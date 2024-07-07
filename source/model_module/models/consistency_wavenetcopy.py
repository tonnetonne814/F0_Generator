import tempfile
import math
import torch
import torch.nn as nn
from source.model_module.diffusion.diffusion_utils import calc_diffusion_step_embedding
from source.model_module.consistency.consistency_processorcopy import ConsistencyProcessor as ConsistencyTraining
from source.model_module.models.components import modules
from source.utils.audio_utils import commons
from source.utils.audio_utils.commons import generate_path
from source.model_module.models.components.note_encoder import NoteEncoder
from source.model_module.models.components.text_encoder_vits import TextEncoder_VITS as TextEncoder

class ConsistencyModels(nn.Module):
    def __init__(self,
                 hps):
        super(ConsistencyModels, self).__init__()
        hps = hps.__dict__['_content'] # オブジェクト属性を辞書型に変更

        # F0最大値設定
        self.f0_max = float(hps["f0"]["max"])
        self.lf0_max = 2595. * torch.log10(1. + torch.tensor(hps["f0"]["max"]).float() / 700.) / 500

        # Consistency Modules
        self.consistency = ConsistencyTraining(hps=hps)
        self.sample_steps = hps["Consistency"]["sample_steps"]

        # Noise Predictor
        self.noise_predictor = NoisePredictor(hps=hps)

        # EMA
        if hps["Consistency"]["is_distill"] is False:
            self.ema = self.copy_model(self.noise_predictor, hps)
            self.ema.requires_grad_(False)
        else:
          raise ValueError("Consistency Distillution is not implemented.")

        # 複製確認
        assert id(self.ema) != id(self.noise_predictor)

    # DeepCopyだとエラー出るので代替関数
    def copy_model(self, model, hps):
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model.state_dict(), f.name)
            new_model = type(model)(hps)
            new_model.load_state_dict(torch.load(f.name))
        return new_model

    # 入力を受け取り、ロスを返す。
    def forward(self, f0,  f0_len,
                      IDs, IDs_len, IDs_dur,
                      NoteIDs = None, NoteID_len = None,
                      g=None, global_step=None, estimated_stepping_batches=None):
        device = f0.device
        B = f0.size(0)

        f0 = self.norm_f0(f0)
        if NoteIDs is not None:
          NoteIDs = self.norm_f0(NoteIDs)

        # F0 preprocess (F0 → NoisyF0)
        current_times, current_noisy_f0, next_times, next_noisy_f0, bins \
                = self.consistency.get_noisy_f0(f0, global_step, estimated_stepping_batches)

        # Noise prediction
        current_F_theta = self.noise_predictor(f0=current_noisy_f0,
                                        f0_len=f0_len,    # noisyF0
                                        ph_IDs=IDs,
                                        ph_IDs_len=IDs_len,   # Condition (ph)
                                        ph_ID_dur=IDs_dur,
                                        NoteIDs=NoteIDs,
                                        NoteIDS_len=NoteID_len,
                                        timesteps=current_times.view(B, 1),
                                        g=g)
        with torch.no_grad():
            next_F_theta = self.ema(f0=next_noisy_f0,
                                    f0_len=f0_len,    # noisyF0
                                    ph_IDs=IDs,
                                    ph_IDs_len=IDs_len,   # Condition (ph)
                                    ph_ID_dur=IDs_dur,
                                    NoteIDs=NoteIDs,
                                    NoteIDS_len=NoteID_len,
                                    timesteps=next_times.view(B, 1),
                                    g=g)

        # denoise process
        current_f_theta = self.consistency.denoise(noisy_input=current_noisy_f0,
                                  noisy_output=current_F_theta,
                                  times=current_times)
        with torch.no_grad():
            next_f_theta = self.consistency.denoise(noisy_input=next_noisy_f0,
                                    noisy_output=next_F_theta,
                                    times=next_times)

        # Loss calculation for ConsistencyTraining
        loss_CT = self.consistency.get_loss(current_f_theta, next_f_theta)

        return loss_CT, bins

    @torch.inference_mode()
    def sampling(self, condition : list):
        ph_IDs, ph_IDs_len, ph_IDs_dur, NoteIDs, NoteID_len, NoteDur,speakerID = condition
        B = ph_IDs.size(0)
        device = ph_IDs.device
        f0_len = torch.sum(condition[2][0])
        shape = (ph_IDs.size(0), 1, int(f0_len))

        time = self.consistency.get_time(device)
        noise = torch.randn(shape, device=device)
        noise_f0 = noise * time

        current_F_theta = self.noise_predictor(f0=noise_f0,
                                        f0_len=f0_len.view(B),    # noisyF0
                                        ph_IDs=ph_IDs,
                                        ph_IDs_len=ph_IDs_len,   # Condition (ph)
                                        ph_ID_dur=ph_IDs_dur,
                                        NoteIDs=NoteIDs,
                                        NoteIDS_len=NoteID_len,
                                        timesteps=time.view(B, 1),
                                        g=speakerID)
        f0_pd = self.consistency.denoise(noisy_input=noise_f0,
                                         noisy_output=current_F_theta,
                                         times=time)

        if self.sample_steps <= 1:
            return self.reverse_norm_f0(f0_pd)

        times = self.consistency.get_multistep_times(steps=self.sample_steps,
                                                     device=device)
        time_min = self.consistency.get_time_min()

        for time in times:
            noise = torch.randn(shape, device=device)
            noisy_f0 = f0_pd + math.sqrt(time.item() ** 2 - time_min**2) * noise # 論文Sampling 6行目
            current_F_theta = self.noise_predictor(f0=noisy_f0,
                                            f0_len=f0_len.view(B, 1),    # noisyF0
                                            ph_IDs=ph_IDs,
                                            ph_IDs_len=ph_IDs_len,   # Condition (ph)
                                            ph_ID_dur=ph_IDs_dur,
                                            NoteIDs=NoteIDs,
                                            NoteIDS_len=NoteID_len,
                                            timesteps=time.view(B, 1),
                                            g=speakerID)
            f0_pd = self.consistency.denoise(noisy_input=noisy_f0,
                                            noisy_output=current_F_theta,
                                            times=time)

        return self.reverse_norm_f0(f0_pd)

    def norm_f0(self,f0):
        #lf0 = 2595. * torch.log10(1. + f0 / 700.) / 500 / self.lf0_max
        f0 = f0 / self.f0_max
        return f0
    def reverse_norm_f0(self,f0):
        #lf0[lf0<0] = 0
        #f0 = (700 * (torch.pow(10, lf0 *self.lf0_max * 500 / 2595) - 1))
        f0[f0<0] = 0
        f0 = f0 * self.f0_max
        return f0

class NoisePredictor(nn.Module):
    def __init__(self,hps):
        super().__init__()

        out_channels            = hps["NoisePredictor"]['out_channels']
        inner_channels          = hps["NoisePredictor"]['inner_channels']
        WN_in_channels          = hps["NoisePredictor"]['WN_in_channels']
        WN_kernel_size          = hps["NoisePredictor"]['WN_kernel_size']
        WN_dilation_rate        = hps["NoisePredictor"]['WN_dilation_rate']
        WN_n_layers             = hps["NoisePredictor"]['WN_n_layers']
        WN_p_dropout            = hps["NoisePredictor"]['WN_p_dropout']
        Attn_filter_channels    = hps["NoisePredictor"]['Attn_filter_channels']
        Attn_kernel_size        = hps["NoisePredictor"]['Attn_kernel_size']
        Attn_n_layers           = hps["NoisePredictor"]['Attn_n_layers']
        Attn_n_heads            = hps["NoisePredictor"]['Attn_n_heads']
        Attn_p_dropout          = hps["NoisePredictor"]['Attn_p_dropout']
        Attn_vocab_size         = hps["NoisePredictor"]["vocab_size"]+1 # mask用に0を取っておく。
        n_speakers              = hps["NoisePredictor"]['n_speakers']
        Diff_step_embed_in      = hps["NoisePredictor"]['Diff_step_embed_in']
        Diff_step_embed_mid     = hps["NoisePredictor"]['Diff_step_embed_mid']
        Diff_step_embed_out     = hps["NoisePredictor"]['Diff_step_embed_out']
        gin_channels            = hps["NoisePredictor"]['n_speakers']
        NoteEnc_n_note          = int(hps["NoteEncoder"]["n_note"])+1
        NoteEnc_hidden_channels = hps["NoteEncoder"]["hidden_channels"]
        self.out_channels=out_channels
        self.diffusion_step_embed_dim_in = Diff_step_embed_in

        # Timestep Embedding
        self.fc_t = nn.ModuleList()
        self.fc_t1 = nn.Linear(Diff_step_embed_in, Diff_step_embed_mid)
        self.fc_t2 = nn.Linear(Diff_step_embed_mid, Diff_step_embed_out)

        # Speaker / VUV Embedding
        self.emb_g  = nn.Embedding(n_speakers, gin_channels)

        # NoteEncoder
        self.note_enc = NoteEncoder(n_note=NoteEnc_n_note,
                                    hidden_channels=NoteEnc_hidden_channels)

        # Attention
        self.text_encoder = TextEncoder(n_vocab         =Attn_vocab_size,
                                        out_channels    =inner_channels,
                                        hidden_channels =inner_channels,
                                        filter_channels =Attn_filter_channels,
                                        n_heads         =Attn_n_heads,
                                        n_layers        =Attn_n_layers,
                                        kernel_size     =Attn_kernel_size,
                                        p_dropout       =Attn_p_dropout   )

        # WaveNet
        self.pre = nn.Conv1d(WN_in_channels, inner_channels, 1)
        self.WaveNet = modules.WN(hidden_channels=inner_channels,
                                  Diff_step_embed_out=Diff_step_embed_out,
                                  kernel_size=WN_kernel_size,
                                  dilation_rate=WN_dilation_rate,
                                  n_layers=WN_n_layers,
                                  gin_channels=gin_channels,
                                  p_dropout=WN_p_dropout)
        # projection
        output_ch = 1
        self.proj_1 = nn.Conv1d(inner_channels, output_ch, 1)
        self.relu = ACTIVATION_FUNCTIONS["mish"]
        self.proj_2 = nn.Conv1d(output_ch, output_ch, 1)

    def forward(self,   f0,
                        f0_len,    # noisyF0
                        ph_IDs,
                        ph_IDs_len,   # Condition (ph)
                        ph_ID_dur,
                        timesteps,
                        NoteIDs=None,
                        NoteIDS_len=None,
                        g=None):
        # Embedding timesteps
        emb_timesteps = calc_diffusion_step_embedding(timesteps, self.diffusion_step_embed_dim_in).to(f0.device)
        emb_timesteps = swish(self.fc_t1(emb_timesteps))
        emb_timesteps = swish(self.fc_t2(emb_timesteps))

        # Embedding speakerID
        g = torch.unsqueeze(self.emb_g(g),dim=2) # [Batch, n_speakers] to [Batch, 1, gin_channels]

        # Projection
        f0_mask = torch.unsqueeze(commons.sequence_mask(f0_len, f0.size(2)), 1).to(f0.device)
        f0 = self.pre(f0) * f0_mask                 # [Batch, Channel=1, f0_len] to [Batch, inner_channels, f0_len]

        # Encoding (output is masked) Attention Embedding
        ph_IDs, ph_IDs_mask = self.text_encoder(ph_IDs, ph_IDs_len)                 # [Batch, inner_channel, IDs_len]

        if NoteIDs is not None:
            noteIDs, noteIDs_mask = self.note_enc(  noteID=NoteIDs,
                                                noteID_lengths=NoteIDS_len)

        # expand ph_len to f0_len
        attn_mask = torch.unsqueeze(ph_IDs_mask, 2) * torch.unsqueeze(f0_mask, -1)
        attn = generate_path(duration=torch.unsqueeze(ph_ID_dur,dim=1), mask=attn_mask )
        attn = torch.squeeze(attn, dim=1).permute(0,2,1)                          # [Batch, IDs_len, f0_len]
        ph_IDs = torch.matmul(ph_IDs, torch.tensor(attn, dtype=torch.float32))    # to [Batch, inner_channel, f0_len]

        if NoteIDs is not None:
            noteIDs = torch.matmul(noteIDs, torch.tensor(attn, dtype=torch.float32))  # to [Batch, inner_channel, f0_len]
            hidden = ph_IDs + noteIDs
        else:
            hidden = ph_IDs

        # NoisePrediction Process
        f0 = self.WaveNet(  x         =f0 ,#+ vuv,
                            x_mask    =f0_mask,
                            IDs       =hidden,
                            IDs_mask  =ph_IDs_mask,
                            timesteps =emb_timesteps,
                            g         =g)

        # Projection
        f0 = self.proj_1(f0)
        f0 = self.relu  (f0)
        f0 = self.proj_2(f0) * f0_mask

        return f0

def swish(x):
    return x * torch.sigmoid(x)

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}

if __name__ == "__main__":
    from source.utils.read_write import read_yml
    yaml_path = "./SiFi-VITS2-44100-Ja-main/configs/config.yaml"
    config = read_yml(yaml_path)
    _ = DiffusionModels(hps=config, vocab_size=192, training=True)
    pass