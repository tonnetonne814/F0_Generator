
import copy, torch, copy
import torch
import torch.nn as nn
#from source.model_module.ddiffusion.diffusion_multinomial import MultinomialDiffusion
from source.model_module.diffusion.diffusion_utils import calc_diffusion_step_embedding
from source.model_module.diffusion.diffusion_processor import DiffusionProcessor
from source.model_module.models.components import modules
from tqdm import tqdm
from source.utils.audio_utils import commons
from source.utils.audio_utils.commons import generate_path
from source.model_module.models.components.note_encoder import NoteEncoder
from source.model_module.models.components.text_encoder_vits import TextEncoder_VITS as TextEncoder

class DiffusionModels(nn.Module):
    def __init__(self,
                 hps,
                 training=True):
        super(DiffusionModels, self).__init__()
        hps = hps.__dict__['_content'] # オブジェクト属性を辞書型に変更

        # timestep設定
        self.num_timesteps = hps["Diffusion"]["T"]
        if training is True:
          self.infer_timesteps = hps["Diffusion"]["T"]
          self.ddim = False
        else:
          self.infer_timesteps = hps["Diffusion"]["N"]
          self.ddim = hps["Diffusion"]["ddim"]

        # F0最大値設定
        self.f0_max = float(hps["f0"]["max"])
        self.lf0_max = 2595. * torch.log10(1. + torch.tensor(hps["f0"]["max"]).float() / 700.) / 500

        # F0 diffusion modules
        self.f0_diff = DiffusionProcessor(hps=hps)

        # Noise Predictor
        self.noise_predictor = NoisePredictor(hps=hps)

        # Voice/UnVoice diffusion modules
        #self.vuv_diff = MultinomialDiffusion(num_classes=3,   # vとuvとmaskの3通り
        #                                     timesteps=self.num_timesteps,
        #                                     loss_type="vb_stochastic",
        #                                     parametrization="x0")
        self.noise_schedule = None
        self.N = None
        self.step_infer = None


    # 入力を受け取り、ロスを返す。
    def forward(self, f0,  f0_len,
                      IDs, IDs_len, IDs_dur,
                      NoteIDs = None, NoteID_len = None,
                      g=None):
        device = f0.device
        if device.type == "cpu":
          pass
        f0 = self.norm_f0(f0)
        if NoteIDs is not None:
          NoteIDs = self.norm_f0(NoteIDs)

        # Generate timesteps
        B, _, _ = f0.shape
        timesteps= torch.randint(self.num_timesteps, size=(B, 1, 1)).to(device)
        #timesteps, pt = self.vuv_diff.sample_time(B, device="cuda", method='importance')

        # F0 preprocess (F0 → NoisyF0)
        noisy_f0, noise_gt = self.f0_diff.get_noisy_f0(f0=f0, ts=timesteps)

        # vuv preprocess (VUV → NoisyVUV)
        # noisy_vuv, vuv_diff_params = self.vuv_diff.preprocess(x=vuv, t_int=timesteps.view(-1))

        # Noise prediction
        f0_noise = self.noise_predictor(f0=noisy_f0,
                                        f0_len=f0_len,    # noisyF0
                                        ph_IDs=IDs,
                                        ph_IDs_len=IDs_len,   # Condition (ph)
                                        ph_ID_dur=IDs_dur,
                                        NoteIDs=NoteIDs,
                                        NoteIDS_len=NoteID_len,
                                        timesteps=timesteps.view(B, 1),
                                        g=g)

        # Loss calculation for F0
        loss_f0 = self.f0_diff.get_loss( f0_noise_pd=f0_noise,
                                         noisy_f0=noisy_f0,
                                         noise_gt=noise_gt)

        # Loss calculation for VUV
        #loss_vuv = self.vuv_diff.postprocess(model_out=vuv_noise,
        #                                     parameters=vuv_diff_params,
        #                                     t_int=timesteps.view(-1),
        #                                     t_float=pt)
        #loss_vuv = loss_vuv.sum() / (math.log(2) * torch.sum(vuv_len))

        return loss_f0  #, loss_vuv

    @torch.inference_mode()
    def sampling(self, condition):
        #ph_IDs, ph_IDs_len, ph_IDs_dur, NoteIDs, NoteID_len, NoteDur,speakerID = condition
        f0_len = int(torch.sum(condition[2][0]))

        if self.noise_schedule is None:
          self.noise_schedule = self.f0_diff.get_noise_schedule(timesteps=self.infer_timesteps, training=False)
        # sampling using DDPM reverse process.
        #f0_pd, vuv_pd = self.sampling_given_noise_schedule(size=(1, 1, f0_len),
        #                                                   inference_noise_schedule=noise_schedule,
        #                                                   condition=condition,
        #                                                   ddim=False,
        #                                                   return_sequence=False)
        f0_pd = self.sampling_given_noise_schedule(size=(1, 1, f0_len),
                                                           inference_noise_schedule=self.noise_schedule,
                                                           condition=condition,
                                                           ddim=self.ddim,
                                                           return_sequence=False)

        return self.reverse_norm_f0(f0_pd)
        #return f0_pd, vuv_pd

    @torch.inference_mode()
    def sampling_given_noise_schedule(
        self,
        size,
        inference_noise_schedule,
        condition=None,
        ddim=False,
        return_sequence=False):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the wavenet models
        size (tuple):                   size of tensor to be generated,
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors
        condition (torch.tensor):       ground truth mel spectrogram read from disk
                                        None if used for unconditional generation

        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        #IDs_gt, IDs_gt_len, dur_gt, speaker_ID = condition
        ph_IDs, ph_IDs_len, ph_IDs_dur, NoteIDs, NoteID_len, _, speakerID = condition
        if NoteIDs is not None:
          NoteIDs = self.norm_f0(NoteIDs)
        N , steps_infer = self.f0_diff.get_step_infer(size=size,
                                                     inference_noise_schedule=inference_noise_schedule)

        x = self.f0_diff.generate_x(size).to(ph_IDs.device)
        #log_z = self.vuv_diff.generate_log_z(size)

        if return_sequence:
            x_ = copy.deepcopy(x)
            xs = [x_]

        # need  N と steps_infer
        with torch.no_grad():
            for n in tqdm(range(N - 1, -1, -1),desc="Sampling..."):
                diffusion_steps = (steps_infer[n] * torch.ones((size[0], 1)))

                # Multinomial diffusion preprocess
                #model_in, log_z = self.vuv_diff.sample_preprocess(log_z=log_z, t=n)

                # Noise Prediction
                #f0_noise, vuv_noise = self.noise_predictor(f0=x,
                #                                           f0_len=torch.tensor([dur_gt.size(1)], dtype=torch.int64),
                #                                           IDs=IDs_gt,
                #                                           IDs_len=IDs_gt_len,
                #                                           vuv=model_in,
                #                                           vuv_len=torch.tensor([dur_gt.size(1)], dtype=torch.int64),
                #                                           attn=dur_gt,
                #                                           timesteps=diffusion_steps,
                #                                           g=speaker_ID)

                f0_noise = self.noise_predictor(f0=x,
                                                f0_len=torch.tensor([size[2]], dtype=torch.int64),    # noisyF0
                                                ph_IDs=ph_IDs,
                                                ph_IDs_len=ph_IDs_len,   # Condition (ph)
                                                ph_ID_dur=ph_IDs_dur,
                                                timesteps=diffusion_steps,
                                                NoteIDs=NoteIDs,
                                                NoteIDS_len=NoteID_len,
                                                g=speakerID)

                # Multinomial diffusion postprocess
                #log_z = self.vuv_diff.sample_postprocess(model_out=vuv_noise, log_z=log_z, t_int=n)

                # Denoising
                x = self.f0_diff.denoising(x=x, noise_pd=f0_noise, ddim=ddim, n=n)

                if return_sequence:
                    x_ = copy.deepcopy(x)
                    xs.append(x_)

            # VUV Decoding
            #vuv_pd = self.vuv_diff.decode_log_z(log_z)

        if return_sequence:
            return xs
        return x
        #return x, vuv_pd
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
        #self.emb_uv = nn.Embedding(3, inner_channels) # vとuvとmaskの３つ
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

        #self.rezero = Rezero()

        # projection
        output_ch = 1 # 4 = [f0 vuv]
        self.proj_1 = nn.Conv1d(inner_channels, output_ch, 1)
        self.relu = nn.Mish()
        self.proj_2 = nn.Conv1d(output_ch, output_ch, 1)
        #self.vuv_proj = nn.Conv1d(1,4,1)

    def forward(self,   f0,
                        f0_len,    # noisyF0
                        ph_IDs,
                        ph_IDs_len,   # Condition (ph)
                        ph_ID_dur,
                        #vuv, vuv_len,
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

        # Embedding vuv labels
        #vuv_mask = torch.unsqueeze(commons.sequence_mask(vuv_len, vuv.size(1)), 1).to(vuv.dtype)
        #vuv = self.emb_uv(vuv).permute(0,2,1) * vuv_mask    # [Batch, f0_len]     to [Batch, inner_channel, f0_len]

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

        # ReZero Regularization
        # f0_vuv = self.rezero(f0_vuv)

        # f0, vuv_vector = torch.split(tensor=f0_vuv,
        #                             split_size_or_sections=1,  # ここlabel３つ
        #                             dim=1 )
        # vuv_vector = self.vuv_proj(vuv_vector)


        return f0 # , torch.zeros(size=(f0.size(0), 3, f0.size(2)))

def swish(x):
    return x * torch.sigmoid(x)


if __name__ == "__main__":
  from source.utils.read_write import read_yml
  yaml_path = "./SiFi-VITS2-44100-Ja-main/configs/config.yaml"
  config = read_yml(yaml_path)
  _ = DiffusionModels(hps=config, vocab_size=192, training=True)
  pass