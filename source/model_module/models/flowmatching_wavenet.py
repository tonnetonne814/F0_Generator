
import torch
import torch.nn as nn
from source.model_module.models.components import modules
from source.utils.audio_utils import commons
from source.utils.audio_utils.commons import generate_path
from source.model_module.models.components.note_encoder import NoteEncoder
from source.model_module.models.components.text_encoder_vits import TextEncoder_VITS as TextEncoder
from torchcfm.conditional_flow_matching import \
  ConditionalFlowMatcher, \
  ExactOptimalTransportConditionalFlowMatcher, \
  VariancePreservingConditionalFlowMatcher, \
  SchrodingerBridgeConditionalFlowMatcher

class FlowMatchingModels(nn.Module):
    def __init__(self,
                 hps,
                 training=True):
        super(FlowMatchingModels, self).__init__()
        hps = hps.__dict__['_content'] # オブジェクト属性を辞書型に変更

        # F0最大値設定
        self.f0_max = float(hps["f0"]["max"])
        self.lf0_max = 2595. * torch.log10(1. + torch.tensor(hps["f0"]["max"]).float() / 700.) / 500

        # FlowMatching modules
        sigma = hps["FlowMatching"]["sigma"]
        ot_method = hps["FlowMatching"]["ot_method"]
        cfm_method = hps["FlowMatching"]["cfm_method"]
        if cfm_method == "CFM":
            self.FM = ConditionalFlowMatcher(sigma=sigma)
        elif cfm_method == "OT-CFM":
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        elif cfm_method == "SB-CFM":
            self.FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method=ot_method)
        elif cfm_method == "StochasticInterpolate":
            self.FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
        elif cfm_method == "ActionMatching":
            raise ValueError(f"cfm_method:{cfm_method} is not Implemented.")
            self.FM = ConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(f"cfm_method:{cfm_method} is not Implemented.")

        self.VF_Predictor = VectorFieldPredictor(hps=hps)

    # 入力を受け取り、ロスを返す。
    def forward(self, f0,  f0_len, #X1
                    IDs, IDs_len, IDs_dur,
                    NoteIDs = None, NoteID_len = None,
                    g=None,
                    noise=None, noise_len=None): #X0

        f0 = self.norm_f0(f0)
        if NoteIDs is not None:
            NoteIDs = self.norm_f0(NoteIDs)

        # Generate timesteps
        B, _, _ = f0.shape
        timesteps, xt, ut = self.FM.sample_location_and_conditional_flow(noise, f0) #(X0, X1)

        # VF prediction
        vt = self.VF_Predictor(f0=xt,
                               f0_len=f0_len,    # noisyF0
                               ph_IDs=IDs,
                               ph_IDs_len=IDs_len,   # Condition (ph)
                               ph_ID_dur=IDs_dur,
                               NoteIDs=NoteIDs,
                               NoteIDS_len=NoteID_len,
                               timesteps=timesteps.view(B, 1),
                               g=g)
        return ut, vt

    @torch.inference_mode()
    def sampling_euler(self,IDs, IDs_len, IDs_dur,
                    NoteIDs = None, NoteID_len = None,
                    g=None,
                    t_start=0.0, t_end=1.0, n_timesteps=1000):
        B = int(IDs.size(0))
        assert B == 1

        f0_len = torch.tensor([int(torch.sum(IDs_dur[0]))], dtype=torch.int64, device=IDs.device)
        x = torch.randn(size=(B, 1, f0_len), dtype=torch.float32, device=IDs.device)
        t = t_start
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=IDs.device)
        dt = t_span[1] - t_span[0]
        sol = list()
        t_save = list()

        sol.append(x) # 初期保存
        t_save.append(t)
        for step in range(1, len(t_span)):
            t = torch.tensor(t, dtype=torch.float32, device=IDs.device)
            dphi_dt = self.VF_Predictor(f0=x,
                               f0_len=f0_len,    # noisyF0
                               ph_IDs=IDs,
                               ph_IDs_len=IDs_len,   # Condition (ph)
                               ph_ID_dur=IDs_dur,
                               NoteIDs=NoteIDs,
                               NoteIDS_len=NoteID_len,
                               timesteps=t.view(B, 1),
                               g=g)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            t_save.append(t)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        pd_dict = dict()
        pd_dict["Final"] = self.reverse_norm_f0(sol[-1])
        for idx, (x, t) in enumerate(zip(sol, t_save)):
            if idx % 10 != 0:
                continue
            pd_dict[f"{t}"] = self.reverse_norm_f0(x)
        return pd_dict

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

class VectorFieldPredictor(nn.Module):
    def __init__(self,hps):
        super().__init__()

        out_channels            = hps["VectorFieldPredictor"]['out_channels']
        inner_channels          = hps["VectorFieldPredictor"]['inner_channels']
        WN_in_channels          = hps["VectorFieldPredictor"]['WN_in_channels']
        WN_kernel_size          = hps["VectorFieldPredictor"]['WN_kernel_size']
        WN_dilation_rate        = hps["VectorFieldPredictor"]['WN_dilation_rate']
        WN_n_layers             = hps["VectorFieldPredictor"]['WN_n_layers']
        WN_p_dropout            = hps["VectorFieldPredictor"]['WN_p_dropout']
        Attn_filter_channels    = hps["VectorFieldPredictor"]['Attn_filter_channels']
        Attn_kernel_size        = hps["VectorFieldPredictor"]['Attn_kernel_size']
        Attn_n_layers           = hps["VectorFieldPredictor"]['Attn_n_layers']
        Attn_n_heads            = hps["VectorFieldPredictor"]['Attn_n_heads']
        Attn_p_dropout          = hps["VectorFieldPredictor"]['Attn_p_dropout']
        Attn_vocab_size         = hps["VectorFieldPredictor"]["vocab_size"]+1 # mask用に0を取っておく。
        n_speakers              = hps["VectorFieldPredictor"]['n_speakers']
        Diff_step_embed_in      = hps["VectorFieldPredictor"]['Diff_step_embed_in']
        Diff_step_embed_mid     = hps["VectorFieldPredictor"]['Diff_step_embed_mid']
        Diff_step_embed_out     = hps["VectorFieldPredictor"]['Diff_step_embed_out']
        gin_channels            = hps["VectorFieldPredictor"]['n_speakers']
        NoteEnc_n_note          = int(hps["NoteEncoder"]["n_note"])+1
        NoteEnc_hidden_channels = hps["NoteEncoder"]["hidden_channels"]
        self.out_channels=out_channels
        self.diffusion_step_embed_dim_in = Diff_step_embed_in

        # Timestep Embedding
        self.time_embeddings = SinusoidalPosEmb(dim=Diff_step_embed_in)
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

        #self.rezero = Rezero()

        # projection
        output_ch = 1
        self.proj_1 = nn.Conv1d(inner_channels, inner_channels, 1) # skip projection
        self.relu = nn.Mish()
        self.proj_2 = nn.Conv1d(inner_channels, output_ch, 1) # out projection

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
        emb_timesteps = self.time_embeddings(timesteps).to(f0.device)
        emb_timesteps = swish(self.fc_t1(emb_timesteps))
        emb_timesteps = swish(self.fc_t2(emb_timesteps))
        emb_timesteps = torch.squeeze(emb_timesteps, dim=1)

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

        # ReZero Regularization
        # f0_vuv = self.rezero(f0_vuv)

        # f0, vuv_vector = torch.split(tensor=f0_vuv,
        #                             split_size_or_sections=1,  # ここlabel３つ
        #                             dim=1 )
        # vuv_vector = self.vuv_proj(vuv_vector)


        return f0 # , torch.zeros(size=(f0.size(0), 3, f0.size(2)))

def swish(x):
    return x * torch.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

import math
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


if __name__ == "__main__":
  from source.utils.read_write import read_yml
  yaml_path = "./SiFi-VITS2-44100-Ja-main/configs/config.yaml"
  config = read_yml(yaml_path)
  _ = DiffusionModels(hps=config, vocab_size=192, training=True)
  pass