import tempfile
import math
import torch
import torch.nn as nn
from source.model_module.diffusion.diffusion_utils import calc_diffusion_step_embedding
from source.model_module.consistency.consistency_processor import ConsistencyTraining, ImprovedConsistencyTraining, ConsistencySamplingAndEditing
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
        self.f0_normalize = hps["f0"]["normalize"]

        # Noise Predictor
        self.student_model = UNet1D_Denoiser(hps=hps) # NoisePredictor
        self.ema_student_model = UNet1D_Denoiser(hps=hps)
        self.ema_student_model.load_state_dict(self.student_model.state_dict())

        # BP off
        for param in self.ema_student_model.parameters():
            param.requires_grad = False
        self.ema_student_model = self.ema_student_model.eval()

        # Consistency Modules
        self.is_improved = hps["Consistency"]["use_improved_consistency"]
        if self.is_improved is True:
            self.consistency = ImprovedConsistencyTraining(hps=hps)
        else:
            self.consistency = ConsistencyTraining(hps=hps)
            self.teacher_model = UNet1D_Denoiser(hps=hps)
            self.teacher_model.load_state_dict(self.student_model.state_dict())
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model = self.teacher_model.eval()
        self.sampler = ConsistencySamplingAndEditing(hps=hps)
        self.sampling_sigmas = hps["Consistency"]["sampling_sigmas"]

        # EMA
        self.num_timesteps = self.consistency.initial_timesteps

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
        if self.is_improved is True:
            outputs = self.consistency(model=self.student_model,
                                       x=f0,
                                       current_training_step=global_step,
                                       total_training_steps=estimated_stepping_batches,
                                       f0_len=f0_len,    # noisyF0
                                       ph_IDs=IDs,
                                       ph_IDs_len=IDs_len,   # Condition (ph)
                                       ph_ID_dur=IDs_dur,
                                       NoteIDs=NoteIDs,
                                       NoteIDS_len=NoteID_len,
                                       g=g)
        else:
            outputs = self.consistency(student_model=self.student_model,
                                       teacher_model=self.teacher_model,
                                       x=f0,
                                       current_training_step=global_step,
                                       total_training_steps=estimated_stepping_batches,
                                       f0_len=f0_len,    # noisyF0
                                       ph_IDs=IDs,
                                       ph_IDs_len=IDs_len,   # Condition (ph)
                                       ph_ID_dur=IDs_dur,
                                       NoteIDs=NoteIDs,
                                       NoteIDS_len=NoteID_len,
                                       g=g)
        return outputs

    @torch.inference_mode()
    def sampling(self, condition : list):
        ph_IDs, ph_IDs_len, ph_IDs_dur, NoteIDs, NoteID_len, NoteDur,speakerID = condition
        B = ph_IDs.size(0)
        if B != 1:
            raise ValueError(f"Batchsize is 1 only supported in Inference")
        device = ph_IDs.device
        f0_len = torch.sum(condition[2][0])
        shape = (ph_IDs.size(0), 1, int(f0_len))
        noise_f0 = torch.randn(shape, device=device)

        sample_dict = dict()
        for sigmas in self.sampling_sigmas:
            samples = self.sampler(
                self.ema_student_model, noise_f0, sigmas, clip_denoised=True, verbose=True,
                f0_len=f0_len.view(B),    # noisyF0
                ph_IDs=ph_IDs,
                ph_IDs_len=ph_IDs_len,   # Condition (ph)
                ph_ID_dur=ph_IDs_dur,
                NoteIDs=NoteIDs,
                NoteIDS_len=NoteID_len,
                g=speakerID)
            samples = samples.clamp(min=-1.0, max=1.0)
            denormalize = self.reverse_norm_f0(samples)
            sample_dict[f"sigmas={sigmas}"] = denormalize

        return sample_dict

    def norm_f0(self,f0):
        # log scale normalize
        if self.f0_normalize=="log_scale":
            lf0 = 2595. * torch.log10(1. + f0 / 700.) / 500 / self.lf0_max
        # [0,1]normalize
        elif self.f0_normalize == "[0,1]":
            f0 = f0 / self.f0_max
        # [-1, 1] normalize
        elif self.f0_normalize == "[-1,1]":
            f0 = (f0 - (self.f0_max / 2) ) / (self.f0_max / 2)
        else:
            raise ValueError(f"f0_normalize:{self.f0_normalize} is not Implemented.")
        return f0

    def reverse_norm_f0(self,f0):
        # log scale denormalize
        if self.f0_normalize == "log_scale":
            f0[f0<0] = 0
            f0 = (700 * (torch.pow(10, f0 *self.lf0_max * 500 / 2595) - 1))
        # [0,1] denormalize
        elif self.f0_normalize == "[0,1]":
            f0[f0<0] = 0
            f0 = f0 * self.f0_max
        # [-1,1]denormalize
        elif self.f0_normalize == "[-1,1]":
            f0 = f0*(self.f0_max/2) + (self.f0_max/2)
            f0[f0<0] = 0
        else:
            raise ValueError(f"f0_normalize:{self.f0_normalize} is not Implemented.")
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
                        timesteps,
                        f0_len,    # noisyF0
                        ph_IDs,
                        ph_IDs_len,   # Condition (ph)
                        ph_ID_dur,
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


import torch.nn.functional as F
class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ResidualBlock(nn.Module):
    """ Residual Block """

    def __init__(self, d_encoder, residual_channels, dropout, multi_speaker=True):
        super(ResidualBlock, self).__init__()
        self.multi_speaker = multi_speaker
        self.conv_layer = ConvNorm(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            stride=1,
            padding=int((3 - 1) / 2),
            dilation=1,
        )
        self.diffusion_projection = LinearNorm(residual_channels, residual_channels)
        if multi_speaker:
            self.speaker_projection = LinearNorm(d_encoder, residual_channels)
        self.conditioner_projection = ConvNorm(
            d_encoder, residual_channels, kernel_size=1
        )
        self.output_projection = ConvNorm(
            residual_channels, 2 * residual_channels, kernel_size=1
        )

    def forward(self, x, conditioner, diffusion_step, speaker_emb, mask=None):

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        if self.multi_speaker:
            speaker_emb = self.speaker_projection(speaker_emb).unsqueeze(1).expand(
                -1, conditioner.shape[-1], -1
            ).transpose(1, 2)

        residual = y = x + diffusion_step
        y = self.conv_layer(
            (y + conditioner + speaker_emb) if self.multi_speaker else (y + conditioner)
        )
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        x, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class DiffusionEmbedding(nn.Module):
    """ Diffusion Step Embedding """

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        x = 12.5 * x # Maxは80なので
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

from einops.layers.torch import Rearrange
from torch import Tensor
class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 16.0) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)

class Denoiser(nn.Module):
    """ Conditional Diffusion Denoiser """

    def __init__(self, hps):
        super(Denoiser, self).__init__()

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
        gin_channels            = hps["NoisePredictor"]['gin_channels']
        NoteEnc_n_note          = int(hps["NoteEncoder"]["n_note"])+1
        NoteEnc_hidden_channels = hps["NoteEncoder"]["hidden_channels"]
        self.diffusion_step_embed_dim_in = Diff_step_embed_in

        d_encoder = inner_channels #model_config["transformer"]["encoder_hidden"]
        residual_channels =inner_channels#model_config["denoiser"]["residual_channels"]
        residual_layers = WN_n_layers #model_config["denoiser"]["residual_layers"]
        dropout = WN_p_dropout #model_config["denoiser"]["denoiser_dropout"]
        multi_speaker = True

        # Speaker / VUV Embedding
        self.emb_g  = nn.Embedding(n_speakers, gin_channels)

        # f0 Pre projection
        self.input_projection = nn.Sequential(
            ConvNorm(1, residual_channels, kernel_size=1),
            nn.ReLU()
        )

        # Timestep Embedding
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            ACTIVATION_FUNCTIONS["mish"],
            LinearNorm(residual_channels * 4, residual_channels)
        )
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

        # Noise Predict Layer
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_encoder, residual_channels, dropout=dropout, multi_speaker=multi_speaker
                )
                for _ in range(residual_layers)
            ]
        )

        # ReInput Projection
        self.skip_projection = ConvNorm(
            residual_channels, residual_channels, kernel_size=1
        )

        # Output Projection
        self.output_projection = ConvNorm(
            residual_channels, out_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.conv.weight)


    def forward(self,
                f0,
                timesteps,
                f0_len,    # noisyF0
                ph_IDs,
                ph_IDs_len,   # Condition (ph)
                ph_ID_dur,
                NoteIDs=None,
                NoteIDS_len=None,
                g=None):
        """

        :param mel: [B, 1, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :param speaker_emb: [B, M]
        :return:
        """
        # Pre Projection
        f0_mask = torch.unsqueeze(commons.sequence_mask(f0_len, f0.size(2)), 1).to(f0.device)
        f0 = self.input_projection(f0)  # x [B, residual_channel, T]
        f0 = F.relu(f0)

        # Embedding speakerID
        speaker_emb = self.emb_g(g)

        # Embedding timesteps
        timesteps = self.diffusion_embedding(timesteps.view(-1))
        timesteps = self.mlp(timesteps)

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
            conditioner = ph_IDs + noteIDs
        else:
            conditioner = ph_IDs

        skip = []
        for layer in self.residual_layers:
            f0, skip_connection = layer(f0*f0_mask, conditioner, timesteps, speaker_emb)
            skip.append(skip_connection*f0_mask)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x*f0_mask)
        x = F.relu(x)
        x = self.output_projection(x)*f0_mask  # [B, 80, T]

        return x

from source.model_module.diffusion.diffusers_1d_modules import ResBlock1D, Downsample1D, Upsample1D, OutConv1DBlock
from source.model_module.models.components.attentions import Encoder as SelfAttn
from source.model_module.models.components.attentions import Decoder as CrossAttn
class U_Net_1D(torch.nn.Module):
    def __init__(self,in_ch=192,             # Hidden
                    inner_ch=192,          #
                    filter_ch=768,
                    out_ch=4,              # f0, mask, v, u
                    time_embed_dim=192,
                    attn_filter_ch=768,
                    ):
        super().__init__()

        ########################
        ### For Down Process ###
        ########################
        self.ResBlock_D1 = ResBlock1D(inp_channels=in_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_D1 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        self.DownSample_D1 =Downsample1D(channels=inner_ch,
                                         use_conv=True,
                                         out_channels=inner_ch)

        self.ResBlock_D2 = ResBlock1D(inp_channels=inner_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_D2 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        self.DownSample_D2 =Downsample1D(channels=inner_ch,
                                         use_conv=True,
                                         out_channels=inner_ch)

        #######################
        ### For Mid Process ###
        #######################
        self.ResBlock_M1 = ResBlock1D(inp_channels=inner_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.SelfAttn    = SelfAttn(hidden_channels=inner_ch,
                                    filter_channels=filter_ch,
                                    n_heads=4,
                                    n_layers=1,
                                    kernel_size=3,
                                    p_dropout=0.1)
        self.ResBlock_M2 = ResBlock1D(inp_channels=inner_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)


        ######################
        ### For Up Process ###
        ######################
        self.UpSample_U1 = Upsample1D(channels=inner_ch,
                                      use_conv=True,
                                      out_channels=inner_ch)
        self.ResBlock_U1 = ResBlock1D(inp_channels=inner_ch*2,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_U1 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)

        self.UpSample_U2 = Upsample1D(channels=inner_ch,
                                      use_conv=True,
                                      out_channels=inner_ch)
        self.ResBlock_U2 = ResBlock1D(inp_channels=in_ch*2,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_U2 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)

        # self.final = OutConv1DBlock(num_groups_out=8,
        #                             out_channels=out_ch,
        #                             embed_dim=inner_ch,
        #                             act_fn="mish")

    def forward(self, x, x_mask, h, h_mask, t):

        # Down Process
        x = self.ResBlock_D1(inputs=x, t=t)
        x_0 = self.CrossAttn_D1(x=x, x_mask=x_mask, h=h, h_mask=h_mask)
        _, _, x_0_len = x_0.shape
        x = self.DownSample_D1(inputs=x_0) # length/2

        x = self.ResBlock_D2(inputs=x, t=t)
        x_mask_1 = self.mask_downsample(downsample_x=x, mask=x_mask)
        x_1 = self.CrossAttn_D2(x=x, x_mask=x_mask_1, h=h, h_mask=h_mask)
        _, _, x_1_len = x_1.shape
        x = self.DownSample_D2(inputs=x_1) # length/4

        # Mid Process
        x = self.ResBlock_M1(inputs=x, t=t)
        x_mask_2 = self.mask_downsample(downsample_x=x, mask=x_mask_1)
        x = self.SelfAttn(x=x,x_mask=x_mask_2)
        x = self.ResBlock_M2(inputs=x, t=t) # length/4

        # Up Process
        x = self.UpSample_U1(inputs=x) # length/2
        x = torch.cat(tensors=[x_1, x[:,:,:x_1_len]], dim=1) # concat in "channel direction"
        x = self.ResBlock_U1(inputs=x, t=t)
        x = self.CrossAttn_U1(x=x, x_mask=x_mask_1, h=h, h_mask=h_mask)

        x = self.UpSample_U2(inputs=x)
        x = torch.cat(tensors=[x_0, x[:,:,:x_0_len]], dim=1) # concat in "channel direction"
        x = self.ResBlock_U2(inputs=x, t=t)
        x = self.CrossAttn_U2(x=x, x_mask=x_mask, h=h, h_mask=h_mask)

        # Final Process
        #x = self.final(x)
        return x

    def mask_downsample(self, downsample_x, mask):
        _,_,mask_length = mask.shape
        _,_,x_length = downsample_x.shape
        if mask_length % 2 == 0: # even
            mask = mask[:,:,1::2]
        else:
            mask = mask[:,:,::2]

        if x_length == mask.size(2):
            return mask

def swish(x):
    return x * torch.sigmoid(x)
from einops.layers.torch import Rearrange
from torch import Tensor
class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 16.0) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            #Rearrange("b c -> b c () ()"),
            Rearrange("b c -> b c"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)


class UNet1D_Denoiser(nn.Module):
    """ Conditional Diffusion Denoiser """

    def __init__(self, hps):
        super(UNet1D_Denoiser, self).__init__()

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
        gin_channels            = hps["NoisePredictor"]['gin_channels']
        NoteEnc_n_note          = int(hps["NoteEncoder"]["n_note"])+1
        NoteEnc_hidden_channels = hps["NoteEncoder"]["hidden_channels"]
        self.diffusion_step_embed_dim_in = Diff_step_embed_in

        d_encoder = inner_channels #model_config["transformer"]["encoder_hidden"]
        residual_channels =inner_channels#model_config["denoiser"]["residual_channels"]
        residual_layers = WN_n_layers #model_config["denoiser"]["residual_layers"]
        dropout = WN_p_dropout #model_config["denoiser"]["denoiser_dropout"]
        multi_speaker = True

        # Speaker / VUV Embedding
        self.emb_g  = nn.Embedding(n_speakers, gin_channels)

        # f0 Pre projection
        self.input_projection = nn.Sequential(
            ConvNorm(1, residual_channels, kernel_size=1),
            nn.ReLU()
        )

        # Timestep Embedding
        self.diffusion_embedding = NoiseLevelEmbedding(channels=residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            ACTIVATION_FUNCTIONS["mish"],
            LinearNorm(residual_channels * 4, residual_channels)
        )
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

        self.unet = U_Net_1D()

        # ReInput Projection
        self.skip_projection = ConvNorm(
            residual_channels, residual_channels, kernel_size=1
        )

        # Output Projection
        self.output_projection = ConvNorm(
            residual_channels, out_channels, kernel_size=1
        )
        # nn.init.zeros_(self.output_projection.conv.weight)


    def forward(self,
                f0,
                timesteps,
                f0_len,    # noisyF0
                ph_IDs,
                ph_IDs_len,   # Condition (ph)
                ph_ID_dur,
                NoteIDs=None,
                NoteIDS_len=None,
                g=None):
        """

        :param mel: [B, 1, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :param speaker_emb: [B, M]
        :return:
        """
        # Pre Projection
        f0_mask = torch.unsqueeze(commons.sequence_mask(f0_len, f0.size(2)), 1).to(f0.device)
        f0 = self.input_projection(f0)  # x [B, residual_channel, T]
        f0 = F.mish(f0)

        # Embedding speakerID
        speaker_emb = self.emb_g(g)

        # Embedding timesteps
        timesteps = self.diffusion_embedding(timesteps.view(-1))
        timesteps = self.mlp(timesteps)

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
            conditioner = ph_IDs + noteIDs
        else:
            conditioner = ph_IDs

        x =  self.unet(x=f0,
                       x_mask=f0_mask,
                       h =conditioner,
                       h_mask = f0_mask,
                       t=timesteps)

        x = self.skip_projection(x)
        x = F.mish(x)
        x = self.output_projection(x)  # [B, 80, T]

        return x*f0_mask

if __name__ == "__main__":
    Batch = 2
    F0_len = 187
    ids_len = 25
    inner_channel = 192

        # :param mel: [B, 1, M, T]
        # :param diffusion_step: [B,]
        # :param conditioner: [B, M, T]
        # :param speaker_emb: [B, M]
    f0 = torch.randn(size=(Batch, 1, F0_len)) #  [Batch, inner_channel, f0_len]
    timesteps = torch.randint(low=0, high=10, size=(Batch,1 ))
    f0_len = torch.randint(low=10, high=F0_len, size=(Batch,))    # noisyF0
    ph_IDs = torch.randint(low=10, high=F0_len, size=(Batch, ids_len))
    ph_IDs_len = torch.randint(low=10, high=ids_len, size=(Batch,))   # Condition (ph)
    ph_ID_dur = torch.randint(low=10, high=F0_len, size=(Batch, ids_len))
    NoteIDs= torch.randn(size=(Batch, ids_len))
    NoteIDS_len=torch.randint(low=10, high=ids_len, size=(Batch,))
    g=torch.randint(low=0, high=1, size=(Batch,))

    from source.utils.read_write import read_yml
    yml = read_yml("configs/model/consistency_wavenet.yaml")
    # denoiser = Denoiser(hps=yml["net_g"]["hps"])
    denoiser = UNet1D_Denoiser(hps=yml["net_g"]["hps"])
    out = denoiser(f0,
                timesteps,
                f0_len,    # noisyF0
                ph_IDs,
                ph_IDs_len,   # Condition (ph)
                ph_ID_dur,
                NoteIDs=NoteIDs,
                NoteIDS_len=NoteIDS_len,
                g=g)
    print(out)
    pass