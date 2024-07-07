import torch.nn.functional as F
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
import numpy as np
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
        self.student_model = NoisePredictor(hps=hps) # NoisePredictor
        self.ema_student_model = NoisePredictor(hps=hps)
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
            self.teacher_model = NoisePredictor(hps=hps)
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
        remainder = int(f0_len) % 16
        shape = (ph_IDs.size(0), 1, int(f0_len + (16 - remainder)))
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
        gin_channels            = hps["NoisePredictor"]['n_speakers']
        NoteEnc_n_note          = int(hps["NoteEncoder"]["n_note"])+1
        NoteEnc_hidden_channels = hps["NoteEncoder"]["hidden_channels"]
        self.out_channels=out_channels
        self.diffusion_step_embed_dim_in = Diff_step_embed_in

        self.pre = nn.Conv1d(WN_in_channels, inner_channels, 1)

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
        self.unet = UNetModel_original(in_channels=hps["OpenAI_UnetConfig"]['in_channels'],
                                model_channels=hps["OpenAI_UnetConfig"]["model_channels"],
                                out_channels=hps["OpenAI_UnetConfig"]["out_channels"],
                                num_res_blocks=hps["OpenAI_UnetConfig"]["num_res_blocks"],
                                attention_resolutions=hps["OpenAI_UnetConfig"]["attention_resolutions"],
                                dropout=hps["OpenAI_UnetConfig"]["dropout"],
                                channel_mult=hps["OpenAI_UnetConfig"]["channel_mult"],
                                conv_resample=hps["OpenAI_UnetConfig"]["conv_resample"],
                                dims=hps["OpenAI_UnetConfig"]["dims"],
                                num_classes=hps["OpenAI_UnetConfig"]["num_classes"],
                                use_checkpoint=hps["OpenAI_UnetConfig"]["use_checkpoint"],
                                use_fp16=hps["OpenAI_UnetConfig"]["use_fp16"],
                                num_heads=hps["OpenAI_UnetConfig"]["num_heads"],
                                num_head_channels=hps["OpenAI_UnetConfig"]["num_head_channels"],
                                num_heads_upsample=hps["OpenAI_UnetConfig"]["num_heads_upsample"],
                                use_scale_shift_norm=hps["OpenAI_UnetConfig"]["use_scale_shift_norm"],
                                resblock_updown=hps["OpenAI_UnetConfig"]["resblock_updown"],
                                use_new_attention_order=hps["OpenAI_UnetConfig"]["use_new_attention_order"])

    def forward(self,   f0,
                        timesteps,
                        f0_len,    # noisyF0
                        ph_IDs,
                        ph_IDs_len,   # Condition (ph)
                        ph_ID_dur,
                        NoteIDs=None,
                        NoteIDS_len=None,
                        g=None):
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
        x = torch.concat([f0, hidden], dim=1)
        x = self.unet(x=x, t=timesteps)

        return x * f0_mask

def swish(x):
    return x * torch.sigmoid(x)

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}
from abc import abstractmethod
from source.model_module.models.components.cm_fp16_util import convert_module_to_f16, convert_module_to_f32
from source.model_module.models.components.cm_nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, condition=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                if hasattr(layer, 'attention'):
                    name = str(layer.attention.__class__).split(".")[-1].split("'")[0]
                    # name = str(attention.__class__).split(".")[-1].split("'")[0]
                    if name == "QKVAttention":
                        x = layer(x, condition)
                else:
                    x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            #zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            #)
            ,
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="flash",
        encoder_channels=None,
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        self.attention_type = attention_type
        if attention_type == "flash":
            # self.attention = QKVFlashAttention(channels, self.num_heads)
            raise ValueError(f"{attention_type} is not Implemented.")
        else:
            # split heads before split qkv
            # self.attention = QKVAttentionLegacy(self.num_heads)
            self.attention = QKVAttention(self.num_heads)

        self.use_attention_checkpoint = not (
            self.use_checkpoint or self.attention_type == "flash"
        )
        if encoder_channels is not None:
            assert attention_type != "flash"
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        #elf.proj_out = zero_module(conv_nd(dims, channels, channels, 1))
        self.proj_out = conv_nd(dims, channels, channels, 1)

    def forward(self, x, encoder_out=None):
        if encoder_out is None:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, encoder_out), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, encoder_out=None):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = checkpoint(
                self.attention, (qkv, encoder_out), (), self.use_attention_checkpoint
            )
        else:
            h = checkpoint(self.attention, (qkv,), (), self.use_attention_checkpoint)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange
        # from flash_attn.flash_attention import FlashAttention

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = FlashAttention(
            attention_dropout=attention_dropout, **factory_kwargs
        )
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        qkv, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        from einops import rearrange
        self.rearrange = rearrange


    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        qkv = qkv.half()

        qkv =   self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.n_heads
        ) 
        q, k, v = qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        q = q.reshape(bs*self.n_heads, ch, length)
        k = k.reshape(bs*self.n_heads, ch, length)
        v = v.reshape(bs*self.n_heads, ch, length)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight, dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.float()
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == 2 * ch * self.n_heads
            ek, ev = encoder_kv.chunk(2, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        # image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            dims=dims, # add
                            attention_type = None, # add
                            encoder_channels=self.model_channels, # add
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                dims=dims, # add
                attention_type = None, # add
                encoder_channels=self.model_channels, # add
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            dims=dims, # add
                            attention_type = None, # add
                            encoder_channels=self.model_channels, # add
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, condition, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # print(f"h:{h.shape}")# add
            # print(f"condition:{condition.shape}")# add
            # print(f"emb:{emb.shape}")# add
            h = module(h, emb, condition)
            hs.append(h)
        h = self.middle_block(h, emb, condition)
        # print(f"{h.shape}")# add
        for module in self.output_blocks:
            pop_hs = hs.pop() # add
            if pop_hs.shape[-1] != h.shape[-1]: # add
                h = h[..., :pop_hs.shape[-1]] # add
            h = torch.cat([h, pop_hs], dim=1)
            h = module(h, emb, condition)
            #print(f"{h.shape}")# add
        h = h.type(x.dtype)
        return self.out(h)

class UNetModel_original(nn.Module):
    """The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            dims=dims, # add
                            attention_type = None, # add
                            encoder_channels=self.model_channels, # add
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                            dims=dims, # add
                            attention_type = None, # add
                            encoder_channels=self.model_channels, # add
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            dims=dims, # add
                            attention_type = None, # add
                            encoder_channels=self.model_channels, # add
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            #ero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, t, x, y=None):
        """Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        timesteps = t
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        while timesteps.dim() > 1:
            print(timesteps.shape)
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)



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

if __name__ == "__main__":
    # Data Config
    Batch = 2
    F0_len = 48
    ids_len = 25
    inner_channel = 192

    # Data Dummy
    f0 = torch.randn(size=(Batch, 384, F0_len)) #  [Batch, inner_channel, f0_len]
    timesteps = torch.randint(low=0, high=10, size=(Batch,))
    conditioner = torch.randn(size=(Batch, inner_channel, F0_len))
    # f0_len = torch.randint(low=10, high=F0_len, size=(Batch,))    # noisyF0
    # ph_IDs = torch.randint(low=10, high=F0_len, size=(Batch, ids_len))
    # ph_IDs_len = torch.randint(low=10, high=ids_len, size=(Batch,))   # Condition (ph)
    # ph_ID_dur = torch.randint(low=10, high=F0_len, size=(Batch, ids_len))
    # NoteIDs= torch.randn(size=(Batch, ids_len))
    # NoteIDS_len=torch.randint(low=10, high=ids_len, size=(Batch,))
    # g=torch.randint(low=0, high=1, size=(Batch,))

    # model config
    # :param in_channels: channels in the input Tensor.
    in_channels=384
    # :param model_channels: base channel count for the model.
    model_channels=192
    # :param out_channels: channels in the output Tensor.
    out_channels=1
    # :param num_res_blocks: number of residual blocks per downsample.
    num_res_blocks=3
    # :param attention_resolutions: a collection of downsample rates at which
    #     attention will take place. May be a set, list, or tuple.
    #     For example, if this contains 4, then at 4x downsampling, attention
    #     will be used.
    attention_resolutions=[32,16,8,4]
    # :param dropout: the dropout probability.
    dropout=0
    # :param channel_mult: channel multiplier for each level of the UNet.
    channel_mult=[1, 2, 4, 8]
    # :param conv_resample: if True, use learned convolutions for upsampling and
    #     downsampling.
    conv_resample=True
    # :param dims: determines if the signal is 1D, 2D, or 3D.
    dims=1 # 1D
    # :param num_classes: if specified (as an int), then this model will be
    #     class-conditional with `num_classes` classes.
    num_classes=None # SpeakerEmbeddings
    # :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    use_checkpoint=False
    use_fp16=False
    # :param num_heads: the number of attention heads in each attention layer.
    num_heads=4
    # :param num_heads_channels: if specified, ignore num_heads and instead use
    #                            a fixed channel width per attention head.
    num_head_channels=-1
    # :param num_heads_upsample: works with num_heads to set a different number
    #                            of heads for upsampling. Deprecated.
    num_heads_upsample=-1
    # :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    use_scale_shift_norm=False
    # :param resblock_updown: use residual blocks for up/downsampling.
    resblock_updown=False
    # :param use_new_attention_order: use a different attention pattern for potentially
    #                                 increased efficiency.
    use_new_attention_order=False

    # model dummy
    model = UNetModel_original(
    #model = UNetModel(
        # image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=conv_resample,
        dims=dims,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,)

    #out = model(x=f0, timesteps=timesteps, condition=conditioner,)
    # padded process
    remainder = f0.size(-1) % 16
    if remainder != 0:
        B, C, length = f0.size()
        new = length + (16 - remainder)
        output = torch.zeros(size=(B,C,new))
        output[:,:, :length] = f0
        f0 = output
    out = model(x=f0, t=timesteps)
    print(out.shape)
    # from source.utils.read_write import read_yml
    # yml = read_yml("configs/model/consistency_wavenet.yaml")
    # # denoiser = Denoiser(hps=yml["net_g"]["hps"])
    # denoiser = UNet1D_Denoiser(hps=yml["net_g"]["hps"])
    # out = denoiser(f0,
    #             timesteps,
    #             f0_len,    # noisyF0
    #             ph_IDs,
    #             ph_IDs_len,   # Condition (ph)
    #             ph_ID_dur,
    #             NoteIDs=NoteIDs,
    #             NoteIDS_len=NoteIDS_len,
    #             g=g)
    # print(out)
    pass