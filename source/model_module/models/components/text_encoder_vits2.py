import torch.nn as nn
import torch
import math
from source.utils.audio_utils import commons
from source.model_module.models.components import attentions

class TextEncoder_VITS2(nn.Module):
  def __init__(self,
      n_vocab,
      n_note,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      gin_channels=0):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.x_emb = nn.Embedding(n_vocab, hidden_channels)
    self.ph_w_idx_emb = nn.Embedding(2,  hidden_channels)
    self.dur_emb = nn.Linear(1, hidden_channels)
    nn.init.normal_(self.x_emb.weight, 0.0, hidden_channels**-0.5)
    nn.init.normal_(self.ph_w_idx_emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder_VITS2(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      gin_channels=self.gin_channels)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, w_dur_ms=None, ph_w_idx=None,  g=None):
    x = self.x_emb(x) / math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]

    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if w_dur_ms is not None:
      w_dur_ms = self.dur_emb(w_dur_ms.unsqueeze(2)).permute(0,2,1) / math.sqrt(self.hidden_channels)
      x = x + w_dur_ms
    if ph_w_idx is not None:
      ph_w_idx = torch.diff(ph_w_idx, dim=1, prepend=ph_w_idx.new_zeros(x.size(0), 1)) >= 0
      ph_w_idx = self.ph_w_idx_emb(ph_w_idx.long()).permute(0,2,1) / math.sqrt(self.hidden_channels)  # [B, T_ph, H]
      x = x + ph_w_idx

    enc_in = x * x_mask
    x = self.encoder(enc_in, x_mask, g=g)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask
