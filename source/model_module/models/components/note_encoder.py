import torch.nn as nn
import torch
import math
from source.utils.audio_utils import commons

class NoteEncoder(nn.Module):
    def __init__(self,
                 n_note,
                 hidden_channels,
                 ):
        super(NoteEncoder, self).__init__()
        self.hidden_ch = hidden_channels
        #self.note_emb = nn.Embedding(num_embeddings=n_note, 
        #                             embedding_dim=hidden_channels)
        self.dur_emb  = nn.Linear   (in_features=1,  # max_len
                                     out_features=hidden_channels)
        self.note_emb  = nn.Linear   (in_features=1,  # max_len
                                     out_features=hidden_channels)
        self.hidden_channels=hidden_channels

        ### wordIDをselfAttnして加算してもよいかも
    def forward(self, noteID, noteID_lengths, note_dur=None):
        noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths, noteID.size(1)), 1).to(noteID.dtype)

        emb = self.note_emb(noteID.unsqueeze(2)) / math.sqrt(self.hidden_ch)
        if note_dur is not None:
          emb += self.dur_emb(note_dur.unsqueeze(2)) / math.sqrt(self.hidden_ch)
        return emb.permute(0,2,1), noteID_mask

