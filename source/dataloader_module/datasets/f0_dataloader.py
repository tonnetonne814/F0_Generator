
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
import torchaudio
from source.utils.audio_utils.singDB_loader import get_g2p_dict_from_tabledata, get_g2p_dict_from_training_data
from source.utils.audio_utils.mel_processing import spectrogram_torch
from source.utils.audio_utils.utils import load_wav_to_torch, load_filepaths_and_text
from source.utils.audio_utils import commons

# データセット読み込み君本体
class AudioTextF0_Dataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hps):

        self.basepath_list = load_filepaths_and_text(audiopaths_and_text, split="|")

        self.sampling_rate  = hps["audio_profile"]["sampling_rate"]
        self.hop_length     = hps["audio_profile"]["hop_length"]
        self.filter_length  = hps["audio_profile"]["filter_length"]
        self.win_length     = hps["audio_profile"]["win_length"]
        self.wav_max_ms     = hps["audio_profile"]["wav_max_ms"]
        self.wav_min_ms     = hps["audio_profile"]["wav_min_ms"]
        self.f0_max         = hps["audio_profile"]["f0_max"]

        self.oto2lab, self.ph_symbol_to_id,   self.id_to_ph_symbol, \
                      self.word_symbol_to_id,self.id_to_word_symbol = get_g2p_dict_from_tabledata(table_path=hps["oto_profile"]["oto2lab_path"], include_converter=True)
        with open(hps["oto_profile"]["noteid2hz_txt_path"], mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        self.id_to_hz = {}
        for idx, line in enumerate(lines):
            id, hz = line.split(",")

            self.id_to_hz[idx-1] = float(hz)

        # random.shuffle(self.basepath_list)
        self._filter()

    # 最小以下及び最大以上のデータを弾くフィルター関数
    def _filter(self):
        filtered_list = []
        lengths = []
        for basepath in tqdm(self.basepath_list, desc="Dataset Filtering..."):
            wav, sr = torchaudio.load(basepath[0]+".wav")
            ch, wav_len = wav.shape
            ms = wav_len / sr * 1000
            if self.wav_min_ms <= ms  and ms  <= self.wav_max_ms:
                filtered_list.append(basepath[0])
                lengths.append(wav_len // (2 * self.hop_length))
            else:
                print(f"EXCEEDED LENGTH : {basepath[0]}")
        self.basepath_list = filtered_list
        self.lengths = lengths

    def get_ph_vocab_size(self):
        return len(self.ph_symbol_to_id) + 1 # mask用

    def get_ph_ID(self, ph_list):
        sequence = []
        for symbol in ph_list:
            symbol_id = self.ph_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)

    def get_word_ID(self, word_list):
        sequence = []
        for symbol in word_list:
            symbol_id = self.word_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)

    def expand_note_info(self, ph_IDs, noteID, note_dur, n_ph_pooling):
        ph_IDs_lengths = torch.tensor(ph_IDs.size(1), dtype=torch.int64)
        ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths.view(1), ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
        noteID_lengths = torch.tensor(noteID.size(1), dtype=torch.int64)
        noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths.view(1), noteID.size(1)), 1).to(noteID.dtype) # [B, 1, ph_len]

        attn_mask     = torch.unsqueeze(noteID_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
        attn          = commons.generate_path(duration=torch.unsqueeze(n_ph_pooling,dim=1), mask=attn_mask)
        attn          = torch.squeeze(attn, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 
        # expand
        noteID        = torch.matmul(noteID.float().unsqueeze(1), attn)                                            # to [Batch, inner_channel, ph_len] 
        note_dur      = torch.matmul(note_dur.float().unsqueeze(1), attn)                     # to [Batch, inner_channel, ph_len] 

        return noteID.view(-1), note_dur.view(-1)

    def get_audio(self, filename):
        #print(filename)
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_dur_frame_from_e_ms(self, e_ms):
        e_ms = torch.tensor(e_ms,dtype=torch.float32)
        #frames = torch.ceil(  (e_ms/1000)*self.sampling_rate / self.hop_length )   # 切り上げ
        frames = torch.floor(  (e_ms / 1000)*self.sampling_rate / self.hop_length )   # 切り捨て
        frames = torch.diff(frames, dim=0, prepend=frames.new_zeros(1))
        #for idx in reversed(range(len(frames))):
        #    if idx == 0:
        #        continue
        #    frames[idx] = frames[idx] - frames[idx-1]
        return torch.tensor(frames, dtype=torch.int64)

    def get_ph_pooling_dur(self, ph_e, word_e):
        out = list()
        z_t = 0
        word_idx = 0
        for idx, e in enumerate(ph_e):
            idx += 1
            if word_e[word_idx] == e:
                out.append(int(idx - z_t))
                z_t = idx
                word_idx += 1
        return out

    def get_item(self, basepath):

        # labのデータは推論時存在しない
        f0          = torch.load(basepath + "_f0.pt"          )
        ph          = torch.load(basepath + "_ph.pt"          ) # ust or lab
        ph_e_ms     = torch.load(basepath + "_ph_e_ms.pt"     )     # lab
        #word        = torch.load(basepath + "_word.pt"        ) # ust
        #word_dur_ms = torch.load(basepath + "_word_dur_ms.pt" )     # lab
        word_e_ms   = torch.load(basepath + "_word_e_ms.pt"   )     # lab
        noteID      = torch.load(basepath + "_noteID.pt"      ) # ust
        notedur     = torch.load(basepath + "_notedur.pt"     ) # ust
        speakerID   = 0 # 未実装

        # tokenize and get duration
        ph_IDs              = self.get_ph_ID(ph_list=ph)
        #word_IDs            = self.get_word_ID(word_list=word)
        ph_frame_dur        = self.get_dur_frame_from_e_ms(e_ms=ph_e_ms)
        n_ph_pooling        = self.get_ph_pooling_dur(ph_e=ph_e_ms, word_e=word_e_ms)
        f0_len            = len(f0)
        notes = []
        for id in noteID:
            notes += [self.id_to_hz[int(id)]]
        noteID, notedur     = self.expand_note_info(ph_IDs=ph_IDs.view(1,-1),
                                                    noteID      =torch.tensor(notes, dtype=torch.float32).view(1,-1),
                                                    note_dur    =torch.tensor(notedur,dtype=torch.int64).view(1,-1),
                                                    n_ph_pooling=torch.tensor(n_ph_pooling, dtype=torch.int64).view(1,-1))

        # padの影響で1ずれる。無問題と妄想
        if f0_len != int(torch.sum(ph_frame_dur)):
            ph_frame_dur[-1] += 1

        # 保障
        assert sum(n_ph_pooling) == len(ph_IDs)
        #assert spec_len == len(vuv)
        assert len(ph_IDs) == len(noteID)
        #assert len(word_IDs) == len(word_dur_ms)
        #assert len(word_IDs) == len(word_e_ms)

        return (torch.tensor(f0          ,            dtype=torch.float32),
                #torch.tensor(vuv,                       dtype=torch.int64)+1,       # maskを0とする。
                torch.tensor(ph_IDs,                    dtype=torch.int64)+1,        # maskを0とする。
                torch.tensor(ph_frame_dur,              dtype=torch.int64),
                #torch.tensor(word_dur_ms,               dtype=torch.float32) / 1000, # ここで秒になる
                torch.tensor(noteID,                    dtype=torch.float32) ,       # maskを0とする。z
                torch.tensor(speakerID,                 dtype=torch.int64)  )

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)


class AudioTextF0_Collater():
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[0].size(0) for x in batch] ),dim=0, descending=True )

        max_f0_len              = max([len(x[0]) for x in batch])
        #max_vuv_len            = max([len(x[1]) for x in batch])
        max_ph_IDs_len          = max([len(x[1]) for x in batch])
        max_ph_frame_dur_len    = max([len(x[2]) for x in batch])
        max_noteID_len          = max([len(x[3]) for x in batch])

        f0_lengths              = torch.LongTensor(len(batch))
        #vuv_lengths             = torch.LongTensor(len(batch))
        ph_IDs_lengths          = torch.LongTensor(len(batch))
        ph_frame_dur_lengths    = torch.LongTensor(len(batch))
        noteID_lengths          = torch.LongTensor(len(batch))
        spkids                  = torch.LongTensor(len(batch))

        f0_padded               = torch.FloatTensor(len(batch), max_f0_len)
        #vuv_padded              = torch.LongTensor(len(batch),  max_vuv_len) 
        ph_IDs_padded           = torch.LongTensor(len(batch),   max_ph_IDs_len)
        ph_frame_dur_padded     = torch.LongTensor(len(batch),   max_ph_frame_dur_len)
        noteID_padded           = torch.FloatTensor(len(batch), max_noteID_len)

        f0_padded.zero_()
        ph_IDs_padded.zero_()
        ph_frame_dur_padded.zero_()
        noteID_padded.zero_()
        spkids.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            f0 = row[0]
            f0_padded[i, :f0.size(0)] = f0
            f0_lengths[i] = f0.size(0)

            #vuv = row[1]
            #vuv_padded[i, :vuv.size(0)] = vuv
            #vuv_lengths[i] = vuv.size(0)

            ph_IDs = row[1]
            ph_IDs_padded[i, :ph_IDs.size(0)] = ph_IDs
            ph_IDs_lengths[i] = ph_IDs.size(0)

            ph_frame_dur = row[2]
            ph_frame_dur_padded[i,     :ph_frame_dur.size(0)] = ph_frame_dur
            ph_frame_dur_lengths[i] =   ph_frame_dur.size(0)

            noteID = row[3]
            noteID_padded[i, :noteID.size(0)] = noteID
            noteID_lengths[i] = noteID.size(0)

            spkids[i] = row[4]


        # 次元調整
        f0_padded = torch.unsqueeze(f0_padded, dim=1)

        if self.return_ids:
            return  f0_padded,              f0_lengths,             \
                    ph_IDs_padded,          ph_IDs_lengths,         \
                    ph_frame_dur_padded,                            \
                    noteID_padded,          noteID_lengths,         \
                    spkids,                                   \
                    ids_sorted_decreasing

        return  f0_padded,              f0_lengths,             \
                ph_IDs_padded,          ph_IDs_lengths,         \
                ph_frame_dur_padded,                            \
                noteID_padded,          noteID_lengths,         \
                spkids
