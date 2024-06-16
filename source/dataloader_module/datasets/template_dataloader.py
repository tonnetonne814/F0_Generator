import random
import torch
import torch.utils.data

# 要素を一つ一つ読み込む処理
class TemplateDataset(torch.utils.data.Dataset):
    def __init__(self, hps):
        self.hps = hps

        # basepath_listにファイルのパス群を入れる処理を書く
        self.basepath_list = list()

        #シャッフルする
        random.seed(hps["seed"])
        random.shuffle(self.basepath_list)

    # ファイルを一つ一つ取得する処理をする
    def get_item(self, basepath):
        dataA = torch.load(basepath + "dataA.pt")
        dataB = torch.load(basepath + "dataB.pt")
        dataC = torch.load(basepath + "dataC.pt")

        # A,B,Cとデータを返す
        return (torch.tensor(dataA, dtype=torch.float32),
                torch.tensor(dataB, dtype=torch.float32),
                torch.tensor(dataC, dtype=torch.float32),
                )

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)

# バッチ内部の要素達の長さを、最大長に揃えて行列化させる処理
class TemplateCollater():
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[0].size(0) for x in batch] ),dim=0, descending=True )

        max_dataA_len = max([len(x[0]) for x in batch])
        max_dataB_len = max([len(x[1]) for x in batch])
        max_dataC_len = max([len(x[2]) for x in batch])

        dataA_lengths = torch.LongTensor(len(batch))
        dataB_lengths = torch.LongTensor(len(batch))
        dataC_lengths = torch.LongTensor(len(batch))

        dataA_padded = torch.FloatTensor(len(batch), max_dataA_len)
        dataB_padded = torch.LongTensor(len(batch),   max_dataB_len)
        dataC_padded = torch.LongTensor(len(batch),   max_dataC_len)

        dataA_padded.zero_()
        dataB_padded.zero_()
        dataC_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            dataA = row[0]
            dataA_padded[i, :dataA.size(0)] = dataA
            dataA_lengths[i] = dataA.size(0)

            dataB = row[1]
            dataB_padded[i, :dataB.size(0)] = dataB
            dataB_lengths[i] = dataB.size(0)

            dataC = row[2]
            dataC_padded[i, :dataC.size(0)] = dataC
            dataC_lengths[i] = dataC.size(0)

        if self.return_ids:
            return  dataA_padded, dataA_lengths,\
                    dataB_padded, dataB_lengths,\
                    dataC_padded, dataC_lengths,\
                    ids_sorted_decreasing

        return  dataA_padded, dataA_lengths,\
                dataB_padded, dataB_lengths,\
                dataC_padded, dataC_lengths,\
                ids_sorted_decreasing
