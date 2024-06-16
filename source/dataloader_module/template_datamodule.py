from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, random_split
import torch
from lightning import LightningDataModule

from source.dataloader_module.datasets.template_dataloader import TemplateDataset, TemplateCollater
from source.dataloader_module.datasets.components.distributed_bucket_sampler import DistributedBucketSampler

# pytorch_lightning用データローダ構成
class TemplateDataModule(LightningDataModule):
    def __init__(self, hps,) -> None:
        super().__init__()
        self.hps = hps.__dict__['_content']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.collate_fn = TemplateCollater(return_ids=False)

    def setup(self, stage: str = None) -> None:


        trainset = TemplateDataset(self.hps["dataset"]["data_dir"])
        testset = TemplateDataset(self.hps["dataset"]["data_dir"])

    def train_dataloader(self):
        sampler = DistributedBucketSampler(
            self.train_dataset,
            self.hps.ml.batch_size,
            boundaries=self.hps.dataset.boundaries,
            num_replicas=1,
            rank=0,
            shuffle=self.dataset.shuffle
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.hps.ml.num_workers,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            num_workers=self.hps.ml.num_workers,
            batch_size=4,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

if __name__ == "__main__":
    pass