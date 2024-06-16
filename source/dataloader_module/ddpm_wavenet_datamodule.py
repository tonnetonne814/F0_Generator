from typing import Any, Dict, Optional, Tuple #need
import os
import torch #need
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split#need
import torch
from lightning import LightningDataModule

from source.dataloader_module.datasets.f0_dataloader import AudioTextF0_Dataset, AudioTextF0_Collater
from source.dataloader_module.datasets.components.distributed_bucket_sampler import DistributedBucketSampler

# pytorch_lightning用データローダ構成
class TextAudioF0DataModule(LightningDataModule):
    """`LightningDataModule`

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    def __init__(self, hps ) -> None:
        super().__init__()
        self.hps = hps.__dict__['_content']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.collate_fn = AudioTextF0_Collater(return_ids=False)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        ### random_splitを行うと、DistributedBucketSamplerのlength値が消えるので未使用。preprocess時点でtrain等に分ける
        # dataset = list()
        # if self.hps["dataset"]["train_filelist_path"] is not None:
        #     dataset.append(AudioTextF0_Dataset(audiopaths_and_text=self.hps["dataset"]["train_filelist_path"],
        #                                 hps=self.hps))
        # if self.hps["dataset"]["valid_filelist_path"] is not None:
        #     dataset.append(AudioTextF0_Dataset(audiopaths_and_text=self.hps["dataset"]["valid_filelist_path"],
        #                                 hps=self.hps))
        # if self.hps["dataset"]["test_filelist_path"] is not None:
        #     dataset.append(AudioTextF0_Dataset(audiopaths_and_text=self.hps["dataset"]["test_filelist_path"],
        #                                 hps=self.hps))
        # dataset = ConcatDataset(datasets=dataset)
        # self.data_train, self.data_val, self.data_test = random_split(
        #         dataset=dataset,
        #         lengths=self.hps["dataset"]["split_ratio"],
        #         generator=torch.Generator().manual_seed(42),
        #     )

        # 存在するか否か
        train_path = self.hps["dataset"]["train_filelist_path"]
        if os.path.exists(train_path) is False:
            print(f"[WARNING] Does not exist {train_path}'")
        valid_path = self.hps["dataset"]["valid_filelist_path"]
        if os.path.exists(valid_path) is False:
            print(f"[WARNING] Does not exist {valid_path}'")
        test_path = self.hps["dataset"]["test_filelist_path"]
        if os.path.exists(test_path) is False:
            print(f"[WARNING] Does not exist {test_path}'")

        pass

    def train_dataloader(self):
        """Create and return the train dataloader."""
        training_dataset= AudioTextF0_Dataset(audiopaths_and_text=self.hps["dataset"]["train_filelist_path"], hps=self.hps)
        sampler = DistributedBucketSampler(
            dataset=training_dataset,
            batch_size=self.hps["training"]["batch_size"],
            boundaries=self.hps["dataset"]["boundaries"],
            num_replicas=torch.cuda.device_count(),
            rank=0,
            shuffle=self.hps["dataset"]["is_shuffle"]
        )
        return DataLoader(
            dataset=training_dataset,
            num_workers=self.hps["training"]["num_workers"],
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset = AudioTextF0_Dataset(audiopaths_and_text=self.hps["dataset"]["valid_filelist_path"], hps=self.hps),
            num_workers=self.hps["training"]["num_workers"],
            batch_size=2,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False
        )
    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=AudioTextF0_Dataset(audiopaths_and_text=self.hps["dataset"]["test_filelist_path"], hps=self.hps),
            num_workers=self.hps["training"]["num_workers"],
            batch_size=1,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False
        )


    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    pass