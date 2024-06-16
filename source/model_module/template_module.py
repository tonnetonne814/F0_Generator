from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule

class DDPMWaveNetModule(LightningModule):
    """Example of a `LightningModule`.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net_g: torch.nn.Module,
        # net_d: torch.nn.Module, #GAN
        optimizer_g: torch.optim.Optimizer,
        # optimizer_d: torch.optim.Optimizer, #GAN
        scheduler_g: torch.optim.lr_scheduler,
        # scheduler_d: torch.optim.lr_scheduler, #GAN
        compile: bool,
    ) -> None:
        super().__init__()
        """
        初期化処理。ここでの引数は、config上で設定できる。
        GAN処理等の2モデル以上の学習であれば"#GAN"部分参照
        """

        self.net_g = net_g
        self.optimizer_g = optimizer_g
        self.scheduler_g = scheduler_g
        # self.optimizer_d = optimizer_d #GAN
        # self.scheduler_d = scheduler_d #GAN
        self.is_compile = compile

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        x, y = batch
        logits = self.net_g(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

        """

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
        """
        ### GAN # 片側の学習処理記述方法 ###
        # optimizer_g, optimizer_d = self.optimizers()
        # self.net_g.train() #GAN
        # self.net_d.eval() #GAN
        # self.toggle_optimizer(optimizer_g) #GAN
        #
        # outputs = self.net_g(A, B, C) #GAN
        # losses = 0
        #
        # self.manual_backward(losses)    # 勾配計算 #GAN
        # optimizer_g.step()              # パラメータ更新 #GAN
        # optimizer_g.zero_grad()         # 勾配初期化 #GAN
        ########################
        # return None

        losses = 0
        return losses

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a Training Epoch Ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        """
        # self.net_A.eval() #GAN
        # self.net_B.eval() #GAN

        # valid process

        # self.net_A.train() #GAN
        # self.net_B.train() #GAN
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a Validation Epoch Ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        """
        # self.net_A.eval() #GAN
        # self.net_B.eval() #GAN

        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a Test Epoch Ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the Beginning of Fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        # self.automatic_optimization = False # 手動でBackProp処理を回す #GAN

        # Fit時のみ、モデルをcompileする
        if self.is_compile and stage == "fit":
            self.net_g = torch.compile(self.net_g)
            # self.net_d = torch.compile(self.net_d) #GAN

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        # Single Model #
        optimizer_g = self.optimizer_g(params=self.trainer.model.parameters()) # モデルの全てのパラメータを渡す
        if self.scheduler_g is not None:
            scheduler_g = self.scheduler_g(optimizer=optimizer_g)
            return {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer_g}
        ################

        #GAN
        # optimizer_g = self.optimizer_g(params=self.net_g.parameters())
        # optimizer_d = self.optimizer_d(params=self.net_d.parameters())
        # if self.scheduler_g is not None:
        #     scheduler_g = self.scheduler_g(optimizer=optimizer_g)
        # else:
        #     scheduler_g = None
        # if self.scheduler_d is not None:
        #     scheduler_d = self.scheduler_d(optimizer=optimizer_d)
        # else:
        #     scheduler_d = None
        # return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
        ###########
