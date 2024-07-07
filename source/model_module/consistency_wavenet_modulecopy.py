from typing import Any, Dict, Tuple
import torch, os
from lightning import LightningModule
from source.utils.plot_utils import generate_graph, generate_graph_overwrite
from source.utils.audio_utils.commons import clip_grad_value_
from source.utils.audio_utils.f0_extractor import f0_exchange

class ConsistencyWaveNetModule(LightningModule):
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
        valid_sampling_in_n_epoch: int,
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
        self.valid_sampling_in_n_epoch = valid_sampling_in_n_epoch

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        global_step = self.trainer.global_step
        estimated_stepping_batches = self.trainer.estimated_stepping_batches
        ema_decay = self.net_g.consistency.ema_update(
                                model = self.net_g.noise_predictor,
                                model_ema = self.net_g.ema,
                                global_step = global_step,
                                estimated_stepping_batches = estimated_stepping_batches
                                )
        self.log("train/ema_decay", ema_decay, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass

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

        f0,             f0_lengths,\
        ph_IDs,         ph_IDs_lengths,\
        ph_frame_dur,\
        noteID,         noteID_lengths,\
        speakerID, _ = batch

        global_step = self.trainer.global_step
        estimated_stepping_batches = self.trainer.estimated_stepping_batches

        if f0.device.type == "cpu":
          pass

        loss_CT = self.net_g(f0 = f0,
                             f0_len = f0_lengths,
                             IDs = ph_IDs,
                             IDs_len = ph_IDs_lengths,
                             IDs_dur = ph_frame_dur,
                             NoteIDs = noteID,
                             NoteID_len = noteID_lengths,
                             g = speakerID,
                             global_step = global_step,
                             estimated_stepping_batches = estimated_stepping_batches
                             )

        return loss_CT

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
        epoch = self.current_epoch
        global_step = self.global_step
        ###GAN # Generatorの学習処理記述方法 ###
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

        loss_CT, bins = self.model_step(batch)
        self.log("train/loss_CT", loss_CT, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/bins", bins, on_step=True, on_epoch=True, prog_bar=False)

        # .zero_grad()する前に行う
        self.log("gradient_norm/generator", clip_grad_value_(self.net_g.parameters(), None),\
                                                on_step=True, on_epoch=True, prog_bar=False)
        return loss_CT

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a Training Epoch Ends."
        optimizer_g = self.optimizers()
        last_lr = optimizer_g.optimizer.param_groups[0]["lr"]
        self.log("optimizer_g/lr [x10^4]", last_lr*10000, on_step=False, on_epoch=True, prog_bar=False)

        log_dir = self.logger.log_dir.replace("tensorboard/version_0", "checkpoints")
        self.trainer.save_checkpoint(filepath=os.path.join(log_dir, "last_epoch.ckpt"))
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
        epoch = self.current_epoch
        global_step = self.global_step
        # self.net_A.eval() #GAN
        # self.net_B.eval() #GAN
        # self.net_A.train() #GAN
        # self.net_B.train() #GAN

        self.val_batch = batch # 最終時の試し生成の結果見る用
        loss_CT, bins = self.model_step(batch)
        self.log("val/loss_CT", loss_CT, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/bins", bins, on_step=True, on_epoch=True, prog_bar=False)

        # このロスに基づいてschedulerが動く
        self.log("monitor", loss_CT, on_step=False, on_epoch=True, prog_bar=True)
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a Validation Epoch Ends."

        epoch = self.current_epoch
        global_step = self.global_step
        if epoch % self.valid_sampling_in_n_epoch == 0:
            # print(f"Try Sampling... Epoch:{epoch} GlobalStep:{global_step}")
            f0_gt=self.val_batch[0][0][0].view(1, 1,-1)
            f0_gt_len=self.val_batch[1][0].view(-1)
            ph_IDs=self.val_batch[2][0].view(1,-1)
            ph_IDs_len=self.val_batch[3][0].view(-1)
            ph_IDs_dur=self.val_batch[4][0].view(1,-1)
            NoteIDs=self.val_batch[5][0].view(1,-1)
            NoteID_len=self.val_batch[6][0].view(-1)
            speakerID=self.val_batch[7][0].view(-1)
            basepath = self.val_batch[8][0]

            f0_pd = self.net_g.sampling(condition=[ph_IDs, ph_IDs_len, ph_IDs_dur,
                                        NoteIDs, NoteID_len, None, speakerID])

            self.log("val/f0_RMSE", torch.sqrt(torch.nn.functional.mse_loss(f0_pd, f0_gt)), on_step=False, on_epoch=True, prog_bar=False)

            f0_pd = f0_pd[0][0].to('cpu').detach().numpy().copy()
            f0_gt = f0_gt[0][0].to('cpu').detach().numpy().copy()

            f0_pd_image= generate_graph(vector=f0_pd ,
                                        label="F0 pd",
                                        color="blue",
                                        x_label = 'Frames',
                                        y_label = "Hz")
            f0_gtpd_image = generate_graph_overwrite(vector1=f0_gt,
                                                     vector2=f0_pd,
                                                     title="Comparison",
                                                     x_label = 'Frames',
                                                     y_label = "Hz")
            self.logger.experiment.add_image("val/f0_pd", torch.from_numpy(f0_pd_image).clone().permute(2, 0, 1), epoch)
            self.logger.experiment.add_image("val/f0_gt-pd", torch.from_numpy(f0_gtpd_image).clone().permute(2, 0, 1), epoch)

            audio_f0pd, audio_original, fs = f0_exchange(basepath+".wav", f0_pd)
            self.logger.experiment.add_audio("val/f0_exchanged_audio", audio_f0pd, epoch, fs)
            if global_step == 0:
                f0_gt_image= generate_graph(vector=f0_gt,
                                            label="F0 gt",
                                            color="red",
                                            x_label='Frames',
                                            y_label="Hz")
                self.logger.experiment.add_image('val/f0_gt', torch.from_numpy(f0_gt_image).clone().permute(2, 0, 1), self.current_epoch)
                self.logger.experiment.add_audio("val/gt_audio", audio_original, epoch, fs)


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
        idx = str(batch_idx).zfill(3)

        f0_gt=batch[0][0][0].view(1, 1,-1)
        f0_gt_len=batch[1][0].view(-1)
        ph_IDs=batch[2][0].view(1,-1)
        ph_IDs_len=batch[3][0].view(-1)
        ph_IDs_dur=batch[4][0].view(1,-1)
        NoteIDs=batch[5][0].view(1,-1)
        NoteID_len=batch[6][0].view(-1)
        speakerID=batch[7][0].view(-1)
        basepath = batch[8][0]

        f0_pd = self.net_g.sampling(condition=[ph_IDs, ph_IDs_len, ph_IDs_dur,
                                    NoteIDs, NoteID_len, None, speakerID])

        self.log("test_f0_RMSE", torch.sqrt(torch.nn.functional.mse_loss(f0_pd, f0_gt)), on_step=False, on_epoch=True, prog_bar=False)

        f0_pd = f0_pd[0][0].to('cpu').detach().numpy().copy()
        f0_gt = f0_gt[0][0].to('cpu').detach().numpy().copy()

        f0_pd_image= generate_graph(vector=f0_pd ,
                                    label="F0 pd",
                                    color="blue",
                                    x_label = 'Frames',
                                    y_label = "Hz")
        f0_gtpd_image = generate_graph_overwrite(vector1=f0_gt,
                                                    vector2=f0_pd,
                                                    title="Comparison",
                                                    x_label = 'Frames',
                                                    y_label = "Hz")
        self.logger.experiment.add_image(f"test_{idx}/f0_pd", torch.from_numpy(f0_pd_image).clone().permute(2, 0, 1), self.current_epoch)
        self.logger.experiment.add_image(f"test_{idx}/f0_gt-pd", torch.from_numpy(f0_gtpd_image).clone().permute(2, 0, 1), self.current_epoch)

        f0_gt_image= generate_graph(vector=f0_gt,
                                    label="F0 gt",
                                    color="red",
                                    x_label='Frames',
                                    y_label="Hz")
        self.logger.experiment.add_image(f'test_{idx}/f0_gt', torch.from_numpy(f0_gt_image).clone().permute(2, 0, 1), self.current_epoch)


        audio_f0pd, audio_original, fs = f0_exchange(basepath+".wav", f0_pd)
        self.logger.experiment.add_audio(f"test_{idx}/audio_gt", audio_original, 0, fs)
        self.logger.experiment.add_audio(f"test_{idx}/audio_pd", audio_f0pd, 0, fs)
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a Test Epoch Ends."""
        epoch = self.current_epoch
        global_step = self.global_step
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the Beginning of Fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        #GAN
        # self.automatic_optimization = False # 手動でBackProp処理を回す

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
                    "monitor": "monitor",
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
