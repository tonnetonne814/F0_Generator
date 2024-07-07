
import torch.nn as nn
import torch
import math

class ConsistencyProcessor(nn.Module):
    def __init__(self, hps ):
        super(ConsistencyProcessor, self).__init__()
        self.data_std = hps["Consistency"]["data_std"]
        self.time_min = hps["Consistency"]["time_min"]
        self.time_max = hps["Consistency"]["time_max"]
        self.bins_min = hps["Consistency"]["bins_min"]
        self.bins_max = hps["Consistency"]["bins_max"]
        self.bins_rho = hps["Consistency"]["bins_rho"]
        self.initial_ema_decay = hps["Consistency"]["initial_ema_decay"]
        # self.num_samples = hps["Consistency"]["num_samples"]
        self.sample_steps = hps["Consistency"]["sample_steps"]
        self.is_distill = hps["Consistency"]["is_distill"]
        self.is_denoise_clip = hps["Consistency"]["is_denoise_clip"]

        loss_fn = hps["Consistency"]["LOSS_FN"]
        if loss_fn == "L1":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "L2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"{loss_fn} is not implemented.")

    # EDM Denoise Calclation
    def denoise(self, noisy_input, noisy_output, times):

        # EDM用係数計算
        c_skip = self.data_std**2 / ( (times - self.time_min).pow(2) + self.data_std**2)
        c_out = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        # (論文5式)
        c_skip_X = self.Xt_time_product(noisy_input, c_skip)
        c_out_F_theta = self.Xt_time_product(noisy_output, c_out)
        f_theta = c_skip_X + c_out_F_theta

        if self.is_denoise_clip is True:
            return f_theta.clamp(-1.0, 1.0)

        return f_theta

    def get_noisy_f0(self, f0: torch.Tensor,
                     global_step: int,
                     estimated_stepping_batches: int):
        bins = self.bins(global_step, estimated_stepping_batches)
        noise = torch.randn(f0.shape, device=f0.device)
        timesteps = torch.randint(0,
                                  bins - 1,
                                  (f0.shape[0],),
                                  device=f0.device,
                                  ).long()

        current_times = self.timesteps_to_times(timesteps, bins)
        current_noisy_f0 = f0 + self.Xt_time_product(noise, current_times,)

        next_times = self.timesteps_to_times(timesteps + 1, bins)
        next_noisy_f0 = f0 + self.Xt_time_product(noise, next_times)

        return current_times, current_noisy_f0, next_times, next_noisy_f0, bins

    # 予測一貫性保証計算
    def get_loss(self, denoised_model_output, denoised_ema_model_output):
        return self.loss_fn(denoised_model_output, denoised_ema_model_output)

    @torch.no_grad()
    def ema_update(self, model, model_ema,
                   global_step: int, estimated_stepping_batches: int):

        param = [p.data for p in model.parameters()]
        param_ema = [p.data for p in model_ema.parameters()]

        # μ計算
        ema_decay = self.ema_decay(global_step, estimated_stepping_batches)

        # (論文8式)
        torch._foreach_mul_(param_ema, ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1-ema_decay)
        return ema_decay

    def ema_decay(self, global_step: int, estimated_stepping_batches: int):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins(global_step, estimated_stepping_batches))

    def bins(self, global_step: int, estimated_stepping_batches: int) -> int:
        return math.ceil(
            math.sqrt(
                #self.trainer.global_step
                #/ self.trainer.estimated_stepping_batches
                global_step / estimated_stepping_batches
                * (self.bins_max**2 - self.bins_min**2)
                + self.bins_min**2
            )
        )

    @staticmethod
    def Xt_time_product(Xt: torch.Tensor, times: torch.Tensor):
        # return torch.einsum("b c h w, b -> b c h w", Xt, times)
        return torch.einsum("b l f, b -> b l f", Xt, times)

    # 論文p4左最下部式
    def timesteps_to_times(self, timesteps: torch.LongTensor, bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )

    def get_time(self, device):
        return torch.tensor(data=[self.time_max], device=device)

    def get_time_min(self):
        return self.time_min

    # 多段サンプリング用の時刻計算
    def get_multistep_times(self, steps, device):
        _timesteps = list(
            reversed(range(0, self.bins_max, self.bins_max // steps - 1))
        )[1:]
        _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

        times = self.timesteps_to_times(
            torch.tensor(_timesteps, device=device), bins=150
        )
        return times

if __name__ == "__main__":

    data_std = 0.5
    time_min = 0.002
    times = 2.5152
    c_skip = data_std**2 / ((times - time_min)**2 + data_std**2)
    c_out = data_std * times / ((times**2 + data_std**2)**0.5)
    print(c_skip)
    print(c_out)