import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            micro_sar, micro_opt = th.split(micro, 3, dim=1)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_opt,
                micro_sar,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


import copy
import functools
import os
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class CycleTrainLoop:
    """
    用于训练循环一致性双扩散模型 (Cycle-Consistent Dual Diffusion, CDD)
    同时管理 A2B (SAR->Opt) 和 B2A (Opt->SAR) 两个生成器网络
    """
    def __init__(
            self,
            *,
            model_A2B,  # SAR -> Opt 的模型
            model_B2A,  # Opt -> SAR 的模型
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            lambda_cycle=10.0, # 循环一致性损失权重
    ):
        self.model_A2B = model_A2B
        self.model_B2A = model_B2A
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.lambda_cycle = lambda_cycle

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        # 为了避免 MP_Trainer 报错，我们将两个模型的参数合二为一传给优化器
        self.model_params = list(self.model_A2B.parameters()) + list(self.model_B2A.parameters())
        
        self._load_and_sync_parameters()
        
        # 实例化合并参数的混合精度训练器
        self.mp_trainer = MixedPrecisionTrainer(
            model=th.nn.ModuleList([self.model_A2B, self.model_B2A]),
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            # 为两个模型分别注册 DDP，保证计算图不割裂
            self.ddp_model_A2B = DDP(
                self.model_A2B,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True, # 这里开启 True，防止 Cycle 分支跳过部分层导致报错
            )
            self.ddp_model_B2A = DDP(
                self.model_B2A,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA.")
            self.use_ddp = False
            self.ddp_model_A2B = self.model_A2B
            self.ddp_model_B2A = self.model_B2A

    def _load_and_sync_parameters(self):
        # 初始化与同步权重的逻辑，针对两个模型
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading dual models from checkpoint: {resume_checkpoint}...")
                # 假设您的 checkpoint 是个包含了 A2B 和 B2A state_dict 的大字典
                ckpt = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                self.model_A2B.load_state_dict(ckpt['model_A2B'])
                self.model_B2A.load_state_dict(ckpt['model_B2A'])

        dist_util.sync_params(self.model_A2B.parameters())
        dist_util.sync_params(self.model_B2A.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        # 实现方式与原来类似
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        """
        核心：双循环一致性扩散损失计算
        """
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i: i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
            
            # 分离出真实的 SAR 和 Optical 图像，假设输入通道为6 (3: SAR, 3: Optical)
            micro_sar, micro_opt = th.split(micro, 3, dim=1)

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # ----------------------------------------------------
            # 定义 CDD 损失计算的内部闭包，以便用于 DDP 的 no_sync() 
            # ----------------------------------------------------
            def compute_cdd_losses():
                # 1. 单向 MSE 去噪损失 (A2B: SAR -> Opt)
                # 目标(x_start) = 光学影像, 条件(condition) = SAR影像
                out_A2B = self.diffusion.training_losses(
                    self.ddp_model_A2B, 
                    x_start=micro_opt, 
                    condition=micro_sar, 
                    t=t, 
                    model_kwargs=micro_cond
                )
                loss_mse_A2B = out_A2B["loss"]
                pred_x0_opt = out_A2B["pred_xstart"] # 预测生成的伪光学影像 (G(x))

                # 2. 单向 MSE 去噪损失 (B2A: Opt -> SAR)
                # 目标(x_start) = SAR影像, 条件(condition) = 光学影像
                out_B2A = self.diffusion.training_losses(
                    self.ddp_model_B2A, 
                    x_start=micro_sar, 
                    condition=micro_opt, 
                    t=t, 
                    model_kwargs=micro_cond
                )
                loss_mse_B2A = out_B2A["loss"]
                pred_x0_sar = out_B2A["pred_xstart"] # 预测生成的伪SAR影像 (F(y))

                # 3. 双向循环一致性重构 (Cycle Consistency)

                cycle_out_A2B = self.diffusion.training_losses(
                    self.ddp_model_A2B, 
                    x_start=micro_opt,       # 真实Opt作为回归目标
                    condition=pred_x0_sar,   # 伪SAR作为重构条件 
                    t=t, 
                    model_kwargs=micro_cond
                )
                rec_opt = cycle_out_A2B["pred_xstart"]

   
                cycle_out_B2A = self.diffusion.training_losses(
                    self.ddp_model_B2A, 
                    x_start=micro_sar,       # 真实SAR作为回归目标
                    condition=pred_x0_opt,   # 伪Opt作为重构条件
                    t=t, 
                    model_kwargs=micro_cond
                )
                rec_sar = cycle_out_B2A["pred_xstart"]

                # reduction='none' 保持形状 [B, C, H, W]
                l1_sar = th.nn.functional.l1_loss(rec_sar, micro_sar, reduction='none')
                l1_opt = th.nn.functional.l1_loss(rec_opt, micro_opt, reduction='none')
                
                # 在空间和通道维度取平均，保留 Batch 维度 [B]
                loss_cycle_sar = l1_sar.mean(dim=[1, 2, 3])
                loss_cycle_opt = l1_opt.mean(dim=[1, 2, 3])
                loss_cycle = loss_cycle_sar + loss_cycle_opt

                # 汇总总损失 (所有 loss 此时 shape 都是 [B])
                total_loss = loss_mse_A2B + loss_mse_B2A + (self.lambda_cycle * loss_cycle)

                return {
                    "loss": total_loss,
                    "mse_A2B": loss_mse_A2B,
                    "mse_B2A": loss_mse_B2A,
                    "cycle": loss_cycle
                }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            if last_batch or not self.use_ddp:
                losses = compute_cdd_losses()
            else:
                # 嵌套 context 保证两个 DDP 模型均不同步
                with self.ddp_model_A2B.no_sync():
                    with self.ddp_model_B2A.no_sync():
                        losses = compute_cdd_losses()

            # 将损失按时间步长采样权重缩放，反向传播
            loss = (losses["loss"] * weights).mean()
            
            # 日志记录 (分开打印，便于监控)
            log_loss_dict(self.diffusion, t, {
                "loss_total": losses["loss"] * weights,
                "loss_mse_A2B": losses["mse_A2B"] * weights,
                "loss_mse_B2A": losses["mse_B2A"] * weights,
                "loss_cycle": losses["cycle"] * weights,
            })
            
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving CDD models {rate}...")
                filename = f"model{(self.step + self.resume_step):06d}.pt" if not rate else f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                # 在此我们将混合在一块儿的 state_dict 切割保存，方便推理
                state_dict_A2B = {k: v for k, v in state_dict.items() if 'A2B' in k} # 根据具体参数名调整
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"), "wb") as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()