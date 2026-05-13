# Copyright 2026 Feiyan Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import datetime
import logging
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torchaudio
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset.multiaudio_dataset import (
    build_audio_dataset_from_config,
    collate_fn_filter_none,
    derive_audio_shapes,
)
from .model.flow_matching import FlowMatching, log_normal_sample
from .model.networks import get_wavflow_model
from .trainer_utils import (
    add_weight_decay,
    adjust_learning_rate,
    is_main_process,
    load_config,
    log_audio_backend_info,
    wavflowConfig,
)


logger = logging.getLogger(__name__)


# Re-export for backward compatibility (e.g. `from wavflow.trainer import load_config`).
__all__ = ["Trainer", "load_config", "wavflowConfig"]


class Trainer:
    def _setup_multiaudio_dataset(self, job_config):
        """Set up CSV dataset and distributed dataloader."""
        batch_size = job_config.training.batch_size
        num_workers = job_config.training.num_workers
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        dataset = build_audio_dataset_from_config(job_config)
        self.train_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

        self.dataloader_train = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            collate_fn=collate_fn_filter_none,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=job_config.training.dataloader_pin_memory,
            prefetch_factor=job_config.training.dataloader_prefetch_factor
            if num_workers > 0
            else None,
        )

    def _setup_model(self, job_config: wavflowConfig) -> None:
        """Set up model, flow matching, and related config."""
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False

        # Clear GPU cache before loading model to free up memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Load empty_string_feat on CPU first to save GPU memory.
        # Auto-download from HuggingFace if the configured path is missing.
        from wavflow.utils.auto_download import resolve_weight

        empty_string_feat = torch.load(
            resolve_weight(
                job_config.model.empty_string_ckpt, weight_key="empty_string"
            ),
            weights_only=True,
            map_location="cpu",
        )[0]

        latent_seq_len, _, target_samples = derive_audio_shapes(
            int(job_config.data.target_sample_rate),
            float(job_config.data.audio_duration),
            int(job_config.data.audio_dim),
        )
        self.target_samples = target_samples

        model = get_wavflow_model(
            job_config.model.name,
            latent_seq_len=latent_seq_len,
            empty_string_feat=empty_string_feat,
            gradient_checkpointing=job_config.training.gradient_checkpointing,
        )
        self.fm = FlowMatching(
            min_sigma=job_config.sampling.min_sigma,
            inference_mode=job_config.sampling.method,
            num_steps=job_config.sampling.num_steps,
            prediction_type=job_config.sampling.prediction_type,
            noise_scale=job_config.sampling.noise_scale,
            noise_shift=job_config.sampling.noise_shift,
        )

        self.loss_type = job_config.training.loss_type

        self.clip_grad_norm = job_config.training.clip_grad_norm
        self.log_normal_sampling_scale = job_config.sampling.scale
        self.log_normal_sampling_mean = job_config.sampling.mean
        self.null_condition_probability = job_config.training.label_drop_prob

        # EMA decay rates
        self.ema_decay1 = job_config.training.ema_decay1
        self.ema_decay2 = job_config.training.ema_decay2

        # Audio and generation config
        self.audio_scale = job_config.data.audio_scale
        self.sample_rate = job_config.data.target_sample_rate
        self.audio_duration = job_config.data.audio_duration
        self.cfg_strength = job_config.eval.cfg
        self.experiment_id = job_config.experiment_id
        self.output_dir = job_config.training.output_dir
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(job_config.training.seed + self.local_rank)

        if is_main_process():
            logger.info("Model loaded successfully")
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of trainable parameters: {n_params / 1e6:.6f}M")

        # Clear cache again before moving model to GPU
        torch.cuda.empty_cache()

        model.to(self.device)

        self.model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], broadcast_buffers=False
        )
        self.model_without_ddp = self.model.module

        if job_config.training.gradient_checkpointing:
            if is_main_process():
                logger.info(
                    "Gradient checkpointing enabled for activation memory savings"
                )

        if job_config.training.compile:
            if is_main_process():
                logger.info("Compiling model with torch.compile...")
            # pyrefly: ignore [bad-assignment]
            self.model = torch.compile(self.model)

    def _setup_optimizer(self, job_config, world_size):
        """Set up optimizer with learning rate and weight decay."""
        batch_size = job_config.training.batch_size
        eff_batch_size = batch_size * world_size
        training = job_config.training

        if training.lr is None:
            # Use OmegaConf.update to set lr since config may be frozen
            lr = training.blr * eff_batch_size / 256
            # pyrefly: ignore [bad-argument-type]
            OmegaConf.update(self.config, "training.lr", lr, merge=True)
        else:
            lr = training.lr

        if is_main_process():
            logger.info(f"Base lr: {lr * 256 / eff_batch_size:.2e}")
            logger.info(f"Actual lr: {lr:.2e}")
            logger.info(f"Effective batch size: {eff_batch_size}")

        # Set up optimizer with weight decay adjustment for bias and norm layers
        param_groups = add_weight_decay(self.model_without_ddp, training.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        logger.info(f"Optimizer: {self.optimizer}")

    def _resolve_checkpoint_path(self, job_config):
        """Resolve checkpoint path and load mode.

        Priority:
        1. output_dir/experiment_id/checkpoint_latest.pth (resume)
        2. initial_load_path from config (pretrained)
        3. None (train from scratch)
        """
        latest_path = os.path.join(
            self.output_dir, self.experiment_id, "checkpoint_latest.pth"
        )
        if os.path.exists(latest_path):
            if is_main_process():
                logger.info(
                    f"Found latest checkpoint: {latest_path}, resuming training"
                )
            return latest_path, False

        initial_path = job_config.checkpoint.initial_load_path
        if not initial_path:
            return None, False

        resolved = initial_path
        if not os.path.exists(resolved) and "/tmp/ckptcache" in str(resolved):
            fallback = os.path.join(
                self.output_dir,
                self.experiment_id,
                os.path.basename(str(initial_path)),
            )
            if is_main_process():
                logger.info(f"Checkpoint not found at {resolved}, trying {fallback}")
            if os.path.exists(fallback):
                resolved = fallback

        if is_main_process():
            logger.info(f"initial_load_path: {initial_path}")
            logger.info(f"Resolved checkpoint_path: {resolved}")
            logger.info(f"File exists: {os.path.exists(resolved)}")
            logger.info(
                f"Load model only: {job_config.checkpoint.initial_load_model_only}"
            )

        checkpoint_path = resolved if os.path.exists(resolved) else None
        return checkpoint_path, job_config.checkpoint.initial_load_model_only

    def _load_checkpoint(self, job_config):
        """Load checkpoint if available, or initialize EMA params."""
        self.global_step = 0
        checkpoint_path, load_model_only = self._resolve_checkpoint_path(job_config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "model" in checkpoint:
                self.model_without_ddp.load_state_dict(checkpoint["model"])

                ema_state_dict1 = checkpoint["model_ema1"]
                ema_state_dict2 = checkpoint["model_ema2"]
                self.model_without_ddp.ema_params1 = [
                    ema_state_dict1[name].to(self.device)
                    for name, _ in self.model_without_ddp.named_parameters()
                ]
                self.model_without_ddp.ema_params2 = [
                    ema_state_dict2[name].to(self.device)
                    for name, _ in self.model_without_ddp.named_parameters()
                ]
                if is_main_process():
                    logger.info(f"Loaded full checkpoint from {checkpoint_path}")

                if (
                    not load_model_only
                    and "optimizer" in checkpoint
                    and "epoch" in checkpoint
                ):
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    OmegaConf.update(
                        # pyrefly: ignore [bad-argument-type]
                        self.config,
                        "training.start_epoch",
                        checkpoint["epoch"],
                        merge=True,
                    )
                    if "global_step" in checkpoint:
                        self.global_step = checkpoint["global_step"]
                    if is_main_process():
                        logger.info(
                            f"Loaded optimizer state, resuming from epoch {checkpoint['epoch'] + 1}"
                        )
                        if "global_step" in checkpoint:
                            logger.info(f"Restored global_step: {self.global_step}")
                elif is_main_process():
                    logger.info("Model only load, starting training from epoch 0")
            else:
                self.model_without_ddp.load_state_dict(checkpoint)
                self.model_without_ddp.ema_params1 = copy.deepcopy(
                    list(self.model_without_ddp.parameters())
                )
                self.model_without_ddp.ema_params2 = copy.deepcopy(
                    list(self.model_without_ddp.parameters())
                )
                if is_main_process():
                    logger.info(
                        f"Loaded EMA-only checkpoint from {checkpoint_path}, "
                        "initialized ema_params from loaded weights"
                    )
            del checkpoint
        else:
            self.model_without_ddp.ema_params1 = copy.deepcopy(
                list(self.model_without_ddp.parameters())
            )
            self.model_without_ddp.ema_params2 = copy.deepcopy(
                list(self.model_without_ddp.parameters())
            )
            if is_main_process():
                logger.info("Training from scratch")

    def _setup_logging(self, world_size):
        """Set up logging and distributed flags."""
        self.log_writer = None  # TODO: Set up tensorboard writer if needed
        self.distributed = world_size > 1

        # Set up logging to local file
        if is_main_process():
            # Create experiment output directory
            exp_output_dir = os.path.join(self.output_dir, self.experiment_id)
            os.makedirs(exp_output_dir, exist_ok=True)

            # Configure file handler to save training logs locally
            log_file = os.path.join(exp_output_dir, "training.log")
            file_handler = logging.FileHandler(
                log_file, mode="a"
            )  # Append mode for resume training
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

            # Add handler to root logger to capture all module logs (e.g., audio_dataset warnings)
            # Note: Only add to root logger, not module logger, to avoid duplicate logs
            # (module logger propagates to root logger by default)
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.INFO)
            logger.info(f"Log file saved to: {log_file}")

            # Log ffmpeg and torchaudio backend availability for mp4 audio loading
            log_audio_backend_info()

    @record
    def __init__(self, job_config: wavflowConfig) -> None:
        self.config = job_config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not torch.cuda.is_available():
            raise RuntimeError("Local-only training requires CUDA GPUs.")

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(
                    seconds=job_config.comm.init_timeout_seconds
                ),
            )

        self._setup_multiaudio_dataset(job_config)
        self._setup_model(job_config)
        self._setup_optimizer(job_config, world_size)
        self._load_checkpoint(job_config)
        self._setup_logging(world_size)

    def train_one_epoch(
        self,
        model,
        model_without_ddp,
        data_loader,
        optimizer,
        device,
        epoch,
        config,
        log_writer=None,
    ):
        """Train for one epoch."""
        model.train(True)

        optimizer.zero_grad()

        if log_writer is not None and is_main_process():
            logger.info(f"log_dir: {log_writer.log_dir}")

        # MultiDatasetWeightedBatchSampler may not have __len__
        try:
            num_batches = len(data_loader)
        except TypeError:
            num_batches = None

        log_freq = config.training.log_freq

        epoch_start_time = time.time()
        log_start_time = time.time()  # Time of last log
        steps_since_last_log = 0  # Steps processed since last log

        for data_iter_step, batch_data in enumerate(data_loader):
            # if batch_data is None, continue
            if not batch_data or "audio_tokens" not in batch_data:
                logger.info(f"Skipping empty batch at step {data_iter_step}")
                continue

            # Cache batch for generate_training_data() to avoid creating
            # a new DataLoader iterator (which resets the sampler)
            self._cached_gen_batch = batch_data

            # Compute loss for this batch
            batch_size, mean_loss, current_lr, grad_norm = self._train_step(
                batch_data,
                optimizer,
                device,
                epoch,
                data_iter_step,
                num_batches,
            )

            # Increment cumulative global step
            self.global_step += 1
            steps_since_last_log += 1

            # Log metrics at the configured frequency
            self._handle_step_logging_and_checkpoints(
                data_iter_step=data_iter_step,
                num_batches=num_batches,
                log_freq=log_freq,
                batch_size=batch_size,
                mean_loss=mean_loss,
                current_lr=current_lr,
                grad_norm=grad_norm,
                epoch=epoch,
                config=config,
                log_start_time=log_start_time,
                steps_since_last_log=steps_since_last_log,
            )

            # Reset timer and counter after logging
            should_log = self.global_step % log_freq == 0
            if num_batches is not None:
                should_log = should_log or data_iter_step == num_batches - 1
            if should_log:
                log_start_time = time.time()
                steps_since_last_log = 0

        epoch_time = time.time() - epoch_start_time
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

    def _train_step(
        self,
        batch_data,
        optimizer,
        device,
        epoch,
        data_iter_step,
        num_batches,
    ):
        """Execute a single training step and return metrics."""
        # Use cumulative global step (maintained across epochs)
        global_step = self.global_step

        # per iteration (instead of per epoch) lr scheduler with step-based warmup
        if num_batches is not None:
            lr_epoch = data_iter_step / num_batches + epoch
        else:
            lr_epoch = epoch
        current_lr = adjust_learning_rate(
            optimizer, lr_epoch, self.config, global_step=global_step
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            clip_f, sync_f, text_f, audio_tokens = self._prepare_batch_tensors(
                batch_data, device
            )

            # Smoke-test trick: when training on our 4-sample demo dataset, uncomment
            # the block below to repeat each batch 32x along the batch dim. The model
            # will overfit and converge in a few hundred steps, letting you verify the
            # full pipeline (data loading, forward, backward, audio generation)
            # end-to-end without waiting for a real training run.
            # repeat_factor = 32
            # clip_f = clip_f.repeat(repeat_factor, 1, 1)
            # sync_f = sync_f.repeat(repeat_factor, 1, 1)
            # text_f = text_f.repeat(repeat_factor, 1, 1)
            # audio_tokens = audio_tokens.repeat(repeat_factor, 1, 1)

            bs = audio_tokens.shape[0]

            x1 = audio_tokens
            t = log_normal_sample(
                x1,
                generator=self.rng,
                m=self.log_normal_sampling_mean,
                s=self.log_normal_sampling_scale,
            )
            x0, x1, xt, t_shifted, (clip_f, sync_f, text_f) = self.fm.get_x0_xt_c(
                x1, t, Cs=[clip_f, sync_f, text_f], generator=self.rng
            )
            # classifier-free training: use torch.where for compile compatibility
            clip_f, sync_f, text_f = self._apply_cfg_masking(
                bs, x1.device, clip_f, sync_f, text_f
            )

            pred_x1 = self.model(xt, clip_f, sync_f, text_f, t_shifted)
            if self.fm.prediction_type == "x" and self.loss_type == "x":
                loss = self.fm.x_pred_x_loss(pred_x1, x0, xt, x1, t_shifted)
            elif self.fm.prediction_type == "x" and self.loss_type == "v":
                loss = self.fm.x_pred_v_loss(pred_x1, x0, xt, x1, t_shifted)
            elif self.fm.prediction_type == "v" and self.loss_type == "v":
                loss = self.fm.v_pred_v_loss(pred_x1, x0, xt, x1, t_shifted)
            elif self.fm.prediction_type == "v" and self.loss_type == "x":
                loss = self.fm.v_pred_x_loss(pred_x1, x0, xt, x1, t_shifted)
            # pyrefly: ignore [unbound-name]
            mean_loss = loss.mean()

        optimizer.zero_grad(set_to_none=True)
        mean_loss.backward()
        # clip grad norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model_without_ddp.parameters(), self.clip_grad_norm
        )
        optimizer.step()
        self.update_ema()

        return bs, mean_loss, current_lr, grad_norm

    def _prepare_batch_tensors(self, batch_data, device):
        """Prepare batch tensors and handle missing video/text conditions."""
        clip_f = batch_data["clip_features"].to(
            device, non_blocking=True
        )  # [B, 64, 1024]
        sync_f = batch_data["sync_features"].to(
            device, non_blocking=True
        )  # [B, 192, 768]
        text_f = batch_data["text_features"].to(
            device, non_blocking=True
        )  # [B, 77, 1024]
        audio_tokens = batch_data["audio_tokens"].to(
            device, non_blocking=True
        )  # [B, 250, 512]

        # Handle missing video/text conditions (before CFG random masking)
        if "video_exist" in batch_data:
            video_exist = batch_data["video_exist"].to(device, non_blocking=True)
            clip_f[~video_exist] = self.model_without_ddp.empty_clip_feat
            sync_f[~video_exist] = self.model_without_ddp.empty_sync_feat
        if "text_exist" in batch_data:
            text_exist = batch_data["text_exist"].to(device, non_blocking=True)
            text_f[~text_exist] = self.model_without_ddp.empty_string_feat

        return clip_f, sync_f, text_f, audio_tokens

    def _apply_cfg_masking(self, bs, device, clip_f, sync_f, text_f):
        """Apply classifier-free guidance masking to conditions."""
        samples = torch.rand(bs, device=device, generator=self.rng)
        null_video = (samples < self.null_condition_probability)[
            :, None, None
        ]  # [B, 1, 1]
        clip_f = torch.where(null_video, self.model.module.empty_clip_feat, clip_f)
        sync_f = torch.where(null_video, self.model.module.empty_sync_feat, sync_f)

        samples = torch.rand(bs, device=device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)[
            :, None, None
        ]  # [B, 1, 1]
        text_f = torch.where(null_text, self.model.module.empty_string_feat, text_f)

        return clip_f, sync_f, text_f

    def _handle_step_logging_and_checkpoints(
        self,
        data_iter_step: int,
        num_batches: Optional[int],
        log_freq: int,
        batch_size: int,
        mean_loss: torch.Tensor,
        current_lr: float,
        grad_norm: torch.Tensor,
        epoch: int,
        config: wavflowConfig,
        log_start_time: float,
        steps_since_last_log: int,
    ) -> None:
        """Handle logging at the configured frequency. Checkpointing and
        sample generation are handled at epoch granularity in `train()`."""
        # Log metrics at specified frequency (based on cumulative global_step
        # so log_freq works correctly even when each epoch has only 1 step).
        should_log = self.global_step % log_freq == 0
        if num_batches is not None:
            should_log = should_log or data_iter_step == num_batches - 1
        if should_log:
            elapsed_time = time.time() - log_start_time
            # Throughput = total samples processed / elapsed time
            total_samples = batch_size * steps_since_last_log
            samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
            if is_main_process():
                step_str = (
                    f"{data_iter_step}/{num_batches}"
                    if num_batches
                    else f"{data_iter_step}"
                )
                logger.info(
                    f"Epoch [{epoch + 1}/{config.training.epochs}] Step [{step_str}] GlobalStep {self.global_step} Loss: {mean_loss.item():.4f} LR: {current_lr:.2e} GradNorm: {grad_norm:.4f} Throughput: {samples_per_sec:.1f} samples/s"
                )

    @torch.no_grad()
    def update_ema(self):
        """Update EMA parameters with exponential moving average.

        Fused loop iterates source params once, updating both EMA copies.
        Uses lerp_ which is more cache-friendly than mul_ + add_.
        """
        for targ1, targ2, src in zip(
            self.model_without_ddp.ema_params1,
            self.model_without_ddp.ema_params2,
            self.model_without_ddp.parameters(),
        ):
            targ1.lerp_(src, 1 - self.ema_decay1)
            targ2.lerp_(src, 1 - self.ema_decay2)

    def train(self):
        """Main training loop."""
        training = self.config.training
        checkpoint_save_epoch = training.checkpoint_save_epoch
        checkpoint_freq = training.checkpoint_freq
        gen_train_freq = training.gen_train_freq
        ema_save_epoch = training.ema_checkpoint_save_epoch

        # Create output directory with experiment_id
        exp_output_dir = os.path.join(self.output_dir, self.experiment_id)

        if is_main_process():
            os.makedirs(exp_output_dir, exist_ok=True)
            logger.info(f"Starting training for {training.epochs} epochs")
            logger.info(f"Output directory: {exp_output_dir}")

        start_time = time.time()

        for epoch in range(training.start_epoch, training.epochs):
            self.train_sampler.set_epoch(epoch)
            if is_main_process():
                logger.info(f"Starting epoch {epoch + 1}")
            self.train_one_epoch(
                self.model,
                self.model_without_ddp,
                self.dataloader_train,
                self.optimizer,
                self.device,
                epoch,
                self.config,
                log_writer=self.log_writer,
            )

            display_epoch = epoch + 1
            is_last_epoch = display_epoch == training.epochs

            # Refresh checkpoint_latest.pth every checkpoint_freq epochs
            if checkpoint_freq > 0 and (
                display_epoch % checkpoint_freq == 0 or is_last_epoch
            ):
                self.save_checkpoint(
                    display_epoch,
                    exp_output_dir,
                    filename="checkpoint_latest.pth",
                )

            # Generate training samples every gen_train_freq epochs
            if gen_train_freq > 0 and (
                display_epoch % gen_train_freq == 0 or is_last_epoch
            ):
                self.generate_training_data(display_epoch, exp_output_dir)

            # Save permanent checkpoint every checkpoint_save_epoch epochs
            if checkpoint_save_epoch > 0 and display_epoch % checkpoint_save_epoch == 0:
                self.save_checkpoint(
                    display_epoch,
                    exp_output_dir,
                    filename=f"checkpoint_epoch_{display_epoch}.pth",
                )

            # Save standalone EMA checkpoint at configured frequency
            if ema_save_epoch > 0 and display_epoch % ema_save_epoch == 0:
                self.save_ema_checkpoint(display_epoch, exp_output_dir)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training completed. Total time: {total_time_str}")

    def save_ema_checkpoint(self, epoch: int, output_dir: Optional[str] = None) -> None:
        """Save EMA weights as a standalone model checkpoint for inference."""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, self.experiment_id)
            os.makedirs(output_dir, exist_ok=True)

        if not is_main_process():
            return

        param_names = [n for n, _ in self.model_without_ddp.named_parameters()]
        ema_state_dict = {
            name: p.cpu()
            for name, p in zip(param_names, self.model_without_ddp.ema_params1)
        }
        path = os.path.join(output_dir, f"ema_epoch_{epoch}.pth")
        torch.save(ema_state_dict, path)
        logger.info(f"Saved EMA checkpoint to {path}")

    def save_checkpoint(self, epoch, output_dir=None, filename="checkpoint_latest.pth"):
        """Save checkpoint with specified filename."""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, self.experiment_id)
            os.makedirs(output_dir, exist_ok=True)

        # Only rank 0 saves checkpoint to avoid conflicts
        if not is_main_process():
            return

        checkpoint = {
            "model": self.model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "model_ema1": {
                name: p.cpu()
                for name, p in zip(
                    [n for n, _ in self.model_without_ddp.named_parameters()],
                    self.model_without_ddp.ema_params1,
                )
            },
            "model_ema2": {
                name: p.cpu()
                for name, p in zip(
                    [n for n, _ in self.model_without_ddp.named_parameters()],
                    self.model_without_ddp.ema_params2,
                )
            },
        }
        checkpoint_path = os.path.join(output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Checkpoint saved to {checkpoint_path} (epoch {epoch})"
        )  # Already guarded by is_main_process() check above

    def generate_training_data(self, epoch, output_dir, step=None):
        """
        Generate audio samples for visualization.
        Only the first 8 global ranks (node 0) generate and save audio to avoid file conflicts.

        Args:
            epoch: Current epoch number
            output_dir: Base output directory
            step: Optional step number for step-based generation
        """
        # Only first 8 global ranks (node 0) generate audio
        global_rank = int(os.environ.get("RANK", 0))
        if global_rank >= 8:
            return

        # Create samples directory with optional step suffix
        if step is not None:
            samples_dir = os.path.join(
                output_dir, "samples", f"epoch_{epoch}_step_{step}"
            )
        else:
            samples_dir = os.path.join(output_dir, "samples", f"epoch_{epoch}")
        os.makedirs(samples_dir, exist_ok=True)

        # Reuse the last training batch (cached during training) instead of
        # creating a new iterator with next(iter(dataloader)), which resets
        # the sampler and wastes prefetched data.
        batch_data = getattr(self, "_cached_gen_batch", None)
        if batch_data is None:
            return  # No training batch available yet

        # Get sample ID for debugging (GT audio ID)
        sample_ids = batch_data.get("id", [])
        gt_sample_id = sample_ids[0] if len(sample_ids) > 0 else "unknown"
        # Sanitize ID for filename (replace special chars)
        gt_sample_id_safe = str(gt_sample_id).replace("/", "_").replace("\\", "_")

        # Move data to device and take the first sample
        clip_f = batch_data["clip_features"][0:1].to(self.device)
        sync_f = batch_data["sync_features"][0:1].to(self.device)
        text_f = batch_data["text_features"][0:1].to(self.device)
        audio_tokens = batch_data["audio_tokens"][0:1].to(self.device)

        # Enter eval mode
        self.model.eval()

        try:
            with (
                torch.amp.autocast("cuda", dtype=torch.bfloat16),
                torch.inference_mode(),
            ):
                # Save GT audio
                audio_gt = (
                    audio_tokens[0].cpu().float() / self.audio_scale
                )  # de-normalize
                num_samples = self.target_samples
                audio_gt = audio_gt.reshape(-1)[:num_samples]

                gt_path = os.path.join(
                    samples_dir, f"gt_rank{global_rank}_id_{gt_sample_id_safe}.wav"
                )
                torchaudio.save(gt_path, audio_gt.unsqueeze(0), self.sample_rate)

                # Generate audio from noise
                x0 = (
                    torch.randn(
                        audio_tokens.shape,
                        generator=self.rng,
                        device=audio_tokens.device,
                        dtype=audio_tokens.dtype,
                    )
                    * self.fm.noise_scale
                )

                # Preprocess conditions
                conditions = self.model_without_ddp.preprocess_conditions(
                    clip_f, sync_f, text_f
                )
                empty_conditions = self.model_without_ddp.get_empty_conditions(
                    x0.shape[0]
                )

                # ODE wrapper with CFG
                def cfg_ode_wrapper(t, x):
                    return self.model_without_ddp.ode_wrapper(
                        t, x, conditions, empty_conditions, self.cfg_strength
                    )

                # Generate
                x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)

                # Save generated audio (uses same sample ID since conditions come from same sample)
                audio_gen = x1_hat[0].cpu().float() / self.audio_scale
                audio_gen = audio_gen.reshape(-1)[:num_samples]

                gen_path = os.path.join(
                    samples_dir, f"gen_rank{global_rank}_id_{gt_sample_id_safe}.wav"
                )
                torchaudio.save(gen_path, audio_gen.unsqueeze(0), self.sample_rate)

        except Exception as e:
            logger.warning(f"Error in generate_training_data: {e}")
        finally:
            # Return to train mode
            self.model.train()

    def close(self) -> None:
        """Cleanup resources."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
