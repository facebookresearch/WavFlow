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

import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import torchaudio
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "medium_16k"
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    class_num: int = 1000
    empty_string_ckpt: str = "path/to/empty_string_ckpt.pth"


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 200
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0
    batch_size: int = 128
    global_batch_size: int = -1
    seed: int = 0
    start_epoch: int = 0
    num_workers: int = 12
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    compile: bool = False
    gradient_checkpointing: bool = False
    output_dir: str = "output"

    # Optimizer settings
    lr: Optional[float] = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "step"
    lr_schedule_steps: list = field(default_factory=lambda: [60, 90])
    lr_schedule_gamma: float = 0.1
    weight_decay: float = 0.0
    log_freq: int = 50
    gen_train_freq: int = 20
    checkpoint_freq: int = 20
    checkpoint_save_epoch: int = 1
    warmup_steps: int = 60  # Number of steps for linear warmup

    # EMA settings
    ema_decay1: float = 0.9999
    ema_decay2: float = 0.9996
    ema_checkpoint_save_epoch: int = 50

    # Loss type (prediction_type lives in SamplingConfig since it's shared with inference)
    loss_type: str = "v"  # "x" (x-loss) or "v" (v-loss)

    # CFG training
    label_drop_prob: float = 0.1


@dataclass
class SamplingConfig:
    """Sampling configuration for flow matching."""

    mean: float = 0.0
    scale: float = 1.0
    min_sigma: float = 0.0
    method: str = "euler"
    num_steps: int = 50
    noise_scale: float = 1.0
    noise_shift: float = 1.0
    prediction_type: str = "x"  # shared between train (loss target) and infer (ODE)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    sampling_method: str = "euler"
    num_sampling_steps: int = 50
    cfg: float = 1.0
    interval_min: float = 0.0
    interval_max: float = 1.0
    eval_freq: int = 40


@dataclass
class DataConfig:
    """Dataset configuration."""

    csv_path: str = ""
    data_root: Optional[str] = None
    audio_scale: float = 3.0
    target_sample_rate: int = 16000
    audio_duration: float = 8.0
    audio_dim: int = 200
    clip_seq_len: int = 64
    clip_dim: int = 1024
    sync_seq_len: int = 192
    sync_dim: int = 768
    text_seq_len: int = 77
    text_dim: int = 1024


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    enable_checkpoint: bool = True
    folder: str = "checkpoint"
    initial_load_path: Optional[str] = None
    initial_load_model_only: bool = True
    interval: int = 5
    export_dtype: str = "float32"
    keep_latest_k: int = -1
    load_step: int = -1


@dataclass
class MetricsConfig:
    """Logging/metrics configuration."""

    disable_color_printing: bool = False
    compact_logging: bool = True


@dataclass
class ParallelismConfig:
    """Distributed training configuration."""

    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = -1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    context_parallel_degree: int = 1
    expert_parallel_degree: int = 1


@dataclass
class CommConfig:
    """Communication/timeout configuration."""

    init_timeout_seconds: int = 1800
    train_timeout_seconds: int = 1800


@dataclass
class JobConfig:
    """Job-level configuration."""

    description: str = "wavflow Audio Generation Training"
    dump_folder: str = "./output_dir"


@dataclass
class ExperimentalConfig:
    """Experimental features configuration."""

    custom_import: str = ""


@dataclass
class wavflowConfig:
    """Main configuration for wavflow audio generation training."""

    experiment_id: str = "default_exp"
    job: JobConfig = field(default_factory=JobConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)


# ============================================================================
# Config loading
# ============================================================================


def _merge_base(cfg: "DictConfig", config_path: str) -> "DictConfig":
    """If cfg has a top-level ``_base_`` key, load that file (resolved
    relative to ``config_path``) and merge it under cfg so cfg's keys win.
    Recursive merging is supported (a base may itself reference another base).
    """
    if "_base_" not in cfg:
        return cfg
    base_rel = str(cfg._base_)
    base_path = os.path.join(os.path.dirname(config_path), base_rel)
    base = OmegaConf.load(base_path)
    base = _merge_base(base, base_path)
    merged = OmegaConf.merge(base, cfg)
    del merged["_base_"]
    return merged


def load_config(
    config_path: Optional[str] = None, overrides: Optional[dict] = None
) -> wavflowConfig:
    """Load configuration from YAML file and apply overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        overrides: Dictionary of overrides to apply on top of loaded config.

    Returns:
        wavflowConfig object with all settings.
    """
    # Start with default config
    schema = OmegaConf.structured(wavflowConfig)

    if config_path is not None and os.path.exists(config_path):
        # Load from YAML file (with optional ``_base_`` include).
        file_config = OmegaConf.load(config_path)
        file_config = _merge_base(file_config, config_path)
        config = OmegaConf.merge(schema, file_config)
    else:
        config = schema

    # Apply overrides
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    # pyrefly: ignore [bad-return]
    return config


# ============================================================================
# Optimizer / LR helpers
# ============================================================================


def adjust_learning_rate(optimizer, epoch, config, global_step=None):
    """Decay the learning rate with linear warmup based on steps.

    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch (used for lr_schedule after warmup)
        config: Training configuration
        global_step: Current global step count (used for warmup)
    """
    training = config.training
    warmup_steps = training.warmup_steps

    # Step-based linear warmup
    if global_step is not None and global_step < warmup_steps:
        # Linear warmup from 0 to target lr over warmup_steps
        lr = training.lr * (global_step + 1) / warmup_steps
    else:
        # After warmup, apply lr_schedule
        if training.lr_schedule == "constant":
            lr = training.lr
        elif training.lr_schedule == "cosine":
            lr = training.min_lr + (training.lr - training.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * epoch / training.epochs)
            )
        elif training.lr_schedule == "step":
            lr = training.lr
            for step in training.lr_schedule_steps:
                if epoch >= step:
                    lr *= training.lr_schedule_gamma
        else:
            raise NotImplementedError(f"Unknown lr_schedule: {training.lr_schedule}")

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def add_weight_decay(model, weight_decay=0, skip_list=()):
    """Build optimizer param groups, excluding bias / norm / diffloss from weight decay."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or "diffloss" in name
        ):
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


# ============================================================================
# Distributed helpers
# ============================================================================


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


# ============================================================================
# Audio backend diagnostics
# ============================================================================


def log_audio_backend_info():
    """Log ffmpeg and torchaudio backend availability for debugging audio loading issues."""
    logger.info("=" * 50)
    logger.info("Audio Backend Information")
    logger.info("=" * 50)

    # Check ffmpeg availability
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Extract first line of ffmpeg version
            version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            logger.info(f"ffmpeg: AVAILABLE - {version_line}")
        else:
            logger.warning("ffmpeg: NOT AVAILABLE (command failed)")
    except FileNotFoundError:
        logger.warning("ffmpeg: NOT FOUND in PATH - mp4 audio loading will fail!")
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg: TIMEOUT checking version")
    except Exception as e:
        logger.warning(f"ffmpeg: ERROR checking availability - {e}")

    # Check torchaudio backend
    try:
        # List available backends
        available_backends = torchaudio.list_audio_backends()
        logger.info(f"torchaudio available backends: {available_backends}")

        # Check if ffmpeg backend is available
        if "ffmpeg" in available_backends:
            logger.info("torchaudio ffmpeg backend: AVAILABLE")
        else:
            logger.warning("torchaudio ffmpeg backend: NOT AVAILABLE")

        # Log current backend (if applicable)
        try:
            current_backend = torchaudio.get_audio_backend()
            logger.info(f"torchaudio current backend: {current_backend}")
        except Exception:
            logger.info("torchaudio: using new dispatcher (no global backend)")

    except Exception as e:
        logger.warning(f"torchaudio backend check failed: {e}")

    # Log supported audio formats
    logger.info("Supported audio formats: .flac, .wav, .mp4 (mp4 requires ffmpeg)")
    logger.info("=" * 50)
