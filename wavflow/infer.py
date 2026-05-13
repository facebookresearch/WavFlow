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

"""
Wavflow end-to-end inference (single GPU).

Given a CSV with columns ``video_path, caption, video_exist, text_exist`` (and
optionally ``id``), this script:
  1. extracts CLIP frame, Synchformer, and CLIP text features on the fly
     (or substitutes empty learned tokens when video_exist=0 / text_exist=0),
  2. runs the WavFlow flow-matching ODE with classifier-free guidance,
  3. saves each generated audio clip as ``<output_dir>/<id>.wav``.

Usage:
    python -m wavflow.infer --config wavflow/configs/infer.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import torch
import torchaudio
from feature_extract.ext.features_utils import FeaturesUtils
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from tqdm import tqdm
from wavflow.dataset.multiaudio_dataset import derive_audio_shapes
from wavflow.model.flow_matching import FlowMatching
from wavflow.model.networks import get_wavflow_model
from wavflow.trainer_utils import _merge_base

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_CLIP_SIZE = 384
_CLIP_FPS = 8.0
_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wavflow inference")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> DictConfig:
    """Load infer.yaml with optional ``_base_:`` include (no schema validation)."""
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config at {path} must be a mapping")
    cfg = _merge_base(cfg, path)
    return cfg


def _build_clip_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(
                (_CLIP_SIZE, _CLIP_SIZE),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def _build_sync_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _decode_video(
    video_path: str, duration_sec: float
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(_CLIP_FPS * duration_sec),
        frame_rate=_CLIP_FPS,
        format="rgb24",
    )
    reader.add_basic_video_stream(
        frames_per_chunk=int(_SYNC_FPS * duration_sec),
        frame_rate=_SYNC_FPS,
        format="rgb24",
    )
    reader.fill_buffer()
    chunks = reader.pop_chunks()
    return chunks[0], chunks[1]


def _row_id(row: dict, idx: int) -> str:
    if "id" in row and not pd.isna(row["id"]):
        return str(row["id"])
    vp = row.get("video_path", "")
    if isinstance(vp, str) and vp:
        return Path(vp).stem
    return f"row_{idx:05d}"


def _load_ckpt(model: torch.nn.Module, ckpt_path: str, use_ema: bool) -> None:
    """Load a wavflow checkpoint. Handles three formats:
    - full training ckpt: dict with 'model' / 'model_ema1' / 'optimizer' / ...
    - ema-only ckpt: flat state_dict {param_name: tensor}
    - plain weights ckpt: flat state_dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_ema1" in ckpt and use_ema:
        state = ckpt["model_ema1"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded EMA params from {ckpt_path}. "
            f"Missing (expected: rotary buffers): {missing}. Unexpected: {unexpected}"
        )
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        for buf_key in ("t_embed.freqs", "latent_rot", "clip_rot"):
            state.pop(buf_key, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded full model from {ckpt_path}. "
            f"Missing: {missing}. Unexpected: {unexpected}"
        )
    else:
        # flat state_dict (e.g., ema_epoch_*.pth saved by save_ema_checkpoint)
        state = ckpt
        for buf_key in ("t_embed.freqs", "latent_rot", "clip_rot"):
            state.pop(buf_key, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded flat state_dict from {ckpt_path}. "
            f"Missing: {missing}. Unexpected: {unexpected}"
        )


class WavflowInferencer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        if not torch.cuda.is_available():
            raise RuntimeError("Wavflow inference requires a CUDA GPU.")
        self.device = torch.device("cuda:0")

        # Audio shape (shared with training via base.yaml)
        self.duration_sec = float(config.data.audio_duration)
        self.sample_rate = int(config.data.target_sample_rate)
        self.audio_dim = int(config.data.audio_dim)
        self.audio_scale = float(config.data.audio_scale)
        self.latent_seq_len, _, self.target_samples = derive_audio_shapes(
            self.sample_rate, self.duration_sec, self.audio_dim
        )

        # Inference-only knobs
        self.cfg_strength = float(config.inference.cfg)
        self.batch_size = int(config.inference.batch_size)
        self.trim_to_duration = bool(config.inference.get("trim_to_duration", True))

        self.clip_transform = _build_clip_transform()
        self.sync_transform = _build_sync_transform()
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self._setup_features_extractor(config)
        self._setup_wavflow_model(config)
        self._setup_flow_matching(config)

        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loudness_norm = bool(config.output.get("loudness_norm", True))
        self.target_lufs = float(config.output.get("loudness_target_lufs", -22.0))

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(config.inference.seed))

    def _setup_features_extractor(self, config: DictConfig) -> None:
        self.features_utils = (
            FeaturesUtils(
                synchformer_ckpt=config.model.synchformer_ckpt,
                clip_model_name=config.model.get(
                    "clip_model_name", "ViT-H-14-378-quickgelu"
                ),
                clip_pretrained=config.model.get("clip_pretrained", None),
                enable_conditions=True,
            )
            .eval()
            .to(self.device)
        )
        logger.info("FeaturesUtils loaded (CLIP image + Synchformer + CLIP text)")

    def _setup_wavflow_model(self, config: DictConfig) -> None:
        # For inference: empty_string_feat is restored from the trained checkpoint
        # (it's a registered Parameter), so we just init with zeros and let
        # _load_ckpt overwrite it. No need to download/compute it here.
        self.model = get_wavflow_model(
            config.model.name,
            latent_seq_len=self.latent_seq_len,
            empty_string_feat=None,  # init with zeros, ckpt will overwrite
        )
        _load_ckpt(self.model, config.model.ckpt_path, bool(config.model.use_ema))
        self.model = self.model.eval().to(self.device)

        self.latent_dim = self.model.latent_dim
        logger.info(
            f"WavflowModel '{config.model.name}' loaded. "
            f"latent_seq_len={self.latent_seq_len}, latent_dim={self.latent_dim}"
        )

    def _setup_flow_matching(self, config: DictConfig) -> None:
        self.fm = FlowMatching(
            min_sigma=float(config.sampling.get("min_sigma", 0.0)),
            inference_mode=str(config.sampling.get("method", "euler")),
            num_steps=int(config.sampling.num_steps),
            prediction_type=str(config.sampling.prediction_type),
            noise_scale=float(config.sampling.noise_scale),
            noise_shift=float(config.sampling.noise_shift),
        )

    @torch.inference_mode()
    def _encode_video(self, video_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        clip_chunk, sync_chunk = _decode_video(video_path, self.duration_sec)
        if clip_chunk is None or clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f"Video {video_path} too short for CLIP "
                f"({None if clip_chunk is None else clip_chunk.shape[0]} < {self.clip_expected_length})"
            )
        if sync_chunk is None or sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f"Video {video_path} too short for sync "
                f"({None if sync_chunk is None else sync_chunk.shape[0]} < {self.sync_expected_length})"
            )
        clip_chunk = self.clip_transform(clip_chunk[: self.clip_expected_length])
        sync_chunk = self.sync_transform(sync_chunk[: self.sync_expected_length])

        clip_video = clip_chunk.unsqueeze(0).to(self.device)  # (1, T, 3, 384, 384)
        sync_video = sync_chunk.unsqueeze(0).to(self.device)  # (1, T, 3, 224, 224)
        clip_f = self.features_utils.encode_video_with_clip(clip_video)  # (1, 64, 1024)
        sync_f = self.features_utils.encode_video_with_sync(sync_video)  # (1, 192, 768)
        return clip_f.squeeze(0), sync_f.squeeze(0)

    @torch.inference_mode()
    def _encode_text(self, caption: str) -> torch.Tensor:
        text_f = self.features_utils.encode_text([caption])  # (1, 77, 1024)
        return text_f.squeeze(0)

    def _features_for_row(
        self, row: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_exist = bool(int(row.get("video_exist", 0)))
        text_exist = bool(int(row.get("text_exist", 0)))

        if video_exist:
            clip_f, sync_f = self._encode_video(str(row["video_path"]))
        else:
            clip_f = self.model.get_empty_clip_sequence(1).squeeze(0)
            sync_f = self.model.get_empty_sync_sequence(1).squeeze(0)

        if text_exist:
            text_f = self._encode_text(str(row["caption"]))
        else:
            text_f = self.model.get_empty_string_sequence(1).squeeze(0)

        return clip_f, sync_f, text_f

    @torch.inference_mode()
    def _generate_batch(
        self, clip_f: torch.Tensor, sync_f: torch.Tensor, text_f: torch.Tensor
    ) -> torch.Tensor:
        bs = clip_f.shape[0]
        x0 = (
            torch.randn(
                bs,
                self.latent_seq_len,
                self.latent_dim,
                device=self.device,
                generator=self.rng,
            )
            * self.fm.noise_scale
        )

        conditions = self.model.preprocess_conditions(clip_f, sync_f, text_f)
        empty_conditions = self.model.get_empty_conditions(bs)

        def cfg_wrapper(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.model.ode_wrapper(
                t, x, conditions, empty_conditions, self.cfg_strength
            )

        x1 = self.fm.to_data(cfg_wrapper, x0)  # (B, latent_seq_len, latent_dim)
        audio = (x1 / self.audio_scale).cpu().float()
        audio = audio.reshape(bs, -1)  # (B, latent_seq_len*latent_dim)
        return audio

    def _save_wav(self, audio: torch.Tensor, sample_id: str) -> Path:
        if self.trim_to_duration:
            audio = audio[: self.target_samples]

        wav_np = audio.numpy()
        if self.loudness_norm:
            meter = pyln.Meter(self.sample_rate)
            current = meter.integrated_loudness(wav_np)
            if current > -70.0:
                wav_np = pyln.normalize.loudness(wav_np, current, self.target_lufs)
                peak = np.max(np.abs(wav_np))
                if peak > 1.0:
                    wav_np = wav_np / peak * 0.99
        out = self.output_dir / f"{sample_id}.wav"
        torchaudio.save(
            str(out),
            torch.from_numpy(wav_np).float().unsqueeze(0),
            self.sample_rate,
        )
        return out

    @torch.inference_mode()
    def run(self) -> None:
        df = pd.read_csv(self.config.data.csv_path)
        for col in ("video_path", "caption", "video_exist", "text_exist"):
            if col not in df.columns:
                raise ValueError(
                    f"Infer csv {self.config.data.csv_path} must contain column "
                    f"'{col}', found columns: {list(df.columns)}"
                )
        rows = df.to_dict("records")
        logger.info(
            f"Loaded {len(rows)} rows from {self.config.data.csv_path}, "
            f"batch_size={self.batch_size}"
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch_start in tqdm(range(0, len(rows), self.batch_size), desc="infer"):
                batch_rows = rows[batch_start : batch_start + self.batch_size]
                ids: list[str] = []
                clip_list: list[torch.Tensor] = []
                sync_list: list[torch.Tensor] = []
                text_list: list[torch.Tensor] = []

                for offset, row in enumerate(batch_rows):
                    sid = _row_id(row, batch_start + offset)
                    try:
                        clip_f, sync_f, text_f = self._features_for_row(row)
                    except Exception as e:
                        logger.warning(f"[SKIP] {sid}: feature extraction failed: {e}")
                        continue
                    ids.append(sid)
                    clip_list.append(clip_f)
                    sync_list.append(sync_f)
                    text_list.append(text_f)

                if not ids:
                    continue

                clip_b = torch.stack(clip_list, dim=0)
                sync_b = torch.stack(sync_list, dim=0)
                text_b = torch.stack(text_list, dim=0)

                audios = self._generate_batch(clip_b, sync_b, text_b)
                for j, sid in enumerate(ids):
                    out = self._save_wav(audios[j], sid)
                    logger.info(f"Saved {out}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    inferencer = WavflowInferencer(config)
    inferencer.run()


if __name__ == "__main__":
    main()
