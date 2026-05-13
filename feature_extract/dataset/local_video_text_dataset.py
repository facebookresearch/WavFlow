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
Local CSV-based Video + Text dataset for VT2A feature extraction.

Reads a manifest CSV with columns ``id, audio_path, video_path, caption``.
Decodes the local video file with ``torio.io.StreamingMediaDecoder``, taking:
- 8 fps clip frames (resized to 384x384)
- 25 fps sync frames (resized & center-cropped to 224x224)

A single ``duration_sec`` window (default 8 s) is taken from the start of the
video. Videos shorter than the window are skipped.

The ``audio_path`` column is NOT decoded — it is just passed through to the
output CSV so users can pair their existing wavs with the extracted features.

Skipping logic:
- If the output ``pth/<id>.pth`` already exists, ``__getitem__`` returns
  ``None`` (resume-friendly).
- Decode failures or too-short videos return ``None``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

logger = logging.getLogger(__name__)

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def collate_fn_filter_none(batch: list[dict | None]) -> dict:
    """Collate batch of samples, filtering out None values."""
    valid_batch = [x for x in batch if x is not None]
    output: dict = {}
    if len(valid_batch) == 0:
        return output
    for key in valid_batch[0].keys():
        values = [x[key] for x in valid_batch if x.get(key) is not None]
        if len(values) == 0:
            continue
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output


def _resolve_path(value: str, data_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or data_root is None:
        return path
    return data_root / path


class LocalVideoTextDataset(Dataset):
    """Dataset over a local CSV manifest for VT2A feature extraction.

    Each row of the manifest must contain
    ``id, audio_path, video_path, caption``.

    ``__getitem__`` returns a dict with:
        - ``id``: sample id (str)
        - ``caption``: text caption (str)
        - ``audio_path``: original audio path string (passed through)
        - ``clip_video``: (T_clip, 3, 384, 384) float tensor (T_clip = 8*duration)
        - ``sync_video``: (T_sync, 3, 224, 224) float tensor (T_sync = 25*duration)
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str | None,
        duration_sec: float = 8.0,
        features_output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path, dtype={"id": str})
        for col in ("id", "audio_path", "video_path", "caption"):
            if col not in self.df.columns:
                raise ValueError(
                    f"VT2A manifest {csv_path} must contain column '{col}', "
                    f"found columns: {list(self.df.columns)}"
                )
        self.data_root = Path(data_root) if data_root else None
        self.duration_sec = duration_sec
        self.features_output_dir = (
            Path(features_output_dir) if features_output_dir else None
        )

        self.clip_expected_length = int(_CLIP_FPS * duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * duration_sec)

        self.clip_transform = v2.Compose(
            [
                v2.Resize(
                    (_CLIP_SIZE, _CLIP_SIZE),
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.sync_transform = v2.Compose(
            [
                v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(_SYNC_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _already_processed(self, sample_id: str) -> bool:
        if self.features_output_dir is None:
            return False
        return (self.features_output_dir / f"{sample_id}.pth").exists()

    def _decode_video(
        self, video_path: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Decode local video file into clip frames and sync frames."""
        reader = StreamingMediaDecoder(video_path)
        # Stream 0: CLIP frames (8 fps)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format="rgb24",
        )
        # Stream 1: Sync frames (25 fps)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="rgb24",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        # pyrefly: ignore [bad-index]
        return data_chunk[0], data_chunk[1]

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int) -> dict | None:
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        if self._already_processed(sample_id):
            return None

        try:
            audio_path = str(row["audio_path"])
            video_path_str = str(row["video_path"])
            caption = str(row["caption"])

            video_path = _resolve_path(video_path_str, self.data_root)
            if not video_path.exists():
                logger.warning(f"[SKIP] {sample_id}: video not found {video_path}")
                return None

            clip_chunk, sync_chunk = self._decode_video(str(video_path))

            if clip_chunk is None or clip_chunk.shape[0] < self.clip_expected_length:
                got = clip_chunk.shape[0] if clip_chunk is not None else "None"
                logger.warning(
                    f"[SKIP] {sample_id}: clip_too_short "
                    f"({got}/{self.clip_expected_length})"
                )
                return None
            clip_chunk = clip_chunk[: self.clip_expected_length]
            clip_chunk = self.clip_transform(clip_chunk)

            if sync_chunk is None or sync_chunk.shape[0] < self.sync_expected_length:
                got = sync_chunk.shape[0] if sync_chunk is not None else "None"
                logger.warning(
                    f"[SKIP] {sample_id}: sync_too_short "
                    f"({got}/{self.sync_expected_length})"
                )
                return None
            sync_chunk = sync_chunk[: self.sync_expected_length]
            sync_chunk = self.sync_transform(sync_chunk)

            return {
                "id": sample_id,
                "caption": caption,
                "audio_path": audio_path,
                "clip_video": clip_chunk,
                "sync_video": sync_chunk,
            }
        except Exception as e:
            logger.warning(
                f"[SKIP] {sample_id}: exception during decode/transform: {e}"
            )
            return None
