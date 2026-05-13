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
Local CSV-based Audio + Text dataset for T2A feature extraction.

Reads a manifest CSV with columns ``id, audio_path, caption``. The audio file
itself is NOT read here — its path is just passed through to the output CSV.
We only need the caption to compute CLIP text features.

Skipping logic:
- If the output ``pth/<id>.pth`` already exists, ``__getitem__`` returns
  ``None`` (resume-friendly).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


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


class LocalAudioTextDataset(Dataset):
    """Dataset over a local CSV manifest for T2A feature extraction.

    Each row of the manifest must contain ``id, audio_path, caption``.

    ``__getitem__`` returns a dict with:
        - ``id``: sample id (str)
        - ``caption``: text caption (str)
        - ``audio_path``: original audio path string (passed through verbatim)
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str | None,
        features_output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path, dtype={"id": str})
        for col in ("id", "audio_path", "caption"):
            if col not in self.df.columns:
                raise ValueError(
                    f"T2A manifest {csv_path} must contain column '{col}', "
                    f"found columns: {list(self.df.columns)}"
                )
        self.data_root = Path(data_root) if data_root else None
        self.features_output_dir = (
            Path(features_output_dir) if features_output_dir else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def _already_processed(self, sample_id: str) -> bool:
        if self.features_output_dir is None:
            return False
        return (self.features_output_dir / f"{sample_id}.pth").exists()

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int) -> dict | None:
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        if self._already_processed(sample_id):
            return None

        try:
            audio_path = str(row["audio_path"])
            caption = str(row["caption"])
            # Validate audio path resolves (purely diagnostic; we don't read it)
            _ = _resolve_path(audio_path, self.data_root)
            return {
                "id": sample_id,
                "caption": caption,
                "audio_path": audio_path,
            }
        except Exception as e:
            logger.warning(f"[SKIP] {sample_id}: exception while reading row: {e}")
            return None
