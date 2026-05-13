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
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)


def derive_audio_shapes(
    target_sample_rate: int, audio_duration: float, audio_dim: int
) -> tuple[int, int, int]:
    """Derive (audio_seq_len, padded_samples, target_samples) from audio config.

    target_samples = round(sr * duration) is the actual number of audio samples
    we care about. audio_seq_len = ceil(target_samples / audio_dim) is how many
    latent tokens the model produces, and padded_samples = audio_seq_len *
    audio_dim is the (possibly zero-padded) length the dataset uses to reshape
    the wav into (audio_seq_len, audio_dim).
    """
    target_samples = int(round(target_sample_rate * audio_duration))
    audio_seq_len = (target_samples + audio_dim - 1) // audio_dim
    padded_samples = audio_seq_len * audio_dim
    return audio_seq_len, padded_samples, target_samples


class CSVAudioFeaturesDataset(Dataset):
    """Dataset that loads audio and feature tensors from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        data_root: str | None,
        *,
        target_sample_rate: int,
        audio_duration: float,
        audio_dim: int,
        audio_scale: float,
        clip_seq_len: int,
        clip_dim: int,
        sync_seq_len: int,
        sync_dim: int,
        text_seq_len: int,
        text_dim: int,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root) if data_root else None
        self.audio_scale = audio_scale
        self.target_sample_rate = target_sample_rate
        self.audio_duration = audio_duration
        self.audio_samples_per_token = audio_dim
        self.audio_seq_len, self.padded_samples, self.target_samples = (
            derive_audio_shapes(target_sample_rate, audio_duration, audio_dim)
        )
        # Pre-allocated zero placeholders for rows whose pth lacks a modality.
        # When video_exist=0 the pth typically has only text_features; when
        # text_exist=0 it has only clip/sync. Trainer's ~exist masking will
        # later swap these zeros for the model's learned empty_*_feat.
        self.fake_clip_features = torch.zeros(clip_seq_len, clip_dim)
        self.fake_sync_features = torch.zeros(sync_seq_len, sync_dim)
        self.fake_text_features = torch.zeros(text_seq_len, text_dim)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or self.data_root is None:
            return path
        return self.data_root / path

    def _normalize_to_target_rms(self, wav: torch.Tensor) -> torch.Tensor:
        """Normalize audio to target 0.33 rms, samples in [-1, 1] range."""
        target_rms = 0.33
        rms = torch.sqrt(torch.mean(torch.pow(wav, 2)))
        if rms > 1e-8:
            wav = wav * target_rms / rms
        # Conditional Peak Limiting
        # Goal: protect overall loudness, not the peaks themselves.
        # After RMS normalization above, RMS == target_rms (0.33). If a few
        # samples still spike past peak_limit (1.0), there are two options:
        #   (a) scale the whole wav by peak_limit/peak so nothing exceeds 1
        #       -> no clipping distortion, but the body of the audio gets
        #          quieter (new_rms drops below target_rms).
        #   (b) skip the scaling and let the final torch.clamp() flatten
        #       just those spike samples -> body keeps its loudness, but the
        #       spike samples get hard-clipped (mild distortion).
        # We choose (a) only when the resulting RMS would still be > min_rms,
        # i.e. when the peak is only slightly above 1.0 so scaling barely
        # costs loudness. For very large peaks (e.g. percussive transients
        # with peak >> 1) the scaling would crush the whole signal, so we
        # fall through to (b) and let clamp eat the spikes.
        peak_limit = 1.0
        min_rms = 0.3
        rms_after_gain = torch.sqrt(torch.mean(wav**2))
        peak = torch.max(torch.abs(wav))
        if peak > peak_limit:
            scale = peak_limit / peak
            new_rms = rms_after_gain * scale
            if new_rms > min_rms:
                wav = wav * scale

        wav = torch.clamp(wav, -1.0, 1.0)
        return wav

    def _process_audio(self, audio_source: str) -> torch.Tensor | None:
        try:
            waveform, sample_rate = torchaudio.load(audio_source)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.target_sample_rate:
                waveform = torchaudio.transforms.Resample(
                    sample_rate, self.target_sample_rate
                )(waveform)

            waveform = waveform.squeeze(0)
            waveform = self._normalize_to_target_rms(waveform)

            current_samples = waveform.shape[0]
            if current_samples < self.padded_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.padded_samples - current_samples)
                )
            elif current_samples > self.padded_samples:
                waveform = waveform[: self.padded_samples]

            audio_tokens = waveform.view(
                self.audio_seq_len, self.audio_samples_per_token
            )
            return audio_tokens * self.audio_scale
        except Exception as e:
            log.warning(f"Error processing audio: {e}, return null")
            return None

    def _process_features(
        self, features_source: str, video_exist: bool, text_exist: bool
    ) -> dict[str, torch.Tensor] | None:
        """Load .pth features. Missing modalities are substituted with zeros.

        For pth files that only contain a subset of {clip,sync,text}_features
        (e.g. T2A extract_t2a_pth.py output only writes text_features), the
        absent keys fall back to pre-allocated zero tensors of the correct
        shape. Trainer's ``clip_f[~video_exist] = empty_clip_feat`` swap then
        replaces those zeros with the model's learned empty tokens.

        Per-row video_exist / text_exist from the csv are also used to force
        zeros even if the pth happens to contain a stale value (defensive).
        """
        try:
            features_dict = torch.load(
                features_source,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as e:
            log.warning(f"Error processing features: {e}, returning None features")
            return None

        if video_exist:
            clip_features = features_dict.get("clip_features", self.fake_clip_features)
            sync_features = features_dict.get("sync_features", self.fake_sync_features)
        else:
            clip_features = self.fake_clip_features
            sync_features = self.fake_sync_features

        if text_exist:
            text_features = features_dict.get("text_features", self.fake_text_features)
        else:
            text_features = self.fake_text_features

        return {
            "clip_features": clip_features,
            "sync_features": sync_features,
            "text_features": text_features,
        }

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int) -> dict[str, object] | None:
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        audio_path = self._resolve_path(str(row["audio_path"]))
        features_path = self._resolve_path(str(row["features_path"]))
        video_exist = bool(int(row.get("video_exist", 1)))
        text_exist = bool(int(row.get("text_exist", 1)))

        audio_tokens = self._process_audio(str(audio_path))
        features = self._process_features(
            str(features_path), video_exist=video_exist, text_exist=text_exist
        )
        if audio_tokens is None or features is None:
            return None

        return {
            "id": sample_id,
            "audio_tokens": audio_tokens,
            "clip_features": features["clip_features"],
            "sync_features": features["sync_features"],
            "text_features": features["text_features"],
            "video_exist": torch.tensor(video_exist, dtype=torch.bool),
            "text_exist": torch.tensor(text_exist, dtype=torch.bool),
        }


def collate_fn_filter_none(batch: list[dict[str, object] | None]) -> dict[str, object]:
    """Collate batch of samples, filtering out None values."""
    valid_batch = [x for x in batch if x is not None]
    output = {}
    if len(valid_batch) == 0:
        return output
    for key in valid_batch[0].keys():
        values = [x[key] for x in valid_batch if x.get(key) is not None]
        if len(values) == 0:
            continue
        if isinstance(values[0], torch.Tensor):
            tensor_values: list[torch.Tensor] = [
                v for v in values if isinstance(v, torch.Tensor)
            ]
            output[key] = torch.stack(tensor_values)
        else:
            output[key] = values
    return output


def build_audio_dataset_from_config(config) -> CSVAudioFeaturesDataset:
    """Build CSV dataset from training config."""
    data_cfg = config.data

    return CSVAudioFeaturesDataset(
        csv_path=str(data_cfg.csv_path),
        data_root=getattr(data_cfg, "data_root", None),
        target_sample_rate=int(data_cfg.target_sample_rate),
        audio_duration=float(data_cfg.audio_duration),
        audio_dim=int(data_cfg.audio_dim),
        audio_scale=float(data_cfg.audio_scale),
        clip_seq_len=int(data_cfg.clip_seq_len),
        clip_dim=int(data_cfg.clip_dim),
        sync_seq_len=int(data_cfg.sync_seq_len),
        sync_dim=int(data_cfg.sync_dim),
        text_seq_len=int(data_cfg.text_seq_len),
        text_dim=int(data_cfg.text_dim),
    )
