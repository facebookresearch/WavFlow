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
T2A feature extraction (text -> CLIP text features).

Reads a local CSV manifest (``id, audio_path, caption``), runs the CLIP text
encoder over the captions on each rank's slice of the data (DistributedSampler
+ torchrun + DDP-style init), and writes:

- ``<output.pth_dir>/<id>.pth`` containing ``{"text_features": (77, 1024)}``
- ``<output.csv_path>`` with columns
  ``id, audio_path, features_path, video_exist, text_exist``. ``features_path``
  is automatically computed as the relative path from the CSV to each pth file.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
        -m feature_extract.extract_t2a_pth \\
        --config feature_extract/configs/extract_t2a.yaml
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from feature_extract.dataset.local_audio_text_dataset import (
    collate_fn_filter_none,
    LocalAudioTextDataset,
)
from feature_extract.ext.features_utils import FeaturesUtils
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2A feature extraction (torchrun)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config at {path} must be a mapping")
    return cfg


class T2AExtractor:
    @record
    def __init__(self, config: DictConfig) -> None:
        self.config = config

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))

        if not torch.cuda.is_available():
            raise RuntimeError("T2A extraction requires CUDA GPUs.")

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(
                    seconds=int(config.comm.init_timeout_seconds)
                ),
            )

        seed = int(config.extraction.seed) * self.world_size + self.rank
        torch.manual_seed(seed)

        self.pth_dir = Path(config.output.pth_dir)
        self.csv_path = Path(config.output.csv_path)
        self.data_root = (
            Path(config.data.data_root) if config.data.get("data_root", None) else None
        )
        if self.rank == 0:
            self.pth_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        self._setup_file_logging()
        self._setup_model(config)
        self._setup_dataloader(config)

        self.save_interval = int(config.extraction.get("save_interval", 128))

    def _setup_file_logging(self) -> None:
        log_file = self.pth_dir / f"extract_t2a_rank{self.rank}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s][Rank %(process)d] %(asctime)s - %(message)s"
            )
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
        logger.info(f"Logging to file: {log_file}")

    def _setup_model(self, config: DictConfig) -> None:
        self.feature_extractor = (
            FeaturesUtils(
                synchformer_ckpt=None,
                clip_model_name=config.model.get(
                    "clip_model_name", "ViT-H-14-378-quickgelu"
                ),
                clip_pretrained=config.model.get("clip_pretrained", None),
                enable_conditions=True,
            )
            .eval()
            .to(self.device)
        )
        if self.rank == 0:
            logger.info("Feature extractor loaded (CLIP text encoder only)")

    def _setup_dataloader(self, config: DictConfig) -> None:
        batch_size = int(config.extraction.batch_size)
        num_workers = int(config.extraction.num_workers)

        self.dataset = LocalAudioTextDataset(
            csv_path=str(config.data.csv_path),
            data_root=config.data.get("data_root", None),
            features_output_dir=str(self.pth_dir),
        )

        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn_filter_none,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        if self.rank == 0:
            logger.info(
                f"Dataset setup complete: {len(self.dataset)} rows, "
                f"batch_size={batch_size}, num_workers={num_workers}, "
                f"world_size={self.world_size}"
            )

    @record
    @torch.inference_mode()
    def run(self) -> None:
        rank_csv_path = (
            self.csv_path.parent / f"{self.csv_path.stem}_rank{self.rank}.csv"
        )
        error_log_path = (
            self.csv_path.parent / f"{self.csv_path.stem}_errors_rank{self.rank}.log"
        )

        rows_buffer: list[dict] = []
        pth_buffer: list[tuple[str, dict]] = []
        error_records: list[str] = []

        def make_row(sid: str, audio_path: str) -> dict:
            ap = Path(audio_path)
            if not ap.is_absolute() and self.data_root is not None:
                ap = self.data_root / ap
            return {
                "id": sid,
                "audio_path": str(ap),
                "features_path": f"{self.pth_dir}/{sid}.pth",
                "video_exist": 0,
                "text_exist": 1,
            }

        def flush() -> None:
            for sample_id, pth_data in pth_buffer:
                pth_path = self.pth_dir / f"{sample_id}.pth"
                try:
                    torch.save(pth_data, pth_path)
                except Exception as e:
                    error_records.append(f"{sample_id}\tpth_save_failed: {e}\n")
            if rows_buffer:
                df = pd.DataFrame(rows_buffer)
                write_header = not rank_csv_path.exists()
                df.to_csv(
                    rank_csv_path,
                    index=False,
                    mode="a",
                    header=write_header,
                )
            pth_buffer.clear()
            rows_buffer.clear()

        iterator = (
            tqdm(self.dataloader, desc=f"Rank {self.rank}")
            if self.rank == 0
            else self.dataloader
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in iterator:
                if not batch:
                    continue
                ids: list[str] = batch["id"]
                captions: list[str] = batch["caption"]
                audio_paths: list[str] = batch["audio_path"]

                new_indices: list[int] = []
                for i, sid in enumerate(ids):
                    if (self.pth_dir / f"{sid}.pth").exists():
                        rows_buffer.append(make_row(sid, audio_paths[i]))
                    else:
                        new_indices.append(i)
                if not new_indices:
                    continue

                new_captions = [captions[i] for i in new_indices]
                text_features = self.feature_extractor.encode_text(new_captions).cpu()

                for j, i in enumerate(new_indices):
                    sid = ids[i]
                    pth_buffer.append(
                        (sid, {"text_features": text_features[j].clone().float()})
                    )
                    rows_buffer.append(make_row(sid, audio_paths[i]))

                if len(pth_buffer) >= self.save_interval:
                    flush()

        flush()

        if error_records:
            with open(error_log_path, "w") as f:
                f.writelines(error_records)

        dist.barrier()

        if self.rank == 0:
            self._merge_results()

        logger.info(f"Rank {self.rank}: extraction completed")

    def _merge_results(self) -> None:
        all_records: list[dict] = []
        for r in range(self.world_size):
            rank_csv = self.csv_path.parent / f"{self.csv_path.stem}_rank{r}.csv"
            if rank_csv.exists():
                df = pd.read_csv(rank_csv, dtype={"id": str})
                all_records.extend(df.to_dict("records"))
                rank_csv.unlink()

        seen_ids: set[str] = set()
        unique_records: list[dict] = []
        for rec in all_records:
            sid = str(rec["id"])
            if sid not in seen_ids:
                seen_ids.add(sid)
                unique_records.append(rec)

        pd.DataFrame(
            unique_records,
            columns=[
                "id",
                "audio_path",
                "features_path",
                "video_exist",
                "text_exist",
            ],
        ).to_csv(self.csv_path, index=False)

        all_errors: list[str] = []
        for r in range(self.world_size):
            err = self.csv_path.parent / f"{self.csv_path.stem}_errors_rank{r}.log"
            if err.exists():
                all_errors.extend(err.read_text().splitlines(keepends=True))
                err.unlink()
        if all_errors:
            (self.csv_path.parent / f"{self.csv_path.stem}_errors.log").write_text(
                "".join(all_errors)
            )

        pth_count = len(list(self.pth_dir.glob("*.pth")))
        logger.info(
            f"Summary: {pth_count} .pth files in {self.pth_dir}, "
            f"{len(unique_records)} CSV rows -> {self.csv_path}, "
            f"{len(all_errors)} errors"
        )

    def close(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    extractor = T2AExtractor(config)
    try:
        extractor.run()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
