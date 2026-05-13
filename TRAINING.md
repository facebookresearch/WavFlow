# Training Guide

This document covers feature extraction and training for **WavFlow**.

For installation and inference, see the [main README](README.md).

---

## Step 1 — Feature extraction

Two independent pipelines:
- **T2A**: text → CLIP text feature only (use this when you don't have video)
- **VT2A**: video + text → CLIP frame + Synchformer + CLIP text features

### Input CSV

T2A `feature_extract/configs/extract_t2a.yaml` → `data.csv_path`:
```csv
id,audio_path,caption
sample1,/abs/or/relative/wav/sample1.wav,a whistling rocket explodes
```

VT2A `feature_extract/configs/extract_vt2a.yaml` → `data.csv_path`:
```csv
id,audio_path,video_path,caption
sample1,/abs/or/relative/wav/sample1.wav,/abs/or/relative/video/sample1.mp4,a whistling rocket explodes
```

Notes:
- Captions with commas must be quoted (`"a, b, c"`).
- Paths can be absolute, or relative to `data.data_root` set in the same yaml.
- Videos must be ≥ `extraction.duration_sec` (default 8 s); shorter clips are skipped.

### Output

```
<output.pth_dir>/<id>.pth        # one pth per sample
<output.csv_path>                # final CSV (consumed by training)
```

Per-pth contents:
- T2A: `{"text_features": (77, 1024)}`
- VT2A: `{"clip_features": (T_clip, 1024), "sync_features": (T_sync, 768), "text_features": (77, 1024)}`

The output csv has columns `id, audio_path, features_path, video_exist, text_exist` with both path columns absolute, so the resulting csv can be fed directly to training (no `data_root` needed).

### Run

```bash
# T2A — defaults to NPROC_PER_NODE=2
bash scripts/launch/extract_t2a.sh

# VT2A — defaults to NPROC_PER_NODE=2
bash scripts/launch/extract_vt2a.sh

# Use a custom GPU count or config
NPROC_PER_NODE=4 bash scripts/launch/extract_vt2a.sh
CONFIG_PATH=path/to/your_extract.yaml bash scripts/launch/extract_t2a.sh
```

Re-runs are resume-friendly: any `<output.pth_dir>/<id>.pth` that already exists is skipped on the second run.

---

## Step 2 — Training

Edit `wavflow/configs/train.yaml` (which extends `base.yaml` for shared shape / model fields):

| Field | What to set |
|---|---|
| `data.csv_path` | the merged csv produced by Step 1, e.g. `training_samples/train_vt2a.csv` |
| `data.data_root` | leave `null` if csv paths are absolute (default with our extract scripts) |
| `model.empty_string_ckpt` | path to `empty_string.pth` |
| `training.output_dir` | where to write checkpoints / training.log / samples |
| `training.batch_size`, `epochs`, etc. | as needed |

### Single-node multi-GPU

```bash
# Default: 2 GPUs
bash scripts/launch/train_single_node.sh

# 8 GPUs
NPROC_PER_NODE=8 bash scripts/launch/train_single_node.sh

# Custom config + extra args forwarded to wavflow.train
bash scripts/launch/train_single_node.sh --config wavflow/configs/train.yaml
```

### Multi-node

Set `NNODES`, `NODE_RANK`, `MASTER_ADDR` on **every** node, then launch the same script:

```bash
# Node 0
NNODES=4 NODE_RANK=0 MASTER_ADDR=node0 NPROC_PER_NODE=8 \
    bash scripts/launch/train_multi_node.sh

# Node 1
NNODES=4 NODE_RANK=1 MASTER_ADDR=node0 NPROC_PER_NODE=8 \
    bash scripts/launch/train_multi_node.sh

# ... and so on for the remaining nodes
```

| Env var | Default | Description |
|---|---|---|
| `NNODES` | *(required)* | total number of nodes |
| `NODE_RANK` | *(required)* | rank of this node, 0..NNODES-1 |
| `MASTER_ADDR` | *(required)* | hostname / IP of node 0 |
| `MASTER_PORT` | `29500` | rendezvous port |
| `NPROC_PER_NODE` | `8` | GPUs per node |
| `CONFIG_PATH` | `wavflow/configs/train.yaml` | YAML config |

### Training outputs

Inside `<training.output_dir>/<experiment_id>/`:

- `checkpoint_latest.pth` — refreshed every `training.checkpoint_freq` epochs
- `checkpoint_epoch_<N>.pth` — every `training.checkpoint_save_epoch` epochs
- `ema_epoch_<N>.pth` — EMA-only weights for inference
- `samples/epoch_<N>/{gt,gen}_rank{R}_id_<sample_id>.wav` — generated audio every `training.gen_train_freq` epochs
- `training.log` — full training log

Resume happens automatically: if `checkpoint_latest.pth` exists in the experiment dir, training picks up from there.

---

## End-to-end example (the included demo data)

```bash
cd wavflow_project
conda activate wavflow

# 1. extract VT2A features for the 4 demo videos in training_samples/video/
bash scripts/launch/extract_vt2a.sh
#   -> writes training_samples/features/vt2a/sample{1..4}.pth
#   -> writes training_samples/train_vt2a.csv

# 2. train (train.yaml already points data.csv_path at train_vt2a.csv)
bash scripts/launch/train_single_node.sh

# 3. once a checkpoint exists, generate audio for any video/text via inference
bash scripts/launch/predict.sh
#   -> writes output/infer/<id>.wav
```

> 💡 **Smoke-test tip:** with only 4 demo samples the loss curve will be very
> noisy and the model will need many epochs to produce listenable audio. To
> verify the pipeline end-to-end in just a few hundred steps, open
> `wavflow/trainer.py`, find the `# Smoke-test trick` block, and uncomment the
> `repeat_factor = 32` lines. This makes the model overfit on the 4 samples
> and quickly start generating recognizable sounds — useful as a sanity check
> before launching a real training run on a larger dataset.

---

## Mixing T2A and VT2A in the same training run

The codebase supports joint training on **VT2A** (video + text), **T2A** (text-only), and **V2A** (video-only) samples — a single model can learn from all three modalities at once. Concatenate the per-task csvs into one and point `data.csv_path` at the merged file:

```bash
head -n 1 training_samples/train_vt2a.csv > training_samples/train_mixed.csv
tail -n +2 -q training_samples/train_vt2a.csv training_samples/train_t2a.csv \
    >> training_samples/train_mixed.csv
```

Then set `data.csv_path: training_samples/train_mixed.csv` in `wavflow/configs/train.yaml`. The trainer reads each row's `video_exist` / `text_exist` flags and handles the missing modality automatically.
