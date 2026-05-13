# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Auto-resolve helper for WavFlow's external weights.

Three external artifacts are needed at runtime:

1. CLIP (DFN5B-CLIP-ViT-H-14-384) - public on HuggingFace; loaded directly by
   ``open_clip.create_model_from_pretrained("hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")``
   so it does not need this module.
2. Synchformer state dict - mirrored on the public MMAudio GitHub release;
   downloaded via direct URL the first time it is needed.
3. ``empty_string.pth`` - this is just CLIP("")'s per-token text embedding.
   We do **not** download it: we recompute it locally on first use (~5s, runs
   once) and cache the result. For inference with a trained checkpoint this
   file is not even needed - the empty_string buffer is restored from the ckpt.

Public entry point: :func:`resolve_weight`. It returns a usable local path:

* If the user-configured path exists, it is returned as-is.
* Otherwise the file is fetched / computed into ``$WAVFLOW_CACHE_DIR``
  (default ``~/.cache/wavflow``) and that path is returned.
"""

# pyre-strict

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlopen


logger: logging.Logger = logging.getLogger(__name__)

CLIP_MODEL_HF: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
CLIP_TOKENIZER_NAME: str = "ViT-H-14-378-quickgelu"

# Direct URL for Synchformer weights (mirrored by MMAudio).
SYNCHFORMER_URL: str = (
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
    "synchformer_state_dict.pth"
)


def cache_dir() -> Path:
    """Local cache directory used for auto-fetched weights."""
    root = os.environ.get("WAVFLOW_CACHE_DIR") or (Path.home() / ".cache" / "wavflow")
    p = Path(root).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_url(url: str, dest: Path) -> None:
    """Stream a remote file to ``dest`` using stdlib only (no extra deps)."""
    logger.info("[wavflow] Downloading %s ...", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def _resolve_synchformer(user_path: Optional[str]) -> str:
    """Return a local path to ``synchformer_state_dict.pth``."""
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_file():
            return str(p)
        logger.warning(
            "[wavflow] Synchformer ckpt not found at %s - falling back to "
            "auto-download.",
            p,
        )
    local = cache_dir() / "synchformer_state_dict.pth"
    if not local.is_file():
        _download_url(SYNCHFORMER_URL, local)
    return str(local)


def _compute_empty_string(dest: Path) -> None:
    """Compute and save the per-token CLIP("") features.

    The output tensor has shape ``(1, 77, 1024)`` matching what trainer.py and
    infer.py expect (i.e. the ``[0]`` indexing yields a ``(77, 1024)`` tensor).
    """
    import torch

    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError(
            "Computing empty_string.pth requires `open_clip_torch`. "
            "Run `pip install open_clip_torch` and retry, or pre-place the "
            "file at the configured `model.empty_string_ckpt` path."
        ) from e

    logger.info(
        "[wavflow] Computing empty_string features locally with CLIP "
        "(runs once, cached at %s).",
        dest,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = open_clip.create_model_from_pretrained(
        CLIP_MODEL_HF, return_transform=False
    )
    if isinstance(clip_model, tuple):  # older open_clip API returns a tuple
        clip_model = clip_model[0]
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_TOKENIZER_NAME)

    # Replicate the per-token text encoding used by feature_extract/ext/features_utils.py.
    with torch.no_grad():
        text = tokenizer([""]).to(device)
        cast_dtype = clip_model.transformer.get_cast_dtype()
        x = clip_model.token_embedding(text).to(cast_dtype)
        x = x + clip_model.positional_embedding.to(cast_dtype)
        x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
        x = clip_model.ln_final(x)  # (1, 77, 1024)

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.detach().cpu(), dest)


def _resolve_empty_string(user_path: Optional[str]) -> str:
    """Return a local path to ``empty_string.pth``, computing it if missing."""
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_file():
            return str(p)
        logger.warning(
            "[wavflow] empty_string ckpt not found at %s - falling back to "
            "local CLIP computation.",
            p,
        )
    local = cache_dir() / "empty_string.pth"
    if not local.is_file():
        _compute_empty_string(local)
    return str(local)


def resolve_weight(
    user_path: Optional[str],
    *,
    weight_key: str,
) -> str:
    """Resolve a weight path, fetching/computing it if missing.

    Args:
        user_path: Path the user configured (may be ``None``, missing, or valid).
        weight_key: One of ``"synchformer"`` or ``"empty_string"``.

    Returns:
        Absolute local path to a file that exists on disk.
    """
    if weight_key == "synchformer":
        return _resolve_synchformer(user_path)
    if weight_key == "empty_string":
        return _resolve_empty_string(user_path)
    raise ValueError(
        f"resolve_weight: unknown weight_key '{weight_key}'. "
        f"Expected 'synchformer' or 'empty_string'."
    )
