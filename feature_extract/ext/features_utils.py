# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize

from .synchformer import Synchformer


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):
    def __init__(
        self,
        *,
        synchformer_ckpt: Optional[str] = None,
        clip_model_name: str = "ViT-H-14-378-quickgelu",
        clip_pretrained: Optional[str] = None,
        enable_conditions: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            if clip_pretrained is not None:
                # Load from local/manifold path (for MAST without internet)
                clip_model, _, _ = open_clip.create_model_and_transforms(
                    clip_model_name, pretrained=clip_pretrained
                )
            else:
                # Load from HuggingFace Hub (for local dev with internet)
                clip_model = create_model_from_pretrained(
                    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False
                )
            self.clip_model = patch_clip(clip_model)
            self.clip_preprocess = Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            self.synchformer = Synchformer()
            # Auto-download synchformer weights from HuggingFace if missing.
            from wavflow.utils.auto_download import resolve_weight

            sync_path = resolve_weight(synchformer_ckpt, weight_key="synchformer")
            self.synchformer.load_state_dict(
                torch.load(sync_path, weights_only=True, map_location="cpu")
            )

            self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        else:
            self.clip_model = None
            # pyrefly: ignore [bad-assignment]
            self.synchformer = None
            self.tokenizer = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            # pyrefly: ignore [bad-assignment]
            self.synchformer = torch.compile(self.synchformer)

    # pyrefly: ignore [bad-override]
    def train(self, mode: bool) -> "FeaturesUtils":
        super().train(False)
        return self

    @torch.inference_mode()
    def encode_video_with_clip(
        self, x: torch.Tensor, batch_size: int = -1
    ) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(
                self.clip_model.encode_image(x[i : i + batch_size], normalize=True)
            )
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, "(b t) d -> b t d", b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_sync(
        self, x: torch.Tensor, batch_size: int = -1
    ) -> torch.Tensor:
        assert self.synchformer is not None, "Synchformer is not loaded"
        # x: (B, T, C, H, W) H/W: 224

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size : i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i : i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
        return x

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        assert self.tokenizer is not None, "Tokenizer is not loaded"
        # x: (B, L)
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
