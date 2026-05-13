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

from typing import Union

import torch
from einops import rearrange
from torch import Tensor

# Ref: https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
# Ref: https://github.com/lucidrains/rotary-embedding-torch


def compute_rope_rotations(
    length: int,
    dim: int,
    theta: int,
    *,
    freq_scaling: float = 1.0,
    device: Union[torch.device, str] = "cpu",
) -> Tensor:
    assert dim % 2 == 0

    with torch.amp.autocast(device_type="cuda", enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        inv_freq_exponent = torch.arange(
            0, dim, 2, dtype=torch.float32, device=device
        ).div(dim)
        freqs = torch.pow(theta, -inv_freq_exponent)
        freqs *= freq_scaling

        rot = torch.einsum("..., f -> ... f", pos, freqs)
        rot = torch.stack(
            [torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1
        )
        rot = rearrange(rot, "n d (i j) -> 1 n d i j", i=2, j=2)
        return rot


def apply_rope(x: Tensor, rot: Tensor) -> Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        _x = x.float()
        _x = _x.view(*_x.shape[:-1], -1, 1, 2)
        x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
        return x_out.reshape(*x.shape).to(dtype=x.dtype)
