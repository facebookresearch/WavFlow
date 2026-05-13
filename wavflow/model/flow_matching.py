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
from typing import Callable, Optional

import torch
from torchdiffeq import odeint

log = logging.getLogger()


def log_normal_sample(
    x: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0,
) -> torch.Tensor:
    bs = x.shape[0]
    sample = torch.randn(bs, device=x.device, generator=generator) * s + m
    return torch.sigmoid(sample)


# Partially from https://github.com/gle-bellier/flow-matching
class FlowMatching:
    def __init__(
        self,
        min_sigma: float = 0.0,
        inference_mode: str = "euler",
        num_steps: int = 25,
        prediction_type: str = "x",
        noise_scale: float = 1.0,
        noise_shift: float = 1.0,
    ) -> None:
        # inference_mode: 'euler' or 'adaptive'
        # num_steps: number of steps in the euler inference mode
        # prediction_type: 'x' (predict x1) or 'v' (predict velocity)
        # noise_scale: scale factor for noise magnitude (for maintaining SNR across resolutions)
        # noise_shift: shift factor for noise schedule (>1 pushes t towards 0/noise end)
        #   formula: t_s = t / (t + shift * (1 - t)), SNR reduced by shift^2
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        self.prediction_type = prediction_type
        self.noise_scale = noise_scale
        self.noise_shift = noise_shift

        assert self.inference_mode in ["euler", "adaptive"]
        assert self.prediction_type in ["x", "v"]
        if self.inference_mode == "adaptive" and num_steps > 0:
            log.info("The number of steps is ignored in adaptive inference mode ")

    def shift_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Apply noise shift: t_s = t / (t + shift * (1 - t)).
        When shift=1, t_s=t (no change).
        When shift>1, t_s < t, pushing towards noise end (t=0).
        """
        if self.noise_shift == 1.0:
            return t
        return t / (t + self.noise_shift * (1 - t))

    def get_conditional_flow(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # which is psi_t(x), eq 22 in flow matching for generative models
        # t should already be in shifted space if noise_shift is used
        t = t[:, None, None].expand_as(x0)
        return (1 - (1 - self.min_sigma) * t) * x0 + t * x1

    def x_pred_x_loss(
        self,
        pred: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # x-prediction + x-loss: (pred_x1 - x1)^2
        reduce_dim = list(range(1, len(pred.shape)))
        return (pred - x1).pow(2).mean(dim=reduce_dim)

    def x_pred_v_loss(
        self,
        pred: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # x-prediction + v-loss: convert x1 pred to velocity, then (pred_v - target_v)^2
        t_expanded = t[:, None, None].expand_as(pred)
        one_minus_t = (1 - t_expanded).clamp(min=1e-6)
        predicted_v = (pred - xt) / one_minus_t
        target_v = (x1 - xt) / one_minus_t
        reduce_dim = list(range(1, len(pred.shape)))
        return (predicted_v - target_v).pow(2).mean(dim=reduce_dim)

    def v_pred_v_loss(
        self,
        pred: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # v-prediction + v-loss: (pred_v - target_v)^2
        # target_v is the derivative of conditional flow: dx/dt = x1 - (1 - min_sigma) * x0
        target_v = x1 - (1 - self.min_sigma) * x0
        reduce_dim = list(range(1, len(pred.shape)))
        return (pred - target_v).pow(2).mean(dim=reduce_dim)

    def v_pred_x_loss(
        self,
        pred: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # v-prediction + x-loss: convert velocity to x1, then (pred_x1 - x1)^2
        # x1 = xt + (1 - t) * v
        t_expanded = t[:, None, None].expand_as(pred)
        one_minus_t = 1 - t_expanded
        predicted_x1 = xt + one_minus_t * pred
        reduce_dim = list(range(1, len(pred.shape)))
        return (predicted_x1 - x1).pow(2).mean(dim=reduce_dim)

    def get_x0_xt_c(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        Cs: list[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]
    ]:
        x0 = torch.empty_like(x1).normal_(generator=generator) * self.noise_scale
        t_shifted = self.shift_timestep(t)
        xt = self.get_conditional_flow(x0, x1, t_shifted)
        return x0, x1, xt, t_shifted, Cs

    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1, 1, 0)

    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0, 0, 1)

    def run_t0_to_t1(
        self, fn: Callable, x0: torch.Tensor, t0: float, t1: float
    ) -> torch.Tensor:
        # fn: a function that takes (t, x) and returns model prediction
        # prediction_type='x': fn returns predicted_x1, converted to velocity for ODE
        # prediction_type='v': fn returns predicted_velocity, used directly for ODE

        if self.inference_mode == "adaptive":

            def velocity_fn(t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                t_s = self.shift_timestep(t_scalar)
                pred = fn(t_s, x)
                if self.prediction_type == "x":
                    one_minus_t = (1 - t_s).clamp(min=1e-6)
                    return (pred - x) / one_minus_t
                else:
                    return pred

            result = odeint(
                velocity_fn,
                x0,
                torch.tensor([t0, t1], device=x0.device, dtype=x0.dtype),
            )
            return result[-1]
        elif self.inference_mode == "euler":
            x = x0
            raw_steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1)
            steps = self.shift_timestep(raw_steps)
            for ti, t in enumerate(steps[:-1]):
                pred = fn(t, x)
                if self.prediction_type == "x":
                    one_minus_t = max(1 - t, 1e-6)
                    v = (pred - x) / one_minus_t
                else:
                    v = pred
                next_t = steps[ti + 1]
                dt = next_t - t
                x = x + dt * v
            return x
        raise ValueError(f"Unknown inference mode: {self.inference_mode}")
