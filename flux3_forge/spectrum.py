"""Experimental Spectrum feature forecasting for FLUX 3 shared blocks.

Adapted from hanjq17/Spectrum commit 4e41f91c93ff66b045332530dd7bcb4c60eafbab
(MIT, Copyright 2026 Jiaqi Han). This module forecasts the shared-transformer
feature residual with a ridge-fitted Chebyshev basis blended with local
discrete Taylor differences. It never extrapolates diffusion velocity.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import torch

SOURCE_REPOSITORY = "https://github.com/hanjq17/Spectrum"
SOURCE_COMMIT = "4e41f91c93ff66b045332530dd7bcb4c60eafbab"
SOURCE_LICENSE = "MIT"


@dataclass(frozen=True)
class SpectrumConfig:
    enabled: bool = False
    order: int = 2
    history_points: int = 4
    ridge: float = 0.1
    blend: float = 0.5
    taylor_order: int = 1
    max_consecutive_skips: int = 1
    max_horizon_ratio: float = 1.5
    warmup_steps: int = 4
    final_steps: int = 3
    max_forecast_steps: int = 3
    max_cache_mb: int = 2048
    work_mb: int = 64

    def __post_init__(self):
        integer_fields = ("order", "history_points", "taylor_order", "max_consecutive_skips",
                          "warmup_steps", "final_steps", "max_forecast_steps", "max_cache_mb", "work_mb")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Spectrum {name} must be an integer")
        for name in ("ridge", "blend", "max_horizon_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                raise ValueError(f"Spectrum {name} must be a finite number")
        if not 1 <= self.order <= 4 or self.history_points < self.order + 2:
            raise ValueError("Spectrum history must be at least order + 2, with order from 1 to 4")
        if self.taylor_order != 1:
            raise ValueError("Spectrum currently supports only linear Taylor for irregular target times")
        if self.ridge < 0 or not 0.0 <= self.blend <= 1.0 or self.max_horizon_ratio <= 0:
            raise ValueError("Spectrum ridge/blend/horizon values are invalid")
        if self.max_consecutive_skips != 1:
            raise ValueError("experimental Spectrum currently permits exactly one consecutive skip")
        if self.warmup_steps < 0 or self.final_steps < 1 or self.max_forecast_steps < 0:
            raise ValueError("Spectrum warmup/final/forecast step limits are invalid")
        if not 16 <= self.max_cache_mb <= 2048 or not 1 <= self.work_mb <= 256:
            raise ValueError("Spectrum memory limits are invalid")

    @classmethod
    def from_request(cls, value) -> "SpectrumConfig":
        if not value:
            return cls()
        if value is True:
            value = {"enabled": True}
        if not isinstance(value, dict):
            raise ValueError("spectrum must be false, true, or an options dictionary")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown Spectrum options: {sorted(unknown)}")
        config = cls(**value)
        if not 1 <= config.order <= 4:
            raise ValueError("Spectrum order must be from 1 to 4")
        if config.history_points < config.order + 2:
            raise ValueError("Spectrum history_points must be at least order + 2")
        if config.taylor_order != 1:
            raise ValueError("Spectrum currently supports only linear Taylor for irregular target times")
        if config.ridge < 0:
            raise ValueError("Spectrum ridge must be nonnegative")
        if not 0.0 <= config.blend <= 1.0:
            raise ValueError("Spectrum blend must be from 0 to 1")
        if not 16 <= config.max_cache_mb <= 2048:
            raise ValueError("Spectrum max_cache_mb must be from 16 to 2048")
        if not 1 <= config.work_mb <= 256:
            raise ValueError("Spectrum work_mb must be from 1 to 256")
        if config.max_consecutive_skips != 1:
            raise ValueError("experimental Spectrum currently permits exactly one consecutive skip")
        if config.max_horizon_ratio <= 0:
            raise ValueError("Spectrum max_horizon_ratio must be positive")
        if config.warmup_steps < 0 or config.final_steps < 1 or config.max_forecast_steps < 0:
            raise ValueError("Spectrum warmup/final/forecast step limits are invalid")
        return config


class SpectrumRejected(RuntimeError):
    """Forecast was unsafe; caller must run the real shared block stack."""


class SpectrumState:
    """Request-scoped bounded history and spectral residual forecaster."""

    def __init__(self, config: SpectrumConfig):
        self.config = config
        self.times: list[float] = []
        self.features: list[torch.Tensor] = []
        self.cache_bytes = 0
        self.disabled_reason: str | None = None
        self.consecutive_skips = 0
        self.full_steps = 0
        self.forecast_steps = 0

    @property
    def active(self) -> bool:
        return self.config.enabled and self.disabled_reason is None

    @property
    def ready(self) -> bool:
        return self.active and len(self.features) >= self.config.order + 2

    def _horizon_ok(self, target_time: float) -> bool:
        if len(self.times) < 2:
            return False
        last_step = abs(self.times[-1] - self.times[-2])
        if last_step <= 1e-12:
            return False
        return abs(float(target_time) - self.times[-1]) <= self.config.max_horizon_ratio * last_step

    def should_compute(self, step: int, total: int, target_time: float) -> bool:
        """True runs all shared blocks; False permits one forecasted residual."""
        if (not self.ready or step < self.config.warmup_steps or
                step >= total - self.config.final_steps or
                self.forecast_steps >= self.config.max_forecast_steps or
                not self._horizon_ok(target_time)):
            self.consecutive_skips = 0
            return True
        if self.consecutive_skips >= self.config.max_consecutive_skips:
            self.consecutive_skips = 0
            return True
        self.consecutive_skips += 1
        self.forecast_steps += 1
        return False

    @torch.no_grad()
    def observe(self, target_time: float, residual: torch.Tensor) -> None:
        """Store a BF16 shared-stack residual after a real block evaluation."""
        if not self.active:
            return
        if not math.isfinite(float(target_time)):
            self.disable("nonfinite target time")
            return
        if self.features and residual.shape != self.features[-1].shape:
            self.disable("feature shape changed")
            return
        item_bytes = residual.numel() * torch.tensor([], dtype=torch.bfloat16).element_size()
        minimum_bytes = item_bytes * (self.config.order + 2)
        limit = self.config.max_cache_mb * 1024 * 1024
        if minimum_bytes > limit:
            self.disable(f"minimum history needs {minimum_bytes / 2**20:.1f} MiB, cap is {self.config.max_cache_mb} MiB")
            return
        while self.features and (len(self.features) >= self.config.history_points or self.cache_bytes + item_bytes > limit):
            removed = self.features.pop(0)
            self.times.pop(0)
            self.cache_bytes -= removed.numel() * removed.element_size()
        value = residual.detach().to(dtype=torch.bfloat16).clone()
        self.features.append(value)
        self.times.append(float(target_time))
        self.cache_bytes += item_bytes
        self.consecutive_skips = 0
        self.full_steps += 1

    @staticmethod
    def _chebyshev(values: torch.Tensor, order: int) -> torch.Tensor:
        columns = [torch.ones_like(values)]
        if order:
            columns.append(values)
        for _ in range(2, order + 1):
            columns.append(2 * values * columns[-1] - columns[-2])
        return torch.stack(columns, dim=-1)

    def _spectral_weights(self, target_time: float, device) -> torch.Tensor:
        times = torch.tensor(self.times, dtype=torch.float32, device="cpu")
        low, high = times.min(), times.max()
        span = (high - low).abs().clamp_min(1e-6)
        tau = (times - (low + high) * 0.5) * (2.0 / span)
        target = (torch.tensor(float(target_time), device="cpu") - (low + high) * 0.5) * (2.0 / span)
        design = self._chebyshev(tau, self.config.order)
        target_design = self._chebyshev(target.reshape(1), self.config.order)[0]
        gram = design.T @ design + self.config.ridge * torch.eye(design.shape[1], device="cpu")
        return (target_design @ torch.linalg.solve(gram, design.T)).to(device)

    def _taylor_chunk(self, chunk: list[torch.Tensor], target_time: float) -> torch.Tensor:
        current, previous = chunk[-1], chunk[-2]
        spacing = self.times[-1] - self.times[-2]
        if abs(spacing) <= 1e-12:
            return current
        k = (float(target_time) - self.times[-1]) / spacing
        result = current + k * (current - previous)
        return result

    @torch.no_grad()
    def predict(self, target_time: float) -> torch.Tensor:
        """Forecast the shared-stack feature residual at a flow-map target time."""
        if not self.ready:
            raise RuntimeError("Spectrum needs more real shared-block observations")
        if not self._horizon_ok(target_time):
            raise RuntimeError("Spectrum target is beyond the conservative forecast horizon")
        reference = self.features[-1]
        flat = [feature.reshape(-1) for feature in self.features]
        try:
            weights = self._spectral_weights(target_time, reference.device)
        except RuntimeError as error:
            self.disable(f"spectral solve failed: {error}")
            raise SpectrumRejected(self.disabled_reason) from error
        output = torch.empty(reference.numel(), dtype=reference.dtype, device=reference.device)
        difference_sq = torch.zeros((), dtype=torch.float32, device=reference.device)
        reference_sq = torch.zeros((), dtype=torch.float32, device=reference.device)
        bytes_per_feature = 4 * len(flat)
        chunk_size = max(1, self.config.work_mb * 1024 * 1024 // bytes_per_feature)
        for start in range(0, output.numel(), chunk_size):
            end = min(start + chunk_size, output.numel())
            history = [feature[start:end].float() for feature in flat]
            spectral = sum(weight * value for weight, value in zip(weights, history))
            taylor = self._taylor_chunk(history, target_time)
            predicted = ((1.0 - self.config.blend) * taylor + self.config.blend * spectral).to(reference.dtype)
            output[start:end] = predicted
            difference_sq += (predicted.float() - flat[-1][start:end].float()).square().sum()
            reference_sq += flat[-1][start:end].float().square().sum()
        finite = bool(torch.isfinite(output).all().item())
        relative = float(torch.sqrt(difference_sq / reference_sq.clamp_min(1e-12)).item())
        if not finite or not math.isfinite(relative) or relative > 0.5:
            reason = "nonfinite forecast" if not finite else f"relative feature change {relative:.3f} exceeds 0.5"
            self.disable(reason)
            raise SpectrumRejected(reason)
        return output.reshape(reference.shape)

    def disable(self, reason: str) -> None:
        self.disabled_reason = reason
        self.release(keep_reason=True)

    def release(self, keep_reason: bool = False) -> None:
        reason = self.disabled_reason if keep_reason else None
        self.times.clear()
        self.features.clear()
        self.cache_bytes = 0
        self.consecutive_skips = 0
        self.disabled_reason = reason

    def diagnostics(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "active": self.active,
            "ready": self.ready,
            "disabled_reason": self.disabled_reason,
            "history_points": len(self.features),
            "cache_mb": round(self.cache_bytes / 2**20, 2),
            "full_steps": self.full_steps,
            "forecast_steps": self.forecast_steps,
            "source_commit": SOURCE_COMMIT,
        }
