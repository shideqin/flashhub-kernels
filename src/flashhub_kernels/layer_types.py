"""Kernelize/layer support types.

Plain data types with no platform dependency: device capability constraints
and kernelize mode flags used for kernel selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import Any


class Mode(Flag):
    """Kernelize mode: selects kernels for the given mode in a mapping."""

    INFERENCE = auto()
    TRAINING = auto()
    TORCH_COMPILE = auto()
    FALLBACK = auto()


@dataclass(frozen=True)
class CUDAProperties:
    """CUDA compute-capability constraints for kernel selection.

    ``min_capability`` / ``max_capability`` are ints like 75 (compute
    capability 7.5) or 90 (9.0).
    """

    min_capability: int
    max_capability: int


@dataclass(frozen=True)
class ROCMProperties:
    """ROCm compute-capability constraints for kernel selection."""

    min_capability: int
    max_capability: int


@dataclass(frozen=True)
class Device:
    """A compute device with optional capability properties.

    ``type`` is e.g. "cuda", "mps", "npu", "rocm", "xpu".
    """

    type: str
    properties: CUDAProperties | ROCMProperties | None = None

    def __str__(self) -> str:
        return self.type
