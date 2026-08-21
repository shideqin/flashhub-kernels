"""Benchmark base class for kernel benchmarks."""
from __future__ import annotations

from typing import Any


class Benchmark:
    """Base class for kernel benchmarks.

    Subclass this to create a benchmark script with automatic timing,
    verification, and reproducibility support. The kernel is loaded
    automatically from the repo_id passed at construction.

    Example:
        ```python
        class MyBenchmark(Benchmark):
            seed = 42

            def setup(self):
                self.x = torch.randn(128, 1024, device=self.device, dtype=torch.float16)
                self.out = torch.empty(128, 512, device=self.device, dtype=torch.float16)

            def benchmark_silu(self):
                self.kernel.silu_and_mul(self.out, self.x)

            def verify_silu(self):
                return torch.nn.functional.silu(self.x[..., :512]) * self.x[..., 512:]
        ```
    """

    seed: int | None = None  # Optional: seed for reproducibility
    device: str = "cpu"  # Set automatically by runner

    def __init__(self, kernel: Any = None) -> None:
        self.kernel: Any = kernel
        self.out: Any = None  # Output tensor, set by setup methods

    def setup(self) -> None:
        """Override to set up tensors as instance attributes."""
        pass
