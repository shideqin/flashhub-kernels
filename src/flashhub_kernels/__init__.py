"""flashhub-kernels: client for CUDA kernel packages on the flashhub CDN.

Downloads kernel artifacts from the flashhub CDN
(`https://flashhub-cdn.aodianyun.com` by default), matches the build variant
to the local torch/CUDA/arch, and imports the kernel module. The artifact
layout is:

    <prefix>/<org>/<pkg>/v<version>/build/<variant>/
        metadata.json
        <py_pkg>/__init__.py
        <py_pkg>/_ops.py
        <py_pkg>/_<pkg>_cuda_<id>.so

Usage:

    from flashhub_kernels import get_kernel
    kernel = get_kernel("sdq/audio-codebook-primitives", version=1)
"""

from ._config import configure, get_config
from ._api import get_kernel, get_kernel_variants, get_local_kernel, has_kernel
from ._cdn import (
    LoadedKernel,
    RepoInfo,
    VariantAccepted,
    VariantRejected,
    get_loaded_kernels,
)
from .benchmark import Benchmark
from .layer_types import CUDAProperties, Device, Mode, ROCMProperties

__version__ = "0.1.2"

__all__ = [
    "configure",
    "get_config",
    "get_kernel",
    "get_kernel_variants",
    "get_loaded_kernels",
    "get_local_kernel",
    "has_kernel",
    "Benchmark",
    "CUDAProperties",
    "Device",
    "LoadedKernel",
    "Mode",
    "ROCMProperties",
    "RepoInfo",
    "VariantAccepted",
    "VariantRejected",
    "__version__",
]
