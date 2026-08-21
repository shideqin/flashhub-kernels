# Usage on another machine

Step-by-step guide for using the 51 flashhub CUDA kernel packages on a new
machine (loading from the CDN or from the local artifact tarball).

## Prerequisites

| Requirement | Value |
|---|---|
| GPU | NVIDIA Thor (SM110) or an aarch64 CUDA machine matching the artifacts |
| OS | Linux aarch64 (the published variants are `aarch64-linux`) |
| CUDA | 13.x (published variant uses `cu130`) |
| Python | >= 3.10 |
| torch | >= 2.0 (2.11 matches the published builds) |

## 1. Install flashhub-kernels

```bash
pip install flashhub-kernels
```

or install from source:

```bash
git clone https://github.com/shideqin/flashhub-kernels.git
pip install -e ./flashhub-kernels
```

Note: the pip package name uses a hyphen (`flashhub-kernels`); the Python
import name uses an underscore (`flashhub_kernels`).

## 2. Load a kernel

### Option A: from the CDN (needs network)

```python
from flashhub_kernels import get_kernel

kernel = get_kernel("flashhub/audio-codebook-primitives", version=1)
```

The CDN org is `flashhub`, so `repo_id` is `flashhub/<pkg>`.

### Option B: from the local artifact tarball (offline)

```bash
wget https://github.com/shideqin/flashhub-kernels/releases/download/packages-20260813/flashhub-kernels-packages-20260813.tar.gz
tar xzf flashhub-kernels-packages-20260813.tar.gz
```

This unpacks to `hf_build_clean/<pkg>/build/<variant>/...`. Load it with:

```python
from flashhub_kernels import get_local_kernel

kernel = get_local_kernel("hf_build_clean/audio-codebook-primitives")
```

## 3. Run a kernel

```python
import torch

kernel = get_kernel("flashhub/audio-codebook-primitives", version=1)

logits = torch.randn((4, 16), dtype=torch.bfloat16, device="cuda")
codebook = torch.randn((4, 16, 8), dtype=torch.bfloat16, device="cuda")
codes, emb = kernel.delayed_codebook_argmax_embed_bf16(
    logits, codebook, delay=2, boc=1
)
```

## 4. Run the official package tests

Clone the source repo and use `tools/run_hf_pkg_tests.py` against the
`hf_build` artifact layout:

```bash
git clone https://github.com/shideqin/FlashRT-HF-kernels.git
python tools/run_hf_pkg_tests.py audio-codebook-primitives
```

## 5. Rebuild from source (only when modifying code)

```bash
# Source: the SM110/Thor portable-SIMT work lives on the fork's feat/* branches,
# not on upstream main.
git clone -b feat/gated-delta-attention-portable-simt \
    https://github.com/shideqin/FlashRT-HF-kernels.git

# Build scripts: tools/build_hf.py + tools/build_common.py + tools/build_hf_all.py
python tools/build_hf_all.py <pkg_dir>
```

`tools/build_common.py` hard-codes machine paths (`/data/venv...`,
`/usr/local/cuda-13.2`, cutlass dirs); adjust them for the new machine before
rebuilding.

## Known issues

- `fa2-seqused-runtime` is missing from the CDN (404 for every layout/version).
  Use Option B (local artifact) for it, or ask the flashhub platform to upload
  it under `flashhub/fa2-seqused-runtime/v1/`.
- Published versions that differ from `build.toml`:
  - `fp8-kv-attention` -> `version=2` (toml says 3)
  - `gated-delta-attention` -> `version=5`
  - `grouped-moe-gemv` -> `version=2`
  - `int4-blackwell` -> `version=2`
  - everything else -> `version=1`
- `MiniMaxAI-msa-blackwell` is published under the lowercase name
  `minimaxai-msa-blackwell`.
