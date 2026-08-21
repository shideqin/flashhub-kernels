# flashhub-kernels

Load CUDA kernel packages published on the flashhub CDN
(`https://flashhub-cdn.aodianyun.com` by default).

Kernel packages on the flashhub CDN are made to be:

- **Portable**: a kernel can be loaded from a local path or downloaded from
  the CDN into a cache, without being installed into `PYTHONPATH`.
- **Variant-matched**: each package publishes one build per environment
  (torch version, CUDA version, CPU architecture); the client picks the
  variant that fits the local runtime and downloads only that build.
- **Verified**: every artifact is checked against the sha256 digests in its
  `metadata.json` before loading.

## 🚀 Quick Start

Install the `flashhub-kernels` package (requires `torch>=2.0`):

```bash
pip install flashhub-kernels
```

Load a kernel from the flashhub CDN:

```python
import torch

from flashhub_kernels import get_kernel

kernel = get_kernel("sdq/audio-codebook-primitives", version=1)

logits = torch.randn((4, 16), dtype=torch.bfloat16, device="cuda")
codebook = torch.randn((4, 16, 8), dtype=torch.bfloat16, device="cuda")
codes, emb = kernel.delayed_codebook_argmax_embed_bf16(
    logits, codebook, delay=2, boc=1
)
```

## API

- `get_kernel(repo_id, revision=None, version=None)` — download from the CDN
  into the local cache (variant-matched to torch/CUDA/arch), import and return
  the kernel module. `version` selects `v<version>`; `revision` accepts
  `"v1"`/`"1"`. Exactly one of `revision`/`version` must be given.
- `get_local_kernel(repo_path)` — import a kernel from a local repo dir.
- `has_kernel(repo_id, version=None)` — whether the variant exists on the CDN.
- `get_kernel_variants(repo_id, version=None)` — resolve every published
  variant against the current system (`VariantAccepted`/`VariantRejected`).
- `get_loaded_kernels()` — snapshot of kernels loaded in this process.
- `configure(cdn_url=None, prefix=None, cache_dir=None)` — override CDN
  settings for the process (defaults read from `FLASHHUB_CDN_URL`,
  `FLASHHUB_CDN_PREFIX`, `FLASHHUB_CACHE_DIR`).

## CDN layout

`repo_id` is the full `<org>/<pkg>` (e.g. `sdq/audio-codebook-primitives`); the
org is part of the object path.

```
<prefix>/<org>/<pkg>/v<version>/build/<variant>/
    metadata.json          # {"digest": {"files": {rel: b64-sha256}}}
    <py_pkg>/__init__.py
    <py_pkg>/_ops.py
    <py_pkg>/_<pkg>_cuda_<id>.so
<prefix>/<org>/<pkg>/v<version>/variants.json   # optional: published variant names
```

`variant` is derived from the runtime: `torch<major><minor>-cxx11-cu<cuda>-<arch>-linux`.
