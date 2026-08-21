"""Public API for loading kernel packages from the flashhub CDN."""
from __future__ import annotations

from pathlib import Path

from ._cdn import (
    VariantAccepted,
    VariantRejected,
    load_variant_from_cdn,
    resolve_variants,
)
from ._config import get_config


def get_kernel(
    repo_id: str,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    user_agent: str | dict | None = None,
    trust_remote_code: bool | list[str] = False,
):
    """Load a kernel from the flashhub CDN.

    ``version`` selects the ``v<version>`` artifact; ``revision`` accepts
    ``"v1"``/``"1"``. One of ``version`` or ``revision`` must be specified.
    Downloads to the local cache and imports the package module
    (variant-matched to torch/CUDA/arch).

    Example:
        ```python
        from flashhub_kernels import get_kernel
        kernel = get_kernel("sdq/audio-codebook-primitives", version=1)
        codes, emb = kernel.delayed_codebook_argmax_embed_bf16(
            logits, codebook, delay=2, boc=1)
        ```
    """
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")
    return load_variant_from_cdn(repo_id, revision=revision, version=version, backend=backend)


def get_kernel_variants(
    repo_id: str,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
) -> list:
    """Resolve all build variants of a kernel against the current environment.

    Returns one ``VariantAccepted`` or ``VariantRejected`` per published
    variant, sorted with compatible variants first (most preferred leading).
    When no variants.json manifest exists (single-variant CDN), a single
    decision for the exact derived variant is returned.
    """
    from ._cdn import _fetch_variants, _resolve_version, _variant_name

    cfg = get_config()
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")
    if revision is not None:
        from ._cdn import _revision_to_version

        version = _revision_to_version(revision, 1)
    else:
        version = _resolve_version(cfg, repo_id, version)
    names = _fetch_variants(cfg, repo_id, version)
    if names is None:
        return [VariantAccepted(variant=_variant_name())]
    return resolve_variants(names, backend=backend)


def get_local_kernel(repo_path, backend: str | None = None):
    """Import a kernel from a local kernel repository path (kernels-compatible)."""
    return _import_local(Path(repo_path), backend=backend)


def has_kernel(
    repo_id: str,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
) -> bool:
    """Whether a kernel version exists on the flashhub CDN for this environment."""
    from ._cdn import _fetch_variants, _match_variants, _resolve_version, _variant_name, _http_get

    cfg = get_config()
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")
    if revision is not None:
        from ._cdn import _revision_to_version

        version = _revision_to_version(revision, 1)
    else:
        version = _resolve_version(cfg, repo_id, version)
    names = _fetch_variants(cfg, repo_id, version)
    if names is not None:
        best, _ = _match_variants(names, backend=backend)
        return best is not None
    url = f"{cfg.cdn_url.rstrip('/')}/{cfg.prefix}/{repo_id}/v{version}/build/{_variant_name()}/metadata.json"
    try:
        _http_get(url)
        return True
    except Exception:
        return False


def _import_local(local: Path, backend: str | None = None):
    """Import the python package of a local kernel repo dir (with build/variant)."""
    from ._cdn import _import_pkg, _match_variants

    if (local / "build").is_dir():
        names = sorted(d.name for d in (local / "build").iterdir() if d.is_dir())
        if names:
            best, _ = _match_variants(names, backend=backend)
            if best is None:
                raise FileNotFoundError(
                    f"no compatible build variant under {local / 'build'}"
                )
            return _import_pkg(local, only=best)
        return _import_pkg(local)
    pkg_dirs = [d for d in local.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    if not pkg_dirs:
        raise FileNotFoundError(f"no kernel module found at: {local}")
    import sys

    sys.path.insert(0, str(local))
    import importlib

    return importlib.import_module(pkg_dirs[0].name)
