"""flashhub CDN configuration (env-driven, override via configure())."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# Default flashhub CDN (CDN domain fronting the Aliyun OSS bucket; object
# path layout `<prefix>/<org>/<pkg>/...`). Override via FLASHHUB_CDN_URL or
# configure().
DEFAULT_CDN_URL = "https://flashhub-cdn.aodianyun.com"


@dataclass(frozen=True)
class Config:
    cdn_url: str = os.environ.get("FLASHHUB_CDN_URL", DEFAULT_CDN_URL)
    prefix: str = os.environ.get("FLASHHUB_CDN_PREFIX", "kernels")
    cache_dir: str = os.environ.get("FLASHHUB_CACHE_DIR", "/tmp/flashhub_kernel_cache")


_cfg = Config()


def get_config() -> Config:
    return _cfg


def configure(*, cdn_url: str | None = None, prefix: str | None = None, cache_dir: str | Path | None = None) -> None:
    """Override the flashhub CDN settings for this process.

    Defaults to the domestic CDN (``https://flashhub-cdn.aodianyun.com``);
    override with the international CDN domain when needed. Falls back to env
    vars `FLASHHUB_CDN_URL`, `FLASHHUB_CDN_PREFIX`, `FLASHHUB_CACHE_DIR` when
    not given.
    """
    global _cfg
    _cfg = replace(
        _cfg,
        cdn_url=cdn_url if cdn_url is not None else _cfg.cdn_url,
        prefix=prefix if prefix is not None else _cfg.prefix,
        cache_dir=str(cache_dir) if cache_dir is not None else _cfg.cache_dir,
    )
