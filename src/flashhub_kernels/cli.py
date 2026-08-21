"""CLI for flashhub-kernels."""
from __future__ import annotations

import argparse
import os

from flashhub_kernels import configure, get_config, get_kernel, has_kernel


def main() -> int:
    parser = argparse.ArgumentParser(prog="flashhub-kernel-load")
    parser.add_argument("repo_id")
    parser.add_argument("--cdn-url", default=None, help="CDN base URL (default: built-in default)")
    parser.add_argument("--prefix", default=None, help="object prefix (default: kernels)")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--has", action="store_true", help="just check availability")
    args = parser.parse_args()

    if args.version is None:
        parser.error("--version is required (version or revision must be specified)")

    cfg = get_config()
    cdn_url = args.cdn_url or os.environ.get("FLASHHUB_CDN_URL") or cfg.cdn_url
    prefix = args.prefix or os.environ.get("FLASHHUB_CDN_PREFIX") or cfg.prefix
    cache_dir = args.cache_dir or os.environ.get("FLASHHUB_CACHE_DIR") or cfg.cache_dir
    configure(cdn_url=cdn_url, prefix=prefix, cache_dir=cache_dir)
    if args.has:
        print(f"{args.repo_id} v{args.version}:", has_kernel(args.repo_id, version=args.version))
        return 0
    k = get_kernel(args.repo_id, version=args.version)
    print(f"loaded {args.repo_id} v{args.version} as {k.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
