"""CDN download + variant matching + import (self-contained)."""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ._config import Config, get_config

UA = "flashhub-kernels/0.1"

# torch211-cxx11-cu130-aarch64-linux
# torch: single-digit major + two-digit minor (211 -> 2.11).
# Variant name grammar:
#   torch<M><m>(-cxx11|-cxx98)?-<backend>-<platform>-<os>      arch kernel
#   torch-stable-abi<M><m>-<backend>-<platform>-<os>           stable ABI arch kernel
#   torch-<backend>                                            noarch kernel
# e.g. torch211-cxx11-cu130-aarch64-linux, torch211-cu130-aarch64-linux,
#      torch-stable-abi211-cu130-aarch64-linux, torch-cpu.
_TORCH_RE = re.compile(r"^torch(\d)(\d+)(?:-(cxx11|cxx98))?$")
_STABLE_ABI_RE = re.compile(r"^torch-stable-abi(\d)(\d+)$")
_ARCH_RE = re.compile(r"^(?P<backend>cu(?P<cmaj>\d{1,2})(?P<cmin>\d)|cpu|rocm\d+|metal)-(?P<platform>aarch64|x86_64|amd64|arm64)-(?P<os>\w+)$")


@dataclass(frozen=True)
class _Variant:
    name: str
    kind: str  # "torch" | "stable-abi" | "noarch"
    torch_major: int | None
    torch_minor: int | None
    cxx11_abi: bool | None  # None = no ABI tag
    cuda_major: int | None
    cuda_minor: int | None
    backend_name: str  # "cuda" / "cpu" / "rocm" / "metal" (noarch: backend name)
    platform: str | None
    os: str | None


@dataclass(frozen=True)
class VariantAccepted:
    """A build variant that is compatible with the current system.

    ``variant`` is the variant name string; ``variant_str`` is an alias.
    """

    variant: str

    @property
    def variant_str(self) -> str:
        return self.variant


@dataclass(frozen=True)
class VariantRejected:
    """A build variant that is incompatible, with the rejection reason."""

    variant: str
    reason: str

    @property
    def variant_str(self) -> str:
        return self.variant


Decision = VariantAccepted | VariantRejected


@dataclass(frozen=True)
class RepoInfo:
    """Origin of a loaded kernel."""

    repo_id: str
    revision: str


@dataclass(frozen=True)
class LoadedKernel:
    """Information about a loaded kernel.

    ``metadata`` is the parsed ``metadata.json`` dict; ``module`` is the
    imported kernel module; ``repo_info`` records the CDN origin when loaded
    via ``get_kernel``.
    """

    metadata: dict
    module: object
    repo_info: RepoInfo | None


_loaded_kernels: dict[Path, LoadedKernel] = {}


def _register_loaded(variant_local: Path, module, repo_info: RepoInfo | None = None) -> None:
    """Record a loaded kernel variant in the process registry."""
    meta_path = variant_local / "metadata.json"
    metadata = {}
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            metadata = {}
    _loaded_kernels[variant_local] = LoadedKernel(
        metadata=metadata, module=module, repo_info=repo_info
    )


def get_loaded_kernels() -> list[LoadedKernel]:
    """Snapshot of every kernel loaded in this process."""
    return list(_loaded_kernels.values())


def _backend() -> dict:
    """Detect the system compute backend (neuron/cuda/rocm/metal/xpu/cann/cpu)."""
    import torch

    if hasattr(torch, "neuron"):
        return {"name": "neuron", "major": None, "minor": None, "variant_str": "neuron"}
    if torch.version.cuda is not None:
        major, minor = (int(p) for p in torch.version.cuda.split(".")[:2])
        return {"name": "cuda", "major": major, "minor": minor, "variant_str": f"cu{major}{minor}"}
    if torch.version.hip is not None:
        hip = torch.version.hip.split("-")[0]
        major, minor = (int(p) for p in hip.split(".")[:2])
        return {"name": "rocm", "major": major, "minor": minor, "variant_str": f"rocm{major}{minor}"}
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return {"name": "metal", "major": None, "minor": None, "variant_str": "metal"}
    xpu = getattr(torch.version, "xpu", None)
    if xpu is not None:
        return {"name": "xpu", "major": None, "minor": None, "variant_str": "xpu"}
    try:
        if torch._C._get_privateuse1_backend_name() == "npu":
            return {"name": "cann", "major": None, "minor": None, "variant_str": "cann"}
    except Exception:  # pragma: no cover - exotic builds
        pass
    return {"name": "cpu", "major": None, "minor": None, "variant_str": "cpu"}


def _system() -> dict:
    """System parameters used for variant matching."""
    import torch

    torch_ver = torch.__version__.split("+")[0]
    t_major, t_minor = (int(p) for p in torch_ver.split(".")[:2])
    arch = "aarch64" if os.uname().machine in ("aarch64", "arm64") else "x86_64"
    if os.uname().machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = "x86_64" if os.uname().machine in ("x86_64", "amd64") else os.uname().machine
    try:
        cxx11 = bool(torch.compiled_with_cxx11_abi()) if sys.platform.startswith("linux") else None
    except Exception:  # pragma: no cover - exotic builds
        cxx11 = None
    be = _backend()
    cuda_major = cuda_minor = None
    if be["name"] == "cuda":
        cuda_major, cuda_minor = be["major"], be["minor"]
    return {
        "torch_major": t_major, "torch_minor": t_minor,
        "cuda_major": cuda_major, "cuda_minor": cuda_minor,
        "arch": arch, "os": sys.platform, "cxx11_abi": cxx11,
        "backend": be,
    }


def _parse_variant(name: str) -> _Variant | None:
    """Parse a variant name; None when unrecognized."""
    s = name.strip()

    if s.startswith("torch-stable-abi"):
        # torch-stable-abi<M><m>-<backend>-<platform>-<os>
        m = re.match(r"^(torch-stable-abi\d+\d+)-", s)
        if m is None:
            return None
        head = m.group(1)
        tail = s[len(head) + 1:]
        am = _STABLE_ABI_RE.match(head)
        if am is None:
            return None
        arch = _ARCH_RE.match(tail)
        if arch is None:
            return None
        return _Variant(
            name=s, kind="stable-abi",
            torch_major=int(am.group(1)), torch_minor=int(am.group(2)),
            cxx11_abi=None,
            cuda_major=int(arch.group("cmaj")) if arch.group("cmaj") else None,
            cuda_minor=int(arch.group("cmin")) if arch.group("cmin") else None,
            backend_name="cuda" if arch.group("backend").startswith("cu") else arch.group("backend"),
            platform=arch.group("platform"), os=arch.group("os"),
        )

    # noarch: torch-cpu, torch-neuron, ...
    if s.startswith("torch-") and "-cu" not in s:
        backend = s[len("torch-"):]
        if backend and "/" not in backend:
            return _Variant(
                name=s, kind="noarch",
                torch_major=None, torch_minor=None, cxx11_abi=None,
                cuda_major=None, cuda_minor=None,
                backend_name=backend, platform=None, os=None,
            )

    if not s.startswith("torch"):
        return None
    parts = s.split("-")
    # torch<M><m>(-cxx11|-cxx98)?-<backend>-<platform>-<os>
    if len(parts) >= 2 and parts[1] in ("cxx11", "cxx98"):
        head, tail = "-".join(parts[:2]), "-".join(parts[2:])
    else:
        head, tail = parts[0], "-".join(parts[1:])
    if not tail:
        return None
    m = _TORCH_RE.match(head)
    if m is None:
        return None
    abi_str = m.group(3)
    arch = _ARCH_RE.match(tail)
    if arch is None:
        return None
    return _Variant(
        name=s, kind="torch",
        torch_major=int(m.group(1)), torch_minor=int(m.group(2)),
        cxx11_abi=None if abi_str is None else (abi_str == "cxx11"),
        cuda_major=int(arch.group("cmaj")) if arch.group("cmaj") else None,
        cuda_minor=int(arch.group("cmin")) if arch.group("cmin") else None,
        backend_name="cuda" if arch.group("backend").startswith("cu") else arch.group("backend"),
        platform=arch.group("platform"), os=arch.group("os"),
    )


def _variant_name() -> str:
    """Derive the build variant dir from the runtime (torch/backend/arch)."""
    s = _system()
    be = s["backend"]
    if be["name"] == "cuda":
        backend_str = f"cu{be['major']}{be['minor']}"
    else:
        backend_str = be["variant_str"]
    return f"torch{s['torch_major']}{s['torch_minor']:02d}-cxx11-{backend_str}-{s['arch']}-linux"


def _select_backend(backend: str | None, sys_params: dict) -> dict:
    """Resolve the requested backend; None = system-detected.

    Only the system backend and "cpu" are valid; anything else raises.
    """
    if backend is None:
        return sys_params["backend"]
    supported = {"cpu": {"name": "cpu", "major": None, "minor": None, "variant_str": "cpu"},
                 sys_params["backend"]["name"]: sys_params["backend"]}
    if backend not in supported:
        raise ValueError(
            f"Invalid backend '{backend}', system supported backends: {', '.join(sorted(supported))}"
        )
    return supported[backend]


def _match_variants(names: list[str], backend: str | None = None) -> tuple[str | None, list[str]]:
    """Return the best matching variant name and a decision trace.

    Variant resolution rules:
      - arch kernels: platform & os must match; torch version must equal
        (stable-abi: ABI version <= system torch); backend must match
        (CUDA: major equal, minor <= system; other backends: exact);
        ABI tag must match when present;
      - noarch kernels: backend name matches the system backend (or universal);
      - preference order: stable-abi (newest ABI first) > torch (untagged
        preferred) > noarch; within a framework, highest CUDA minor wins.
    """
    sys_params = _system()
    selected = _select_backend(backend, sys_params)
    trace: list[str] = []
    accepted: list[_Variant] = []
    for name in names:
        v = _parse_variant(name)
        if v is None:
            trace.append(f"{name}: unrecognized variant name format")
            continue

        if v.kind == "noarch":
            noarch_name = "npu" if selected["name"] == "cann" else selected["name"]
            if v.backend_name != noarch_name and v.backend_name != "universal":
                trace.append(
                    f"{name}: backend ({v.backend_name}) does not match system backend ({noarch_name}) and is not universal"
                )
                continue
            accepted.append(v)
            trace.append(f"{name}: compatible")
            continue

        if v.platform != sys_params["arch"]:
            trace.append(
                f"{name}: CPU ({v.platform}) does not match system CPU ({sys_params['arch']})"
            )
            continue
        if v.os != "linux":
            trace.append(f"{name}: OS ({v.os}) does not match system OS (linux)")
            continue
        # Backend match: CUDA allows older minor; other backends must be exact.
        if selected["name"] == "cuda":
            if v.backend_name != "cuda":
                trace.append(
                    f"{name}: backend ({v.backend_name}) does not match selected backend (cuda)"
                )
                continue
            if v.cuda_major != selected["major"]:
                trace.append(
                    f"{name}: CUDA major ({v.cuda_major}) does not match system CUDA major ({selected['major']})"
                )
                continue
            if v.cuda_minor > selected["minor"]:
                trace.append(
                    f"{name}: CUDA version (cu{v.cuda_major}.{v.cuda_minor}) is newer than "
                    f"system CUDA (cu{selected['major']}.{selected['minor']})"
                )
                continue
        else:
            if v.backend_name != selected["name"]:
                trace.append(
                    f"{name}: backend ({v.backend_name}) does not match selected backend ({selected['name']})"
                )
                continue
        if v.kind == "torch":
            if (v.torch_major, v.torch_minor) != (sys_params["torch_major"], sys_params["torch_minor"]):
                trace.append(
                    f"{name}: Torch version (torch{v.torch_major}.{v.torch_minor}) "
                    f"does not match environment Torch version (torch{sys_params['torch_major']}.{sys_params['torch_minor']})"
                )
                continue
            if v.cxx11_abi is not None and sys_params["cxx11_abi"] is not None and v.cxx11_abi != sys_params["cxx11_abi"]:
                trace.append(
                    f"{name}: Torch CXX11 ABI ({'cxx11' if v.cxx11_abi else 'cxx98'}) "
                    f"does not match environment ABI ({'cxx11' if sys_params['cxx11_abi'] else 'cxx98'})"
                )
                continue
        else:  # stable-abi: ABI version must be <= system torch
            if (v.torch_major, v.torch_minor) > (sys_params["torch_major"], sys_params["torch_minor"]):
                trace.append(
                    f"{name}: Torch stable ABI version (torch{v.torch_major}.{v.torch_minor}) is too new "
                    f"for environment Torch version (torch{sys_params['torch_major']}.{sys_params['torch_minor']})"
                )
                continue
        accepted.append(v)
        trace.append(f"{name}: compatible")

    def sort_key(v: _Variant) -> tuple:
        # (accepted order, framework order, abi order, cuda order, name)
        if v.kind == "stable-abi":
            fw = (0, -v.torch_major, -v.torch_minor)
        elif v.kind == "torch":
            fw = (1, 0, 1 if v.cxx11_abi is not None else 0)
        else:  # noarch
            fw = (2, 0, 0)
        return (fw[0], fw[1], fw[2], -(v.cuda_minor or 0), v.name)

    accepted.sort(key=sort_key)
    best = accepted[0].name if accepted else None
    if best is not None:
        trace = [f"{name}: compatible (preferred)" if name == best else entry
                 for name, entry in zip(names, trace)]
    return best, trace


def resolve_variants(names: list[str], backend: str | None = None) -> list[Decision]:
    """Resolve every variant against the current system.

    Returns one ``VariantAccepted`` or ``VariantRejected`` per variant name,
    sorted with compatible variants first (most preferred leading).
    """
    best, trace = _match_variants(names, backend=backend)
    decisions: list[Decision] = []
    for name, entry in zip(names, trace):
        if ": compatible" in entry:
            decisions.append(VariantAccepted(variant=name))
        else:
            reason = entry.split(": ", 1)[1] if ": " in entry else entry
            decisions.append(VariantRejected(variant=name, reason=reason))
    # Order: accepted first, best leading, then by name.
    def sort_key(d: Decision):
        return (0 if isinstance(d, VariantAccepted) else 1,
                d.variant != best,
                d.variant)
    decisions.sort(key=sort_key)
    return decisions


def _fetch_variants(cfg: Config, repo_id: str, version: int) -> list[str] | None:
    """Fetch the published variant list for a version; None when unavailable.

    The platform writes ``<prefix>/<org>/<pkg>/v<version>/variants.json`` at
    publish time (static file, served by the CDN like any other artifact).
    """
    base = cfg.cdn_url.rstrip("/")
    url = f"{base}/{cfg.prefix}/{repo_id}/v{version}/variants.json"
    try:
        data = json.loads(_http_get(url))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    if not isinstance(data, list):
        raise ValueError(f"variants.json at {url} must be a JSON list of variant names")
    return [str(x) for x in data]


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _resolve_version(cfg: Config, repo_id: str, version: int | None) -> int:
    """Require an explicit version (version or revision must be
    given; there is no "latest" resolution)."""
    if version is not None:
        return version
    raise ValueError(
        "A kernel version or revision must be specified. "
        "Use `version=<major>` for a stable kernel API version or "
        "`revision=<vN>` for an explicit version."
    )


def _download_variant(cfg: Config, repo_id: str, version: int, *, variant: str | None = None, refresh: bool = False, verify: bool = True) -> Path:
    """Download `<repo_id>/v<version>/build/<variant>/` into the cache; return the repo path."""
    variant = variant or _variant_name()
    local = Path(cfg.cache_dir) / repo_id / f"v{version}"
    variant_local = local / "build" / variant
    meta_local = variant_local / "metadata.json"
    if refresh or not meta_local.exists():
        base = f"{cfg.cdn_url.rstrip('/')}/{cfg.prefix}/{repo_id}/v{version}/build/{variant}"
        meta = json.loads(_http_get(f"{base}/metadata.json"))
        files = meta.get("digest", {}).get("files", {})
        if not files:
            raise ValueError(f"metadata.json at {base}/metadata.json has no digest.files")
        variant_local.mkdir(parents=True, exist_ok=True)
        meta_local.write_bytes(json.dumps(meta, indent=2).encode() + b"\n")
        for rel, b64digest in files.items():
            dest = variant_local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = _http_get(f"{base}/{rel}")
            if verify:
                want = base64.b64decode(b64digest)
                got = hashlib.sha256(data).digest()
                if got != want:
                    raise ValueError(f"sha256 mismatch for {rel}")
            dest.write_bytes(data)
    return local


def _import_pkg(local: Path, only: str | None = None, repo_info: RepoInfo | None = None):
    """Import the python package of a downloaded variant dir.

    Supports both kernel layouts:
      - `<variant>/<py_pkg>/__init__.py` (package subdir);
      - `<variant>/__init__.py` (module directly in the variant dir).
    ``only`` restricts to a single variant dir name (post-variant-match).
    Loaded kernels are recorded in the process registry.
    """
    build = local / "build"
    for variant in sorted(build.iterdir()) if build.is_dir() else []:
        if not variant.is_dir():
            continue
        if only is not None and variant.name != only:
            continue
        # Layout 1: package subdir with __init__.py (our hf_build layout).
        pkg_dirs = [d for d in variant.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
        if pkg_dirs:
            sys.path.insert(0, str(variant))
            module = importlib.import_module(pkg_dirs[0].name)
            _register_loaded(variant, module, repo_info)
            return module
        # Layout 2: __init__.py directly in the variant dir.
        if (variant / "__init__.py").is_file():
            spec = importlib.util.spec_from_file_location(variant.name, variant / "__init__.py")
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[variant.name] = module
            spec.loader.exec_module(module)
            _register_loaded(variant, module, repo_info)
            return module
    raise FileNotFoundError(f"no importable kernel variant under {local}")


def load_from_cdn(
    repo_id: str,
    *,
    version: int | None = None,
    revision: str | None = None,
    backend: str | None = None,
    refresh: bool = False,
    verify: bool = True,
    cfg: Config | None = None,
):
    cfg = cfg or get_config()
    if backend is not None:
        # Validate the backend up front, even on the
        # exact-variant fallback path.
        _select_backend(backend, _system())
    if revision is not None:
        # Explicit revision takes precedence; resolve straight to the vN path.
        version = _revision_to_version(revision, 1)
    else:
        version = _resolve_version(cfg, repo_id, version)
    local = _download_variant(cfg, repo_id, version, refresh=refresh, verify=verify)
    repo_info = RepoInfo(repo_id=repo_id, revision=f"v{version}")
    return _import_pkg(local, repo_info=repo_info)


def load_variant_from_cdn(
    repo_id: str,
    *,
    version: int | None = None,
    revision: str | None = None,
    backend: str | None = None,
    refresh: bool = False,
    verify: bool = True,
    cfg: Config | None = None,
):
    """Load a kernel, matching against the published variant list.

    Fetches ``v<version>/variants.json`` (a JSON list of variant names) and
    selects the best variant for this system with ``_match_variants``. When the
    manifest is absent (single-variant CDN / older packages), falls back to the
    exact derived variant name.
    """
    cfg = cfg or get_config()
    if revision is not None:
        version = _revision_to_version(revision, 1)
    else:
        version = _resolve_version(cfg, repo_id, version)
    names = _fetch_variants(cfg, repo_id, version)
    if names is None:
        return load_from_cdn(repo_id, version=version, backend=backend, cfg=cfg)
    best, trace = _match_variants(names, backend=backend)
    if best is None:
        raise ValueError(
            f"no compatible variant for {repo_id} v{version}:\n" + "\n".join(trace)
        )
    local = _download_variant(cfg, repo_id, version, variant=best, refresh=refresh, verify=verify)
    repo_info = RepoInfo(repo_id=repo_id, revision=f"v{version}")
    return _import_pkg(local, repo_info=repo_info)


def _revision_to_version(revision: str, fallback: int) -> int:
    """`v1` / `1` -> 1. Explicit commits are not resolvable via CDN paths."""
    r = revision.strip()
    if r.startswith("v"):
        r = r[1:]
    try:
        return int(r)
    except ValueError:
        return fallback
