import os
import glob
import shutil

REPO = "/data/FlashRT-HF-kernels"
OUT = "/data/build"
OUT_HF = "/data/hf_build"
TMP = "/data/build_tmp"
VENV_PY = "/data/venv/bin/python"
CUDA_HOME = "/usr/local/cuda-13.2"
REGISTRATION_DIR = "/data"
UNIQUE_ID = "b1c9a42"

CUTLASS_DIRS = {
    "cutlass_2_10": "/data/third_party/cutlass-2.10.0",
    "cutlass_3_5": "/data/third_party/cutlass-3.5.1",
    "cutlass_3_6": "/data/third_party/cutlass-3.6.0",
    "cutlass_3_8": "/data/third_party/cutlass-3.8.0",
    "cutlass_3_9": "/data/third_party/cutlass-3.9.2",
    "cutlass_4_0": "/data/third_party/cutlass-4.0.0",
    "cutlass_4_4": "/data/third_party/cutlass-4.4.0",
    "cutlass_4_5": "/data/third_party/cutlass-4.5.2",
}

# Extra include dirs (sibling package csrc) for packages with missing
# vendored headers (e.g. fp8-gemm references cutlass/util/packed_stride.hpp
# which is only vendored by fp4-gemm).
EXTRA_INCLUDES = {
    "fp8-gemm": ["/data/FlashRT-HF-kernels/fp4-gemm/csrc"],
    "fp8-cross-attention-blackwell": ["/data/FlashRT-HF-kernels/fp4-gemm/csrc"],
}

COMPILABLE_EXTS = (".cu", ".cpp", ".cc", ".cxx", ".c", ".c++")

# Normalize e.g. "11.0" -> "110", "12.0a" -> "120a", "8.0" -> "80"
def normalize_cap(cap):
    cap = cap.strip()
    suffix = ""
    if cap.endswith("a"):
        suffix = "a"
        cap = cap[:-1]
    if "." in cap:
        major, minor = cap.split(".")[:2]
        cap = major + minor
    return cap + suffix

# Map a CUDA capability string to nvcc arch tokens.
def cap_to_gencode(cap):
    a = normalize_cap(cap)
    return f"-gencode=arch=compute_{a},code=sm_{a}"

def collect_compile_sources(pkg_dir, torch_cfg, kernels):
    """Return list of absolute paths of compilable sources for a package."""
    sources = []
    for s in torch_cfg.get("src", []):
        p = os.path.join(pkg_dir, s)
        if os.path.splitext(p)[1] in COMPILABLE_EXTS:
            sources.append(p)
    for kname, kcfg in kernels.items():
        if kcfg.get("backend") != "cuda":
            continue
        for s in kcfg.get("src", []):
            p = os.path.join(pkg_dir, s)
            if os.path.splitext(p)[1] in COMPILABLE_EXTS:
                sources.append(p)
    # dedupe while preserving order
    seen = set()
    out = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def collect_includes(pkg_dir, torch_cfg, kernels):
    dirs = []
    for inc in torch_cfg.get("include", []):
        dirs.append(os.path.join(pkg_dir, inc))
    for kname, kcfg in kernels.items():
        if kcfg.get("backend") != "cuda":
            continue
        for inc in kcfg.get("include", []):
            dirs.append(os.path.join(pkg_dir, inc))
    seen = set()
    out = []
    for d in dirs:
        d = os.path.normpath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out

def collect_cuda_caps(kernels):
    caps = []
    for kname, kcfg in kernels.items():
        if kcfg.get("backend") != "cuda":
            continue
        for c in kcfg.get("cuda-capabilities", []) or []:
            if c not in caps:
                caps.append(c)
    return caps

def collect_cuda_flags(kernels):
    flags = []
    for kname, kcfg in kernels.items():
        if kcfg.get("backend") != "cuda":
            continue
        for f in kcfg.get("cuda-flags", []) or []:
            if f not in flags:
                flags.append(f)
    return flags

def collect_cutlass_deps(kernels):
    deps = []
    for kname, kcfg in kernels.items():
        for d in kcfg.get("depends", []) or []:
            if d.startswith("cutlass") and d not in deps:
                deps.append(d)
    return deps

def python_pkg_dir(pkg_dir):
    ext = os.path.join(pkg_dir, "torch-ext")
    if not os.path.isdir(ext):
        return None
    for d in sorted(os.listdir(ext)):
        if os.path.isdir(os.path.join(ext, d)) and os.path.exists(os.path.join(ext, d, "__init__.py")):
            return d
    return None
