"""Build one FlashRT package into the HF Kernel Hub artifact layout.

Produces, for each package:

  /data/hf_build/<pkg>/build/<variant>/
    metadata.json
    <py_pkg>/
      __init__.py
      _ops.py
      _<py_pkg>_cuda_<id>.so

Compilation follows the HF kernel-builder structure: every ``[kernel.*]``
section is treated as an independent component. Each compilable source is
built with only the gencode targets declared by the sections that list it
(plus ``11.0a`` when it does not already cover the Thor target), and each
section's ``cuda-flags``/defines apply only to the sources of that section.
All objects are then linked into a single extension ``_<py_pkg>_cuda_<id>.so``
registered under one ``torch.ops`` namespace.

This per-source gencode gating is required so that arch-specific sources
(e.g. fp8-gemm SM120 kernels using ``f8f6f4``) are never compiled for older
architectures they do not support.

Compilation targets the current Thor hardware: torch from the venv,
CUDA 13.2, SM110 (`11.0a`). Stable-ABI (.abi3) is intentionally NOT used
because the installed torch does not ship torch/stable.h.

Usage: python build_hf.py <pkg_dir>
"""
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import sysconfig
import traceback

sys.path.insert(0, "/data")
import build_common as bc

import torch
from torch.utils import cpp_extension as ce

ce._get_cuda_arch_flags = lambda *a: []

FORCED_CAP = "11.0a"


def normalize_arch_for_metadata(cap: str) -> str:
    """`11.0a` -> `11.0` (kernel-builder drops the arch-specific suffix)."""
    cap = cap.strip()
    if cap.endswith("a"):
        cap = cap[:-1]
    if "." not in cap and len(cap) >= 2:
        cap = f"{cap[:-1]}.{cap[-1]}"
    return cap


def package_unique_id(pkg_dir, py_pkg, sources):
    """Deterministic per-package id from build.toml + package sources."""
    h = hashlib.sha256()
    btoml = os.path.join(pkg_dir, "build.toml")
    if os.path.exists(btoml):
        h.update(open(btoml, "rb").read())
    ext = os.path.join(pkg_dir, "torch-ext", py_pkg)
    if os.path.isdir(ext):
        for root, _dirs, files in sorted(os.walk(ext)):
            for f in sorted(files):
                if f.endswith((".pyc", ".so")):
                    continue
                p = os.path.join(root, f)
                h.update(os.path.relpath(p, pkg_dir).encode())
                h.update(open(p, "rb").read())
    for s in sorted(sources):
        p = os.path.join(pkg_dir, s)
        if os.path.exists(p):
            h.update(s.encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()[:7]


def variant_name():
    torch_ver = torch.__version__.split("+")[0]
    major, minor = torch_ver.split(".")[:2]
    cuda_major, cuda_minor = torch.version.cuda.split(".")[:2]
    arch = "aarch64" if os.uname().machine in ("aarch64", "arm64") else "x86_64"
    return (f"torch{major}{minor}-cxx11-cu{cuda_major}{cuda_minor}"
            f"-{arch}-linux")


def digest_files(variant_dir):
    """sha256+base64 digest over artifact files (excludes metadata.json)."""
    files = {}
    for root, dirs, names in sorted(os.walk(variant_dir)):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(names):
            if name.endswith(".pyc"):
                continue
            p = os.path.join(root, name)
            rel = os.path.relpath(p, variant_dir).replace(os.sep, "/")
            if rel == "metadata.json":
                continue
            data = open(p, "rb").read()
            files[rel] = base64.b64encode(hashlib.sha256(data).digest()).decode()
    return files


def write_metadata(variant_dir, name, ops_id, version, license, archs):
    meta = {
        "name": name,
        "id": ops_id,
        "version": version,
        "license": license,
        "python-depends": [],
        "backend": {"type": "cuda", "archs": archs},
        "digest": {"algorithm": "sha256", "files": digest_files(variant_dir)},
    }
    with open(os.path.join(variant_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def per_source_caps(section_caps_by_src, all_caps):
    """HF kernel-builder per-source gencode: section-declared caps (+ Thor)."""
    result = {}
    for src, caps in section_caps_by_src.items():
        caps = _dedup(caps)
        if not caps:
            caps = list(all_caps)
        else:
            if FORCED_CAP not in caps:
                caps = list(caps) + [FORCED_CAP]
        result[src] = caps
    return result


def write_ninja_build(ops_name, build_dir, sources, src_specs, extra_include,
                      extra_cflags, extra_cuda, extra_ldflags):
    """Write build.ninja with per-source CUDA gencode/flags, then run ninja.

    src_specs: dict mapping absolute source path ->
        {"kind": "cpp"|"cuda", "cuda_post": [flags...]}.
    Returns the path of the produced shared library.
    """
    ce.verify_ninja_availability()
    compiler = ce.get_cxx_compiler()
    nvcc = os.path.join(bc.CUDA_HOME, "bin", "nvcc")

    user_includes = [os.path.abspath(i) for i in extra_include]
    system_includes = ce.include_paths("cuda")
    py_inc = sysconfig.get_path("include", scheme="posix_prefix")
    if py_inc:
        system_includes.append(py_inc)

    common = [f"-DTORCH_EXTENSION_NAME={ops_name}",
              "-DTORCH_API_INCLUDE_EXTENSION_H"]
    for i in user_includes:
        common.append(f"-I{shlex_quote(i)}")
    for i in system_includes:
        common.append(f"-isystem {shlex_quote(i)}")

    cflags = common + ["-fPIC", "-std=c++17"] + extra_cflags
    cuda_cflags = (common + list(ce.COMMON_NVCC_FLAGS)
                   + ["--compiler-options", "'-fPIC'"] + extra_cuda)
    if not any(f.startswith("-std=") for f in cuda_cflags):
        cuda_cflags.append("-std=c++17")
    ldflags = [ce.SHARED_FLAG] + extra_ldflags

    objects = []
    for i, src in enumerate(sources):
        spec = src_specs[src]
        obj = f"obj_{i}.cuda.o" if spec["kind"] == "cuda" else f"obj_{i}.o"
        objects.append(obj)

    lines = [
        "ninja_required_version = 1.3",
        f"cxx = {compiler}",
        f"nvcc = {nvcc}",
        f'cflags = {" ".join(cflags)}',
        f'cuda_cflags = {" ".join(cuda_cflags)}',
        f'ldflags = {" ".join(ldflags)}',
        "",
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule cuda_compile",
        "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]
    for i, (src, obj) in enumerate(zip(sources, objects)):
        spec = src_specs[src]
        esc = src.replace(" ", "$ ")
        lines.append(f"build {obj}: {'cuda_compile' if spec['kind'] == 'cuda' else 'compile'} {esc}")
        if spec["kind"] == "cuda":
            lines.append(f"  cuda_post_cflags = {' '.join(spec['cuda_post'])}")
        else:
            lines.append("  post_cflags =")
        lines.append("")
    so = f"{ops_name}.so"
    lines.append(f"build {so}: link {' '.join(objects)}")
    lines.append(f"default {so}")
    lines.append("")

    os.makedirs(build_dir, exist_ok=True)
    with open(os.path.join(build_dir, "build.ninja"), "w") as f:
        f.write("\n".join(lines))
    ce._run_ninja_build(build_dir, False, f"Error building extension '{ops_name}'")
    so_path = os.path.join(build_dir, so)
    if not os.path.exists(so_path):
        raise RuntimeError("no .so produced")
    return so_path


def shlex_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def build_with_fallback(ops_name, build_dir, sources, src_specs, extra_include,
                        extra_cflags, extra_cuda, extra_ldflags):
    """Build, retrying per-source when the forced 11.0a target is impossible.

    A source whose own sections do not declare ``11.0a`` gets the Thor target
    added so the artifact runs on SM110. Some SM120-only sources cannot emit
    SM110 SASS (e.g. ``mma with block scale``); for those the forced gencode is
    dropped so the package still builds with its declared capability targets.
    """
    forced = {s for s in sources if src_specs[s].get("forced11")}
    for _attempt in range(8):
        try:
            return write_ninja_build(ops_name, build_dir, sources, src_specs,
                                     extra_include, extra_cflags, extra_cuda,
                                     extra_ldflags)
        except RuntimeError as e:
            err = str(e)
            failing = [s for s in forced if os.path.basename(s) in err]
            if not failing:
                raise
            for s in failing:
                src_specs[s]["cuda_post"] = [
                    f for f in src_specs[s]["cuda_post"]
                    if not f.startswith(f"-gencode=arch=compute_{bc.normalize_cap(FORCED_CAP)}")
                ]
                src_specs[s]["forced11"] = False
                forced.discard(s)
            print(f"    [fallback] {len(failing)} source(s) cannot emit "
                  f"compute_{bc.normalize_cap(FORCED_CAP)}: "
                  + ", ".join(os.path.basename(s) for s in failing), flush=True)
    raise RuntimeError(f"could not stabilize build for {ops_name}")


def build_one(pkg_dir):
    import tomli
    with open(os.path.join(pkg_dir, "build.toml"), "rb") as f:
        build = tomli.load(f)

    general = build.get("general", {})
    name = general.get("name", os.path.basename(pkg_dir.rstrip("/")))
    version = general.get("version", 1)
    license_name = general.get("license", "Apache-2.0")

    torch_cfg = build.get("torch", {})
    kernels = build.get("kernel", {})
    cuda_kernels = {k: v for k, v in kernels.items() if v.get("backend") == "cuda"}
    if not cuda_kernels:
        return {"status": "skipped", "reason": "no cuda kernel sections"}

    py_pkg = bc.python_pkg_dir(pkg_dir)
    if not py_pkg:
        return {"status": "skipped", "reason": "no python package dir"}

    all_sources = bc.collect_compile_sources(pkg_dir, torch_cfg, cuda_kernels)
    includes = bc.collect_includes(pkg_dir, torch_cfg, cuda_kernels)
    cutlass_deps = bc.collect_cutlass_deps(cuda_kernels)

    if not all_sources:
        return {"status": "skipped", "reason": "no compilable sources"}

    # SM110 dispatch override: when a `*_sm110` section provides a dispatch
    # source that *defines* the same top-level symbols as native SM120 sources
    # (e.g. grouped-moe-gemv's sm110_dispatch.cu redefines nexn2_*_bf16 and
    # grouped_w4a4_gemv_sm120_bf16), both cannot coexist in one .so. Drop the
    # SM120 sources whose symbols the dispatch redefines, so the single
    # artifact links once. quantize_activations_nvfp4.cu etc. that the
    # dispatch does NOT redefine stay compiled for both.
    sm110_srcs = []
    for kname, kcfg in cuda_kernels.items():
        caps = kcfg.get("cuda-capabilities", []) or []
        if any("11.0a" in c for c in caps):
            for s in kcfg.get("src", []):
                if os.path.basename(s) == "sm110_dispatch.cu":
                    sm110_srcs.append(os.path.join(pkg_dir, s))
    if sm110_srcs:
        dispatch_symbols = set()
        for ds in sm110_srcs:
            try:
                for line in open(ds):
                    m = re.match(r"\s*(?:int|void|bool|float|double|long|size_t|unsigned|char|uint\d+_t|int\d+_t)\s+([A-Za-z_]\w*)\s*\(", line)
                    if m:
                        dispatch_symbols.add(m.group(1))
            except OSError:
                pass
        if dispatch_symbols:
            excluded = []
            for s in list(all_sources):
                base = os.path.basename(s)
                if base == "sm110_dispatch.cu" or base == "w4a16_edge_sm120.cu":
                    continue
                try:
                    defined = [m.group(1) for line in open(s)
                               if (m := re.match(r"\s*(?:int|void|bool|float|double|long|size_t|unsigned|char|uint\d+_t|int\d+_t)\s+([A-Za-z_]\w*)\s*\(", line))]
                except OSError:
                    continue
                if dispatch_symbols & set(defined):
                    all_sources.remove(s)
                    excluded.append(os.path.basename(s))
            if excluded:
                print(f"    [sm110-dispatch] excluded native sources shadowed by "
                      f"dispatch: {', '.join(excluded)}", flush=True)

    declared_all = []
    section_map = {}
    flags_by_src = {}
    for kname, kcfg in cuda_kernels.items():
        caps = kcfg.get("cuda-capabilities", []) or []
        for c in caps:
            if c not in declared_all:
                declared_all.append(c)
        kflags = kcfg.get("cuda-flags", []) or []
        for s in kcfg.get("src", []):
            p = os.path.join(pkg_dir, s)
            if os.path.splitext(p)[1] not in bc.COMPILABLE_EXTS:
                continue
            sec = section_map.setdefault(p, [])
            for c in caps:
                if c not in sec:
                    sec.append(c)
            for f in kflags:
                if f not in flags_by_src.setdefault(p, []):
                    flags_by_src[p].append(f)

    global_undefs = []
    for kcfg in cuda_kernels.values():
        for f in kcfg.get("cuda-flags", []) or []:
            if f.startswith("-U") and f not in global_undefs:
                global_undefs.append(f)

    all_caps = list(declared_all)
    if FORCED_CAP not in all_caps:
        all_caps.append(FORCED_CAP)
    archs = list(all_caps)

    uid = package_unique_id(pkg_dir, py_pkg, all_sources)
    ops_name = f"_{py_pkg}_cuda_{uid}"

    extra_include = list(includes)
    extra_include.append(bc.REGISTRATION_DIR)
    extra_include.append(f"{bc.CUDA_HOME}/include")
    pkg_short = os.path.basename(pkg_dir.rstrip("/"))
    for inc in bc.EXTRA_INCLUDES.get(pkg_short, []):
        if inc not in extra_include:
            extra_include.append(inc)
    for dep in cutlass_deps:
        cut = bc.CUTLASS_DIRS.get(dep)
        if cut and os.path.isdir(os.path.join(cut, "include")):
            extra_include.append(os.path.join(cut, "include"))
            util = os.path.join(cut, "tools", "util", "include")
            if os.path.isdir(util):
                extra_include.append(util)

    extra_cflags = ["-O3", "-DCUDA_KERNEL", "-DUSE_CUDA=1", "-fno-gnu-unique"]
    extra_cuda = [
        "-O3", "-DCUDA_KERNEL", "-DUSE_CUDA=1", "-std=c++17",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
    ]

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    nvidia_lib = os.path.join(
        os.path.dirname(torch.__file__), "..", "nvidia", "cu13", "lib"
    )
    cuda_lib = os.path.join(bc.CUDA_HOME, "lib64")
    extra_ldflags = [
        f"-L{torch_lib}", f"-L{nvidia_lib}", f"-L{cuda_lib}",
        f"-Wl,-rpath,{torch_lib}", f"-Wl,-rpath,{nvidia_lib}",
        f"-Wl,-rpath,{cuda_lib}",
        "-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda", "-ltorch",
        "-ltorch_python", "-lcudart",
        "-l:libcublasLt.so.13", "-l:libcublas.so.13",
    ]

    per_src = per_source_caps(section_map, all_caps)

    src_specs = {}
    for src in all_sources:
        if src.endswith((".cu",)):
            gencodes = [bc.cap_to_gencode(c) for c in per_src[src]]
            cuda_post = (gencodes + global_undefs
                         + [f for f in flags_by_src.get(src, [])
                            if f not in global_undefs])
            declared = section_map.get(src, [])
            forced11 = bool(declared) and FORCED_CAP not in declared
            src_specs[src] = {"kind": "cuda", "cuda_post": cuda_post,
                              "forced11": forced11}
        else:
            src_specs[src] = {"kind": "cpp", "cuda_post": []}

    build_dir = os.path.join("/data/build_tmp", pkg_short, ops_name)
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    so = build_with_fallback(ops_name, build_dir, all_sources, src_specs,
                             extra_include, extra_cflags, extra_cuda,
                             extra_ldflags)

    variant = variant_name()
    variant_dir = os.path.join(bc.OUT_HF, pkg_short, "build", variant)
    out_pkg = os.path.join(variant_dir, py_pkg)
    if os.path.exists(out_pkg):
        shutil.rmtree(out_pkg)
    os.makedirs(out_pkg, exist_ok=True)

    src_pkg = os.path.join(pkg_dir, "torch-ext", py_pkg)
    for root, dirs, files in os.walk(src_pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = os.path.relpath(root, src_pkg)
        dest = os.path.join(out_pkg, rel)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            if f.endswith(".so"):
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(dest, f))

    shutil.copy2(so, os.path.join(out_pkg, os.path.basename(so)))

    ops_py = (
        "import torch\n"
        "from . import " + ops_name + "\n"
        "ops = torch.ops." + ops_name + "\n"
        "\n"
        "def add_op_namespace_prefix(op_name: str):\n"
        "    \"\"\"\n"
        "    Prefix op by namespace.\n"
        "    \"\"\"\n"
        "    return f\"" + ops_name + "::{op_name}\"\n"
    )
    with open(os.path.join(out_pkg, "_ops.py"), "w") as f:
        f.write(ops_py)

    meta_archs = sorted({normalize_arch_for_metadata(a) for a in archs})
    write_metadata(variant_dir, name, ops_name, version, license_name, meta_archs)

    return {
        "status": "built",
        "ops_name": ops_name,
        "archs": archs,
        "metadata_archs": meta_archs,
        "variant": variant,
        "so": os.path.join(out_pkg, os.path.basename(so)),
        "variant_dir": variant_dir,
    }


def main():
    pkg_dir = sys.argv[1]
    result = {"pkg": os.path.basename(pkg_dir.rstrip("/"))}
    try:
        r = build_one(pkg_dir)
        result.update(r)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()[-3000:]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
