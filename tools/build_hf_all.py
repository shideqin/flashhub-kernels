"""Build all FlashRT packages into the HF Kernel Hub artifact layout.

Wraps build_hf.py for every package under /data/FlashRT-HF-kernels that has a
build.toml with CUDA kernels. Runs with the Thor venv (venv_thor).

Usage: python build_hf_all.py [pkg_dir_name ...]
  With no args, builds every package. With args, only those packages.
"""
import json
import os
import subprocess
import sys
import time

import build_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/data/venv_thor211/bin/python"
RESULTS = os.path.join(bc.OUT_HF, "build_hf_status.json")

env = dict(os.environ)
env["PATH"] = f"{bc.CUDA_HOME}/bin:{env.get('PATH', '')}"
env["CUDA_HOME"] = bc.CUDA_HOME
env["TORCH_CUDA_ARCH_LIST"] = ""


def package_dirs(only):
    dirs = []
    for f in sorted(os.listdir(bc.REPO)):
        d = os.path.join(bc.REPO, f)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "build.toml")):
            continue
        if only and f not in only:
            continue
        dirs.append(d)
    return dirs


def main():
    only = set(sys.argv[1:])
    os.makedirs(bc.OUT_HF, exist_ok=True)

    results = {}
    if os.path.exists(RESULTS):
        try:
            results = json.load(open(RESULTS))
        except Exception:
            results = {}

    dirs = package_dirs(only)
    for i, pkg_dir in enumerate(dirs, 1):
        pkg = os.path.basename(pkg_dir.rstrip("/"))
        if pkg in results and results[pkg].get("status") == "built":
            print(f"[{i}/{len(dirs)}] {pkg} ... skip (already built)", flush=True)
            continue
        t0 = time.time()
        print(f"[{i}/{len(dirs)}] {pkg} ...", flush=True)
        try:
            r = subprocess.run(
                [PY, os.path.join(HERE, "build_hf.py"), pkg_dir],
                env=env, capture_output=True, text=True, timeout=36000,
            )
            line = r.stdout.strip().splitlines()
            try:
                res = json.loads(line[-1] if line else "")
            except Exception:
                res = {"pkg": pkg, "status": "failed",
                       "error": line[-1][:200] if line else "no output",
                       "stderr": r.stderr[-500:]}
        except subprocess.TimeoutExpired:
            res = {"pkg": pkg, "status": "failed", "error": "timeout"}
        res["elapsed_s"] = round(time.time() - t0, 1)
        results[pkg] = res
        json.dump(results, open(RESULTS, "w"), indent=1)
        print(f"    -> {res['status']} {res.get('ops_name', '')} "
              f"{res.get('metadata_archs', '')} {res['elapsed_s']}s "
              f"{res.get('error', '')}", flush=True)

    ok = sum(1 for v in results.values() if v["status"] == "built")
    print(f"\nDONE: {ok}/{len(results)} built -> {RESULTS}")


if __name__ == "__main__":
    main()
