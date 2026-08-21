"""Run every package's own correctness test against its HF build artifact.

Usage: python run_hf_pkg_tests.py [pkg ...]
  With no args, tests every package under /data/hf_build.
"""
import json
import os
import shutil
import subprocess
import sys

PY = "/data/venv_thor211/bin/python"
LD = ("/data/venv_thor211/lib/python3.10/site-packages/torch/lib:"
      "/data/venv_thor211/lib/python3.10/site-packages/nvidia/cu13/lib:"
      "/data/venv_thor211/lib/python3.10/site-packages/nvidia/cudnn/lib:"
      "/data/venv_thor211/lib/python3.10/site-packages/nvidia/nccl/lib:"
      "/data/venv_thor211/lib/python3.10/site-packages/nvidia/cusparselt/lib:"
      "/data/venv_thor211/lib/python3.10/site-packages/nvidia/nvshmem/lib:"
      "/usr/local/cuda-13.2/compat:/usr/local/cuda-13.2/lib64")
REPO = "/data/FlashRT-HF-kernels"
HF = "/data/hf_build"
VARIANT = "torch211-cxx11-cu130-aarch64-linux"
RESULTS = "/data/hf_pkgtest_results.json"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = LD
env["PYTHONPATH"] = "/data"

only = set(sys.argv[1:])

tests = []
for p in sorted(os.listdir(REPO)):
    if only and p not in only:
        continue
    d = os.path.join(REPO, p)
    variant = os.path.join(HF, p, "build", VARIANT)
    if not os.path.isdir(variant):
        continue
    td = os.path.join(d, "tests")
    for f in sorted(os.listdir(td)) if os.path.isdir(td) else []:
        if f.startswith("test_") and f.endswith(".py"):
            src = open(os.path.join(td, f)).read()
            if "--backend" in src and "installed" in src:
                top_level = "artifact_path / \"__init__.py\"" in src or \
                            'artifact_path / "__init__.py"' in src
                tests.append((p, os.path.join(td, f), top_level))

def summarize(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line in ("{", "}", "[" , "]"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        passed = payload.get("passed")
        total = payload.get("total", payload.get("failed"))
        if total is None and isinstance(payload.get("rows"), list):
            total = len(payload["rows"])
        if passed is not None and total is not None:
            return f"passed {passed}/{total}"
    return (stdout.strip().splitlines()[-1] if stdout.strip() else "")[:300]


ARTIFACT_CACHE = {}
def artifact_for(pkg, top_level):
    variant = os.path.join(HF, pkg, "build", VARIANT)
    if not top_level:
        return variant
    if pkg in ARTIFACT_CACHE:
        return ARTIFACT_CACHE[pkg]
    top = f"/tmp/hf_art_{pkg}"
    shutil.rmtree(top, ignore_errors=True)
    os.makedirs(top)
    py_pkgs = [x for x in os.listdir(variant) if x != "metadata.json"]
    for x in py_pkgs:
        for entry in os.listdir(os.path.join(variant, x)):
            os.symlink(os.path.join(variant, x, entry), os.path.join(top, entry))
    ARTIFACT_CACHE[pkg] = top
    return top

results = {}
if os.path.exists(RESULTS):
    try:
        results = json.load(open(RESULTS))
    except Exception:
        results = {}

for i, (pkg, test, top_level) in enumerate(tests, 1):
    key = f"{pkg}::{os.path.basename(test)}"
    if key in results and results[key].get("status") == "pass":
        print(f"[{i}/{len(tests)}] {key} ... skip", flush=True)
        continue
    artifact = artifact_for(pkg, top_level)
    print(f"[{i}/{len(tests)}] {key} ...", flush=True)
    try:
        r = subprocess.run(
            [PY, test, "--backend", "installed", "--artifact", artifact],
            env=env, capture_output=True, text=True, timeout=2400,
        )
        out = r.stdout.strip().splitlines()
        if "Traceback" in r.stdout or r.returncode != 0:
            status = "fail"
            detail = (r.stdout + r.stderr)[-500:]
        else:
            status = "pass"
            detail = summarize(r.stdout)
    except subprocess.TimeoutExpired:
        status = "fail"
        detail = "timeout"
    results[key] = {"status": status, "detail": detail}
    json.dump(results, open(RESULTS, "w"), indent=1)
    print(f"    -> {status} {detail}", flush=True)

fails = {k: v for k, v in results.items() if v["status"] != "pass"}
print(f"\nDONE: {len(results) - len(fails)}/{len(results)} passed; "
      f"{len(fails)} failed -> {RESULTS}")
for k, v in fails.items():
    print(f"FAIL {k}: {v['detail'][:200]}")
