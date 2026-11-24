#!/usr/bin/env python3
"""Run CPU/GPU comparisons from the checked-in, validated reference inputs."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PARAM_DIR = Path(__file__).resolve().parent
MODES = {
    "laplace": [("cpu_cg", 0, 0, 0), ("gpu_cg", 1, 0, 0),
                ("cpu_pcg", 0, 1, 0), ("gpu_pcg", 1, 1, 0)],
    "allen-cahn": [("cpu", 0, 0, 0), ("gpu", 0, 0, 2)],
}

def replace_setting(text, key, value, required=True):
    pattern = rf"(?m)^(\s*set\s+{re.escape(key)}\s*=)[^\r\n]*"
    new, count = re.subn(pattern, lambda m: f"{m.group(1)} {value}", text)
    if count != 1 and (required or count != 0):
        raise ValueError(f"Expected one setting {key!r}, found {count}")
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("problem", choices=MODES)
    p.add_argument("--refs", type=int, nargs="+")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--build-dir", type=Path, default=ROOT / "build/gpu")
    a = p.parse_args()
    if a.repeats < 1 or a.threads < 1:
        p.error("--repeats and --threads must be positive")
    refs = a.refs or ([5] if a.problem == "laplace" else [7])
    build = a.build_dir.resolve()
    exe = build / a.problem / "main"
    param = PARAM_DIR / ("laplace.prm" if a.problem == "laplace" else "allen_cahn.prm")
    if not exe.is_file() or not param.is_file():
        p.error(f"Missing executable or reference input: {exe}; {param}")
    base = param.read_text()
    run_root = build / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"{a.problem}_", dir=run_root))
    print(f"Results: {out}", flush=True)
    section = "Linear Solver" if a.problem == "laplace" else "Solve Blocks"
    pattern = rf"(?m)^\s*\|-\s+{re.escape(section)}\s+\d+\s+([0-9.]+)\s"
    with (out / "timing.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["refinement", "mode", "run", "section", "wall_seconds"])
        for ref in refs:
            prm = replace_setting(base, "global refinement", ref)
            if a.problem == "laplace":
                prm = replace_setting(prm, "max iterations", 10000)
            else:
                prm = replace_setting(prm, "max refinement", ref, required=False)
                prm = replace_setting(prm, "mesh adaptivity", "false")
            for mode, cg, jacobi, explicit in MODES[a.problem]:
                for run in range(1, a.repeats + 1):
                    d = out / f"ref{ref}" / mode / f"run{run}"
                    d.mkdir(parents=True)
                    (d / "parameters.prm").write_text(prm)
                    env = dict(os.environ, DEAL_II_NUM_THREADS=str(a.threads),
                               CUDA_VISIBLE_DEVICES=a.device, PRISMS_PF_C_GPU_VMULT="0",
                               PRISMS_PF_C_GPU_VERIFY="0", PRISMS_PF_D_GPU_CG=str(cg),
                               PRISMS_PF_D_JACOBI=str(jacobi), PRISMS_PF_E_GPU_EXPLICIT=str(explicit))
                    print(f"ref{ref} {mode} run{run}", flush=True)
                    with (d / "run.log").open("w") as log:
                        status = subprocess.run([str(exe), "-i", "parameters.prm"], cwd=d,
                                                env=env, stdout=log, stderr=subprocess.STDOUT)
                    raw = (d / "run.log").read_text(errors="replace")
                    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
                    if status.returncode or (a.problem == "laplace" and "did not converge" in raw.lower()):
                        raise RuntimeError(f"Run failed or did not converge: {d / 'run.log'}")
                    match = re.search(pattern, raw)
                    if not match:
                        raise RuntimeError(f"Timing section not found: {d / 'run.log'}")
                    writer.writerow([ref, mode, run, section, match.group(1)])
                    f.flush()
    print(f"Timings: {out / 'timing.csv'}")

if __name__ == "__main__":
    main()
