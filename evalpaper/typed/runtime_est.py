"""Runtime model for the A9 typed grid, from the smoke timings (typed/runtime_smoke.csv, written by smoke_check.py).

Per method x generator: seconds/epoch (training) and seconds/evaluation point (all per-type passes + combined pass +
held-out), measured in the smoke job (49 concurrent runs on one 128-core EPYC 7742 debug node, i.e. at grid-like load),
at the thread count the grid will use (THREADS below: 4 for the long contrastive loops when a 4-thread timing exists,
else 2). Single-run estimate = num_epoch * s/epoch + n_eval * s/eval + setup. Writes typed/runtime_est.json (read by
jobs/gen_tasks.py for longest-first ordering and thread/memory requests) and prints the grid totals.

  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B runtime_est.py
"""
import sys

sys.dont_write_bytecode = True

import json
import math
import os.path as osp

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "jobs"))
import gen_tasks as G  # noqa: E402

THREADS = {"SIGNET": 4, "CVTGAD": 4, "GOOD-D": 4, "GraphDE": 4}
GPU_METHODS = ("CVTGAD", "GOOD-D")     # RUN_NOTES §0b
SETUP_S = 150.0          # interpreter + torch/PyG import from the shared file system under load, data build (measured ~100-140 s)
MEM_FLOOR_GB = 3.0


def main():
    rt_all = pd.read_csv(osp.join(HERE, "runtime_smoke.csv"))
    rt = rt_all[rt_all.device == "cpu"]
    est, rows = {}, []
    for m in G.METHODS:
        for g in ("GX", "MS"):
            th = THREADS.get(m, 2)
            s = rt[(rt.method == m) & (rt.generator == g) & (rt.threads == th)]
            src = f"smoke {th} threads"
            if not len(s):          # 4-thread timing only measured on MS: scale GX 2-thread timing by the MS ratio
                s2 = rt[(rt.method == m) & (rt.generator == g) & (rt.threads == 2)]
                r2 = rt[(rt.method == m) & (rt.generator == "MS") & (rt.threads == 2)]
                r4 = rt[(rt.method == m) & (rt.generator == "MS") & (rt.threads == th)]
                f_ep = float(r4.sec_per_epoch_train.max() / r2.sec_per_epoch_train.max()) if len(r4) and len(r2) else 1.0
                f_ev = float(r4.sec_per_eval.max() / r2.sec_per_eval.max()) if len(r4) and len(r2) else 1.0
                spe, spv = float(s2.sec_per_epoch_train.max()) * f_ep, float(s2.sec_per_eval.max()) * f_ev
                mem = float(s2.maxrss_gb.max())
                src = f"smoke 2 threads x MS {th}/2-thread ratio ({f_ep:.2f}, {f_ev:.2f})"
            else:
                spe, spv, mem = float(s.sec_per_epoch_train.max()), float(s.sec_per_eval.max()), float(s.maxrss_gb.max())
            ne, _ = G.EPOCHS[m]
            sec = ne * spe + G.n_eval(m) * spv + SETUP_S
            mem_req = max(MEM_FLOOR_GB, math.ceil(1.5 * mem + 1.0))
            est.setdefault(m, {})[g] = dict(sec_per_epoch=spe, sec_per_eval=spv, setup_s=SETUP_S, threads=th,
                                            mem_gb=mem_req, peak_rss_gb_smoke=mem, source=src, device="cpu")
            cpu_sec = sec
            if m in GPU_METHODS:      # A100 timings of the gpu8 smoke (GX measured on F2; F1 assumed equal)
                q = rt_all[(rt_all.method == m) & (rt_all.generator == g) & (rt_all.tag == "gpu8")]
                assert len(q), (m, g, "no GPU timing")
                gspe, gspv = float(q.sec_per_epoch_train.max()), float(q.sec_per_eval.max())
                gsec = ne * gspe + G.n_eval(m) * gspv + SETUP_S
                est[m][g]["gpu"] = dict(sec_per_epoch=gspe, sec_per_eval=gspv, setup_s=SETUP_S, threads=4, mem_gb=4,
                                        device="cuda (A100 40GB, 1 run per GPU)", source="gpu8 smoke job <jobid>")
                sec = gsec
            n_runs = 10 if g == "GX" else 5
            dev = "gpu" if m in GPU_METHODS else "cpu"
            rows.append(dict(method=m, generator=g, device=dev, threads=th, num_epoch=ne, n_eval=G.n_eval(m),
                             est_run_h=sec / 3600.0, cpu_est_run_h=cpu_sec / 3600.0, runs=n_runs,
                             core_h=(0.0 if dev == "gpu" else n_runs * sec * th / 3600.0),
                             gpu_h=(n_runs * sec / 3600.0 if dev == "gpu" else 0.0), peak_rss_gb=mem, mem_req_gb=mem_req))
    with open(osp.join(HERE, "runtime_est.json"), "w") as f:
        json.dump(est, f, indent=1)
    df = pd.DataFrame(rows).sort_values("est_run_h", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    c = df[df.device == "cpu"]
    print(f"plan 'mixed' (RUN_NOTES §0b): CPU part {int(c.runs.sum())} runs, {c.core_h.sum():.1f} core-hours, longest "
          f"{c.est_run_h.max():.2f} h; GPU part {int(df[df.device == 'gpu'].runs.sum())} runs, {df.gpu_h.sum():.1f} "
          f"A100-hours, longest {df[df.device == 'gpu'].est_run_h.max():.2f} h")
    allcpu = sum(r.runs * r.cpu_est_run_h * r.threads for r in df.itertuples())
    print(f"fallback 'all-cpu': {allcpu:.1f} core-hours, longest single run {df.cpu_est_run_h.max():.2f} h")


if __name__ == "__main__":
    main()
