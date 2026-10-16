"""A9 typed grid task list for run_pool.py, LONGEST FIRST. Stdlib only (runs anywhere).

Full grid: 12 methods x (GX seeds 0-4 x {F2, F1} + MS seeds 0-4) = 180 runs (RUN_NOTES §0 item 1).
Excluded from the list:
  * finished runs (typed/parts/<run_id>.json exists);
  * runs with >= 2 failed attempts (crash records under <TYPED_ROOT>/crash/, or pool failures in
    logs/failed_attempts.tsv, e.g. a kill without a Python exception): never scheduled again (no retry loop); listed as
    DIVERGED candidates (A12), registered as DIVERGED only if both attempts failed the same way
    on non-finite values;
  * anything matched by jobs/EXCLUDE.txt (one token per line: a run id, a method, 'GX', 'MS', 'F1', ... ; lines
    starting with # are comments). It is used to drop tasks by decision, e.g. after a verdict.
Ordering: estimated single-run wall time (runtime_est.json: per method x generator seconds/epoch and seconds/evaluation
measured in the smoke jobs, at the thread count and on the device used here), descending.
Device plans (RUN_NOTES §0b; --plan):
  mixed     (default; for jobs with GPUs) CVTGAD and GOOD-D as GPU tasks ('gpu' flag), all other methods as CPU tasks;
  gpu-only  only the GPU-planned methods (GPU job);
  cpu-only  only the CPU-planned methods (CPU-only job; GPU-planned methods are never run on CPU by accident);
  all-cpu   every method on CPU (fallback; needs an explicit decision, recorded in the run log, and then applies to
            every remaining run of CVTGAD and GOOD-D).
Prints on stdout: 'remaining N threads T est_core_h H longest_h L', then the excluded / DIVERGED-candidate lists.

  python -B jobs/gen_tasks.py --out logs/tasks_<job>.txt [--generators GX,MS] [--methods A,B]
"""
import argparse
import glob
import json
import os
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
TYPED = osp.dirname(HERE)
PARTS = os.environ.get("TYPED_PARTS", osp.join(TYPED, "parts"))
TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
CRASH = osp.join(TYPED_ROOT, "crash")
EST = osp.join(TYPED, "runtime_est.json")
EXCLUDE = osp.join(HERE, "EXCLUDE.txt")
GPU_METHODS = {"CVTGAD", "GOOD-D"}      # RUN_NOTES §0b
FAILED = osp.join(TYPED, "logs", "failed_attempts.tsv")   # written by run_pool.py for every rc not in (0, 3)

METHODS = ["SIGNET", "CVTGAD", "GOOD-D", "GraphDE", "OCGTL", "OCGIN", "GLocalKD", "GLADC", "GraphCL_IF",
           "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"]
# modal UB-GOLD TU configuration (RUN_NOTES §0 item 6; resolved by run_typed.modal_hparams and recorded in every run)
EPOCHS = {"GLADC": (100, 4), "CVTGAD": (800, 4), "GOOD-D": (400, 4), "GraphDE": (400, 4), "GraphCL_IF": (30, 4),
          "GraphCL_OCSVM": (30, 2), "InfoGraph_IF": (30, 2), "InfoGraph_OCSVM": (30, 2), "SIGNET": (1000, 4),
          "OCGTL": (500, 4), "OCGIN": (500, 4), "GLocalKD": (150, 4)}
SEEDS = [0, 1, 2, 3, 4]
UNITS = [("GX", s, "F2") for s in SEEDS] + [("GX", s, "F1") for s in SEEDS] + [("MSV2", s, "MS") for s in SEEDS]   # CHANGE-A28.1: MS-v2 replaces retired MolSwap v1


def n_eval(m):
    ne, ef = EPOCHS[m]
    return len(set(range(ef, ne + 1, ef)) | {ne})


def estimates(plan):
    est = json.load(open(EST)) if osp.exists(EST) else {}
    out = {}
    for m in METHODS:
        for g in ("GX", "MSV2"):   # CHANGE-A28.1: MSV2 has no smoke timing; its test sets are GX-sized and its
            # training set is MS-sized, so use the larger of the GX and MS (v1) estimates field by field
            if g == "MSV2":
                eg, em = est.get(m, {}).get("GX", {}), est.get(m, {}).get("MS", {})
                e = {k: max(float(eg.get(k, 0)), float(em.get(k, 0))) for k in ("sec_per_epoch", "sec_per_eval", "setup_s", "mem_gb")}
                e["threads"] = int(em.get("threads", eg.get("threads", 2)))
                if "gpu" in eg or "gpu" in em:
                    gg, gm = eg.get("gpu", {}), em.get("gpu", {})
                    e["gpu"] = {k: max(float(gg.get(k, 0)), float(gm.get(k, 0))) for k in ("sec_per_epoch", "sec_per_eval", "setup_s", "mem_gb")}
                    e["gpu"]["threads"] = int(gm.get("threads", gg.get("threads", 2)))
            else:
                e = est.get(m, {}).get(g, {})
            dev = "gpu" if (m in GPU_METHODS and plan in ("mixed", "gpu-only")) else "cpu"
            if dev == "gpu":
                e = e.get("gpu", e)
            spe, spv = float(e.get("sec_per_epoch", 2.0)), float(e.get("sec_per_eval", 2.0))
            ne, _ = EPOCHS[m]
            out[(m, g)] = dict(seconds=ne * spe + n_eval(m) * spv + float(e.get("setup_s", 60.0)),
                               threads=int(e.get("threads", 2)), mem=float(e.get("mem_gb", 4.0)), dev=dev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--generators", default="GX,MSV2")   # CHANGE-A28.1
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--plan", default="mixed", choices=["mixed", "gpu-only", "cpu-only", "all-cpu"])
    a = ap.parse_args()
    gens = set(a.generators.split(","))
    meths = [m for m in METHODS if m in set(a.methods.split(","))]
    if a.plan == "gpu-only":
        meths = [m for m in meths if m in GPU_METHODS]
    elif a.plan == "cpu-only":
        meths = [m for m in meths if m not in GPU_METHODS]
    excl = set()
    if osp.exists(EXCLUDE):
        excl = {ln.split()[0] for ln in open(EXCLUDE) if ln.strip() and not ln.startswith("#")}
    fails = {}
    if osp.exists(FAILED):
        for ln in open(FAILED):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                fails[p[2]] = fails.get(p[2], 0) + 1
    E = estimates(a.plan)
    tasks, done, excluded, diverged = [], 0, [], []
    for m in meths:
        for g, s, fv in UNITS:
            if g not in gens:
                continue
            rid = f"{m}_{g}_{fv}_s{s}"
            if osp.exists(osp.join(PARTS, rid + ".json")):
                done += 1
                continue
            if excl & {rid, m, g, fv, f"{m}_{g}", f"{m}_{g}_{fv}"}:
                excluded.append(rid)
                continue
            if max(len(glob.glob(osp.join(CRASH, rid + ".attempt*.json"))), fails.get(rid, 0)) >= 2:
                diverged.append(rid)
                continue
            e = E[(m, g)]
            tasks.append((e["seconds"], m, g, s, fv, e))
    tasks.sort(key=lambda t: (-t[0], t[1], t[2], t[4], t[3]))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"# A9 typed grid, plan {a.plan}, longest first (estimated single-run seconds in the comment)\n")
        for sec, m, g, s, fv, e in tasks:
            f.write(f"{m} {g} {s} {fv} mem={e['mem']:.0f} threads={e['threads']} {e['dev']}   # est {sec:.0f}s\n")
    os.replace(tmp, a.out)
    th = sum(t[5]["threads"] for t in tasks if t[5]["dev"] == "cpu")
    core_h = sum(t[0] * t[5]["threads"] for t in tasks if t[5]["dev"] == "cpu") / 3600.0
    gpu_h = sum(t[0] for t in tasks if t[5]["dev"] == "gpu") / 3600.0
    n_gpu = sum(1 for t in tasks if t[5]["dev"] == "gpu")
    longest = (tasks[0][0] / 3600.0) if tasks else 0.0
    print(f"remaining {len(tasks)} cpu_threads {th} est_core_h {core_h:.1f} longest_h {longest:.2f} done {done} "
          f"gpu_tasks {n_gpu} est_gpu_h {gpu_h:.1f} plan {a.plan}")
    if excluded:
        print(f"excluded by jobs/EXCLUDE.txt: {len(excluded)}")
    if diverged:
        print(f"failed twice, not scheduled (A12 DIVERGED candidates; check crash records): {' '.join(diverged)}")


if __name__ == "__main__":
    main()
