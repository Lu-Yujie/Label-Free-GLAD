"""A33 task lists and device plans (grail-prelim env; reads metadata only, never a test metric).

  $GLAD_ENVS/grail-prelim/bin/python -B jobs/gen_tasks.py [--xs zeros|ones]

Writes (in jobs/):
  tasks_orient.txt          12 detectors x BZR/COX2/PROTEINS_full x 5 folds, flipped convention (A33.1)
  tasks_ablation.txt        CVTGAD/GOOD-D x (UB9 x 5 folds + GX:F2 / MSV2:MS x 5 seeds), x_s = 0 (A33.2 as written)
  tasks_ablation_ones.txt   the same with x_s = 1 (only with --xs ones; NOTES.md §7 option, needs ratification)
  device_plan_orient.json   per (method, dataset, fold): main-grid device (eval_folds.csv) and its source
  device_plan_ablation.json per unit: device of the unablated counterpart (harness / typed parts meta)
Each line carries est= (seconds, for longest-first ordering and the runtime estimate), mem= / gmem= (GB) and threads=.
Orientation estimates scale the main-grid run time by n_train_normal(flipped) / n_train_normal(original) per fold.
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import argparse
import glob
import json
import os.path as osp
import sys
from collections import defaultdict

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
FU = osp.dirname(HERE)
EVALPAPER = osp.dirname(FU)
PRELIM = osp.dirname(EVALPAPER)
sys.path.insert(0, FU)
import a33_common as A  # noqa: E402

DEEP12 = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD", "GLADC", "CVTGAD", "GOOD-D", "GraphDE",
          "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"]
ODS = ["BZR", "COX2", "PROTEINS_full"]
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
FOLD_DIR = osp.join(A.DATA_ROOT, "orientation", "folds")
SPLITS = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_splits")


def dev_of(s):
    return "cuda" if "cuda" in str(s) else "cpu"


def part_index(pattern):
    out = {}
    for f in glob.glob(pattern):
        m = json.load(open(f))["meta"]
        out[m["run_id"]] = m
    return out


def fmt(t):
    return (f"{t['exp']} {t['method']} {t['unit']} {t['fold']} dev={t['dev']} mem={t['mem']:.1f} gmem={t['gmem']:.1f} "
            f"threads={t['threads']} est={int(t['est'])}" + (f" xs={t['xs']}" if t.get("xs") else ""))


def orient_tasks():
    ef = pd.read_csv(osp.join(PRELIM, "results", "eval_folds.csv"),
                     usecols=["method", "dataset", "fold", "protocol", "device", "orientation", "source", "run_id"])
    ef = ef[(ef.protocol == "oracle") & (ef.orientation != "ubgold") & ef.dataset.isin(ODS) & ef.method.isin(DEEP12)]
    hp = part_index(osp.join(EVALPAPER, "harness", "parts", "*.json"))
    dr = part_index(osp.join(PRELIM, "results", "deep_rerun_parts", "*.json"))
    ratio = {}
    for ds in ODS:
        new = json.load(open(osp.join(FOLD_DIR, f"{ds}.json")))["folds"]
        old = json.load(open(osp.join(SPLITS, f"{ds}.json")))["folds"]
        for k in range(5):
            ratio[(ds, k)] = len(new[k]["train_idx"]) / len(old[k]["train_idx"])
    tasks, plan = [], {}
    for r in ef.itertuples():
        m = (hp if r.source == "evalpaper_harness" else dr).get(r.run_id)
        assert m is not None, r.run_id
        mem = m.get("memory", {}) or {}
        rr = ratio[(r.dataset, int(r.fold))]
        dev = dev_of(r.device)
        t = dict(exp="orient", method=r.method, unit=r.dataset, fold=int(r.fold), dev=dev,
                 threads=int(m.get("threads") or 2) if dev == "cpu" else 2,
                 est=float(m.get("seconds", 600)) * rr,
                 mem=max(2.0, float(mem.get("maxrss_gb", 2.0)) * min(rr, 2.0) + 1.0),
                 gmem=max(1.5, float(mem.get("cuda_max_alloc_gb", 0.5) or 0.5) * rr + 1.0) if dev == "cuda" else 0.0)
        if dev == "cuda" and "cuda_max_alloc_gb" not in mem:     # deep_rerun parts carry no GPU-memory record
            t["gmem"] = 4.0
        tasks.append(t)
        plan[f"{r.method} {r.dataset} {int(r.fold)}"] = dict(device=dev, source=r.source, main_run_id=r.run_id,
                                                              threads=t["threads"], train_ratio=round(rr, 4))
    assert len(tasks) == 12 * 3 * 5, len(tasks)
    return tasks, plan


def ablation_tasks(xs):
    hp = part_index(osp.join(EVALPAPER, "harness", "parts", "*.json"))
    tp = part_index(osp.join(EVALPAPER, "typed", "parts", "*.json"))
    tasks, plan = [], {}
    for M in ("CVTGAD", "GOOD-D"):
        for ds in UB9:
            for k in range(5):
                m = hp[f"{M}_{ds}_fold{k}"]
                dev = dev_of(m["device_all"])
                mem = m.get("memory", {}) or {}
                t = dict(exp="abl_ub9", method=M, unit=ds, fold=k, dev=dev, xs=xs,
                         threads=int(m.get("threads") or 2) if dev == "cpu" else 2, est=float(m["seconds"]),
                         mem=max(2.0, float(mem.get("maxrss_gb", 2.0)) + 1.0),
                         gmem=max(1.5, float(mem.get("cuda_max_alloc_gb", 0.5) or 0.5) + 1.0) if dev == "cuda" else 0.0)
                tasks.append(t)
                plan[f"abl_ub9 {M} {ds} {k}"] = dict(device=dev, unablated_run_id=m["run_id"], threads=t["threads"])
        for unit, (gen, fv) in A.TYPED_UNITS.items():
            for s in range(5):
                m = tp[f"{M}_{gen}_{fv}_s{s}"]
                dev = dev_of(m["device_all"])
                mem = m.get("memory", {}) or {}
                t = dict(exp="abl_typed", method=M, unit=unit, fold=s, dev=dev, xs=xs,
                         threads=int(m.get("threads") or 4), est=float(m["seconds"]),
                         mem=max(3.0, float(mem.get("maxrss_gb", 3.0)) + 1.0),
                         gmem=max(2.0, float(mem.get("cuda_max_alloc_gb", 1.0) or 1.0) + 1.0) if dev == "cuda" else 0.0)
                tasks.append(t)
                plan[f"abl_typed {M} {unit} {s}"] = dict(device=dev, unablated_run_id=m["run_id"], threads=t["threads"])
    assert len(tasks) == 110, len(tasks)
    return tasks, plan


def summary(name, tasks):
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for t in tasks:
        b = by[t["dev"]]
        b[0] += 1
        b[1] += t["est"]
        b[2] = max(b[2], t["est"])
    for dev, (n, s, mx) in sorted(by.items()):
        print(f"{name}: {dev:4s} {n:3d} tasks, sum {s / 3600:6.2f} h, longest {mx / 3600:5.2f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xs", choices=["zeros", "ones"], default="zeros")
    a = ap.parse_args()
    ot, op = orient_tasks()
    at, aplan = ablation_tasks(a.xs)
    aname = "ablation" if a.xs == "zeros" else "ablation_ones"
    for name, tasks, plan in (("orient", ot, op), (aname, at, aplan)):
        tasks.sort(key=lambda t: -t["est"])
        with open(osp.join(HERE, f"tasks_{name}.txt"), "w") as f:
            f.write(f"# A33 {name} tasks (jobs/gen_tasks.py --xs {a.xs}); longest first; est = seconds (estimate)\n")
            for t in tasks:
                f.write(fmt(t) + "\n")
        pname = "ablation" if name.startswith("ablation") else name      # device plan does not depend on x_s
        with open(osp.join(HERE, f"device_plan_{pname}.json"), "w") as f:
            json.dump(dict(generated_by="followups/jobs/gen_tasks.py", plan=plan), f, indent=1)
        summary(name, tasks)


if __name__ == "__main__":
    main()
