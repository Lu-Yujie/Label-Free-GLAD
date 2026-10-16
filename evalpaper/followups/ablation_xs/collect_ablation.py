"""A33.2: x_s-ablation CSVs from the ablation parts. NOTES.md §5.

  $GLAD_ENVS/grail-prelim/bin/python -B evalpaper/followups/ablation_xs/collect_ablation.py [--xs zeros]   # dry run
  ... collect_ablation.py --xs zeros --final
        -> results/eval_ablation_xs_folds.csv   (UB9; eval_folds schema + a33 columns)
           results/eval_ablation_xs_typed.csv   (typed; eval_typed schema + a33 columns)
        (--xs ones writes ..._ones_folds.csv / ..._ones_typed.csv)
  ... collect_ablation.py --xs ones --smoke --parts_ub9 <dir> --parts_typed <dir> --out_dir <tmp>   # tests

Checks every part (meta.a33.variant of the chosen --xs, device type == jobs/device_plan_ablation.json unless --smoke),
then runs harness/collect.py (UB9) and typed/collect_typed.py (typed) UNCHANGED with inputs/outputs redirected, keeps
only these runs and adds a33_variant / planned_device. Prints counts only, never a metric.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
FU = osp.dirname(HERE)
sys.path.insert(0, FU)
import a33_common as A  # noqa: E402

import pandas as pd  # noqa: E402

PLAN = osp.join(FU, "jobs", "device_plan_ablation.json")
VARIANTS = {"zeros": "x_s=0", "ones": "x_s=1"}


def check(parts, typed, xs, plan, smoke):
    bad, mism, info = [], [], {}
    use = [(fn, m) for fn, m in parts if m["_tagged"] == smoke]
    for fn, m in use:
        a33 = m.get("a33", {})
        if a33.get("variant") != VARIANTS[xs] or a33.get("path") != ("typed" if typed else "ub9"):
            bad.append(m["run_id"])
            continue
        if typed:
            unit = {"GX": "GX:F2", "MSV2": "MSV2:MS"}[m["generator"]]
            key = f"abl_typed {m['method']} {unit} {m['seed']}"
        else:
            key = f"abl_ub9 {m['method']} {m['dataset']} {m['fold']}"
        dev, want = A.device_type(m), plan[key]["device"]
        if dev != want and not smoke:
            mism.append(f"{m['run_id']} ran on {dev}, planned {want}")
        info[m["_stem"]] = dict(planned_device=want, unablated_run_id=plan[key]["unablated_run_id"])
    return use, bad, mism, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xs", choices=sorted(VARIANTS), default="zeros")
    ap.add_argument("--parts_ub9", default=None)
    ap.add_argument("--parts_typed", default=None)
    ap.add_argument("--out_dir", default=None, help="tests: write both CSVs here")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tests: tagged parts, no device-plan check")
    a = ap.parse_args()
    grid_ub9 = osp.join(HERE, f"parts_ub9_{a.xs}")
    grid_typed = osp.join(HERE, f"parts_typed_{a.xs}")
    a.parts_ub9 = a.parts_ub9 or grid_ub9
    a.parts_typed = a.parts_typed or grid_typed
    if a.final and (a.out_dir or a.smoke or osp.realpath(a.parts_ub9) != osp.realpath(grid_ub9)
                    or osp.realpath(a.parts_typed) != osp.realpath(grid_typed)):
        sys.exit("--final reads only the grid parts directories and writes only the official files")
    plan = json.load(open(PLAN))["plan"]
    sfx = "" if a.xs == "zeros" else "_ones"
    outs = {"ub9": osp.join(A.PRELIM, "results", f"eval_ablation_xs{sfx}_folds.csv"),
            "typed": osp.join(A.PRELIM, "results", f"eval_ablation_xs{sfx}_typed.csv")}
    if a.out_dir:
        outs = {k: osp.join(a.out_dir, osp.basename(v)) for k, v in outs.items()}
    os.makedirs(A.SCRATCH, exist_ok=True)
    for path, pdir in (("ub9", a.parts_ub9), ("typed", a.parts_typed)):
        typed = path == "typed"
        parts = A.scan_parts(pdir, typed=typed) if osp.isdir(pdir) else []
        use, bad, mism, info = check(parts, typed, a.xs, plan, a.smoke)
        print(f"[{path}] parts: {len(use)} {'tagged (smoke)' if a.smoke else 'untagged'} runs "
              f"({VARIANTS[a.xs]}), {len(parts) - len(use)} other (ignored), {len(bad)} invalid, {len(mism)} device mismatches")
        if bad:
            sys.exit(f"[{path}] parts without the A33.2 {VARIANTS[a.xs]} marker: {bad}")
        if mism:
            sys.exit(f"[{path}] device mismatches:\n  " + "\n  ".join(mism))
        if not use:
            continue
        tmp = tempfile.mkdtemp(prefix=f"collect_abl_{path}_", dir=A.SCRATCH)
        try:
            src = pdir
            if a.smoke:
                src = osp.join(tmp, "fixture")
                A.untagged_fixture(use, src)
            res = osp.join(tmp, "res")
            os.makedirs(res)
            if typed:
                A.run_collect(A.TYPED_COLLECT, args=("--parts", src, "--out", osp.join(res, "eval_typed.csv")))
                df = pd.read_csv(osp.join(res, "eval_typed.csv"))
            else:
                A.run_collect(A.HARNESS_COLLECT, dict(EP_PARTS=src, EP_RES=res))
                df = pd.read_csv(osp.join(res, "eval_folds.csv"))
                df = df[df.source == "evalpaper_harness"].copy()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        df["a33_variant"] = VARIANTS[a.xs]
        df["planned_device"] = df.run_id.map(lambda r: info[r]["planned_device"])
        df["unablated_run_id"] = df.run_id.map(lambda r: info[r]["unablated_run_id"])
        assert df.planned_device.notna().all()
        print(f"[{path}] rows kept {len(df)}; runs {df.run_id.nunique()}")
        if a.final or a.out_dir:
            df.to_csv(outs[path], index=False)
            print(f"[{path}] wrote -> {outs[path]}")
    if not (a.final or a.out_dir):
        print("dry run: nothing written (use --final at the end of the grid, or --out_dir for a test)")


if __name__ == "__main__":
    main()
