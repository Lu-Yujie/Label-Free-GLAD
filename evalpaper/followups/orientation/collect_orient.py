"""A33.1: eval_orientation_folds.csv (eval_folds schema + A33 columns) from the orientation parts. NOTES.md §6.

  $GLAD_ENVS/grail-prelim/bin/python -B evalpaper/followups/orientation/collect_orient.py           # dry run
  ... collect_orient.py --final      # writes results/eval_orientation_folds.csv (end of the grid)
  ... collect_orient.py --parts <smoke parts> --smoke --out <tmp.csv>    # tests (tagged parts via an untagged fixture)

1. Checks every part: meta.a33.variant == 'flipped', fold-cache hash equal to the current cache, hp 'ubgold', device
   type == jobs/device_plan_orient.json (unless --smoke).
2. Runs the UNCHANGED harness/collect.py with EP_PARTS=<parts> and EP_RES=<temporary dir>.
3. Keeps only rows from these parts (collect.py also appends c940457 deep_rerun rows of the old 4 for cells it does not
   find; they are dropped and counted), adds convention / a33_variant / anomalous_class / planned_device, writes the CSV.
   The temporary directory (extra, coverage, repro files) is deleted unread. Prints counts only, never a metric.
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

GRID_PARTS = osp.join(HERE, "parts")
FINAL_OUT = osp.join(A.PRELIM, "results", "eval_orientation_folds.csv")
PLAN = osp.join(FU, "jobs", "device_plan_orient.json")
FOLD_DIR = osp.join(A.DATA_ROOT, "orientation", "folds")
DS3 = ["BZR", "COX2", "PROTEINS_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=GRID_PARTS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--final", action="store_true", help=f"write {FINAL_OUT}")
    ap.add_argument("--smoke", action="store_true", help="tests: tagged parts, no device-plan check")
    a = ap.parse_args()
    if a.final and (a.out or a.smoke or osp.realpath(a.parts) != osp.realpath(GRID_PARTS)):
        sys.exit("--final reads only the grid parts directory and writes only the official file")
    plan = json.load(open(PLAN))["plan"]
    cache_sha = {ds: A.sha256_file(osp.join(FOLD_DIR, f"{ds}.json")) for ds in DS3}
    parts = A.scan_parts(a.parts)
    use = [(fn, m) for fn, m in parts if m["_tagged"] == a.smoke]
    bad, mism, info = [], [], {}
    for fn, m in use:
        a33 = m.get("a33", {})
        if a33.get("variant") != "flipped" or a33.get("fold_cache_sha256") != cache_sha.get(m["dataset"]) \
                or m.get("hp", "ubgold") != "ubgold":
            bad.append(m["run_id"])
            continue
        dev = A.device_type(m)
        want = plan[f"{m['method']} {m['dataset']} {m['fold']}"]["device"]
        if dev != want and not a.smoke:
            mism.append(f"{m['run_id']} ran on {dev}, planned {want}")
        info[m["_stem"]] = dict(planned_device=want, run_device=dev)
    print(f"parts: {len(use)} {'tagged (smoke)' if a.smoke else 'untagged'} A33.1 runs, {len(parts) - len(use)} "
          f"other (ignored), {len(bad)} invalid, {len(mism)} device mismatches")
    if bad:
        sys.exit(f"parts without the A33.1 flipped marker / current fold cache / 'ubgold': {bad}")
    if mism:
        sys.exit("device mismatches (A33 pairs device types):\n  " + "\n  ".join(mism))
    os.makedirs(A.SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="collect_orient_", dir=A.SCRATCH)
    try:
        src = a.parts
        if a.smoke:
            src = osp.join(tmp, "fixture")
            A.untagged_fixture(use, src)
        res = osp.join(tmp, "res")
        os.makedirs(res)
        A.run_collect(A.HARNESS_COLLECT, dict(EP_PARTS=src, EP_RES=res))
        df = pd.read_csv(osp.join(res, "eval_folds.csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_all = len(df)
    df = df[df.source == "evalpaper_harness"].copy()
    df["convention"] = "flipped"
    df["a33_variant"] = "A33.1 flipped: class 1 (minority) anomalous"
    df["anomalous_class"] = 1
    df["planned_device"] = df.run_id.map(lambda r: info[r]["planned_device"])
    assert df.planned_device.notna().all() and set(df.dataset) <= set(DS3), set(df.dataset)
    print(f"collect.py rows: {n_all}; dropped {n_all - len(df)} non-A33 rows (deep_rerun); kept {len(df)}")
    cov = df[df.protocol == "oracle"].groupby(["method", "dataset"]).fold.nunique()
    print(f"cells with 5 folds: {int((cov == 5).sum())} of 36; protocols {sorted(df.protocol.unique())}")
    out = FINAL_OUT if a.final else a.out
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows -> {out}")
    else:
        print("dry run: nothing written (use --final at the end of the grid, or --out for a test)")


if __name__ == "__main__":
    main()
