"""A17: eval_version_a96137b_folds.csv (schema: stats/FROZEN.md) from the a96137b parts. NOTES.md §3.9.

  $GLAD_ENVS/grail-prelim/bin/python -B evalpaper/version_a96137b/collect_a96137b.py            # dry run
  ... collect_a96137b.py --final        # writes results/eval_version_a96137b_folds.csv (end of the grid only)
  ... collect_a96137b.py --parts <dir> --out <tmp.csv> --allow_device_mismatch    # smoke tests

1. Checks every untagged part in --parts: meta.ubgold_commit == a96137b..., no max_epochs override, patched-tree digest
   present, device type == the A17 device plan (jobs/device_plan.json) unless --allow_device_mismatch.
2. Runs the UNCHANGED harness/collect.py with EP_PARTS=<parts> and EP_RES=<temporary dir>.
3. Keeps only rows with source == evalpaper_harness (collect.py also appends c940457 deep_rerun rows of the old-4
   methods for cells it does not find in EP_PARTS; they are dropped and counted), adds `ubgold_commit` from each
   part's meta, and writes the CSV. The temporary directory (eval_folds_extra, eval_coverage, eval_repro) is deleted
   unread. No metric value is printed; only row counts and fold coverage.
"""
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
COLLECT = osp.join(EVALPAPER, "harness", "collect.py")
GRID_PARTS = osp.join(HERE, "parts")
FINAL_OUT = osp.join(PRELIM, "results", "eval_version_a96137b_folds.csv")
PLAN = osp.join(HERE, "jobs", "device_plan.json")
A96 = "a96137b0f67dbb1f97cb1867c77849dd1bf5032f"
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SCRATCH = os.environ.get("A17_TMP", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_a96137b/tmp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=GRID_PARTS)
    ap.add_argument("--out", default=None, help="output CSV (smoke tests); the official file needs --final")
    ap.add_argument("--final", action="store_true", help=f"write {FINAL_OUT}")
    ap.add_argument("--allow_device_mismatch", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    if a.final and a.out:
        sys.exit("--final writes the official file; do not combine with --out")
    if a.final and osp.realpath(a.parts) != osp.realpath(GRID_PARTS):
        sys.exit("--final reads only the grid parts directory")
    plan = json.load(open(PLAN))["plan"]
    commits, bad, mism, n_tagged = {}, [], [], 0
    for fn in sorted(glob.glob(osp.join(a.parts, "*.json"))):
        m = json.load(open(fn))["meta"]
        stem = f"{m['method']}_{m['dataset']}_fold{m['fold']}"
        if m["run_id"] != stem or m.get("max_epochs_override") is not None:
            n_tagged += 1          # tagged smoke/timing parts: collect.py skips them too
            continue
        if m.get("ubgold_commit") != A96 or "ubgold_tree_sha256" not in m or m.get("hp", "ubgold") != "ubgold":
            bad.append(m["run_id"])
            continue
        dev = "+".join(sorted({"cuda" if "cuda" in s["device"] else "cpu" for s in m["segments"]}))
        want = plan.get(f"{m['method']} {m['dataset']}", {}).get("device")
        if dev != want:
            mism.append(f"{m['run_id']} ran on {dev}, planned {want}")
        commits[m["run_id"]] = m["ubgold_commit"]
    print(f"parts: {len(commits)} untagged a96137b runs, {n_tagged} tagged (ignored), {len(bad)} invalid, "
          f"{len(mism)} device mismatches")
    if bad:
        sys.exit(f"parts without the a96137b commit / tree digest or not 'ubgold': {bad}")
    if mism and not a.allow_device_mismatch:
        sys.exit("device mismatches (A17 pairs device types):\n  " + "\n  ".join(mism))
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="collect_", dir=SCRATCH)
    try:
        env = dict(os.environ, EP_PARTS=a.parts, EP_RES=tmp, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-B", COLLECT], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"harness/collect.py failed:\n{r.stderr[-4000:]}")
        df = pd.read_csv(osp.join(tmp, "eval_folds.csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)    # extra / coverage / repro files deleted unread
    n_all = len(df)
    n_dr = int((df.source != "evalpaper_harness").sum())
    df = df[df.source == "evalpaper_harness"].copy()
    df["ubgold_commit"] = df.run_id.map(commits)
    assert df.ubgold_commit.notna().all(), "row without an a96137b part"
    assert set(df.method) <= {"GLocalKD", "GLADC"} and set(df.dataset) <= set(UB9), (set(df.method), set(df.dataset))
    print(f"collect.py rows: {n_all}; dropped {n_dr} non-harness rows (c940457 deep_rerun); kept {len(df)}")
    cov = df[df.protocol == "oracle"].groupby(["method", "dataset"]).fold.nunique()
    for m in ("GLocalKD", "GLADC"):
        print(f"  {m:8s} folds per UB9 dataset: " + " ".join(f"{ds}:{int(cov.get((m, ds), 0))}" for ds in UB9))
    complete = all(int(cov.get((m, ds), 0)) == 5 for m in ("GLocalKD", "GLADC") for ds in UB9)
    print(f"  grid complete (90/90): {complete}; protocols: {sorted(df.protocol.unique())}")
    out = FINAL_OUT if a.final else a.out
    if out:
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows -> {out}")
    else:
        print("dry run: nothing written (use --final at the end of the grid, or --out for a smoke test)")


if __name__ == "__main__":
    main()
