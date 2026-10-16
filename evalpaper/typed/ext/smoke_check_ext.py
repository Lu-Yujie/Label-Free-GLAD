"""Phase B smoke / timing analysis. Prints NO metric value of any typed run (blinding; RUN_NOTES_ext §0.1 item 12).

Reads $GLAD_DATA_ROOT/typed/smoke_ext/parts (tagged runs only) and checks:
  1. automatic sanity per run: complete trajectory; every per-type AUROC/AUPRC/FPR95 finite and within [0, 1]
     (counts only); held-out signal finite (count); slice checks (MUSE / ProtoGLAD) as max |difference|;
  2. resume equivalence: tag 'rs' (straight) vs tag 'rr' (stopped at an intermediate epoch, then resumed): max |diff|
     over the trajectory (every numeric field except wall-clock times) and every per-graph score array;
  3. timings and peak memory -> typed/ext/runtime_smoke_ext.csv; from the 'time' runs -> typed/ext/runtime_est_ext.json
     (seconds per epoch / per evaluation point, setup, memory with margin) used by jobs/gen_tasks_ext.py;
  4. the collector on all smoke runs into a temp dir (counts only).
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B smoke_check_ext.py [--est]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import tempfile

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
SMOKE = osp.join(os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed")), "smoke_ext")
sys.path.insert(0, HERE)
import collect_typed_ext as CX  # noqa: E402

SKIP_KEYS = {"t", "timings", "seconds", "epoch_time_gladpro", "seconds_stage1", "seconds_stage2"}


def numeric_leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP_KEYS:
                continue
            yield from numeric_leaves(v, f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from numeric_leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def max_diff(a, b):
    la, lb = dict(numeric_leaves(a)), dict(numeric_leaves(b))
    if set(la) != set(lb):
        return float("inf"), f"field sets differ ({len(set(la) ^ set(lb))})"
    d = 0.0
    for k in la:
        x, y = la[k], lb[k]
        if np.isnan(x) and np.isnan(y):
            continue
        d = max(d, abs(x - y)) if np.isfinite(x - y) else float("inf")
    return d, ""


def sanity(J):
    traj = J["traj"]
    n_bad = n_nonfinite = n_vals = 0
    for r in traj:
        for k, t in r["types"].items():
            for m in ("auroc", "auprc", "fpr95"):
                v = t[m]
                n_vals += 1
                if v is None or not np.isfinite(v):
                    n_nonfinite += 1
                elif not (0.0 <= v <= 1.0):
                    n_bad += 1
    hold_ok = sum(1 for r in traj if r.get("holdout_score") is not None and np.isfinite(r["holdout_score"]))
    return dict(n_values=n_vals, n_nonfinite=n_nonfinite, n_out_of_range=n_bad, hold_finite=f"{hold_ok}/{len(traj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", action="store_true", help="write runtime_est_ext.json from the 'time' runs")
    a = ap.parse_args()
    parts = sorted(glob.glob(osp.join(SMOKE, "parts", "*.json")))
    rows, byid = [], {}
    for fn in parts:
        J = json.load(open(fn))
        m = J["meta"]
        byid[m["run_id"]] = (J, fn)
        try:
            CX.Run(J, "smoke").check_complete()
            complete = True
        except ValueError:
            complete = False
        s = sanity(J)
        fl = m.get("flags", {})
        rows.append(dict(run_id=m["run_id"], method=m["method"], generator=m["generator"], fv=m["feature_variant"],
                         tag=m.get("tag"), dry=bool(fl.get("dry_n")), complete=complete, n_eval=m["n_eval"],
                         num_epoch=m["hparams"]["num_epoch"], threads=m["threads"], device=m.get("device_all"),
                         seconds=round(m["seconds"], 1), sec_per_epoch_train=round(m["sec_per_epoch_train"], 3),
                         sec_per_eval=round(m["sec_per_eval"], 3), maxrss_gb=round(m["memory"]["maxrss_gb"], 2),
                         segments=len(m.get("segments", [])), slice_check=json.dumps(m.get("slice_check")), **s))
    df = pd.DataFrame(rows)
    out_csv = osp.join(HERE, "runtime_smoke_ext.csv")
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} smoke parts; all complete: {bool(df.complete.all()) if len(df) else None}; "
          f"non-finite per-type metric values: {int(df.n_nonfinite.sum()) if len(df) else 0}; "
          f"out of [0,1]: {int(df.n_out_of_range.sum()) if len(df) else 0} -> {out_csv}")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(df[["run_id", "complete", "n_eval", "threads", "seconds", "sec_per_epoch_train", "sec_per_eval",
                  "maxrss_gb", "n_nonfinite", "n_out_of_range", "hold_finite", "segments", "slice_check"]].to_string())
    # resume equivalence
    print("\nresume equivalence (tag rs = straight, rr = stop + resume):")
    for rid, (J, fn) in sorted(byid.items()):
        if not rid.endswith("_rr"):
            continue
        sid = rid[:-3] + "_rs"
        if sid not in byid:
            print(f"  {rid}: no straight partner")
            continue
        Js = byid[sid][0]
        d_traj, note = max_diff(J["traj"], Js["traj"])
        d_loss, _ = max_diff(J["train_loss"], Js["train_loss"])
        za, zb = np.load(J["meta"]["scores_npz"]), np.load(Js["meta"]["scores_npz"])
        d_npz = 0.0
        common = sorted(set(za.files) & set(zb.files))
        for k in common:
            x, y = za[k], zb[k]
            if x.dtype.kind in "fiu" and x.shape == y.shape:
                with np.errstate(invalid="ignore"):
                    diff = np.abs(x.astype(float) - y.astype(float))
                diff = diff[~(np.isnan(x.astype(float)) & np.isnan(y.astype(float)))]
                d_npz = max(d_npz, float(diff.max()) if diff.size else 0.0)
            elif x.dtype.kind in "fiu":
                d_npz = float("inf")
        segs = [s["start_epoch"] for s in J["meta"]["segments"]]
        print(f"  {rid}: segments start {segs}; max|diff| traj {d_traj:.3g}{(' ' + note) if note else ''}, "
              f"train_loss {d_loss:.3g}, npz {d_npz:.3g} over {len(common)} arrays")
    if a.est:
        est = {}
        t = df[(df.tag == "time") & df.complete & ~df.dry]
        for _, r in t.iterrows():
            setup = max(0.0, r.seconds - r.sec_per_epoch_train * r.num_epoch - r.sec_per_eval * r.n_eval)
            if r.method == "ProtoGLAD":   # one fit: the whole grid is the 'epoch'
                e = dict(threads=int(r.threads), sec_per_epoch=float(r.seconds), sec_per_eval=0.0, setup_s=0.0)
            else:
                e = dict(threads=int(r.threads), sec_per_epoch=float(r.sec_per_epoch_train),
                         sec_per_eval=float(r.sec_per_eval), setup_s=float(setup))
            e["mem_gb"] = float(np.ceil(1.3 * r.maxrss_gb + 2))
            est.setdefault(r.method, {})[r.generator] = e
        with open(osp.join(HERE, "runtime_est_ext.json"), "w") as f:
            json.dump(est, f, indent=1)
        print(f"\nruntime_est_ext.json: {sorted(est)}")
    tmp = tempfile.mkdtemp(prefix="smoke_cx_", dir=os.environ.get("TMPDIR"))
    p = subprocess.run([sys.executable, "-B", osp.join(HERE, "collect_typed_ext.py"), "--parts", osp.join(SMOKE, "parts"),
                        "--only-tag", "smoke", "--out", osp.join(tmp, "eval_typed_ext.csv")], capture_output=True, text=True)
    print("\ncollector on smoke runs:", (p.stdout.strip().splitlines() or ["(no output)"])[0], p.stderr[-500:])


if __name__ == "__main__":
    main()
