"""A17 smoke S1 + timing: automatic sanity checks of tagged smoke parts and the runtime/memory table.

  python -B tests/smoke_check.py --tags smoke,tcpu,tgpu [--out runtime_smoke.json]

Sanity (pass/fail only, no metric value printed): evaluation epochs = harness schedule, every test AUROC finite and in
[0, 1], every held-out mean finite, score matrices (n_eval x n_test / n_hold), meta has the a96137b commit, the patched
tree digest and code hashes, and the planned device (if any) equals the device used.
Timing (first segment only): s/epoch (training incl. process_graph), process_graph s/epoch, s/evaluation, setup, peak
host RSS and peak CUDA memory; full-run estimate = setup + num_epoch * s/epoch + n_eval * s/eval with the UB-GOLD
budget (GLADC 100 epochs / 25 evaluations, GLocalKD 150 / 38).
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os.path as osp  # noqa: E402

import numpy as np  # noqa: E402

SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_a96137b/smoke")
A96 = "a96137b0f67dbb1f97cb1867c77849dd1bf5032f"
FULL = {"GLADC": (100, 4), "GLocalKD": (150, 4)}


def n_evals(ne, ef):
    return len(set(range(ef, ne + 1, ef)) | {ne})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated run-id tags")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, allok = [], True
    for tag in a.tags.split(","):
        for fn in sorted(glob.glob(osp.join(SMOKE, "parts", f"*_{tag}.json"))):
            J = json.load(open(fn))
            m, t = J["meta"], J["traj"]
            h = m["hparams"]
            ne, ef = int(h["num_epoch"]), int(h["eval_freq"])
            sched = sorted(set(range(ef, ne + 1, ef)) | {ne})
            z = np.load(m["scores_npz"])
            chk = dict(schedule=[r["epoch"] for r in t] == sched,
                       auroc_finite_in_01=all(isinstance(r["auroc"], float) and math.isfinite(r["auroc"])
                                              and 0.0 <= r["auroc"] <= 1.0 for r in t),
                       holdout_finite=all(math.isfinite(r["holdout_score"]) for r in t),
                       scores_shape=z["test"].shape == (len(t), m["n_test"]) and z["hold"].shape == (len(t), m["n_holdout"]),
                       commit=m.get("ubgold_commit") == A96 and "ubgold_tree_sha256" in m
                       and "harness/run_eval.py" in m.get("code_sha256", {})
                       and "ubgold_a96137b/GAOOD/detector/mybase.py" in m.get("code_sha256", {}),
                       device=(m.get("a17", {}).get("planned_device") in (None, m["device"].split(":")[0])))
            ok = all(chk.values())
            allok &= ok
            seg0 = m["segments"][0]
            single = len(m["segments"]) == 1
            ep = m["epochs_this_segment"]
            tr, ev = m["seconds_train_this_segment"], m["seconds_eval_this_segment"]
            tm = m.get("a17_timing_this_segment", {})
            s_ep = tr / max(1, ep)
            s_ev = ev / max(1, len(t))
            setup = m["seconds"] - tr - ev
            fe, fef = FULL[m["method"]]
            est = setup + fe * s_ep + n_evals(fe, fef) * s_ev
            r = dict(run_id=m["run_id"], method=m["method"], dataset=m["dataset"], device=m["device"].split(":")[0],
                     threads=seg0.get("threads"), epochs=ep, n_eval=len(t), single_segment=single,
                     s_epoch=round(s_ep, 3), pg_s_epoch=round(tm.get("pg_train_s", float("nan")) / max(1, ep), 3),
                     s_eval=round(s_ev, 3), pg_s_eval=round(tm.get("pg_eval_s", float("nan")) / max(1, tm.get("evals", 1)), 3),
                     setup_s=round(setup, 1), maxrss_gb=round(m["memory"]["maxrss_gb"], 2),
                     cuda_max_alloc_gb=round(m["memory"].get("cuda_max_alloc_gb", 0.0), 2),
                     cuda_max_reserved_gb=round(m["memory"].get("cuda_max_reserved_gb", 0.0), 2),
                     est_full_h=round(est / 3600, 3) if single else None, sanity_ok=ok, **{"chk_" + k: v for k, v in chk.items()})
            rows.append(r)
            print(" ".join(f"{k}={v}" for k, v in r.items() if not k.startswith("chk_")) + ("" if ok else f" FAILED {chk}"))
    print(f"S1 sanity: {sum(r['sanity_ok'] for r in rows)}/{len(rows)} parts PASS")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    sys.exit(0 if allok and rows else 1)


if __name__ == "__main__":
    main()
