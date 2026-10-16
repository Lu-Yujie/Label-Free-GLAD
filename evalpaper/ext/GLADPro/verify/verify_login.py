"""Login-node verification of the GLADPro adapter (< 2 min, <= 4 cores per call).

  python -B verify/verify_login.py folds DS [DS ...]     # folds/features/held-out == harness (writes JSON)
  python -B verify/verify_login.py resume DS FOLD HP EPOCHS STOP   # resume-equivalence + determinism (CPU)

folds : for each fold, the adapter's load_fold() (the code path the runs use) is compared with
        (1) $GLAD_DATA_ROOT/ubgold_splits/<DS>.json (asserted inside load_fold),
        (2) every harness score file $GLAD_DATA_ROOT/evalpaper/scores/*_<DS>_fold<k>.npz
            (test_index, y_test, hold_pos, test/hold graph sizes),
        (3) the common fold cache (compared by verify/check_fold_cache.py with the grail-prelim env,
            from the JSON written here).
resume: run A = straight, run B = stop at epoch STOP (EP_TEST_STOP_EPOCH) then resume, run C = straight again;
        trajectories and per-graph scores must be bit-identical (A==B: resume; A==C: determinism).
"""
import glob
import json
import os
import os.path as osp
import subprocess
import sys
import time

import numpy as np

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_gladpro/smoke")
OUT = osp.join(HERE, "verify")


def folds(dsets):
    import run_method as RM
    H = RM.load_harness()
    res = {}
    for DS in dsets:
        for k in range(5):
            F = RM.load_fold(H, DS, k, 128, 9999, 0.2)
            hp = np.asarray(F["hold_pos"])
            rec = dict(n_fit=len(F["fit_list"]), n_hold=F["n_hold"], n_test=len(F["test_list"]),
                       n_anom=int(F["y_test"].sum()), num_feat=int(F["meta"]["num_feat"]),
                       x_dim=int(F["test_list"][0].x.shape[1]), harness_npz_checked=[])
            tn = np.array([int(d.num_nodes) for d in F["test_list"]])
            hn = np.array([int(d.num_nodes) for d in F["hold_list"]])
            for f in sorted(glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + f"/evalpaper/scores/*_{DS}_fold{k}*.npz"))):
                z = np.load(f)
                assert np.array_equal(z["test_index"], np.asarray(F["split_test_index"])), f
                assert np.array_equal(z["y_test"], F["y_test"]), f
                assert np.array_equal(z["hold_pos"], hp), f
                assert np.array_equal(z["test_n_nodes"], tn), f
                assert np.array_equal(z["hold_n_nodes"], hn), f
                rec["harness_npz_checked"].append(osp.basename(f))
            # original dataset indices of the training normals, held-out and fit graphs (for the cache check)
            tr = [i for i in F["split_train_index"]]
            rec.update(split_test_index=F["split_test_index"], y_test=F["y_test"].tolist(),
                       split_train_index=tr, hold_pos=F["hold_pos"])
            res[f"{DS}/{k}"] = rec
            print(DS, k, {kk: v for kk, v in rec.items() if kk not in ("split_test_index", "y_test",
                                                                      "split_train_index", "hold_pos")}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = osp.join(OUT, "folds_" + "_".join(dsets) + ".json")
    json.dump(res, open(path, "w"))
    print("wrote", path)


def run_one(DS, fold, hp, epochs, tag, stop=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", EP_PARTS=f"{SMOKE}/parts", EP_SCORES=f"{SMOKE}/scores",
               EP_CKPT=f"{SMOKE}/ckpt", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if stop:
        env["EP_TEST_STOP_EPOCH"] = str(stop)
    if os.environ.get("FORCE_DEL"):
        env["EP_TEST_FORCE_DELPROTO_EPOCH"] = os.environ["FORCE_DEL"]
    cmd = [sys.executable, "-B", osp.join(HERE, "run_method.py"), "--method", "GLADPro", "--dataset", DS,
           "--fold", str(fold), "--hp", hp, "--max_epochs", str(epochs), "--tag", tag, "--threads", "4"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr[-2000:]


def resume(DS, fold, hp, epochs, stop, phase="all"):
    """phase: all | A | B1 | B2 | C | compare (separate calls keep each login-node command < 2 min)."""
    rid = lambda t: f"GLADPro_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + f"_{t}"
    log = osp.join(OUT, f"resume_{DS}_fold{fold}_{hp}.log")
    if phase in ("all", "A"):
        for t in ("rsA", "rsB", "rsC"):
            for f in (f"{SMOKE}/parts/{rid(t)}.json", f"{SMOKE}/scores/{rid(t)}.npz", f"{SMOKE}/ckpt/{DS}/{rid(t)}.pt"):
                if osp.exists(f):
                    os.remove(f)
        open(log, "w").close()
    steps = {"A": ("rsA", None, 0), "B1": ("rsB", stop, 3), "B2": ("rsB", stop, 0), "C": ("rsC", None, 0)}
    for ph in (["A", "B1", "B2", "C"] if phase == "all" else ([phase] if phase in steps else [])):
        t0 = time.time()
        tag, st, want = steps[ph]
        rc, out = run_one(DS, fold, hp, epochs, tag, stop=st)
        ck = osp.exists(f"{SMOKE}/ckpt/{DS}/{rid('rsB')}.pt")
        ok = rc == want and (ph != "B1" or ck) and (ph != "B2" or "[resume]" in out)
        with open(log, "a") as f:
            f.write(f"{ph} rc={rc} want={want} ckpt_after={ck} resumed={'[resume]' in out} ok={ok} {time.time()-t0:.0f}s\n")
        print(open(log).read())
        assert ok, out[-3000:]
    if phase not in ("all", "compare"):
        return
    J = {t: json.load(open(f"{SMOKE}/parts/{rid(t)}.json")) for t in ("rsA", "rsB", "rsC")}
    Z = {t: np.load(f"{SMOKE}/scores/{rid(t)}.npz") for t in ("rsA", "rsB", "rsC")}
    keys = ["auroc", "auprc", "fpr95", "holdout_score", "train_loss", "n_prot", "auroc_gladpro"]
    out = {}
    for a, b in (("rsA", "rsB"), ("rsA", "rsC")):
        d_traj = max(abs(float(x[k]) - float(y[k])) if x[k] == x[k] else (0.0 if y[k] != y[k] else np.inf)
                     for x, y in zip(J[a]["traj"], J[b]["traj"]) for k in keys)
        d_sc = max(float(np.nanmax(np.abs(Z[a][k] - Z[b][k]))) for k in ("test", "hold"))
        same_nan = all(np.array_equal(np.isnan(Z[a][k]), np.isnan(Z[b][k])) for k in ("test", "hold"))
        out[f"{a}_vs_{b}"] = dict(max_abs_diff_traj=d_traj, max_abs_diff_scores=d_sc, same_nan_pattern=same_nan)
    nprot = [r["n_prot"] for r in J["rsA"]["traj"]]
    out["n_prot_trajectory_changes"] = [(r["epoch"], r["n_prot"]) for i, r in enumerate(J["rsA"]["traj"])
                                        if i == 0 or r["n_prot"] != nprot[i - 1]]
    out["segments_B"] = J["rsB"]["meta"]["segments"]
    out["final_auroc_A"] = J["rsA"]["traj"][-1]["auroc"]
    print(json.dumps(out, indent=1))
    json.dump(out, open(osp.join(OUT, f"resume_{DS}_fold{fold}_{hp}.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        folds(sys.argv[2:])
    elif sys.argv[1] == "resume":
        DS, fold, hp, ep, stop = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
        resume(DS, fold, hp, ep, stop, sys.argv[7] if len(sys.argv) > 7 else "all")
