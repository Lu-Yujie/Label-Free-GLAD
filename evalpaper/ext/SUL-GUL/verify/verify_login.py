"""Login-node verification of the SUL/GUL adapter (< 2 min, <= 4 cores per call).

  python -B verify/verify_login.py folds DS [DS ...]               # folds/features/held-out == harness (JSON)
  python -B verify/verify_login.py resume DS FOLD HP EPOCHS STOP [PHASE]   # resume-equivalence + determinism (CPU)
  python -B verify/verify_login.py hpident DS FOLD EPOCHS          # ubgold vs hpfree short runs bit-identical
  python -B verify/verify_login.py batchinv DS FOLD EPOCHS         # score independent of batch composition

folds   : for each fold, the adapter's load_fold() (the code path the runs use) is compared with
          (1) $GLAD_DATA_ROOT/ubgold_splits/<DS>.json (asserted inside load_fold),
          (2) every harness score file $GLAD_DATA_ROOT/evalpaper/scores/*_<DS>_fold<k>.npz and every GLADPro
              port score file $GLAD_DATA_ROOT/ext_gladpro/scores/*_<DS>_fold<k>*.npz
              (test_index, y_test, hold_pos, test/hold graph sizes),
          (3) the common fold cache (verify/check_fold_cache.py, grail-prelim env, from the JSON written here).
resume  : run A = straight, run B = stop at epoch STOP (EP_TEST_STOP_EPOCH) then resume, run C = straight again;
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
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_sulgul/smoke")
OUT = osp.join(HERE, "verify")
KEYS = ["auroc", "auprc", "fpr95", "holdout_score", "holdout_obj", "train_loss", "auroc_gul", "n_anchor"]


def folds(dsets):
    import run_method as RM
    H = RM.load_harness()
    res = {}
    for DS in dsets:
        for k in range(5):
            F = RM.load_fold(H, DS, k, 256, 9999, 0.2)
            hp = np.asarray(F["hold_pos"])
            rec = dict(n_fit=len(F["fit_list"]), n_hold=F["n_hold"], n_test=len(F["test_list"]),
                       n_anom=int(F["y_test"].sum()), num_feat=int(F["meta"]["num_feat"]),
                       x_dim=int(F["test_list"][0].x.shape[1]), harness_npz_checked=[])
            tn = np.array([int(d.num_nodes) for d in F["test_list"]])
            hn = np.array([int(d.num_nodes) for d in F["hold_list"]])
            files = sorted(glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + f"/evalpaper/scores/*_{DS}_fold{k}.npz"))
                           + glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + f"/evalpaper/scores/*_{DS}_fold{k}_hpfree.npz"))
                           + glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + f"/ext_gladpro/scores/GLADPro_{DS}_fold{k}.npz")))
            for f in files:
                z = np.load(f)
                assert np.array_equal(z["test_index"], np.asarray(F["split_test_index"])), f
                assert np.array_equal(z["y_test"], F["y_test"]), f
                assert np.array_equal(z["hold_pos"], hp), f
                assert np.array_equal(z["test_n_nodes"], tn), f
                assert np.array_equal(z["hold_n_nodes"], hn), f
                rec["harness_npz_checked"].append(osp.basename(f))
            assert rec["x_dim"] == rec["num_feat"], rec
            rec.update(split_test_index=F["split_test_index"], y_test=F["y_test"].tolist(),
                       split_train_index=F["split_train_index"], hold_pos=F["hold_pos"])
            res[f"{DS}/{k}"] = rec
            print(DS, k, {kk: (v if kk != "harness_npz_checked" else len(v)) for kk, v in rec.items()
                          if kk not in ("split_test_index", "y_test", "split_train_index", "hold_pos")}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = osp.join(OUT, "folds_" + "_".join(dsets) + ".json")
    json.dump(res, open(path, "w"))
    print("wrote", path)


def run_one(DS, fold, hp, epochs, tag, stop=None, threads=4):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", EP_PARTS=f"{SMOKE}/parts", EP_SCORES=f"{SMOKE}/scores",
               EP_CKPT=f"{SMOKE}/ckpt", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    if stop:
        env["EP_TEST_STOP_EPOCH"] = str(stop)
    cmd = [sys.executable, "-B", osp.join(HERE, "run_method.py"), "--method", "SUL-GUL", "--dataset", DS,
           "--fold", str(fold), "--hp", hp, "--max_epochs", str(epochs), "--tag", tag, "--threads", str(threads)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd="/tmp")
    return p.returncode, p.stdout + p.stderr[-2000:]


def rid(DS, fold, hp, t):
    return f"SUL-GUL_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + f"_{t}"


def clean(DS, r):
    for f in (f"{SMOKE}/parts/{r}.json", f"{SMOKE}/scores/{r}.npz", f"{SMOKE}/ckpt/{DS}/{r}.pt"):
        if osp.exists(f):
            os.remove(f)


def compare(ja, jb, za, zb):
    d_traj = 0.0
    for x, y in zip(ja["traj"], jb["traj"]):
        for k in KEYS:
            a, b = float(x[k]), float(y[k])
            d_traj = max(d_traj, 0.0 if (a == b or (a != a and b != b)) else abs(a - b) if a == a and b == b else np.inf)
    d_sc = max(float(np.nanmax(np.abs(za[k] - zb[k]))) for k in ("test", "hold"))
    same_nan = all(np.array_equal(np.isnan(za[k]), np.isnan(zb[k])) for k in ("test", "hold"))
    return dict(n_points=(len(ja["traj"]), len(jb["traj"])), max_abs_diff_traj=d_traj, max_abs_diff_scores=d_sc,
                same_nan_pattern=same_nan,
                bit_identical=bool(d_traj == 0.0 and d_sc == 0.0 and same_nan and len(ja["traj"]) == len(jb["traj"])))


def resume(DS, fold, hp, epochs, stop, phase="all"):
    """phase: all | A | B1 | B2 | C | compare (separate calls keep each login-node command < 2 min)."""
    R = lambda t: rid(DS, fold, hp, t)
    log = osp.join(OUT, f"resume_{DS}_fold{fold}_{hp}.log")
    if phase in ("all", "A"):
        for t in ("rsA", "rsB", "rsC"):
            clean(DS, R(t))
        open(log, "w").close()
    steps = {"A": ("rsA", None, 0), "B1": ("rsB", stop, 3), "B2": ("rsB", stop, 0), "C": ("rsC", None, 0)}
    for ph in (["A", "B1", "B2", "C"] if phase == "all" else ([phase] if phase in steps else [])):
        t0 = time.time()
        tag, st, want = steps[ph]
        rc, out = run_one(DS, fold, hp, epochs, tag, stop=st)
        ck = osp.exists(f"{SMOKE}/ckpt/{DS}/{R('rsB')}.pt")
        ok = rc == want and (ph != "B1" or ck) and (ph != "B2" or "[resume]" in out)
        with open(log, "a") as f:
            f.write(f"{ph} rc={rc} want={want} ckpt_after={ck} resumed={'[resume]' in out} ok={ok} "
                    f"{time.time()-t0:.0f}s\n")
        print(open(log).read())
        assert ok, out[-3000:]
    if phase not in ("all", "compare"):
        return
    J = {t: json.load(open(f"{SMOKE}/parts/{R(t)}.json")) for t in ("rsA", "rsB", "rsC")}
    Z = {t: np.load(f"{SMOKE}/scores/{R(t)}.npz") for t in ("rsA", "rsB", "rsC")}
    out = {f"{a}_vs_{b}": compare(J[a], J[b], Z[a], Z[b]) for a, b in (("rsA", "rsB"), ("rsA", "rsC"))}
    out["segments_B"] = J["rsB"]["meta"]["segments"]
    out["anchor_updates_A"] = sum(bool(r["anchor_update"]) for r in J["rsA"]["traj"])
    out["final_auroc_A"] = J["rsA"]["traj"][-1]["auroc"]
    out["topk_effective"] = J["rsA"]["meta"]["hparams"]["topk_effective"]
    print(json.dumps(out, indent=1))
    json.dump(out, open(osp.join(OUT, f"resume_{DS}_fold{fold}_{hp}.json"), "w"), indent=1)


def hpident(DS, fold, epochs):
    a, b = rid(DS, fold, "ubgold", "hpid"), rid(DS, fold, "hpfree", "hpid")
    clean(DS, a), clean(DS, b)
    for hp in ("ubgold", "hpfree"):
        rc, out = run_one(DS, fold, hp, epochs, "hpid")
        assert rc == 0, out[-3000:]
    ja, jb = (json.load(open(f"{SMOKE}/parts/{r}.json")) for r in (a, b))
    za, zb = (np.load(f"{SMOKE}/scores/{r}.npz") for r in (a, b))
    out = compare(ja, jb, za, zb)
    ha = {k: v for k, v in ja["meta"]["hparams"].items()}
    hb = {k: v for k, v in jb["meta"]["hparams"].items()}
    out["hparams_identical"] = ha == hb
    out["hparams"] = ha
    print(json.dumps(out, indent=1))
    json.dump(out, open(osp.join(OUT, f"hpident_{DS}_fold{fold}.json"), "w"), indent=1)


def batchinv(DS, fold, epochs):
    """Train `epochs` epochs (in-process, verbatim training step), then score the held-out and test graphs (eval
    mode) (a) all in one batch, (b) one graph at a time, (c) in reversed chunks of 7; report the max difference."""
    import random
    import torch
    import run_method as RM
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    H = RM.load_harness()
    utils, layers, model_mod = RM.load_gul()
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn.models.autoencoder import GAE
    args, _, _ = RM.resolve_hparams(utils, DS, "ubgold")
    F = RM.load_fold(H, DS, fold, args.batch_size, args.batch_size_test)
    tr = F["fit_list"]
    topk = min(args.topk, len(tr))
    model = GAE(model_mod.U_GLAD(args=args, input_dim=tr[0].num_node_features))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    anchors = random.sample(tr, topk)
    ba = next(iter(DataLoader(anchors, batch_size=args.batch_size_test, shuffle=False)))
    for ep in range(epochs):
        model.train()
        for b in DataLoader(tr, batch_size=args.batch_size, shuffle=True):
            opt.zero_grad()
            loss, var, ne = model.encode(b, ba)
            loss = loss + args.w_g * model.recon_loss(ne, b.edge_index)
            loss.backward()
            opt.step()
    model.eval()
    res = {}
    with torch.no_grad():
        for name, L in (("hold", F["hold_list"]), ("test", F["test_list"])):
            one = model.encode(next(iter(DataLoader(L, batch_size=9999))), ba)[1].numpy()
            single = np.concatenate([model.encode(next(iter(DataLoader([g], batch_size=1))), ba)[1].numpy() for g in L])
            rev = L[::-1]
            ch = np.concatenate([model.encode(next(iter(DataLoader(rev[i:i + 7], batch_size=7))), ba)[1].numpy()
                                 for i in range(0, len(rev), 7)])[::-1]
            # anchors in a different order / other anchors must not change log_var either
            ba2 = next(iter(DataLoader(anchors[::-1][: max(1, topk // 2)], batch_size=9999)))
            other_anchor = model.encode(next(iter(DataLoader(L, batch_size=9999))), ba2)[1].numpy()
            res[name] = dict(n=len(L), max_abs_one_vs_single=float(np.max(np.abs(one - single))),
                             max_abs_one_vs_rev7=float(np.max(np.abs(one - ch))),
                             max_abs_one_vs_other_anchors=float(np.max(np.abs(one - other_anchor))),
                             max_rel_one_vs_single=float(np.max(np.abs(one - single) / np.abs(one))),
                             mean_score=float(one.mean()))
    out = dict(dataset=DS, fold=fold, epochs_trained=epochs, **res)
    print(json.dumps(out, indent=1))
    json.dump(out, open(osp.join(OUT, f"batchinv_{DS}_fold{fold}.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "folds":
        folds(sys.argv[2:])
    elif sys.argv[1] == "resume":
        DS, fold, hp, ep, stop = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
        resume(DS, fold, hp, ep, stop, sys.argv[7] if len(sys.argv) > 7 else "all")
    elif sys.argv[1] == "hpident":
        hpident(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "batchinv":
        batchinv(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
