"""Verification of the ProtoGLAD adapter (login node, < 2 min per call, <= 4 cores).

  python -B verify.py folds [DS ...]   # folds / held-out carve-out / test order == the harness's (via its score files)
  python -B verify.py pipeline [DS ...] # shortcut == from-scratch == original run_single_trial; determinism; threads

Results are appended to verify/verify.jsonl (one JSON line per check).
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]

import importlib.util
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
spec = importlib.util.spec_from_file_location("pg_run_method", osp.join(HERE, "run_method.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
OUT = osp.join(HERE, "verify", "verify.jsonl")
HARNESS_SCORES = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper/scores")
GLADPRO_SCORES = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_gladpro/scores")


def log(d):
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    d = dict(d, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.uname().nodename,
             adapter_sha256=R.sha256(osp.join(HERE, "run_method.py")))
    with open(OUT, "a") as f:
        f.write(json.dumps(d) + "\n")
    print(json.dumps(d), flush=True)


def check_folds(datasets):
    H = R.load_harness()
    for DS in datasets:
        for fold in range(5):
            F = R.load_fold(H, DS, fold)
            ref = (osp.join(GLADPRO_SCORES, f"GLADPro_MUTAG_fold{fold}.npz") if DS == "MUTAG"
                   else osp.join(HARNESS_SCORES, f"OCGIN_{DS}_fold{fold}_hpfree.npz"))
            Z = np.load(ref)
            ok = dict(
                test_index=bool(np.array_equal(Z["test_index"], np.asarray(F["split_test_index"]))),
                y_test=bool(np.array_equal(Z["y_test"], F["y_test"])),
                hold_pos=bool(np.array_equal(Z["hold_pos"], np.asarray(F["hold_pos"]))),
                test_n_nodes=bool(np.array_equal(Z["test_n_nodes"], [int(d.num_nodes) for d in F["test_list"]])),
                hold_n_nodes=bool(np.array_equal(Z["hold_n_nodes"], [int(d.num_nodes) for d in F["hold_list"]])),
                test_n_edges=bool(np.array_equal(Z["test_n_edges"],
                                                 [int(d.edge_index.size(1)) // 2 for d in F["test_list"]])))
            log(dict(check="folds", dataset=DS, fold=fold, reference=osp.basename(ref), n_fit=len(F["fit_list"]),
                     n_hold=F["n_hold"], n_test=len(F["test_list"]), all_ok=all(ok.values()), **ok))


def original_trial_scores(P, H, fit, test, y_test, h, psi1, psi2, tau, seed):
    """run_single_trial's body with the ORIGINAL ik_transform / compute_graph_embeddings / detector (scores kept)."""
    R.ubgold_set_seed(H)(seed)
    train_x_ik, test_x_ik = P.emb.ik_transform([x for _, x in fit], [x for _, x in test], psi=psi1)
    X_train, _ = P.emb.compute_graph_embeddings([a for a, _ in fit], train_x_ik, h)
    X_test, _ = P.emb.compute_graph_embeddings([a for a, _ in test], test_x_ik, h)
    det = P.det.ProtoGLADDetector(psi=psi2, tau=tau)
    det.fit(X_train, k_neighbors=P.cfg.DEFAULT_TOP_K)
    scores, _ = det.score(X_test)
    return np.asarray(-scores, dtype=np.float64), det.n_prototypes


def check_pipeline(datasets):
    H = R.load_harness()
    P = R.load_protoglad(with_run_detection=True)
    cells2 = [(2, 0.001), (32, 0.01), (64, 0.05)]
    for DS in datasets:
        for fold in (0, 1):
            seed = 3407 + fold
            F = R.load_fold(H, DS, fold)
            fit, hold, test, _ = R.fold_graphs(P, F, DS, "ubgold")
            y = F["y_test"]
            for psi1 in (4, 64):
                torch.set_num_threads(1)
                Xtr, Xte, Xho, _ = R.stage1(P, H, fit, hold, test, psi1, 2, seed)
                st = H.rng_state()
                for psi2, tau in cells2:
                    H.set_rng_state(st)
                    sc = R.stage2(P, H, Xtr, Xte, Xho, y, psi2, tau)          # shortcut (grid replay path)
                    # from scratch through the adapter (= tuned/hpfree part path), twice
                    a1 = R.stage2(P, H, *R.stage1(P, H, fit, hold, test, psi1, 2, seed)[:3], y, psi2, tau)
                    a2 = R.stage2(P, H, *R.stage1(P, H, fit, hold, test, psi1, 2, seed)[:3], y, psi2, tau)
                    # original code path (ik_transform / compute_graph_embeddings without the held-out graphs)
                    o_scores, o_np = original_trial_scores(P, H, fit, test, y, 2, psi1, psi2, tau, seed)
                    # run_detection.run_single_trial itself (verbatim), same seed
                    R.ubgold_set_seed(H)(seed)
                    fd = dict(train_adj=[a for a, _ in fit], test_adj=[a for a, _ in test],
                              train_x=[x for _, x in fit], test_x=[x for _, x in test],
                              train_labels=np.zeros(len(fit), dtype=np.int64), test_labels=y)
                    rs = P.rd.run_single_trial(fd, 2, psi1, psi2, tau)
                    # thread invariance
                    torch.set_num_threads(4)
                    t4 = R.stage2(P, H, *R.stage1(P, H, fit, hold, test, psi1, 2, seed)[:3], y, psi2, tau)
                    torch.set_num_threads(1)
                    ok = dict(
                        shortcut_eq_scratch=bool(np.array_equal(sc["test_scores"], a1["test_scores"])
                                                 and np.array_equal(sc["hold_scores"], a1["hold_scores"])),
                        deterministic=bool(np.array_equal(a1["test_scores"], a2["test_scores"])
                                           and np.array_equal(a1["hold_scores"], a2["hold_scores"])),
                        eq_original_code=bool(np.array_equal(a1["test_scores"], o_scores)
                                              and a1["n_prototypes"] == o_np),
                        eq_run_single_trial=bool(rs["auc"] == a1["auc"] and rs["n_prototypes"] == a1["n_prototypes"]),
                        threads_1_eq_4=bool(np.array_equal(a1["test_scores"], t4["test_scores"])
                                            and np.array_equal(a1["hold_scores"], t4["hold_scores"])))
                    log(dict(check="pipeline", dataset=DS, fold=fold, psi1=psi1, psi2=psi2, tau=tau,
                             auc=a1["auc"], n_prototypes=a1["n_prototypes"],
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             all_ok=all(ok.values()), **ok))


if __name__ == "__main__":
    what = sys.argv[1]
    ds = sys.argv[2:] or R.DATASETS
    torch.set_num_threads(1)
    {"folds": check_folds, "pipeline": check_pipeline}[what](ds)
