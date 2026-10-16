"""CPU-vs-GPU comparison (NOTES.md "Device and determinism"): re-run grid cells on the GPU (the released code's
device, no D1 needed) and compare with the CPU grid-replay units (same seeds, same cells).

  python -B gpu_equiv.py --datasets MUTAG BZR COX2 DHFR --folds 0 1 2 3 4 --psi1 4 64 --minutes 20

Appends one JSON line per (dataset, fold, psi1) to verify/gpu_equiv.jsonl.
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]

import argparse
import importlib.util
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = osp.dirname(osp.abspath(__file__))
spec = importlib.util.spec_from_file_location("pg_run_method", osp.join(HERE, "run_method.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
OUT = osp.join(HERE, "verify", "gpu_equiv.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MUTAG", "BZR", "COX2", "DHFR"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--psi1", nargs="+", type=int, default=[4, 64])
    ap.add_argument("--minutes", type=float, default=20)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU required"
    torch.set_num_threads(1)
    t_end = time.time() + 60 * a.minutes
    H = R.load_harness()
    P = R.load_protoglad()
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    for DS in a.datasets:
        for fold in a.folds:
            F = R.load_fold(H, DS, fold)
            fit, hold, test, _ = R.fold_graphs(P, F, DS, "ubgold")
            y = F["y_test"]
            for psi1 in a.psi1:
                if time.time() > t_end:
                    raise SystemExit("time budget reached")
                unit = R.grid_unit_paths(DS, fold, psi1, "ubgold")[1]
                if not osp.exists(unit):
                    continue
                Z = np.load(unit)
                t0 = time.time()
                Xtr, Xte, Xho, _ = R.stage1(P, H, fit, hold, test, psi1, 2, 3407 + fold)
                st = H.rng_state()
                d_score, d_auc, same_np, bit_eq, n = [], [], [], [], 0
                for j, (h, p1, psi2, tau) in enumerate(Z["cells"].tolist()):
                    H.set_rng_state(st)
                    res = R.stage2(P, H, Xtr, Xte, Xho, y, int(psi2), float(tau))
                    cpu = Z["test"][j]
                    d_score.append(float(np.max(np.abs(res["test_scores"] - cpu))))
                    d_auc.append(float(res["auc"] - roc_auc_score(y, cpu)))
                    bit_eq.append(bool(np.array_equal(res["test_scores"], cpu)))
                    n += 1
                rec = dict(dataset=DS, fold=fold, psi1=psi1, cells=n, bit_identical_cells=int(sum(bit_eq)),
                           max_abs_score_diff=max(d_score), max_abs_auroc_diff=float(np.max(np.abs(d_auc))),
                           mean_auroc_diff=float(np.mean(d_auc)), seconds_gpu=time.time() - t0,
                           gpu=torch.cuda.get_device_name(0), job=os.environ.get("SLURM_JOB_ID"),
                           time=time.strftime("%Y-%m-%d %H:%M:%S"))
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
