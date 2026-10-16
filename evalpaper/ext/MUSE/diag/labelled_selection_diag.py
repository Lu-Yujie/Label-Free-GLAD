"""DIAGNOSTIC ONLY (investigation of the R2 reproduction misses; never a protocol row, never used for selection
in the study). Uses TEST labels.

On the stored error representations of a finished grid run (npz `rep`: every snapshot, every graph), it replays
MUSE's own detector stage and selection with src.py code, verbatim, with the TEST set standing in for MUSE's
labelled validation set:
  for every snapshot and every MLP cell (h, lr): MUSE_oneclass_classification.train_MLP(valid = test), i.e. the
  MLP is early-stopped (patience 10) on test AUROC and its best-test-AUROC state is scored (main.py lines 131-157).
Writes one JSON per run with the 10 x 9 matrix of (val=test) AUROCs; `diag_summary.py` applies main.py's
selection (best cell per fold/snapshot, snapshot by the mean over folds, lines 147-175).
  python -B diag/labelled_selection_diag.py MUSE_BZR_fold0
"""
import importlib.util
import json
import os
import os.path as osp
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

torch.set_num_threads(1)
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
spec = importlib.util.spec_from_file_location("rm", osp.join(HERE, "run_method.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
OUT = osp.join(HERE, "results", "diag")


def main(run_id):
    os.makedirs(OUT, exist_ok=True)
    J = json.load(open(osp.join(rm.PARTS, run_id + ".json")))
    Z = np.load(osp.join(rm.SCORES, run_id + ".npz"))
    m = J["meta"]
    G, S_ = rm.load_muse()["GNNs"], rm.load_muse()["src"]
    n_fit, n_hold, n_test = int(Z["n_fit"]), len(Z["hold_pos"]), len(Z["y_test"])
    n_all = n_fit + n_hold + n_test
    rep = Z["rep"].reshape(len(Z["epochs"]), n_all, 4)
    fit_idx = list(range(n_fit))
    test_idx = list(range(n_fit + n_hold, n_all))
    test_labels = list(1 - Z["y_test"])          # MUSE orientation: normal = 1
    seed = int(m["seed"])
    res = dict(run_id=run_id, dataset=m["dataset"], fold=m["fold"], hp=m["hp"], epochs=Z["epochs"].tolist(),
               cells=[f"h{h}_lr{l:g}" for h, l in rm.MLP_GRID], test_auroc=[], last_epoch_check=None)
    for e in range(len(Z["epochs"])):
        ae = S_.MUSE_oneclass_classification(model=None, feature_encoder=None, edge_encoder=None, datasets=None,
                                             device="cpu", labels=None, pos_weights=None, B_size=1)
        ae.TX = torch.tensor(rep[e], dtype=torch.float32)       # exactly the adapter's TX (float32 -> float64 -> float32)
        row = []
        for (h, l) in rm.MLP_GRID:
            torch.manual_seed(seed); torch.random.manual_seed(seed); np.random.seed(seed)
            mlp = G.AutoEncoderOneclassClassifier(in_dim=4, hid_dim=h, n_layers=3, drop_p=0.0)
            out = ae.train_MLP(encoder_param=None, classifier=mlp, train_idxs=fit_idx, valid_idxs=test_idx,
                               valid_labels=test_labels, test_idxs=test_idx, test_labels=test_labels, lr=l, epochs=500,
                               saving_interval=1, w_decay=1e-4, seed=seed)
            row.append(float(out[3]))
        res["test_auroc"].append(row)
    json.dump(res, open(osp.join(OUT, run_id + ".json"), "w"))
    print(run_id, "done", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
