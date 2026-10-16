"""UB-GOLD folds exactly as the evalpaper harness builds them (evalpaper/harness/run_eval.py _run()).

The harness module is imported (not copied), so the split code, the verified-split assertions, the trial
seed and the 20% held-out-normal carve-out are the harness's own. Graphs are returned as dicts with the
UB-GOLD node features (x, float32) and the PyG edge_index (no self-loops; LGKDE adds them itself).
"""
import hashlib
import importlib.util
import json
import os
import os.path as osp

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
HARNESS = osp.normpath(osp.join(HERE, "..", "..", "harness"))
RUN_EVAL = osp.join(HARNESS, "run_eval.py")


def load_harness():
    spec = importlib.util.spec_from_file_location("evalpaper_run_eval", RUN_EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def harness_sha256():
    return hashlib.sha256(open(RUN_EVAL, "rb").read()).hexdigest()


class _Args:
    pass


def load_fold(DS, fold, holdout_frac=0.2, H=None):
    """Returns dict(fit, hold, test: lists of graph dicts; y_test; split_test_index; hold_pos; n_all; seed).
    Same steps as harness _run(): set_seed(3407+fold) -> get_ad_split_TU / get_ad_dataset_TU (cwd = the
    UB-GOLD data dir) -> assert against ubgold_splits -> RandomState(2026+1000*fold) 20% hold-out."""
    H = H or load_harness()
    ns = H.load_ubgold_data_functions()
    assert not DS.startswith("Tox21"), "Tox21 not supported by this adapter (optional for E5)"
    args = _Args()
    args.DS = DS
    args.batch_size = 1 << 30         # loaders are rebuilt below; UB-GOLD's DataLoaders are never iterated
    args.batch_size_test = 1 << 30
    args.num_trial = 5
    seed = 3407 + fold
    ns["set_seed"](seed)
    cwd = os.getcwd()
    os.chdir(H.DATA_RUN)
    try:
        splits = ns["get_ad_split_TU"](args, fold=args.num_trial)
        out = ns["get_ad_dataset_TU"](args, splits[fold], need_str_enc=False)
    finally:
        os.chdir(cwd)
    (dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta) = out
    ref = json.load(open(osp.join(H.SPLITS, f"{DS}.json")))
    rf = ref["folds"][fold]
    assert list(map(int, splits[fold][0])) == list(map(int, rf["split_train_index"]))
    assert list(map(int, splits[fold][1])) == list(map(int, rf["split_test_index"]))
    train_all = list(dataloader.dataset)
    n_all = len(train_all)
    n_hold = int(round(holdout_frac * n_all))
    perm = np.random.RandomState(H.HOLDOUT_SEED + 1000 * fold).permutation(n_all)
    hold_pos = sorted(perm[:n_hold].tolist())
    fit_pos = sorted(perm[n_hold:].tolist())
    assert n_hold > 0
    test_data = list(dataloader_test.dataset)
    y_test = np.array([int(d.y) for d in test_data], dtype=np.int64)

    def conv(d):
        x = d.x
        if x is None:
            x = torch.ones((d.num_nodes, 1), dtype=torch.float32)
        return dict(x=x.to(torch.float32).contiguous(), edge_index=d.edge_index.long().contiguous(),
                    n=int(d.num_nodes))

    return dict(fit=[conv(train_all[i]) for i in fit_pos], hold=[conv(train_all[i]) for i in hold_pos],
                test=[conv(d) for d in test_data], y_test=y_test,
                split_test_index=np.asarray(list(map(int, splits[fold][1])), dtype=np.int64),
                hold_pos=np.asarray(hold_pos, dtype=np.int64), n_all=n_all, seed=seed,
                num_feat=int(meta["num_feat"]), ubgold_meta=meta)
