"""Fold / label / feature identity: GLADMamba's own loaders (run_method.load_fold) vs the harness's UB-GOLD
loaders (harness/run_eval.py: load_ubgold_data_functions) on the same data. No model, no test metrics.
  python -B tests/check_folds.py DS [DS ...]  [--folds 0,1,2,3,4] [--no_xs]
"""
import importlib.util, json, os, os.path as osp, sys, time
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, torch
import run_method as R
spec = importlib.util.spec_from_file_location("harness_run_eval", osp.join(HERE, "..", "..", "harness", "run_eval.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
args_cli = [a for a in sys.argv[1:] if not a.startswith("--")]
folds = [0, 1, 2, 3, 4]
for a in sys.argv[1:]:
    if a.startswith("--folds="):
        folds = [int(x) for x in a.split("=")[1].split(",")]
no_xs = "--no_xs" in sys.argv
hns = H.load_ubgold_data_functions()
dfn = R.load_data_functions()
for DS in args_cli:
    t0 = time.time()
    args, _, _ = R.resolve_hparams(DS, "ubgold")
    for fold in folds:
        # GLADMamba side (asserts vs ubgold_splits inside)
        splits, dl, dlt, meta = R.load_fold(args, DS, fold, dfn)
        # harness side (UB-GOLD loader, cwd = UB-GOLD data dir)
        cwd = os.getcwd(); os.chdir(H.DATA_RUN)
        try:
            hs = hns["get_ad_split_TU"](args, fold=5)
            out = hns["get_ad_dataset_TU"](args, hs[fold], need_str_enc=not no_xs)
        finally:
            os.chdir(cwd)
        assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(splits, hs))
        ga, gb = list(dl.dataset), list(out[3].dataset)
        ta, tb = list(dlt.dataset), list(out[5].dataset)
        assert len(ga) == len(gb) and len(ta) == len(tb)
        assert meta["num_feat"] == out[6]["num_feat"] and meta["num_train"] == out[6]["num_train"]
        for A, B in ((ga, gb), (ta, tb)):
            for a, b in zip(A, B):
                assert int(a.y) == int(b.y)
                assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)
                if not no_xs:
                    assert torch.equal(a.x_s, b.x_s)
                assert a.XLX_f.shape[1] == a.x.shape[1] and a.XLX_s.shape[1] == a.x_s.shape[1]
        print(f"{DS} fold {fold}: OK  n_train={len(ga)} n_test={len(ta)} anomalies={sum(int(d.y) for d in ta)} "
              f"num_feat={meta['num_feat']} x_s={'checked' if not no_xs else 'skipped'}", flush=True)
    print(f"{DS}: {time.time()-t0:.0f}s", flush=True)
