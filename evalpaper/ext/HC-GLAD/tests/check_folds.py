"""Fold / label / feature identity: HC-GLAD's own loaders (run_method.load_fold) vs the harness's UB-GOLD loaders
(harness/run_eval.py: load_ubgold_data_functions) on the same data. No model, no test metrics.
For REDDIT-BINARY run_method uses the hyperedge cache; with --check_hyper the cache is compared with the verbatim
get_hyperedge_index on the first K graphs (--hyper_k=K, default all).
  python -B tests/check_folds.py DS [DS ...]  [--folds=0,1,2,3,4] [--hp=ubgold] [--check_hyper] [--hyper_k=K]
"""
import importlib.util, json, os, os.path as osp, sys, time
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, torch
import run_method as R
spec = importlib.util.spec_from_file_location("harness_run_eval", osp.join(HERE, "..", "..", "harness", "run_eval.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
args_cli = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
folds = [int(x) for x in opts.get("folds", "0,1,2,3,4").split(",")]
hp = opts.get("hp", "ubgold")
hns = H.load_ubgold_data_functions()
for DS in args_cli:
    t0 = time.time()
    dfn = R.load_data_functions(DS)
    args, _, _ = R.resolve_hparams(DS, hp)
    for fold in folds:
        splits, dl, dlt, meta = R.load_fold(args, DS, fold, dfn)       # asserts vs ubgold_splits inside
        cwd = os.getcwd(); os.chdir(H.DATA_RUN)
        try:
            hs = hns["get_ad_split_TU"](args, fold=5)
            out = hns["get_ad_dataset_TU"](args, hs[fold])
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
                assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index) and torch.equal(a.x_s, b.x_s)
                assert a.hyperedge_index.dtype == torch.int64 and int(a.hyperedge_index[0].max()) < a.num_nodes
        print(f"{DS} fold {fold} hp {hp}: OK  n_train={len(ga)} n_test={len(ta)} anomalies={sum(int(d.y) for d in ta)} "
              f"num_feat={meta['num_feat']} x/edge_index/x_s/y identical to the UB-GOLD loader", flush=True)
    if "check_hyper" in opts and DS in R.HYPER_CACHED_DS:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.transforms import Constant
        ds = TUDataset(osp.join(R.DATA_ROOT, "data", DS), name=DS, transform=Constant(1, cat=False))
        k = int(opts.get("hyper_k", len(ds)))
        ref = dfn["get_hyperedge_index_orig"]([ds[i] for i in range(k)], None)
        C = torch.load(osp.join(R.HYPER_CACHE, f"{DS}.pt"), weights_only=False)
        assert all(torch.equal(ref[i].hyperedge_index, C["hyperedge_index"][i]) for i in range(k))
        print(f"{DS}: hyperedge cache == verbatim get_hyperedge_index on the first {k} graphs", flush=True)
    print(f"{DS}: {time.time()-t0:.0f}s", flush=True)
