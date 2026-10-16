"""Run UB-GOLD's released KernelGLAD pipeline verbatim (WL-IF, `benchmark/Source/WL_IF.sh` lines).

For every dataset and fold this executes, unchanged:
  * mymain.set_seed(3407 + i), get_ad_split_TU / get_ad_dataset_TU (or get_ad_dataset_Tox21),
  * benchmark/utils.init_model (KernelGLAD branch) and the KernelGLAD class from
    GAOOD/detector/KernelGLAD.py (extracted with `ast`, exec'd unchanged),
  * model.fit(dataset=dataset_train, ..., dataloader=dataloader, dataloader_val=dataloader_val)
    and model.predict(dataset=dataset_test, dataloader=dataloader_test) exactly as mymain.main(),
  * UB-GOLD's metric functions (GAOOD/metric/metric.py).
CHANGES (documented in NOTES.md): `build_save_path` is replaced by a no-op (it creates and empties
model_save/ directories inside the source tree); the structural encodings are skipped
(need_str_enc=False; KernelGLAD never reads them; computing them consumes no RNG).

Instrumentation: dataloader_val is wrapped in a pass-through proxy that records, in iteration
order, each graph's label and a structural fingerprint. KernelGLAD.fit iterates it exactly once
(in to_grakel_dataset), so the recorded labels are the labels aligned with UB-GOLD's scores.
We report, per fold:
  * auroc_released: UB-GOLD's own pairing (scores from fit()'s pass, labels from predict()'s
    second pass over the shuffled test loader);
  * auroc_aligned: the same scores paired with the recorded aligned labels;
  * frac_label_mismatch: share of positions where the two label sequences differ;
  * kernel_maxabs_vs_ours: max |K_ubgold - K_ours| after mapping UB-GOLD's graph order onto the
    cache order by fingerprint (K_ours = kglad.py test-mode WL kernel, WL_iter = 5).

  python -B ubgold_verbatim_check.py [DS ...]   -> results/eval_kernels_ubgold_verbatim.csv
"""
import ast
import csv
import hashlib
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
DEEP = osp.join(PRELIM, "deep_rerun")
SRC = osp.join(DEEP, "ubgold_src")
sys.path.insert(0, HERE)
sys.path.insert(0, DEEP)
sys.path.insert(0, PRELIM)

import torch  # noqa: E402
import run_rerun as RR  # noqa: E402  (read-only reuse of the verified UB-GOLD extraction helpers)
import kglad  # noqa: E402
from kglad_data import load_folds  # noqa: E402

OUT = osp.join(PRELIM, "results", "eval_kernels_ubgold_verbatim.csv")
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY",
            "REDDIT-BINARY", "DD", "COLLAB", "Tox21_p53", "Tox21_MMP"]


def load_kernelglad_class():
    from sklearn.metrics import auc, roc_curve
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from grakel.kernels import WeisfeilerLehman, VertexHistogram, Propagation, ShortestPath, PropagationAttr

    class DeepDetector:  # stub base: KernelGLAD uses none of DeepDetector's machinery
        def __init__(self, in_dim=None, **kwargs):
            self.in_dim = in_dim

    path = osp.join(SRC, "GAOOD", "detector", "KernelGLAD.py")
    tree = ast.parse(open(path).read())
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KernelGLAD"]
    ns = dict(np=np, torch=torch, os=os, time=time, auc=auc, roc_curve=roc_curve,
              IsolationForest=IsolationForest, OneClassSVM=OneClassSVM, LocalOutlierFactor=LocalOutlierFactor,
              WeisfeilerLehman=WeisfeilerLehman, VertexHistogram=VertexHistogram, Propagation=Propagation,
              ShortestPath=ShortestPath, PropagationAttr=PropagationAttr, DeepDetector=DeepDetector)
    exec(compile(ast.Module(body=cls, type_ignores=[]), path, "exec"), ns)
    K = ns["KernelGLAD"]
    K.build_save_path = lambda self: None  # CHANGE: no directory creation/deletion in the source tree
    upath = osp.join(SRC, "benchmark", "utils.py")
    utree = ast.parse(open(upath).read())
    fn = [n for n in utree.body if isinstance(n, ast.FunctionDef) and n.name == "init_model"]
    uns = {"KernelGLAD": K}
    exec(compile(ast.Module(body=fn, type_ignores=[]), upath, "exec"), uns)
    return K, uns["init_model"]


def load_tox21_functions(ns):
    import networkx as nx
    import re
    from torch_geometric.utils import from_networkx
    ns.update(nx=nx, re=re, from_networkx=from_networkx)
    RR.extract_functions(osp.join(SRC, "dataloader", "data_loader.py"),
                         ["read_graph_file", "get_ad_dataset_Tox21"], ns)


class Recorder:
    """Pass-through iterable over a DataLoader that records what each pass yields."""

    def __init__(self, loader):
        self.loader = loader
        self.passes = []

    def __iter__(self):
        rec = []
        self.passes.append(rec)
        for data in self.loader:
            rec.append((int(data.y.item()), fp_pyg(data)))
            yield data

    def __len__(self):
        return len(self.loader)


def fp_edges(n, undirected_pairs, labels):
    e = sorted(set((min(i, j), max(i, j)) for i, j in undirected_pairs))
    h = hashlib.sha1()
    h.update(str(n).encode())
    h.update(np.asarray(e, dtype=np.int64).tobytes())
    h.update(np.asarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()


def fp_pyg(data):
    ei = data.edge_index.numpy().T
    lab = data.x.argmax(1).numpy() if data.x is not None else np.zeros(data.num_nodes, dtype=np.int64)
    return fp_edges(int(data.num_nodes), [tuple(p) for p in ei], lab)


def fp_cache(g):
    lab = g["node_label"] if g["node_label"] is not None else np.zeros(g["n"], dtype=np.int64)
    return fp_edges(int(g["n"]), [tuple(p) for p in np.asarray(g["edges"])], lab)


def main():
    names = sys.argv[1:] or DATASETS
    KernelGLAD, init_model = load_kernelglad_class()
    ns = RR.load_ubgold_data_functions()
    metric = __import__("importlib").import_module
    RR.stub_package("GAOOD", osp.join(SRC, "GAOOD"))
    RR.stub_package("GAOOD.metric", osp.join(SRC, "GAOOD", "metric"))
    M = metric("GAOOD.metric.metric")
    rows = []
    if osp.exists(OUT):
        rows = list(csv.DictReader(open(OUT)))
    done = {(r["dataset"], r["fold"]) for r in rows}
    cwd = os.getcwd()
    os.chdir(RR.DATA_RUN)  # UB-GOLD reads ./data/<DS>
    try:
        for DS in names:
            if all((DS, str(i)) in done for i in range(5)):
                continue
            args, line = RR.ubgold_hparams("WL_IF", DS)
            tox = DS.startswith("Tox21")
            if tox and "get_ad_dataset_Tox21" not in ns:
                load_tox21_functions(ns)
            cache = load_folds(DS)
            for i in range(args.num_trial):
                if (DS, str(i)) in done:
                    continue
                t0 = time.time()
                ns["set_seed"](3407 + i)
                if tox:
                    out = ns["get_ad_dataset_Tox21"](args, need_str_enc=False)
                else:
                    splits = ns["get_ad_split_TU"](args, fold=args.num_trial)
                    out = ns["get_ad_dataset_TU"](args, splits[i], need_str_enc=False)
                dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = out
                args.max_nodes_num = meta["max_nodes_num"]
                args.dataset_num_features = meta["num_feat"]
                args.n_train = meta["num_train"]
                args.n_edge_feat = meta["num_edge_feat"]
                model = init_model(args)
                rec_val = Recorder(dataloader_val)
                model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=rec_val)
                rec_test = Recorder(dataloader_test)
                score, y_all = model.predict(dataset=dataset_test, dataloader=rec_test, args=args, return_score=False)
                y_rel = np.asarray(y_all)
                y_fit = np.asarray([y for y, _ in rec_val.passes[0]])
                assert len(rec_val.passes) == 1 and len(rec_test.passes) == 1
                assert np.array_equal(y_rel, np.asarray([y for y, _ in rec_test.passes[0]]))
                s = np.asarray(score)
                # equivalence with our grakel input (cache order, test-mode WL, WL_iter = 5)
                cf = cache[i]
                pos = {}
                for j, g in enumerate(cf["test"]):
                    pos.setdefault(fp_cache(g), []).append(j)
                order = []
                used = {}
                for _, fp in rec_val.passes[0]:
                    k = used.get(fp, 0)
                    order.append(pos[fp][k] if fp in pos and k < len(pos[fp]) else -1)
                    used[fp] = k + 1
                order = np.asarray(order)
                maxabs = float("nan")
                y_cache_match = False
                if (order >= 0).all() and len(order) == len(cf["test"]):
                    teg = kglad.to_grakel(cf["test"])
                    Kc, _, _, _ = kglad.kernel_matrices("WL", int(args.WL_iter), None, teg, "test", 3407 + i)
                    maxabs = float(np.max(np.abs(Kc[np.ix_(order, order)] - model.kernel_matrix)))
                    y_cache_match = bool(np.array_equal(np.asarray(cf["y_test"])[order], y_fit))
                r = {"dataset": DS, "fold": i, "method": "WL-IF", "ubgold_line": line,
                     "n_test": len(s), "n_anom": int(y_fit.sum()),
                     "auroc_released": 100 * M.ood_auc(y_rel, s), "auroc_aligned": 100 * M.ood_auc(y_fit, s),
                     "auprc_released": 100 * M.ood_aupr(y_rel, s), "auprc_aligned": 100 * M.ood_aupr(y_fit, s),
                     "fpr95_released": 100 * M.fpr95(y_rel, s), "fpr95_aligned": 100 * M.fpr95(y_fit, s),
                     "frac_label_mismatch": float(np.mean(y_rel != y_fit)),
                     "kernel_maxabs_vs_ours": maxabs, "labels_match_cache": y_cache_match,
                     "seconds": round(time.time() - t0, 1)}
                rows.append({k: str(v) for k, v in r.items()})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "ubgold_line"},
                      flush=True)
                with open(OUT + ".tmp", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                    w.writeheader()
                    w.writerows(rows)
                os.replace(OUT + ".tmp", OUT)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
