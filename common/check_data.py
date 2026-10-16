"""Sanity checks for the cached folds -> common/DATA_CHECK.md (+ data_check.json).

Usage: PYTHONDONTWRITEBYTECODE=1 python -I -B common/check_data.py
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import json
import os.path as osp
import sys
import time

PRELIM = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PRELIM)

import numpy as np

from common.data import list_datasets, load_raw_cache, load_folds, TU12
from common.metrics import evaluate

UBG_DIR = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_splits")
OUT_MD = osp.join(PRELIM, "common", "DATA_CHECK.md")
OUT_JSON = osp.join(PRELIM, "common", "data_check.json")
WL_ITERS = 3


def wl_fingerprints(graphs, use_labels, iters=WL_ITERS, seed=0):
    """128-bit WL(iters) fingerprint per graph (multiset hashing with random uint64 tables).
    Equal fingerprints <=> WL-indistinguishable (up to negligible hash collisions)."""
    rng = np.random.default_rng(seed)
    ns = np.array([g["n"] for g in graphs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(ns)])
    N = int(off[-1])
    src = np.concatenate([g["edges"][:, 0] + off[i] for i, g in enumerate(graphs)] + [np.zeros(0, np.int64)])
    dst = np.concatenate([g["edges"][:, 1] + off[i] for i, g in enumerate(graphs)] + [np.zeros(0, np.int64)])
    s = np.concatenate([src, dst]); d = np.concatenate([dst, src])
    order = np.argsort(s, kind="stable"); s = s[order]; d = d[order]
    deg = np.bincount(s, minlength=N)
    has = deg > 0
    starts = np.concatenate([[0], np.cumsum(deg)])[:-1][has]
    if use_labels and graphs[0]["node_label"] is not None:
        col = np.concatenate([g["node_label"] for g in graphs]).astype(np.int64)
    else:
        col = np.zeros(N, dtype=np.int64)
    _, col = np.unique(col, return_inverse=True)
    for _ in range(iters):
        R = rng.integers(0, 2**63 - 1, size=col.max() + 1, dtype=np.int64).astype(np.uint64)
        agg = np.zeros(N, dtype=np.uint64)
        if s.size:
            agg[has] = np.add.reduceat(R[col[d]], starts)
        sig = np.stack([col.astype(np.uint64), agg], 1)
        _, col = np.unique(sig, axis=0, return_inverse=True)
        col = col.ravel()
    R1 = rng.integers(0, 2**63 - 1, size=col.max() + 1, dtype=np.int64).astype(np.uint64)
    R2 = rng.integers(0, 2**63 - 1, size=col.max() + 1, dtype=np.int64).astype(np.uint64)
    gid = np.repeat(np.arange(len(graphs)), ns)
    h1 = np.zeros(len(graphs), dtype=np.uint64); h2 = np.zeros(len(graphs), dtype=np.uint64)
    nz = ns > 0
    if N:
        st = off[:-1][nz]
        h1[nz] = np.add.reduceat(R1[col], st); h2[nz] = np.add.reduceat(R2[col], st)
    ms = np.array([g["edges"].shape[0] for g in graphs], dtype=np.uint64)
    return [(int(a), int(b), int(n), int(m)) for a, b, n, m in zip(h1, h2, ns, ms)]


def check_dataset(DS):
    t0 = time.time()
    obj = load_raw_cache(DS)
    meta = obj["meta"]
    graphs = obj["graphs"]
    ub_path = osp.join(UBG_DIR, DS + ".json")
    ub = json.load(open(ub_path)) if osp.exists(ub_path) else None
    fps = wl_fingerprints(graphs, use_labels=meta["ubgold_uses_node_labels"])
    ns = np.array([g["n"] for g in graphs]); ms = np.array([g["edges"].shape[0] for g in graphs])
    n_iso_graphs = 0
    for g in graphs:
        deg = np.bincount(g["edges"].ravel(), minlength=g["n"]) if g["n"] else np.zeros(0)
        n_iso_graphs += int(g["n"] > 0 and (deg == 0).any())
    folds = load_folds(DS)
    rows = []
    for f, fr in zip(folds, obj["folds"]):
        tr = np.asarray(fr["train_idx"]); te = np.asarray(fr["test_idx"]); y = f["y_test"]
        overlap = int(len(np.intersect1d(tr, te)))
        gid_overlap = len({g["gid"] for g in f["train"]} & {g["gid"] for g in f["test"]})
        train_fp = {fps[i] for i in tr}
        dup = np.array([fps[i] in train_fp for i in te])
        label_ok = None
        if obj["labels_orig"] is not None:
            yo = obj["labels_orig"]
            label_ok = bool(np.all(yo[tr] != 0) and np.array_equal(y, (yo[te] == 0).astype(int)))
        size_auc = evaluate([g["n"] for g in f["test"]], y)["auroc"]
        rows.append({"fold": f["fold"], "seed": f["meta"]["seed"], "n_train": len(tr),
                     "n_test_normal": int((y == 0).sum()), "n_test_anom": int(y.sum()),
                     "anom_ratio_test": float(y.mean()), "idx_overlap": overlap, "gid_overlap": gid_overlap,
                     "labels_ok": label_ok,
                     "wl_dup_test_normal": int(dup[y == 0].sum()), "wl_dup_test_anom": int(dup[y == 1].sum()),
                     "size_auroc": float(size_auc)})
    # within-dataset WL collisions across classes (TU only): graphs identical to a graph of the other class
    cross = None
    if obj["labels_orig"] is not None:
        yo = obj["labels_orig"]
        fp_cls = {}
        for i, fp in enumerate(fps):
            fp_cls.setdefault(fp, set()).add(int(yo[i] == 0))
        cross = int(sum(1 for i, fp in enumerate(fps) if len(fp_cls[fp]) > 1))
    res = {"dataset": DS, "meta": {k: v for k, v in meta.items() if k not in ("reader_info", "file_ids")},
           "reader_info": meta.get("reader_info"),
           "ubgold_export": None if ub is None else {"n_graph_mismatches": ub["n_graph_mismatches"],
                                                      "num_feat": ub["num_feat"], "versions": ub.get("versions")},
           "n_graphs_cached": len(graphs), "avg_n": float(ns.mean()), "avg_m": float(ms.mean()),
           "max_n": int(ns.max()), "min_n": int(ns.min()), "n_graphs_no_edges": int((ms == 0).sum()),
           "n_graphs_with_isolated_nodes": n_iso_graphs, "wl_graphs_shared_across_classes": cross,
           "folds": rows, "seconds": round(time.time() - t0, 1)}
    print(f"[check_data] {DS} done in {res['seconds']}s", flush=True)
    return res


def fmt_md(results, skl):
    L = []
    A = L.append
    A("# DATA_CHECK: UB-GOLD-faithful folds for the Grail pilot")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `prelim/common/check_data.py`. "
      "Machine-readable copy: `prelim/common/data_check.json`. Caches: `$GLAD_DATA_ROOT/folds/<DS>.pkl`.")
    A("")
    A("## Protocol as implemented (UB-GOLD `-exp_type ad`)")
    A("- Source of truth: the official UB-GOLD repository (github.com/UB-GOLD/UB-GOLD, commit c940457, 2026-06-03), "
      "cloned to `$GLAD_DATA_ROOT/UB-GOLD`. Its functions `get_ad_split_TU`, `get_ad_dataset_TU`, "
      "`read_graph_file` and `get_ad_dataset_Tox21` were extracted verbatim from `dataloader/data_loader.py` and executed "
      "unchanged with PyG 2.6.1 (UB-GOLD's pinned version) + CPU torch (`common/ubgold_xcheck.py`). The exported index lists "
      "are what the caches use, and `common/build_folds.py` asserts they equal an independent sklearn re-implementation.")
    A("- TU datasets: graph labels remapped to 0..C-1 (sorted raw values, as PyG). `StratifiedKFold(n_splits=5, shuffle=True, "
      "random_state=0)` over all graphs; UB-GOLD trial i = fold i with seed 3407+i (`meta['seed']`). "
      "Train = training-fold graphs with label != 0 (normal only). Test = the entire test fold, y=1 iff label == 0. "
      "No anomaly down-sampling and no validation set (UB-GOLD sets val = test; we expose no validation split: "
      "label-free calibration must cross-fit inside `fold['train']`).")
    A("- Features (`load_folds(name)` default `features='ubgold'`): what UB-GOLD's models see. Labelled TU sets: "
      "`node_label` = TU node label (UB-GOLD one-hot, dim = `meta['node_label_dim']`); IMDB-BINARY/REDDIT-BINARY/COLLAB: "
      "no node features (UB-GOLD `Constant(1)`); `x` = None everywhere (UB-GOLD uses `TUDataset(use_node_attr=False)`). "
      "`features='all'` additionally exposes continuous node attributes / Tox21 atom types; results with it are NOT "
      "comparable to published UB-GOLD numbers.")
    A("- Tox21 (Type I): fixed TU `_training`/`_testing` files, UB-GOLD's per-file first-appearance label remap, "
      "isolated nodes and edge-less graphs dropped (as `nx.from_edgelist` does), constant node features; the same split "
      "is returned 5 times (`meta['fixed_split']=True`, seeds 3407..3411).")
    A("- Metrics (`common.metrics.evaluate`) mirror UB-GOLD `GAOOD/metric/metric.py`: sklearn AUROC, average precision, "
      "FPR at the first ROC point with TPR >= 0.95. Anomaly = positive class; higher score = more anomalous.")
    A("")
    A("## Verification summary")
    A("| Dataset | UB-GOLD code export == cache (split + every graph: n, edge set, node-label one-hot) | UB-GOLD num_feat | sklearn 1.0.2 folds == cache | train∩test (indices) | all train graphs normal / y_test consistent |")
    A("|---|---|---|---|---|---|")
    for r in results:
        DS = r["dataset"]; m = r["meta"]
        ubx = r["ubgold_export"]
        ok = "yes (0 mismatches)" if (ubx and ubx["n_graph_mismatches"] == 0 and m["ubgold_verified"]) else "NO"
        sk = skl.get("datasets", {}).get(DS, {}).get("same_102_vs_cache")
        sk = "n/a (fixed files)" if DS.startswith("Tox21") else ("yes" if sk else ("NO" if sk is False else "not run"))
        ov = max(f["idx_overlap"] for f in r["folds"]) + max(f["gid_overlap"] for f in r["folds"])
        lab = r["folds"][0]["labels_ok"]
        lab = "n/a (Tox21: by construction)" if lab is None else ("yes" if all(f["labels_ok"] for f in r["folds"]) else "NO")
        A(f"| {DS} | {ok} | {ubx['num_feat'] if ubx else '-'} | {sk} | {'none' if ov == 0 else ov} | {lab} |")
    A("")
    A("## Dataset summary vs. UB-GOLD Table 1 (arXiv 2406.15523v2)")
    A("Counts are (# normal train, # normal test, # anomalous test). UB-GOLD Table 1 lists one fold; ours = fold 0 and the per-fold range.")
    A("")
    A("| Dataset | #graphs | class counts (raw label: count) | anomaly class (raw) | UB-GOLD features | node_label_dim | cont. attr dim (`features='all'`) | avg n / avg m / max n | UB-GOLD Table 1 | ours fold 0 | ours range over folds | match |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        DS = r["dataset"]; m = r["meta"]; F = r["folds"]
        t1 = m.get("ubgold_table1")
        f0 = (F[0]["n_train"], F[0]["n_test_normal"], F[0]["n_test_anom"])
        rng = "/".join(f"{min(f[k] for f in F)}-{max(f[k] for f in F)}" for k in ("n_train", "n_test_normal", "n_test_anom"))
        match = "n/a (not in UB-GOLD)" if t1 is None else ("exact" if tuple(t1) == f0 else
                ("within ±1" if all(abs(a - b) <= 1 for a, b in zip(t1, f0)) else "DIFFERENT"))
        if m["kind"] == "TU":
            raw = m["normal_classes_raw"]; cc = m["class_counts"]
            allraw = [m["anomaly_class_raw"]] + raw
            ccs = ", ".join(f"{v}: {c}" for v, c in zip(allraw, cc))
            anom = f"{m['anomaly_class_raw']} (class 0)"
        else:
            ri = r["reader_info"]
            ccs = "train file " + str(ri["training"]["raw_label_counts_file"]) + "; test file " + str(ri["testing"]["raw_label_counts_file"])
            anom = f"test raw {m['anomaly_raw_label_test']}; train normal raw {m['normal_raw_label_train']}"
        A(f"| {DS} | {r['n_graphs_cached']} | {ccs} | {anom} | {m['ubgold_node_features']} | {m['node_label_dim']} | "
          f"{m['x_attr_dim'] if m['x_attr_dim'] else '-'} | {r['avg_n']:.1f} / {r['avg_m']:.1f} / {r['max_n']} | "
          f"{'/'.join(map(str, t1)) if t1 else '-'} | {'/'.join(map(str, f0))} | {rng} | {match} |")
    A("")
    A("## Deviations from UB-GOLD and caveats (read before reporting numbers)")
    A("1. **Edge format.** UB-GOLD/PyG keep both directions in `edge_index`; we store each undirected edge once (i<j). "
      "Equivalent here: every TU set has 0 asymmetric `A.txt` entries; the only self-loops are 5674 `A.txt` entries in COLLAB, "
      "which PyG's `read_tu_data` also removes (checked: 0 self-loops and 24,572,158 = 2 x 12,286,079 directed edges in "
      "UB-GOLD's PyG COLLAB data, matching our undirected count). "
      "Node order is UB-GOLD's (TU file order; for Tox21, first appearance in the edge list).")
    A("2. **No validation split (same as UB-GOLD).** UB-GOLD passes the test loader as the validation loader and tunes "
      "hyperparameters on test, so its published numbers are test-selected upper bounds. Our folds expose only "
      "`train` (normal) and `test`; any calibration, model selection or score orientation must use training normals only "
      "(e.g. cross-fitting inside `fold['train']`).")
    A("3. **PROTEINS = `PROTEINS_full`.** UB-GOLD's run scripts (`benchmark/Source/*.sh`) use `PROTEINS_full` (same 1113 "
      "graphs and labels as PROTEINS; under UB-GOLD features both give the same 3-d node-label one-hot). Its 29-d continuous "
      "attributes are only exposed with `features='all'`. Aliases `PROTEINS`, `IMDB-B`, `REDDIT-B` are accepted by `load_folds`.")
    A("4. **MUTAG is not a UB-GOLD dataset.** It is in the LGKDE/GLASS 12-TU set. We ran UB-GOLD's own `get_ad_dataset_TU` "
      "on it unchanged: anomaly = class 0 = raw label -1 (non-mutagenic, 63 graphs), normal = raw 1 (mutagenic, 125). "
      "We did not verify that LGKDE/GLASS used the same direction for MUTAG.")
    A("5. **The normal class is often the minority (UB-GOLD code ≠ its own text).** Anomaly = class 0 everywhere. "
      "That is the MAJORITY class for BZR, COX2, PROTEINS_full, DD and COLLAB, contrary to UB-GOLD App. B (\"minority class "
      "is treated as anomalies\"). ENZYMES and COLLAB have several normal classes (5 and 2). This is what UB-GOLD's published "
      "numbers use, so we keep it.")
    A("6. **Tox21: HSE and PPAR-gamma are inverted in UB-GOLD and not cached.** UB-GOLD remaps labels by first appearance per "
      "file and trains on mapped label 1. For p53 and MMP the training file starts with an active (raw 1), so training = "
      "inactives (raw 0) and test anomalies = actives (raw 1), which is consistent. For HSE and PPAR-gamma the training file starts "
      "with an inactive, so UB-GOLD trains on the ACTIVE molecules (423 and 219 graphs, exactly its Table 1 counts) while still "
      "scoring actives as test anomalies. Those two tasks are internally inconsistent, so we only cross-checked them "
      "(`ubgold_splits/Tox21_HSE.json`, `Tox21_PPAR-gamma.json`) and did not add them to `list_datasets()`. Tox21 features in "
      "UB-GOLD are constant; atom types are only exposed with `features='all'`. Tox21 test sets are tiny (28 and 38 anomalies), "
      "and the split is fixed, so the std over the 5 'folds' only reflects method randomness.")
    A("7. **Variance vs std.** UB-GOLD's CSVs (and probably its Table 8 '±') are `statistics.variance` values ×100. "
      "`common.metrics.summarize` reports the std (`np.std`, ddof=0).")
    A("8. **Library versions.** UB-GOLD pinned scikit-learn 1.0.2 and PyG 2.6.1; we ran PyG 2.6.1 with torch 2.14.1+cpu and "
      "scikit-learn 1.9.1. We also ran the 1.0.2 `StratifiedKFold._make_test_folds` source on every TU label vector "
      "(`common/skl_version_check.py`). It reproduces the cached folds exactly (verification table above).")
    A("9. **Isomorphic duplicates.** Some TU sets (notably IMDB-BINARY) contain many WL-identical graphs, including across "
      "the normal/anomaly classes (see `wl-dup` and 'WL-identical to a graph of the other class' below). This is a property "
      "of the data that UB-GOLD keeps, not leakage. No graph index appears in both train and test of any fold.")
    A("10. **Graph size alone is strongly informative, and its direction varies.** See the `size AUROC` column (score = n): "
      "the anomaly class is much larger on AIDS and smaller on MUTAG and NCI1. A fixed one-sided size-like score therefore "
      "flips on some datasets. Channels should be two-sided or calibrated on training normals only.")
    A("")
    A("## Per-fold details")
    A("`wl-dup` = test graphs whose WL(3) fingerprint (node labels included when UB-GOLD uses them) equals that of some "
      "training graph, split by test normal / test anomaly. Duplicates are a property of the raw TU data (isomorphic copies), "
      "not index leakage; anomalous test graphs that are WL-identical to a normal training graph cap the attainable AUROC of "
      "any WL-bounded detector. `size AUROC` = AUROC of the raw node count n as the score (a data diagnostic, not a baseline).")
    A("")
    for r in results:
        A(f"### {r['dataset']}")
        extra = []
        if r["wl_graphs_shared_across_classes"] is not None:
            extra.append(f"graphs WL-identical to a graph of the other class (whole dataset): {r['wl_graphs_shared_across_classes']}")
        extra.append(f"graphs with no edges: {r['n_graphs_no_edges']}; graphs with isolated nodes: {r['n_graphs_with_isolated_nodes']}")
        ri = r["reader_info"]
        if r["meta"]["kind"] == "TU":
            extra.append(f"self-loop entries removed: {ri['self_loops_removed']}; directed A.txt entries without reverse: {ri['asymmetric_directed_entries']}")
        else:
            for part in ("training", "testing"):
                p = ri[part]
                extra.append(f"{part} file: {p['n_graphs_file']} graphs, dropped (no edges) {p['n_dropped_no_edges']}, "
                             f"isolated nodes dropped {p['n_isolated_nodes_dropped']}, first-appearance label order {p['first_appearance_label_order']}")
        for e in extra:
            A(f"- {e}")
        A("")
        A("| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |")
        A("|---|---|---|---|---|---|---|---|")
        for f in r["folds"]:
            A(f"| {f['fold']} | {f['seed']} | {f['n_train']} | {f['n_test_normal']} | {f['n_test_anom']} | "
              f"{f['anom_ratio_test']:.3f} | {f['wl_dup_test_normal']}/{f['wl_dup_test_anom']} | {f['size_auroc']:.3f} |")
        A("")
    return "\n".join(L) + "\n"


def main(names=None):
    names = names or list_datasets()
    skl_path = osp.join(UBG_DIR, "sklearn_1.0.2_check.json")
    skl = json.load(open(skl_path)) if osp.exists(skl_path) else {}
    results = [check_dataset(DS) for DS in names]
    with open(OUT_JSON, "w") as f:
        json.dump({"results": results, "sklearn_check": skl}, f, indent=1, default=str)
    md = fmt_md(results, skl)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print("wrote", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main(sys.argv[1:])
