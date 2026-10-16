"""Build the cached UB-GOLD-faithful fold files $GLAD_DATA_ROOT/folds/<DS>.pkl.

Protocol (UB-GOLD ``exp_type ad``, benchmark/mymain.py + dataloader/data_loader.py):
  TU datasets (get_ad_split_TU / get_ad_dataset_TU):
    * graph labels remapped to 0..C-1 (sorted raw values, as PyG TUDataset does);
    * StratifiedKFold(n_splits=5, shuffle=True, random_state=0) over ALL graphs;
    * trial i uses fold i and seed 3407+i;
    * train = training-fold graphs whose label != 0 (all other classes are normal);
    * test  = the whole test fold; y_test = 1 iff label == 0 (anomaly), no down-sampling;
    * node features: one-hot TU node labels (no continuous attributes,
      use_node_attr=False); Constant(1) for IMDB-BINARY / REDDIT-BINARY / COLLAB.
  Tox21 (get_ad_dataset_Tox21 / read_graph_file): fixed TU ``_training`` / ``_testing``
    files; per-file first-appearance label remap; train = training-file graphs with
    mapped label 1; y_test = 1 iff mapped label 1 in the testing file; isolated nodes and
    edge-less graphs dropped; constant node features. Same split for all 5 trials.

The split indices are taken from the export of UB-GOLD's own code
(common/ubgold_xcheck.py -> ubgold_splits/<DS>.json) and independently recomputed
here with sklearn; the build asserts they agree.

Usage: PYTHONDONTWRITEBYTECODE=1 python -I -B common/build_folds.py AIDS BZR ...
"""
import json
import os
import os.path as osp
import pickle
import sys
import time

PRELIM = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PRELIM)

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common.tu_io import read_tu_pyg_like, read_tox21_ubgold_like

DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data"))
RAW = osp.join(DATA_ROOT, "raw")
FOLD_DIR = osp.join(DATA_ROOT, "folds")
UBG_DIR = osp.join(DATA_ROOT, "ubgold_splits")
CONSTANT_FEAT = ("IMDB-BINARY", "REDDIT-BINARY", "COLLAB")
N_FOLDS = 5
BASE_SEED = 3407

# UB-GOLD arXiv 2406.15523v2 Table 1: (# ID train, # ID test, # OOD/anomaly test)
UBGOLD_TABLE1 = {
    "Tox21_p53": (8088, 241, 28), "Tox21_HSE": (423, 257, 10), "Tox21_MMP": (6170, 200, 38),
    "Tox21_PPAR-gamma": (219, 252, 15), "COLLAB": (1920, 480, 520), "IMDB-BINARY": (400, 100, 100),
    "REDDIT-BINARY": (800, 200, 200), "ENZYMES": (400, 100, 20), "PROTEINS_full": (360, 90, 133),
    "DD": (390, 97, 139), "BZR": (69, 17, 64), "AIDS": (1280, 320, 80), "COX2": (81, 21, 73),
    "NCI1": (1646, 411, 411), "DHFR": (368, 93, 59),
}


def native_tu_splits(y):
    kfd = StratifiedKFold(n_splits=N_FOLDS, random_state=0, shuffle=True)
    out = []
    for tr, te in kfd.split(np.zeros(len(y)), y):
        out.append((tr[y[tr] != 0], te, (y[te] == 0).astype(np.int64), tr))
    return out


def build_tu(DS):
    graphs, y, info = read_tu_pyg_like(osp.join(RAW, DS), DS)
    nat = native_tu_splits(y)
    ub_path = osp.join(UBG_DIR, DS + ".json")
    ub = json.load(open(ub_path)) if osp.exists(ub_path) else None
    verified = ub is not None and ub["n_graph_mismatches"] == 0
    folds = []
    for k in range(N_FOLDS):
        tr, te, yt, tr_full = nat[k]
        if ub is not None:
            f = ub["folds"][k]
            # UB-GOLD keeps test-fold order and train-fold order; compare exactly
            same = (np.array_equal(np.asarray(f["train_idx"]), tr) and
                    np.array_equal(np.asarray(f["test_idx"]), te) and
                    np.array_equal(np.asarray(f["y_test"]), yt) and
                    np.array_equal(np.asarray(f["split_train_index"]), tr_full))
            if not same:
                raise RuntimeError(f"{DS} fold {k}: native split != UB-GOLD export")
            tr = np.asarray(f["train_idx"], dtype=np.int64)
            te = np.asarray(f["test_idx"], dtype=np.int64)
            yt = np.asarray(f["y_test"], dtype=np.int64)
        # leakage / consistency checks
        assert len(np.intersect1d(tr, te)) == 0, "train/test overlap"
        assert np.all(y[tr] != 0), "anomaly class in training"
        assert np.array_equal(yt, (y[te] == 0).astype(np.int64))
        folds.append({"fold": k, "seed": BASE_SEED + k, "train_idx": tr, "test_idx": te, "y_test": yt})
    raw_vals = info["graph_label_raw_values"]
    meta = {
        "dataset": DS, "kind": "TU", "fixed_split": False,
        "protocol": ("UB-GOLD ad / get_ad_dataset_TU: StratifiedKFold(5, shuffle=True, random_state=0) "
                     "over all graphs; train = train-fold graphs with label!=0; test = test fold; "
                     "anomaly = class 0 (smallest raw graph label)"),
        "anomaly_class": 0, "anomaly_class_raw": raw_vals[0],
        "normal_classes": list(range(1, len(raw_vals))), "normal_classes_raw": raw_vals[1:],
        "class_counts": info["class_counts"], "n_graphs": info["n_graphs"],
        "ubgold_node_features": "constant" if DS in CONSTANT_FEAT else "node_label_onehot",
        "ubgold_uses_node_labels": DS not in CONSTANT_FEAT and info["node_label_dim"] is not None,
        "node_label_dim": info["node_label_dim"], "x_attr_dim": info["x_attr_dim"],
        "ubgold_num_feat": None if ub is None else ub["num_feat"],
        "ubgold_verified": bool(verified),
        "ubgold_table1": UBGOLD_TABLE1.get(DS),
        "reader_info": info,
    }
    return {"name": DS, "graphs": graphs, "labels_orig": y, "folds": folds, "meta": meta}


def build_tox21(DS):
    g_tr, y_tr_m, y_tr_raw, info_tr = read_tox21_ubgold_like(osp.join(RAW, f"{DS}_training"), f"{DS}_training")
    g_te, y_te_m, y_te_raw, info_te = read_tox21_ubgold_like(osp.join(RAW, f"{DS}_testing"), f"{DS}_testing")
    train_graphs = [g for g, yy in zip(g_tr, y_tr_m) if yy == 1]
    graphs = []
    file_ids = []
    for g in train_graphs:
        file_ids.append(("training", g["gid"]))
    for g in g_te:
        file_ids.append(("testing", g["gid"]))
    for i, g in enumerate(train_graphs + g_te):
        g = dict(g)
        g["gid"] = i                                     # position in the cached graph list
        graphs.append(g)
    tr = np.arange(len(train_graphs), dtype=np.int64)
    te = np.arange(len(train_graphs), len(graphs), dtype=np.int64)
    yt = (y_te_m == 1).astype(np.int64)
    ub_path = osp.join(UBG_DIR, DS + ".json")
    ub = json.load(open(ub_path)) if osp.exists(ub_path) else None
    verified = False
    if ub is not None:
        verified = (ub["n_graph_mismatches"] == 0 and ub["n_train"] == len(tr) and
                    ub["n_test"] == len(te) and ub["y_test"] == yt.tolist() and
                    ub["train_gid"] == [fid for s, fid in file_ids[:len(tr)]] and
                    ub["test_gid"] == [fid for s, fid in file_ids[len(tr):]])
        if not verified:
            raise RuntimeError(f"{DS}: native Tox21 split != UB-GOLD export")
    # raw semantics of the classes
    tr_order = info_tr["first_appearance_label_order"]
    te_order = info_te["first_appearance_label_order"]
    normal_raw_train = tr_order[1] if len(tr_order) > 1 else None
    anomaly_raw_test = te_order[1] if len(te_order) > 1 else None
    folds = [{"fold": k, "seed": BASE_SEED + k, "train_idx": tr, "test_idx": te, "y_test": yt}
             for k in range(N_FOLDS)]
    has_nl = graphs[0]["node_label"] is not None
    nl_min = min(int(g["node_label"].min()) for g in graphs if g["n"] > 0) if has_nl else None
    if has_nl:
        for g in graphs:
            g["node_label"] = g["node_label"] - nl_min
    meta = {
        "dataset": DS, "kind": "Tox21", "fixed_split": True,
        "protocol": ("UB-GOLD ad / get_ad_dataset_Tox21: fixed TU _training/_testing files, per-file "
                     "first-appearance label remap; train = training-file graphs with mapped label 1; "
                     "y_test = mapped label 1 in testing file; isolated nodes and edge-less graphs dropped; "
                     "same split for all 5 trials (seeds 3407+i)"),
        "normal_raw_label_train": normal_raw_train, "anomaly_raw_label_test": anomaly_raw_test,
        "semantics_consistent": normal_raw_train is not None and normal_raw_train != anomaly_raw_test,
        "ubgold_node_features": "constant", "ubgold_uses_node_labels": False,
        "node_label_dim": None if not has_nl else int(max(int(g["node_label"].max()) for g in graphs if g["n"] > 0)) + 1,
        "x_attr_dim": None, "ubgold_num_feat": None if ub is None else ub["num_feat"],
        "ubgold_verified": bool(verified), "ubgold_table1": UBGOLD_TABLE1.get(DS),
        "reader_info": {"training": info_tr, "testing": info_te},
        "file_ids": file_ids,
    }
    return {"name": DS, "graphs": graphs, "labels_orig": None, "folds": folds, "meta": meta}


def main(names):
    os.makedirs(FOLD_DIR, exist_ok=True)
    for DS in names:
        t0 = time.time()
        obj = build_tox21(DS) if DS.startswith("Tox21") else build_tu(DS)
        tmp = osp.join(FOLD_DIR, DS + ".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, osp.join(FOLD_DIR, DS + ".pkl"))      # atomic: list_datasets never sees partial files
        m = obj["meta"]
        cnt = [(len(f["train_idx"]), int((f["y_test"] == 0).sum()), int(f["y_test"].sum())) for f in obj["folds"]]
        print(f"[build_folds] {DS}: verified={m['ubgold_verified']} folds(train,test_norm,test_anom)={cnt} "
              f"table1={m['ubgold_table1']} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
