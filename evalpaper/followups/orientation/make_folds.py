"""A33.1 fold builder: UB-GOLD's own split code with the class roles swapped (NOTES.md §2).

UB-GOLD's get_ad_split_TU and get_ad_dataset_TU are taken verbatim from the harness copy (ast-extracted by
harness/run_eval.py's load_ubgold_data_functions, imported read-only) and run on a TUDataset whose labels are
y' = 1 - y (# CHANGE-A33, a33_common.make_flipped_tudataset). Same split call (StratifiedKFold(5, shuffle, rs 0)),
same trial seeds set_seed(3407 + fold). Writes $GLAD_DATA_ROOT/followups/orientation/folds/<DS>.json
(same keys as the verified export $GLAD_DATA_ROOT/ubgold_splits/<DS>.json, plus A33 fields) and
folds_check.json with every check result. No model, no metric.

  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B make_folds.py [--datasets BZR COX2 PROTEINS_full]
  (--out_dir DIR for a test build; refuses to overwrite an existing fold file whose content differs)
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
import a33_common as A  # noqa: E402

FOLD_DIR = osp.join(A.DATA_ROOT, "orientation", "folds")
DATASETS = ["BZR", "COX2", "PROTEINS_full"]
NFOLD = 5


def build(DS, RE, ns, Flipped, Plain, ref):
    args = SimpleNamespace(DS=DS, num_trial=NFOLD, batch_size=128, batch_size_test=9999, rw_dim=16, dg_dim=16)
    checks = {}
    cwd = os.getcwd()
    os.chdir(RE.DATA_RUN)
    try:
        ns["TUDataset"] = Plain
        ds_plain = Plain(osp.join(os.getcwd(), ".", "data", DS), name=DS)
        y = [int(d.y.item()) for d in ds_plain]
        n = len(y)
        cnt = [y.count(0), y.count(1)]
        assert sorted(set(y)) == [0, 1], set(y)
        assert cnt[0] > cnt[1], f"{DS}: class 0 is not the majority {cnt}"   # minority = class 1 (verified)
        checks["class_counts"] = cnt
        checks["majority_class"], checks["minority_class"] = 0, 1
        splits_plain = ns["get_ad_split_TU"](args, fold=args.num_trial)
        ns["TUDataset"] = Flipped
        splits_flip = ns["get_ad_split_TU"](args, fold=args.num_trial)
        folds = []
        for k in range(NFOLD):
            tr_p, te_p = (list(map(int, a)) for a in splits_plain[k])
            tr_f, te_f = (list(map(int, a)) for a in splits_flip[k])
            rf = ref["folds"][k]
            # (a) swapped labels give the same split as UB-GOLD's and the verified export
            assert tr_f == tr_p == list(map(int, rf["split_train_index"])), (DS, k, "train split")
            assert te_f == te_p == list(map(int, rf["split_test_index"])), (DS, k, "test split")
            # UB-GOLD's dataset function on the swapped labels (trial seed as UB-GOLD's main loop)
            ns["set_seed"](3407 + k)
            out = ns["get_ad_dataset_TU"](args, splits_flip[k], need_str_enc=False)
            train_list, test_list, meta = out[3].dataset, out[5].dataset, out[6]
            got_train = [int(d.a33_index) for d in train_list]
            got_test = [int(d.a33_index) for d in test_list]
            got_y = [int(d.y) for d in test_list]
            # (c) independent derivation: normals = class 0 graphs of the training split, in split order
            exp_train = [i for i in tr_f if y[i] == 0]
            exp_y = [int(y[i] == 1) for i in te_f]
            assert got_train == exp_train, (DS, k, "train normals")
            assert got_test == te_f, (DS, k, "test order")
            assert got_y == exp_y, (DS, k, "y_test")                 # (d)
            assert set(got_train) <= set(tr_f) and not (set(got_train) & set(te_f)), (DS, k, "disjoint")
            assert meta["num_train"] == len(exp_train)
            # (e) prevalence = class-1 share of the test fold
            prev = sum(exp_y) / len(exp_y)
            assert abs(prev - sum(y[i] for i in te_f) / len(te_f)) < 1e-12
            # unflipped reproduction of the verified export (sanity of the extraction)
            assert [i for i in tr_p if y[i] != 0] == list(map(int, rf["train_idx"])), (DS, k, "orig train_idx")
            assert [int(y[i] == 0) for i in te_p] == list(map(int, rf["y_test"])), (DS, k, "orig y_test")
            folds.append(dict(fold=k, seed=3407 + k, split_train_index=tr_f, split_test_index=te_f,
                              train_idx=exp_train, test_idx=te_f, y_test=exp_y, num_train_meta=int(meta["num_train"]),
                              train_n_nodes=[int(d.num_nodes) for d in train_list],
                              n_test=len(te_f), n_test_anomalous=int(sum(exp_y)), prevalence=prev))
    finally:
        os.chdir(cwd)
        ns["TUDataset"] = Plain
    # (b) test folds pairwise disjoint and covering all graphs
    allte = [i for f in folds for i in f["test_idx"]]
    assert len(allte) == len(set(allte)) == n and set(allte) == set(range(n)), (DS, "test folds partition")
    # (e) anomalies over the 5 folds = class-1 count
    assert sum(f["n_test_anomalous"] for f in folds) == cnt[1]
    # (f) every class-0 graph is a training normal in exactly 4 folds; no class-1 graph is ever a training graph
    from collections import Counter
    c = Counter(i for f in folds for i in f["train_idx"])
    assert set(c) == {i for i in range(n) if y[i] == 0} and set(c.values()) == {NFOLD - 1}, (DS, "train coverage")
    checks.update(n_graphs=n, test_partition=True, train_normals_each_in_4_folds=True,
                  split_equals_unflipped_and_export=True, anomalies_total=cnt[1],
                  per_fold=[dict(fold=f["fold"], n_train_normal=len(f["train_idx"]), n_test=f["n_test"],
                                 n_test_anomalous=f["n_test_anomalous"], prevalence=round(f["prevalence"], 6),
                                 n_train_normal_original=len(ref["folds"][f["fold"]]["train_idx"]))
                            for f in folds])
    obj = dict(DS=DS, kind="TU", convention="flipped (A33.1): class 1 (minority) anomalous, class 0 normal",
               anomalous_class=1, normal_class=0, class_counts=cnt, n_graphs=n, num_feat=ref["num_feat"],
               source_export=osp.join(RE.SPLITS, f"{DS}.json"), source_export_sha256=A.sha256_file(osp.join(RE.SPLITS, f"{DS}.json")),
               builder="prelim/evalpaper/followups/orientation/make_folds.py",
               ubgold_functions="get_ad_split_TU, get_ad_dataset_TU (verbatim, harness/ubgold_src c940457) on y' = 1 - y",
               created=time.strftime("%Y-%m-%d %H:%M:%S"), folds=folds)
    return obj, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out_dir", default=FOLD_DIR)
    a = ap.parse_args()
    RE = A.load_harness()
    RE.setup_overlay()
    RE.install_torch_scatter_shim()
    ns = RE.load_ubgold_data_functions()
    Plain = ns["TUDataset"]
    Flipped = A.make_flipped_tudataset(Plain, tag_index=True)
    os.makedirs(a.out_dir, exist_ok=True)
    report = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), builder_sha256=A.sha256_file(__file__),
                  run_eval_sha256=A.sha256_file(A.RUN_EVAL), versions=A.versions(), datasets={})
    for DS in a.datasets:
        ref = json.load(open(osp.join(RE.SPLITS, f"{DS}.json")))
        obj, checks = build(DS, RE, ns, Flipped, Plain, ref)
        path = osp.join(a.out_dir, f"{DS}.json")
        if osp.exists(path):
            old = json.load(open(path))
            same = old["folds"] == json.loads(json.dumps(obj["folds"]))
            assert same, f"{path} exists with different folds; refusing to overwrite"
            print(f"[same] {path} (unchanged)")
        else:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(obj, f)
            os.replace(tmp, path)
            print(f"[wrote] {path}")
        checks["fold_file"] = path
        checks["fold_file_sha256"] = A.sha256_file(path)
        report["datasets"][DS] = checks
        print(f"{DS}: classes {checks['class_counts']}; per fold (train normals, test, anomalies): "
              + " ".join(f"({r['n_train_normal']},{r['n_test']},{r['n_test_anomalous']})" for r in checks["per_fold"]))
    with open(osp.join(a.out_dir, "folds_check.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("all checks passed ->", osp.join(a.out_dir, "folds_check.json"))


if __name__ == "__main__":
    main()
