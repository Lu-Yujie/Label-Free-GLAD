"""A33.1 training-free references under the flipped convention (NOTES.md §5): size only (ctrl_size_lw), graphlet count
+ LW (ctrl_gc_lw), GRAIL-0 (grail0_sumz), plus the other rows grail0/run.py's fold function writes.

grail0/run.py is imported unchanged; its per-fold function run_fold() runs as in the main reference runs, with
common.data.load_folds replaced in THIS process (# CHANGE-A33) by a loader that returns the flipped folds: graphs from
the same fold-cache pickle ($GLAD_DATA_ROOT/folds/<DS>.pkl, same views), train = the flipped training normals
(class 0 graphs of the training split, split order), test = the same test fold, y_test = [class == 1]. The label-free
graphlet and WL caches (computed over all graphs) are reused. As in the main runs, references fit on ALL training
normals (no 20% carve-out). Outputs: followups/orientation/refs_parts/<DS>__fold<k>.csv/.json; --collect writes
results/eval_orientation_refs_folds.csv (grail0_folds schema + convention). Prints counts only.

  cd <repo> && PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B \
      evalpaper/followups/orientation/refs_flipped.py [--datasets BZR COX2 PROTEINS_full] [--threads 4] [--collect]
  test: add --out_parts <dir> (never the grid refs_parts) --folds 0
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(osp.dirname(HERE)))
sys.path.insert(0, PRELIM)
FOLD_DIR = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/followups/orientation/folds")
GRID_PARTS = osp.join(HERE, "refs_parts")
FINAL_OUT = osp.join(PRELIM, "results", "eval_orientation_refs_folds.csv")
DS3 = ["BZR", "COX2", "PROTEINS_full"]
A33_ROWS = {"ctrl_size_lw": "size-only", "ctrl_gc_lw": "graphlet-count+LW", "grail0_sumz": "GRAIL-0"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_flipped_loader(orig_load_folds):
    import common.data as CD

    def load_folds(name, features="ubgold", folds=None):          # CHANGE-A33: flipped folds
        if features not in ("ubgold", "all"):
            raise ValueError("features must be 'ubgold' or 'all'")
        obj = CD.load_raw_cache(name)
        flip = json.load(open(osp.join(FOLD_DIR, f"{name}.json")))
        meta0 = obj["meta"]
        use_nl = features == "all" or meta0["ubgold_uses_node_labels"]
        use_x = features == "all"
        for i, g in enumerate(obj["graphs"]):
            assert g["gid"] == i
            CD._ro(g["edges"]); CD._ro(g["node_label"]); CD._ro(g["x_attr"])

        def view(g):          # = common.data.load_folds.view
            return {"n": g["n"], "edges": g["edges"], "x": g["x_attr"] if use_x else None,
                    "node_label": g["node_label"] if use_nl else None, "gid": g["gid"]}
        meta_public = {k: v for k, v in meta0.items() if k not in ("reader_info", "file_ids")}
        meta_public["features"] = features
        out = []
        for f0, ff in zip(obj["folds"], flip["folds"]):
            assert int(f0["fold"]) == ff["fold"]
            if folds is not None and ff["fold"] not in folds:
                continue
            assert list(map(int, f0["test_idx"])) == ff["test_idx"], "fold cache test set != flipped test set"
            tr, te = ff["train_idx"], ff["test_idx"]
            y = np.asarray(ff["y_test"], dtype=np.int64).copy()
            meta = dict(meta_public)
            meta.update({"fold": int(ff["fold"]), "seed": int(f0["seed"]), "n_train": len(tr), "n_test": len(te),
                         "n_test_anomalies": int(y.sum()), "train_idx": np.asarray(tr).copy(),
                         "test_idx": np.asarray(te).copy(), "convention": "flipped (A33.1)"})
            out.append({"train": [view(obj["graphs"][i]) for i in tr], "test": [view(obj["graphs"][i]) for i in te],
                        "y_test": y, "fold": int(ff["fold"]), "meta": meta})
        return out
    return load_folds


def collect(parts_dir, out):
    import pandas as pd
    files = sorted(glob.glob(osp.join(parts_dir, "*__fold*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["convention"] = "flipped"
    df["a33_reference"] = df.method.map(A33_ROWS).fillna("")
    cov = df[df.method.isin(list(A33_ROWS))].groupby(["dataset", "method"]).fold.nunique()
    print(f"refs rows {len(df)} from {len(files)} fold files; A33 reference cells with 5 folds: {int((cov == 5).sum())} of 9")
    df.to_csv(out, index=False)
    print(f"wrote -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DS3)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out_parts", default=GRID_PARTS)
    ap.add_argument("--collect", action="store_true", help=f"after the runs, write {FINAL_OUT} (grid parts only)")
    a = ap.parse_args()
    assert set(a.datasets) <= set(DS3)
    import common.data as CD
    import grail0.run as GR
    GR.PARTS = a.out_parts                                    # CHANGE-A33: outputs -> A33 directory
    CD.load_folds = make_flipped_loader(CD.load_folds)        # CHANGE-A33: run_fold imports it at call time
    os.makedirs(a.out_parts, exist_ok=True)
    prov = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), script_sha256=sha(__file__),
                grail0_run_sha256=sha(GR.__file__), common_data_sha256=sha(CD.__file__),
                fold_cache_sha256={ds: sha(osp.join(FOLD_DIR, f"{ds}.json")) for ds in a.datasets})
    for ds in a.datasets:
        for k in a.folds:
            p, info = GR.run_fold(ds, k, a.threads)
            if info is None:
                print(f"[skip] {p} exists")
                continue
            info["a33"] = dict(prov, convention="flipped: class 1 (minority) anomalous")
            with open(p.replace(".csv", ".json"), "w") as f:
                json.dump(info, f, indent=1)
            import pandas as pd
            d = pd.read_csv(p)
            assert d.auroc.notna().all(), (ds, k, "non-finite reference metric")
            print(f"[fold] {ds} {k}: n_train {info['n_train']} n_test {info['n_test']} n_anom {info['n_anom']} "
                  f"rows {len(d)} ({info['seconds']:.1f}s)")
    if a.collect:
        if osp.realpath(a.out_parts) != osp.realpath(GRID_PARTS):
            sys.exit("--collect writes the official file from the grid refs_parts only")
        collect(a.out_parts, FINAL_OUT)


if __name__ == "__main__":
    main()
