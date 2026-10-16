"""A33.1 orientation-sensitivity runner: the 12 UB-GOLD detectors on BZR / COX2 / PROTEINS_full with the class roles
swapped (class 1 = minority = anomalous). NOTES.md §3.

harness/run_eval.py is imported READ-ONLY (a33_common.load_harness) and its own run()/_run() do everything: UB-GOLD
hyperparameter line ('ubgold'), set_seed(3407 + fold), UB-GOLD's verbatim split/dataset code, held-out 20% normals
(RandomState(2026 + 1000*fold)), canonical test order, evaluation schedule, add_eval, scores npz, checkpoints, resume.
Overridden in THIS process only (module attributes, # CHANGE-A33):
  * SPLITS -> the flipped fold caches (make_folds.py); harness asserts the split indices against them;
  * load_ubgold_data_functions -> the harness function + TUDataset with y' = 1 - y, and a post-check of the UB-GOLD
    function output (training-normal node counts in order, test labels) against the fold cache;
  * PARTS / SCORES / CKPT_ROOT -> A33 directories; save_ckpt -> marks checkpoints as A33 / flipped;
  * fit_* -> wrappers: meta.a33 + code hashes + planned-device check (loops unchanged);
  * _run -> wrapper writing A12 crash records; add_eval -> + training-side progress line (harness verbose off:
    it would print test AUROC).
Usage (ubgold-gpu env):
  PYTHONDONTWRITEBYTECODE=1 python -B run_orient.py --method GLADC --dataset BZR --fold 0 --require_device cuda
  smoke: add --out_root smoke --max_epochs 4 --tag smoke
Exit codes: 0 done, 3 checkpointed/locked, 5 device mismatch / precondition, 4 crash (recorded).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import traceback  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
import a33_common as A  # noqa: E402

import torch  # noqa: E402

DATASETS = ("BZR", "COX2", "PROTEINS_full")
ROOT = osp.join(A.DATA_ROOT, "orientation")
FOLD_DIR = osp.join(ROOT, "folds")
GRID_PARTS = osp.join(HERE, "parts")
OWN_FILES = {"followups/orientation/run_orient.py": osp.join(HERE, "run_orient.py"),
             "followups/orientation/make_folds.py": osp.join(HERE, "make_folds.py")}
VARIANT = "flipped"
_CFG = {}
_RE = []


def roots(out_root):
    """(parts, scores, ckpt, crash)."""
    if out_root == "grid":
        return GRID_PARTS, osp.join(ROOT, "scores"), osp.join(ROOT, "ckpt"), osp.join(ROOT, "crash")
    assert out_root == "smoke", out_root
    b = osp.join(ROOT, "smoke")
    return osp.join(b, "parts"), osp.join(b, "scores"), osp.join(b, "ckpt"), osp.join(b, "crash")


def configure(out_root, require_device):
    if _RE:
        return _RE[0]
    RE = A.load_harness()
    parts, scores, ckpt, crash = roots(out_root)
    RE.PARTS, RE.SCORES, RE.CKPT_ROOT = parts, scores, ckpt
    RE.SPLITS = FOLD_DIR                                   # CHANGE-A33: harness split assert -> flipped caches
    _CFG.update(out_root=out_root, require_device=require_device, crash_dir=crash)

    orig_ldf = RE.load_ubgold_data_functions

    def load_ubgold_data_functions():                      # CHANGE-A33
        ns = orig_ldf()
        ns["TUDataset"] = A.make_flipped_tudataset(ns["TUDataset"])
        orig_get = ns["get_ad_dataset_TU"]

        def get_ad_dataset_TU(args, split, need_str_enc=True):
            out = orig_get(args, split, need_str_enc=need_str_enc)
            cache = json.load(open(osp.join(FOLD_DIR, f"{args.DS}.json")))
            f = [f for f in cache["folds"] if list(map(int, split[1])) == f["split_test_index"]]
            assert len(f) == 1, "split not found in the flipped fold cache"
            f = f[0]
            tr, te = out[3].dataset, out[5].dataset
            assert [int(d.num_nodes) for d in tr] == f["train_n_nodes"], "training normals differ from fold cache"
            assert [int(d.y) for d in te] == f["y_test"], "flipped test labels differ from fold cache"
            _CFG["fold_check"] = dict(fold=f["fold"], n_train_normal=len(tr), n_test=len(te),
                                      n_test_anomalous=int(sum(f["y_test"])), ok=True)
            return out
        ns["get_ad_dataset_TU"] = get_ad_dataset_TU
        return ns
    RE.load_ubgold_data_functions = load_ubgold_data_functions

    orig_run = RE._run
    codes = (orig_run.__code__,)
    _CFG["codes"] = codes

    def _run_with_crash_record(method, DS, fold, hp, run_id, out_json, ckpath, *rest):
        try:
            return orig_run(method, DS, fold, hp, run_id, out_json, ckpath, *rest)
        except Exception as e:
            A.record_crash(_CFG["crash_dir"], run_id, e, ckpath, dict(a33="A33.1 orientation flipped"))
            raise
    RE._run = _run_with_crash_record

    for name in ("fit_signet", "fit_ocg", "fit_glocalkd", "fit_gladc", "fit_cvtgad", "fit_goodd", "fit_graphde",
                 "fit_two_stage"):
        setattr(RE, name, wrap_fit(RE, getattr(RE, name)))
    A.install_ckpt_marker(RE, dict(experiment="A33.1 orientation", variant=VARIANT))
    A.Progress.install(RE)
    _RE.append(RE)
    return RE


def wrap_fit(RE, fn):
    def fit(C):
        meta = A.find_meta_out(_CFG["codes"])
        cache = osp.join(FOLD_DIR, f"{C.DS}.json")
        meta.update(A.provenance(OWN_FILES))
        meta["a33"] = dict(experiment="A33.1 orientation sensitivity", amendment=A.AMENDMENT, variant=VARIANT,
                           convention="class 1 (minority) anomalous, class 0 normal; UB-GOLD split code on y' = 1 - y",
                           anomalous_class=1, fold_cache=cache, fold_cache_sha256=A.sha256_file(cache),
                           fold_check=_CFG.get("fold_check"), planned_device=_CFG["require_device"],
                           runner="prelim/evalpaper/followups/orientation/run_orient.py",
                           out_root=_CFG["out_root"], hp="ubgold (tuned for the original orientation)")
        assert _CFG.get("fold_check", {}).get("ok"), "fold post-check did not run"
        st = A.device_check(C, _CFG["require_device"], meta)
        if st:
            return st
        if C.state is not None and C.state.get("a33", {}).get("variant") != VARIANT:
            print("[resume] checkpoint variant mismatch; refusing", flush=True)
            return "device_mismatch"
        return fn(C)
    fit.__name__ = fn.__name__
    return fit


def run(method, DS, fold, require_device, out_root="grid", deadline=None, max_epochs=None, ckpt_every_s=300.0,
        tag=""):
    assert DS in DATASETS, DS
    if out_root == "grid" and (tag or max_epochs is not None):
        raise SystemExit("tagged / --max_epochs runs must use --out_root smoke")
    have = "cuda" if torch.cuda.is_available() else "cpu"
    if require_device is not None and have != require_device:
        print(f"[device] task planned on {require_device}, this process sees {have}; not started", flush=True)
        return "device_mismatch"
    RE = configure(out_root, require_device)
    return RE.run(method, DS, fold, hp="ubgold", deadline=deadline, max_epochs=max_epochs, ckpt_every_s=ckpt_every_s,
                  tag=tag, verbose=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--require_device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--out_root", choices=["grid", "smoke"], default="grid")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EP_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    a = ap.parse_args()
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag")
    torch.set_num_threads(a.threads)
    try:
        st = run(a.method, a.dataset, a.fold, a.require_device, out_root=a.out_root, deadline=a.deadline,
                 max_epochs=a.max_epochs, ckpt_every_s=a.ckpt_every_s, tag=a.tag)
    except Exception:
        traceback.print_exc()
        sys.exit(4)
    sys.exit(0 if st == "done" else (5 if st == "device_mismatch" else 3))
