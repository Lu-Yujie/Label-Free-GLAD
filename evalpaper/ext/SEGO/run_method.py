"""SEGO (AAAI 2025; github.com/name-is-what/SEGO @45bdda7, MIT) under the evalpaper label-free protocols
(E5 stratum, evalpaper/PREREG_AMENDMENTS.md A10). Pre-registration: ./NOTES.md Part 1.

Same CLI and per-run JSON "part" schema as evalpaper/harness/run_eval.py:
    PYTHONDONTWRITEBYTECODE=1 python -B run_method.py --method SEGO --dataset BZR --fold 0 [--hp hpfree]

* SEGO's own code (src/, pinned, unmodified) supplies the model (model.HCL), the coding-tree encoder
  (tree_utli.HRN/HRNEncoder/GraphTransform, codingTree), the tree-augmented TU dataset (aug.TUDataset_aug),
  the AD split/loaders (data_loader.get_ad_split_TU / get_ad_dataset_TU / init_structural_encoding, run
  from the pinned file via `ast`), its argparse parser (treecl.arg_parse) and setup_seed. The training and
  scoring loop below is a line-by-line copy of treecl.py's `__main__` loop. Every deviation is marked
  "# CHANGE" (list in NOTES.md).
* Two PyG-2.0 -> 2.6 API fixes are applied as exact, asserted textual replacements on the pinned source of
  the two affected methods (patch_method), recompiled in their own module namespace.
* hp in {ubgold, hpfree}: 'ubgold' = SEGO's released configuration (per-option mode of the 10 lines of
  src/script/sego.sh; no AD line is released), 'hpfree' = treecl.py argparse defaults (crash fallback:
  modal released value). Run ids: SEGO_<DS>_fold<k>[_hpfree][_<tag>].
* Every evaluation point (epoch % eval_freq == 0, plus the final epoch) records test AUROC/AUPRC/FPR95
  (UB-GOLD's metric functions), per-graph test scores in canonical test order, the held-out-normal scores
  and their mean (VAL-NORMAL signal, A3) and the training objective on the held-out graphs (secondary).
* Resumable: checkpoints every --ckpt_every_s and at --deadline (model, optimizer, tree encoder, adaptive
  statistics, RNG states, trajectory, scores).
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # never import from cwd

import argparse
import ast
import contextlib
import importlib.util
import json
import os
import os.path as osp
import random
import re
import resource
import shlex
import subprocess
import sys
import textwrap
import time
import warnings
from collections import Counter

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
SRC = osp.join(HERE, "src")
SEGO_COMMIT = "45bdda7cd36c77379605f8a24641ecb82ec9f32f"
EVALPAPER = osp.dirname(osp.dirname(HERE))
UBG_SRC = osp.join(EVALPAPER, "harness", "ubgold_src")      # read-only: UB-GOLD metric functions
OVERLAYS = [(os.environ.get("GLAD_ENVS", "envs") + "/evalpaper-overlay"),          # torch_sparse 0.6.18 (read-only)
            (os.environ.get("GLAD_ENVS", "envs") + "/ext-sego")]                   # numba 0.60.0, llvmlite 0.43.0, ogb 1.3.6
EXT_ROOT = os.environ.get("SEGO_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_sego"))
SEGO_FILE_ANCHOR = osp.join(EXT_ROOT, "data_loader.py")       # SEGO loaders read dirname(__file__)/data/<DS>
SPLITS = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_splits")
CKPT_ROOT = os.environ.get("EP_CKPT", osp.join(EXT_ROOT, "ckpt"))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
SCORES = os.environ.get("EP_SCORES", osp.join(EXT_ROOT, "scores"))
HOLDOUT_SEED = 2026
METHOD = "SEGO"
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
DATASETS = UB9 + ["MUTAG"]
# HP-FREE crash fallback (NOTES.md Part 1): option -> modal released value where the default crashes
HPFREE_CRASH_FALLBACK = {"REDDIT-BINARY": ["batch_size_test"]}   # 9999: ~1.8e5-node test batch, ~550 GB
PREP_PROCS = int(os.environ.get("SEGO_PREP_PROCS", "1"))
TEST_STOP_EPOCH = int(os.environ.get("EP_TEST_STOP_EPOCH", "0"))   # resume tests only


# ----------------------------------------------------------------------------------------------------
# loading SEGO's code (pinned src/, never modified on disk)
# ----------------------------------------------------------------------------------------------------
_PM = {}


def _pm_worker(i):
    import pickle
    out = _PM["fn"](_PM["items"][i])
    for k in list(out.keys()):
        v = out[k]
        if torch.is_tensor(v):
            # tensors of dataset.get(i) are views into the whole collated dataset tensor; pickling a view
            # serialises the entire storage (REDDIT-B: ~32 MB per graph). clone() keeps only its elements.
            out[k] = v.clone()
    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)


def _parallel_map(fn, items):
    """CHANGE (NOTES 2): the coding-tree pre-transform mapped over a process pool, largest graphs first;
    results are put back in dataset order, so the collated dataset equals the sequential one. With
    SEGO_PREP_PROCS=1 this is exactly `[fn(d) for d in items]`. Inputs reach the forked workers by
    inheritance and results come back as plain pickles (no torch shared-memory file descriptors)."""
    if PREP_PROCS <= 1 or len(items) < 2:
        return [fn(d) for d in items]
    import multiprocessing as mp
    import pickle
    order = sorted(range(len(items)), key=lambda i: -int(items[i].num_nodes))
    _PM.update(fn=fn, items=items)
    try:
        with mp.get_context("fork").Pool(PREP_PROCS) as pool:
            res = pool.map(_pm_worker, order, chunksize=1)
    finally:
        _PM.clear()
    out = [None] * len(items)
    for i, r in zip(order, res):
        out[i] = pickle.loads(r)
    return out


def patch_method(module, cls_name, meth_name, replacements, label):
    """Recompile `cls.meth` from the pinned source file with exact textual replacements (each must occur
    exactly once) inside the module's own namespace."""
    path = module.__file__
    text = open(path).read()
    tree = ast.parse(text)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name][0]
    fn = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == meth_name][0]
    code = textwrap.dedent(ast.get_source_segment(text, fn))
    for old, new in replacements:
        assert code.count(old) == 1, (cls_name, meth_name, old)
        code = code.replace(old, new)
    assert "super()" not in code
    # compiled inside a class of the same name so that private names (self.__x) are mangled as in the original
    code = f"class {cls_name}:\n" + textwrap.indent(code, "    ")
    loc = {}
    exec(compile(code, f"{path} [{label}]", "exec"), module.__dict__, loc)
    setattr(getattr(module, cls_name), meth_name, loc[cls_name].__dict__[meth_name])
    return label


_SEGO = None


def load_sego():
    """Import SEGO's modules from src/ and return a namespace with its functions."""
    global _SEGO
    if _SEGO is not None:
        return _SEGO
    for p in OVERLAYS:
        if p not in sys.path:
            sys.path.append(p)           # appended: the ubgold-gpu venv keeps precedence
    if SRC not in sys.path:
        sys.path.insert(0, SRC)

    # treecl.py, lines 1-4 (every warnings.warn is silenced). CHANGE (cosmetic): applied after torch_sparse
    # is imported; on torch 2.5 TorchScript would otherwise try to compile the no-op `warn` inside
    # torch_sparse's scripted functions and fail.
    import torch_sparse  # noqa: F401
    import torch_geometric  # noqa: F401

    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    import model as sego_model
    import tree_utli
    import aug
    import codingTree  # noqa: F401
    for m, f in ((sego_model, "model.py"), (tree_utli, "tree_utli.py"), (aug, "aug.py")):
        assert osp.realpath(m.__file__) == osp.realpath(osp.join(SRC, f)), m.__file__
    patches = [
        # CHANGE (NOTES 1): PyG >= 2.4 made Data.keys a method (PyG 2.0.0: property)
        patch_method(tree_utli, "GraphTransform", "transform",
                     [("for key in odata.keys:", "for key in odata.keys():")],
                     "CHANGE GraphTransform.transform: odata.keys -> odata.keys()"),
        # CHANGE (NOTES 1): PyG >= 2.3 read_tu_data returns (data, slices, sizes);
        # CHANGE (NOTES 2): the pre-transform runs through _parallel_map (identical output)
        patch_method(aug, "TUDataset_aug", "process",
                     [("self.data, self.slices = read_tu_data(self.raw_dir, self.name)",
                       "self.data, self.slices, _ = read_tu_data(self.raw_dir, self.name)"),
                      ("data_list = [self.pre_transform(data) for data in data_list]",
                       "data_list = _parallel_map(self.pre_transform, data_list)")],
                     "CHANGE TUDataset_aug.process: read_tu_data 3-tuple; pre_transform via _parallel_map"),
        # CHANGE (NOTES 1): MessagePassing `size` semantics for flow='target_to_source'. PyG 2.0.0
        # (__collect__) checks x_j against size[0] and aggregates into size[1] rows whatever the flow;
        # PyG >= 2.1 uses size[j]/size[i] = size[1]/size[0] for target_to_source. SEGO passes
        # (n_children, n_parents) in the old convention; the same computation in PyG 2.6 needs
        # (n_parents, n_children). Without it: "ValueError: Encountered tensor with size <n_children>
        # in dimension -2, but expected size <n_parents>".
        patch_method(tree_utli, "HRNEncoder", "forward",
                     [("size = data.treeNodeSize[:, [_, _+1]].sum(dim=0)",
                       "size = data.treeNodeSize[:, [_+1, _]].sum(dim=0)")],
                     "CHANGE HRNEncoder.forward: MessagePassing size tuple in PyG>=2.1 order"),
    ]
    aug._parallel_map = _parallel_map
    # SEGO's AD data functions, executed from the pinned data_loader.py with __file__ pointing into
    # EXT_ROOT so that `osp.dirname(osp.realpath(__file__))/data/<DS>` is our data directory (CHANGE, NOTES 3)
    from scipy import sparse as sp
    import networkx as nx
    from sklearn.model_selection import StratifiedKFold
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Constant
    from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
    ns = dict(os=os, osp=osp, re=re, sp=sp, torch=torch, np=np, nx=nx, DataLoader=DataLoader, Constant=Constant,
              TUDataset=TUDataset, to_scipy_sparse_matrix=to_scipy_sparse_matrix, degree=degree,
              from_networkx=from_networkx, StratifiedKFold=StratifiedKFold, __file__=SEGO_FILE_ANCHOR)
    extract_functions(osp.join(SRC, "data_loader.py"),
                      ["get_ad_split_TU", "get_ad_dataset_TU", "init_structural_encoding"], ns)
    # treecl.py: arg_parse() and setup_seed(), verbatim
    import torch_geometric
    tns = dict(argparse=argparse, torch=torch, np=np, random=random, torch_geometric=torch_geometric)
    extract_functions(osp.join(SRC, "treecl.py"), ["arg_parse", "setup_seed"], tns)
    _SEGO = dict(HCL=sego_model.HCL, HRN=tree_utli.HRN, HRNEncoder=tree_utli.HRNEncoder, aug=aug,
                 tree_utli=tree_utli, get_ad_split_TU=ns["get_ad_split_TU"],
                 get_ad_dataset_TU=ns["get_ad_dataset_TU"], init_structural_encoding=ns["init_structural_encoding"],
                 arg_parse=tns["arg_parse"], setup_seed=tns["setup_seed"], patches=patches, DataLoader=DataLoader)
    return _SEGO


def extract_functions(path, names, ns):
    tree = ast.parse(open(path).read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in funcs) == sorted(names), [f.name for f in funcs]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns


def load_ubgold_metric():
    """UB-GOLD's GAOOD/metric/metric.py (ood_auc, ood_aupr, fpr95), loaded from the harness copy."""
    spec = importlib.util.spec_from_file_location("ubgold_metric", osp.join(UBG_SRC, "GAOOD", "metric", "metric.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------
# hyperparameters (NOTES.md Part 1)
# ----------------------------------------------------------------------------------------------------
def sego_parse(S, toks):
    """SEGO's own parser (treecl.arg_parse reads sys.argv) on an explicit token list."""
    old = sys.argv
    sys.argv = ["treecl.py"] + list(toks)
    try:
        return S["arg_parse"]()
    finally:
        sys.argv = old


def released_lines():
    out = []
    for raw in open(osp.join(SRC, "script", "sego.sh")):
        s = raw.strip()
        if re.match(r"^python3? +treecl\.py", s):
            out.append(shlex.split(s, comments=True)[2:])
    assert len(out) == 10, len(out)
    return out


def modal_released(S):
    """Per-option modal value over the 10 released lines (each resolved by SEGO's parser; ties -> smaller)."""
    res = [vars(sego_parse(S, t)) for t in released_lines()]
    dflt = vars(sego_parse(S, []))
    modal = {}
    for k in dflt:
        if k in ("exp_type", "DS", "DS_ood", "DS_pair"):
            continue
        c = Counter(r[k] for r in res)
        top = max(c.values())
        modal[k] = sorted([v for v, n in c.items() if n == top], key=lambda v: (str(type(v)), v))[0]
    return modal, dflt


def resolve_hparams(S, DS, hp):
    modal, dflt = modal_released(S)
    toks = ["-exp_type", "ad", "-DS", DS]
    notes = []
    if hp == "ubgold":
        diff = {k: v for k, v in modal.items() if v != dflt[k]}
        for k, v in sorted(diff.items()):
            toks += ["-" + k, str(v)]
        tag = "released-modal (no AD configuration is released; mode over the 10 sego.sh lines)"
    else:
        assert hp == "hpfree", hp
        diff = {}
        for k in HPFREE_CRASH_FALLBACK.get(DS, []):
            diff[k] = modal[k]
            toks += ["-" + k, str(modal[k])]
            notes.append(f"hpfree crash fallback: {k} {dflt[k]} -> modal released {modal[k]} (NOTES Part 1)")
        tag = "hpfree (treecl.py argparse defaults)"
    args = sego_parse(S, toks)
    for k, v in diff.items():           # the resolution really produced the intended values
        assert getattr(args, k) == v, (k, getattr(args, k), v)
    line = "python treecl.py " + " ".join(toks) + "  # " + tag
    return args, line, notes


# ----------------------------------------------------------------------------------------------------
# data preparation (coding trees are pre-computed once per dataset; SEGO caches them as data_tree<d>.pt)
# ----------------------------------------------------------------------------------------------------
def processed_paths(DS, tree_depth=5):
    d = osp.join(EXT_ROOT, "data", DS, DS, "processed")
    return osp.join(d, "data.pt"), osp.join(d, "data_tree%s.pt" % tree_depth)


def ensure_prepared(S, args, DS):
    """Build the plain and the tree-augmented processed files once, under a cross-job lock."""
    p_plain, p_tree = processed_paths(DS, args.tree_depth)
    if osp.exists(p_plain) and osp.exists(p_tree):
        return "cached"
    lock = osp.join(EXT_ROOT, "data", DS, "prep.lock")
    t0 = time.time()
    while True:
        if acquire_lock(lock):
            try:
                if not (osp.exists(p_plain) and osp.exists(p_tree)):
                    print(f"[prep] building {DS} processed files (procs={PREP_PROCS})", flush=True)
                    splits = S["get_ad_split_TU"](args, fold=args.num_trial)       # plain TUDataset
                    S["get_ad_dataset_TU"](args, splits[0], pre=True)              # TUDataset_aug (trees)
                return f"built in {time.time() - t0:.0f}s"
            finally:
                try:
                    os.remove(lock)
                except OSError:
                    pass
        print(f"[prep] waiting for another job to build {DS}", flush=True)
        time.sleep(30)


# ----------------------------------------------------------------------------------------------------
# utilities copied from harness/run_eval.py (unchanged)
# ----------------------------------------------------------------------------------------------------
def eval_metrics(metric, y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        return dict(auroc=float("nan"), auprc=float("nan"), fpr95=float("nan"), nonfinite=int((~np.isfinite(s)).sum()))
    return dict(auroc=float(metric.ood_auc(y, s)), auprc=float(metric.ood_aupr(y, s)),
                fpr95=float(metric.fpr95(y, s)), nonfinite=0)


def finite_mean(t):
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(t.mean()) if t.size and np.all(np.isfinite(t)) else float("nan")


def rng_state():
    st = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


@contextlib.contextmanager
def rng_isolated():
    """Anything inside (held-out scoring) leaves python/numpy/torch CPU+CUDA RNG states untouched."""
    py, npst = random.getstate(), np.random.get_state()
    devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devs):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(npst)


def save_ckpt(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def acquire_lock(path):
    """One process per task across concurrent SLURM jobs (O_EXCL lock holding the job id); a lock
    whose job is no longer RUNNING is stale and taken over."""
    me = f"{os.environ.get('SLURM_JOB_ID', 'nojob')} {os.getpid()} {os.uname().nodename}"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = open(path).read().split()
            except OSError:
                owner = []
            job = owner[0] if owner else "nojob"
            alive = False
            if job != "nojob":
                q = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
                alive = q.returncode == 0 and "RUNNING" in q.stdout
            elif len(owner) >= 3 and owner[2] == os.uname().nodename:   # interactive run on this host
                try:
                    os.kill(int(owner[1]), 0)
                    alive = True
                except (OSError, ValueError):
                    alive = False
            if alive and job != os.environ.get("SLURM_JOB_ID", "nojob-x"):
                return False
            if alive and job == os.environ.get("SLURM_JOB_ID", "nojob-x") and len(owner) >= 2:
                try:                     # same job: a sibling process still holds it
                    if owner[2] == os.uname().nodename:
                        os.kill(int(owner[1]), 0)
                        return False
                except (OSError, ValueError, IndexError):
                    pass
            try:
                os.remove(path)
            except OSError:
                pass
    return False


def save_scores(path, epochs, S, extra):
    os.makedirs(osp.dirname(path), exist_ok=True)
    arrs = dict(epochs=np.asarray(epochs, dtype=np.int64))
    for k, v in S.items():
        arrs[k] = np.stack([np.asarray(r, dtype=np.float64) for r in v], 0) if len(v) else np.zeros((0, 0))
    arrs.update(extra)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def peak_mem():
    out = dict(maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
    if torch.cuda.is_available():
        out["cuda_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        out["cuda_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3
    return out


def chunks(lst, n):
    n = max(1, int(n))
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class Ctx:
    pass


def add_eval(C, epoch, test_main, hold_main, extra_test=None, extra_hold=None, rec_extra=None, t_eval0=None):
    """Record one evaluation point (as harness/run_eval.py). test_main: canonical-order anomaly scores."""
    rec = eval_metrics(C.metric, C.y_test, test_main)
    C.S["test"].append(np.asarray(test_main, dtype=np.float64))
    C.S["hold"].append(np.asarray(hold_main, dtype=np.float64).ravel())
    for k, v in (extra_test or {}).items():
        C.S.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
    for k, v in (extra_hold or {}).items():
        C.S.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
    rec["holdout_score"] = finite_mean(hold_main)
    rec["holdout_loss"] = rec["holdout_score"]      # deep_rerun naming (VAL-NORMAL signal)
    rec.update(rec_extra or {})
    rec["epoch"] = epoch
    rec["train_loss"] = C.train_loss[-1] if C.train_loss else float("nan")
    rec["t"] = time.time() - C.t_start
    C.traj.append(rec)
    if t_eval0 is not None:
        C.t_eval += time.time() - t_eval0
    if C.verbose:
        print(f"[{C.run_id}] ep {epoch} train {rec['train_loss']:.4f} test_auc {rec['auroc']:.4f} "
              f"hold {rec['holdout_score']:.4f} t={rec['t']:.0f}s", flush=True)


def maybe_stop(C, epoch, payload_fn):
    now = time.time()
    stop = (C.deadline is not None and now > C.deadline) or (TEST_STOP_EPOCH > 0 and epoch == TEST_STOP_EPOCH
                                                              and C.segments[-1]["start_epoch"] <= epoch)
    if stop or now - C.last_save > C.ckpt_every_s:
        p = payload_fn()
        p.update(epoch=epoch, traj=C.traj, train_loss=C.train_loss, rng=rng_state(), S=C.S, segments=C.segments)
        save_ckpt(C.ckpath, p)
        C.last_save = now
    if stop:
        print(f"[{C.run_id}] deadline reached at epoch {epoch}; checkpoint saved", flush=True)
    return stop


# ----------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------
def run(method, DS, fold, hp="ubgold", holdout_frac=0.2, deadline=None, max_epochs=None, ckpt_every_s=300.0,
        tag="", verbose=True):
    global PARTS, SCORES, CKPT_ROOT
    assert method == METHOD, method
    assert DS in DATASETS, DS
    if max_epochs is not None and "EP_PARTS" not in os.environ:   # smoke/timing runs never reach ext/parts
        smoke = osp.join(EXT_ROOT, "smoke")
        PARTS, SCORES, CKPT_ROOT = osp.join(smoke, "parts"), osp.join(smoke, "scores"), osp.join(smoke, "ckpt")
    t_start = time.time()
    run_id = f"{METHOD}_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + (f"_{tag}" if tag else "")
    os.makedirs(PARTS, exist_ok=True)
    out_json = osp.join(PARTS, run_id + ".json")
    if osp.exists(out_json):
        print(f"[skip] {out_json} exists")
        return "done"
    ckdir = osp.join(CKPT_ROOT, METHOD, DS)
    os.makedirs(ckdir, exist_ok=True)
    ckpath = osp.join(ckdir, run_id + ".pt")
    lock = ckpath + ".lock"
    if not acquire_lock(lock):
        print(f"[locked] {run_id} is running in another job")
        return "locked"
    try:
        return _run(DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s,
                    verbose, t_start)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def load_fold(S, args, DS, fold):
    """SEGO's own AD loaders for UB-GOLD trial `fold` (exactly treecl.py's two calls), split asserted."""
    splits = S["get_ad_split_TU"](args, fold=args.num_trial)
    ref = json.load(open(osp.join(SPLITS, f"{DS}.json")))
    rf = ref["folds"][fold]
    assert list(map(int, splits[fold][0])) == list(map(int, rf["split_train_index"]))
    assert list(map(int, splits[fold][1])) == list(map(int, rf["split_test_index"]))
    dataloader, _, meta = S["get_ad_dataset_TU"](args, splits[fold], pre=True)
    _, dataloader_test, _ = S["get_ad_dataset_TU"](args, splits[fold])
    return splits, dataloader, dataloader_test, meta


def _run(DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s, verbose,
         t_start):
    S = load_sego()
    from torch_geometric.data import Batch
    DataLoader = S["DataLoader"]
    HCL, HRN, HRNEncoder = S["HCL"], S["HRN"], S["HRNEncoder"]
    metric = load_ubgold_metric()
    import sklearn.metrics as skm
    impl = list(S["patches"])

    args, sh_line, hp_notes = resolve_hparams(S, DS, hp)
    if max_epochs is not None:   # smoke / test runs only (tag must say so)
        args.num_epoch = max_epochs
    seed = 3407 + fold
    S["setup_seed"](seed)        # CHANGE (NOTES 4): treecl.py uses setup_seed(trial + 1); A10: UB-GOLD's 3407+i
    prep = ensure_prepared(S, args, DS)
    splits, dataloader, dataloader_test, meta = load_fold(S, args, DS, fold)
    dataset_num_features = meta['num_feat']
    tree_input_dim = meta['deg_x']
    n_train_all = meta['num_train']

    # ---- CHANGE (NOTES 4): held-out normal set, identical carve-out to harness/run_eval.py -------------
    train_all = list(dataloader.dataset)
    n_all = len(train_all)
    assert n_all == n_train_all
    n_hold = int(round(holdout_frac * n_all))
    perm = np.random.RandomState(HOLDOUT_SEED + 1000 * fold).permutation(n_all)
    hold_pos = sorted(perm[:n_hold].tolist())
    fit_pos = sorted(perm[n_hold:].tolist())
    fit_list = [train_all[i] for i in fit_pos]
    hold_list = [train_all[i] for i in hold_pos]
    assert n_hold > 0
    for i, d in enumerate(fit_list):   # idx = position in the 80% fit set (unused by SEGO's loss)
        d["idx"] = i
    fb_keys = sorted(k for k in train_all[0].keys() if k.find('treePHLayer') >= 0)
    # CHANGE (NOTES 5): follow_batch=fb_keys, as SEGO's OOD loader does; the released AD loader omits it
    # and HRNEncoder then fails on data['treePHLayer1_batch'] (verify.py, probe_follow_batch)
    dataloader = DataLoader(fit_list, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)
    n_train = len(fit_list)

    # ---- canonical test order --------------------------------------------------------------------------
    test_data = dataloader_test.dataset
    n_test = len(test_data)
    for i, d in enumerate(test_data):
        d.gid = i
    y_test = np.array([int(d.y) for d in test_data], dtype=np.int64)

    C = Ctx()
    C.metric, C.y_test, C.run_id, C.verbose, C.t_start = metric, y_test, run_id, verbose, t_start
    C.traj, C.train_loss = [], []
    C.S = {"test": [], "hold": []}
    C.t_train, C.t_eval = 0.0, 0.0
    eval_epochs = set(range(args.eval_freq, args.num_epoch + 1, args.eval_freq)) | {args.num_epoch}  # CHANGE: + final

    def unscramble(scores, gids, y_shuffled=None):
        o = np.full(n_test, np.nan)
        gids = np.asarray(gids, dtype=np.int64).ravel()
        assert len(gids) == n_test and len(np.unique(gids)) == n_test
        if y_shuffled is not None:
            assert np.array_equal(y_test[gids], np.asarray(y_shuffled, dtype=np.int64).ravel())
        o[gids] = np.asarray(scores, dtype=np.float64).ravel()
        return o

    # held-out graphs pre-collated in chunks of batch_size_test (fixed order, no DataLoader -> no RNG)
    hold_batches = [Batch.from_data_list(c, follow_batch=fb_keys) for c in chunks(hold_list, min(args.batch_size_test, n_hold))]

    gi = np.loadtxt(osp.join(EXT_ROOT, "data", DS, DS, "raw", f"{DS}_graph_indicator.txt"), dtype=np.int64)
    meta_out = dict(method=METHOD, dataset=DS, fold=fold, hp=hp, seed=seed, run_id=run_id,
                    ubgold_sh_line=sh_line, hp_notes=hp_notes,
                    hparams={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in sorted(vars(args).items())},
                    n_train_normal_ubgold=n_all, n_fit=n_train, n_holdout=n_hold,
                    holdout_seed=HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=int(y_test.sum()),
                    max_nodes_num=int(np.bincount(gi).max()), device=None, torch=torch.__version__,
                    impl=None, cuda_name=None, threads=torch.get_num_threads(), max_epochs_override=max_epochs,
                    sego_commit=SEGO_COMMIT, tree_input_dim=int(tree_input_dim), prep=prep,
                    protocol_notes=dict(oracle="UB-GOLD rule (labelled; SEGO has no test-label rule)",
                                        last="SEGO's own reporting rule (final epoch)",
                                        valnormal="min mean held-out-normal SEGO score (A3)"))

    # ---- treecl.py: model / tree encoder / optimizers (verbatim order) -----------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    encoder = HRNEncoder(args.tree_depth, args.tree_pooling_type,
                         tree_input_dim, args.tree_hidden_dim,
                         args.hidden_dim*args.num_layer, args.tree_dropout,
                         args.tree_link_input, args.tree_drop_root,
                         device)
    treeM = HRN(encoder, args.hidden_dim*args.num_layer).to(device)
    treeOpt = torch.optim.Adam(treeM.parameters(), lr=args.tree_learning_rate)

    meta_out["device"] = str(device)
    meta_out["cuda_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    meta_out["impl"] = "; ".join(impl + [f"device {device.type}"])
    hbs = [b.to(device) for b in hold_batches]

    state = None
    if osp.exists(ckpath):
        state = torch.load(ckpath, map_location="cpu", weights_only=False)
        print(f"[resume] {ckpath} at epoch {state['epoch']}", flush=True)
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(state["epoch"] + 1 if state is not None else 1),
               t0=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.segments = (list(state.get("segments", [])) if state is not None else []) + [seg]
    meta_out["segments"] = C.segments
    meta_out["device_all"] = "+".join(sorted({x["device"] for x in C.segments}))
    C.last_save = time.time()
    C.deadline, C.ckpt_every_s, C.ckpath = deadline, ckpt_every_s, ckpath

    start_epoch = 1
    mean_b = std_b = mean_g = std_g = mean_n = std_n = None
    if state is not None:
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["opt"])
        treeM.load_state_dict(state["treeM"]); treeOpt.load_state_dict(state["treeOpt"])
        mean_b, std_b, mean_g, std_g, mean_n, std_n = state["adaptive"]
        C.traj, C.train_loss = state["traj"], state["train_loss"]
        C.S.update(state["S"])
        start_epoch = state["epoch"] + 1
        set_rng_state(state["rng"])
    payload = lambda: dict(model=model.state_dict(), opt=optimizer.state_dict(), treeM=treeM.state_dict(),
                           treeOpt=treeOpt.state_dict(), adaptive=(mean_b, std_b, mean_g, std_g, mean_n, std_n))

    def sego_score(m, data):
        """treecl.py's test-time scoring, verbatim (returns score and its g / n components)."""
        b, g_f, g_s, n_f, n_s = m(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
        y_score_g = m.calc_loss_g(g_f, g_s)
        y_score_n = m.calc_loss_n(n_f, n_s, data.batch)
        y_score_t = m.calc_loss_g(g_f, g_s)

        if args.is_adaptive:
            y_score = (y_score_t - mean_b)/std_b + (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
        else:
            y_score = y_score_t + y_score_g + y_score_n
        return y_score, y_score_g, y_score_n, g_f

    # ---- treecl.py training loop (verbatim apart from # CHANGE) -----------------------------------------
    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        weight_b = weight_g = weight_n = 1
        if args.is_adaptive:
            if epoch == 1:
                weight_b, weight_g, weight_n = 1, 1, 1
            else:
                weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                weight_sum = (weight_b  + weight_g  + weight_n) / 3
                weight_b, weight_g, weight_n = weight_b/weight_sum, weight_g/weight_sum, weight_n/weight_sum

        model.train()
        treeM.train()

        loss_all = 0
        if args.is_adaptive:
            loss_t_all, loss_g_all, loss_n_all = [], [], []

        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            treeOpt.zero_grad()

            b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
            x_hrn = treeM(data)
            loss_g = model.calc_loss_g(g_f, g_s)
            loss_n = model.calc_loss_n(n_f, n_s, data.batch)
            loss_t = model.calc_loss_tree(x_hrn, g_f)

            if args.is_adaptive:
                loss = weight_b * loss_t.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                loss_t_all = loss_t_all + loss_t.detach().cpu().tolist()
                loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
            else:
                loss = loss_t.mean() + loss_g.mean() + loss_n.mean()

            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        C.train_loss.append(loss_all / n_train)    # treecl prints loss_all / n_train ('[TRAIN] Epoch')

        if args.is_adaptive:
            mean_b, std_b = np.mean(loss_t_all), np.std(loss_t_all)
            mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
            mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
        C.t_train += time.time() - t0

        if epoch in eval_epochs:   # treecl: epoch % args.eval_freq == 0 (CHANGE: + final epoch)
            t1 = time.time()
            model.eval()
            treeM.eval()

            y_score_all = []
            y_true_all = []
            gid_all, g_all, n_all_ = [], [], []    # CHANGE: bookkeeping (canonical order, components)
            for data in dataloader_test:
                data = data.to(device)
                with torch.no_grad():   # CHANGE: treecl scores with autograd on (identical values)
                    y_score, y_score_g, y_score_n, _ = sego_score(model, data)
                y_true = data.y

                y_score_all = y_score_all + y_score.detach().cpu().tolist()
                y_true_all = y_true_all + y_true.detach().cpu().tolist()
                gid_all += data.gid.cpu().tolist()
                g_all += y_score_g.cpu().tolist()
                n_all_ += y_score_n.cpu().tolist()

            auc = skm.roc_auc_score(y_true_all, y_score_all)    # SEGO's own printed number
            test = unscramble(y_score_all, gid_all, y_true_all)
            # CHANGE: held-out normals, same scoring, fixed batches, RNG isolated; + objective (secondary)
            hs, hg, hn, objs = [], [], [], []
            with rng_isolated(), torch.no_grad():
                for hb in hbs:
                    s_, sg_, sn_, g_f = sego_score(model, hb)
                    x_hrn = treeM(hb)
                    l_t = model.calc_loss_tree(x_hrn, g_f)
                    objs.append(float(weight_b * l_t.mean() + weight_g * sg_.mean() + weight_n * sn_.mean()) * hb.num_graphs)
                    hs.append(s_.cpu().numpy()); hg.append(sg_.cpu().numpy()); hn.append(sn_.cpu().numpy())
            rec = dict(auroc_sego=float(auc), holdout_obj=sum(objs) / n_hold,
                       mean_b=float(mean_b) if mean_b is not None else None, std_b=float(std_b) if std_b is not None else None,
                       mean_g=float(mean_g) if mean_g is not None else None, std_g=float(std_g) if std_g is not None else None,
                       mean_n=float(mean_n) if mean_n is not None else None, std_n=float(std_n) if std_n is not None else None,
                       weights=[float(weight_b), float(weight_g), float(weight_n)])
            add_eval(C, epoch, test, np.concatenate(hs),
                     extra_test=dict(test_g=unscramble(g_all, gid_all), test_n=unscramble(n_all_, gid_all)),
                     extra_hold=dict(hold_g=np.concatenate(hg), hold_n=np.concatenate(hn)),
                     rec_extra=rec, t_eval0=t1)
            assert abs(C.traj[-1]["auroc"] - auc) < 1e-12, (C.traj[-1]["auroc"], auc)
        if maybe_stop(C, epoch, payload):
            return "incomplete"

    seg["t1"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_out["seconds"] = time.time() - t_start
    meta_out["seconds_train_this_segment"] = C.t_train
    meta_out["seconds_eval_this_segment"] = C.t_eval
    meta_out["epochs_this_segment"] = args.num_epoch - seg["start_epoch"] + 1
    meta_out["sec_per_epoch_train"] = C.t_train / max(1, meta_out["epochs_this_segment"])
    meta_out["n_eval"] = len(C.traj)
    meta_out["memory"] = peak_mem()
    meta_out["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    epochs = [r["epoch"] for r in C.traj]
    for k, v in C.S.items():
        assert len(v) == len(epochs), (k, len(v), len(epochs))
    npz = osp.join(SCORES, run_id + ".npz")
    save_scores(npz, epochs, C.S, dict(
        y_test=y_test, test_index=np.asarray(list(map(int, splits[fold][1])), dtype=np.int64),
        hold_pos=np.asarray(hold_pos, dtype=np.int64),
        test_n_nodes=np.array([int(d.num_nodes) for d in test_data], dtype=np.int64),
        test_n_edges=np.array([int(d.edge_index.size(1)) // 2 for d in test_data], dtype=np.int64),
        hold_n_nodes=np.array([int(d.num_nodes) for d in hold_list], dtype=np.int64),
        hold_n_edges=np.array([int(d.edge_index.size(1)) // 2 for d in hold_list], dtype=np.int64),
        num_epoch=np.int64(args.num_epoch), eval_freq=np.int64(args.eval_freq)))
    meta_out["scores_npz"] = npz
    res = dict(meta=meta_out, traj=C.traj, train_loss=C.train_loss)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_json)
    if osp.exists(ckpath):
        os.remove(ckpath)
    print(f"[done] {run_id} in {meta_out['seconds']:.0f}s -> {out_json}", flush=True)
    return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--hp", default="ubgold", choices=["ubgold", "hpfree"])
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke tests only (use with --tag)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EP_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    a = ap.parse_args()
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag (smoke runs must not be mistaken for real runs)")
    torch.set_num_threads(a.threads)
    st = run(a.method, a.dataset, a.fold, hp=a.hp, holdout_frac=a.holdout_frac, deadline=a.deadline,
             max_epochs=a.max_epochs, ckpt_every_s=a.ckpt_every_s, tag=a.tag)
    sys.exit(0 if st == "done" else 3)
