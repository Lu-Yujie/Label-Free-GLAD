"""evalpaper week-1 harness: instrumented UB-GOLD runs for all 12 deep detectors.

COPY-AND-EXTEND of m1_selection/run_m1.py (itself a copy of deep_rerun/run_rerun.py);
no existing prelim module is imported or edited. UB-GOLD (commit c940457) is copied unmodified to
./ubgold_src. Every change to UB-GOLD's training code is marked "# CHANGE".

One run = (method, dataset, fold, hp) where hp in {ubgold, hpfree}:
  * ubgold : UB-GOLD's per-dataset line from benchmark/Source/<M>.sh, parsed with UB-GOLD's own
             argparse parser (extracted verbatim from benchmark/mymain.py).
  * hpfree : label-free hyperparameters (see hpfree_args() and PREREG_AMENDMENTS.md A1).

At every UB-GOLD evaluation point (epoch % eval_freq == 0, plus the final epoch) we record
  - test AUROC / AUPRC / FPR95 (UB-GOLD's GAOOD/metric functions) and the per-graph test scores in
    canonical test order (labels are used ONLY to report metrics);
  - per-graph anomaly scores of the held-out 20% normal training graphs (same carve-out as deep_rerun:
    RandomState(2026 + 1000*fold), the model trains on the other 80%), their mean ("holdout_score",
    the VAL-NORMAL signal; for SIGNET/OCGTL/OCGIN/GLocalKD it is their per-graph loss, identical to
    deep_rerun's "holdout_loss"), and where cheap the training objective on the held-out graphs
    ("holdout_obj", secondary).
Held-out scoring never touches the training RNG streams (rng_isolated()).

Method-specific notes (details in NOTES.md):
  * GraphDE: UB-GOLD scores with infer_e_gx = P(in-distribution) (higher = more normal). Main score
    = -infer_e_gx (paper orientation, fixed by PREREG); the UB-GOLD orientation is recorded as
    auroc_ubgold/... (reproduction column only).
  * CVTGAD: UB-GOLD selects the checkpoint (fit) with the adaptive-normalised score but reports
    (predict) the raw score loss_g + loss_n. Main score = raw; normalised score = test_norm/auroc_sel.
  * GraphCL/InfoGraph + IF/OCSVM: the "checkpoint" is (encoder state, IF/OCSVM as fitted by UB-GOLD:
    refit on the embeddings of every training batch, so the detector at an evaluation point is the one
    fitted on the last training batch of that epoch). Main score = -detector.decision_function(g).
    UB-GOLD's fit/predict compute labels and scores from two different passes over the shuffled test
    loader (misaligned); we replicate that pass (RNG fidelity) and record its AUROC as
    auroc_ubgold_misaligned, but all reported metrics use aligned labels.

Outputs per run (EP_PARTS / EP_SCORES override):
  parts/<run_id>.json                       meta + per-eval-point metrics
  $GLAD_DATA_ROOT/evalpaper/scores/<run_id>.npz  per-graph score matrices
Resumable: model/optimizer/RNG/trajectory checkpoints every 5 min and at the job deadline. GLADC and
GLocalKD (batches fixed once before training) resume by deterministic replay of the pre-training
steps, verified with a fingerprint of the fixed batches.

Usage:
  PYTHONDONTWRITEBYTECODE=1 python -B run_eval.py --method GLADC --dataset BZR --fold 0 [--hp hpfree]
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # never import from cwd

import argparse
import ast
import contextlib
import json
import os
import os.path as osp
import pickle
import random
import re
import resource
import shlex
import sys
import time
import types
import importlib

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
SRC = osp.join(HERE, "ubgold_src")
OVERLAY = (os.environ.get("GLAD_ENVS", "envs") + "/evalpaper-overlay")   # faiss-cpu, PyGCL, torch_sparse/cluster, texttable, packaging
DATA_RUN = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_run")     # cwd for UB-GOLD loaders (./data/<DS>)
SPLITS = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_splits")     # verified UB-GOLD index lists
EP_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper")
CKPT_ROOT = os.environ.get("EP_CKPT", osp.join(EP_ROOT, "ckpt"))
PARTS = os.environ.get("EP_PARTS", osp.join(HERE, "parts"))
SCORES = os.environ.get("EP_SCORES", osp.join(EP_ROOT, "scores"))
HOLDOUT_SEED = 2026
PATIENCE = 30

OLD4 = ("SIGNET", "OCGTL", "OCGIN", "GLocalKD")
NEW8 = ("GLADC", "CVTGAD", "GOOD-D", "GraphDE", "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM")
ALL12 = NEW8 + OLD4
ALIASES = {"GOOD_D": "GOOD-D", "GOODD": "GOOD-D"}
SUBPARSER = {"CVTGAD", "GLADC", "SIGNET", "GLocalKD", "OCGTL", "OCGIN", "GraphCL_IF", "GraphCL_OCSVM",
             "InfoGraph_IF", "InfoGraph_OCSVM"}          # GOOD-D / GraphDE have no subparser
NEEDS_STR_ENC = {"GOOD-D", "CVTGAD"}                   # only these read data.x_s
TWO_STAGE = {"GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"}
REPLAY = {"GLADC", "GLocalKD"}                         # fixed pre-training batches -> replay on resume

# HP-FREE (PREREG_AMENDMENTS.md A1): mymain.py defaults except these per-dataset compute settings
HPFREE_KEEP = {"model", "exp_type", "DS", "DS_ood", "DS_pair", "data_root", "gpu", "num_trial",
               "batch_size", "batch_size_test"}
# options whose mymain default makes the method crash -> modal value over its UB-GOLD TU lines (A1)
HPFREE_CRASH_FALLBACK = {"InfoGraph_OCSVM": ["num_layer"]}

TU_DATASETS = ["AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES", "IMDB-BINARY", "NCI1",
               "PROTEINS_full", "REDDIT-BINARY"]


# ----------------------------------------------------------------------------------------
# environment / UB-GOLD code loading (no edits to the copied files)
# ----------------------------------------------------------------------------------------
def setup_overlay():
    if OVERLAY not in sys.path:
        sys.path.append(OVERLAY)       # appended: the ubgold-gpu venv keeps precedence
    if "dgl" not in sys.modules:       # PyGCL's GCL/utils.py imports dgl only for an unused helper
        m = types.ModuleType("dgl")
        m.DGLGraph = object

        def _no_dgl(*a, **k):
            raise RuntimeError("dgl stub: not available")
        m.graph = _no_dgl
        sys.modules["dgl"] = m


def install_faiss_gpu_fallback():
    """UB-GOLD pinned faiss-gpu 1.7.2 (no wheel for py3.12). With faiss-cpu, GOOD_D.run_kmeans
    calls faiss.StandardGpuResources() outside its try-block; we provide stand-ins whose GPU index
    constructor raises, so UB-GOLD's own `except: ... Try CPU` branch (IndexFlatL2) runs."""
    import faiss
    if hasattr(faiss, "StandardGpuResources"):
        return "faiss " + faiss.__version__ + " (gpu build)"

    class _Res:
        pass

    class _Cfg:
        useFloat16 = False

    def _gpu_index(*a, **k):
        raise RuntimeError("faiss-cpu build: no GPU index")
    faiss.StandardGpuResources = _Res
    faiss.GpuIndexFlatConfig = _Cfg
    faiss.GpuIndexFlatL2 = _gpu_index
    return "faiss-cpu " + faiss.__version__ + " (UB-GOLD CPU k-means fallback)"


def install_torch_scatter_shim():
    try:
        import torch_scatter  # noqa: F401
        return "torch_scatter " + getattr(torch_scatter, "__version__", "?")
    except ImportError:
        raise RuntimeError("run with $GLAD_ENVS/ubgold-gpu (torch_scatter required)")


def install_cpu_device_remap():
    """CPU mode (same approach as m1/m2): UB-GOLD hard-codes 'cuda' in several places
    (OCGTL/OCGIN/GLocalKD, GOOD_D.run_kmeans .cuda()). Map every CUDA target to cpu at runtime."""
    _to = torch.Tensor.to

    def _map(a):
        if isinstance(a, int) and not isinstance(a, bool):
            return "cpu"
        if isinstance(a, str) and a.startswith("cuda"):
            return "cpu"
        if isinstance(a, torch.device) and a.type == "cuda":
            return torch.device("cpu")
        return a

    def to(self, *args, **kw):
        args = tuple(_map(a) for a in args)
        if "device" in kw:
            kw["device"] = _map(kw["device"])
        return _to(self, *args, **kw)

    torch.Tensor.to = to
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    return "cpu (UB-GOLD 'cuda' remapped to cpu at runtime)"


def stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


def load_ubgold(method):
    """Import UB-GOLD modules without running GAOOD/__init__ or GAOOD/detector/__init__ (they import
    every detector incl. KernelGLAD/grakel). Returns a dict of modules."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    stub_package("GAOOD", osp.join(SRC, "GAOOD"))
    stub_package("GAOOD.nn", osp.join(SRC, "GAOOD", "nn"))
    stub_package("GAOOD.utils", osp.join(SRC, "GAOOD", "utils"))
    stub_package("GAOOD.detector", osp.join(SRC, "GAOOD", "detector"))
    M = {"metric": importlib.import_module("GAOOD.metric.metric")}
    importlib.import_module("GAOOD.metric")
    if method in OLD4:
        for k in ("signet", "ocgtl", "ocgin", "glocalkd"):
            M[k] = importlib.import_module("GAOOD.nn." + k)
        return M
    modname = {"GOOD-D": "GOOD_D"}.get(method, method)
    M["det_mod"] = importlib.import_module("GAOOD.detector." + modname)
    cls_name = {"GOOD-D": "GOOD_D"}.get(method, method)
    cls = getattr(M["det_mod"], cls_name)
    # CHANGE: build_save_path() creates and CLEARS GAOOD/model_save/<model>/<exp>/<DS> (shared by
    # concurrent folds); we never save best-test models, so it is disabled (in memory only).
    cls.build_save_path = lambda self: setattr(self, "path", None)
    M["cls"] = cls
    # UB-GOLD's init_model(args), extracted verbatim from benchmark/utils.py
    ns = {cls_name: cls}
    extract_functions(osp.join(SRC, "benchmark", "utils.py"), ["init_model"], ns)
    M["init_model"] = ns["init_model"]
    return M


def extract_functions(path, names, ns):
    tree = ast.parse(open(path).read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in funcs) == sorted(names), [f.name for f in funcs]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns


def load_ubgold_data_functions():
    import os.path as osp_
    import networkx as nx
    from scipy import sparse as sp
    from sklearn.model_selection import StratifiedKFold
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Constant
    from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
    ns = dict(os=os, osp=osp_, re=re, sp=sp, torch=torch, np=np, random=random, nx=nx,
              DataLoader=DataLoader, Constant=Constant, TUDataset=TUDataset,
              to_scipy_sparse_matrix=to_scipy_sparse_matrix, degree=degree, from_networkx=from_networkx,
              StratifiedKFold=StratifiedKFold)
    extract_functions(osp.join(SRC, "dataloader", "data_loader.py"),
                      ["get_ad_split_TU", "get_ad_dataset_TU", "init_structural_encoding",
                       "read_graph_file", "get_ad_dataset_Tox21"], ns)
    extract_functions(osp.join(SRC, "benchmark", "mymain.py"), ["set_seed"], ns)
    return ns


def build_ubgold_parser():
    """UB-GOLD's argparse parser, extracted verbatim from benchmark/mymain.py's __main__ block."""
    path = osp.join(SRC, "benchmark", "mymain.py")
    tree = ast.parse(open(path).read())
    main_if = [n for n in tree.body if isinstance(n, ast.If)][-1]
    body = [s for s in main_if.body
            if not (isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "args")
            and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                     and getattr(s.value.func, "id", "") == "main")]
    ns = {"argparse": argparse}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns["parser"]


def sh_lines(method):
    """All `-exp_type ad` lines of benchmark/Source/<M>.sh as {DS: (line, tokens-after-script)}."""
    sh = osp.join(SRC, "benchmark", "Source", f"{method}.sh")
    out = {}
    for raw in open(sh):
        s = raw.strip()
        if not re.match(r"^python3? +benchmark/mymain\.py", s):
            continue
        toks = shlex.split(s, comments=True)[2:]
        if "-exp_type" in toks and toks[toks.index("-exp_type") + 1] == "ad" and "-DS" in toks:
            DS = toks[toks.index("-DS") + 1]
            out.setdefault(DS, (s, toks))     # first line wins (as a reader of the script would)
    return out


def fix_known_defects(method, DS, toks):
    """UB-GOLD script defects that make its own parser fail (PREREG_AMENDMENTS.md A2)."""
    notes = []
    if method == "GraphDE" and DS == "REDDIT-BINARY":
        i = toks.index("-batch_size_test")
        if i + 1 < len(toks) and toks[i + 1].startswith("-"):
            toks = toks[:i] + toks[i + 1:]   # dangling '-batch_size_test' (no value) -> mymain default 9999
            notes.append("dropped dangling '-batch_size_test' (no value in GraphDE.sh) -> default 9999")
    return toks, notes


def modal_value(method, key, parser):
    from collections import Counter
    vals = []
    for DS, (line, toks) in sh_lines(method).items():
        if DS in TU_DATASETS:
            t2, _ = fix_known_defects(method, DS, list(toks))
            vals.append(getattr(parser.parse_args(t2), key))
    cnt = Counter(vals)
    top = max(cnt.values())
    return sorted([v for v, c in cnt.items() if c == top], key=lambda v: (str(type(v)), v))[0]


def resolve_hparams(method, DS, hp):
    parser = build_ubgold_parser()
    lines = sh_lines(method)
    if DS not in lines:
        raise KeyError(f"no UB-GOLD line for {method} {DS}")
    line, toks = lines[DS]
    toks, notes = fix_known_defects(method, DS, list(toks))
    args = parser.parse_args(toks)
    if hp == "ubgold":
        return args, line, notes
    assert hp == "hpfree", hp
    dtoks = ["-exp_type", "ad", "-DS", DS, "-model", method] + ([method] if method in SUBPARSER else [])
    dflt = parser.parse_args(dtoks)
    changed = {}
    for k, v in vars(dflt).items():
        if k not in HPFREE_KEEP and getattr(args, k, None) != v:
            changed[k] = (getattr(args, k, None), v)
            setattr(args, k, v)
    for k in vars(args):
        if k not in vars(dflt) and k not in HPFREE_KEEP:   # option absent from the default namespace
            raise RuntimeError(f"hpfree: {k} has no mymain default for {method}")
    for k in HPFREE_CRASH_FALLBACK.get(method, []):
        mv = modal_value(method, k, parser)
        notes.append(f"hpfree crash fallback: {k}={getattr(args, k)} -> modal {mv}")
        setattr(args, k, mv)
    notes.append("hpfree: reset to mymain defaults: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(changed.items())))
    return args, "HPFREE " + " ".join(dtoks) + f"  [batch_size {args.batch_size} batch_size_test {args.batch_size_test} from UB-GOLD line]", notes


# ----------------------------------------------------------------------------------------
# metrics / RNG / checkpoint utilities
# ----------------------------------------------------------------------------------------
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
    import subprocess
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


# ----------------------------------------------------------------------------------------
# GLocalKD.process_graph, verbatim (method; `self` only supplies max_nodes_num)
# ----------------------------------------------------------------------------------------
def glocalkd_process_graph(self, data):
    import scipy.sparse as sp
    num_graphs = data.ptr.size(0) - 1
    adj_matrixs = []
    adj_labels = []
    graph_appends = []
    graph_labes = []
    for i in range(num_graphs):
        graph_labes.append(int(data.y[i]))
        graph_x_size = int(data.ptr[i + 1]) - int(data.ptr[i])
        graph_x = data.x[data.ptr[i]:data.ptr[i + 1]]
        graph_append = torch.zeros((self.max_nodes_num, graph_x.shape[1]), dtype=float)
        for graph_index in range(graph_x_size):
            graph_append[graph_index, :] = graph_x[graph_index, :]
        node_indices = torch.tensor(range(data.ptr[i], data.ptr[i + 1]))
        adj_matrix = torch.zeros((graph_x_size, graph_x_size), dtype=torch.float32)
        edge_indices = torch.stack(
            [data.edge_index[0][torch.where(torch.isin(data.edge_index[1], node_indices))[0]],
             data.edge_index[1][torch.where(torch.isin(data.edge_index[1], node_indices))[0]]], dim=0)
        edge_indices = torch.sub(edge_indices, int(data.ptr[i]))
        adj_matrix[edge_indices[0], edge_indices[1]] = 1
        if self.max_nodes_num > graph_x_size:
            adj_padded = torch.zeros((self.max_nodes_num, self.max_nodes_num))
            adj_padded[:graph_x_size, :graph_x_size] = adj_matrix
            adj_label = adj_padded.numpy() + sp.eye(adj_padded.shape[0])
            adj_label = torch.tensor(adj_label)
            adj_matrixs.append(adj_padded)
            adj_labels.append(adj_label)
            graph_appends.append(graph_append)
        else:
            adj_label = adj_matrix.numpy() + sp.eye(adj_matrix.shape[0])
            adj_label = torch.tensor(adj_label)
            adj_matrixs.append(adj_matrix)
            adj_labels.append(adj_label)
            graph_appends.append(graph_append)
    adj_matrixs = torch.stack(adj_matrixs)
    adj_labels = torch.stack(adj_labels)
    graph_appends = torch.stack(graph_appends)
    graph_labes = torch.tensor(graph_labes)
    return adj_matrixs, adj_labels, graph_appends, graph_labes


def fingerprint(tensors):
    """Order-sensitive summary of fixed pre-training batches (replay check on resume): per batch its
    shape and the per-graph sums (no full-size temporaries; REDDIT batches are ~17 GB)."""
    return [[list(t.shape)] + [round(float(v), 4) for v in t.reshape(t.shape[0], -1).sum(1, dtype=torch.float64)]
            for t in tensors]


# ----------------------------------------------------------------------------------------
# run context
# ----------------------------------------------------------------------------------------
class Ctx:
    pass


def run(method, DS, fold, hp="ubgold", holdout_frac=0.2, deadline=None, max_epochs=None, ckpt_every_s=300.0,
        tag="", verbose=True):
    method = ALIASES.get(method, method)
    assert method in ALL12, method
    t_start = time.time()
    run_id = f"{method}_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + (f"_{tag}" if tag else "")
    os.makedirs(PARTS, exist_ok=True)
    out_json = osp.join(PARTS, run_id + ".json")
    if osp.exists(out_json):
        print(f"[skip] {out_json} exists")
        return "done"
    ckdir = osp.join(CKPT_ROOT, method, DS)
    os.makedirs(ckdir, exist_ok=True)
    ckpath = osp.join(ckdir, run_id + ".pt")
    lock = ckpath + ".lock"
    if not acquire_lock(lock):
        print(f"[locked] {run_id} is running in another job")
        return "locked"
    try:
        return _run(method, DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs,
                    ckpt_every_s, verbose, t_start)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _run(method, DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s,
         verbose, t_start):
    setup_overlay()
    impl = [install_torch_scatter_shim()]
    if method == "GOOD-D":
        impl.append(install_faiss_gpu_fallback())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        impl.append(install_cpu_device_remap())
    M = load_ubgold(method)
    metric = M["metric"]
    ns = load_ubgold_data_functions()

    args, sh_line, hp_notes = resolve_hparams(method, DS, hp)
    args.gpu = 0
    if max_epochs is not None:  # smoke tests only (tag must say so)
        args.num_epoch = max_epochs

    # ---- UB-GOLD trial `fold`: set_seed(3407 + fold), then its split + dataset code -------------
    seed = 3407 + fold
    ns["set_seed"](seed)
    is_tox = DS.startswith("Tox21")
    cwd = os.getcwd()
    os.chdir(DATA_RUN)
    try:
        if is_tox:
            # CHANGE (as deep_rerun): need_str_enc only for the methods that read x_s (no RNG used)
            out = ns["get_ad_dataset_Tox21"](args, need_str_enc=method in NEEDS_STR_ENC)
            split_test_index = list(range(len(out[5].dataset)))
        else:
            splits = ns["get_ad_split_TU"](args, fold=args.num_trial)
            out = ns["get_ad_dataset_TU"](args, splits[fold], need_str_enc=method in NEEDS_STR_ENC)
            split_test_index = list(map(int, splits[fold][1]))
    finally:
        os.chdir(cwd)
    (dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta) = out
    ref = json.load(open(osp.join(SPLITS, f"{DS}.json")))
    if is_tox:
        assert len(dataloader.dataset) == ref["n_train"] and len(dataloader_test.dataset) == ref["n_test"]
        assert [int(d.y) for d in dataloader_test.dataset] == list(map(int, ref["y_test"]))
    else:
        rf = ref["folds"][fold]
        assert list(map(int, splits[fold][0])) == list(map(int, rf["split_train_index"]))
        assert list(map(int, splits[fold][1])) == list(map(int, rf["split_test_index"]))

    args.max_nodes_num = meta["max_nodes_num"]
    args.dataset_num_features = meta["num_feat"]
    args.n_train = meta["num_train"]
    args.n_edge_feat = meta["num_edge_feat"]

    # ---- CHANGE: held-out normal set (fixed seed, independent of torch RNG; = deep_rerun) ---------
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch
    train_all = list(dataloader.dataset)
    n_all = len(train_all)
    n_hold = int(round(holdout_frac * n_all))
    perm = np.random.RandomState(HOLDOUT_SEED + 1000 * fold).permutation(n_all)
    hold_pos = sorted(perm[:n_hold].tolist())
    fit_pos = sorted(perm[n_hold:].tolist())
    fit_list = [train_all[i] for i in fit_pos]
    hold_list = [train_all[i] for i in hold_pos]
    assert n_hold > 0
    for i, d in enumerate(fit_list):   # CHANGE: idx = position in the 80% fit set (GOOD-D's b_all rows)
        d["idx"] = i
    dataloader = DataLoader(fit_list, batch_size=args.batch_size, shuffle=True)
    args.n_train = len(fit_list)

    # ---- canonical test order: tag test graphs with their position (collated as data.gid) --------
    assert dataloader_val is dataloader_test
    test_data = dataloader_test.dataset
    n_test = len(test_data)
    for i, d in enumerate(test_data):
        d.gid = i
    y_test = np.array([int(d.y) for d in test_data], dtype=np.int64)
    n_test_anom = int(y_test.sum())

    C = Ctx()
    C.method, C.DS, C.fold, C.args, C.device, C.metric = method, DS, fold, args, device, metric
    C.dataloader, C.dataloader_val, C.fit_list, C.hold_list = dataloader, dataloader_val, fit_list, hold_list
    C.n_test, C.n_hold, C.y_test, C.run_id, C.verbose, C.t_start = n_test, n_hold, y_test, run_id, verbose, t_start
    C.M = M
    C.traj, C.train_loss = [], []
    C.S = {"test": [], "hold": []}
    C.t_train, C.t_eval = 0.0, 0.0
    C.eval_epochs = set(range(args.eval_freq, args.num_epoch + 1, args.eval_freq)) | {args.num_epoch}

    def unscramble(scores, gids, y_shuffled):
        o = np.full(n_test, np.nan)
        gids = np.asarray(gids, dtype=np.int64).ravel()
        assert len(gids) == n_test and len(np.unique(gids)) == n_test
        assert np.array_equal(y_test[gids], np.asarray(y_shuffled, dtype=np.int64).ravel())
        o[gids] = np.asarray(scores, dtype=np.float64).ravel()
        return o
    C.unscramble = unscramble
    # held-out graphs, pre-collated in chunks of batch_size_test (fixed order, no DataLoader -> no RNG)
    C.hold_batches = [Batch.from_data_list(c) for c in chunks(hold_list, min(args.batch_size_test, n_hold))]

    meta_out = dict(method=method, dataset=DS, fold=fold, hp=hp, seed=seed, run_id=run_id,
                    ubgold_sh_line=sh_line, hp_notes=hp_notes,
                    hparams={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in sorted(vars(args).items())},
                    n_train_normal_ubgold=n_all, n_fit=len(fit_list), n_holdout=n_hold,
                    holdout_seed=HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=n_test_anom,
                    max_nodes_num=int(args.max_nodes_num), device=str(device), torch=torch.__version__,
                    impl="; ".join(impl), cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                    threads=torch.get_num_threads(), max_epochs_override=max_epochs)

    C.state = None
    if osp.exists(ckpath):
        C.state = torch.load(ckpath, map_location="cpu", weights_only=False)
        print(f"[resume] {ckpath} at epoch {C.state['epoch']}", flush=True)
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(C.state["epoch"] + 1 if C.state is not None else 1),
               t0=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.segments = (list(C.state.get("segments", [])) if C.state is not None else []) + [seg]
    meta_out["segments"] = C.segments
    meta_out["device_all"] = "+".join(sorted({x["device"] for x in C.segments}))
    C.last_save = time.time()
    C.deadline, C.ckpt_every_s, C.ckpath = deadline, ckpt_every_s, ckpath

    fn = {"SIGNET": fit_signet, "OCGTL": fit_ocg, "OCGIN": fit_ocg, "GLocalKD": fit_glocalkd,
          "GLADC": fit_gladc, "CVTGAD": fit_cvtgad, "GOOD-D": fit_goodd, "GraphDE": fit_graphde}.get(method, fit_two_stage)
    st = fn(C)
    if st != "done":
        return st

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
        y_test=y_test, test_index=np.asarray(split_test_index, dtype=np.int64),
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


# ---------------------------------------------------------------------------------------------
# shared bookkeeping
# ---------------------------------------------------------------------------------------------
def add_eval(C, epoch, test_main, hold_main, extra_test=None, extra_hold=None, rec_extra=None, t_eval0=None):
    """Record one evaluation point. test_main: canonical-order anomaly scores (main orientation)."""
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


TEST_STOP_EPOCH = int(os.environ.get("EP_TEST_STOP_EPOCH", "0"))   # resume tests only


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


def restore_common(C):
    st = C.state
    C.traj, C.train_loss = st["traj"], st["train_loss"]
    C.S.update(st["S"])
    return st["epoch"] + 1


# =============================================================================================
# SIGNET / OCGTL / OCGIN / GLocalKD : copies of m1_selection/run_m1.py (pseudo-set removed)
# =============================================================================================
def fit_signet(C):
    args, device, M = C.args, C.device, C.M
    model = M["signet"].SIGNET(input_dim=args.dataset_num_features, input_dim_edge=args.n_edge_feat,
                               args=args, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    hbs = [b.to(device) for b in C.hold_batches]
    start_epoch = 1
    if C.state is not None:
        model.load_state_dict(C.state["model"]); optimizer.load_state_dict(C.state["opt"])
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
    payload = lambda: dict(model=model.state_dict(), opt=optimizer.state_dict())
    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        model.train()
        loss_all = 0
        num_sample = 0
        for data in C.dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            y, y_hyper, node_imp, edge_imp = model(data)
            loss = model.loss_nce(y, y_hyper).mean()
            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
            loss.backward()
            optimizer.step()
        C.train_loss.append(loss_all / max(num_sample, 1))
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:  # UB-GOLD: epoch % eval_freq == 0 (+ CHANGE: final epoch)
            t1 = time.time()
            model.eval()
            all_ad_true, all_ad_score, all_gid = [], [], []
            for data in C.dataloader_val:
                all_ad_true.append(data.y.cpu())
                all_gid.append(data.gid.cpu())
                data = data.to(device)
                with torch.no_grad():
                    y, y_hyper, _, _ = model(data)
                    ano_score = model.loss_nce(y, y_hyper)
                all_ad_score.append(ano_score.cpu())
            test = C.unscramble(torch.cat(all_ad_score).numpy(), torch.cat(all_gid).numpy(), torch.cat(all_ad_true).numpy())
            with rng_isolated(), torch.no_grad():   # CHANGE: held-out normals, same scoring as test
                hs = []
                for hb in hbs:
                    y, y_hyper, _, _ = model(hb)
                    hs.append(model.loss_nce(y, y_hyper).cpu().numpy())
            add_eval(C, epoch, test, np.concatenate(hs), t_eval0=t1)
        # CHANGE: no early stopping and no torch.save of the best-test-AUROC model
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


def fit_ocg(C):
    args, device, M = C.args, C.device, C.M
    if C.method == "OCGTL":
        model = M["ocgtl"].ocgtl(dim_features=args.dataset_num_features, args=args).to(device)
    else:
        model = M["ocgin"].OCGIN(dim_features=args.dataset_num_features, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    hbs = [b.to(device) for b in C.hold_batches]
    start_epoch = 1
    if C.state is not None:
        model.load_state_dict(C.state["model"]); optimizer.load_state_dict(C.state["opt"])
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
        if start_epoch > args.eval_freq:
            model.eval()  # UB-GOLD never switches back to train() after the first evaluation
    payload = lambda: dict(model=model.state_dict(), opt=optimizer.state_dict())

    def forward_model(data, eval_):
        emb = model(data)
        return model.loss_func(emb, eval_)

    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        all_loss, n_bw = 0, 0
        for data in C.dataloader:
            n_bw += 1
            data = data.to(device)
            loss_epoch = forward_model(data, False)
            loss_mean = loss_epoch.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            all_loss += loss_epoch.sum().detach()
        mean_loss = all_loss.item() / args.n_train
        C.train_loss.append(mean_loss)
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:
            t1 = time.time()
            model.eval()
            y_val, score_val, gid_val = [], [], []
            for data in C.dataloader_val:
                data = data.to(device)
                with torch.no_grad():  # (UB-GOLD computes without no_grad; values identical)
                    score_epoch = forward_model(data, True)
                score_val = score_val + score_epoch.detach().cpu().tolist()
                y_val = y_val + data.y.detach().cpu().tolist()
                gid_val = gid_val + data.gid.detach().cpu().tolist()
            test = C.unscramble(score_val, gid_val, y_val)
            with rng_isolated(), torch.no_grad():
                hs = np.concatenate([forward_model(hb, True).cpu().numpy() for hb in hbs])
            add_eval(C, epoch, test, hs, t_eval0=t1)
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


def fit_glocalkd(C):
    """GAOOD/detector/GLocalKD.py fit(). Batches are fixed once before training; on resume the
    pre-training steps are replayed (same RNG consumption) and verified by fingerprint."""
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch_geometric.data import Batch
    args, device, M = C.args, C.device, C.M
    G = M["glocalkd"]
    selfobj = types.SimpleNamespace(max_nodes_num=args.max_nodes_num)
    model_teacher = G.GcnEncoderGraph_teacher(input_dim=args.dataset_num_features, hidden_dim=args.hidden_dim,
                                              embedding_dim=args.output_dim, label_dim=2,
                                              num_layers=args.num_layer, bn=args.bn, args=args).to(device)
    model_student = G.GcnEncoderGraph_student(input_dim=args.dataset_num_features, hidden_dim=args.hidden_dim,
                                              embedding_dim=args.output_dim, label_dim=2,
                                              num_layers=args.num_layer, bn=args.bn, args=args).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    preprocessed = []
    for batch_idx, data in enumerate(C.dataloader):
        adj, _, h0, _ = glocalkd_process_graph(selfobj, data)
        preprocessed.append(tuple([adj, h0]))
    preprocessed_val = []
    val_gids = []
    for batch_idx, data in enumerate(C.dataloader_val):
        adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, data)
        preprocessed_val.append(tuple([adj_matrixs, graph_appends, graph_label]))
        val_gids += data.gid.cpu().tolist()
    preprocessed_hold = []   # CHANGE: held-out normals, one graph at a time like the test set
    for g in C.hold_list:
        adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, Batch.from_data_list([g]))
        preprocessed_hold.append(tuple([adj_matrixs, graph_appends, graph_label]))
    fp = fingerprint([a for a, _ in preprocessed]) + [val_gids]
    start_epoch = 1
    if C.state is not None:   # CHANGE: replay-resume
        assert C.state["fp"] == fp, "GLocalKD replay mismatch"
        model_student.load_state_dict(C.state["student"]); model_teacher.load_state_dict(C.state["teacher"])
        optimizer.load_state_dict(C.state["opt"]); scheduler.load_state_dict(C.state["sched"])
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
    payload = lambda: dict(student=model_student.state_dict(), teacher=model_teacher.state_dict(),
                           opt=optimizer.state_dict(), sched=scheduler.state_dict(), fp=fp)

    def forward_model(adj, h0):
        adj = Variable(adj.float(), requires_grad=False).to("cuda")
        h0 = Variable(h0.float(), requires_grad=False).to("cuda")
        embed_node, embed = model_student(h0, adj)
        embed_teacher_node, embed_teacher = model_teacher(h0, adj)
        embed_teacher = embed_teacher.detach()
        embed_teacher_node = embed_teacher_node.detach()
        loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction="none"), dim=2).mean(dim=1).mean(dim=0)
        loss = F.mse_loss(embed, embed_teacher, reduction="none").mean(dim=1).mean(dim=0)
        return loss + loss_node

    def score_list(prep):
        loss, y = [], []
        for adj_matrixs, graph_appends, graph_label in prep:
            adj = Variable(adj_matrixs.float(), requires_grad=False).to("cuda")
            h0 = Variable(graph_appends.float(), requires_grad=False).to("cuda")
            with torch.no_grad():
                embed_node, embed = model_student(h0, adj)
                embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction="none"), dim=2).mean(dim=1)
            loss_graph = F.mse_loss(embed, embed_teacher, reduction="none").mean(dim=1)
            loss_ = loss_graph + loss_node
            loss.append(np.array(loss_.cpu().detach()))
            y.append(1 if int(graph_label) != 0 else 0)
        return np.array(loss).ravel(), y

    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        total_loss = 0.0
        nb = 0
        model_student.train()
        for batch_idx, data in enumerate(C.dataloader):
            model_student.zero_grad()
            adj, h0 = preprocessed[batch_idx]
            loss = forward_model(adj, h0)
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            nb += 1
        C.train_loss.append(total_loss / max(nb, 1))
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:
            t1 = time.time()
            model_student.eval()
            s, y = score_list(preprocessed_val)
            test = C.unscramble(s, val_gids, y)
            with rng_isolated():
                sh, _ = score_list(preprocessed_hold)
            add_eval(C, epoch, test, sh, t_eval0=t1)
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# =============================================================================================
# GLADC: GAOOD/detector/GLADC.py fit()
# =============================================================================================
def fit_gladc(C):
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch_geometric.data import Batch
    args, device = C.args, C.device
    det = C.M["init_model"](args)                 # UB-GOLD benchmark/utils.py init_model (verbatim)
    det.device = device                           # fit(): torch.device('cuda:'+str(args.gpu) if cuda else 'cpu')
    det.NetGe, det.noise_NetG = det.init_model(**det.kwargs)
    optimizerG = torch.optim.Adam(det.NetGe.parameters(), lr=det.lr)
    preprocessed = []
    for batch_idx, data in enumerate(C.dataloader):
        adj_matrixs, adj_labels, graph_appends, _ = det.process_graph(data)
        preprocessed.append(tuple([adj_matrixs, adj_labels, graph_appends]))
    preprocessed_val = []
    val_gids = []
    for batch_idx, data in enumerate(C.dataloader_val):
        adj_matrixs, _, graph_appends, graph_label = det.process_graph(data)
        preprocessed_val.append(tuple([adj_matrixs, graph_appends, graph_label]))
        val_gids += data.gid.cpu().tolist()
    preprocessed_hold = []   # CHANGE: held-out normals, one graph at a time like the test set
    for g in C.hold_list:
        adj_matrixs, _, graph_appends, graph_label = det.process_graph(Batch.from_data_list([g]))
        preprocessed_hold.append(tuple([adj_matrixs, graph_appends, graph_label]))
    # secondary signal: training objective on the held-out graphs (only when cheap in memory)
    hold_obj_batches = None
    if args.max_nodes_num <= 1000:
        hold_obj_batches = []
        for ch in chunks(C.hold_list, det.batch_size):
            a, al, ga, _ = det.process_graph(Batch.from_data_list(ch))
            hold_obj_batches.append((a, al, ga))
    fp = fingerprint([a for a, _, _ in preprocessed]) + [val_gids]
    start_epoch = 1
    if C.state is not None:   # CHANGE: replay-resume
        assert C.state["fp"] == fp, "GLADC replay mismatch"
        det.NetGe.load_state_dict(C.state["NetGe"]); det.noise_NetG.load_state_dict(C.state["noise_NetG"])
        optimizerG.load_state_dict(C.state["opt"])
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
    payload = lambda: dict(NetGe=det.NetGe.state_dict(), noise_NetG=det.noise_NetG.state_dict(),
                           opt=optimizerG.state_dict(), fp=fp)

    def score_prep(prep):
        loss, y = [], []
        for adj_matrixs, graph_appends, graph_label in prep:
            adj = Variable(adj_matrixs.float().float(), requires_grad=False).to(det.device)
            h0 = Variable(graph_appends.float(), requires_grad=False).to(det.device)
            with torch.no_grad():   # CHANGE: UB-GOLD scores with autograd on (identical values)
                x1_r, Feat_0 = det.NetGe.shared_encoder(h0, adj)
                x_fake, s_fake, x2, Feat_1 = det.NetGe(x1_r, adj)
                loss_node = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
                loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
                loss_ = loss_node + loss_graph
            loss_ = np.array(loss_.cpu().detach())
            loss.append(loss_)
            if int(graph_label) != 0:
                y.append(1)
            else:
                y.append(0)
        return np.array(loss).ravel(), y

    for epoch in range(start_epoch, det.num_epochs + 1):
        t0 = time.time()
        total_lossG = 0.0
        det.NetGe.train()
        for adj_matrixs, adj_labels, graph_appends in preprocessed:
            lossG = det.forward_model(adj_matrixs, adj_labels, graph_appends)
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            total_lossG += lossG.detach()   # CHANGE: detach (UB-GOLD keeps every graph alive; same value)
        C.train_loss.append(float(total_lossG) / max(1, len(preprocessed)))
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:   # UB-GOLD: (epoch) % args.eval_freq == 0 (+ CHANGE: final epoch)
            t1 = time.time()
            det.NetGe.eval()
            s, y = score_prep(preprocessed_val)
            test = C.unscramble(s, val_gids, y)
            rec = {}
            with rng_isolated():
                sh, _ = score_prep(preprocessed_hold)
                if hold_obj_batches is not None:
                    with torch.no_grad():
                        objs = [float(det.forward_model(a, al, ga)) * a.shape[0] for a, al, ga in hold_obj_batches]
                    rec["holdout_obj"] = sum(objs) / C.n_hold
            add_eval(C, epoch, test, sh, rec_extra=rec, t_eval0=t1)
        # CHANGE: no early stopping, no torch.save of the best-test-AUROC model
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# =============================================================================================
# CVTGAD: GAOOD/detector/CVTGAD.py fit()
# =============================================================================================
def fit_cvtgad(C):
    args, device = C.args, C.device
    det = C.M["init_model"](args)
    det.device = device
    det.model = det.init_model(**det.kwargs)      # NB: sets args.dg_dim = args.rw_dim (UB-GOLD)
    optimizer = torch.optim.Adam(det.model.parameters(), lr=args.lr)
    hbs = [b.to(device) for b in C.hold_batches]
    start_epoch = 1
    std_g = std_n = mean_g = mean_n = None
    if C.state is not None:
        det.model.load_state_dict(C.state["model"]); optimizer.load_state_dict(C.state["opt"])
        std_g, std_n, mean_g, mean_n = C.state["adaptive"]
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
    payload = lambda: dict(model=det.model.state_dict(), opt=optimizer.state_dict(),
                           adaptive=(std_g, std_n, mean_g, mean_n))
    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        if args.is_adaptive:
            if epoch == 1:
                weight_g, weight_n = 1, 1
            else:
                weight_g, weight_n = std_g ** args.alpha, std_n ** args.alpha
                weight_sum = (weight_g + weight_n) / 2
                weight_g, weight_n = weight_g / weight_sum, weight_n / weight_sum
        det.model.train()
        loss_all = 0
        if args.is_adaptive:
            loss_g_all, loss_n_all = [], []
        for data in C.dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss_g, loss_n = det.forward_model(data, C.dataloader, args)
            if args.is_adaptive:
                loss = weight_g * loss_g.mean() + weight_n * loss_n.mean()
                loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
            else:
                loss = loss_g.mean() + loss_n.mean()
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        if args.is_adaptive:
            mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
            mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
        C.train_loss.append(loss_all / args.n_train)
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:
            t1 = time.time()
            det.model.eval()
            y_score_all, y_raw_all, y_true_all, gid_all = [], [], [], []
            for data in C.dataloader_val:
                data = data.to(device)
                with torch.no_grad():   # CHANGE: UB-GOLD scores with autograd on (identical values)
                    y_score_g, y_score_n = det.forward_model(data, C.dataloader_val, args)
                if args.is_adaptive:
                    y_score = (y_score_g - mean_g) / std_g + (y_score_n - mean_n) / std_n
                else:
                    y_score = y_score_g + y_score_n
                y_true = data.y
                y_score_all = y_score_all + y_score.detach().cpu().tolist()
                y_raw_all = y_raw_all + (y_score_g + y_score_n).detach().cpu().tolist()   # = predict() score
                y_true_all = y_true_all + y_true.detach().cpu().tolist()
                gid_all = gid_all + data.gid.cpu().tolist()
            test_raw = C.unscramble(y_raw_all, gid_all, y_true_all)
            test_norm = C.unscramble(y_score_all, gid_all, y_true_all)
            sel = eval_metrics(C.metric, C.y_test, test_norm)
            hg, hn = [], []
            with rng_isolated(), torch.no_grad():
                for hb in hbs:
                    g_, n_ = det.forward_model(hb, None, args)
                    hg.append(g_.cpu().numpy()); hn.append(n_.cpu().numpy())
            hg, hn = np.concatenate(hg), np.concatenate(hn)
            hold_raw = hg + hn
            hold_norm = (hg - mean_g) / std_g + (hn - mean_n) / std_n if args.is_adaptive else hold_raw
            wg, wn = (weight_g, weight_n) if args.is_adaptive else (1, 1)
            rec = dict(auroc_sel=sel["auroc"], auprc_sel=sel["auprc"], fpr95_sel=sel["fpr95"],
                       holdout_score_norm=finite_mean(hold_norm),
                       holdout_obj=float(wg * hg.mean() + wn * hn.mean()))
            add_eval(C, epoch, test_raw, hold_raw, extra_test={"test_norm": test_norm},
                     extra_hold={"hold_norm": hold_norm}, rec_extra=rec, t_eval0=t1)
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# =============================================================================================
# GOOD-D: GAOOD/detector/GOOD_D.py fit()
# =============================================================================================
def fit_goodd(C):
    args = C.args
    det = C.M["init_model"](args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # fit()
    det.device = device                          # (DeepDetector: self.device = args.gpu = 0 -> cuda:0)
    det.model = det.init_model(**det.kwargs)
    optimizer = torch.optim.Adam(det.model.parameters(), lr=args.lr)
    det.model.train()
    det.decision_score_ = None
    det.train_dataloader = C.dataloader
    hbs = [b.to(device) for b in C.hold_batches]
    start_epoch = 1
    std_b = std_g = std_n = None
    if C.state is not None:
        det.model.load_state_dict(C.state["model"]); optimizer.load_state_dict(C.state["opt"])
        std_b, std_g, std_n = C.state["adaptive"]
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
        det.model.eval()   # get_cluster_result() leaves the model in eval() from the first batch on
    payload = lambda: dict(model=det.model.state_dict(), opt=optimizer.state_dict(), adaptive=(std_b, std_g, std_n))
    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        if args.is_adaptive:
            if epoch == 1:
                weight_b, weight_g, weight_n = 1, 1, 1
            else:
                weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                weight_sum = (weight_b + weight_g + weight_n) / 3
                weight_b, weight_g, weight_n = weight_b / weight_sum, weight_g / weight_sum, weight_n / weight_sum
        if args.is_adaptive:
            loss_b_all, loss_g_all, loss_n_all = [], [], []
        y_score_all = []
        loss_all = 0
        for data in C.dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss_epoch, score_epoch = det.forward_model(data, C.dataloader, args)
            loss_g, loss_b, loss_n = loss_epoch
            y_score_b, y_score_g, y_score_n = score_epoch
            if args.is_adaptive:
                loss = weight_b * loss_b.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                loss_b_all = loss_b_all + loss_b.detach().cpu().tolist()
                loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
            else:
                loss = loss_b.mean() + loss_g.mean() + loss_n.mean()
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            if args.is_adaptive:
                mean_b, std_b = np.mean(loss_b_all), np.std(loss_b_all)
                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
            if args.is_adaptive:
                y_score = (y_score_b - mean_b) / std_b + (y_score_g - mean_g) / std_g + (y_score_n - mean_n) / std_n
            else:
                y_score = y_score_b + y_score_g + y_score_n
            y_score_all = y_score_all + y_score.detach().cpu().tolist()
        C.train_loss.append(loss_all / args.n_train)
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:
            t1 = time.time()
            det.model.eval()
            y_val, score_val, gid_val = [], [], []
            cluster_result = None
            for data in C.dataloader_val:
                data = data.to(device)
                with torch.no_grad():   # CHANGE: UB-GOLD scores with autograd on (identical values)
                    emb, emb_list = det.model(data)
                    cluster_result = det.get_cluster_result(det.train_dataloader, det.model, args)
                    y_score_b, y_score_g, y_score_n = det.model.score_func(emb, emb_list, data, cluster_result)
                y_score = y_score_b + y_score_g + y_score_n
                y_true = data.y
                y_val = y_val + y_true.detach().cpu().tolist()
                score_val = score_val + y_score.detach().cpu().tolist()
                gid_val = gid_val + data.gid.cpu().tolist()
            test = C.unscramble(score_val, gid_val, y_val)
            hs = []
            with rng_isolated(), torch.no_grad():   # same model state -> same k-means as the last test batch
                for hb in hbs:
                    emb, emb_list = det.model(hb)
                    b_, g_, n_ = det.model.score_func(emb, emb_list, hb, cluster_result)
                    hs.append((b_ + g_ + n_).cpu().numpy())
            add_eval(C, epoch, test, np.concatenate(hs), t_eval0=t1)
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# =============================================================================================
# GraphDE: GAOOD/detector/GraphDE.py fit()
# =============================================================================================
def fit_graphde(C):
    args = C.args
    det = C.M["init_model"](args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det.device = device                          # (DeepDetector: self.device = args.gpu = 0 -> cuda:0)
    det.model = det.init_model(**det.kwargs)
    optimizer = torch.optim.Adam(det.model.parameters(), lr=args.lr)
    det.model.train()
    det.decision_score_ = None
    hbs = [b.to(device) for b in C.hold_batches]
    start_epoch = 1
    if C.state is not None:
        det.model.load_state_dict(C.state["model"]); optimizer.load_state_dict(C.state["opt"])
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
        if start_epoch > args.eval_freq:
            det.model.eval()   # UB-GOLD never switches back to train() after the first evaluation
    payload = lambda: dict(model=det.model.state_dict(), opt=optimizer.state_dict())
    for epoch in range(start_epoch, det.epoch + 1):
        t0 = time.time()
        all_loss, n_bw = 0, 0
        for data in C.dataloader:
            n_bw += 1
            data = data.to(device)
            loss_epoch, score_epoch = det.forward_model(data, C.dataloader, args)
            all_loss += loss_epoch
        all_loss /= n_bw
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        C.train_loss.append(float(all_loss))
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:
            t1 = time.time()
            det.model.eval()
            y_val, score_val, gid_val = [], [], []
            for data in C.dataloader_val:
                data = data.to(device)
                with torch.no_grad():   # CHANGE: UB-GOLD scores with autograd on (identical values)
                    score_epoch = det.model.infer_e_gx(data.x, data.edge_index, data.batch)
                score_val = score_val + score_epoch.detach().cpu().tolist()
                y_true = data.y
                y_val = y_val + y_true.detach().cpu().tolist()
                gid_val = gid_val + data.gid.cpu().tolist()
            e_test = C.unscramble(score_val, gid_val, y_val)          # UB-GOLD orientation (P(in-dist))
            ub = eval_metrics(C.metric, C.y_test, e_test)
            he, objs = [], []
            with rng_isolated(), torch.no_grad():
                for hb in hbs:
                    he.append(det.model.infer_e_gx(hb.x, hb.edge_index, hb.batch).cpu().numpy())
                    l_, _ = det.model.loss_func(hb)
                    objs.append(float(l_) * hb.num_graphs)
            he = np.concatenate(he)
            rec = dict(auroc_ubgold=ub["auroc"], auprc_ubgold=ub["auprc"], fpr95_ubgold=ub["fpr95"],
                       holdout_score_ubgold=finite_mean(he), holdout_obj=sum(objs) / C.n_hold)
            # main (paper) orientation: anomaly score = -infer_e_gx  (PREREG: fixed, never from test)
            add_eval(C, epoch, -e_test, -he, rec_extra=rec, t_eval0=t1)
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# =============================================================================================
# GraphCL_IF / GraphCL_OCSVM / InfoGraph_IF / InfoGraph_OCSVM: GAOOD/detector/<M>.py fit()
# =============================================================================================
def fit_two_stage(C):
    from torch.optim import Adam
    args = C.args
    is_gcl = C.method.startswith("GraphCL")
    det = C.M["init_model"](args)
    det.device = C.device                        # fit(): cuda:args.gpu if available else cpu
    device = det.device
    det.init_model(**det.kwargs)
    optimizer = Adam(det.encoder_model.parameters(), lr=args.lr)
    hbs = [b.to(device) for b in C.hold_batches]
    start_epoch = 1
    last_fit = None   # embeddings the detector was last fitted on (diagnostic)
    if C.state is not None:
        det.encoder_model.load_state_dict(C.state["encoder"]); optimizer.load_state_dict(C.state["opt"])
        det.detector = pickle.loads(C.state["detector"])
        last_fit = C.state["last_fit"]
        start_epoch = restore_common(C)
        set_rng_state(C.state["rng"])
    payload = lambda: dict(encoder=det.encoder_model.state_dict(), opt=optimizer.state_dict(),
                           detector=pickle.dumps(det.detector), last_fit=last_fit)

    def encode(data):
        if is_gcl:
            _, g, _, _, g1, g2 = det.encoder_model(data.x, data.edge_index, data.batch)
        else:
            z, g = det.encoder_model(data.x, data.edge_index, data.batch)
        return g

    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        det.encoder_model.train()
        epoch_loss = 0
        for data in C.dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            if is_gcl:
                _, g, _, _, g1, g2 = det.encoder_model(data.x, data.edge_index, data.batch)
                last_fit = g.cpu().detach().numpy()
                det.detector.fit(g.cpu().detach().numpy())
                g1, g2 = [det.encoder_model.encoder.project(g) for g in [g1, g2]]
                loss = det.contrast_model(g1=g1, g2=g2, batch=data.batch)
            else:
                z, g = det.encoder_model(data.x, data.edge_index, data.batch)
                z, g = det.encoder_model.project(z, g)
                last_fit = g.cpu().detach().numpy()
                det.detector.fit(g.cpu().detach().numpy())
                loss = det.contrast_model(h=z, g=g, batch=data.batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        C.train_loss.append(epoch_loss / max(1, len(C.dataloader)))
        C.t_train += time.time() - t0
        if epoch in C.eval_epochs:
            t1 = time.time()
            det.encoder_model.eval()
            # UB-GOLD: labels from one pass and scores from a SECOND pass over the shuffled test loader
            ys = torch.cat([data.y for data in C.dataloader_val])
            y_score_all, y_true_all, gid_all = [], [], []
            for data in C.dataloader_val:
                data = data.to(device)
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                with torch.no_grad():   # CHANGE: UB-GOLD scores with autograd on (identical values)
                    g = encode(data)
                anomaly_scores = -det.detector.decision_function(g.cpu().detach().numpy())
                y_score_all = y_score_all + list(anomaly_scores)
                y_true_all = y_true_all + data.y.cpu().tolist()       # aligned labels (ours)
                gid_all = gid_all + data.gid.cpu().tolist()
            mis = C.metric.ood_auc(ys, y_score_all)                    # UB-GOLD's (misaligned) val_auc
            test = C.unscramble(y_score_all, gid_all, y_true_all)
            hs, objs = [], []
            with rng_isolated(), torch.no_grad():
                for hb in hbs:
                    hs.append(-det.detector.decision_function(encode(hb).cpu().numpy()))
                    if is_gcl:
                        _, g, _, _, g1, g2 = det.encoder_model(hb.x, hb.edge_index, hb.batch)
                        g1, g2 = [det.encoder_model.encoder.project(g) for g in [g1, g2]]
                        objs.append(float(det.contrast_model(g1=g1, g2=g2, batch=hb.batch)) * hb.num_graphs)
                    else:
                        z, g = det.encoder_model(hb.x, hb.edge_index, hb.batch)
                        z, g = det.encoder_model.project(z, g)
                        objs.append(float(det.contrast_model(h=z, g=g, batch=hb.batch)) * hb.num_graphs)
                fitref = float(np.mean(-det.detector.decision_function(last_fit)))
            rec = dict(auroc_ubgold_misaligned=float(mis), holdout_obj=sum(objs) / C.n_hold,
                       fitref_score=fitref, n_last_fit=int(len(last_fit)))
            add_eval(C, epoch, test, np.concatenate(hs), rec_extra=rec, t_eval0=t1)
        if maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
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
