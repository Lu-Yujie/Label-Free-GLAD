"""M1 kill test: label-free checkpoint selection with typed pseudo-anomalies.

COPY of deep_rerun/run_rerun.py (deep_rerun is never edited), extended so that at EVERY
UB-GOLD evaluation point (epoch % eval_freq == 0, plus the final epoch) it saves per-graph
anomaly scores of
  (a) the test set (canonical test-index order; labels are used ONLY for reporting metrics),
  (b) the held-out 20% normal training graphs (identical carve-out as deep_rerun),
  (c) the pseudo-anomaly validation set P built by pseudo.py from the held-out normals only
      (one perturbed copy per held-out graph and applicable family; cached on disk).
Everything else (UB-GOLD models, data/split code, per-dataset hyperparameters from the .sh
files, seeds 3407+fold, held-out seed 2026+1000*fold, training loops, evaluation schedule) is
identical to deep_rerun, so ORACLE/LAST/VAL-NORMAL are reproducible from the new runs.
Changes relative to deep_rerun are marked "# M1".

P scoring never touches the training RNG streams: P graphs are pre-collated (no DataLoader) and
scored under torch.random.fork_rng; the forwards are in eval mode and the four models consume no
RNG there anyway. SIGNET's score (NCE loss) depends on the batch, so for SIGNET each family is
scored as ONE batch = [held-out normals ; their perturbed copies] (mixed, like the mixed test
batch); the normals' scores in that batch are stored as pneg_<family>. For OCGTL/OCGIN/GLocalKD
per-graph scores do not depend on the batch and pneg_<family> = held-out scores.

Outputs per run:
  $GLAD_DATA_ROOT/m1_scores/<run_id>.npz  epochs (E,), test (E,n_test), y_test, hold (E,n_hold),
        ppos_<fam>/pneg_<fam> (E,n_hold), test_index, hold_pos, n_nodes/n_edges of test/hold, ...
  m1_selection/parts/<run_id>.json           meta + per-eval-point metrics (like deep_rerun)

Usage:
  PYTHONDONTWRITEBYTECODE=1 python -B run_m1.py --method SIGNET --dataset AIDS --fold 0
"""
import os.path as _osp, sys as _sys  # M1: this directory holds select.py, which would shadow the
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))  # stdlib module `select` (used by
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # subprocess): drop it from sys.path

import argparse
import ast
import json
import os
import os.path as osp
import random
import re
import shlex
import sys
import time
import types
import importlib

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
SRC = osp.join(HERE, "ubgold_src")
DATA_RUN = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_run")      # cwd for UB-GOLD loaders (./data/<DS>/<DS>/raw)
SPLITS = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_splits")      # verified UB-GOLD index lists
CKPT_ROOT = os.environ.get("M1_CKPT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m1_ckpt"))  # M1: resumable training state
PARTS = os.environ.get("M1_PARTS", osp.join(HERE, "parts"))                  # M1: per-run trajectories (JSON)
SCORES = os.environ.get("M1_SCORES", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m1_scores"))  # M1: per-graph score matrices
PSEUDO_DIR = os.environ.get("M1_PSEUDO", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m1_pseudo"))  # M1: cached P (pseudo.py)
HOLDOUT_SEED = 2026  # fixed; the held-out normal set depends only on (dataset, fold)
PATIENCE = 30        # UB-GOLD's early-stopping patience (N = 30), used only in aggregate.py
GPU_ONLY = {"OCGTL", "OCGIN", "GLocalKD"}  # UB-GOLD hard-codes 'cuda' in these


# ----------------------------------------------------------------------------------------
# UB-GOLD code loading (no edits to the copied files)
# ----------------------------------------------------------------------------------------
def install_torch_scatter_shim():
    """Use the real torch_scatter when installed (GPU venv). Otherwise (CPU venv) provide the
    two functions UB-GOLD uses: scatter(..., reduce='mean'/'sum') and scatter_max(...)[0]."""
    try:
        import torch_scatter  # noqa: F401
        return "torch_scatter " + getattr(torch_scatter, "__version__", "?")
    except ImportError:
        pass
    from torch_geometric.utils import scatter as pyg_scatter
    m = types.ModuleType("torch_scatter")

    def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        assert out is None
        return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)

    class _ScatterMax(torch.autograd.Function):
        """torch_scatter.scatter_max semantics: empty segments -> 0, gradient routed to the
        (first) argmax element only, and src is NOT saved for backward (SIGNET modifies it
        in place right after the call, which torch_scatter tolerates)."""

        @staticmethod
        def forward(ctx, src, index, d, n_out):
            shape = [1] * src.dim()
            shape[d] = -1
            idx = index.view(shape).expand_as(src)
            size = list(src.shape)
            size[d] = n_out
            out = src.new_zeros(size).scatter_reduce(d, idx, src, reduce="amax", include_self=False)
            L = src.size(d)
            pos = torch.arange(L, device=src.device).view(shape).expand_as(src)
            cand = torch.where(src == out.gather(d, idx), pos, torch.full_like(pos, L))
            arg = torch.full(size, L, dtype=torch.long, device=src.device).scatter_reduce(
                d, idx, cand, reduce="amin", include_self=True)
            ctx.save_for_backward(arg)
            ctx.d, ctx.src_shape = d, list(src.shape)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (arg,) = ctx.saved_tensors
            size = list(ctx.src_shape)
            size[ctx.d] += 1
            g = grad_out.new_zeros(size).scatter_(ctx.d, arg, grad_out)
            return g.narrow(ctx.d, 0, ctx.src_shape[ctx.d]), None, None, None

    def scatter_max(src, index, dim=-1, out=None, dim_size=None):
        assert out is None
        d = dim if dim >= 0 else src.dim() + dim
        n_out = int(index.max()) + 1 if dim_size is None else dim_size
        return _ScatterMax.apply(src, index, d, n_out), None  # argmax is never used by UB-GOLD

    m.scatter = scatter
    m.scatter_max = scatter_max
    sys.modules["torch_scatter"] = m
    return "shim (torch_geometric.utils.scatter / Tensor.scatter_reduce amax)"


def install_cpu_device_remap():
    """M1 (CPU mode, M1_CPU_REMAP=1; same approach as m2_typed/run_m2.py): run the
    CUDA-hard-coded UB-GOLD models (OCGTL/OCGIN/GLocalKD) on a CPU node WITHOUT editing them.
    UB-GOLD moves data/parameters to the GPU only through Tensor.to('cuda' | 0 | torch.device('cuda'))
    (incl. PyG Data.to(args.gpu)), Tensor.cuda() and Module.cuda(); these entry points are wrapped
    so that any CUDA device maps to 'cpu'. Every operation is otherwise unchanged."""
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
    """Register a package without executing its __init__ (GAOOD/__init__ imports every
    detector, which pulls in dgl/faiss/ogb that these four models never use)."""
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


def load_ubgold_nn():
    stub_package("GAOOD", osp.join(SRC, "GAOOD"))
    stub_package("GAOOD.nn", osp.join(SRC, "GAOOD", "nn"))
    stub_package("GAOOD.metric", osp.join(SRC, "GAOOD", "metric"))
    mods = {k: importlib.import_module("GAOOD.nn." + k) for k in ("signet", "ocgtl", "ocgin", "glocalkd")}
    mods["metric"] = importlib.import_module("GAOOD.metric.metric")
    return mods


def extract_functions(path, names, ns):
    tree = ast.parse(open(path).read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in funcs) == sorted(names), [f.name for f in funcs]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns


def load_ubgold_data_functions():
    import os.path as osp_
    from scipy import sparse as sp
    from sklearn.model_selection import StratifiedKFold
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Constant
    from torch_geometric.utils import to_scipy_sparse_matrix, degree
    ns = dict(os=os, osp=osp_, re=re, sp=sp, torch=torch, np=np, random=random,
              DataLoader=DataLoader, Constant=Constant, TUDataset=TUDataset,
              to_scipy_sparse_matrix=to_scipy_sparse_matrix, degree=degree,
              StratifiedKFold=StratifiedKFold)
    extract_functions(osp.join(SRC, "dataloader", "data_loader.py"),
                      ["get_ad_split_TU", "get_ad_dataset_TU", "init_structural_encoding"], ns)
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


def ubgold_hparams(method, DS):
    """Parse the UB-GOLD run line for (method, DS) from benchmark/Source/<method>.sh."""
    sh = osp.join(SRC, "benchmark", "Source", f"{method}.sh")
    line = None
    for raw in open(sh):
        s = raw.strip()
        if not s.startswith("python benchmark/mymain.py"):
            continue
        toks = shlex.split(s)
        if "-exp_type" in toks and toks[toks.index("-exp_type") + 1] == "ad" and \
                "-DS" in toks and toks[toks.index("-DS") + 1] == DS:
            line = s
            break
    parser = build_ubgold_parser()
    if line is None:
        if DS != "MUTAG":
            raise KeyError(f"no UB-GOLD line for {method} {DS}")
        line = modal_line(method, sh, parser)  # MUTAG is not a UB-GOLD dataset
    args = parser.parse_args(shlex.split(line, comments=True)[2:])
    return args, line


def modal_line(method, sh, parser):
    """Label-free config for a dataset UB-GOLD never tuned (MUTAG): for every option, the most
    frequent value over the method's 11 TU `-exp_type ad` lines (Tox21 excluded); ties go to
    mymain's default if it is among the tied values, else to the smaller value."""
    from collections import Counter
    tu = ["AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES", "IMDB-BINARY", "NCI1",
          "PROTEINS_full", "REDDIT-BINARY"]
    parsed = []
    for raw in open(sh):
        toks = shlex.split(raw.strip()) if raw.strip().startswith("python benchmark/mymain.py") else []
        if toks and toks[toks.index("-exp_type") + 1] == "ad" and toks[toks.index("-DS") + 1] in tu:
            parsed.append(vars(parser.parse_args(toks[2:])))
    assert len(parsed) == 11, len(parsed)
    default = vars(parser.parse_args(["-model", method, method]))
    out = {}
    for k in parsed[0]:
        if k in ("DS",):
            continue
        cnt = Counter(d[k] for d in parsed)
        top = max(cnt.values())
        tied = [v for v, c in cnt.items() if c == top]
        out[k] = default[k] if default[k] in tied else sorted(tied, key=lambda v: (str(type(v)), v))[0]
    toks = ["python", "benchmark/mymain.py", "-exp_type", "ad", "-DS", "MUTAG", "-model", method]
    opt = {a.dest: a.option_strings[0] for a in parser._actions if a.option_strings}
    sub = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"][0].choices[method]
    sopt = {a.dest: a.option_strings[0] for a in sub._actions if a.option_strings}
    main_opts = [f"{opt[k]} {out[k]}" for k in sorted(out) if k in opt and k not in ("model", "exp_type", "DS")
                 and out[k] != default[k] and out[k] is not None]
    sub_opts = [f"{sopt[k]} {out[k]}" for k in sorted(out) if k in sopt and k not in opt and out[k] != default[k]]
    return " ".join(toks + main_opts + [method] + sub_opts) + "   # modal UB-GOLD TU config (MUTAG)"


# ----------------------------------------------------------------------------------------
# metrics
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
    return float(t.mean()) if np.all(np.isfinite(t)) else float("nan")


# ----------------------------------------------------------------------------------------
# GLocalKD.process_graph, verbatim (it is a method; `self` only supplies max_nodes_num)
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


# ----------------------------------------------------------------------------------------
# checkpointing (resumable across SLURM jobs)
# ----------------------------------------------------------------------------------------
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


def save_ckpt(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def acquire_lock(path):
    """One process per task across concurrent SLURM jobs: O_EXCL lock file holding the job id;
    a lock whose job is no longer in the queue is stale and taken over."""
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
            if alive and job != os.environ.get("SLURM_JOB_ID"):
                return False
            try:
                os.remove(path)
            except OSError:
                pass
    return False


# ----------------------------------------------------------------------------------------
# M1: fold preparation for pseudo.py (same code path as _run; consumes no RNG)
# ----------------------------------------------------------------------------------------
def holdout_positions(n_all, fold, holdout_frac=0.2):
    """Identical to deep_rerun's carve-out (fixed seed, independent of torch RNG)."""
    n_hold = int(round(holdout_frac * n_all))
    perm = np.random.RandomState(HOLDOUT_SEED + 1000 * fold).permutation(n_all)
    return sorted(perm[:n_hold].tolist()), sorted(perm[n_hold:].tolist())


def prepare_fold(DS, fold, holdout_frac=0.2):
    """UB-GOLD split + dataset for (DS, fold) and the held-out positions, without training.
    Used by pseudo.py to build P from exactly the held-out normals that _run carves out
    (_run re-checks this via a per-graph signature)."""
    install_torch_scatter_shim()
    ns = load_ubgold_data_functions()
    args, _ = ubgold_hparams("OCGIN", DS)  # data code only reads args.DS / batch sizes / num_trial
    cwd = os.getcwd()
    os.chdir(DATA_RUN)
    try:
        splits = ns["get_ad_split_TU"](args, fold=args.num_trial)
        out = ns["get_ad_dataset_TU"](args, splits[fold], need_str_enc=False)
    finally:
        os.chdir(cwd)
    ref = json.load(open(osp.join(SPLITS, f"{DS}.json")))["folds"][fold]
    assert list(map(int, splits[fold][0])) == list(map(int, ref["split_train_index"]))
    train_all = list(out[3].dataset)
    hold_pos, fit_pos = holdout_positions(len(train_all), fold, holdout_frac)
    return dict(train_all=train_all, hold_pos=hold_pos, fit_pos=fit_pos, test=list(out[5].dataset),
                test_index=list(map(int, splits[fold][1])), meta=out[6])


def load_pseudo(DS, fold, hold_list, hold_pos):
    """M1: load the cached P for (DS, fold), check it was built from exactly these held-out
    graphs, and return {family: [Data, ...]} aligned with hold_list."""
    import zlib
    C = torch.load(osp.join(PSEUDO_DIR, f"{DS}_fold{fold}.pt"), weights_only=False)
    assert C["hold_pos"] == list(map(int, hold_pos)), "P cache built from a different held-out set"
    for g, s in zip(hold_list, C["hold_sig"]):
        assert int(g.num_nodes) == s["n"] and \
            zlib.crc32(np.ascontiguousarray(g.edge_index.numpy()).tobytes()) == s["ei_hash"], "held-out graph mismatch"
    P = {}
    for f in C["families"]:
        lst = []
        for g, p in zip(hold_list, C["P"][f]):
            d = g.clone()
            assert p["x"].shape == g.x.shape and p["x"].dtype == g.x.dtype
            d.x = p["x"].clone()
            d.edge_index = p["edge_index"].clone().long()
            assert d.edge_index.size(1) == g.edge_index.size(1)  # size-preserving (m)
            lst.append(d)
        P[f] = lst
    return C["families"], P


def auc_pos_neg(neg, pos):
    """AUROC (%) of pos (pseudo-anomalies) vs neg (held-out normals); NaN if non-finite."""
    from sklearn.metrics import roc_auc_score
    neg, pos = np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64)
    s = np.r_[neg, pos]
    if not np.all(np.isfinite(s)) or len(neg) == 0 or len(pos) == 0:
        return float("nan")
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], s))


def save_scores(path, epochs, S, extra):
    """M1: per-graph score matrices (eval points x graphs) as a compressed npz."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    arrs = dict(epochs=np.asarray(epochs, dtype=np.int64))
    for k, v in S.items():
        arrs[k] = np.stack([np.asarray(r, dtype=np.float64) for r in v], 0) if len(v) else np.zeros((0, 0))
    arrs.update(extra)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
def run(method, DS, fold, holdout_frac=0.2, deadline=None, max_epochs=None, ckpt_every_s=300.0,
        tag="", verbose=True):
    t_start = time.time()
    run_id = f"{method}_{DS}_fold{fold}" + (f"_{tag}" if tag else "")
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
        return _run(method, DS, fold, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs,
                    ckpt_every_s, verbose, t_start)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _run(method, DS, fold, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s,
         verbose, t_start):

    scatter_impl = install_torch_scatter_shim()
    M = load_ubgold_nn()
    metric = M["metric"]
    ns = load_ubgold_data_functions()

    args, sh_line = ubgold_hparams(method, DS)
    if args is None:
        raise KeyError(f"no UB-GOLD hyperparameters for {method} {DS}")
    args.gpu = 0
    if max_epochs is not None:  # smoke tests only
        args.num_epoch = max_epochs

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    cpu_remap = os.environ.get("M1_CPU_REMAP", "0") == "1"  # M1: CPU mode (debug CPU partition)
    if cpu_remap:
        assert device.type == "cpu", "M1_CPU_REMAP=1 is meant for CPU nodes"
        scatter_impl += "; " + install_cpu_device_remap()
    if method in GPU_ONLY and device.type != "cuda" and not cpu_remap:
        raise RuntimeError(f"{method} hard-codes CUDA in UB-GOLD; run it on a GPU node (or set M1_CPU_REMAP=1)")

    # ---- UB-GOLD trial `fold`: set_seed(3407 + fold), then its split + dataset code -------
    seed = 3407 + fold
    ns["set_seed"](seed)
    cwd = os.getcwd()
    os.chdir(DATA_RUN)
    try:
        splits = ns["get_ad_split_TU"](args, fold=args.num_trial)
        # CHANGE: need_str_enc=False. The RW/degree encodings (x_s) are only read by
        # GOOD-D/GraphDE, never by SIGNET/OCGTL/OCGIN/GLocalKD; computing them uses no RNG.
        (dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test,
         meta) = ns["get_ad_dataset_TU"](args, splits[fold], need_str_enc=False)
    finally:
        os.chdir(cwd)
    # sanity: identical to the verified UB-GOLD split export
    ref = json.load(open(osp.join(SPLITS, f"{DS}.json")))["folds"][fold]
    assert list(map(int, splits[fold][0])) == list(map(int, ref["split_train_index"]))
    assert list(map(int, splits[fold][1])) == list(map(int, ref["split_test_index"]))

    args.max_nodes_num = meta["max_nodes_num"]
    args.dataset_num_features = meta["num_feat"]
    args.n_train = meta["num_train"]
    args.n_edge_feat = meta["num_edge_feat"]

    # ---- CHANGE: carve out a held-out normal set (fixed seed, independent of torch RNG) ----
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
    if n_hold > 0:
        dataloader = DataLoader(fit_list, batch_size=args.batch_size, shuffle=True)
        args.n_train = len(fit_list)
    # pre-collated (no DataLoader -> consumes no torch RNG)
    hold_batch = Batch.from_data_list(hold_list) if n_hold > 0 else None
    n_test = len(dataloader_test.dataset)
    n_test_anom = int(sum(int(d.y) for d in dataloader_test.dataset))

    # ---- M1: canonical test order. Tag every test graph with its position in UB-GOLD's
    # test list (= split test_index order); the shuffled test loader then reports which graph
    # each score belongs to. Adds one field to the collated batch; consumes no RNG.
    assert dataloader_val is dataloader_test
    test_data = dataloader_test.dataset
    assert isinstance(test_data, list) and len(test_data) == n_test
    for i, d in enumerate(test_data):
        d.gid = i
    y_test_canon = np.array([int(d.y) for d in test_data], dtype=np.int64)
    test_n_nodes = np.array([int(d.num_nodes) for d in test_data], dtype=np.int64)
    test_n_edges = np.array([int(d.edge_index.size(1)) // 2 for d in test_data], dtype=np.int64)
    hold_n_nodes = np.array([int(d.num_nodes) for d in hold_list], dtype=np.int64)
    hold_n_edges = np.array([int(d.edge_index.size(1)) // 2 for d in hold_list], dtype=np.int64)
    # ---- M1: pseudo-anomaly validation set P (cached, built from the held-out normals only)
    families, P = load_pseudo(DS, fold, hold_list, hold_pos)
    S = {"test": [], "hold": []}           # per-eval-point per-graph scores
    for f in families:
        S["ppos_" + f] = []
        S["pneg_" + f] = []

    def unscramble(scores, gids, y_shuffled):
        out = np.full(n_test, np.nan)
        gids = np.asarray(gids, dtype=np.int64).ravel()
        assert len(gids) == n_test and len(np.unique(gids)) == n_test
        # the labels the loader delivered must match the canonical labels of those graphs
        assert np.array_equal(y_test_canon[gids], np.asarray(y_shuffled, dtype=np.int64).ravel())
        out[gids] = np.asarray(scores, dtype=np.float64).ravel()
        return out

    def add_pseudo(rec, neg_by_fam, pos_by_fam):
        aucs = []
        for f in families:
            S["pneg_" + f].append(np.asarray(neg_by_fam[f], dtype=np.float64).ravel())
            S["ppos_" + f].append(np.asarray(pos_by_fam[f], dtype=np.float64).ravel())
            a = auc_pos_neg(S["pneg_" + f][-1], S["ppos_" + f][-1])
            rec["pauc_" + f] = a
            aucs.append(a)
        rec["pauc_mean"] = float(np.mean(aucs)) if aucs and np.all(np.isfinite(aucs)) else float("nan")

    fork_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []

    meta_out = dict(method=method, dataset=DS, fold=fold, seed=seed, run_id=run_id,
                    ubgold_sh_line=sh_line, hparams={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                     for k, v in sorted(vars(args).items())},
                    n_train_normal_ubgold=n_all, n_fit=len(fit_list), n_holdout=n_hold,
                    holdout_seed=HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=n_test_anom,
                    pseudo_families=families, n_pseudo_per_family=n_hold,
                    device=str(device), torch=torch.__version__, scatter_impl=scatter_impl,
                    cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None))

    traj = []          # one dict per evaluation point
    train_loss = []    # per epoch
    start_epoch = 1
    state = None
    if osp.exists(ckpath):
        state = torch.load(ckpath, map_location="cpu", weights_only=False)
        print(f"[resume] {ckpath} at epoch {state['epoch']}")
    # M1: device of every (resumed) training segment, so mixed CPU/GPU runs are disclosed
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(state["epoch"] + 1 if state is not None else 1))
    segments = (list(state.get("segments", [])) if state is not None else []) + [seg]
    meta_out["segments"] = segments
    meta_out["device_all"] = "+".join(sorted({x["device"] for x in segments}))

    hb = hold_batch.to(device) if hold_batch is not None else None
    last_save = time.time()

    def maybe_stop(epoch, payload_fn):
        """Save a checkpoint periodically and when the job deadline is near."""
        nonlocal last_save
        now = time.time()
        stop = deadline is not None and now > deadline
        if stop or now - last_save > ckpt_every_s:
            save_ckpt(ckpath, payload_fn())
            last_save = now
        return stop

    eval_epochs = set(range(args.eval_freq, args.num_epoch + 1, args.eval_freq)) | {args.num_epoch}

    # =====================================================================================
    if method == "SIGNET":
        # GAOOD/detector/SIGNET.py fit(); model built exactly as SIGNET.init_model()
        model = M["signet"].SIGNET(input_dim=args.dataset_num_features, input_dim_edge=args.n_edge_feat,
                                   args=args, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # M1: pre-collated mixed batches [held-out ; P_f] (no DataLoader -> no RNG)
        pb = {f: Batch.from_data_list(hold_list + P[f]).to(device) for f in families}
        if state is not None:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["opt"])
            traj, train_loss, start_epoch = state["traj"], state["train_loss"], state["epoch"] + 1
            S.update(state["S"])  # M1
            set_rng_state(state["rng"])

        def payload(epoch):
            return dict(epoch=epoch, model=model.state_dict(), opt=optimizer.state_dict(), traj=traj,
                        train_loss=train_loss, rng=rng_state(), S=S, segments=segments)  # M1

        for epoch in range(start_epoch, args.num_epoch + 1):
            model.train()
            loss_all = 0
            num_sample = 0
            for data in dataloader:
                optimizer.zero_grad()
                data = data.to(device)
                y, y_hyper, node_imp, edge_imp = model(data)          # forward_model()
                loss = model.loss_nce(y, y_hyper).mean()
                loss_all += loss.item() * data.num_graphs
                num_sample += data.num_graphs
                loss.backward()
                optimizer.step()
            train_loss.append(loss_all / max(num_sample, 1))
            if epoch in eval_epochs:  # UB-GOLD: (epoch) % args.eval_freq == 0  (+ CHANGE: final epoch)
                model.eval()
                all_ad_true, all_ad_score, all_gid = [], [], []
                for data in dataloader_val:                          # == test loader in UB-GOLD
                    all_ad_true.append(data.y.cpu())
                    all_gid.append(data.gid.cpu())                   # M1
                    data = data.to(device)
                    with torch.no_grad():
                        y, y_hyper, _, _ = model(data)
                        ano_score = model.loss_nce(y, y_hyper)
                    all_ad_score.append(ano_score.cpu())
                rec = eval_metrics(metric, torch.cat(all_ad_true).numpy(), torch.cat(all_ad_score).numpy())
                S["test"].append(unscramble(torch.cat(all_ad_score).numpy(), torch.cat(all_gid).numpy(),
                                            torch.cat(all_ad_true).numpy()))  # M1
                if hb is not None:  # CHANGE: held-out normal loss, same scoring as test (one batch)
                    with torch.no_grad():
                        y, y_hyper, _, _ = model(hb)
                        hs = model.loss_nce(y, y_hyper).cpu().numpy()
                        rec["holdout_loss"] = finite_mean(hs)
                    S["hold"].append(hs.astype(np.float64))          # M1
                # M1: pseudo-anomaly set; SIGNET's NCE score is batch-dependent -> one mixed batch
                # [held-out normals ; perturbed copies] per family (scored without touching any RNG)
                neg, pos = {}, {}
                with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
                    for f in families:
                        y, y_hyper, _, _ = model(pb[f])
                        sc = model.loss_nce(y, y_hyper).cpu().numpy()
                        neg[f], pos[f] = sc[:n_hold], sc[n_hold:]
                add_pseudo(rec, neg, pos)
                rec.update(epoch=epoch, train_loss=train_loss[-1])
                traj.append(rec)
                if verbose:
                    print(f"[{run_id}] ep {epoch} train {train_loss[-1]:.4f} test_auc {rec['auroc']:.4f} "
                          f"hold {rec.get('holdout_loss', float('nan')):.4f} t={time.time()-t_start:.0f}s", flush=True)
            # CHANGE: no early stopping and no torch.save of the best-test-AUROC model
            if maybe_stop(epoch, lambda: payload(epoch)):
                print(f"[{run_id}] deadline reached at epoch {epoch}; checkpoint saved", flush=True)
                return "incomplete"

    # =====================================================================================
    elif method in ("OCGTL", "OCGIN"):
        # GAOOD/detector/OCGTL.py / OCGIN.py fit(); model built as their init_model()
        if method == "OCGTL":
            model = M["ocgtl"].ocgtl(dim_features=args.dataset_num_features, args=args).to(device)
        else:
            model = M["ocgin"].OCGIN(dim_features=args.dataset_num_features, args=args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        pb_all = Batch.from_data_list([g for f in families for g in P[f]]).to(device)  # M1 (no RNG)
        if state is not None:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["opt"])
            traj, train_loss, start_epoch = state["traj"], state["train_loss"], state["epoch"] + 1
            S.update(state["S"])  # M1
            set_rng_state(state["rng"])
            if start_epoch > args.eval_freq:
                model.eval()  # UB-GOLD never switches back to train() after the first evaluation

        def payload(epoch):
            return dict(epoch=epoch, model=model.state_dict(), opt=optimizer.state_dict(), traj=traj,
                        train_loss=train_loss, rng=rng_state(), S=S, segments=segments)  # M1

        def forward_model(data, eval_):
            emb = model(data)
            return model.loss_func(emb, eval_)

        for epoch in range(start_epoch, args.num_epoch + 1):
            all_loss, n_bw = 0, 0
            for data in dataloader:
                n_bw += 1
                data = data.to(device)
                loss_epoch = forward_model(data, False)
                loss_mean = loss_epoch.mean()
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                all_loss += loss_epoch.sum().detach()   # (detach: avoids keeping graphs alive; same value)
            mean_loss = all_loss.item() / args.n_train
            train_loss.append(mean_loss)
            if epoch in eval_epochs:
                model.eval()
                y_val, score_val, gid_val = [], [], []
                for data in dataloader_val:
                    data = data.to(device)
                    with torch.no_grad():  # (UB-GOLD computes without no_grad; values identical)
                        score_epoch = forward_model(data, True)
                    score_val = score_val + score_epoch.detach().cpu().tolist()
                    y_val = y_val + data.y.detach().cpu().tolist()
                    gid_val = gid_val + data.gid.detach().cpu().tolist()   # M1
                rec = eval_metrics(metric, y_val, score_val)
                S["test"].append(unscramble(score_val, gid_val, y_val))     # M1
                if hb is not None:
                    with torch.no_grad():
                        hs = forward_model(hb, True).cpu().numpy()
                        rec["holdout_loss"] = finite_mean(hs)
                    S["hold"].append(hs.astype(np.float64))                 # M1
                # M1: pseudo-anomaly set; per-graph scores are batch-independent here (no BN;
                # GraphNorm is per graph), so all P graphs are scored in one pre-collated batch
                with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
                    ps = forward_model(pb_all, True).cpu().numpy().astype(np.float64)
                pos = {f: ps[k * n_hold:(k + 1) * n_hold] for k, f in enumerate(families)}
                add_pseudo(rec, {f: S["hold"][-1] for f in families}, pos)
                rec.update(epoch=epoch, train_loss=mean_loss)
                traj.append(rec)
                if verbose:
                    print(f"[{run_id}] ep {epoch} train {mean_loss:.4f} test_auc {rec['auroc']:.4f} "
                          f"hold {rec.get('holdout_loss', float('nan')):.4f} t={time.time()-t_start:.0f}s", flush=True)
            if maybe_stop(epoch, lambda: payload(epoch)):
                print(f"[{run_id}] deadline reached at epoch {epoch}; checkpoint saved", flush=True)
                return "incomplete"

    # =====================================================================================
    elif method == "GLocalKD":
        # GAOOD/detector/GLocalKD.py fit(); not resumable (its fixed batch composition is drawn
        # once before training), so a run restarts from scratch if interrupted.
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.autograd import Variable
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
        for batch_idx, data in enumerate(dataloader):
            adj, _, h0, _ = glocalkd_process_graph(selfobj, data)
            preprocessed.append(tuple([adj, h0]))
        preprocessed_val = []
        val_gids = []  # M1: test-graph identity in the (fixed) order of preprocessed_val
        for batch_idx, data in enumerate(dataloader_val):
            adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, data)
            preprocessed_val.append(tuple([adj_matrixs, graph_appends, graph_label]))
            val_gids += data.gid.cpu().tolist()
        # CHANGE: held-out normals preprocessed one graph at a time, like the test set (batch_size_test=1)
        preprocessed_hold = []
        for g in hold_list:
            adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, Batch.from_data_list([g]))
            preprocessed_hold.append(tuple([adj_matrixs, graph_appends, graph_label]))
        # M1: pseudo-anomalies preprocessed and scored one graph at a time, like the held-out set
        preprocessed_p = {}
        for f in families:
            preprocessed_p[f] = []
            for g in P[f]:
                adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, Batch.from_data_list([g]))
                preprocessed_p[f].append(tuple([adj_matrixs, graph_appends, graph_label]))

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

        for epoch in range(1, args.num_epoch + 1):
            total_loss = 0.0
            nb = 0
            model_student.train()
            for batch_idx, data in enumerate(dataloader):
                model_student.zero_grad()
                adj, h0 = preprocessed[batch_idx]
                loss = forward_model(adj, h0)
                loss.backward(loss.clone().detach())
                nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                nb += 1
            train_loss.append(total_loss / max(nb, 1))
            if epoch in eval_epochs:
                model_student.eval()
                s, y = score_list(preprocessed_val)
                rec = eval_metrics(metric, y, s)
                S["test"].append(unscramble(s, val_gids, y))                # M1
                if preprocessed_hold:
                    sh, _ = score_list(preprocessed_hold)
                    rec["holdout_loss"] = finite_mean(sh)
                    S["hold"].append(np.asarray(sh, dtype=np.float64))      # M1
                pos = {}
                with torch.random.fork_rng(devices=fork_devices):           # M1
                    for f in families:
                        pos[f], _ = score_list(preprocessed_p[f])
                add_pseudo(rec, {f: S["hold"][-1] for f in families}, pos)
                rec.update(epoch=epoch, train_loss=train_loss[-1])
                traj.append(rec)
                if verbose:
                    print(f"[{run_id}] ep {epoch} train {train_loss[-1]:.4f} test_auc {rec['auroc']:.4f} "
                          f"hold {rec.get('holdout_loss', float('nan')):.4f} t={time.time()-t_start:.0f}s", flush=True)
            if deadline is not None and time.time() > deadline:
                print(f"[{run_id}] deadline reached; GLocalKD is not resumable, will restart", flush=True)
                return "incomplete"
    else:
        raise ValueError(method)

    meta_out["seconds"] = time.time() - t_start
    meta_out["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # M1: per-graph scores (written BEFORE the JSON, whose existence marks the run complete)
    epochs = [r["epoch"] for r in traj]
    for k, v in S.items():
        assert len(v) == len(epochs), (k, len(v), len(epochs))
    # labels in the shuffled loader must agree with the canonical order (sanity of the gid map)
    npz = osp.join(SCORES, run_id + ".npz")
    save_scores(npz, epochs, S, dict(
        y_test=y_test_canon, test_index=np.asarray(splits[fold][1], dtype=np.int64),
        hold_pos=np.asarray(hold_pos, dtype=np.int64), test_n_nodes=test_n_nodes, test_n_edges=test_n_edges,
        hold_n_nodes=hold_n_nodes, hold_n_edges=hold_n_edges, families=np.array(families),
        num_epoch=np.int64(args.num_epoch), eval_freq=np.int64(args.eval_freq)))
    meta_out["scores_npz"] = npz
    out = dict(meta=meta_out, traj=traj, train_loss=train_loss)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, out_json)
    if osp.exists(ckpath):
        os.remove(ckpath)
    print(f"[done] {run_id} in {meta_out['seconds']:.0f}s -> {out_json}", flush=True)
    return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DR_THREADS", "2")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    st = run(a.method, a.dataset, a.fold, holdout_frac=a.holdout_frac, deadline=a.deadline,
             max_epochs=a.max_epochs, tag=a.tag)
    sys.exit(0 if st == "done" else 3)
