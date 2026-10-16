"""Label-free re-evaluation of UB-GOLD deep GLAD baselines (SIGNET, OCGTL, OCGIN, GLocalKD).

ONE training run per (method, dataset, fold) records, at every UB-GOLD evaluation point
(epoch % eval_freq == 0, plus the final epoch):
  * test AUROC / AUPRC / FPR95, computed with UB-GOLD's own scoring code and metric functions;
  * the label-free signal: the method's own per-graph training loss (= its anomaly score),
    averaged over a held-out 20% of the NORMAL training graphs (carved out before training
    with a fixed seed; the model is trained on the remaining 80%);
  * the mean training loss of that epoch.
From the trajectory, aggregate.py derives ORACLE (UB-GOLD protocol: max test AUROC with its
patience-30 early stopping), LAST (final epoch) and VAL-NORMAL (argmin held-out normal loss).

Everything that defines the model, data and training is UB-GOLD code (commit c940457, copied
verbatim to ./ubgold_src): the nn modules are imported unchanged from ubgold_src/GAOOD/nn,
the data functions (get_ad_split_TU, get_ad_dataset_TU) and set_seed are extracted verbatim
via `ast` and executed unchanged, the per-dataset hyperparameters are parsed from
ubgold_src/benchmark/Source/<METHOD>.sh with UB-GOLD's own argparse parser (extracted from
benchmark/mymain.py), and the training loops below are line-by-line copies of the detectors'
fit() methods (ubgold_src/GAOOD/detector/<METHOD>.py). The changes are listed in NOTES.md and
marked "# CHANGE" below.

Usage:
  PYTHONDONTWRITEBYTECODE=1 python -B run_rerun.py --method SIGNET --dataset AIDS --fold 0
"""
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
CKPT_ROOT = os.environ.get("DR_CKPT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/deep_rerun/ckpt"))  # resumable training state
PARTS = os.environ.get("DR_PARTS", osp.join(osp.dirname(HERE), "results", "deep_rerun_parts"))  # per-run trajectories
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
    if method in GPU_ONLY and device.type != "cuda":
        raise RuntimeError(f"{method} hard-codes CUDA in UB-GOLD; run it on a GPU node")

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

    meta_out = dict(method=method, dataset=DS, fold=fold, seed=seed, run_id=run_id,
                    ubgold_sh_line=sh_line, hparams={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                     for k, v in sorted(vars(args).items())},
                    n_train_normal_ubgold=n_all, n_fit=len(fit_list), n_holdout=n_hold,
                    holdout_seed=HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=n_test_anom,
                    device=str(device), torch=torch.__version__, scatter_impl=scatter_impl,
                    cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None))

    traj = []          # one dict per evaluation point
    train_loss = []    # per epoch
    start_epoch = 1
    state = None
    if osp.exists(ckpath):
        state = torch.load(ckpath, map_location="cpu", weights_only=False)
        print(f"[resume] {ckpath} at epoch {state['epoch']}")

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
        if state is not None:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["opt"])
            traj, train_loss, start_epoch = state["traj"], state["train_loss"], state["epoch"] + 1
            set_rng_state(state["rng"])

        def payload(epoch):
            return dict(epoch=epoch, model=model.state_dict(), opt=optimizer.state_dict(), traj=traj,
                        train_loss=train_loss, rng=rng_state())

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
                all_ad_true, all_ad_score = [], []
                for data in dataloader_val:                          # == test loader in UB-GOLD
                    all_ad_true.append(data.y.cpu())
                    data = data.to(device)
                    with torch.no_grad():
                        y, y_hyper, _, _ = model(data)
                        ano_score = model.loss_nce(y, y_hyper)
                    all_ad_score.append(ano_score.cpu())
                rec = eval_metrics(metric, torch.cat(all_ad_true).numpy(), torch.cat(all_ad_score).numpy())
                if hb is not None:  # CHANGE: held-out normal loss, same scoring as test (one batch)
                    with torch.no_grad():
                        y, y_hyper, _, _ = model(hb)
                        rec["holdout_loss"] = finite_mean(model.loss_nce(y, y_hyper).cpu().numpy())
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
        if state is not None:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["opt"])
            traj, train_loss, start_epoch = state["traj"], state["train_loss"], state["epoch"] + 1
            set_rng_state(state["rng"])
            if start_epoch > args.eval_freq:
                model.eval()  # UB-GOLD never switches back to train() after the first evaluation

        def payload(epoch):
            return dict(epoch=epoch, model=model.state_dict(), opt=optimizer.state_dict(), traj=traj,
                        train_loss=train_loss, rng=rng_state())

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
                y_val, score_val = [], []
                for data in dataloader_val:
                    data = data.to(device)
                    with torch.no_grad():  # (UB-GOLD computes without no_grad; values identical)
                        score_epoch = forward_model(data, True)
                    score_val = score_val + score_epoch.detach().cpu().tolist()
                    y_val = y_val + data.y.detach().cpu().tolist()
                rec = eval_metrics(metric, y_val, score_val)
                if hb is not None:
                    with torch.no_grad():
                        rec["holdout_loss"] = finite_mean(forward_model(hb, True).cpu().numpy())
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
        for batch_idx, data in enumerate(dataloader_val):
            adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, data)
            preprocessed_val.append(tuple([adj_matrixs, graph_appends, graph_label]))
        # CHANGE: held-out normals preprocessed one graph at a time, like the test set (batch_size_test=1)
        preprocessed_hold = []
        for g in hold_list:
            adj_matrixs, _, graph_appends, graph_label = glocalkd_process_graph(selfobj, Batch.from_data_list([g]))
            preprocessed_hold.append(tuple([adj_matrixs, graph_appends, graph_label]))

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
                if preprocessed_hold:
                    sh, _ = score_list(preprocessed_hold)
                    rec["holdout_loss"] = finite_mean(sh)
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
