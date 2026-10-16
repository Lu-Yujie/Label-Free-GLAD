"""E4 extension (PREREG_AMENDMENTS A11): instrumented GLASS (arXiv 2609.05253) runs on UB-GOLD folds.

Same CLI and per-run JSON "part" schema as evalpaper/harness/run_eval.py:
  PYTHONDONTWRITEBYTECODE=1 python -B run_method.py --method GLASS --dataset BZR --fold 0 [--hp hpfree]
  PYTHONDONTWRITEBYTECODE=1 python -B run_method.py --prep --dataset BZR        # features + Qwen3 embeddings (GPU)

* GLASS's code is the official repository at commit 3bbba39 (src/, byte-identical). The training loop below is a
  line-by-line copy of src/glass_training_core.py: train_single_run(); evaluate_all_scorers() is copied with score
  capture; the reference-set scorer is src/evaluate_reference.py's own functions. Every change is marked "# CHANGE"
  and listed in NOTES.md (D1-D8).
* Folds, features, seeds and held-out normals come from the harness (its run_eval.py is imported, never modified).
* hp 'ubgold' = the README single-domain command line (the only released configuration); hp 'hpfree' =
  train_single.py argparse defaults. Both resolve to the same values (NOTES.md Part 1).
* Evaluation points = GLASS's own schedule: epoch 0 (baseline), epoch 1 and every 5 epochs (32 points for 150 epochs).
  All epochs are trained; GLASS's test-label checkpoint selection / early stopping / final evaluation are replayed online.

Outputs (EP_PARTS / EP_SCORES / EP_CKPT override):
  evalpaper/ext/parts/<run_id>.json                       meta + per-evaluation-point metrics
  $GLAD_DATA_ROOT/ext_glass/scores/<run_id>.npz        per-graph test / held-out scores (all scorers/rules)
  $GLAD_DATA_ROOT/ext_glass/ckpt/<DS>/<run_id>.pt      resumable state (removed when done)
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # never import from cwd

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import os.path as osp
import pickle
import re
import shlex
import sys
import time
import types

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
GLASS_SRC = osp.join(HERE, "src")                                  # GLASS @ 3bbba39 (unmodified)
GLASS_COMMIT = "3bbba39a1d714e8c3583461be59bba978f3d1f37"
HARNESS = osp.join(osp.dirname(osp.dirname(HERE)), "harness")     # evalpaper/harness
OVERLAY = (os.environ.get("GLAD_ENVS", "envs") + "/ext-glass")                          # transformers 4.55.2 stack (--no-deps)
ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_glass")
DATA_ROOT = os.environ.get("EP_GLASS_DATA", osp.join(ROOT, "data_ubgold"))       # GLASS feature caches
EMB_CACHE = os.environ.get("EP_GLASS_EMB", osp.join(ROOT, "embedding_cache"))    # GLASS text-embedding cache
TEXT_MODEL = osp.join(ROOT, "models", "Qwen3-Embedding-0.6B")                    # revision in models/REVISION.txt
CKPT_ROOT = os.environ.get("EP_CKPT", osp.join(ROOT, "ckpt"))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
SCORES = os.environ.get("EP_SCORES", osp.join(ROOT, "scores"))
METHOD = "GLASS"
GLASS_NAME = {"PROTEINS_full": "PROTEINS"}        # CHANGE (D8): same graphs; GLASS's DOMAIN_MAP / domain prior name
DEFAULT_RULE = "rcsm_full_test_cdf"               # NOTES.md Part 1: a-priori label-free scorer (A11)
R0_RULE = "fixed_test_raw"                        # first rule in evaluate_reference.py's enumeration (sensitivity)
TRAIN_SCORERS = ['graph_knn', 'text_knn', 'ensemble', 'fusion_knn', 'vmf', 'sms']   # glass_training_core l.585
UBGOLD_SOCIAL = ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_harness():
    """Import evalpaper/harness/run_eval.py as a module (read-only use of its data/RNG/IO helpers)."""
    spec = importlib.util.spec_from_file_location("ep_harness_run_eval", osp.join(HARNESS, "run_eval.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    return H


def load_ubgold_metric(H):
    if H.SRC not in sys.path:
        sys.path.append(H.SRC)            # appended: GLASS's own top-level modules keep precedence
    H.stub_package("GAOOD", osp.join(H.SRC, "GAOOD"))
    importlib.import_module("GAOOD.metric")
    return importlib.import_module("GAOOD.metric.metric")


def load_glass():
    """Import GLASS's modules from src/ (unmodified)."""
    for k in ("GLASS_GPU_ACCEL", "GLASS_ABLATION", "GLASS_GRAPHDP_VERSION"):
        if os.environ.get(k):
            raise RuntimeError(f"{k} must be unset (README single-domain configuration)")
    if OVERLAY not in sys.path:
        sys.path.append(OVERLAY)          # appended: the ubgold-gpu env keeps precedence
    if GLASS_SRC not in sys.path:
        sys.path.insert(0, GLASS_SRC)
    G = types.SimpleNamespace()
    G.T = importlib.import_module("glass_training_core")
    G.R = importlib.import_module("evaluate_reference")
    G.cfg = importlib.import_module("src.config")
    G.ds = importlib.import_module("src.dataset")
    G.te = importlib.import_module("src.text_encoder")
    G.ec = importlib.import_module("src.embedding_cache")
    G.ut = importlib.import_module("src.utils")
    G.gm = importlib.import_module("src.glass_model")
    G.gdp = importlib.import_module("src.graph_dp")
    for mod in (G.T, G.R, G.cfg, G.ds, G.te, G.ec, G.ut, G.gm, G.gdp):
        assert osp.realpath(mod.__file__).startswith(osp.realpath(GLASS_SRC)), mod.__file__
    # CHANGE (D5): GLASS's global embedding cache would live in src/embedding_cache; create it on slate first
    os.makedirs(EMB_CACHE, exist_ok=True)
    G.ec._global_cache = None
    G.ec.get_embedding_cache(cache_dir=EMB_CACHE)
    assert G.T.get_embedding_cache().cache_dir == EMB_CACHE
    assert G.R.get_embedding_cache().cache_dir == EMB_CACHE
    assert G.T.GPU_ACCEL is False and G.T._batch_cache is None
    return G


# ----------------------------------------------------------------------------------------------
# hyperparameters (NOTES.md Part 1, pre-registered 2026-10-13 12:56:38 EDT)
# ----------------------------------------------------------------------------------------------
def train_single_parser():
    """train_single.py main()'s argparse parser, extracted verbatim (statements before parse_args)."""
    path = osp.join(GLASS_SRC, "train_single.py")
    tree = ast.parse(open(path).read())
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    body = []
    for s in main.body:
        if isinstance(s, ast.Global):
            continue
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "args":
            break
        body.append(s)
    ns = {"argparse": argparse, "os": os, "PROJECT_ROOT": __import__("pathlib").Path(GLASS_SRC),
          "CKPT_ROOT": "checkpoints", "RESULT_ROOT": "results"}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns["parser"]


def readme_single_domain_tokens():
    """Tokens of the README 'Single-Domain Training and Evaluation' train_single.py command (the only released line),
    minus the dataset/device/path options."""
    txt = open(osp.join(GLASS_SRC, "README.md")).read()
    sec = txt.split("## Single-Domain Training and Evaluation", 1)[1].split("\n## ", 1)[0]
    block = sec.split("```bash", 1)[1].split("```", 1)[0]
    cmd = block.split("python evaluate_reference.py")[0].replace("\\\n", " ")
    toks = shlex.split(cmd, posix=True)
    assert toks[:2] == ["python", "train_single.py"], toks
    toks = toks[2:]
    out, i = [], 0
    while i < len(toks):
        k = toks[i]
        if k in ("--dataset", "--device", "--text_device", "--text_model", "--data_root"):
            i += 2
            continue
        out.append(k)
        i += 1
    return out


def resolve_hparams(DS, hp):
    parser = train_single_parser()
    if hp == "ubgold":
        toks = readme_single_domain_tokens()
        line = "README single-domain line: python train_single.py " + " ".join(toks)
    else:
        assert hp == "hpfree", hp
        toks = []
        line = "HPFREE train_single.py argparse defaults"
    a = parser.parse_args(["--dataset", GLASS_NAME.get(DS, DS)] + toks)
    return a, line


def make_config(G, a, device, max_epochs=None):
    """Copy of train_single.py main() l.118-136 (config construction)."""
    config = G.cfg.GLASSConfig(
        dataset_name=a.dataset,
        data_root=DATA_ROOT,                    # CHANGE (D1): feature caches on slate
        device=device,
        epochs=a.epochs if max_epochs is None else max_epochs,   # max_epochs: smoke/timing runs only (tagged)
        num_seeds=a.num_seeds,
        batch_size=a.batch_size,
        lr=a.lr,
        alpha=a.alpha,
        patience=a.patience,
        num_prototypes=8,
    )
    config.gin_layers = 3
    config.mrl_mode = "native"
    config.text_model_path = TEXT_MODEL         # CHANGE (D6): pinned local Qwen3-Embedding-0.6B
    config.text_embed_dim = a.text_embed_dim
    if a.use_cache:
        raise NotImplementedError("--use_cache is not part of the released single-domain line")
    return config


# ----------------------------------------------------------------------------------------------
# data: UB-GOLD folds / features (CHANGE D1, D2)
# ----------------------------------------------------------------------------------------------
def load_full_dataset(H, DS):
    """All graphs of the TU dataset in TU order with UB-GOLD's features (copy of the loading lines of UB-GOLD's
    get_ad_dataset_TU: Constant(1) for the social datasets, edge_attr dropped)."""
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import Constant
    path = osp.join(H.DATA_RUN, "data", DS)
    if DS in UBGOLD_SOCIAL:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=DS)
    out = []
    for data in dataset:
        data.edge_attr = None
        out.append(data)
    return out


def _same_graph(a, b):
    return (int(a.num_nodes) == int(b.num_nodes) and torch.equal(a.edge_index, b.edge_index)
            and torch.equal(a.x, b.x))


def load_fold(H, DS, fold, holdout_frac=0.2):
    """UB-GOLD trial `fold` exactly as the harness (set_seed(3407+fold), get_ad_split_TU / get_ad_dataset_TU asserted
    against ubgold_splits, 20% held-out carve-out RandomState(2026+1000*fold)); returns indices into the full TU list."""
    ns = H.load_ubgold_data_functions()
    ns["set_seed"](3407 + fold)
    assert not DS.startswith("Tox21"), "Tox21 not supported by this adapter"
    uargs = types.SimpleNamespace(DS=DS, num_trial=5, batch_size=64, batch_size_test=9999)
    cwd = os.getcwd()
    os.chdir(H.DATA_RUN)
    try:
        splits = ns["get_ad_split_TU"](uargs, fold=uargs.num_trial)
        out = ns["get_ad_dataset_TU"](uargs, splits[fold], need_str_enc=False)
        full = load_full_dataset(H, DS)
    finally:
        os.chdir(cwd)
    (dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta) = out
    ref = json.load(open(osp.join(H.SPLITS, f"{DS}.json")))
    rf = ref["folds"][fold]
    train_index, test_index = list(map(int, splits[fold][0])), list(map(int, splits[fold][1]))
    assert train_index == list(map(int, rf["split_train_index"]))
    assert test_index == list(map(int, rf["split_test_index"]))
    labels = [int(d.y) for d in full]
    orig_train = [i for i in train_index if labels[i] != 0]          # UB-GOLD: training normals = class != 0
    train_all = list(dataloader.dataset)
    assert len(orig_train) == len(train_all) == meta["num_train"]
    for j, i in enumerate(orig_train):
        assert _same_graph(train_all[j], full[i]), (DS, fold, j, i)
    test_list = list(dataloader_test.dataset)
    assert len(test_list) == len(test_index)
    for j, i in enumerate(test_index):
        assert _same_graph(test_list[j], full[i]), (DS, fold, j, i)
    y_test = np.array([int(d.y) for d in test_list], dtype=np.int64)            # UB-GOLD: 1 = anomaly (class 0)
    assert list(map(int, y_test)) == list(map(int, rf["y_test"])), "test labels differ from ubgold_splits"
    assert all((labels[i] == 0) == bool(y) for i, y in zip(test_index, y_test))
    n_all = len(train_all)
    n_hold = int(round(holdout_frac * n_all))
    perm = np.random.RandomState(H.HOLDOUT_SEED + 1000 * fold).permutation(n_all)
    hold_pos = sorted(perm[:n_hold].tolist())
    fit_pos = sorted(perm[n_hold:].tolist())
    assert n_hold > 0
    return dict(full=full, labels=labels, y_test=y_test, n_all=n_all, n_hold=n_hold, hold_pos=hold_pos,
                fit_pos=fit_pos, fit_orig=[orig_train[p] for p in fit_pos], hold_orig=[orig_train[p] for p in hold_pos],
                test_orig=test_index, train_index=train_index, meta=meta)


def feature_cache_path(DS):
    return osp.join(DATA_ROOT, f"{DS}_glass_features.pkl")


def build_glass_dataset(G, config, DS, full, allow_build=False):
    """CHANGE (D1): GLASS's GLASSDataset over UB-GOLD's data list. GLASSDataset.__init__ only loads
    TUDataset(use_node_attr=True) and the feature cache; the precomputation and accessors are GLASS's own methods."""
    ds = G.ds.GLASSDataset.__new__(G.ds.GLASSDataset)
    ds.config = config
    ds.dataset_name = config.dataset_name
    ds.data_root = config.data_root
    ds.raw_dataset = None
    ds.data_list = full
    ds.num_features = int(full[0].x.shape[1])
    ds.domain = G.gdp.DOMAIN_MAP.get(ds.dataset_name, 'generic')
    ds.graphdp_version = os.environ.get("GLASS_GRAPHDP_VERSION", "v1").lower()
    assert ds.graphdp_version == "v1"
    path = feature_cache_path(DS)
    if osp.exists(path):
        with open(path, "rb") as f:
            c = pickle.load(f)
        assert c["n_graphs"] == len(full) and c["DS"] == DS
        ds.ltd_list, ds.spectral_list, ds.graph_dp_list = c["ltd"], c["spectral"], c["graph_dp"]
    elif allow_build:
        ds._precompute_features()                       # GLASS's own (LTD, spectral sketch, GraphDP v1)
        os.makedirs(DATA_ROOT, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(dict(DS=DS, glass_name=ds.dataset_name, n_graphs=len(full), ltd=ds.ltd_list,
                             spectral=ds.spectral_list, graph_dp=ds.graph_dp_list, glass_commit=GLASS_COMMIT,
                             created=time.strftime("%Y-%m-%d %H:%M:%S")), f)
        os.replace(tmp, path)
    else:
        raise FileNotFoundError(f"{path} missing: run --prep for {DS} first")
    assert len(ds.ltd_list) == len(full) and len(ds.graph_dp_list) == len(full)
    for i in (0, len(full) - 1):
        assert ds.ltd_list[i].shape[0] == full[i].num_nodes
    return ds


def make_text_encoder(G, config, text_device, load_model):
    """CHANGE (D6): when the embedding cache exists, GLASS's TextEncoder is built without loading Qwen3 (the
    constructor consumes no RNG and native-MRL projections are an empty ModuleList, as in TextEncoder.__init__)."""
    if load_model:
        return G.te.TextEncoder(config, text_device=text_device)
    te = G.te.TextEncoder.__new__(G.te.TextEncoder)
    torch.nn.Module.__init__(te)
    te.config = config
    te.text_device = text_device or config.device
    te.mrl_mode = getattr(config, 'mrl_mode', 'learned')
    assert te.mrl_mode == "native"
    for d_s in config.matryoshka_dims:
        assert d_s <= config.text_embed_dim
    te.projections = torch.nn.ModuleList()
    return te


def text_embeddings(G, dataset, config, text_device, device, allow_encode=False):
    """GLASS's own precompute_text_embeddings (content-hashed cache; batches of 16 in dataset order)."""
    te = make_text_encoder(G, config, text_device, load_model=allow_encode)
    if allow_encode:
        te.to_text_device()
    else:
        domain = G.gdp.DOMAIN_MAP.get(config.dataset_name, 'generic')
        inst = G.gdp.get_instruction(domain=domain, mode='train')
        if G.T.get_embedding_cache().get(config.dataset_name, dataset.graph_dp_list, inst) is None:
            raise FileNotFoundError(f"text-embedding cache miss for {config.dataset_name}: run --prep on a GPU node")
    raw = G.T.precompute_text_embeddings(dataset, te, config, text_device, device)
    if allow_encode:
        del te.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return te, raw


# ----------------------------------------------------------------------------------------------
# reference-set scorer (evaluate_reference.py, verbatim functions)
# ----------------------------------------------------------------------------------------------
def reference_rules(R):
    """Copy of evaluate_reference.py main() l.289-295 (the 24 rules, in its enumeration order)."""
    rules = []
    for score_key in ["test_raw", "test_cdf", "test_z"]:
        rules.append((f"fixed_{score_key}", lambda ds, ch, key=score_key: (R.fixed_slice_weights(ch), key)))
        rules.append((f"domain_{score_key}", lambda ds, ch, key=score_key: (R.domain_prior_weights(ds, ch), key)))
        for mode in ["agree", "full", "align"]:
            rules.append((f"rc_{mode}_{score_key}", lambda ds, ch, key=score_key, mode=mode: (R.rc_weights(ch, mode=mode), key)))
            rules.append((f"rcsm_{mode}_{score_key}", lambda ds, ch, key=score_key, mode=mode: (R.slice_modality_rc_weights(ch, mode=mode), key)))
    return rules


@torch.no_grad()
def reference_scores(G, rules, model, text_emb_all, dataset, config, device, ref_idx, queries):
    """evaluate_reference.py l.342-359 on the current model: R = ref_idx (fit-set normals), k = 5; returns
    {query_name: {rule: per-graph scores}} and the per-rule channel weights (reference statistics only)."""
    R = G.R
    train_g, train_t = R.get_embeddings(model, text_emb_all, dataset, None, ref_idx, config, device)
    out, weights = {}, None
    for qname, qidx in queries.items():
        q_g, q_t = R.get_embeddings(model, text_emb_all, dataset, None, qidx, config, device)
        channels = R.collect_channels(train_g, train_t, q_g, q_t, config.matryoshka_dims, k=5)
        out[qname] = {}
        w_here = {}
        for name, fn in rules:
            w, score_key = fn(config.dataset_name, channels)
            out[qname][name] = np.asarray(R.score_with_weights(channels, w, score_key), dtype=np.float64)
            w_here[name] = [float(x) for x in w]
        if weights is None:
            weights = w_here
        else:
            assert all(np.allclose(weights[k], w_here[k], rtol=0, atol=1e-12) for k in weights)  # reference-only
    return out, weights


@torch.no_grad()
def holdout_train_scorers(G, model, text_encoder, raw_text_cache, dataset, train_indices, hold_indices, config, device):
    """Per-graph training-time scorers that do not normalise over the scored set, on the held-out normals
    (same computations as evaluate_all_scorers 1., 2., 4., 5. with the prototypes as set by that evaluation)."""
    T = G.T
    F = torch.nn.functional
    train_graph, train_text = T.get_embeddings(model, text_encoder, raw_text_cache, dataset, train_indices, config, device)
    hold_graph, hold_text = T.get_embeddings(model, text_encoder, raw_text_cache, dataset, hold_indices, config, device)
    train_graph, hold_graph = F.normalize(train_graph, dim=-1), F.normalize(hold_graph, dim=-1)
    train_text, hold_text = F.normalize(train_text, dim=-1), F.normalize(hold_text, dim=-1)
    out = dict(graph_knn=T.score_knn(train_graph, hold_graph, k=5), text_knn=T.score_knn(train_text, hold_text, k=5))
    train_fused = F.normalize(torch.cat([train_graph, train_text], dim=-1), dim=-1)
    hold_fused = F.normalize(torch.cat([hold_graph, hold_text], dim=-1), dim=-1)
    out["fusion_knn"] = T.score_knn(train_fused, hold_fused, k=5)
    bs = min(256, len(hold_indices))
    vm = []
    for start in range(0, len(hold_indices), bs):
        bi = hold_indices[start:min(start + bs, len(hold_indices))]
        x, ei, b, sp = T.prepare_batch(dataset, bi, config, device)
        z_slices, _ = model.encode_graph(x, ei, b, sp)
        vm.append(model.compute_anomaly_score(z_slices).cpu())
    out["vmf"] = torch.cat(vm).numpy()
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------
# copy of glass_training_core.py evaluate_all_scorers (l.199-331) with score capture (CHANGE D4)
# ----------------------------------------------------------------------------------------------
def make_evaluate_all_scorers(T):
    get_embeddings, score_knn, normalize_scores = T.get_embeddings, T.score_knn, T.normalize_scores
    get_per_slice_embeddings, prepare_batch, compute_metrics = T.get_per_slice_embeddings, T.prepare_batch, T.compute_metrics
    SphericalMultiModalScoring = T.SphericalMultiModalScoring
    F = torch.nn.functional

    @torch.no_grad()
    def evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                             train_indices, test_indices, test_labels, config, device, cap):
        """Evaluate multiple scoring strategies and pick the best."""
        model.eval()
        _batch_cache = T._batch_cache

        # Get embeddings
        train_graph, train_text = get_embeddings(model, text_encoder, raw_text_cache, dataset, train_indices, config, device)
        test_graph, test_text = get_embeddings(model, text_encoder, raw_text_cache, dataset, test_indices, config, device)

        train_graph = F.normalize(train_graph, dim=-1)
        test_graph = F.normalize(test_graph, dim=-1)
        train_text = F.normalize(train_text, dim=-1)
        test_text = F.normalize(test_text, dim=-1)

        results = {}

        # 1. Graph-only kNN
        s_graph = score_knn(train_graph, test_graph, k=5)
        results['graph_knn'] = compute_metrics(test_labels, s_graph)
        cap['graph_knn'] = s_graph                                       # CHANGE (D4)

        # 2. Text-only kNN
        s_text = score_knn(train_text, test_text, k=5)
        results['text_knn'] = compute_metrics(test_labels, s_text)
        cap['text_knn'] = s_text                                         # CHANGE (D4)

        # 3. Adaptive ensemble: weighted sum with learned weight
        # Try multiple weights and pick best (this is a simple grid search)
        sg = normalize_scores(s_graph)
        st = normalize_scores(s_text)
        cap['ens_sg'], cap['ens_st'] = sg, st                            # CHANGE (D4)
        best_ens_auroc = 0
        best_w = 0.5
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            s_ens = w * sg + (1 - w) * st
            m = compute_metrics(test_labels, s_ens)
            if m['auroc'] > best_ens_auroc:
                best_ens_auroc = m['auroc']
                best_w = w

        s_ens = best_w * sg + (1 - best_w) * st
        results['ensemble'] = compute_metrics(test_labels, s_ens)
        results['ensemble']['weight_graph'] = best_w
        cap['ensemble'] = s_ens                                          # CHANGE (D4)

        # 4. Concatenation kNN
        train_fused = F.normalize(torch.cat([train_graph, train_text], dim=-1), dim=-1)
        test_fused = F.normalize(torch.cat([test_graph, test_text], dim=-1), dim=-1)
        s_fused = score_knn(train_fused, test_fused, k=5)
        results['fusion_knn'] = compute_metrics(test_labels, s_fused)
        cap['fusion_knn'] = s_fused                                      # CHANGE (D4)

        # 5. vMF prototype scoring (fit prototypes post-hoc)
        bs = min(256, len(train_indices))
        train_slices_list = []
        for start in range(0, len(train_indices), bs):
            end = min(start + bs, len(train_indices))
            bi = train_indices[start:end]
            if _batch_cache is not None:
                x, ei, b, sp = _batch_cache.get_batch(bi, device)
            else:
                x, ei, b, sp = prepare_batch(dataset, bi, config, device)
            z_slices, _ = model.encode_graph(x, ei, b, sp)
            train_slices_list.append(z_slices)

        # Concatenate per-slice
        train_z_per_slice = []
        for s in range(config.num_slices):
            train_z_per_slice.append(torch.cat([sl[s] for sl in train_slices_list], dim=0))

        model.prototypes.init_from_data(train_z_per_slice)

        test_slices_list = []
        test_vmf_scores = []
        for start in range(0, len(test_indices), bs):
            end = min(start + bs, len(test_indices))
            bi = test_indices[start:end]
            if _batch_cache is not None:
                x, ei, b, sp = _batch_cache.get_batch(bi, device)
            else:
                x, ei, b, sp = prepare_batch(dataset, bi, config, device)
            z_slices, _ = model.encode_graph(x, ei, b, sp)
            scores = model.compute_anomaly_score(z_slices)
            test_vmf_scores.append(scores.cpu())

        s_vmf = torch.cat(test_vmf_scores).numpy()
        results['vmf'] = compute_metrics(test_labels, s_vmf)
        cap['vmf'] = s_vmf                                               # CHANGE (D4)

        # 6. SMS (vMF KDE) scoring - finite-kappa
        try:
            train_graph_slices, train_text_slices = get_per_slice_embeddings(
                model, text_encoder, raw_text_cache, dataset, train_indices, config, device)
            test_graph_slices, test_text_slices = get_per_slice_embeddings(
                model, text_encoder, raw_text_cache, dataset, test_indices, config, device)

            sms = SphericalMultiModalScoring(config.matryoshka_dims, alpha=0.5)
            sms.fit(train_graph_slices, train_text_slices)

            # SMS with alpha=0.5 (equal modality weight)
            s_sms = sms.score(test_graph_slices, test_text_slices).numpy()
            results['sms'] = compute_metrics(test_labels, s_sms)
            results['sms']['kappa_graph'] = sms.kappa_graph
            results['sms']['kappa_text'] = sms.kappa_text
            cap['sms_a0.5'] = s_sms                                      # CHANGE (D4)

            # SMS with optimal alpha (grid search)
            best_sms_auroc = results['sms']['auroc']
            best_sms_alpha = 0.5
            for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
                sms_a = SphericalMultiModalScoring(config.matryoshka_dims, alpha=a)
                sms_a.fit(train_graph_slices, train_text_slices)
                s_a = sms_a.score(test_graph_slices, test_text_slices).numpy()
                cap[f'sms_a{a:.1f}'] = s_a                               # CHANGE (D4)
                m_a = compute_metrics(test_labels, s_a)
                if m_a['auroc'] > best_sms_auroc:
                    best_sms_auroc = m_a['auroc']
                    best_sms_alpha = a

            if best_sms_alpha != 0.5:
                sms_best = SphericalMultiModalScoring(config.matryoshka_dims, alpha=best_sms_alpha)
                sms_best.fit(train_graph_slices, train_text_slices)
                s_sms_best = sms_best.score(test_graph_slices, test_text_slices).numpy()
                results['sms'] = compute_metrics(test_labels, s_sms_best)
                results['sms']['kappa_graph'] = sms.kappa_graph
                results['sms']['kappa_text'] = sms.kappa_text
                results['sms']['best_alpha'] = best_sms_alpha
                cap['sms'] = s_sms_best                                  # CHANGE (D4)
            else:
                results['sms']['best_alpha'] = 0.5
                cap['sms'] = s_sms                                       # CHANGE (D4)

            # SMS with uniform slice weights (ablation)
            s_sms_uni = sms.score_uniform_weights(test_graph_slices, test_text_slices).numpy()
            results['sms_uniform'] = compute_metrics(test_labels, s_sms_uni)
            cap['sms_uniform'] = s_sms_uni                               # CHANGE (D4)
        except Exception as e:
            print(f"  [WARN] SMS scoring failed: {e}")
            results['sms'] = {'auroc': 0.0, 'auprc': 0.0, 'fpr95': 1.0}
            results['sms_uniform'] = {'auroc': 0.0, 'auprc': 0.0, 'fpr95': 1.0}
            cap['sms_failed'] = str(e)                                   # CHANGE (D4)

        return results

    return evaluate_all_scorers


CAP_TEST_KEYS = (['graph_knn', 'text_knn', 'ens_sg', 'ens_st', 'ensemble', 'fusion_knn', 'vmf', 'sms', 'sms_uniform']
                 + [f'sms_a{a / 10:.1f}' for a in range(11)])
HOLD_TRAIN_KEYS = ['graph_knn', 'text_knn', 'fusion_knn', 'vmf']


def glass_summary(res):
    """GLASS's own metrics (x100, sklearn) of one evaluate_all_scorers call, JSON-safe."""
    out = {}
    for k, v in res.items():
        out[k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else
                       [float(x) for x in vv] if isinstance(vv, (list, tuple)) else vv) for kk, vv in v.items()}
    return out


class Ctx:
    pass


def run(DS, fold, hp="ubgold", holdout_frac=0.2, deadline=None, max_epochs=None, ckpt_every_s=300.0, tag="",
        verbose=True):
    t_start = time.time()
    run_id = f"{METHOD}_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + (f"_{tag}" if tag else "")
    os.makedirs(PARTS, exist_ok=True)
    out_json = osp.join(PARTS, run_id + ".json")
    if osp.exists(out_json):
        print(f"[skip] {out_json} exists")
        return "done"
    H = load_harness()
    ckdir = osp.join(CKPT_ROOT, DS)
    os.makedirs(ckdir, exist_ok=True)
    ckpath = osp.join(ckdir, run_id + ".pt")
    lock = ckpath + ".lock"
    if not H.acquire_lock(lock):
        print(f"[locked] {run_id} is running in another job")
        return "locked"
    try:
        return _run(H, DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s,
                    verbose, t_start)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _run(H, DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s, verbose,
         t_start):
    metric = load_ubgold_metric(H)
    G = load_glass()
    T = G.T
    F = torch.nn.functional
    evaluate_all_scorers = make_evaluate_all_scorers(T)
    rules = reference_rules(G.R)
    rule_names = [r[0] for r in rules]
    assert rule_names[0] == R0_RULE and DEFAULT_RULE in rule_names and len(rule_names) == 24
    deterministic = not torch.cuda.is_available()
    if deterministic and os.environ.get("EP_GLASS_NONDET", "") != "1":
        # CHANGE (D9, NOTES Part 2): deterministic CPU kernels (fixes the float summation order only)
        torch.use_deterministic_algorithms(True, warn_only=True)

    a, hp_line = resolve_hparams(DS, hp)
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = make_config(G, a, dev_str, max_epochs)
    text_device = dev_str
    seed = 3407 + fold
    Fd = load_fold(H, DS, fold, holdout_frac)
    y_test = Fd["y_test"]
    n_test = len(y_test)

    # ================= copy of glass_training_core.py: train_single_run() ===============================
    G.ut.set_seed(seed)                       # (T.set_seed is src.utils.set_seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    dataset = build_glass_dataset(G, config, DS, Fd["full"])      # CHANGE (D1)
    T._batch_cache = None                                         # (config._use_cache is not set)

    # GLAD protocol -- CHANGE (D1, D2): UB-GOLD fold; fit set (80% of UB-GOLD training normals) in RandomState(seed)
    # order as GLASS's rng.shuffle(normal_indices); test = UB-GOLD test fold (canonical order); held-out normals
    rng = np.random.RandomState(seed)
    fit_orig = np.asarray(Fd["fit_orig"], dtype=np.int64)
    train_indices = fit_orig[rng.permutation(len(fit_orig))]
    test_indices = np.asarray(Fd["test_orig"], dtype=np.int64)
    test_labels = y_test.astype(np.float64)
    hold_indices = np.asarray(Fd["hold_orig"], dtype=np.int64)

    print(f"\nDataset: {config.dataset_name} | Seed: {seed} | "
          f"Train: {len(train_indices)}, Test: {len(test_indices)} | Held-out: {len(hold_indices)}", flush=True)

    sample_canon = dataset.get_canonicalized_features(0)
    canon_input_dim = sample_canon.shape[1]

    # Text encoder -- CHANGE (D6): cached embeddings (GLASS's own content-hashed cache), Qwen3 not reloaded
    ablation = os.environ.get('GLASS_ABLATION', '')
    assert ablation == ''
    text_encoder, raw_text_cache = text_embeddings(G, dataset, config, text_device, device, allow_encode=False)
    text_encoder.projections.to(device)

    # Model
    model = G.gm.GLASSModel(config, canon_input_dim=canon_input_dim).to(device)

    param_groups = [
        {'params': model.canon_mlp.parameters(), 'lr': config.lr},
        {'params': model.graph_encoder.parameters(), 'lr': config.lr},
        {'params': model.graph_projections.parameters(), 'lr': config.lr},
        {'params': model.prototypes.log_kappas.parameters(), 'lr': config.lr * 0.5},
        {'params': [model.slice_weights], 'lr': config.lr * 0.1},
    ]
    if len(list(text_encoder.projections.parameters())) > 0:
        param_groups.append({'params': text_encoder.projections.parameters(), 'lr': config.text_proj_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    warmup_phase = 20

    # ---- CHANGE (D4): instrumentation state --------------------------------------------------------------
    C = Ctx()
    C.traj, C.train_loss, C.S = [], [], {"test": [], "hold": []}
    C.t_train = C.t_eval = 0.0
    own = None                                   # GLASS's own final results at the replayed stop point

    def record_eval(epoch_done, res, cap, t_eval0):
        """One evaluation point: GLASS in-loop results + reference-set scorer + held-out scores (isolated)."""
        with H.rng_isolated():
            refs, rweights = reference_scores(G, rules, model, raw_text_cache, dataset, config, device,
                                              train_indices, {"test": test_indices, "hold": hold_indices})
            htr = holdout_train_scorers(G, model, text_encoder, raw_text_cache, dataset, train_indices,
                                        hold_indices, config, device)
        test_main, hold_main = refs["test"][DEFAULT_RULE], refs["hold"][DEFAULT_RULE]
        rec = H.eval_metrics(metric, y_test, test_main)
        C.S["test"].append(test_main)
        C.S["hold"].append(hold_main)
        rule_auc, rule_hold = {}, {}
        for r in rule_names:
            C.S.setdefault("r__" + r, []).append(refs["test"][r])
            C.S.setdefault("h__" + r, []).append(refs["hold"][r])
            m = H.eval_metrics(metric, y_test, refs["test"][r])
            rule_auc[r] = [m["auroc"], m["auprc"], m["fpr95"]]
            rule_hold[r] = H.finite_mean(refs["hold"][r])
        for k in CAP_TEST_KEYS:
            C.S.setdefault("t__" + k, []).append(np.asarray(cap.get(k, np.full(n_test, np.nan)), dtype=np.float64).ravel())
        for k in HOLD_TRAIN_KEYS:
            C.S.setdefault("ht__" + k, []).append(htr[k])
        rec["holdout_score"] = H.finite_mean(hold_main)
        rec["holdout_loss"] = rec["holdout_score"]
        r0 = H.eval_metrics(metric, y_test, refs["test"][R0_RULE])
        rec.update(r0_auroc=r0["auroc"], r0_auprc=r0["auprc"], r0_fpr95=r0["fpr95"], r0_nonfinite=r0["nonfinite"],
                   r0_holdout=rule_hold[R0_RULE], rule_metrics=rule_auc, rule_holdout=rule_hold,
                   rule_weights_default=rweights[DEFAULT_RULE],
                   glass=glass_summary(res), glass_ens_auroc=float(res['ensemble']['auroc']),
                   glass_weight_graph=float(res['ensemble']['weight_graph']),
                   glass_sms_best_alpha=float(res['sms'].get('best_alpha', float('nan'))),
                   glass_best_ens_auroc=float(best_ens_auroc), glass_best_epoch=int(best_epoch),
                   glass_patience_counter=int(patience_counter), glass_stopped=bool(stopped),
                   epoch=int(epoch_done), train_loss=(C.train_loss[-1] if C.train_loss else float("nan")),
                   t=time.time() - t_start)
        C.traj.append(rec)
        C.t_eval += time.time() - t_eval0
        if verbose:
            print(f"[{run_id}] ep {epoch_done} loss {rec['train_loss']:.4f} auroc[{DEFAULT_RULE}] {rec['auroc']:.4f} "
                  f"hold {rec['holdout_score']:.4f} | glass ens {rec['glass_ens_auroc']:.1f} "
                  f"best {best_ens_auroc:.1f}@{best_epoch} pat {patience_counter}{' STOPPED' if stopped else ''} "
                  f"t={rec['t']:.0f}s", flush=True)

    def own_final(stop_epoch):
        """CHANGE (D3): GLASS's own end-of-training block (train_single_run l.577-591) at the replayed stop point,
        then evaluate_reference.py on the saved best checkpoint (l.331-360); model and RNG are restored afterwards."""
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved_tp = {k: v.detach().clone() for k, v in text_encoder.projections.state_dict().items()}
        rng0 = H.rng_state()
        # ---- copy of l.577-591 ----
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        text_encoder.projections.load_state_dict({k: v.to(device) for k, v in best_text_proj_state.items()})
        capF = {}
        final = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                                     train_indices, test_indices, test_labels, config, device, capF)
        methods = ['graph_knn', 'text_knn', 'ensemble', 'fusion_knn', 'vmf', 'sms']
        best_method = max(methods, key=lambda m: final[m]['auroc'])
        best_res = final[best_method]
        print(f"  Final: graph={final['graph_knn']['auroc']:.1f}% text={final['text_knn']['auroc']:.1f}% "
              f"ens={final['ensemble']['auroc']:.1f}% vmf={final['vmf']['auroc']:.1f}% "
              f"fus={final['fusion_knn']['auroc']:.1f}% → best={best_method}({best_res['auroc']:.1f}%) @ep{best_epoch}",
              flush=True)
        # ---- evaluate_reference.py on the checkpoint GLASS saves (model_state = best_state) ----
        with H.rng_isolated():
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            refF, rwF = reference_scores(G, rules, model, raw_text_cache, dataset, config, device,
                                         train_indices, {"test": test_indices})
        model.load_state_dict(saved)
        text_encoder.projections.load_state_dict(saved_tp)
        H.set_rng_state(rng0)
        o = dict(stop_epoch=int(stop_epoch), best_epoch=int(best_epoch), best_ens_auroc=float(best_ens_auroc),
                 early_stopped=bool(stop_epoch < config.epochs), glass_final=glass_summary(final),
                 best_method=best_method, best_res={k: float(best_res[k]) for k in ('auroc', 'auprc', 'fpr95')},
                 train_metrics={}, ref_metrics={}, ref_weights=rwF)
        arrays = {}
        for m in methods:
            m_ = H.eval_metrics(metric, y_test, capF[m])
            o["train_metrics"][m] = [m_["auroc"], m_["auprc"], m_["fpr95"]]
            arrays["own_train__" + m] = np.asarray(capF[m], dtype=np.float64)
        for r in rule_names:
            m_ = H.eval_metrics(metric, y_test, refF["test"][r])
            o["ref_metrics"][r] = [m_["auroc"], m_["auprc"], m_["fpr95"]]
            o.setdefault("ref_glass_metrics", {})[r] = G.ut.compute_metrics(test_labels, refF["test"][r])
            arrays["own_ref__" + r] = refF["test"][r]
        o["arrays"] = arrays
        return o

    # ---- resume (CHANGE D7) --------------------------------------------------------------------------------
    state = torch.load(ckpath, map_location="cpu", weights_only=False) if osp.exists(ckpath) else None
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(state["epoch"] + 1 if state is not None else 0),
               t0=time.strftime("%Y-%m-%d %H:%M:%S"))
    segments = (list(state.get("segments", [])) if state is not None else []) + [seg]

    if state is None:
        # Evaluate baseline (no training)
        t_ev = time.time()
        cap = {}
        baseline = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                                        train_indices, test_indices, test_labels, config, device, cap)
        print(f"  Epoch 0: graph={baseline['graph_knn']['auroc']:.1f}% text={baseline['text_knn']['auroc']:.1f}% "
              f"ens={baseline['ensemble']['auroc']:.1f}% (w_g={baseline['ensemble']['weight_graph']:.1f})", flush=True)

        best_ens_auroc = baseline["ensemble"]["auroc"]
        best_epoch = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
        patience_counter = 0
        stopped = False                            # CHANGE (D3): replayed early stop
        record_eval(0, baseline, cap, t_ev)        # CHANGE (D4)
        start_epoch = 0
    else:
        print(f"[resume] {ckpath} at epoch {state['epoch']}", flush=True)
        model.load_state_dict(state["model"])
        text_encoder.projections.load_state_dict(state["text_proj"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        best_ens_auroc, best_epoch = state["best_ens_auroc"], state["best_epoch"]
        best_state, best_text_proj_state = state["best_state"], state["best_text_proj_state"]
        patience_counter, stopped, own = state["patience_counter"], state["stopped"], state["own"]
        C.traj, C.train_loss = state["traj"], state["train_loss"]
        C.S = state["S"]
        start_epoch = state["epoch"] + 1
        H.set_rng_state(state["rng"])
    last_save = time.time()

    def payload(ep):
        return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    text_proj=text_encoder.projections.state_dict(), opt=optimizer.state_dict(),
                    sched=scheduler.state_dict(), best_ens_auroc=best_ens_auroc, best_epoch=best_epoch,
                    best_state=best_state, best_text_proj_state=best_text_proj_state,
                    patience_counter=patience_counter, stopped=stopped, own=own, traj=C.traj,
                    train_loss=C.train_loss, S=C.S, epoch=ep, rng=H.rng_state(), segments=segments)

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        model.train()
        text_encoder.projections.train()

        in_warmup = epoch < warmup_phase

        if len(train_indices) <= 256:
            batches = [train_indices]
        else:
            perm_idx = np.random.permutation(len(train_indices))
            shuffled = train_indices[perm_idx]
            batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]

        epoch_loss = 0
        n_batches = 0
        _epoch_t0 = time.time()

        for bi in batches:
            if len(bi) < 4:
                continue

            if T._batch_cache is not None:
                x, ei, b, sp = T._batch_cache.get_batch(bi, device)
            else:
                x, ei, b, sp = T.prepare_batch(dataset, bi, config, device)
            raw_emb = raw_text_cache[bi].to(device)
            text_slices = text_encoder.project_to_slices(raw_emb)
            z_slices, _ = model.encode_graph(x, ei, b, sp)

            # Loss 1: Soft cosine alignment (graph <-> text per pair)
            loss_align = torch.tensor(0.0, device=device)
            for z_s, e_s in zip(z_slices, text_slices):
                pos_sim = (z_s * e_s).sum(dim=-1)  # [B]
                loss_align = loss_align + (1 - pos_sim).mean()
            loss_align = loss_align / len(z_slices)

            # Loss 2: Prototype compactness + negatives (phase 2)
            loss_proto = torch.tensor(0.0, device=device)
            if not in_warmup:
                energy_normal = model.prototypes.energy(z_slices)
                loss_compact = torch.log(torch.cosh(energy_normal)).mean()

                loss_margin = torch.tensor(0.0, device=device)
                skip_neg = (T._batch_cache is None) and (not T.GPU_ACCEL) and len(dataset.data_list) > 4500  # Skip expensive negatives for large datasets
                if not skip_neg:
                 try:
                  if T._batch_cache is not None:
                    nx, nei, nb, nsp = T._batch_cache.get_negative_batch(bi, device)
                  else:
                    nx, nei, nb, nsp = T.prepare_negatives(dataset, bi, config, device)
                  z_neg, _ = model.encode_graph(nx, nei, nb, nsp)
                  energy_neg = model.prototypes.energy(z_neg)
                  loss_margin = F.relu(config.margin - (energy_neg - energy_normal)).mean()
                 except Exception:
                  pass

                loss_proto = loss_compact + loss_margin

            # Loss 3: Ortho
            loss_ortho = model.ortho_reg_fn(model.graph_projections, text_encoder.projections if config.mrl_mode == 'learned' else None)

            # Weighting
            proto_w = 0 if in_warmup else min(1.0, (epoch - warmup_phase + 1) / 20) * config.alpha
            total_loss = loss_align + proto_w * loss_proto + config.ortho_weight * loss_ortho

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(text_encoder.projections.parameters()), 1.0
            )
            optimizer.step()

            # EMA prototype update (phase 2)
            if not in_warmup:
                with torch.no_grad():
                    model.prototypes.update_prototypes_ema([z.detach() for z in z_slices])

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        # CHANGE (D5): TrainingLogger (JSON under src/results) replaced by the in-memory train_loss list
        C.train_loss.append(float(avg_loss))

        scheduler.step()

        # Initialize prototypes at start of phase 2
        if epoch == warmup_phase - 1:
            with torch.no_grad():
                if T._batch_cache is not None:
                    init_x, init_ei, init_b, init_sp = T._batch_cache.get_batch(train_indices[:min(256, len(train_indices))], device)
                else:
                    init_x, init_ei, init_b, init_sp = T.prepare_batch(
                        dataset, train_indices[:min(256, len(train_indices))], config, device
                    )
                z_init, _ = model.encode_graph(init_x, init_ei, init_b, init_sp)
                model.prototypes.init_from_data(z_init)
            print(f"  [Epoch {epoch+1}] Prototypes initialized [{time.time()-_epoch_t0:.0f}s]", flush=True)
        C.t_train += time.time() - t0

        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            t_ev = time.time()
            cap = {}
            res = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                                       train_indices, test_indices, test_labels, config, device, cap)
            ens_auroc = res['ensemble']['auroc']

            if not stopped:                        # CHANGE (D3): selection state frozen after the replayed stop
                if ens_auroc > best_ens_auroc:
                    best_ens_auroc = ens_auroc
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 5  # We eval every 5 epochs

            record_eval(epoch + 1, res, cap, t_ev)  # CHANGE (D4)

            if not stopped and patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1} (replayed; training continues for the label-free rows)",
                      flush=True)
                stopped = True                      # CHANGE (D3): original `break`
                own = own_final(epoch + 1)

        if epoch == config.epochs - 1 and own is None:
            own = own_final(config.epochs)          # original: after the loop (no early stop)

        # CHANGE (D7): checkpoint every ckpt_every_s and at the deadline; resume tests stop at EP_TEST_STOP_EPOCH
        now = time.time()
        stop = (deadline is not None and now > deadline) or (H.TEST_STOP_EPOCH > 0 and epoch + 1 == H.TEST_STOP_EPOCH
                                                             and seg["start_epoch"] <= epoch)
        if (stop or now - last_save > ckpt_every_s) and epoch < config.epochs - 1:
            H.save_ckpt(ckpath, payload(epoch))
            last_save = now
        if stop and epoch < config.epochs - 1:
            print(f"[{run_id}] deadline reached after epoch {epoch + 1}; checkpoint saved", flush=True)
            return "incomplete"
    # ================= end of train_single_run() copy ====================================================

    seg["t1"] = time.strftime("%Y-%m-%d %H:%M:%S")
    epochs = [r["epoch"] for r in C.traj]
    assert epochs == [0, 1] + list(range(5, config.epochs + 1, 5)), epochs
    meta_out = dict(method=METHOD, dataset=DS, fold=fold, hp=hp, seed=seed, run_id=run_id,
                    ubgold_sh_line=hp_line, hp_notes=[f"GLASS name {config.dataset_name}"],
                    hp_meaning=("README single-domain line (only released configuration; possibly test-tuned)"
                                if hp == "ubgold" else "train_single.py argparse defaults (identical to README line)"),
                    hparams=dict({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                  for k, v in sorted(vars(a).items())},
                                 num_epoch=int(config.epochs), eval_freq=5, eval_epochs="0,1,5,10,...",
                                 gin_layers=config.gin_layers, gin_hidden=config.gin_hidden, mrl_mode=config.mrl_mode,
                                 matryoshka_dims=str(config.matryoshka_dims), num_prototypes=config.num_prototypes,
                                 weight_decay=config.weight_decay, text_model=TEXT_MODEL),
                    default_rule=DEFAULT_RULE, r0_rule=R0_RULE, rules=rule_names,
                    n_train_normal_ubgold=Fd["n_all"], n_fit=len(train_indices), n_holdout=len(hold_indices),
                    holdout_seed=H.HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=int(y_test.sum()),
                    max_nodes_num=int(Fd["meta"]["max_nodes_num"]), num_feat=int(Fd["meta"]["num_feat"]),
                    canon_input_dim=int(canon_input_dim), device=str(device), torch=torch.__version__,
                    cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                    threads=torch.get_num_threads(), max_epochs_override=max_epochs, glass_commit=GLASS_COMMIT,
                    harness_run_eval_sha256=sha256(osp.join(HARNESS, "run_eval.py")),
                    adapter_sha256=sha256(osp.abspath(__file__)), deterministic_algorithms=deterministic,
                    segments=segments, device_all="+".join(sorted({x["device"] for x in segments})))
    own_arrays = own.pop("arrays")
    meta_out["own"] = own
    meta_out["seconds"] = time.time() - t_start
    meta_out["seconds_train_this_segment"] = C.t_train
    meta_out["seconds_eval_this_segment"] = C.t_eval
    meta_out["epochs_this_segment"] = config.epochs - seg["start_epoch"]
    meta_out["sec_per_epoch_train"] = C.t_train / max(1, meta_out["epochs_this_segment"])
    meta_out["n_eval"] = len(C.traj)
    meta_out["memory"] = H.peak_mem()
    meta_out["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    for k, v in C.S.items():
        assert len(v) == len(epochs), (k, len(v), len(epochs))
    npz = osp.join(SCORES, run_id + ".npz")
    full = Fd["full"]
    extra = dict(
        y_test=y_test, test_index=np.asarray(Fd["test_orig"], dtype=np.int64),
        hold_pos=np.asarray(Fd["hold_pos"], dtype=np.int64), hold_index=hold_indices,
        fit_index_order=train_indices,
        test_n_nodes=np.array([int(full[i].num_nodes) for i in test_indices], dtype=np.int64),
        test_n_edges=np.array([int(full[i].edge_index.size(1)) // 2 for i in test_indices], dtype=np.int64),
        hold_n_nodes=np.array([int(full[i].num_nodes) for i in hold_indices], dtype=np.int64),
        hold_n_edges=np.array([int(full[i].edge_index.size(1)) // 2 for i in hold_indices], dtype=np.int64),
        num_epoch=np.int64(config.epochs), eval_freq=np.int64(5))
    extra.update(own_arrays)
    H.save_scores(npz, epochs, C.S, extra)
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


# ----------------------------------------------------------------------------------------------
# --prep: GLASS dataset-level precomputation (labels never used)
# ----------------------------------------------------------------------------------------------
def prep(DS, encode=True):
    H = load_harness()
    G = load_glass()
    a, _ = resolve_hparams(DS, "ubgold")
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = make_config(G, a, dev_str)
    cwd = os.getcwd()
    os.chdir(H.DATA_RUN)
    try:
        full = load_full_dataset(H, DS)
    finally:
        os.chdir(cwd)
    t0 = time.time()
    dataset = build_glass_dataset(G, config, DS, full, allow_build=True)
    t1 = time.time()
    out = dict(DS=DS, glass_name=config.dataset_name, n_graphs=len(full), feature_seconds=round(t1 - t0, 1))
    if encode:
        device = torch.device(dev_str)
        _, raw = text_embeddings(G, dataset, config, dev_str, device, allow_encode=True)
        out.update(text_seconds=round(time.time() - t1, 1), emb_shape=list(raw.shape),
                   emb_finite=bool(torch.isfinite(raw).all()), device=dev_str,
                   cuda_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    print("[prep]", json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--hp", default="ubgold", choices=["ubgold", "hpfree"])
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke tests only (use with --tag)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EP_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    ap.add_argument("--prep", action="store_true", help="build GLASS feature cache (+ text embeddings unless --no_encode)")
    ap.add_argument("--no_encode", action="store_true")
    a = ap.parse_args()
    if a.method != METHOD:
        sys.exit(f"this adapter runs only {METHOD}")
    torch.set_num_threads(a.threads)
    if a.prep:
        prep(a.dataset, encode=not a.no_encode)
        sys.exit(0)
    if a.fold is None:
        sys.exit("--fold is required")
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag (smoke runs must not be mistaken for real runs)")
    st = run(a.dataset, a.fold, hp=a.hp, holdout_frac=a.holdout_frac, deadline=a.deadline, max_epochs=a.max_epochs,
             ckpt_every_s=a.ckpt_every_s, tag=a.tag)
    sys.exit(0 if st == "done" else 3)
