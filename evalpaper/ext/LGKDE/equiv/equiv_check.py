"""Numerical equivalence: original LGKDE (DGL 2.4.0, src/ @1bba5ef, imported unmodified) vs the PyG port.

Run (CPU, login node, <= 4 threads) with the DGL environment:
  DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
    $GLAD_ENVS/ext-lgkde-dgl/bin/python -B equiv/equiv_check.py --out equiv/results_<tag>.json

Checks (UB-GOLD fold graphs, UB-GOLD features):
  T1 GraphConv layer: forward + gradients (in>out and in<=out branches), copied weights.
  T2 Model init from the same seed: parameters bitwise identical (creation order / init calls).
  T3 Train-mode forward with identical RNG (dropout masks), deep-MMD distance matrices (intra, cross):
     original DGL loop vs port loop vs port vectorized (also with tiny blocks to force many chunks),
     KDE scores, the training loss and the gradients of every parameter.
  T4 Negative graphs: original ng.py vs port, same RNG: features and edge sets of every 'sub'/'add'
     negative, for both configurations (ubgold: swap+structure; hpfree: swap+noise+structure).
  T5 End-to-end: the original Trainer.train_epoch/evaluate (verbatim, DGL loaders over the same graphs)
     vs the port Trainer, same seed, E epochs: per-epoch loss and test density scores.
     The original runs with CUDA copy semantics for batched_graph.to('cpu') (see NOTES.md); the same
     run with the original CPU semantics is reported to quantify that device effect.
"""
import argparse
import copy
import json
import os
import os.path as osp
import sys
import time

os.environ.setdefault("DGLBACKEND", "pytorch")
HERE = osp.dirname(osp.abspath(__file__))
LG = osp.dirname(HERE)
SRC = osp.join(LG, "src")
sys.path[:] = [p for p in sys.path if osp.realpath(p or ".") != osp.realpath(HERE)]
sys.path.insert(0, LG)
sys.path.insert(1, SRC)

import numpy as np
import torch
import dgl
import dgl.nn as dglnn
from dgl.dataloading import GraphDataLoader

import model as O_model            # original src/model.py
import ng as O_ng                  # original src/ng.py
import experiment as O_exp         # original src/experiment.py
import dataprocessing as O_dp      # original src/dataprocessing.py (CustomDataset)
import lgkde_pyg as P
import lgkde_trainer as PT
import ubgold_data

UB_CFG = dict(
    base=dict(seed=42, gpu_id="0", save_dir="./outputs"),
    dataset=dict(name="X", type="TU", normal_class=None, train_ratio=0.8, mixed_anomaly_ratio=0.0,
                 self_loop=True, batch_size=None),
    model=dict(in_dim=None, hidden_dim=32, out_dim=16, num_layers=2, bandwidths=[0.01, 0.1, 1, 10, 100],
               dropout=0.2, batch_norm=True, learn_kde_weights=True),
    negative_sampling=dict(perturbation_methods=["feature_swap", "structure"], attr_swap_ratio=0.2,
                           spectral_perturb_ratio=0.2, threshold=0.5, weighted_graph=False, verbose=False,
                           big_energy_threshold=0.5, mid_energy_threshold=0.75, max_ratio=10.0),
    training=dict(epochs=100, lr=0.01, weight_decay=0.1, loss_alpha=1.0, loss_type="LogRatioLossAD",
                  epsilon=1.0e-6),
    early_stopping=dict(patience=5, min_epochs=10, min_delta=0.001),
    evaluation=dict(anomaly_threshold_percentile=0.1))


def rel(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    d = (a - b).abs().max().item() if a.numel() else 0.0
    s = max(b.abs().max().item() if b.numel() else 0.0, 1e-30)
    return dict(max_abs=d, max_rel=d / s)


def copy_graph(g):
    """New storage for every feature tensor (a GPU->CPU transfer never shares storage)."""
    g2 = g.clone()
    for k in list(g2.ndata.keys()):
        g2.ndata[k] = g2.ndata[k].clone()
    for k in list(g2.edata.keys()):
        g2.edata[k] = g2.edata[k].clone()
    return g2


def dgl_graph(g):
    """Original pipeline: raw edges -> dgl.add_self_loop -> ndata['attr']."""
    gg = dgl.graph((g["edge_index"][0], g["edge_index"][1]), num_nodes=g["n"])
    gg = dgl.add_self_loop(gg)
    gg.ndata["attr"] = g["x"].clone()
    gg.ndata["label"] = torch.zeros(g["n"], 1)
    return gg


def port_batch(graphs, gidx0=0):
    gs = [PT.build_graph(g) for g in graphs]
    return P.batch_graphs([g["x"] for g in gs], [g["edge_index"] for g in gs], list(range(gidx0, gidx0 + len(gs))))


def copy_params(o_model, p_model):
    """state_dict of the original -> port (same names: DGL GraphConv has weight/bias too)."""
    sd = o_model.state_dict()
    missing = p_model.load_state_dict(sd, strict=True)
    return missing


def t1_graphconv(graphs, R):
    bg_o = dgl.batch([dgl_graph(g) for g in graphs])
    bg_p = port_batch(graphs)
    so, do = bg_o.edges()
    R["T1_edges_identical"] = bool(torch.equal(so, bg_p.edge_index[0]) and torch.equal(do, bg_p.edge_index[1]))
    x = bg_p.x
    res = {}
    for (fi, fo) in [(x.size(1), 32), (x.size(1), 2)]:
        torch.manual_seed(0)
        lo = dglnn.GraphConv(fi, fo, norm='both', bias=True)
        torch.manual_seed(0)
        lp = P.GraphConv(fi, fo, norm='both', bias=True)
        init_equal = bool(torch.equal(lo.weight, lp.weight) and torch.equal(lo.bias, lp.bias))
        xo = x.clone().requires_grad_(True)
        xp = x.clone().requires_grad_(True)
        yo = lo(bg_o, xo)
        yp = lp(bg_p, xp)
        w = torch.randn_like(yo)
        (yo * w).sum().backward()
        (yp * w).sum().backward()
        res[f"{fi}->{fo}"] = dict(init_bitwise=init_equal, out=rel(yp, yo), grad_x=rel(xp.grad, xo.grad),
                                  grad_w=rel(lp.weight.grad, lo.weight.grad), grad_b=rel(lp.bias.grad, lo.bias.grad))
    R["T1_graphconv"] = res


def make_models(in_dim, cfg, seed):
    m = cfg["model"]
    kw = dict(in_dim=in_dim, hidden_dim=m["hidden_dim"], out_dim=m["out_dim"], num_layers=m["num_layers"],
              bandwidths=m["bandwidths"], dropout=m["dropout"], batch_norm=m["batch_norm"],
              learn_kde_weights=m["learn_kde_weights"])
    torch.manual_seed(seed)
    mo = O_model.LGKDE(**kw)
    torch.manual_seed(seed)
    mp = P.LGKDE(**kw)
    return mo, mp


def _flat_rel(g, ref):
    """relative L2 error over all parameters (pre-BN biases have a true gradient of ~0 and would make
    per-parameter max-relative errors meaningless)."""
    num = sum(float(((g[k].double() - ref[k].double()) ** 2).sum()) for k in ref)
    den = sum(float((ref[k].double() ** 2).sum()) for k in ref)
    return (num / max(den, 1e-300)) ** 0.5


def _to_double_graph(g):
    g2 = dict(g)
    g2["x"] = g["x"].double()
    return g2


def t2_t3_model(fit, neg_src, R, cfg, tag, block_elems=None):
    """fit: normal graphs (batch A); neg_src: graphs used as the cross set (batch Q).
    Every variant starts from the same parameters and the same RNG state (identical dropout masks).
    'orig64' = the original DGL code run in float64 (reference for float32 rounding)."""
    in_dim = fit[0]["x"].size(1)
    mo, mp = make_models(in_dim, cfg, 123)
    so, sp = mo.state_dict(), mp.state_dict()
    R[f"T2_init_bitwise_{tag}"] = bool(so.keys() == sp.keys() and all(torch.equal(so[k], sp[k]) for k in so))
    crit = PT.LogRatioLossAD(epsilon=1e-6)
    variants = [("orig32", "orig", None, None), ("orig64", "orig", None, None),
                ("port_loop", "port", False, None), ("port_vec", "port", True, None)]
    if block_elems:
        variants.append(("port_vec_smallblocks", "port", True, block_elems))
    variants.append(("port_vec_autograd", "port", "autograd", None))   # plain autograd blocks, no exp floor
    res = {}
    mo0, mp0 = copy.deepcopy(mo), copy.deepcopy(mp)
    for name, kind, vec, be in variants:
        dbl = name == "orig64"
        fA = [_to_double_graph(g) for g in fit] if dbl else fit
        fQ = [_to_double_graph(g) for g in neg_src] if dbl else neg_src
        model = copy.deepcopy(mo0 if kind == "orig" else mp0)
        if dbl:
            model = model.double()
            torch.set_default_dtype(torch.float64)
        if kind == "port":
            P.DGMMD.vectorized = bool(vec)
            P.DGMMD.block_impl = "autograd" if vec == "autograd" else "function"
            P.DGMMD.block_elems = be or 2 ** 25
        try:
            model.train()
            model.zero_grad()
            torch.manual_seed(777)
            if kind == "orig":
                bo = dgl.batch([dgl_graph(g) for g in fA])
                qo = dgl.batch([dgl_graph(g) for g in fQ])
                d_in = model.dgmmd.compute_distance_matrix(bo, None)
                d_cr = model.dgmmd.compute_distance_matrix(qo, bo)
            else:
                bp = port_batch(fA)
                qp = port_batch(fQ, gidx0=10 ** 6)
                d_in = model.dgmmd.compute_distance_matrix(bp, None)
                d_cr = model.dgmmd.compute_distance_matrix(qp, bp)
            p_in = model.compute_kde_scores(d_in)
            p_cr = model.compute_kde_scores(d_cr)
            kk = min(p_in.numel(), p_cr.numel())
            loss = crit(p_in[:kk], p_cr[:kk])
            loss.backward()
        finally:
            torch.set_default_dtype(torch.float32)
        res[name] = dict(d_in=d_in.detach().double(), d_cr=d_cr.detach().double(), p_in=p_in.detach().double(),
                         p_cr=p_cr.detach().double(), loss=float(loss),
                         grads={k: v.grad.detach().double() for k, v in model.named_parameters() if v.grad is not None},
                         bn={k: v.double() for k, v in model.state_dict().items() if "running" in k},
                         sym=bool(torch.equal(d_in, d_in.t()) and float(d_in.diagonal().abs().max()) == 0.0))
    P.DGMMD.vectorized = True
    P.DGMMD.block_impl = "function"
    P.DGMMD.block_elems = 2 ** 25
    out = {}
    for name in res:
        if name == "orig64":
            continue
        r = res[name]
        row = {}
        for refname in ("orig32", "orig64"):
            if refname == name:
                continue
            o = res[refname]
            row["vs_" + refname] = dict(
                dist_intra_max_rel=rel(r["d_in"], o["d_in"])["max_rel"], dist_cross_max_rel=rel(r["d_cr"], o["d_cr"])["max_rel"],
                kde_intra_max_rel=rel(r["p_in"], o["p_in"])["max_rel"], kde_cross_max_rel=rel(r["p_cr"], o["p_cr"])["max_rel"],
                loss_rel=abs(r["loss"] - o["loss"]) / max(abs(o["loss"]), 1e-30),
                grad_rel_l2=_flat_rel(r["grads"], o["grads"]),
                grad_rel_l2_kde_logits=_flat_rel({"k": r["grads"]["kde_logits"]}, {"k": o["grads"]["kde_logits"]}),
                bn_running_max_rel=max(rel(r["bn"][k], o["bn"][k])["max_rel"] for k in r["bn"]))
        row["intra_symmetric_zero_diag"] = r["sym"]
        out[name] = row
    R[f"T3_forward_backward_{tag}"] = out
    R[f"T3_sizes_{tag}"] = dict(n_graphs_A=len(fit), n_graphs_Q=len(neg_src),
                               nodes_A=int(sum(g["n"] for g in fit)), nodes_Q=int(sum(g["n"] for g in neg_src)))


def t4_negatives(fit, R, cfg, tag):
    """Same RNG state -> same negatives. Original with CUDA copy semantics (graphs_cpu is a copy)."""
    ns = dict(cfg["negative_sampling"])
    gen_o = O_ng.NegativeSampleGenerator(**ns)
    gen_p = P.NegativeSampleGenerator(**ns)
    bo = dgl.batch([dgl_graph(g) for g in fit])
    torch.manual_seed(99)
    graphs_cpu = copy_graph(bo)      # CUDA semantics: .to('cpu') from a GPU graph is a copy
    gl = dgl.unbatch(graphs_cpu)
    sub_o = [gen_o.generate_negative_graph(g, increase_edges=False) for g in gl]
    add_o = [gen_o.generate_negative_graph(g, increase_edges=True) for g in gl]
    bp = port_batch(fit)
    torch.manual_seed(99)
    tr = PT.Trainer.__new__(PT.Trainer)
    tr.train_graphs = [dict(PT.build_graph(g), gidx=i) for i, g in enumerate(fit)]
    tr.test_graphs = []
    gp = tr._unbatch_cpu(bp)
    sub_p = [gen_p.generate_negative_graph(g, increase_edges=False) for g in gp]
    add_p = [gen_p.generate_negative_graph(g, increase_edges=True) for g in gp]
    nfeat, nedge, nedge_order, changed = 0, 0, 0, []
    for lo, lp in ((sub_o, sub_p), (add_o, add_p)):
        for go, (xp, ep) in zip(lo, lp):
            if not torch.equal(go.ndata["attr"], xp):
                nfeat += 1
            so, do = go.edges()
            eo = sorted(zip(so.tolist(), do.tolist()))
            epl = sorted(zip(ep[0].tolist(), ep[1].tolist()))
            if eo != epl:
                nedge += 1
            if not (torch.equal(so, ep[0]) and torch.equal(do, ep[1])):
                nedge_order += 1
    for go, g in zip(sub_o + add_o, list(gl) + list(gl)):
        changed.append(abs(go.num_edges() - g.num_edges()) / max(1, g.num_edges()))
    R[f"T4_negatives_{tag}"] = dict(n_graphs=len(fit), feature_mismatches=nfeat, edge_set_mismatches=nedge,
                                    edge_order_mismatches=nedge_order,
                                    rng_state_equal_after=bool(True),
                                    mean_rel_edge_count_change=float(np.mean(changed)))
    # is the RNG stream left in the same state? (re-run both and compare next draw)
    torch.manual_seed(99); gl2 = dgl.unbatch(copy_graph(bo))
    [gen_o.generate_negative_graph(g, False) for g in gl2]; [gen_o.generate_negative_graph(g, True) for g in gl2]
    a = torch.rand(3)
    torch.manual_seed(99); gp2 = tr._unbatch_cpu(bp)
    [gen_p.generate_negative_graph(g, False) for g in gp2]; [gen_p.generate_negative_graph(g, True) for g in gp2]
    b = torch.rand(3)
    R[f"T4_negatives_{tag}"]["rng_state_equal_after"] = bool(torch.equal(a, b))


class _OrigTrainer(O_exp.Trainer):
    """Original Trainer with the data loading replaced by DGL loaders over the given graphs; train_epoch and
    evaluate are the original methods (inherited, unmodified)."""

    def __init__(self, config, fit, test, y_test, seed):
        self.config = config
        self.device = torch.device("cpu")
        self.save_dir = config["base"]["save_dir"]
        self.timestamp = "equiv"
        O_exp.set_global_seed(seed)
        tr = O_dp.CustomDataset([dgl_graph(g) for g in fit], torch.zeros(len(fit), dtype=torch.long))
        te = O_dp.CustomDataset([dgl_graph(g) for g in test], torch.as_tensor(y_test, dtype=torch.long))
        self.train_loader = GraphDataLoader(tr, batch_size=len(tr), shuffle=True)
        self.test_loader = GraphDataLoader(te, batch_size=len(te), shuffle=False)
        self.train_loader_noshuffle = GraphDataLoader(tr, batch_size=len(tr), shuffle=False)
        self.init_components()


def _patch_to(cuda_copy):
    orig = dgl.DGLGraph.to

    def to(self, device, **kw):
        g = orig(self, device, **kw)
        if cuda_copy and str(device) == "cpu":
            g = copy_graph(g)          # emulate a GPU->CPU transfer: new storage
        return g
    dgl.DGLGraph.to = to
    return orig


T5_VARIANTS = {   # name: (implementation, float64?, CUDA copy semantics?, vectorized MMD?)
    "orig32_cuda": ("orig", False, True, None), "orig32_cpu": ("orig", False, False, None),
    "orig64_cuda": ("orig", True, True, None),
    "port32_vec": ("port", False, True, True), "port32_loop": ("port", False, True, False),
    "port64_vec": ("port", True, True, True), "port64_loop": ("port", True, True, False),
    # float64 original whose adjacency (SVD structure perturbation) stays float32 as in a normal run: DGL's
    # g.adj() follows torch's default dtype, so plain orig64 also runs the SVD in float64 and changes a few
    # negative graphs (4/160 on MUTAG fold 0)
    "orig64a32_cuda": ("orig", True, True, None)}


def t5_run_variant(fold, cfg, epochs, variant):
    kind, dbl, cuda_copy, vec = T5_VARIANTS[variant]
    seed = fold["seed"]
    fit = [_to_double_graph(g) for g in fold["fit"]] if dbl else fold["fit"]
    test = [_to_double_graph(g) for g in fold["test"]] if dbl else fold["test"]
    losses, scores, t0 = [], [], time.time()
    orig_to = _patch_to(cuda_copy) if kind == "orig" else None
    orig_adj = dgl.DGLGraph.adj
    if variant == "orig64a32_cuda":
        def adj32(self, *a, **k):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                return orig_adj(self, *a, **k)
            finally:
                torch.set_default_dtype(prev)
        dgl.DGLGraph.adj = adj32
    try:
        if kind == "orig":
            T = _OrigTrainer(cfg, fit, test, fold["y_test"], seed)
        else:
            P.DGMMD.vectorized = vec
            T = PT.Trainer(cfg, fit, test, fold["y_test"], torch.device("cpu"), seed)
        if dbl:
            T.model.double()
            torch.set_default_dtype(torch.float64)
        for ep in range(1, epochs + 1):
            losses.append(float(T.train_epoch(ep)))
            if kind == "orig":
                m = T.evaluate()
                scores.append(np.asarray(m["combined_scores"][len(fit):], dtype=np.float64))
            else:
                sc, _ = T.evaluate()
                scores.append(sc.double().numpy())
    finally:
        torch.set_default_dtype(torch.float32)
        P.DGMMD.vectorized = True
        if orig_to is not None:
            dgl.DGLGraph.to = orig_to
        dgl.DGLGraph.adj = orig_adj
    return dict(variant=variant, losses=losses, scores=scores, y=np.asarray(fold["y_test"]),
                seconds=time.time() - t0, params={k: v.detach().double().clone() for k, v in T.model.state_dict().items()})


def t5_compare(runs, ref_names=("orig64_cuda", "orig64a32_cuda", "orig32_cuda")):
    from sklearn.metrics import roc_auc_score
    out = {}
    for name, r in runs.items():
        row = dict(seconds=r["seconds"], loss=r["losses"],
                   auroc=[float(roc_auc_score(r["y"], -s)) for s in r["scores"]])
        for ref in ref_names:
            if ref == name or ref not in runs:
                continue
            o = runs[ref]
            n = min(len(o["losses"]), len(r["losses"]))
            row["vs_" + ref] = dict(
                loss_rel=[abs(r["losses"][i] - o["losses"][i]) / max(abs(o["losses"][i]), 1e-30) for i in range(n)],
                score_rel_l2=[float(np.linalg.norm(r["scores"][i] - o["scores"][i]) / max(np.linalg.norm(o["scores"][i]), 1e-300)) for i in range(n)],
                score_rank_spearman=[float(_spearman(r["scores"][i], o["scores"][i])) for i in range(n)])
        out[name] = row
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="components", choices=["components", "t5", "compare"])
    ap.add_argument("--datasets", default="MUTAG,BZR,IMDB-BINARY,ENZYMES")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_fit", type=int, default=60)
    ap.add_argument("--t5_ds", default="MUTAG")
    ap.add_argument("--t5_epochs", type=int, default=10)
    ap.add_argument("--t5_variant", default="port32_vec")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--t3_only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    R = dict(versions=dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__),
             started=time.strftime("%Y-%m-%d %H:%M:%S"), lgkde_commit="1bba5ef00067230fa327a630d67cabbd53fdf523")
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    if a.mode == "t5":
        fd = ubgold_data.load_fold(a.t5_ds, a.fold)
        r = t5_run_variant(fd, UB_CFG, a.t5_epochs, a.t5_variant)
        r.update(dataset=a.t5_ds, fold=a.fold, versions=R["versions"])
        torch.save(r, a.out)
        print(a.t5_variant, a.t5_ds, f"{r['seconds']:.0f}s", r["losses"], flush=True)
        return
    if a.mode == "compare":
        runs = {}
        for fn in a.inputs:
            r = torch.load(fn, weights_only=False)
            runs.setdefault((r["dataset"], r["fold"]), {})[r["variant"]] = r
        for (ds, fo), rr in runs.items():
            R[f"T5_end_to_end_{ds}_fold{fo}"] = t5_compare(rr)
        json.dump(R, open(a.out, "w"), indent=1, default=float)
        print(json.dumps(R, indent=1, default=float)[:20000])
        return
    H = ubgold_data.load_harness()
    for DS in a.datasets.split(","):
        fd = ubgold_data.load_fold(DS, a.fold, H=H)
        fit = fd["fit"][:a.n_fit]
        q = fd["test"][:40]
        if DS == a.datasets.split(",")[0] and not a.t3_only:
            t1_graphconv(fit, R)
        cfg_h = copy.deepcopy(UB_CFG)
        t2_t3_model(fit, q, R, UB_CFG, DS, block_elems=2 ** 12)
        if a.t3_only:
            continue
        t4_negatives(fd["fit"], R, UB_CFG, DS + "_ubgold")
        cfg_h["negative_sampling"] = {}        # hpfree: NegativeSampleGenerator defaults (adds feature_noise)
        t4_negatives(fd["fit"], R, cfg_h, DS + "_hpfree")
        print(DS, "T1-T4 done", time.strftime("%H:%M:%S"), flush=True)
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float)[:30000])


if __name__ == "__main__":
    main()
