"""GRAIL-0 public API (training-free, permutation-invariant motif-likelihood detector).

    from grail0 import api
    model = api.fit(train_graphs, engine="esu", kmax=5, seed=0, n_labels=L)
    out = api.score(model, test_graphs)      # dict of arrays, higher = more anomalous
    ex = api.explain(model, test_graphs[0])  # per-channel / per-graphlet / per-node contributions

Graphs use the common.data format: {'n', 'edges' (m,2) i<j, 'x' or None, 'node_label' or None,
'gid'}. Precomputed graphlet features (graphlets.graph_features output) can be passed via
`feats=` (list aligned with the graphs) or attached as g['_gl']; otherwise they are computed.

Label-free calibration (cross-fitting): the normal training graphs are split in two halves
(fixed seed). Channel models are fitted on one half and robustly standardized on the other:
z_c = (s_c - median_cal) / (1.4826 * MAD_cal); this is done both ways and the two z's are
averaged. Combined scores (fixed a priori): sum_z = sum over channels, max_z, sum_z_noS0.
Cross-conformal p-values: p(G) = (1 + sum_j #{cal_j : S_j(cal) >= S_j(G)}) / (N_train + 1).
"""
import numpy as np

from grail0 import graphlets as GL
from grail0.channels import Channels, CONFIG

CHANNELS = ["S0", "A", "B", "C"]


def featurize(graphs, engine="esu", kmax=5, feats=None):
    if feats is not None:
        return list(feats)
    out = []
    for g in graphs:
        f = g.get("_gl") if isinstance(g, dict) else None
        out.append(f if f is not None else GL.graph_features(g, engine, kmax))
    return out


def _node_inputs(graphs, use_labels, use_x):
    labels = [np.asarray(g["node_label"]) for g in graphs] if use_labels else None
    X = [np.asarray(g["x"]) for g in graphs] if use_x else None
    return labels, X


def _robust(s):
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(s))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return med, scale


def combos(z, chans):
    """Fixed a-priori combination rules over the available channels."""
    out = {}
    Z = np.stack([z[c] for c in chans], 1)
    out["sum_z"] = Z.sum(1)
    out["max_z"] = Z.max(1)
    noS0 = [c for c in chans if c != "S0"]
    out["sum_z_noS0"] = np.stack([z[c] for c in noS0], 1).sum(1)
    for c in chans:
        out[c] = z[c]
    return out


def fit(train_graphs, engine="esu", kmax=5, seed=0, n_labels=None, feats=None,
        use_labels=None, use_x=None, config=None):
    F = featurize(train_graphs, engine, kmax, feats)
    N = len(F)
    if use_labels is None:
        use_labels = all(g.get("node_label") is not None for g in train_graphs)
    if use_x is None:
        use_x = all(g.get("x") is not None for g in train_graphs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    halves = [np.sort(perm[: N // 2]), np.sort(perm[N // 2:])]
    subs = []
    for j in range(2):
        fi, ci = halves[j], halves[1 - j]
        lab_f, X_f = _node_inputs([train_graphs[i] for i in fi], use_labels, use_x)
        cm = Channels(seed=seed + j, n_labels=n_labels, config=config).fit([F[i] for i in fi], lab_f, X_f)
        lab_c, X_c = _node_inputs([train_graphs[i] for i in ci], use_labels, use_x)
        raw = cm.score([F[i] for i in ci], lab_c, X_c)
        stats = {c: _robust(raw[c]) for c in raw}
        cal_z = {c: (raw[c] - stats[c][0]) / stats[c][1] for c in raw}
        subs.append({"model": cm, "stats": stats, "cal_z": cal_z, "cal_idx": ci, "fit_idx": fi})
    chans = [c for c in CHANNELS if c in subs[0]["stats"]]
    cal_combos = [combos(s["cal_z"], chans) for s in subs]
    return {"subs": subs, "channels": chans, "engine": engine, "kmax": kmax, "seed": seed,
            "use_labels": use_labels, "use_x": use_x, "n_train": N, "cal_combos": cal_combos,
            "config": dict(CONFIG, **(config or {}))}


def score(model, graphs, feats=None, conformal=True):
    F = featurize(graphs, model["engine"], model["kmax"], feats)
    lab, X = _node_inputs(graphs, model["use_labels"], model["use_x"])
    chans = model["channels"]
    zs, raws, per_split = [], [], []
    for s in model["subs"]:
        raw = s["model"].score(F, lab, X)
        z = {c: (raw[c] - s["stats"][c][0]) / s["stats"][c][1] for c in chans}
        zs.append(z); raws.append(raw); per_split.append(combos(z, chans))
    z = {c: np.mean([zz[c] for zz in zs], 0) for c in chans}
    out = combos(z, chans)
    for c in chans:
        out["raw_" + c] = np.mean([r[c] for r in raws], 0)
    if conformal:
        N = model["n_train"]
        for k in list(per_split[0].keys()):
            cnt = np.zeros(len(F))
            for j in range(2):
                cal = np.sort(model["cal_combos"][j][k])
                cnt += len(cal) - np.searchsorted(cal, per_split[j][k], side="left")
            out["p_" + k] = (1.0 + cnt) / (N + 1.0)
        # combination ablations listed a priori in strategy_modern 2.8 (ranking only):
        # Cauchy combination (Liu & Xie 2020) and Stouffer over per-channel conformal p-values
        Pm = np.stack([out["p_" + c] for c in chans], 1) - 0.5 / (N + 1.0)  # mid-p, in (0, 1)
        out["cauchy"] = np.mean(np.tan((0.5 - Pm) * np.pi), 1)
        from scipy.stats import norm
        out["stouffer"] = norm.isf(Pm).sum(1)
    return out


def explain(model, graph, feat=None):
    """Exact additive decomposition of sum_z for one graph:
    sum_z = sum(size parts) + sum(graphlet parts) + sum(node parts B) + sum(node parts C) + const.
    Node importance also distributes positive graphlet parts over the nodes that take part
    in those graphlet occurrences (share from the orbit counts)."""
    F = featurize([graph], model["engine"], model["kmax"], None if feat is None else [feat])
    f = F[0]
    lab, X = _node_inputs([graph], model["use_labels"], model["use_x"])
    T = GL.tables()
    chans = model["channels"]
    nsub = len(model["subs"])
    size_parts = np.zeros(3); g_parts = None; resid = None; expct = None
    nodeB = np.zeros(f["n"]); nodeC = np.zeros(f["n"]); const = 0.0; zc = {c: 0.0 for c in chans}
    roles = None
    for s in model["subs"]:
        raw, P = s["model"].score(F, lab, X, parts=True)
        st = s["stats"]
        size_parts += P["S0"][0] / st["S0"][1] / nsub
        gp = P["A"][0] / st["A"][1] / nsub
        g_parts = gp if g_parts is None else g_parts + gp
        resid = P["A_resid"][0] / nsub if resid is None else resid + P["A_resid"][0] / nsub
        expct = P["A_expected"][0] / nsub if expct is None else expct + P["A_expected"][0] / nsub
        nodeB += P["B_node"] / st["B"][1] / nsub
        if "C" in chans:
            nodeC += P["C_node"] / st["C"][1] / nsub
            roles = P["role"]
        for c in chans:
            const -= st[c][0] / st[c][1] / nsub
            zc[c] += (raw[c][0] - st[c][0]) / st[c][1] / nsub
    total = float(sum(zc.values()))
    recon = float(size_parts.sum() + g_parts.sum() + nodeB.sum() + nodeC.sum() + const)
    # distribute positive graphlet parts to participating nodes
    K = len(g_parts)
    share_imp = np.zeros(f["n"])
    for gr in T["graphlets"][:K]:
        k = gr["id"]
        ck = f["counts"][k]
        if g_parts[k] > 0 and ck > 0:
            part = f["gdv"][:, gr["orbits"]].sum(1) / (gr["size"] * ck)
            share_imp += g_parts[k] * part
    graphlets = []
    for gr in T["graphlets"][:K]:
        k = gr["id"]
        graphlets.append({"graphlet": gr["name"], "count": int(f["counts"][k]),
                          "expected_given_size": float(expct[k]), "log_resid": float(resid[k]),
                          "direction": "excess" if resid[k] > 0 else "deficit",
                          "contribution_z": float(g_parts[k])})
    graphlets.sort(key=lambda d: -d["contribution_z"])
    node_imp = nodeB + nodeC + share_imp
    return {"channels_z": zc, "sum_z": total,
            "size_parts_z": dict(zip(["log_n", "log_m", "density"], size_parts.tolist())),
            "graphlets": graphlets,
            "node_B_z": nodeB, "node_C_z": nodeC if "C" in chans else None,
            "node_graphlet_share_z": share_imp, "node_importance": node_imp,
            "node_role": roles, "constant": const,
            "check_sum_minus_score": recon - total}
