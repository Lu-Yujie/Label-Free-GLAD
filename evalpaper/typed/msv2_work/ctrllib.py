"""Control helpers for the MS-v2 design work (A28). Same functions as typed/controls_typed.control_scores
(grail0 graphlet counter ESU kmax 5, controls.gc_feats / size_feats, Ledoit-Wolf Mahalanobis, edge-label pairs), with
a per-graph feature cache so that many design variants can be scored quickly. No detector of any kind.
"""
import sys

sys.dont_write_bytecode = True
import hashlib
import os.path as osp

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
TYPED = osp.dirname(HERE)
EVALPAPER = osp.dirname(TYPED)
PRELIM = osp.dirname(EVALPAPER)
for _p in (PRELIM, EVALPAPER, TYPED):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grail0 import graphlets as GL  # noqa: E402
from grail0 import controls as C  # noqa: E402
from grail0.channels import CONFIG  # noqa: E402
from common.metrics import evaluate  # noqa: E402
import controls_typed as CT  # noqa: E402

ENGINE = CT.ENGINE
_FC = {}


def gkey(g):
    e = np.ascontiguousarray(np.asarray(g["edges"], np.int64).reshape(-1, 2))
    return hashlib.sha1(np.int64(g["n"]).tobytes() + e.tobytes()).hexdigest()


def feats(g):
    k = gkey(g)
    f = _FC.get(k)
    if f is None:
        f = GL.graph_features(g, *ENGINE)
        _FC[k] = f
    return f


def F(graphs):
    return [feats(g) for g in graphs]


def scores(train, test, L, which=("gc_lw", "size_lw", "elp_lw")):
    """Exactly controls_typed.control_scores (same functions, same order of operations)."""
    Ftr, Fte = F(train), F(test)
    out = {}
    if "size_lw" in which:
        out["size_lw"] = C.mahal(C.size_feats(Ftr), C.size_feats(Fte), C.CONFIG["ridge_S0"])
    if "gc_lw" in which:
        out["gc_lw"] = C.mahal(C.gc_feats(Ftr), C.gc_feats(Fte), C.CONFIG["ridge_A"])
    if "elp_lw" in which:
        Etr, Ete = CT.edgelab(train, L), CT.edgelab(test, L)
        out["elp_lw"] = C.mahal(np.log1p(Etr), np.log1p(Ete), CONFIG["ridge_A"])
    return out


def auroc(s_norm, s_anom):
    s = np.r_[np.asarray(s_norm, float), np.asarray(s_anom, float)]
    y = np.r_[np.zeros(len(s_norm), int), np.ones(len(s_anom), int)]
    return 100.0 * evaluate(s, y)["auroc"]


def cell_aurocs(train, test_normals, anomalies, L, which=("gc_lw", "size_lw", "elp_lw")):
    S = scores(train, list(test_normals) + list(anomalies), L, which)
    nn = len(test_normals)
    return {k: auroc(v[:nn], v[nn:]) for k, v in S.items()}
