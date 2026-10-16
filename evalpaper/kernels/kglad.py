"""UB-GOLD KernelGLAD (WL / PK graph kernels + iForest / OCSVM) on the cached UB-GOLD folds.

Faithful to ubgold_src/GAOOD/detector/KernelGLAD.py (commit c940457):
  * grakel input = UB-GOLD's `to_grakel_graph(data, labeled=True)`: the edge set holds both
    directions of every undirected edge, and every node i gets label argmax(x_i) + 1, i.e. the
    TU node label + 1 on labelled sets and 1 on IMDB-B / REDDIT-B / COLLAB / Tox21 (UB-GOLD's
    constant features). Isolated nodes appear in the label dict but not in the edge set, as in
    UB-GOLD.
  * kernels: WeisfeilerLehman(n_iter=WL_iter, normalize=True, base_graph_kernel=VertexHistogram)
    and Propagation(t_max=WL_iter, w=PK_bin_width, normalize=True) (labeled=True is UB-GOLD's
    default and is never overridden by mymain.py / utils.init_model).
  * detectors: IsolationForest(n_estimators=IF_n_trees, max_samples=IF_sample_ratio,
    contamination='auto') and OneClassSVM(kernel='precomputed', nu=nuOCSVM), both fitted on the
    kernel matrix; anomaly score = -decision_function (UB-GOLD's orientation).
  * randomness: UB-GOLD seeds the global numpy RNG with 3407 + fold (mymain.set_seed) and never
    passes random_state, so PK's LSH and iForest draw from the global RNG. We seed the same way
    before every kernel fit and restore the post-kernel RNG state before every detector fit, so
    each (kernel, detector, config) result equals a fresh UB-GOLD run of that single config.

Two fit modes:
  * 'train' (primary, inductive one-class): kernel.fit_transform(normal training graphs),
    detector.fit(K_train), score kernel.transform(test graphs). Same train/test protocol as
    every other detector in the re-evaluation.
  * 'test' (UB-GOLD's own code path): KernelGLAD.fit() builds the kernel and fits the detector
    on `dataloader_val`, which for TU / Tox21 data IS the test loader (get_ad_dataset_TU:
    `dataloader_val = dataloader_test`), and scores those same graphs (transductive, no labels).
    UB-GOLD's released code additionally pairs these scores with labels collected in a SECOND
    pass over the shuffled test loader (shuffle=True), i.e. in a different random order; we
    pair scores with the correct labels here, and ubgold_verbatim_check.py measures the effect
    of the released pairing.
"""
import random
import time

import numpy as np
from grakel.kernels import WeisfeilerLehman, VertexHistogram, Propagation
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# UB-GOLD mymain.py KernelGLAD sub-parser defaults (the label-free 'default' setting)
DEFAULTS = dict(WL_iter=5, PK_bin_width=1, IF_n_trees=200, IF_sample_ratio=0.5, nuOCSVM=0.1)
# UB-GOLD search_utils.param_space['KernelGLAD'] (n_neighbors / n_leaf only affect LOF, which
# is not run; nuOCSVM and PK_bin_width are not searched by UB-GOLD)
SEARCH = dict(IF_n_trees=[200, 250, 300], IF_sample_ratio=[0.3, 0.4, 0.5, 0.6], WL_iter=[3, 4, 5, 6, 7])


def to_grakel(graphs):
    out = []
    for g in graphs:
        e = np.asarray(g["edges"])
        edges = {(int(i), int(j)) for i, j in e} | {(int(j), int(i)) for i, j in e}
        nl = g["node_label"]
        if nl is None:
            labels = {i: 1 for i in range(g["n"])}
        else:
            labels = {i: int(nl[i]) + 1 for i in range(g["n"])}
        out.append([edges, labels])
    return out


def make_kernel(kind, wl_iter, pk_bin_width=1):
    if kind == "WL":
        return WeisfeilerLehman(n_iter=wl_iter, normalize=True, base_graph_kernel=VertexHistogram)
    if kind == "PK":
        return Propagation(t_max=wl_iter, w=pk_bin_width, normalize=True)
    raise ValueError(kind)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def kernel_matrices(kind, wl_iter, train_g, test_g, mode, seed):
    """Return (K_fit, K_score, rng_state_after_kernel, seconds)."""
    seed_all(seed)
    t0 = time.time()
    k = make_kernel(kind, wl_iter, DEFAULTS["PK_bin_width"])
    if mode == "train":
        K_fit = np.asarray(k.fit_transform(train_g), dtype=np.float64)
        st = np.random.get_state()
        K_score = np.asarray(k.transform(test_g), dtype=np.float64)
    elif mode == "test":
        K_fit = np.asarray(k.fit_transform(test_g), dtype=np.float64)
        st = np.random.get_state()
        K_score = K_fit
    else:
        raise ValueError(mode)
    return K_fit, K_score, st, time.time() - t0


def detector_configs():
    """All detector configurations evaluated: UB-GOLD's IF search grid and the OCSVM default."""
    cfgs = []
    for nt in SEARCH["IF_n_trees"]:
        for sr in SEARCH["IF_sample_ratio"]:
            cfgs.append(("IF", dict(IF_n_trees=nt, IF_sample_ratio=sr)))
    cfgs.append(("OCSVM", dict(nuOCSVM=DEFAULTS["nuOCSVM"])))
    return cfgs


def fit_score(det, cfg, K_fit, K_score, rng_state):
    np.random.set_state(rng_state)
    if det == "IF":
        d = IsolationForest(n_estimators=cfg["IF_n_trees"], max_samples=cfg["IF_sample_ratio"],
                            contamination="auto")
    elif det == "OCSVM":
        d = OneClassSVM(kernel="precomputed", nu=cfg["nuOCSVM"])
    else:
        raise ValueError(det)
    d.fit(K_fit)
    return -d.decision_function(K_score)
