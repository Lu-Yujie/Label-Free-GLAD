"""GRAIL-0 channels. Every model here is fitted on NORMAL graphs only; no anomaly label,
test statistic or score orientation is ever used. All constants are fixed a priori
(CONFIG) and identical for every dataset.

  S0  size           : s(G) = [log n, log(m+1), density] -> Ledoit-Wolf Gaussian; score =
                       squared Mahalanobis distance (two-sided).
  A   composition    : y_k = log(1 + c_k) per graphlet type k, regressed by least squares on
                       [1, log n, log(m+1)]; residual vector d -> Ledoit-Wolf (+ fixed ridge)
                       Gaussian; score = d' P d = sum_k A_k, A_k = d_k (P d)_k (exactly additive).
  B   structural role: diagonal GMM (K roles) on log(1 + GDV) of normal nodes; node NLL b_v;
                       graph score = mean_v b_v + mean of the top-10% b_v.
  C   attribute|role : structure-only role prior pi_v = GMM posterior from B (fixed);
                       per-role categorical over node labels (Dirichlet prior centred on the
                       global label frequencies) and, if continuous attributes are given, a
                       per-role diagonal Gaussian; MAP-EM over role parameters; node NLL
                       a_v = -log sum_r pi_vr p(x_v | r) (full NLL incl. log-variance);
                       graph score = mean + top-10% mean. Absent when the graph has no
                       node features.
"""
import math

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

CONFIG = {
    "ridge_A": 1e-2,          # added to the LW residual covariance (log-count units^2)
    "ridge_S0": 1e-6,
    "gmm_K": 8,
    "gmm_reg_covar": 1e-2,    # variance floor of the role GMM (log(1+count) units^2)
    "gmm_max_nodes": 50000,   # node subsample for fitting the GMM (fixed seed)
    "gmm_max_iter": 200,
    "top_frac": 0.10,
    "dir_beta": 10.0,         # Dirichlet pseudo-count mass for per-role label distributions
    "em_iters": 50,
    "gauss_var_floor": 1e-2,  # relative to the global (standardized) variance
}


def size_feats(F):
    out = np.zeros((len(F), 3))
    for i, f in enumerate(F):
        n, m = f["n"], f["m"]
        out[i] = [math.log(max(n, 1)), math.log(m + 1.0), (2.0 * m / (n * (n - 1))) if n > 1 else 0.0]
    return out


def comp_design(F):
    X = np.ones((len(F), 3))
    for i, f in enumerate(F):
        X[i, 1] = math.log(max(f["n"], 1)); X[i, 2] = math.log(f["m"] + 1.0)
    return X


def comp_y(F):
    return np.log1p(np.stack([f["counts"] for f in F]).astype(np.float64))


def stack_nodes(F):
    Z = np.log1p(np.concatenate([f["gdv"] for f in F], axis=0).astype(np.float64))
    sizes = np.array([f["n"] for f in F], dtype=np.int64)
    off = np.zeros(len(F) + 1, dtype=np.int64); np.cumsum(sizes, out=off[1:])
    return Z, off


def graph_agg(v, off, top_frac):
    """mean + mean of top-ceil(top_frac*n) values per graph; also returns per-node weights so
    that score = sum_v w_v * v_v exactly."""
    G = len(off) - 1
    s = np.zeros(G); w = np.zeros(len(v))
    for i in range(G):
        a, b = off[i], off[i + 1]
        n = b - a
        if n == 0:
            continue
        x = v[a:b]
        q = max(1, int(math.ceil(top_frac * n)))
        top = np.argpartition(-x, q - 1)[:q] if q < n else np.arange(n)
        ww = np.full(n, 1.0 / n); ww[top] += 1.0 / q
        w[a:b] = ww
        s[i] = float(ww @ x)
    return s, w


def _lw(Y, ridge, centered=False):
    lw = LedoitWolf(assume_centered=centered).fit(Y)
    S = lw.covariance_ + ridge * np.eye(Y.shape[1])
    mu = np.zeros(Y.shape[1]) if centered else lw.location_
    return mu, np.linalg.inv(S)


class Channels:
    def __init__(self, seed=0, n_labels=None, config=None):
        self.seed = int(seed)
        self.n_labels = n_labels
        self.cfg = dict(CONFIG); self.cfg.update(config or {})

    # ------------------------------------------------------------------ fit
    def fit(self, F, labels=None, X=None):
        cfg = self.cfg
        # S0
        S = size_feats(F)
        self.s0_mu, self.s0_P = _lw(S, cfg["ridge_S0"])
        # A
        Xd = comp_design(F); Y = comp_y(F)
        self.A_beta = np.linalg.lstsq(Xd, Y, rcond=None)[0]
        R = Y - Xd @ self.A_beta
        _, self.A_P = _lw(R, cfg["ridge_A"], centered=True)
        # B
        Z, off = stack_nodes(F)
        rng = np.random.default_rng(self.seed)
        Zfit = Z if len(Z) <= cfg["gmm_max_nodes"] else Z[rng.choice(len(Z), cfg["gmm_max_nodes"], replace=False)]
        K = min(cfg["gmm_K"], max(1, len(np.unique(Zfit, axis=0))))
        self.gmm = GaussianMixture(n_components=K, covariance_type="diag", reg_covar=cfg["gmm_reg_covar"],
                                   max_iter=cfg["gmm_max_iter"], random_state=self.seed, init_params="kmeans")
        self.gmm.fit(Zfit)
        # C
        self.has_C = labels is not None or X is not None
        if self.has_C:
            Pi = self.gmm.predict_proba(Z)
            self._fit_C(Pi, labels, X)
        return self

    def _fit_C(self, Pi, labels, X):
        cfg = self.cfg
        K = Pi.shape[1]
        self.use_lab = labels is not None
        self.use_x = X is not None
        logp = np.zeros_like(Pi)
        if self.use_lab:
            lab = np.concatenate(labels).astype(np.int64)
            L = int(self.n_labels) if self.n_labels else int(lab.max()) + 1
            self.L = L
            lab = np.minimum(lab, L)            # bin L = out-of-vocabulary
            cnt = np.bincount(lab, minlength=L + 1).astype(float)
            self.p_glob = (cnt + 1.0) / (cnt.sum() + L + 1.0)
            onehot = np.zeros((len(lab), L + 1)); onehot[np.arange(len(lab)), lab] = 1.0
        if self.use_x:
            Xa = np.concatenate(X, axis=0).astype(np.float64)
            self.x_mu = Xa.mean(0); self.x_sd = Xa.std(0) + 1e-8
            Xs = (Xa - self.x_mu) / self.x_sd
        beta = cfg["dir_beta"]
        Q = Pi.copy()
        for it in range(cfg["em_iters"]):
            # M-step
            Nr = Q.sum(0)
            if self.use_lab:
                self.theta = (Q.T @ onehot + beta * self.p_glob[None, :]) / (Nr[:, None] + beta)
            if self.use_x:
                wsum = Nr[:, None] + 1e-12
                self.eta = (Q.T @ Xs) / wsum
                var = (Q.T @ (Xs ** 2)) / wsum - self.eta ** 2
                self.tau2 = np.maximum(var, cfg["gauss_var_floor"])
            # E-step
            logp = self._log_lik(lab if self.use_lab else None, Xs if self.use_x else None)
            lw = np.log(Pi + 1e-300) + logp
            lw -= lw.max(1, keepdims=True)
            Q = np.exp(lw); Q /= Q.sum(1, keepdims=True)

    def _log_lik(self, lab, Xs):
        out = 0.0
        if self.use_lab:
            out = out + np.log(self.theta[:, lab].T)
        if self.use_x:
            d = Xs[:, None, :] - self.eta[None, :, :]
            out = out - 0.5 * np.sum(d ** 2 / self.tau2[None] + np.log(2 * np.pi * self.tau2[None]), axis=2)
        return out

    # ------------------------------------------------------------------ score
    def score(self, F, labels=None, X=None, parts=False):
        cfg = self.cfg
        out = {}
        P = {} if parts else None
        S = size_feats(F) - self.s0_mu
        PS = S @ self.s0_P
        out["S0"] = np.sum(S * PS, 1)
        Xd = comp_design(F); Y = comp_y(F)
        R = Y - Xd @ self.A_beta
        PR = R @ self.A_P
        Acontrib = R * PR
        out["A"] = Acontrib.sum(1)
        Z, off = stack_nodes(F)
        b = -self.gmm.score_samples(Z)
        out["B"], wB = graph_agg(b, off, cfg["top_frac"])
        if parts:
            P["S0"] = S * PS; P["A"] = Acontrib; P["A_resid"] = R
            P["A_expected"] = np.expm1(Xd @ self.A_beta)
            P["B_node"] = b * wB; P["off"] = off
        if self.has_C and (labels is not None or X is not None):
            Pi = self.gmm.predict_proba(Z)
            lab = None; Xs = None
            if self.use_lab:
                if labels is None:
                    raise ValueError("model was fitted with node labels")
                lab = np.minimum(np.concatenate(labels).astype(np.int64), self.L)
            if self.use_x:
                Xs = (np.concatenate(X, axis=0).astype(np.float64) - self.x_mu) / self.x_sd
            ll = np.log(Pi + 1e-300) + self._log_lik(lab, Xs)
            mx = ll.max(1)
            a = -(mx + np.log(np.exp(ll - mx[:, None]).sum(1)))
            out["C"], wC = graph_agg(a, off, cfg["top_frac"])
            if parts:
                P["C_node"] = a * wC
                P["role"] = Pi.argmax(1)
        if parts:
            return out, P
        return out
