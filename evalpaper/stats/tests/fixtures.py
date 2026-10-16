"""Synthetic fixtures for the glad_stats v2 unit tests. Every number here is invented; no real result is read."""
import importlib.util
import json
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = osp.dirname(osp.abspath(__file__))
STATS = osp.dirname(HERE)
EVALPAPER = osp.dirname(STATS)


def load_gs():
    spec = importlib.util.spec_from_file_location("glad_stats_under_test", osp.join(STATS, "glad_stats.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load_gs()
UB9 = G.UB9
HPFREE_KEYS = dict(batch_size=32, batch_size_test=32, lr=1e-4, hidden_dim=16, num_layer=5, alpha=0, num_cluster=2,
                   rw_dim=16, dg_dim=16, dropout=0.3)


def write_part(d, method, ds, fold, hp, auroc, hold, *, epochs=None, nonfinite=None, test_scores=None, n_test=40,
               n_anom=8, eval_freq=1, num_epoch=None, hparams=None, traj_extra=None, meta_extra=None, train_loss=None,
               run_id=None, n_distinct=None, seed=0, write_npz=True, npz_extra=None):
    """One run JSON (+ scores npz) in the union of the formats read by harness/collect.py and the adapters."""
    os.makedirs(d, exist_ok=True)
    n = len(auroc)
    epochs = list(epochs or range(1, n + 1))
    num_epoch = num_epoch or epochs[-1]
    nonfinite = list(nonfinite or [0] * n)
    run_id = run_id or f"{method}_{ds}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "")
    rng = np.random.default_rng(seed + 1000 * fold + 7 * len(ds) + (1 if hp == "hpfree" else 0))
    traj = []
    for i in range(n):
        a = auroc[i]
        r = dict(epoch=epochs[i], auroc=a, auprc=(a if a == a else float("nan")),
                 fpr95=(1 - a if a == a else float("nan")), holdout_score=hold[i], holdout_loss=hold[i],
                 holdout_obj=hold[i], train_loss=0.5 / (i + 1), nonfinite=int(nonfinite[i]), t=1.0)
        for k, v in (traj_extra or {}).items():
            r[k] = v[i] if isinstance(v, (list, tuple, np.ndarray)) else v
        traj.append(r)
    npz = osp.join(d, "scores", run_id + ".npz")
    if write_npz:
        os.makedirs(osp.dirname(npz), exist_ok=True)
        if test_scores is None:
            T = rng.normal(size=(n, n_test))
            if n_distinct is not None:
                T = np.round(rng.integers(0, n_distinct, size=(n, n_test)).astype(float), 3)
            for i in range(n):
                if nonfinite[i]:
                    T[i, 0] = np.nan
        else:
            T = np.asarray(test_scores, float)
        np.savez(npz, epochs=np.asarray(epochs), test=T, hold=rng.normal(size=(n, 10)),
                 y_test=(np.arange(n_test) < n_anom).astype(int), num_epoch=num_epoch, eval_freq=eval_freq,
                 **(npz_extra or {}))
    h = dict(HPFREE_KEYS, eval_freq=eval_freq, num_epoch=num_epoch, topk_effective=96)
    h.update(hparams or {})
    meta = dict(method=method, dataset=ds, fold=fold, hp=hp, run_id=run_id, hparams=h, n_test=n_test,
                n_test_anomalous=n_anom, scores_npz=npz if write_npz else None, max_epochs_override=None,
                device="cpu", device_all="cpu", segments=[dict(device="cpu")], threads=1, variant="GUL",
                ubgold_sh_line=f"python benchmark/mymain.py -exp_type ad -DS {ds} -model {method}",
                hp_meaning="synthetic", completed="2026-10-13 00:00:00")
    meta.update(meta_extra or {})
    J = dict(meta=meta, traj=traj, train_loss=train_loss if train_loss is not None else [0.5 / (i + 1) for i in range(n)])
    with open(osp.join(d, run_id + ".json"), "w") as fh:
        json.dump(J, fh)
    return osp.join(d, run_id + ".json")


def shaped(base, gap_vn, gap_last, n=10, vn_at=2, or_at=5):
    """AUROC (fraction) and held-out trajectory with ORACLE - VAL-NORMAL = gap_vn and ORACLE - LAST = gap_last
    (in AUROC points). VAL-NORMAL picks index vn_at; ORACLE (max) is at or_at; LAST is the final point."""
    a = np.full(n, base - 0.02)
    a[vn_at] = base
    a[or_at] = base + gap_vn / 100
    a[-1] = base + (gap_vn - gap_last) / 100
    h = np.linspace(1.0, 2.0, n)
    h[vn_at] = 0.1
    return a.tolist(), h.tolist()


FROZEN_EXTRA = {"GraphDE": ["auroc_ubgold", "auprc_ubgold", "fpr95_ubgold", "holdout_score_ubgold"],
                "CVTGAD": ["auroc_sel", "holdout_score_norm"]}


def frozen_part(d, method, ds, fold, hp, base, gap_vn, gap_last, **kw):
    a, h = shaped(base, gap_vn, gap_last)
    extra = {}
    for k in FROZEN_EXTRA.get(method, []):
        extra[k] = h if "holdout" in k else a
    return write_part(d, method, ds, fold, hp, a, h, traj_extra=extra, **kw)


def write_eval_folds(parts_dir, res_dir):
    """eval_folds.csv(+extra) from synthetic harness parts with the harness's own run_rows (collect.py logic)."""
    spec = importlib.util.spec_from_file_location("_hc_fixture", osp.join(EVALPAPER, "harness", "collect.py"))
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    rows, extra = [], []
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith(".json"):
            J = json.load(open(osp.join(parts_dir, fn)))
            r, e = hc.run_rows(J, "evalpaper_harness")
            rows += r
            extra += e
    os.makedirs(res_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(osp.join(res_dir, "eval_folds.csv"), index=False)
    pd.DataFrame(extra).to_csv(osp.join(res_dir, "eval_folds_extra.csv"), index=False)


def kernels_csv(res_dir, base=0.6):
    rows = []
    for m in G.KERNELS:
        for ds in UB9:
            for f in range(5):
                for proto, b in (("tuned", base + 0.03), ("default", base)):
                    rows.append(dict(method=m, dataset=ds, fold=f, protocol=proto, auroc=b, auprc=b, fpr95=1 - b,
                                     epoch_selected=0, device="cpu", hp_config="{}", orientation="native"))
                    rows.append(dict(method=m + "@testfit", dataset=ds, fold=f, protocol=proto, auroc=0.99, auprc=0.99,
                                     fpr95=0.01, epoch_selected=0, device="cpu", hp_config="{}", orientation="native"))
    pd.DataFrame(rows).to_csv(osp.join(res_dir, "eval_kernels_folds.csv"), index=False)


def grail0_csv(res_dir):
    rows = [dict(dataset=ds, fold=f, method=m, auroc=0.55 + 0.01 * i, auprc=0.5, fpr95=0.9)
            for i, m in enumerate(G.REF_MAP) for ds in UB9 for f in range(5)]
    pd.DataFrame(rows).to_csv(osp.join(res_dir, "grail0_folds.csv"), index=False)


def typed_frame(deep_vn, ctrl, methods, ms_variants=("matched",), n_anom=200, gx=True, ms=True, seeds=range(5)):
    """deep_vn: {(method, gen, type, variant): VAL-NORMAL AUROC (percent)}, default 55 (blind);
    ctrl: {(gen, type, variant): (gc, size, elp)}."""
    rows = []
    gens = ([("GX", t, "matched") for t in G.GX_TYPES] if gx else []) + \
           ([("MS", t, v) for t in G.MS_TYPES for v in ms_variants] if ms else [])
    for g, t, v in gens:
        gc, sz, el = ctrl.get((g, t, v), (95.0, 50.0, 60.0))
        for s in seeds:
            for name, val in (("GRAIL0:ctrl_gc_lw", gc), ("GRAIL0:ctrl_size_lw", sz), ("POSTHOC:ctrl_edgelab_lw", el)):
                rows.append(dict(method=name, dataset=g, anomaly_type=t, size_variant=v, feature_variant="F2" if g == "GX" else "MS",
                                 seed=s, criterion="training-free", auroc=val / 100, n_anomalies=n_anom))
            for m in methods:
                val = deep_vn.get((m, g, t, v), 55.0)
                for crit in ("oracle", "last", "valnormal"):
                    rows.append(dict(method=m, dataset=g, anomaly_type=t, size_variant=v,
                                     feature_variant="F2" if g == "GX" else "MS", seed=s, criterion=crit,
                                     auroc=(val + (5 if crit == "oracle" else 0)) / 100, n_anomalies=n_anom))
    return pd.DataFrame(rows)


def manifest(path, counts=None, stale_status=False):
    """MolSwap manifest in the A9.1 format; stale_status writes the superseded 150-floor status strings."""
    counts = counts or {}
    st = {}
    for t in G.MS_TYPES:
        st[t] = {}
        for v in ("matched", "sizechg"):
            c = counts.get((t, v), [200] * 5)
            st[t][v] = dict(counts_per_seed=c, min=min(c))
            if stale_status:
                st[t][v]["status"] = "FEASIBLE" if min(c) >= 150 else "INFEASIBLE"
    with open(path, "w") as fh:
        json.dump(dict(generator="synthetic", status=st), fh)
    return path
