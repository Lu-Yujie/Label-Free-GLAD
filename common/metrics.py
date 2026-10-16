"""Metrics for GLAD (anomaly = positive class, higher score = more anomalous).

``evaluate`` mirrors UB-GOLD GAOOD/metric/metric.py exactly:
  AUROC = sklearn roc_auc_score, AUPRC = sklearn average_precision_score,
  FPR95 = FPR at the first ROC point (sklearn roc_curve) with TPR >= 0.95.
Values are fractions in [0,1] (multiply by 100 for tables).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def evaluate(scores, y):
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y).astype(np.int64).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ in length")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{int((~np.isfinite(s)).sum())} non-finite scores; fix the detector "
                         "instead of letting sklearn rank NaNs")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y must contain both 0 (normal) and 1 (anomaly)")
    fpr, tpr, _ = roc_curve(y, s)
    return {"auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "fpr95": float(fpr[np.where(tpr >= 0.95)[0][0]])}


def summarize(results, ddof=0):
    """Mean and std (np.std, ddof=0 by default) over a list of evaluate() dicts.
    Note: UB-GOLD's CSVs report statistics.variance (sample variance), not std."""
    out = {}
    for k in ("auroc", "auprc", "fpr95"):
        v = np.array([r[k] for r in results], dtype=np.float64)
        out[k + "_mean"] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=ddof)) if len(v) > 1 else 0.0
    out["n_folds"] = len(results)
    return out
