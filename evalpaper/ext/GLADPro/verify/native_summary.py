"""Summarise native-split diagnostic logs (native_main.py): GLADPro's own rule (argmax over epochs of the mean AUC
over its 5 splits; main.py:201-211) and the last-epoch mean, per dataset.
  python3 -B verify/native_summary.py logs/native_*.log
Writes results/gladpro_native_diag.csv. Diagnostic only (GLADPro's own splits and anomaly convention).
"""
import collections
import csv
import os.path as osp
import re
import sys

import numpy as np

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
pat = re.compile(r"^Fold_idx:(\d+), Epoch: (\d+), auc: ([0-9.eE+-]+|nan)")
D = collections.defaultdict(dict)   # (DS) -> {split: {epoch: auc}}
for f in sys.argv[1:]:
    # debug-job logs: native_<DS>[_s<k>]_<jobid>.log; grid-pool logs: GLADPro-native_<DS>_s<k>.log
    ds = re.match(r"(?:GLADPro-)?native_(.+?)(?:_s\d)?(?:_\d+)?\.log$", osp.basename(f)).group(1)
    for line in open(f):
        m = pat.match(line)
        if m:
            D[ds].setdefault(int(m.group(1)), {})[int(m.group(2))] = float(m.group(3))
rows = []
for ds, S in sorted(D.items()):
    E = min(len(v) for v in S.values())
    A = np.array([[S[k][e] for e in range(1, E + 1)] for k in sorted(S)])
    idx = int(np.argmax(A.mean(0)))
    rows.append(dict(dataset=ds, n_splits=len(S), epochs=E, own_rule_auc=round(100 * A[:, idx].mean(), 2),
                     own_rule_std=round(100 * A[:, idx].std(), 2), own_rule_epoch=idx + 1,
                     last_auc=round(100 * A[:, -1].mean(), 2), last_std=round(100 * A[:, -1].std(), 2)))
    print(rows[-1])
with open(osp.join(HERE, "results", "gladpro_native_diag.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
