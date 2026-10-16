"""Own-rule ORACLE (main.py:165-170) and LAST for the reproduction-diagnostic runs (tag 'orig': 100% training normals,
seeds trial+1) next to the grid runs (80% normals, seeds 3407+i) and the paper. Login node:
    PYTHONDONTWRITEBYTECODE=1 python3 -B evalpaper/ext/GLADMamba/tests/diag_repro.py
"""
import glob, json, os.path as osp
import numpy as np
PARTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "..", "parts")
PAPER = {"BZR": 77.25, "COX2": 66.38, "AIDS": 99.29, "DHFR": 63.79, "ENZYMES": 68.39, "NCI1": 73.06,
         "IMDB-BINARY": 69.63, "REDDIT-BINARY": 86.12}
def own(runs):
    ef = runs[0]["meta"]["hparams"]["eval_freq"]
    per = {}
    for J in runs:
        for r in J["traj"]:
            if r["epoch"] % ef == 0:
                per.setdefault(r["epoch"], []).append(100 * r["auroc"])
    c = sorted([(np.mean(v), np.std(v), e) for e, v in per.items() if len(v) == len(runs)], key=lambda x: (-x[0], x[1], x[2]))
    last = np.mean([100 * J["traj"][-1]["auroc"] for J in runs])
    return c[0], last
for ds in ["BZR", "COX2"]:
    for label, pat in (("grid (80%, seeds 3407+i)", f"GLADMamba_{ds}_fold[0-4].json"), ("diag orig (100%, seeds i+1)", f"GLADMamba_{ds}_fold[0-4]_orig.json")):
        runs = [json.load(open(f)) for f in sorted(glob.glob(osp.join(PARTS, pat)))]
        if len(runs) < 5:
            print(f"{ds} {label}: {len(runs)}/5 runs"); continue
        (m, s, e), last = own(runs)
        print(f"{ds} {label}: own-rule ORACLE {m:.2f} +- {s:.2f} @ epoch {e}; LAST {last:.2f}; paper {PAPER[ds]} (diff {m-PAPER[ds]:+.2f})")
