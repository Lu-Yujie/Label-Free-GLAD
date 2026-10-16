"""Summarise the native diagnostic (GLASS's own pipeline, own split, seeds 42-46) -> results/glass_native_diag.csv."""
import glob, json, os, os.path as osp
import numpy as np
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
N = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_glass/native/results")
PAPER = {"MUTAG": 91.42, "AIDS": 99.32, "DHFR": 69.38, "IMDB-BINARY": 77.20}
rows = []
for DS in PAPER:
    f = osp.join(N, "reference", f"reference_scorers_{DS}.json")
    if not osp.exists(f):
        continue
    agg = json.load(open(f))["datasets"][DS]["aggregated"]
    per = json.load(open(f))["datasets"][DS]["per_seed"]
    best = max(agg, key=lambda r: agg[r]["auroc_mean"])
    tr = [json.load(open(p))["per_seed"][0] for p in sorted(glob.glob(osp.join(N, "seed4[2-6]", f"{DS}_gin3_native_mrl_results.json")))]
    rows.append(dict(dataset=DS, paper=PAPER[DS], n_seeds=len(per),
                     own_best24_rule=best, own_best24=round(agg[best]["auroc_mean"], 2),
                     own_best24_std=round(agg[best]["auroc_std"], 2),
                     apriori_rcsm_full_test_cdf=round(agg["rcsm_full_test_cdf"]["auroc_mean"], 2),
                     fallback_fixed_test_raw=round(agg["fixed_test_raw"]["auroc_mean"], 2),
                     fixed_test_cdf=round(agg["fixed_test_cdf"]["auroc_mean"], 2),
                     train_single_best6=round(float(np.mean([r["auroc"] for r in tr])), 2) if tr else None,
                     train_single_methods=",".join(r["best_method"] for r in tr),
                     ckpt_epochs=",".join(str(r["checkpoint_epoch"]) for r in per),
                     diff_best24_vs_paper=round(agg[best]["auroc_mean"] - PAPER[DS], 2)))
os.makedirs(osp.join(HERE, "results"), exist_ok=True)
import csv
with open(osp.join(HERE, "results", "glass_native_diag.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
