"""Runtime / memory table for the SUL/GUL grid from the finished grid parts (their meta records per-segment timing).

  python3 -B jobs/runtime_table.py      (login node) -> RUNTIME.md, runtime_table.csv (in ext/SUL-GUL/)
"""
import glob
import json
import os.path as osp
import time

import numpy as np

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
PARTS = osp.join(osp.dirname(HERE), "parts")
DATASETS = ["REDDIT-BINARY", "NCI1", "AIDS", "DHFR", "PROTEINS_full", "IMDB-BINARY", "ENZYMES", "COX2", "BZR", "MUTAG"]


def main():
    R = {}
    for f in glob.glob(osp.join(PARTS, "SUL-GUL_*.json")):
        m = json.load(open(f))["meta"]
        if m.get("max_epochs_override") is not None or m["run_id"] != (
                f"SUL-GUL_{m['dataset']}_fold{m['fold']}" + ("_hpfree" if m["hp"] == "hpfree" else "")):
            continue                                   # smoke / tagged runs are not grid runs
        R.setdefault((m["dataset"], m["hp"]), []).append(m)
    rows = []
    for DS in DATASETS:
        for hp in ("ubgold", "hpfree"):
            M = R.get((DS, hp), [])
            if not M:
                continue
            one_seg = [m for m in M if len(m["segments"]) == 1]
            se = np.mean([m["sec_per_epoch_train"] for m in one_seg]) if one_seg else float("nan")
            sv = np.mean([m["sec_per_eval"] for m in one_seg]) if one_seg else float("nan")
            wall = [m["seconds"] for m in one_seg]
            rows.append(dict(dataset=DS, hp=hp, runs=len(M), resumed=len(M) - len(one_seg), epochs=M[0]["hparams"]["epochs"],
                             threads=M[0]["threads"], s_epoch=se, s_eval=sv,
                             wall_min_mean=np.mean(wall) / 60 if wall else float("nan"),
                             wall_min_max=np.max(wall) / 60 if wall else float("nan"),
                             rss_gb_max=max(m["memory"]["maxrss_gb"] for m in M),
                             core_h=sum(m["seconds"] * m["threads"] for m in M) / 3600,
                             n_fit=M[0]["n_fit"], n_test=M[0]["n_test"], topk_eff=M[0]["hparams"]["topk_effective"],
                             jobs=",".join(sorted({str(s["job"]) for m in M for s in m["segments"]}))))
    import csv
    with open(osp.join(HERE, "runtime_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = [f"# SUL/GUL runtime and memory table (E4 grid)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `jobs/runtime_table.py` from the finished grid parts "
         f"(`ext/parts/SUL-GUL_*.json`, meta timing of single-segment runs). Machine-readable: `runtime_table.csv`.", "",
         "- Device: CPU debug node (AMD EPYC, 128 cores), all tasks of the grid and the native diagnostic running "
         "concurrently in one pool (so times include node contention); deterministic algorithms on (D8).",
         "- s/epoch = training only; s/eval = test pass + held-out pass (+ secondary held-out objective). GUL evaluates "
         "after every epoch, so a run costs epochs x (s/epoch + s/eval) + ~10-20 s setup (REDDIT-B loader ~18 s).",
         "- Placement: CPU only. Small graphs and a small GIN (hidden 128, 2 layers): every run takes minutes, so the "
         "whole grid fits in one 55-min debug job; no GPU and no regular-partition job is needed.", "",
         "| dataset | hp | runs | epochs | threads | s/epoch | s/eval | wall min (mean / max) | peak RSS GB | core-h (5 runs) | n_fit | topk eff |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['hp']} | {r['runs']} | {r['epochs']} | {r['threads']} | {r['s_epoch']:.3f} | "
                 f"{r['s_eval']:.3f} | {r['wall_min_mean']:.1f} / {r['wall_min_max']:.1f} | {r['rss_gb_max']:.2f} | "
                 f"{r['core_h']:.2f} | {r['n_fit']} | {r['topk_eff']} |")
    tot = sum(r["core_h"] for r in rows)
    L += ["", f"Total: {tot:.1f} core-hours for {sum(r['runs'] for r in rows)} runs; longest single run "
              f"{max(r['wall_min_max'] for r in rows):.1f} min."]
    open(osp.join(HERE, "RUNTIME.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
