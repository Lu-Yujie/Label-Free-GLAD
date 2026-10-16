"""Build ext/MUSE/tasks_grid.txt (longest first) and ext/MUSE/runtime_table.csv from the timing probes.

Probes: fold 0, 3 epochs + 1 evaluation point, 2 threads, tag tlogin2 (login node, 2026-10-13), in
$GLAD_DATA_ROOT/ext_muse/smoke/parts; later probes (tag tcpu, compute node) override when present.
Estimate (minutes, 2 threads) = setup + 200 * s/epoch + 10 * s/eval.
  python3 -B jobs/gen_tasks.py
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import glob
import json
import os.path as osp

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_muse/smoke/parts")
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "MUTAG"]
THREADS = {"REDDIT-BINARY": 4, "NCI1": 4}      # the two longest tasks get 4 threads (short tail)


def probes():
    out = {}
    for tag in ("tlogin2", "tcpu"):
        for f in sorted(glob.glob(osp.join(SMOKE, f"MUSE_*_{tag}.json"))):
            m = json.load(open(f))["meta"]
            se = m["sec_per_epoch_train"]
            ev = m["seconds_eval_this_segment"] / max(1, m["n_eval"])
            setup = m["seconds"] - m["seconds_train_this_segment"] - m["seconds_eval_this_segment"]
            out[(m["dataset"], m["hp"])] = dict(s_epoch=se, s_eval=ev, setup=setup, rss=m["memory"]["maxrss_gb"],
                                                src=f"{tag} ({m['segments'][-1]['host']}, {m['threads']} threads)")
    return out


def main():
    P = probes()
    rows = []
    for ds in DATASETS:
        for hp in ("ubgold", "hpfree"):
            p = P[(ds, hp)]
            est = (p["setup"] + 200 * p["s_epoch"] + 10 * p["s_eval"]) / 60.0
            rows.append(dict(ds=ds, hp=hp, est_min=est, **p))
    rows.sort(key=lambda r: -r["est_min"])
    with open(osp.join(HERE, "tasks_grid.txt"), "w") as f:
        f.write("# MUSE E5 grid: UB9 + MUTAG x 5 folds x {ubgold, hpfree}; longest first (est. minutes at 2 threads)\n")
        for r in rows:
            th = THREADS.get(r["ds"], 2)
            mem = max(4, int(3 * r["rss"] + 2))
            for fold in range(5):
                f.write(f"MUSE {r['ds']} {fold} {r['hp']} threads={th} mem={mem}   # est {r['est_min']:.1f} min\n")
    with open(osp.join(HERE, "runtime_table.csv"), "w") as f:
        f.write("dataset,hp,s_per_epoch,s_per_eval,setup_s,est_min_per_run,maxrss_gb,source\n")
        for r in rows:
            f.write(f"{r['ds']},{r['hp']},{r['s_epoch']:.3f},{r['s_eval']:.1f},{r['setup']:.1f},{r['est_min']:.1f},"
                    f"{r['rss']:.2f},{r['src']}\n")
    tot = sum(r["est_min"] for r in rows) * 5
    print(f"{len(rows) * 5} tasks, sum of estimates {tot:.0f} run-minutes ({tot * 2 / 60:.1f} core-hours at 2 threads)")


if __name__ == "__main__":
    main()
