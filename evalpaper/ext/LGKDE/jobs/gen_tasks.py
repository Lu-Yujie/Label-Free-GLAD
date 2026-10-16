"""Generate the LGKDE grid task list (longest first) and RUNTIME.md from measured timings.

Cost model per run = setup + negstruct + epochs x (train s/epoch + eval s/epoch), from
  * timing runs (smoke dir, tags tcpu*/tgpu) and finished real runs (ext/parts), per dataset and device;
  * fallback: node-pair scaling of the vectorized MMD from the login-node benchmark (15 ns per node pair,
    forward+backward, 5 bandwidths, at 4 threads; 6 ns per query pair in evaluation), for datasets without a
    measurement.
Usage: python -B jobs/gen_tasks.py [--device cpu|gpu] [--cores 126]
"""
import argparse
import glob
import json
import os
import os.path as osp

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
PARTS = osp.normpath(osp.join(HERE, "..", "parts"))
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_lgkde/smoke/parts")
DATASETS = ["REDDIT-BINARY", "NCI1", "DHFR", "AIDS", "ENZYMES", "PROTEINS_full", "IMDB-BINARY", "COX2", "BZR", "MUTAG"]
# fold-0 sizes (graphs/nodes): fit nodes, test nodes, hold nodes (K)
SIZES = {"MUTAG": (1.6, 0.7, 0.4), "BZR": (1.8, 3.0, 0.5), "COX2": (2.7, 3.8, 0.7), "DHFR": (12.2, 6.4, 3.0),
         "ENZYMES": (10.1, 4.0, 2.5), "AIDS": (10.4, 5.8, 2.6), "PROTEINS_full": (6.6, 8.5, 1.6),
         "IMDB-BINARY": (6.1, 4.1, 1.5), "NCI1": (44.8, 25.0, 11.2), "REDDIT-BINARY": (140.8, 172.1, 35.2)}
EPOCHS = 100


def model_thread_seconds_per_epoch(ds):
    nf, nt, nh = (x * 1e3 for x in SIZES[ds])
    return 2.5 * nf * nf * 60e-9 + (nt + nh) * nf * 24e-9 + 1.0


def measured():
    """(dataset, device) -> list of (threads, s/epoch train+eval, negstruct s, setup s)"""
    out = {}
    for fn in glob.glob(osp.join(SMOKE, "LGKDE_*_fold0*_t*.json")) + glob.glob(osp.join(PARTS, "LGKDE_*.json")):
        try:
            J = json.load(open(fn))
        except Exception:
            continue
        m = J["meta"]
        if len(m.get("segments", [])) != 1 or not J["traj"]:
            continue
        dev = "gpu" if "cuda" in m.get("device", "") else "cpu"
        ep = max(1, m.get("epochs_this_segment", len(J["traj"])))
        per = (m.get("seconds_train_this_segment", 0) + m.get("seconds_eval_this_segment", 0)) / ep
        setup = m["seconds"] - m.get("seconds_train_this_segment", 0) - m.get("seconds_eval_this_segment", 0)
        out.setdefault((m["dataset"], dev), []).append(dict(threads=m.get("threads", 1), per_epoch=per,
                                                            negstruct=m.get("seconds_negstruct_this_segment", 0.0),
                                                            setup=setup, run=m["run_id"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--out", default=osp.join(HERE, "tasks_grid.txt"))
    a = ap.parse_args()
    M = measured()
    rows = []
    for ds in DATASETS:
        for hp in ("ubgold", "hpfree"):
            for f in range(5):
                rid = f"LGKDE_{ds}_fold{f}" + ("_hpfree" if hp == "hpfree" else "")
                done = osp.exists(osp.join(PARTS, rid + ".json"))
                meas = M.get((ds, a.device))
                if meas:
                    best = max(meas, key=lambda r: r["threads"])
                    th_s = best["per_epoch"] * (best["threads"] if a.device == "cpu" else 1)
                    src = "measured " + best["run"]
                else:
                    th_s = model_thread_seconds_per_epoch(ds)
                    src = "model"
                rows.append(dict(ds=ds, hp=hp, fold=f, rid=rid, done=done, thread_s_epoch=th_s, src=src))
    lines, tot = [], {"gpu_h": 0.0, "cpu_thread_h": 0.0}
    GPU_DS = {"REDDIT-BINARY", "NCI1"}       # measured: GPU 5-10x faster; everything else runs on CPU cores
    GPU_S_EPOCH = {"REDDIT-BINARY": 96.0, "NCI1": 13.0}   # debug <jobid>, A100 shared by 2-3 runs
    items = []
    for r in rows:
        if r["done"]:
            continue
        if a.device == "gpu" and r["ds"] in GPU_DS:
            est = GPU_S_EPOCH[r["ds"]] * EPOCHS / 3600
            tot["gpu_h"] += est
            items.append((0, -est, f"LGKDE {r['ds']} {r['fold']} {r['hp']} threads=2 mem=8   # GPU est {est:.2f} h (shared A100)"))
        else:
            th = 11 if r["ds"] == "REDDIT-BINARY" else 8 if r["ds"] == "NCI1" else 4 if r["ds"] in ("AIDS", "DHFR", "ENZYMES") else 2
            mem = 8 if r["ds"] == "REDDIT-BINARY" else 4
            est = r["thread_s_epoch"] * EPOCHS / th / 3600
            tot["cpu_thread_h"] += est * th
            items.append((1, -est, f"LGKDE {r['ds']} {r['fold']} {r['hp']} cpu threads={th} mem={mem}   # est {est:.2f} h ({r['src']})"))
    lines = [x[2] for x in sorted(items, key=lambda x: (x[0], x[1]))]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {a.out}; GPU {tot['gpu_h']:.1f} run-hours (shared-A100 rate), CPU {tot['cpu_thread_h']:.1f} thread-hours")


if __name__ == "__main__":
    main()
