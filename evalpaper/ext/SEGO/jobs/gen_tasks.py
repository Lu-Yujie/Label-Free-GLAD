"""Task lists for the SEGO pools (longest first) and the runtime table.

  python3 -B jobs/gen_tasks.py            # writes jobs/tasks_debug.txt, jobs/tasks_grid.txt,
                                          #        runtime_table.csv, RUNTIME.md (table part)

Per-run estimate (CPU) = setup + num_epoch * s/epoch + n_eval * s/eval, from the timing runs
(fold 0, 20 epochs, tag tcpu, $GLAD_DATA_ROOT/ext_sego/smoke/parts) when they exist,
otherwise from the provisional values below (code reading + login-node BZR smoke).
"""
import glob
import json
import os
import os.path as osp
import time

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
SMOKE_PARTS = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_sego/smoke/parts")
PARTS = osp.join(osp.dirname(HERE), "parts")
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
DATASETS = UB9 + ["MUTAG"]
REPRO = ["NCI1", "IMDB-BINARY", "ENZYMES", "DHFR", "BZR"]       # SEGO paper Table 2 datasets
NUM_EPOCH, N_EVAL = 400, 40
# provisional (minutes per run, peak host GB, threads); replaced by measurements when available
# REDDIT-BINARY: measured in debug job <jobid> by the probes SEGO_REDDIT-BINARY_fold0_tcpu(32) (checkpoints only):
# setup ~7 min, ~52 s/epoch at 16 or 32 threads (memory-bandwidth bound, loaded node), first eval ~60 s,
# VmHWM 81 GB during training -> ~6.5 h per run, 90 GB reservation, 60 threads (2 per node)
PROV = {"REDDIT-BINARY": (390, 90, 60), "NCI1": (15, 8, 4), "AIDS": (8, 4, 2), "PROTEINS_full": (5, 4, 2),
        "DHFR": (3, 3, 2), "ENZYMES": (3, 3, 2), "IMDB-BINARY": (3, 3, 2), "COX2": (2, 3, 2), "BZR": (1.5, 3, 2),
        "MUTAG": (1, 3, 2)}
THREADS = {"REDDIT-BINARY": 60, "NCI1": 4}


def timing(ds, hp):
    """Measured (setup_s, s/epoch, s/eval, peak GB, threads) from the tcpu timing run, else None."""
    rid = f"SEGO_{ds}_fold0" + ("_hpfree" if hp == "hpfree" else "") + "_tcpu"
    f = osp.join(SMOKE_PARTS, rid + ".json")
    if not osp.exists(f):
        return None
    m = json.load(open(f))["meta"]
    ep = m["epochs_this_segment"]
    setup = m["seconds"] - m["seconds_train_this_segment"] - m["seconds_eval_this_segment"]
    return (setup, m["seconds_train_this_segment"] / ep, m["seconds_eval_this_segment"] / max(1, m["n_eval"]),
            m["memory"]["maxrss_gb"], m["threads"])


def estimate(ds, hp):
    t = timing(ds, hp) or timing(ds, "ubgold" if hp == "hpfree" else "hpfree")
    if t is None:
        mins, mem, th = PROV[ds]
        return dict(est_min=mins, mem=mem, threads=th, source="provisional")
    setup, spe, spv, gb, th = t
    est = (setup + NUM_EPOCH * spe + N_EVAL * spv) / 60.0
    if th < 4:
        est *= 1.3        # grid runs share the node with other tasks
    src = "measured" if timing(ds, hp) else "measured(other hp)"
    return dict(est_min=est, mem=max(3.0, round(1.3 * gb + 1, 1)), threads=THREADS.get(ds, th), source=src,
                setup_s=setup, s_epoch=spe, s_eval=spv, peak_gb=gb, timing_threads=th)


def line(ds, fold, hp, e, extra=""):
    return f"SEGO {ds} {fold} {hp} mem={e['mem']} threads={e['threads']}{extra}"


def main():
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for ds in DATASETS:
        for hp in ("ubgold", "hpfree"):
            rows.append(dict(dataset=ds, hp=hp, **estimate(ds, hp)))
    est = {(r["dataset"], r["hp"]): r for r in rows}
    grid = sorted([(ds, f, hp) for ds in DATASETS for f in range(5) for hp in ("ubgold", "hpfree")],
                  key=lambda t: (-est[(t[0], t[2])]["est_min"], t[0], t[2], t[1]))
    with open(osp.join(HERE, "jobs", "tasks_grid.txt"), "w") as f:
        f.write(f"# SEGO grid ({stamp}, gen_tasks.py): 10 datasets x 5 folds x {{ubgold,hpfree}}, longest first\n")
        for ds, fo, hp in grid:
            f.write(line(ds, fo, hp, est[(ds, hp)]) + "\n")
    # debug: (T) timing fold 0, 20 epochs; (R) reproduction = real ubgold runs on the Table-2 datasets;
    # (F) fill = every other non-REDDIT grid run, longest first (REDDIT real runs go to the grid job)
    dbg = ["# (T) timing runs, fold 0, 20 epochs (2 evaluation points), smoke dir via tag",
           "SEGO REDDIT-BINARY 0 ubgold mem=100 threads=16 max_epochs=20 tag=tcpu",
           "SEGO REDDIT-BINARY 0 ubgold mem=100 threads=32 max_epochs=20 tag=tcpu32",
           "SEGO NCI1 0 hpfree mem=24 threads=4 max_epochs=20 tag=tcpu",
           "SEGO NCI1 0 ubgold mem=8 threads=4 max_epochs=20 tag=tcpu"]
    for ds in DATASETS:
        if ds in ("REDDIT-BINARY", "NCI1"):
            continue
        for hp in ("ubgold", "hpfree"):
            dbg.append(f"SEGO {ds} 0 {hp} mem=4 threads=2 max_epochs=20 tag=tcpu")
    dbg.append("# (R) reproduction: real runs, released-modal ('ubgold') setting, SEGO paper Table-2 datasets")
    rep = [(ds, f, "ubgold") for ds in REPRO for f in range(5)]
    for ds, fo, hp in rep:
        dbg.append(line(ds, fo, hp, est[(ds, hp)]))
    dbg.append("# (F) fill: remaining non-REDDIT grid runs, longest first")
    for ds, fo, hp in grid:
        if ds != "REDDIT-BINARY" and (ds, fo, hp) not in rep:
            dbg.append(line(ds, fo, hp, est[(ds, hp)]))
    with open(osp.join(HERE, "jobs", "tasks_debug.txt"), "w") as f:
        f.write(f"# SEGO debug pool ({stamp}, gen_tasks.py)\n" + "\n".join(dbg) + "\n")
    import csv
    keys = ["dataset", "hp", "est_min", "mem", "threads", "source", "setup_s", "s_epoch", "s_eval", "peak_gb",
            "timing_threads"]
    with open(osp.join(HERE, "runtime_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in keys})
    done = len([1 for ds, fo, hp in grid if osp.exists(osp.join(
        PARTS, f"SEGO_{ds}_fold{fo}" + ("_hpfree" if hp == "hpfree" else "") + ".json"))])
    tot = sum(est[(ds, hp)]["est_min"] for ds, fo, hp in grid)
    print(f"{stamp}: grid {len(grid)} tasks ({done} done), est total {tot/60:.1f} CPU-task-hours; "
          f"debug list {sum(1 for l in dbg if not l.startswith('#'))} tasks")
    for r in rows:
        print(f"  {r['dataset']:14s} {r['hp']:7s} est {r['est_min']:7.1f} min  mem {r['mem']:6} GB  thr {r['threads']:2}  {r['source']}")


if __name__ == "__main__":
    main()
