"""Build the GLADPro grid task list (longest first) and the runtime/memory table from the debug timing runs.

  python3 -B jobs/gen_grid_tasks.py            (login node; reads $GLAD_DATA_ROOT/ext_gladpro/smoke/parts)

Writes jobs/tasks_grid.txt, RUNTIME.md and runtime_table.csv (in ext/GLADPro/).
Grid: UB9 + MUTAG x 5 folds x hp {ubgold, hpfree} = 100 runs (runs already finished in ext/parts are skipped by
the pool at run time, but stay in the list so the list is complete).
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import glob
import json
import os.path as osp
import time

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_gladpro/smoke/parts")
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "MUTAG"]
# released (ubgold) epochs per dataset (NOTES.md); hpfree = 1000 everywhere
UB_EPOCHS = {"MUTAG": 500, "PROTEINS_full": 500, "IMDB-BINARY": 100}
SETUP_S = 30.0          # imports + UB-GOLD loader (REDDIT-B ~15 s)
CONTENTION = 1.25       # timing runs shared the node with ~30 other tasks; grid node is similarly loaded


def timing():
    T = {}
    for f in glob.glob(osp.join(SMOKE, "GLADPro_*_fold0*_tcpu*.json")):
        J = json.load(open(f))
        m = J["meta"]
        tag = m["run_id"].rsplit("_", 1)[1]
        th = {"tcpu": 4, "tcpu8": 8, "tcpu16": 16}[tag]
        T[(m["dataset"], m["hp"], th)] = dict(s_epoch=m["sec_per_epoch_train"], s_eval=m["sec_per_eval"],
                                               rss_gb=m["memory"]["maxrss_gb"], epochs=m["epochs_this_segment"])
    return T


def main():
    T = timing()
    rows = []
    for DS in DATASETS:
        for hp in ("ubgold", "hpfree"):
            ep = 1000 if hp == "hpfree" else UB_EPOCHS.get(DS, 500)
            t4 = T.get((DS, hp, 4)) or T.get((DS, "hpfree" if hp == "ubgold" else "ubgold", 4))
            if t4 is None:
                rows.append(dict(DS=DS, hp=hp, epochs=ep, est_h=None, threads=4, mem=8, src="missing"))
                continue
            per = t4["s_epoch"] + t4["s_eval"]
            est4 = (SETUP_S + ep * per) * CONTENTION / 3600
            # thread choice: long tasks get more threads if the scaling probe shows a gain
            th, est = 4, est4
            # 16 threads only for the single longest group (REDDIT-B hpfree); 8 at most otherwise, so that all
            # 10 REDDIT-B runs (5 x 16 + 5 x 8 = 120 threads) start together on one 126-core pool
            cap = 16 if (DS, hp) == ("REDDIT-BINARY", "hpfree") else 8
            for th2 in [t for t in (8, 16) if t <= cap]:
                s = T.get((DS, "hpfree", th2))
                if s is not None and est4 > 1.0:
                    sc = (s["s_epoch"] + s["s_eval"]) / (T[(DS, "hpfree", 4)]["s_epoch"] + T[(DS, "hpfree", 4)]["s_eval"])
                    e2 = (SETUP_S + ep * per * sc) * CONTENTION / 3600       # probe speed-up applied to this hp
                    if e2 < 0.8 * est:
                        th, est = th2, e2
            if th == 4 and est4 > 1.5 and (DS, "hpfree", 8) not in T:
                th = 8                       # no probe: modest extra threads for the long tail
            mem = max(4, round(2 * t4["rss_gb"] + 2))
            rows.append(dict(DS=DS, hp=hp, epochs=ep, est_h=est, est4_h=est4, threads=th, mem=mem,
                             s_epoch=t4["s_epoch"], s_eval=t4["s_eval"], rss_gb=t4["rss_gb"], src="measured fold 0"))
    rows.sort(key=lambda r: -(r["est_h"] or 99))
    # reproduction investigation (diagnostic only, NOTES.md Part 2): GLADPro's own main.py on its own splits, one
    # task per split (native_main.py, NATIVE_SPLIT=k). Estimates scale the 4-thread timing by GLADPro's own
    # training-set size (all normals of its 80% split, no held-out carve-out).
    native = [("AIDS", "--epochs,500", 0.35), ("ENZYMES", "--epochs,500", 0.2),
              ("PROTEINS", "--hidden_dim,128,--out_dim,64,--epochs,500", 0.15),
              ("BZR", "--epochs,500", 0.12), ("COX2", "--epochs,500", 0.12)]
    items = [(r["est_h"] or 99, "grid", r) for r in rows] + [(e, "native", (d, a)) for d, a, e in native]
    items.sort(key=lambda x: -x[0])
    with open(osp.join(HERE, "jobs", "tasks_grid.txt"), "w") as f:
        f.write(f"# GLADPro grid, longest first (generated {time.strftime('%F %T')} by jobs/gen_grid_tasks.py); "
                "NATIVE lines = native-split diagnostic (not grid data)\n")
        for est, kind, r in items:
            for k in range(5):
                if kind == "grid":
                    f.write(f"GLADPro {r['DS']} {k} {r['hp']} threads={r['threads']} mem={r['mem']}\n")
                else:
                    f.write(f"NATIVE {r[0]} {k} threads=4 mem=3 args={r[1]}\n")
    import csv
    with open(osp.join(HERE, "runtime_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [k for k in ("est4_h", "s_epoch", "s_eval", "rss_gb")
                                                                  if k not in rows[0]])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tot = sum(5 * (r["est_h"] or 0) * r["threads"] for r in rows)
    longest = max(r["est_h"] or 0 for r in rows)
    lines = [
        "# GLADPro runtime and memory table (E5 grid)", "",
        f"Generated {time.strftime('%F %T %Z')} by `jobs/gen_grid_tasks.py` from the debug timing runs "
        "(fold 0, 10 epochs, tag `tcpu*`, `$GLAD_DATA_ROOT/ext_gladpro/smoke/parts`). "
        "Machine-readable: `runtime_table.csv`.", "",
        "How the numbers were measured and used:",
        "- CPU debug node (AMD EPYC, 128 cores), shared with ~30 concurrent timing and reproduction tasks; "
        "deterministic algorithms on (D8).",
        "- s/epoch = training only; s/eval = one evaluation point = test pass + held-out pass. GLADPro evaluates "
        "after every epoch, so every epoch costs s/epoch + s/eval.",
        f"- Estimate = ({SETUP_S:.0f} s setup + epochs x (s/epoch + s/eval)) x {CONTENTION} (contention), at the "
        "chosen thread count; scaling probes (8/16 threads) on REDDIT-B and NCI1 decide extra threads for long tasks.",
        "- Memory = peak RSS of the timing run; the pool budget uses 2 x RSS + 2 GB.",
        "- Placement: all tasks on CPU (`general`): every task fits in one job (longest run ~3.5 h) with checkpoint/resume every "
        "5 min, and the general queue (~10 h) is much shorter than the gpu queue (~1-3 days).", "",
        "| dataset | hp | epochs | s/epoch (4 thr) | s/eval (4 thr) | peak RSS GB | threads | est h / run | 5 folds core-h |",
        "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["est_h"] is None:
            lines.append(f"| {r['DS']} | {r['hp']} | {r['epochs']} | ? | ? | ? | {r['threads']} | ? | ? |")
            continue
        lines.append(f"| {r['DS']} | {r['hp']} | {r['epochs']} | {r['s_epoch']:.3f} | {r['s_eval']:.3f} | "
                     f"{r['rss_gb']:.1f} | {r['threads']} | {r['est_h']:.2f} | {5 * r['est_h'] * r['threads']:.1f} |")
    lines += ["", f"Total: about {tot:.0f} core-hours for the 100 runs; the longest single run is about {longest:.1f} h "
              "(it starts first). One 128-core general node therefore finishes the grid in roughly the longest-run "
              "time plus the tail.",
              "", "The same job also runs 25 `NATIVE` diagnostic tasks (GLADPro's own main.py on its own splits; "
              "AIDS/ENZYMES/PROTEINS/BZR/COX2 x 5 splits, 4 threads, ~0.1-0.35 h each, ~20 core-hours in total; "
              "not grid data).", ""]
    open(osp.join(HERE, "RUNTIME.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
