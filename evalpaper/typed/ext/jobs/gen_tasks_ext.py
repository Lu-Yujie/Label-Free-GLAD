"""Phase B task list for run_pool_ext.py, LONGEST FIRST. Stdlib only (runs anywhere).

Full grid: 9 methods x UNITS_A28 (GX seeds 0-4 x {F2, F1} + MSV2 seeds 0-4) = 135 runs (RUN_NOTES_ext.md §0.1).
Excluded from the list:
  * finished runs (typed/ext/parts/<run_id>.json exists);
  * runs with >= 2 failed attempts (crash records under <TYPED_ROOT>/ext_ckpt/crash/, or pool failures in
    typed/ext/logs/failed_attempts.tsv): never scheduled again (no retry loop); listed as A12 DIVERGED candidates;
  * anything matched by jobs/EXCLUDE.txt (one token per line: a run id, a method, 'GX', 'MSV2', 'F1', '<M>_<GEN>', ...);
  * GLASS units whose GPU prep (features + Qwen3 text embeddings) has not finished: they wait (no crash attempt).
Ordering: estimated single-run wall time (typed/ext/runtime_est_ext.json from the smoke/timing job; defaults below
otherwise), descending.
Plans (--plan):
  cpu       (default) every method on CPU (RUN_NOTES_ext §0.2 devices);
  gpu-prep  only the GLASS prep tasks (one per unit: the GraphDP texts summarise the node features), for a GPU job.
Prints: 'remaining N cpu_threads T est_core_h H longest_h L done D ...' and the excluded / waiting / DIVERGED lists.

  python -B jobs/gen_tasks_ext.py --out logs/tasks_<job>.txt [--plan cpu|gpu-prep] [--generators GX,MSV2] [--methods A,B]
"""
import argparse
import glob
import json
import os
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
EXT = osp.dirname(HERE)
PARTS = os.environ.get("TYPED_EXT_PARTS", osp.join(EXT, "parts"))
TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
CRASH = osp.join(TYPED_ROOT, "ext_ckpt", "crash")
PREP_DONE = osp.join(TYPED_ROOT, "ext_ckpt", "GLASS", "cache", "prep_done")
EST = osp.join(EXT, "runtime_est_ext.json")
EXCLUDE = osp.join(HERE, "EXCLUDE.txt")
FAILED = osp.join(EXT, "logs", "failed_attempts.tsv")

METHODS = ["HC-GLAD", "GLADMamba", "LGKDE", "GLADPro", "SEGO", "GLASS", "SUL-GUL", "MUSE", "ProtoGLAD"]
# (num_epoch, number of evaluation points) of the typed configuration (RUN_NOTES_ext §0.2)
SCHEDULE = {"MUSE": (200, 10), "LGKDE": (100, 100), "GLADPro": (500, 500), "GLADMamba": (400, 80), "SEGO": (400, 40),
            "GLASS": (150, 32), "ProtoGLAD": (1, 1), "HC-GLAD": (2000, 400), "SUL-GUL": (500, 500)}
# defaults before the timing job: seconds per epoch / per evaluation point at `threads`, setup seconds, memory GB
DEFAULTS = {"MUSE": dict(threads=2, sec_per_epoch=0.5, sec_per_eval=4, setup_s=60, mem_gb=3),
            "LGKDE": dict(threads=8, sec_per_epoch=20, sec_per_eval=15, setup_s=90, mem_gb=6),
            "GLADPro": dict(threads=4, sec_per_epoch=2.5, sec_per_eval=2.5, setup_s=60, mem_gb=4),
            "GLADMamba": dict(threads=8, sec_per_epoch=4, sec_per_eval=60, setup_s=90, mem_gb=48),
            "SEGO": dict(threads=4, sec_per_epoch=4, sec_per_eval=8, setup_s=90, mem_gb=6),
            "GLASS": dict(threads=1, sec_per_epoch=8, sec_per_eval=30, setup_s=120, mem_gb=4),
            "ProtoGLAD": dict(threads=2, sec_per_epoch=900, sec_per_eval=0, setup_s=60, mem_gb=12),
            "HC-GLAD": dict(threads=16, sec_per_epoch=8, sec_per_eval=150, setup_s=120, mem_gb=110),
            "SUL-GUL": dict(threads=2, sec_per_epoch=0.6, sec_per_eval=0.6, setup_s=40, mem_gb=3)}
SEEDS = [0, 1, 2, 3, 4]
UNITS = [("GX", s, "F2") for s in SEEDS] + [("GX", s, "F1") for s in SEEDS] + [("MSV2", s, "MS") for s in SEEDS]


def estimates():
    est = json.load(open(EST)) if osp.exists(EST) else {}
    out = {}
    for m in METHODS:
        for g in ("GX", "MSV2"):
            e = dict(DEFAULTS[m])
            e.update(est.get(m, {}).get(g, {}))
            ne, nev = SCHEDULE[m]
            out[(m, g)] = dict(seconds=ne * float(e["sec_per_epoch"]) + nev * float(e["sec_per_eval"]) + float(e["setup_s"]),
                               threads=int(e["threads"]), mem=float(e["mem_gb"]))
    return out


def failed_counts():
    fails = {}
    if osp.exists(FAILED):
        for ln in open(FAILED):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                fails[p[2]] = fails.get(p[2], 0) + 1
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default="cpu", choices=["cpu", "gpu-prep"])
    ap.add_argument("--generators", default="GX,MSV2")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    gens = set(a.generators.split(","))
    meths = [m for m in METHODS if m in set(a.methods.split(","))]
    tmp = a.out + ".tmp"
    if a.plan == "gpu-prep":
        todo = [(g, s, fv) for g, s, fv in UNITS
                if g in gens and not osp.exists(osp.join(PREP_DONE, f"{g}_{fv}_s{s}.json"))]
        with open(tmp, "w") as f:
            f.write("# Phase B GLASS prep (features + Qwen3-Embedding-0.6B text embeddings), GPU\n")
            for g, s, fv in todo:
                f.write(f"PREP GLASS {g} {s} {fv} mem=24 threads=4 gpu\n")
        os.replace(tmp, a.out)
        print(f"remaining {len(todo)} glass_prep_tasks plan gpu-prep")
        return
    excl = set()
    if osp.exists(EXCLUDE):
        excl = {ln.split()[0] for ln in open(EXCLUDE) if ln.strip() and not ln.startswith("#")}
    fails = failed_counts()
    E = estimates()
    tasks, done, excluded, diverged, waiting = [], 0, [], [], []
    for m in meths:
        for g, s, fv in UNITS:
            if g not in gens:
                continue
            rid = f"{m}_{g}_{fv}_s{s}"
            if osp.exists(osp.join(PARTS, rid + ".json")):
                done += 1
                continue
            if excl & {rid, m, g, fv, f"{m}_{g}", f"{m}_{g}_{fv}"}:
                excluded.append(rid)
                continue
            if max(len(glob.glob(osp.join(CRASH, rid + ".attempt*.json"))), fails.get(rid, 0)) >= 2:
                diverged.append(rid)
                continue
            if m == "GLASS" and not osp.exists(osp.join(PREP_DONE, f"{g}_{fv}_s{s}.json")):
                waiting.append(rid)
                continue
            e = E[(m, g)]
            tasks.append((e["seconds"], m, g, s, fv, e))
    tasks.sort(key=lambda t: (-t[0], t[1], t[2], t[4], t[3]))
    with open(tmp, "w") as f:
        f.write(f"# Phase B typed grid, plan {a.plan}, longest first (estimated single-run seconds in the comment)\n")
        for sec, m, g, s, fv, e in tasks:
            f.write(f"{m} {g} {s} {fv} mem={e['mem']:.0f} threads={e['threads']} cpu   # est {sec:.0f}s\n")
    os.replace(tmp, a.out)
    th = sum(t[5]["threads"] for t in tasks)
    core_h = sum(t[0] * t[5]["threads"] for t in tasks) / 3600.0
    longest = (tasks[0][0] / 3600.0) if tasks else 0.0
    print(f"remaining {len(tasks)} cpu_threads {th} est_core_h {core_h:.1f} longest_h {longest:.2f} done {done} "
          f"plan {a.plan} estimates {'measured' if osp.exists(EST) else 'DEFAULTS'}")
    if excluded:
        print(f"excluded by jobs/EXCLUDE.txt: {len(excluded)}")
    if waiting:
        print(f"waiting for the GLASS GPU prep: {len(waiting)}")
    if diverged:
        print(f"failed twice, not scheduled (A12 DIVERGED candidates; check crash records): {' '.join(diverged)}")


if __name__ == "__main__":
    main()
