"""Run many (method, dataset, fold) tasks of run_m1.py inside one SLURM allocation.
(M1 copy of deep_rerun/run_pool.py; only the script name and output paths differ.)

Each task is a separate process (own log). Tasks whose output JSON already exists are skipped;
unfinished ones checkpoint at the deadline and resume in the next job (GLocalKD restarts).

  python -B run_pool.py --tasks tasks.txt --workers 32 --threads 3 --minutes 52 [--gpus 4]
tasks.txt: one "METHOD DATASET FOLD" per line, longest first.
"""
import os.path as _osp, sys as _sys  # M1: this directory holds select.py, which would shadow the
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))  # stdlib module `select` (used by
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # subprocess): drop it from sys.path

import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("M1_PARTS", osp.join(HERE, "parts"))
LOGS = osp.join(HERE, "logs", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=52.0, help="wall budget for this job")
    ap.add_argument("--gpus", type=int, default=0, help="round-robin CUDA_VISIBLE_DEVICES over this many GPUs")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--auto_threads", type=int, default=0,
                    help="M1: if >0, workers = min(workers, #todo) and threads = max(threads, auto_threads // workers)")
    ap.add_argument("--threads_by_method", default="",
                    help="M1: e.g. 'GLocalKD=1,SIGNET=4' overrides --threads per method")
    ap.add_argument("--max_heavy", type=int, default=-1,
                    help="M1: at most this many GLocalKD REDDIT-BINARY runs at once (~140 GB host RAM each); "
                         "default 3 on a GPU node (480 GB), 1 otherwise")
    ap.add_argument("--gl_min_left", type=float, default=0.0,
                    help="M1: do not START a (non-resumable) GLocalKD task with fewer minutes left")
    a = ap.parse_args()
    if a.max_heavy < 0:
        a.max_heavy = 3 if a.gpus > 0 else 1
    t0 = time.time()
    deadline = t0 + a.minutes * 60 - 150  # leave time to write checkpoints
    os.makedirs(LOGS, exist_ok=True)
    tasks = []
    for line in open(a.tasks):
        p = line.split()
        if len(p) >= 3 and not line.startswith("#"):
            extra = p[3:]  # e.g. --holdout_frac 0 --tag full
            tag = extra[extra.index("--tag") + 1] if "--tag" in extra else ""
            tasks.append((p[0], p[1], int(p[2]), tuple(extra), tag))
    rid = lambda t: f"{t[0]}_{t[1]}_fold{t[2]}" + (f"_{t[4]}" if t[4] else "")
    todo = [t for t in tasks if not osp.exists(osp.join(PARTS, rid(t) + ".json"))]
    if a.auto_threads > 0 and todo:  # M1: fewer, fatter workers when few (long) tasks remain
        a.workers = max(1, min(a.workers, len(todo)))
        a.threads = max(a.threads, a.auto_threads // a.workers)
    print(f"{len(tasks)} tasks, {len(todo)} to run, workers={a.workers}, threads={a.threads}, gpus={a.gpus}", flush=True)
    running = {}
    slot_of = {}
    free_slots = list(range(a.workers))
    status = {}
    while todo or running:
        # launch
        while todo and free_slots and time.time() < deadline - 60:
            left_min = (deadline - time.time()) / 60.0
            heavy = lambda t: t[0] == "GLocalKD" and t[1] == "REDDIT-BINARY"
            n_heavy = sum(1 for v in running.values() if v[0] == "GLocalKD" and v[1] == "REDDIT-BINARY")
            gl_min = (lambda t: min(a.gl_min_left, 25.0) if (a.gpus > 0 and heavy(t)) else a.gl_min_left)  # GPU: ~15-21 min/run
            ok = [i for i, t in enumerate(todo) if (t[0] != "GLocalKD" or left_min >= gl_min(t))
                  and not (heavy(t) and n_heavy >= a.max_heavy)]
            if not ok:
                break
            t = todo.pop(ok[0])
            m, ds, f, extra = t[0], t[1], t[2], list(t[3])
            slot = free_slots.pop(0)
            tmap = dict(kv.split("=") for kv in a.threads_by_method.split(",") if kv)
            th = int(tmap.get(m, a.threads))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(th),
                       MKL_NUM_THREADS=str(th), DR_THREADS=str(th))
            if a.gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(slot % a.gpus)
            log = open(osp.join(LOGS, rid(t) + ".log"), "a")
            log.write(f"\n=== start {time.strftime('%H:%M:%S')} job {os.environ.get('SLURM_JOB_ID')} slot {slot}\n")
            log.flush()
            cmd = [a.python, "-B", osp.join(HERE, "run_m1.py"), "--method", m, "--dataset", ds,
                   "--fold", str(f), "--deadline", str(deadline), "--threads", str(th)] + extra
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[p] = (m, ds, f, log, time.time())
            slot_of[p] = slot
        if not todo and not running:
            break
        if todo and not running and (time.time() >= deadline - 60 or
                                     all(t[0] == "GLocalKD" for t in todo) and (deadline - time.time()) / 60.0 < a.gl_min_left):
            break
        time.sleep(5)
        for p in list(running):
            rc = p.poll()
            if rc is None:
                continue
            m, ds, f, log, ts = running.pop(p)
            log.close()
            free_slots.append(slot_of.pop(p))
            status[(m, ds, f)] = rc
            print(f"{time.strftime('%H:%M:%S')} {m} {ds} fold{f} rc={rc} ({time.time()-ts:.0f}s)", flush=True)
    left = [rid(t) for t in tasks if not osp.exists(osp.join(PARTS, rid(t) + ".json"))]
    print(f"finished in {time.time()-t0:.0f}s; {len(left)} tasks still incomplete: {left}", flush=True)


if __name__ == "__main__":
    main()
