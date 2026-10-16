"""Run many (method, dataset, fold) tasks of run_rerun.py inside one SLURM allocation.

Each task is a separate process (own log). Tasks whose output JSON already exists are skipped;
unfinished ones checkpoint at the deadline and resume in the next job (GLocalKD restarts).

  python -B run_pool.py --tasks tasks.txt --workers 32 --threads 3 --minutes 52 [--gpus 4]
tasks.txt: one "METHOD DATASET FOLD" per line, longest first.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
PARTS = osp.join(osp.dirname(HERE), "results", "deep_rerun_parts")
LOGS = osp.join(HERE, "logs", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=52.0, help="wall budget for this job")
    ap.add_argument("--gpus", type=int, default=0, help="round-robin CUDA_VISIBLE_DEVICES over this many GPUs")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
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
    print(f"{len(tasks)} tasks, {len(todo)} to run, workers={a.workers}, threads={a.threads}, gpus={a.gpus}", flush=True)
    running = {}
    slot_of = {}
    free_slots = list(range(a.workers))
    status = {}
    while todo or running:
        # launch
        while todo and free_slots and time.time() < deadline - 60:
            t = todo.pop(0)
            m, ds, f, extra = t[0], t[1], t[2], list(t[3])
            slot = free_slots.pop(0)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(a.threads),
                       MKL_NUM_THREADS=str(a.threads), DR_THREADS=str(a.threads))
            if a.gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(slot % a.gpus)
            log = open(osp.join(LOGS, rid(t) + ".log"), "a")
            log.write(f"\n=== start {time.strftime('%H:%M:%S')} job {os.environ.get('SLURM_JOB_ID')} slot {slot}\n")
            log.flush()
            cmd = [a.python, "-B", osp.join(HERE, "run_rerun.py"), "--method", m, "--dataset", ds,
                   "--fold", str(f), "--deadline", str(deadline), "--threads", str(a.threads)] + extra
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[p] = (m, ds, f, log, time.time())
            slot_of[p] = slot
        if not todo and not running:
            break
        if todo and time.time() >= deadline - 60 and not running:
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
