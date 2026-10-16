"""Run many (method, dataset, variant, seed) tasks of run_m2.py inside one SLURM allocation.
(COPY of deep_rerun/run_pool.py; changes marked CHANGE-M2: task format, paths, script.)

Each task is a separate process (own log). Tasks whose output JSON already exists are skipped;
unfinished ones checkpoint at the deadline and resume in the next job (GLocalKD restarts).

  python -B run_pool.py --tasks tasks.txt --workers 24 --threads 2 --minutes 54 [--gpus 4]
tasks.txt: one "METHOD DATASET VARIANT SEED [extra args]" per line, longest first.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("M2_PARTS", osp.join(HERE, "parts"))  # CHANGE-M2
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
        if len(p) >= 4 and not line.startswith("#"):  # CHANGE-M2: METHOD DATASET VARIANT SEED [extra]
            extra = p[4:]  # e.g. --max_epochs 8 --tag smoke
            tag = extra[extra.index("--tag") + 1] if "--tag" in extra else ""
            tasks.append((p[0], p[1], p[2], int(p[3]), tuple(extra), tag))
    rid = lambda t: f"{t[0]}_{t[1]}_{t[2]}_s{t[3]}" + (f"_{t[5]}" if t[5] else "")
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
            m, ds, v, f, extra = t[0], t[1], t[2], t[3], list(t[4])
            slot = free_slots.pop(0)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(a.threads),
                       MKL_NUM_THREADS=str(a.threads), M2_THREADS=str(a.threads))
            if a.gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(slot % a.gpus)
            log = open(osp.join(LOGS, rid(t) + ".log"), "a")
            log.write(f"\n=== start {time.strftime('%H:%M:%S')} job {os.environ.get('SLURM_JOB_ID')} slot {slot}\n")
            log.flush()
            cmd = [a.python, "-B", osp.join(HERE, "run_m2.py"), "--method", m, "--dataset", ds,
                   "--variant", v, "--seed", str(f), "--deadline", str(deadline), "--threads", str(a.threads)] + extra
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[p] = (m, f"{ds} {v}", f, log, time.time())
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
            print(f"{time.strftime('%H:%M:%S')} {m} {ds} s{f} rc={rc} ({time.time()-ts:.0f}s)", flush=True)
    left = [rid(t) for t in tasks if not osp.exists(osp.join(PARTS, rid(t) + ".json"))]
    print(f"finished in {time.time()-t0:.0f}s; {len(left)} tasks still incomplete: {left}", flush=True)


if __name__ == "__main__":
    main()
