"""Run many LGKDE run_method.py tasks inside one SLURM allocation.

COPY of evalpaper/harness/run_pool.py (itself a copy of m1_selection/run_pool.py). Changes
(marked CHANGE): PARTS = ext/parts (shared E5 parts dir), logs under ext/LGKDE/logs/runs, the child is
run_method.py, and the run id carries the method name (LGKDE_<DS>_fold<k>...).
Original docstring: Tasks share lock/checkpoint files with every other pool, so concurrent jobs never run the
same task twice; unfinished runs checkpoint at the deadline and resume in the next job.

  python -B run_pool.py --tasks tasks.txt --workers 24 --gpus 4 --minutes 2860 --mem_gb 440

tasks.txt: one task per line, longest first:
   METHOD DATASET FOLD [HP] [cpu] [mem=GB] [threads=N] [max_epochs=N tag=T]   (last two: smoke/timing only)
HP in {ubgold, hpfree} (default ubgold); 'cpu' forces CPU even in a GPU job; mem=GB is the task's
estimated peak host RAM (default 8) used to keep the sum of running tasks under --mem_gb.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.normpath(osp.join(HERE, "..", "parts")))   # CHANGE
LOGS = osp.join(HERE, "logs", "runs")


def parse_tasks(path):
    tasks = []
    for line in open(path):
        p = line.split()
        if len(p) < 3 or line.startswith("#"):
            continue
        t = dict(method=p[0], ds=p[1], fold=int(p[2]), hp="ubgold", cpu=False, mem=8.0, threads=None,
                 max_epochs=None, tag="")
        for x in p[3:]:
            if x in ("ubgold", "hpfree"):
                t["hp"] = x
            elif x == "cpu":
                t["cpu"] = True
            elif x.startswith("mem="):
                t["mem"] = float(x[4:])
            elif x.startswith("threads="):
                t["threads"] = int(x[8:])
            elif x.startswith("max_epochs="):
                t["max_epochs"] = int(x[11:])
            elif x.startswith("tag="):
                t["tag"] = x[4:]
        t["rid"] = (f"{t['method']}_{t['ds']}_fold{t['fold']}" + ("_hpfree" if t["hp"] == "hpfree" else "")
                    + (f"_{t['tag']}" if t["tag"] else ""))
        tasks.append(t)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--threads_by_method", default="", help="e.g. 'GLADC=8,GOOD-D=4'")
    ap.add_argument("--minutes", type=float, required=True, help="wall budget for this job")
    ap.add_argument("--gpus", type=int, default=0, help="round-robin CUDA_VISIBLE_DEVICES over this many GPUs")
    ap.add_argument("--mem_gb", type=float, default=200.0, help="host-RAM budget for concurrently running tasks")
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or 0)
    ap.add_argument("--cores", type=int, default=max(0, _cpus - 2),
                    help="if >0: sum of running tasks' threads stays <= cores (default: SLURM CPUs - 2)")
    ap.add_argument("--min_left", type=float, default=10.0, help="do not START a task with fewer minutes left")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    job_mem = os.environ.get("SLURM_MEM_PER_NODE")       # MB; caps the budget if the job was reshaped
    if job_mem:
        a.mem_gb = min(a.mem_gb, int(job_mem) / 1024.0 - 25.0)
    t0 = time.time()
    deadline = t0 + a.minutes * 60 - 240  # leave time to write checkpoints
    os.makedirs(LOGS, exist_ok=True)
    tasks = parse_tasks(a.tasks)
    tmap = dict(kv.split("=") for kv in a.threads_by_method.split(",") if kv)
    # GPUs actually allocated to this job (SLURM sets CUDA_VISIBLE_DEVICES); round-robin over those ids so a
    # job whose GPU count was reduced with scontrol never addresses a GPU it does not own
    vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() != ""]
    gpu_ids = vis[:a.gpus] if (a.gpus > 0 and vis) else [str(i) for i in range(a.gpus)]
    a.gpus = len(gpu_ids)
    todo = [t for t in tasks if not osp.exists(osp.join(PARTS, t["rid"] + ".json"))]
    print(f"{time.strftime('%F %T')} job {os.environ.get('SLURM_JOB_ID')} on {os.uname().nodename}: "
          f"{len(tasks)} tasks, {len(todo)} to run, workers={a.workers}, gpus={a.gpus}, mem_gb={a.mem_gb}", flush=True)
    running = {}
    free_slots = list(range(a.workers))
    failures = {}
    while todo or running:
        while todo and free_slots and time.time() < deadline - a.min_left * 60:
            used = sum(v["t"]["mem"] for v in running.values())
            used_th = sum(v["th"] for v in running.values())
            thr = lambda t: t["threads"] or int(tmap.get(t["method"], a.threads))
            ok = [i for i, t in enumerate(todo) if (used + t["mem"] <= a.mem_gb or not running)
                  and (a.cores <= 0 or used_th + thr(t) <= a.cores or not running)]
            if not ok:
                break
            t = todo.pop(ok[0])
            if osp.exists(osp.join(PARTS, t["rid"] + ".json")):
                continue
            slot = free_slots.pop(0)
            th = t["threads"] or int(tmap.get(t["method"], a.threads))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(th),
                       MKL_NUM_THREADS=str(th), EP_THREADS=str(th))
            if a.gpus > 0 and not t["cpu"]:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[slot % a.gpus]
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = open(osp.join(LOGS, t["rid"] + ".log"), "a")
            log.write(f"\n=== start {time.strftime('%F %T')} job {os.environ.get('SLURM_JOB_ID')} "
                      f"host {os.uname().nodename} slot {slot} cuda='{env['CUDA_VISIBLE_DEVICES']}' threads {th}\n")
            log.flush()
            cmd = [a.python, "-B", osp.join(HERE, "run_method.py"), "--method", t["method"], "--dataset", t["ds"],
                   "--fold", str(t["fold"]), "--hp", t["hp"], "--deadline", str(deadline), "--threads", str(th)]
            if t["max_epochs"] is not None:
                cmd += ["--max_epochs", str(t["max_epochs"]), "--tag", t["tag"] or "smoke"]
            elif t["tag"]:
                cmd += ["--tag", t["tag"]]
            # CHANGE: the child is run_method.py (above)
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[p] = dict(t=t, log=log, ts=time.time(), slot=slot, th=th)
        if not todo and not running:
            break
        if todo and not running and time.time() >= deadline - a.min_left * 60:
            break
        time.sleep(5)
        for p in list(running):
            rc = p.poll()
            if rc is None:
                continue
            v = running.pop(p)
            v["log"].close()
            free_slots.append(v["slot"])
            t = v["t"]
            print(f"{time.strftime('%F %T')} {t['rid']} rc={rc} ({time.time()-v['ts']:.0f}s)", flush=True)
            # rc 0 = done, 3 = checkpointed/locked; anything else = crash -> never retried in a loop
            if rc not in (0, 3):
                failures[t["rid"]] = rc
    left = [t["rid"] for t in tasks if not osp.exists(osp.join(PARTS, t["rid"] + ".json"))]
    print(f"{time.strftime('%F %T')} finished in {time.time()-t0:.0f}s; failures={failures}; "
          f"{len(left)} tasks incomplete: {left}", flush=True)


if __name__ == "__main__":
    main()
