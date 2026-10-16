"""Run many run_eval_a96137b.py tasks inside one SLURM allocation (A17 version sensitivity).

COPY of evalpaper/typed/run_pool.py (itself a copy of harness/run_pool.py; neither is imported or modified).
Changes are marked CHANGE-A17: task format with a planned device per task (A17 device plan, NOTES.md §3.4), per-GPU
memory budget, runner path / run ids / directories, A12 failure log. Tasks share lock and checkpoint files with every
other A17 pool (harness acquire_lock), so concurrent jobs never run the same task twice; unfinished runs checkpoint at
the deadline (epoch boundary) and resume in the next job.

  python -B run_pool_a96137b.py --tasks jobs/tasks_all.txt --workers 40 --minutes 1430 --mem_gb 440 --gpus 4

tasks file, one task per line, longest first:
   METHOD DATASET FOLD dev=cpu|cuda [mem=GB] [gmem=GB] [threads=N] [smoke max_epochs=N tag=T] [stop_at=E]
dev=cuda tasks run only in a job with GPUs (skipped otherwise; never on CPU); dev=cpu tasks always run with
CUDA_VISIBLE_DEVICES="" (also in a GPU job, on spare cores). mem = estimated peak host RAM (GB) for the --mem_gb budget,
gmem = estimated peak GPU memory (GB) for the per-GPU --gpu_mem_gb budget. 'smoke' writes under
$GLAD_DATA_ROOT/evalpaper_a96137b/smoke (runner --out_root smoke); stop_at=E sets EP_TEST_STOP_EPOCH=E
(resume tests only). Exit codes of a run: 0 done, 3 checkpointed / locked, 5 device mismatch, else crash (never retried
within this job; listed in logs/failed_attempts.tsv).
"""
import argparse
import os
import os.path as osp
import subprocess
import time

HERE = osp.dirname(osp.abspath(__file__))
DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_a96137b")
GRID_PARTS = osp.join(HERE, "parts")
LOGS = osp.join(HERE, "logs", "runs")
FAILED = osp.join(HERE, "logs", "failed_attempts.tsv")


def parse_tasks(path):   # CHANGE-A17: task format
    tasks = []
    for line in open(path):
        p = line.split()
        if len(p) < 4 or line.startswith("#"):
            continue
        t = dict(method=p[0], ds=p[1], fold=int(p[2]), dev=None, mem=8.0, gmem=4.0, threads=None, max_epochs=None,
                 tag="", smoke=False, stop_at=None)
        for x in p[3:]:
            if x.startswith("#"):
                break
            if x.startswith("dev="):
                t["dev"] = x[4:]
            elif x == "smoke":
                t["smoke"] = True
            elif x.startswith("mem="):
                t["mem"] = float(x[4:])
            elif x.startswith("gmem="):
                t["gmem"] = float(x[5:])
            elif x.startswith("threads="):
                t["threads"] = int(x[8:])
            elif x.startswith("max_epochs="):
                t["max_epochs"] = int(x[11:])
            elif x.startswith("tag="):
                t["tag"] = x[4:]
            elif x.startswith("stop_at="):
                t["stop_at"] = int(x[8:])
        assert t["dev"] in ("cpu", "cuda"), f"task without dev=cpu|cuda: {line.strip()}"
        t["rid"] = f"{t['method']}_{t['ds']}_fold{t['fold']}" + (f"_{t['tag']}" if t["tag"] else "")
        parts = osp.join(DATA_ROOT, "smoke", "parts") if t["smoke"] else GRID_PARTS
        t["out"] = osp.join(parts, t["rid"] + ".json")
        tasks.append(t)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, required=True, help="wall budget for this job")
    ap.add_argument("--mem_gb", type=float, default=200.0, help="host-RAM budget for concurrently running tasks")
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or 0)
    ap.add_argument("--cores", type=int, default=max(0, _cpus - 2),
                    help="if >0: sum of running tasks' threads stays <= cores (default: SLURM CPUs - 2)")
    ap.add_argument("--min_left", type=float, default=5.0, help="do not START a task with fewer minutes left")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs of this job used for dev=cuda tasks")
    ap.add_argument("--per_gpu", type=int, default=4, help="max concurrent tasks per GPU")
    ap.add_argument("--gpu_mem_gb", type=float, default=38.0, help="CHANGE-A17: per-GPU memory budget (sum of gmem)")
    ap.add_argument("--only_dev", choices=["cpu", "cuda"], default=None, help="run only tasks planned on this device")
    ap.add_argument("--python", default=(os.environ.get("GLAD_ENVS", "envs") + "/ubgold-gpu/bin/python"))
    ap.add_argument("--count_left", action="store_true", help="print 'left <n> cuda <n> cpu <n>' and exit (chaining)")
    a = ap.parse_args()
    if a.count_left:
        left = [t for t in parse_tasks(a.tasks) if not osp.exists(t["out"])]
        print(f"left {len(left)} cuda {sum(t['dev'] == 'cuda' for t in left)} cpu {sum(t['dev'] == 'cpu' for t in left)}")
        return
    job_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if job_mem:
        a.mem_gb = min(a.mem_gb, int(job_mem) / 1024.0 - 20.0)
    t0 = time.time()
    deadline = t0 + a.minutes * 60 - 240  # leave time to write checkpoints
    os.makedirs(LOGS, exist_ok=True)
    tasks = parse_tasks(a.tasks)
    if a.only_dev:
        tasks = [t for t in tasks if t["dev"] == a.only_dev]
    todo = [t for t in tasks if not osp.exists(t["out"])]
    job = os.environ.get("SLURM_JOB_ID")
    vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() != ""]
    gpu_ids = vis[:a.gpus] if (a.gpus > 0 and vis) else [str(i) for i in range(a.gpus)]
    gpu_load = {g: 0 for g in gpu_ids}
    gpu_mem = {g: 0.0 for g in gpu_ids}
    if not gpu_ids:
        skipped = [t["rid"] for t in todo if t["dev"] == "cuda"]
        todo = [t for t in todo if t["dev"] != "cuda"]
        if skipped:
            print(f"{len(skipped)} GPU-planned tasks skipped (no GPU in this job)", flush=True)
    print(f"{time.strftime('%F %T')} job {job} on {os.uname().nodename}: {len(tasks)} tasks, {len(todo)} to run, "
          f"workers={a.workers}, cores={a.cores}, mem_gb={a.mem_gb}, gpus={gpu_ids}, gpu_mem_gb={a.gpu_mem_gb}",
          flush=True)
    running = {}
    free_slots = list(range(a.workers))
    failures = {}
    thr = lambda t: t["threads"] or a.threads

    def gpu_for(t):   # CHANGE-A17: a GPU with room for this task's gmem (least loaded first)
        c = [g for g in gpu_ids if gpu_load[g] < a.per_gpu and gpu_mem[g] + t["gmem"] <= a.gpu_mem_gb]
        return min(c, key=lambda g: (gpu_mem[g], gpu_load[g])) if c else None

    while todo or running:
        while todo and free_slots and time.time() < deadline - a.min_left * 60:
            used = sum(v["t"]["mem"] for v in running.values())
            used_th = sum(v["th"] for v in running.values())
            ok = [i for i, t in enumerate(todo) if (used + t["mem"] <= a.mem_gb or not running)
                  and (a.cores <= 0 or used_th + thr(t) <= a.cores or not running)
                  and (t["dev"] == "cpu" or gpu_for(t) is not None)]
            if not ok:
                break
            # GPU-planned tasks first while a GPU has room (typed CHANGE-SCHED); longest-first within each class
            ok.sort(key=lambda i: 0 if todo[i]["dev"] == "cuda" else 1)
            t = todo.pop(ok[0])
            if osp.exists(t["out"]):
                continue
            slot = free_slots.pop(0)
            th = thr(t)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th),
                       OPENBLAS_NUM_THREADS=str(th), EP_THREADS=str(th), CUDA_VISIBLE_DEVICES="")
            gpu = None
            if t["dev"] == "cuda":
                gpu = gpu_for(t)
                gpu_load[gpu] += 1
                gpu_mem[gpu] += t["gmem"]
                env["CUDA_VISIBLE_DEVICES"] = gpu
            if t["stop_at"] is not None:
                env["EP_TEST_STOP_EPOCH"] = str(t["stop_at"])
            else:
                env.pop("EP_TEST_STOP_EPOCH", None)
            log = open(osp.join(LOGS, ("SMOKE_" if t["smoke"] else "") + t["rid"] + ".log"), "a")
            log.write(f"\n=== start {time.strftime('%F %T')} job {job} host {os.uname().nodename} slot {slot} "
                      f"threads {th} dev {t['dev']} cuda='{env['CUDA_VISIBLE_DEVICES']}' stop_at {t['stop_at']}\n")
            log.flush()
            cmd = [a.python, "-B", osp.join(HERE, "run_eval_a96137b.py"), "--method", t["method"], "--dataset",
                   t["ds"], "--fold", str(t["fold"]), "--require_device", t["dev"], "--deadline", str(deadline),
                   "--threads", str(th), "--out_root", "smoke" if t["smoke"] else "grid"]
            if t["max_epochs"] is not None:
                cmd += ["--max_epochs", str(t["max_epochs"])]
            if t["tag"]:
                cmd += ["--tag", t["tag"]]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[p] = dict(t=t, log=log, ts=time.time(), slot=slot, th=th, gpu=gpu)
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
            if v["gpu"] is not None:
                gpu_load[v["gpu"]] -= 1
                gpu_mem[v["gpu"]] -= v["t"]["gmem"]
            t = v["t"]
            print(f"{time.strftime('%F %T')} {t['rid']} rc={rc} ({time.time()-v['ts']:.0f}s)", flush=True)
            if rc not in (0, 3):   # crash or device mismatch -> recorded, never retried within this job
                failures[t["rid"]] = rc
                if not t["smoke"]:
                    with open(FAILED, "a") as f:
                        f.write(f"{time.strftime('%F %T')}\t{job}\t{t['rid']}\trc={rc}\n")
    left = [t["rid"] for t in tasks if not osp.exists(t["out"]) and (gpu_ids or t["dev"] != "cuda")]
    print(f"{time.strftime('%F %T')} finished in {time.time()-t0:.0f}s; failures={failures}; "
          f"{len(left)} tasks incomplete: {left}", flush=True)


if __name__ == "__main__":
    main()
