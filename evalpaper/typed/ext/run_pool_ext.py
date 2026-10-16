"""Run many Phase B typed tasks inside one SLURM allocation (RUN_NOTES_ext.md).

COPY of evalpaper/typed/run_pool.py (itself a copy of the harness pool; not imported, not modified); changes are
marked CHANGE-TYPED-EXT: task format, adapter scripts typed/ext/<M>_typed.py, GLASS prep tasks, pass-through adapter
options, rc 5 (prerequisite missing; not a failure), crash bookkeeping (A12) in typed/ext/logs/failed_attempts.tsv.
Tasks share lock/checkpoint files with every other pool (typed_ext_common.acquire_lock), so concurrent jobs never run
the same task twice; unfinished runs checkpoint at the deadline and resume in the next job.

  python -B run_pool_ext.py --tasks tasks.txt --workers 60 --minutes 700 --mem_gb 220 [--cores 126] [--gpus N]

tasks.txt, one task per line, longest first:
   METHOD GENERATOR SEED FV [mem=GB] [threads=N] [cpu|gpu] [smoke max_epochs=N tag=T] [stop_at=E] [--adapter-option ...]
   PREP GLASS GENERATOR SEED FV [mem=GB] [threads=N] [gpu] [smoke] [--smoke_random_emb|--no_encode]
'smoke' writes under <TYPED_ROOT>/smoke_ext/ (adapter --smoke); stop_at=E sets EP_TEST_STOP_EPOCH=E (resume tests).
Exit codes of a run: 0 done, 3 checkpointed / locked (resumes later), 5 prerequisite missing, anything else = crash
(recorded; never retried within this job).
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
GRID_PARTS = osp.join(HERE, "parts")
LOGS = osp.join(HERE, "logs", "runs")
FAILED = osp.join(HERE, "logs", "failed_attempts.tsv")


def parse_tasks(path):   # CHANGE-TYPED-EXT: task format
    tasks = []
    for line in open(path):
        p = line.split()
        if len(p) < 4 or line.startswith("#"):
            continue
        prep = p[0] == "PREP"
        if prep:
            p = p[1:]
        t = dict(method=p[0], gen=p[1], seed=int(p[2]), fv=p[3], mem=8.0, threads=None,
                 max_epochs=None, tag="", smoke=False, stop_at=None, cpu=False, gpu=False, prep=prep, opts=[])
        for x in p[4:]:
            if x.startswith("#"):
                break
            if x == "cpu":
                t["cpu"] = True
            elif x == "gpu":
                t["gpu"] = True
            elif x == "smoke":
                t["smoke"] = True
            elif x.startswith("mem="):
                t["mem"] = float(x[4:])
            elif x.startswith("threads="):
                t["threads"] = int(x[8:])
            elif x.startswith("max_epochs="):
                t["max_epochs"] = int(x[11:])
            elif x.startswith("tag="):
                t["tag"] = x[4:]
            elif x.startswith("stop_at="):
                t["stop_at"] = int(x[8:])
            elif x.startswith("--"):
                t["opts"].append(x)
        if prep:
            t["rid"] = f"PREP_{t['method']}_{t['gen']}_{t['fv']}_s{t['seed']}" + ("_smoke" if t["smoke"] else "")
            t["out"] = osp.join(TYPED_ROOT, "ext_ckpt", t["method"], "cache", "prep_done",
                                f"{t['gen']}_{t['fv']}_s{t['seed']}.json")
        else:
            t["rid"] = f"{t['method']}_{t['gen']}_{t['fv']}_s{t['seed']}" + (f"_{t['tag']}" if t["tag"] else "")
            parts = osp.join(TYPED_ROOT, "smoke_ext", "parts") if t["smoke"] else os.environ.get("TYPED_EXT_PARTS", GRID_PARTS)
            t["out"] = osp.join(parts, t["rid"] + ".json")
        if prep and t["smoke"]:
            t["out"] = osp.join(TYPED_ROOT, "smoke_ext", "ckpt", t["method"], "cache", "prep_done",
                                f"{t['gen']}_{t['fv']}_s{t['seed']}.json")
        tasks.append(t)
    return tasks


def command(a, t, deadline, th):
    script = osp.join(HERE, f"{t['method']}_typed.py")
    if t["prep"]:
        cmd = [a.python, "-B", script, "--prep", "--generator", t["gen"], "--seed", str(t["seed"]), "--fv", t["fv"],
               "--threads", str(th)]
        if t["smoke"]:
            cmd += ["--smoke"]
        return cmd + t["opts"]
    cmd = [a.python, "-B", script, "--generator", t["gen"], "--seed", str(t["seed"]), "--fv", t["fv"],
           "--deadline", str(deadline), "--threads", str(th)]
    if t["smoke"]:
        cmd += ["--smoke"]
    if t["max_epochs"] is not None:
        cmd += ["--max_epochs", str(t["max_epochs"])]
    if t["tag"]:
        cmd += ["--tag", t["tag"]]
    return cmd + t["opts"]


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
    ap.add_argument("--gpus", type=int, default=0, help="round-robin CUDA_VISIBLE_DEVICES over this many GPUs")
    ap.add_argument("--per_gpu", type=int, default=1, help="max concurrent GPU tasks per GPU")
    ap.add_argument("--python", default=(os.environ.get("GLAD_ENVS", "envs") + "/ubgold-gpu/bin/python"))
    a = ap.parse_args()
    job_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if job_mem:
        a.mem_gb = min(a.mem_gb, int(job_mem) / 1024.0 - 16.0)
    t0 = time.time()
    deadline = t0 + a.minutes * 60 - 300  # leave time to write checkpoints
    os.makedirs(LOGS, exist_ok=True)
    tasks = parse_tasks(a.tasks)
    todo = [t for t in tasks if not osp.exists(t["out"])]
    job = os.environ.get("SLURM_JOB_ID")
    vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() != ""]
    gpu_ids = vis[:a.gpus] if (a.gpus > 0 and vis) else [str(i) for i in range(a.gpus)]
    gpu_load = {g: 0 for g in gpu_ids}
    if not gpu_ids:
        skipped = [t["rid"] for t in todo if t["gpu"]]
        todo = [t for t in todo if not t["gpu"]]
        if skipped:
            print(f"{len(skipped)} GPU-planned tasks skipped (no GPU in this job): {skipped}", flush=True)
    print(f"{time.strftime('%F %T')} job {job} on {os.uname().nodename}: {len(tasks)} tasks, {len(todo)} to run, "
          f"workers={a.workers}, cores={a.cores}, mem_gb={a.mem_gb}, gpus={gpu_ids}", flush=True)
    running = {}
    free_slots = list(range(a.workers))
    failures = {}
    thr = lambda t: t["threads"] or a.threads
    while todo or running:
        while todo and free_slots and time.time() < deadline - a.min_left * 60:
            used = sum(v["t"]["mem"] for v in running.values())
            used_th = sum(v["th"] for v in running.values())
            free_gpu = [g for g in gpu_ids if gpu_load[g] < a.per_gpu]
            busy = {v["t"]["rid"] for v in running.values()}   # CHANGE-TYPED-EXT: resume-test lines share a run id
            ok = [i for i, t in enumerate(todo) if t["rid"] not in busy and (used + t["mem"] <= a.mem_gb or not running)
                  and (a.cores <= 0 or used_th + thr(t) <= a.cores or not running)
                  and (t["cpu"] or not t["gpu"] or not gpu_ids or free_gpu)]
            if not ok:
                break
            t = todo.pop(ok[0])
            if osp.exists(t["out"]):
                continue
            slot = free_slots.pop(0)
            th = thr(t)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th),
                       OPENBLAS_NUM_THREADS=str(th), NUMBA_NUM_THREADS=str(th), TYPED_THREADS=str(th),
                       TQDM_DISABLE="1", CUDA_VISIBLE_DEVICES="")
            gpu = None
            if gpu_ids and t["gpu"] and not t["cpu"]:
                gpu = min(free_gpu, key=lambda g: gpu_load[g])
                gpu_load[gpu] += 1
                env["CUDA_VISIBLE_DEVICES"] = gpu
            if t["stop_at"] is not None:
                env["EP_TEST_STOP_EPOCH"] = str(t["stop_at"])
            else:
                env.pop("EP_TEST_STOP_EPOCH", None)
            log = open(osp.join(LOGS, t["rid"] + ".log"), "a")
            log.write(f"\n=== start {time.strftime('%F %T')} job {job} host {os.uname().nodename} slot {slot} "
                      f"threads {th} cuda='{env['CUDA_VISIBLE_DEVICES']}' stop_at {t['stop_at']}\n")
            log.flush()
            p = subprocess.Popen(command(a, t, deadline, th), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
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
            t = v["t"]
            print(f"{time.strftime('%F %T')} {t['rid']} rc={rc} ({time.time()-v['ts']:.0f}s)", flush=True)
            if rc not in (0, 3, 5):   # crash -> recorded (A12 bookkeeping), never retried within this job
                failures[t["rid"]] = rc
                if not t["smoke"]:
                    with open(FAILED, "a") as f:
                        f.write(f"{time.strftime('%F %T')}\t{job}\t{t['rid']}\trc={rc}\n")
    left = [t["rid"] for t in tasks if not osp.exists(t["out"]) and (gpu_ids or not t["gpu"])]
    print(f"{time.strftime('%F %T')} finished in {time.time()-t0:.0f}s; failures={failures}; "
          f"{len(left)} tasks incomplete: {left}", flush=True)


if __name__ == "__main__":
    main()
