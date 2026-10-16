"""Run many A33 follow-up tasks inside one SLURM allocation (orientation and x_s ablation).

COPY of evalpaper/version_a96137b/run_pool_a96137b.py (itself a copy of typed/run_pool.py and harness/run_pool.py;
none is imported or modified). Changes are marked CHANGE-A33: task format with an experiment column (a33_common.parse_tasks),
runner per experiment, A12 exclusion of tasks with >= 2 recorded crashes, longest-first order by the task's `est`
(GPU-planned tasks first while a GPU has room). Tasks share lock and checkpoint files with every other A33 pool
(harness acquire_lock), so concurrent jobs never run the same task twice; unfinished runs checkpoint at the deadline and
resume in the next job.

  python -B run_pool_a33.py --tasks jobs/tasks_orient.txt --workers 40 --minutes 700 --gpus 4 [--only_dev cuda]
  python -B run_pool_a33.py --tasks jobs/tasks_orient.txt --count_left [--only_dev cpu]     # chaining

dev=cuda tasks run only in a job with GPUs (skipped otherwise; never on CPU); dev=cpu tasks always run with
CUDA_VISIBLE_DEVICES="" (also in a GPU job, on spare cores). Exit codes of a run: 0 done, 3 checkpointed / locked,
5 device mismatch, else crash (never retried within this job; logs/failed_attempts.tsv; a 2nd crash in a later job
makes the task DIVERGED-candidate and it is no longer scheduled: A12).
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import a33_common as A  # noqa: E402

LOGS = osp.join(HERE, "logs", "runs")
FAILED = osp.join(HERE, "logs", "failed_attempts.tsv")
RUNNER = {"orient": osp.join(HERE, "orientation", "run_orient.py"),
          "abl_ub9": osp.join(HERE, "ablation_xs", "run_ablation.py"),
          "abl_typed": osp.join(HERE, "ablation_xs", "run_ablation.py")}


def command(a, t, th, deadline):   # CHANGE-A33
    cmd = [a.python, "-B", RUNNER[t["exp"]]]
    if t["exp"] != "orient":
        cmd += ["--path", "ub9" if t["exp"] == "abl_ub9" else "typed", "--xs_variant", t["xs"]]
    cmd += ["--method", t["method"], "--dataset", t["unit"], "--fold", str(t["fold"]), "--require_device", t["dev"],
            "--deadline", str(deadline), "--threads", str(th), "--out_root", "smoke" if t["smoke"] else "grid"]
    if t["max_epochs"] is not None:
        cmd += ["--max_epochs", str(t["max_epochs"])]
    if t["tag"]:
        cmd += ["--tag", t["tag"]]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=1.0, help="wall budget for this job")
    ap.add_argument("--mem_gb", type=float, default=200.0, help="host-RAM budget for concurrently running tasks")
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or 0)
    ap.add_argument("--cores", type=int, default=max(0, _cpus - 2),
                    help="if >0: sum of running tasks' threads stays <= cores (default: SLURM CPUs - 2)")
    ap.add_argument("--min_left", type=float, default=5.0, help="do not START a task with fewer minutes left")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs of this job used for dev=cuda tasks")
    ap.add_argument("--per_gpu", type=int, default=6, help="max concurrent tasks per GPU")
    ap.add_argument("--gpu_mem_gb", type=float, default=36.0, help="per-GPU memory budget (sum of gmem)")
    ap.add_argument("--only_dev", choices=["cpu", "cuda"], default=None, help="run only tasks planned on this device")
    ap.add_argument("--python", default=(os.environ.get("GLAD_ENVS", "envs") + "/ubgold-gpu/bin/python"))
    ap.add_argument("--count_left", action="store_true", help="print 'left <n> cuda <n> cpu <n> est_h <x>' and exit")
    a = ap.parse_args()
    tasks = A.parse_tasks(a.tasks)
    if a.only_dev:
        tasks = [t for t in tasks if t["dev"] == a.only_dev]
    if a.count_left:
        left = [t for t in tasks if A.task_open(t)]
        print(f"left {len(left)} cuda {sum(t['dev'] == 'cuda' for t in left)} cpu {sum(t['dev'] == 'cpu' for t in left)}"
              f" est_h {sum(t['est'] for t in left) / 3600:.1f}")
        return
    job_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if job_mem:
        a.mem_gb = min(a.mem_gb, int(job_mem) / 1024.0 - 20.0)
    t0 = time.time()
    deadline = t0 + a.minutes * 60 - 240  # leave time to write checkpoints
    os.makedirs(LOGS, exist_ok=True)
    todo = [t for t in tasks if A.task_open(t)]
    todo.sort(key=lambda t: -t["est"])     # CHANGE-A33: longest first
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

    def gpu_for(t):
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
            ok.sort(key=lambda i: (0 if todo[i]["dev"] == "cuda" else 1, i))   # GPU first; longest first in class
            t = todo.pop(ok[0])
            if not A.task_open(t):
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
            logname = ("SMOKE_" if t["smoke"] else "") + t["key"].replace(":", "__") + ".log"
            log = open(osp.join(LOGS, logname), "a")
            log.write(f"\n=== start {time.strftime('%F %T')} job {job} host {os.uname().nodename} slot {slot} "
                      f"threads {th} dev {t['dev']} cuda='{env['CUDA_VISIBLE_DEVICES']}' stop_at {t['stop_at']}\n")
            log.flush()
            p = subprocess.Popen(command(a, t, th, deadline), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
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
            print(f"{time.strftime('%F %T')} {t['key']} rc={rc} ({time.time()-v['ts']:.0f}s)", flush=True)
            if rc not in (0, 3):   # crash or device mismatch -> recorded, never retried within this job
                failures[t["key"]] = rc
                if not t["smoke"]:
                    with open(FAILED, "a") as f:
                        f.write(f"{time.strftime('%F %T')}\t{job}\t{t['key']}\trc={rc}\n")
    left = [t["key"] for t in tasks if A.task_open(t) and (gpu_ids or t["dev"] != "cuda")]
    print(f"{time.strftime('%F %T')} finished in {time.time()-t0:.0f}s; failures={failures}; "
          f"{len(left)} tasks incomplete: {left}", flush=True)


if __name__ == "__main__":
    main()
