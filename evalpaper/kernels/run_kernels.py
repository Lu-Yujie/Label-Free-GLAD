"""KernelGLAD grid runner (resumable, one JSON per unit).

A unit = (dataset, fold, kernel in {WL, PK}, fit mode in {train, test}, WL_iter in 3..7).
It computes the kernel once and evaluates every detector configuration of kglad.detector_configs()
(12 iForest configurations from UB-GOLD's search grid + OCSVM with nu = 0.1).

  # one unit
  python -B run_kernels.py unit --ds BZR --fold 0 --kernel WL --mode train --h 5
  # pool inside a SLURM allocation (longest units first, memory-weighted concurrency)
  python -B run_kernels.py pool --workers 100 --minutes 52 --mem_gb 200

Environment: $GLAD_ENVS/ubgold-gpu/bin/python with
PYTHONPATH=$GLAD_ENVS/evalpaper-kernels-site (GraKeL 0.1.10, UB-GOLD's pin).
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, PRELIM)

PARTS = osp.join(HERE, "parts")
LOGS = osp.join(HERE, "logs", "units")
LOCKS = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_kernels/locks")

UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SECONDARY = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP"]
KERNELS = ["WL", "PK"]
MODES = ["train", "test"]
HS = [3, 4, 5, 6, 7]


def uid(ds, fold, kernel, mode, h):
    return f"{ds}__f{fold}__{kernel}__{mode}__h{h}"


def run_unit(ds, fold, kernel, mode, h):
    import kglad
    from kglad_data import load_folds
    from common.metrics import evaluate
    out = osp.join(PARTS, uid(ds, fold, kernel, mode, h) + ".json")
    if osp.exists(out):
        print("exists", out)
        return
    t0 = time.time()
    f = [x for x in load_folds(ds) if x["fold"] == fold][0]
    y = np.asarray(f["y_test"])
    trg = kglad.to_grakel(f["train"]) if mode == "train" else None
    teg = kglad.to_grakel(f["test"])
    seed = 3407 + fold
    K_fit, K_score, st, ksec = kglad.kernel_matrices(kernel, h, trg, teg, mode, seed)
    # off-diagonal mean/std of the fitted kernel matrix without materialising index arrays (memory)
    n_ = K_fit.shape[0]
    n_off = n_ * (n_ - 1)
    dg = np.diagonal(K_fit)
    s1 = float(K_fit.sum() - dg.sum())
    s2 = float(np.einsum("ij,ij->", K_fit, K_fit) - (dg * dg).sum())
    k_mean = s1 / n_off if n_off else None
    k_std = float(np.sqrt(max(s2 / n_off - k_mean ** 2, 0.0))) if n_off else None
    res = []
    scores_default = {}
    for det, cfg in kglad.detector_configs():
        t1 = time.time()
        s = kglad.fit_score(det, cfg, K_fit, K_score, st)
        m = evaluate(s, y)
        r = {"detector": det, **cfg, "auroc": m["auroc"], "auprc": m["auprc"], "fpr95": m["fpr95"],
             "score_std": float(np.std(s)), "seconds": time.time() - t1}
        res.append(r)
        is_def = h == kglad.DEFAULTS["WL_iter"] and (
            det == "OCSVM" or (cfg["IF_n_trees"] == kglad.DEFAULTS["IF_n_trees"]
                               and cfg["IF_sample_ratio"] == kglad.DEFAULTS["IF_sample_ratio"]))
        if is_def:
            scores_default[det] = [float(v) for v in s]
    import grakel, sklearn
    payload = {
        "dataset": ds, "fold": fold, "kernel": kernel, "mode": mode, "WL_iter": h, "seed": seed,
        "n_fit": int(K_fit.shape[0]), "n_test": int(len(y)), "n_test_anomalies": int(y.sum()),
        "kernel_seconds": ksec, "kernel_offdiag_mean": k_mean, "kernel_offdiag_std": k_std,
        "kernel_finite": bool(np.all(np.isfinite(K_fit)) and np.all(np.isfinite(K_score))),
        "results": res, "y_test": [int(v) for v in y], "scores_default": scores_default,
        "orientation": "score = -decision_function (UB-GOLD KernelGLAD)",
        "versions": {"grakel_dist": "0.1.10", "grakel_file": grakel.__file__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__},
        "seconds_total": time.time() - t0, "job": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
    }
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, out)
    print(f"done {uid(ds, fold, kernel, mode, h)} kernel {ksec:.1f}s total {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------------------------
def cost_and_mem(ds, kernel, mode, h):
    """Rough cost (s) and peak memory (GB) from the fold-0 benchmarks (NOTES.md)."""
    base = {"Tox21_p53": 60, "Tox21_MMP": 45, "COLLAB": 40, "REDDIT-BINARY": 30, "NCI1": 15, "DD": 8,
            "AIDS": 5, "PROTEINS_full": 3, "IMDB-BINARY": 3, "DHFR": 2, "ENZYMES": 2, "COX2": 1, "BZR": 1}[ds]
    if kernel == "PK":
        # grakel Propagation fills the kernel pair by pair in Python: ~3 us per pair and iteration
        base = {"Tox21_p53": 500, "Tox21_MMP": 300, "NCI1": 30}.get(ds, base * 3)
    if mode == "test":
        base /= 5 if not ds.startswith("Tox21") else 50
    # measured RSS (job <jobid>): Tox21 PK train ~2 GB while filling the kernel, WL train 6.9 GB peak
    mem = {"Tox21_p53": 7, "Tox21_MMP": 5, "COLLAB": 4, "REDDIT-BINARY": 2.5}.get(ds, 1.0)
    if mode == "test":
        mem = min(mem, 3)
    return base * h / 5.0, mem


def running_jobs():
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        return set(out)
    except Exception:
        return None


def clear_stale_locks():
    """Remove locks whose owning SLURM job is no longer in the queue (or login-node locks)."""
    jobs = running_jobs()
    if jobs is None or not osp.isdir(LOCKS):
        return 0
    n = 0
    for fn in os.listdir(LOCKS):
        p = osp.join(LOCKS, fn)
        try:
            owner = open(p).read().split()[0]
        except Exception:
            owner = "?"
        if owner not in jobs:
            os.remove(p)
            n += 1
    return n


def acquire(u):
    os.makedirs(LOCKS, exist_ok=True)
    p = osp.join(LOCKS, u + ".lock")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"{os.environ.get('SLURM_JOB_ID', 'login')} {os.getpid()} {time.time():.0f}\n".encode())
    os.close(fd)
    return p


HEAVY = {"Tox21_p53", "Tox21_MMP", "COLLAB", "REDDIT-BINARY", "NCI1", "DD", "AIDS"}


def pool(a):
    """Tasks = one subprocess each. Heavy datasets: one WL_iter per task; light datasets: all
    missing WL_iter values of a (dataset, fold, kernel, mode) in one task (saves interpreter
    start-up, ~10-15 s on the shared file system)."""
    t0 = time.time()
    deadline = t0 + a.minutes * 60
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    print(f"cleared {clear_stale_locks()} stale locks", flush=True)
    dss = a.datasets.split(",") if a.datasets else UB9 + SECONDARY
    units = [(ds, f, k, m, h) for ds in dss for f in range(5) for k in KERNELS for m in MODES for h in HS]
    missing = lambda u: not osp.exists(osp.join(PARTS, uid(*u) + ".json"))
    todo = []
    for ds in dss:
        for f in range(5):
            for k in KERNELS:
                for m in MODES:
                    hs = tuple(h for h in HS if missing((ds, f, k, m, h)))
                    if not hs:
                        continue
                    if ds in HEAVY:
                        todo += [(ds, f, k, m, (h,)) for h in hs]
                    else:
                        todo.append((ds, f, k, m, hs))
    cost = lambda t: sum(cost_and_mem(t[0], t[2], t[3], h)[0] for h in t[4])
    mem_of = lambda t: max(cost_and_mem(t[0], t[2], t[3], h)[1] for h in t[4])
    tid = lambda t: f"{t[0]}__f{t[1]}__{t[2]}__{t[3]}__h{'-'.join(map(str, t[4]))}"
    todo.sort(key=lambda t: -cost(t))
    print(f"{len(units)} units, {sum(map(missing, units))} missing, {len(todo)} tasks, workers={a.workers}, "
          f"mem budget {a.mem_gb} GB, est. {sum(map(cost, todo))/3600:.1f} CPU-h", flush=True)
    fails = {}
    running = {}
    mem_used = 0.0
    while todo or running:
        launched = True
        while launched and todo and len(running) < a.workers and time.time() < deadline - 120:
            launched = False
            for i, t in enumerate(todo):
                mem = mem_of(t)
                if mem_used + mem > a.mem_gb and running:
                    continue
                todo.pop(i)
                launched = True
                lock = acquire(tid(t))
                if lock is None:
                    break
                if not any(missing((t[0], t[1], t[2], t[3], h)) for h in t[4]):
                    os.remove(lock)
                    break
                log = open(osp.join(LOGS, tid(t) + ".log"), "a")
                log.write(f"\n=== start {time.strftime('%H:%M:%S')} job {os.environ.get('SLURM_JOB_ID')}\n")
                log.flush()
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1")
                cmd = [sys.executable, "-B", osp.abspath(__file__), "unit", "--ds", t[0], "--fold", str(t[1]),
                       "--kernel", t[2], "--mode", t[3], "--h", ",".join(map(str, t[4]))]
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
                running[p] = (t, log, lock, time.time(), mem)
                mem_used += mem
                break
        if not running and (not todo or time.time() >= deadline - 120):
            break
        time.sleep(2)
        for p in list(running):
            rc = p.poll()
            if rc is None:
                if time.time() > deadline:
                    p.kill()
                continue
            t, log, lock, ts, mem = running.pop(p)
            log.close()
            mem_used -= mem
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
            ok = not any(missing((t[0], t[1], t[2], t[3], h)) for h in t[4])
            print(f"{time.strftime('%H:%M:%S')} {tid(t)} rc={rc} ok={ok} ({time.time()-ts:.0f}s) "
                  f"running={len(running)} todo={len(todo)} mem={mem_used:.0f}GB", flush=True)
            if not ok:
                fails[t] = fails.get(t, 0) + 1
                if fails[t] < 2 and time.time() < deadline - 300:
                    todo.append(t)  # one retry at most (e.g. OOM under contention)
    left = [uid(*u) for u in units if missing(u)]
    print(f"pool finished in {time.time()-t0:.0f}s; {len(left)} units missing", flush=True)
    for x in left[:80]:
        print("  missing", x)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unit")
    u.add_argument("--ds", required=True)
    u.add_argument("--fold", type=int, required=True)
    u.add_argument("--kernel", choices=KERNELS, required=True)
    u.add_argument("--mode", choices=MODES, required=True)
    u.add_argument("--h", required=True, help="WL_iter value(s), comma-separated")
    p = sub.add_parser("pool")
    p.add_argument("--workers", type=int, default=100)
    p.add_argument("--minutes", type=float, default=52)
    p.add_argument("--mem_gb", type=float, default=200)
    p.add_argument("--datasets", default="")
    a = ap.parse_args()
    if a.cmd == "unit":
        for h in [int(v) for v in a.h.split(",")]:
            run_unit(a.ds, a.fold, a.kernel, a.mode, h)
    else:
        pool(a)


if __name__ == "__main__":
    main()
