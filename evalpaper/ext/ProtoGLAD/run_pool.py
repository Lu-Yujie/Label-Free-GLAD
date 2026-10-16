"""Resumable task pool for the ProtoGLAD grid inside one SLURM allocation (or several, sequentially).

  python -B run_pool.py --workers 120 --minutes 52 [--no_native] [--no_diag]

Every task writes its output atomically and holds a lock (harness acquire_lock: a lock whose SLURM job is no
longer RUNNING is stale and taken over), so a later job simply continues: finished outputs are skipped.
Order (longest first inside each phase):
  1. R2 native-protocol searches (sequential processes; long, started first)
  2. grid units (UB-GOLD features), then hpfree parts
  3. per dataset, as soon as its 30 units exist: select -> tuned parts
  4. diagnostic (native features): grid units, select, hpfree / tuned parts
Each task is a subprocess with 1 BLAS/torch thread; logs in ext/ProtoGLAD/logs/tasks/.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
PY = (os.environ.get("GLAD_ENVS", "envs") + "/ubgold-gpu/bin/python")
RM = osp.join(HERE, "run_method.py")
NR = osp.join(HERE, "native_repro.py")
LOGS = osp.join(HERE, "logs", "tasks")
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("pg_rm", RM)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

BASE = {"NCI1": 300, "AIDS": 250, "REDDIT-BINARY": 220, "IMDB-BINARY": 40, "DHFR": 30, "PROTEINS_full": 30,
        "ENZYMES": 30, "COX2": 15, "BZR": 10, "MUTAG": 5}
NATIVE_R2 = [("NCI1", 9000), ("AIDS", 8000), ("PROTEINS_full", 2400), ("DHFR", 1200), ("COX2", 800), ("BZR", 700),
             ("MUTAG", 250)]
DIAG = ["AIDS", "BZR", "COX2", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "ENZYMES"]


class Task:
    def __init__(self, name, argv, cost, done, ready=lambda: True, threads=1, cwd=None):
        self.name, self.argv, self.cost, self.done, self.ready, self.threads = name, argv, cost, done, ready, threads
        self.cwd = cwd
        self.proc = None
        self.t0 = None
        self.attempts = 0


def part_done(DS, fold, hp, features="ubgold"):
    rid = (f"ProtoGLAD_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "")
           + ("_nativefeat" if features == "native" else ""))
    parts = os.environ.get("EP_PARTS_DIAG", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_protoglad/diag_parts")) \
        if features == "native" else R.PARTS
    return osp.exists(osp.join(parts, rid + ".json"))


def units_done(DS, features):
    return all(osp.exists(R.grid_unit_paths(DS, f, p, features)[0]) for f in range(5) for p in R.GRID_SPACE["psi1"])


def build(native=True, diag=True):
    T = []
    if native:
        for DS, c in NATIVE_R2:
            meta = osp.join((os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_protoglad/native"), f"run_{DS}", "native_meta.json")

            def _done(meta=meta):
                if not osp.exists(meta):
                    return False
                import json
                return bool(json.load(open(meta)).get("completed"))
            T.append(Task(f"native_{DS}", [PY, "-B", NR, "--ds", DS], c, _done))
    feats = ["ubgold"] + (["native"] if diag else [])
    for features in feats:
        dsets = R.DATASETS if features == "ubgold" else DIAG
        off = 0 if features == "ubgold" else -10000      # diagnostic after every main task
        for DS in dsets:
            for fold in range(5):
                for psi1 in R.GRID_SPACE["psi1"]:
                    js = R.grid_unit_paths(DS, fold, psi1, features)[0]
                    T.append(Task(f"grid_{features}_{DS}_f{fold}_p{psi1}",
                                  [PY, "-B", RM, "grid", "--dataset", DS, "--fold", str(fold), "--psi1", str(psi1),
                                   "--features", features],
                                  off + BASE[DS] * (1 + psi1 / 64.0), (lambda js=js: osp.exists(js))))
            gb = R.gridbest_path(DS, features)
            T.append(Task(f"select_{features}_{DS}", [PY, "-B", RM, "select", "--dataset", DS, "--features", features],
                          off + 1e6, (lambda gb=gb: osp.exists(gb)),
                          ready=(lambda DS=DS, f=features: units_done(DS, f))))
            for fold in range(5):
                for hp in ("hpfree", "ubgold"):
                    env_parts = [] if features == "ubgold" else ["--features", "native"]
                    need_select = hp == "ubgold" and not (DS == "MUTAG" and features == "ubgold")
                    T.append(Task(f"part_{features}_{DS}_f{fold}_{hp}",
                                  [PY, "-B", RM, "--method", "ProtoGLAD", "--dataset", DS, "--fold", str(fold),
                                   "--hp", hp] + env_parts,
                                  off + BASE[DS] / 4.0,
                                  (lambda DS=DS, fo=fold, hp=hp, f=features: part_done(DS, fo, hp, f)),
                                  # parts run after their grid units so the bit-identity check is done
                                  ready=((lambda gb=gb: osp.exists(gb)) if need_select else
                                         (lambda DS=DS, f=features: units_done(DS, f)))))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--minutes", type=float, default=52.0)
    ap.add_argument("--no_native", action="store_true")
    ap.add_argument("--no_diag", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    t_end = time.time() + 60 * a.minutes
    T = [t for t in build(not a.no_native, not a.no_diag) if not t.done()]
    T.sort(key=lambda t: -t.cost)
    running, finished, failed = [], 0, []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", EP_THREADS="1", TQDM_DISABLE="1",
               EP_PARTS_DIAG=os.environ.get("EP_PARTS_DIAG", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_protoglad/diag_parts")))
    print(f"[pool] {len(T)} pending tasks, {a.workers} workers, until {time.ctime(t_end)}", flush=True)
    while T or running:
        for t in list(running):
            rc = t.proc.poll()
            if rc is None:
                continue
            running.remove(t)
            dt = time.time() - t.t0
            if t.done():
                finished += 1
                print(f"[ok] {t.name} {dt:.0f}s", flush=True)
            else:
                print(f"[rc={rc}] {t.name} {dt:.0f}s (log {t.log})", flush=True)
                if rc == 3 and t.attempts < 20:          # locked by another job: retry later
                    T.append(t)
                else:
                    failed.append(t.name)
        if time.time() > t_end - 120:
            if not running:
                break
            time.sleep(5)
            continue
        free = a.workers - len(running)
        for t in list(T):
            if free <= 0:
                break
            if t.done():
                T.remove(t)
                continue
            if not t.ready():
                continue
            T.remove(t)
            t.attempts += 1
            t.log = osp.join(LOGS, t.name + ".log")
            e = dict(env)
            if t.name.startswith("part_native") or t.name.startswith("grid_native") or t.name.startswith("select_native"):
                e["EP_PARTS"] = env["EP_PARTS_DIAG"]
            t.proc = subprocess.Popen(t.argv, stdout=open(t.log, "a"), stderr=subprocess.STDOUT, env=e, cwd="/tmp")
            t.t0 = time.time()
            running.append(t)
            free -= 1
        if not running and T and not any(t.ready() for t in T):
            print("[pool] nothing runnable (blocked dependencies)", flush=True)
            break
        time.sleep(2)
    left = [t.name for t in T if not t.done()]
    print(f"[pool] finished {finished}; failed {len(failed)}: {failed[:20]}; not started/left {len(left)}: {left[:20]}",
          flush=True)


if __name__ == "__main__":
    main()
