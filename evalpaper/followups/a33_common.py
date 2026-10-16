"""A33 follow-ups: pieces shared by orientation/run_orient.py and ablation_xs/run_ablation.py.

harness/run_eval.py and typed/run_typed.py are imported READ-ONLY as modules (never copied, never edited); their
sha256 goes into every run's meta. Everything here only changes attributes of those module objects inside the
calling process. Marked "# CHANGE-A33" where behaviour differs from the harness.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
PROJECT = PRELIM   # release: the repository root
HARNESS = osp.join(EVALPAPER, "harness")
RUN_EVAL = osp.join(HARNESS, "run_eval.py")
RUN_TYPED = osp.join(EVALPAPER, "typed", "run_typed.py")
DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/followups")
AMENDMENT = "A33 (PREREG_AMENDMENTS.md, written 2026-10-14 23:47:22 EDT)"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_harness():
    """harness/run_eval.py as module 'ep_harness_run_eval' (the name typed/run_typed.py also uses, so both paths
    share the one patched module object)."""
    if "ep_harness_run_eval" in sys.modules:
        return sys.modules["ep_harness_run_eval"]
    spec = importlib.util.spec_from_file_location("ep_harness_run_eval", RUN_EVAL)
    RE = importlib.util.module_from_spec(spec)
    sys.modules["ep_harness_run_eval"] = RE
    spec.loader.exec_module(RE)
    return RE


def load_typed_runner():
    """typed/run_typed.py as a module (read-only; it imports typed_data from its own folder)."""
    if "a33_run_typed" in sys.modules:
        return sys.modules["a33_run_typed"]
    spec = importlib.util.spec_from_file_location("a33_run_typed", RUN_TYPED)
    RT = importlib.util.module_from_spec(spec)
    sys.modules["a33_run_typed"] = RT
    spec.loader.exec_module(RT)
    return RT


def find_meta_out(codes):
    """The meta_out dict of the enclosing harness/typed _run frame (same object; keys added here are written by
    that _run). `codes`: the code objects of the original _run functions."""
    f = sys._getframe(1)
    while f is not None and f.f_code not in codes:
        f = f.f_back
    assert f is not None, "fit function not called from a harness/typed _run"
    m = f.f_locals["meta_out"]
    assert isinstance(m, dict) and "run_id" in m
    return m


def versions():
    import numpy as np
    import torch
    out = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    for mod in ("torch_geometric", "scipy", "sklearn", "networkx", "torch_scatter", "faiss"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception as e:   # pragma: no cover
            out[mod] = f"unavailable: {type(e).__name__}"
    return out


_PROV = {}


def provenance(own_files):
    """Code hashes (own files + harness/typed runners + UB-GOLD commit), versions, project git HEAD."""
    if _PROV:
        return dict(_PROV)
    code = {k: sha256_file(v) for k, v in own_files.items() if osp.exists(v)}
    for k, v in (("harness/run_eval.py", RUN_EVAL), ("typed/run_typed.py", RUN_TYPED),
                 ("typed/typed_data.py", osp.join(EVALPAPER, "typed", "typed_data.py")),
                 ("followups/a33_common.py", osp.join(HERE, "a33_common.py"))):
        code[k] = sha256_file(v)
    try:
        head = subprocess.run(["git", "-C", PROJECT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:
        head = None
    _PROV.update(a33_code_sha256=code, a33_versions=versions(), a33_project_git_head=head,
                 ubgold_commit=open(osp.join(HARNESS, "ubgold_src", "COMMIT")).read().strip())
    return dict(_PROV)


def record_crash(crash_dir, rid, e, ckpath, extra=None):
    """A12 bookkeeping (convention of typed/run_typed.py): crash record + checkpoint moved aside, so the next attempt
    starts from epoch 1."""
    os.makedirs(crash_dir, exist_ok=True)
    k = 1 + sum(1 for f in os.listdir(crash_dir) if f.startswith(rid + ".attempt"))
    tb = traceback.format_exc()
    msg = f"{type(e).__name__}: {e}"
    low = (msg + tb[-2000:]).lower()
    rec = dict(run_id=rid, attempt=k, time=time.strftime("%Y-%m-%d %H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"),
               host=platform.node(), error=msg[:2000], traceback=tb[-8000:],
               nonfinite_suspected=any(w in low for w in ("nan", "inf", "non-finite", "nonfinite")))
    rec.update(extra or {})
    if ckpath and osp.exists(ckpath):
        os.replace(ckpath, ckpath + f".crash{k}")
        rec["checkpoint_moved_to"] = ckpath + f".crash{k}"
    with open(osp.join(crash_dir, f"{rid}.attempt{k}.json"), "w") as f:
        json.dump(rec, f, indent=1)


def n_crashes(crash_dir, rid):
    if not osp.isdir(crash_dir):
        return 0
    return sum(1 for f in os.listdir(crash_dir) if f.startswith(rid + ".attempt") and f.endswith(".json"))


def device_check(C, planned, meta):
    """Planned device type vs this process; refuse a cross-device resume. Returns None or 'device_mismatch'."""
    if planned is not None and C.device.type != planned:
        print(f"[device] planned {planned} but running on {C.device.type}; not started", flush=True)
        return "device_mismatch"
    if C.state is not None:
        if "a33" not in C.state:
            print("[resume] checkpoint is not an A33 checkpoint; refusing", flush=True)
            return "device_mismatch"
        prev = {s["device"] for s in C.state.get("segments", [])}
        if prev and prev != {C.device.type}:
            print(f"[resume] checkpoint segments ran on {prev}, now {C.device.type}; refusing", flush=True)
            return "device_mismatch"
    return None


def install_ckpt_marker(RE, marker):
    """CHANGE-A33: every checkpoint written by the harness's maybe_stop carries an 'a33' marker (so a resume can
    verify it is an A33 checkpoint of the same variant). Wraps RE.save_ckpt in this process only."""
    if getattr(RE.save_ckpt, "_a33", False):
        RE.save_ckpt._a33_marker.update(marker)
        return
    orig = RE.save_ckpt

    def save_ckpt(path, payload):
        payload = dict(payload)
        payload["a33"] = dict(save_ckpt._a33_marker)
        return orig(path, payload)
    save_ckpt._a33 = True
    save_ckpt._a33_marker = dict(marker)
    RE.save_ckpt = save_ckpt


def make_flipped_tudataset(TUDataset, tag_index=False):
    """CHANGE-A33 (orientation): a TUDataset whose graphs carry y' = 1 - y. UB-GOLD's verbatim get_ad_split_TU /
    get_ad_dataset_TU then treat original class 0 as normal and original class 1 as anomalous. tag_index (fold builder
    only): each graph also carries its dataset index as `a33_index`."""
    class FlippedTUDataset(TUDataset):
        def get(self, idx):
            d = super().get(idx)
            d.y = 1 - d.y                       # CHANGE-A33: class roles swapped
            if tag_index:
                d.a33_index = int(idx)
            return d
    return FlippedTUDataset


class Progress:
    """Training-side progress line (no test metric): wraps RE.add_eval for the harness path."""

    @staticmethod
    def install(RE):
        if getattr(RE.add_eval, "_a33", False):
            return
        orig = RE.add_eval

        def add_eval(C, epoch, *a, **k):
            r = orig(C, epoch, *a, **k)
            rec = C.traj[-1]
            if epoch == C.args.num_epoch or len(C.traj) % 10 == 1:
                print(f"[{C.run_id}] ep {epoch}/{C.args.num_epoch} train {rec.get('train_loss', float('nan')):.4f} "
                      f"hold {rec.get('holdout_score', float('nan')):.4f} t={rec.get('t', 0):.0f}s", flush=True)
            return r
        add_eval._a33 = True
        RE.add_eval = add_eval


# ----------------------------------------------------------------------------------------------------------------
# task bookkeeping shared by run_pool_a33.py, jobs/gen_tasks.py and the collect scripts (no torch import)
# ----------------------------------------------------------------------------------------------------------------
EXPS = ("orient", "abl_ub9", "abl_typed")
TYPED_UNITS = {"GX:F2": ("GX", "F2"), "MSV2:MS": ("MSV2", "MS")}


def task_rid(exp, method, unit, fold, tag=""):
    if exp == "abl_typed":
        gen, fv = TYPED_UNITS[unit]
        rid = f"{method}_{gen}_{fv}_s{fold}"
    else:
        rid = f"{method}_{unit}_fold{fold}"
    return rid + (f"_{tag}" if tag else "")


def task_dirs(exp, smoke=False, xs="zeros"):
    """(parts, crash) directories of a task (must match run_orient.roots / run_ablation.roots)."""
    if exp == "orient":
        b = osp.join(DATA_ROOT, "orientation")
        if smoke:
            return osp.join(b, "smoke", "parts"), osp.join(b, "smoke", "crash")
        return osp.join(HERE, "orientation", "parts"), osp.join(b, "crash")
    path = "ub9" if exp == "abl_ub9" else "typed"
    if smoke:
        b = osp.join(DATA_ROOT, "ablation_xs", "smoke", xs, path)
        return osp.join(b, "parts"), osp.join(b, "crash")
    return (osp.join(HERE, "ablation_xs", f"parts_{path}_{xs}"),
            osp.join(DATA_ROOT, "ablation_xs", xs, path, "crash"))


def parse_tasks(path):
    """One task per line:
       EXP METHOD UNIT FOLD dev=cpu|cuda [mem=GB] [gmem=GB] [threads=N] [est=SEC] [xs=zeros|ones]
           [smoke max_epochs=N tag=T] [stop_at=E]
    EXP: orient | abl_ub9 | abl_typed; UNIT: dataset, or GX:F2 / MSV2:MS (abl_typed); FOLD: fold or seed."""
    tasks = []
    for line in open(path):
        p = line.split()
        if len(p) < 5 or line.lstrip().startswith("#"):
            continue
        t = dict(exp=p[0], method=p[1], unit=p[2], fold=int(p[3]), dev=None, mem=4.0, gmem=2.0, threads=None,
                 est=0.0, xs="zeros", max_epochs=None, tag="", smoke=False, stop_at=None)
        for x in p[4:]:
            if x.startswith("#"):
                break
            k, _, v = x.partition("=")
            if x == "smoke":
                t["smoke"] = True
            elif k in ("mem", "gmem", "est"):
                t[k] = float(v)
            elif k in ("threads", "max_epochs", "stop_at"):
                t[k] = int(v)
            elif k in ("dev", "tag", "xs"):
                t[k] = v
        assert t["exp"] in EXPS, line
        assert t["dev"] in ("cpu", "cuda"), f"task without dev=cpu|cuda: {line.strip()}"
        assert t["xs"] in ("zeros", "ones"), line
        t["rid"] = task_rid(t["exp"], t["method"], t["unit"], t["fold"], t["tag"])
        parts, crash = task_dirs(t["exp"], t["smoke"], t["xs"])
        t["out"] = osp.join(parts, t["rid"] + ".json")
        t["crash_dir"] = crash
        t["key"] = f"{t['exp']}:{t['xs'] if t['exp'] != 'orient' else '-'}:{t['rid']}"
        tasks.append(t)
    return tasks


def task_open(t, max_crashes=2):
    """Not finished and not DIVERGED-candidate (A12: >= 2 recorded crash attempts are never retried)."""
    return not osp.exists(t["out"]) and n_crashes(t["crash_dir"], t["rid"]) < max_crashes


# ----------------------------------------------------------------------------------------------------------------
# collection helpers (grail-prelim env; the existing collect scripts run UNCHANGED as subprocesses)
# ----------------------------------------------------------------------------------------------------------------
HARNESS_COLLECT = osp.join(HARNESS, "collect.py")
TYPED_COLLECT = osp.join(EVALPAPER, "typed", "collect_typed.py")
SCRATCH = osp.join(DATA_ROOT, "tmp")


def scan_parts(parts_dir, typed=False):
    """[(path, meta)] of every part; tagged = smoke/timing (tag, max_epochs override or run_id != stem)."""
    import glob
    out = []
    for fn in sorted(glob.glob(osp.join(parts_dir, "*.json"))):
        m = json.load(open(fn))["meta"]
        if typed:
            stem = f"{m['method']}_{m['generator']}_{m['feature_variant']}_s{m['seed']}"
        else:
            stem = f"{m['method']}_{m['dataset']}_fold{m['fold']}"
        m["_tagged"] = bool(m.get("tag")) or m.get("max_epochs_override") is not None or m["run_id"] != stem
        m["_stem"] = stem
        out.append((fn, m))
    return out


def untagged_fixture(parts, dest):
    """Smoke tests only: copy tagged parts into `dest` as if untagged (run_id = stem, no override) so the unchanged
    collect scripts accept them. Never used for real outputs."""
    os.makedirs(dest, exist_ok=True)
    for fn, m in parts:
        J = json.load(open(fn))
        J["meta"]["run_id"] = m["_stem"]
        J["meta"]["max_epochs_override"] = None
        J["meta"].pop("tag", None)
        with open(osp.join(dest, m["_stem"] + ".json"), "w") as f:
            json.dump(J, f)


def device_type(meta):
    return "+".join(sorted({"cuda" if "cuda" in s["device"] else "cpu" for s in meta["segments"]}))


def run_collect(script, env_extra=None, args=()):
    """Run an existing collect script unchanged; returns its stdout (counts only)."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    r = subprocess.run([sys.executable, "-B", script, *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{osp.basename(script)} failed:\n{r.stderr[-4000:]}")
    return r.stdout
