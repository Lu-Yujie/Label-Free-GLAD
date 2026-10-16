"""Common machinery of the Phase B typed adapters (the 9 E5/E4 methods on GX and MS-v2; RUN_NOTES_ext.md §0).

Every adapter `typed/ext/<M>_typed.py` imports its port `evalpaper/ext/<M>/run_method.py` READ-ONLY (never
edited; code it has to change is copied into the adapter and marked `# CHANGE-TYPED`) and uses this module for:
  * data: typed/typed_data.py (load_typed with the frozen hashes, UNITS_A28, held-out split, PyG builder);
  * the TypedRun bookkeeping object, which is also the `C` context the ports' loops expect (traj, train_loss, S,
    state, segments, deadline, last_save, ckpt_every_s, ckpath, run_id, eval_epochs, t_train, t_eval);
  * per-type recording (UB-GOLD metric functions, n_distinct for the A21 tie flag, held-out mean = VAL-NORMAL signal);
  * the RUN_NOTES §0a RNG discipline for scoring every group as its own test set (score_groups);
  * checkpoints / locks / crash records (A12), the part JSON and the per-graph score npz, meta (seeds, device,
    versions, configuration, code sha256 incl. molswap_v2.py, third-party commit).
Paths (RUN_NOTES_ext §0.1 item 10): grid parts typed/ext/parts/; scores $GLAD_DATA_ROOT/typed/ext_scores/<M>/;
checkpoints, locks, caches, crash records $GLAD_DATA_ROOT/typed/ext_ckpt/; smoke runs only under
$GLAD_DATA_ROOT/typed/smoke_ext/. Nothing test-related is ever printed (blinding).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import os
import os.path as osp
import platform
import time
import traceback

import numpy as np

EXT = osp.dirname(osp.abspath(__file__))                 # evalpaper/typed/ext
TYPED = osp.dirname(EXT)                                 # evalpaper/typed
EVALPAPER = osp.dirname(TYPED)
PRELIM = osp.dirname(EVALPAPER)
PORTS = osp.join(EVALPAPER, "ext")                       # the 9 ports (read only)
HARNESS = osp.join(EVALPAPER, "harness")
UBGOLD_METRIC = osp.join(HARNESS, "ubgold_src", "GAOOD", "metric", "metric.py")
for _p in (TYPED, PRELIM, EVALPAPER):
    if _p not in sys.path:
        sys.path.append(_p)
import typed_data as TD  # noqa: E402

TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
GRID_PARTS = osp.join(EXT, "parts")
METHODS = ["MUSE", "LGKDE", "GLADPro", "GLADMamba", "SEGO", "GLASS", "ProtoGLAD", "HC-GLAD", "SUL-GUL"]
UNITS = list(TD.UNITS_A28)
TEST_STOP_EPOCH = int(os.environ.get("EP_TEST_STOP_EPOCH", "0"))     # resume tests only (smoke)
COMMON_FILES = {"typed/ext/typed_ext_common.py": osp.join(EXT, "typed_ext_common.py"),
                "typed/typed_data.py": osp.join(TYPED, "typed_data.py"),
                "typed/molswap_v2.py": osp.join(TYPED, "molswap_v2.py"),
                "m2_typed/convert.py": osp.join(PRELIM, "m2_typed", "convert.py"),
                "m2_typed/m2io.py": osp.join(PRELIM, "m2_typed", "m2io.py")}


# ----------------------------------------------------------------------------------------------------------------
# paths and small helpers
# ----------------------------------------------------------------------------------------------------------------
def roots(smoke):
    """dict(parts, scores, ckpt, crash). Smoke runs never touch the grid parts directory."""
    if smoke:
        base = osp.join(TYPED_ROOT, "smoke_ext")
        parts = os.environ.get("TYPED_EXT_PARTS", osp.join(base, "parts"))
        assert osp.realpath(parts) != osp.realpath(GRID_PARTS), "smoke outputs must stay outside typed/ext/parts"
        assert osp.realpath(parts) != osp.realpath(osp.join(TYPED, "parts")), "never write into typed/parts"
        return dict(parts=parts, scores=osp.join(base, "scores"), ckpt=osp.join(base, "ckpt"),
                    crash=osp.join(base, "crash"))
    return dict(parts=os.environ.get("TYPED_EXT_PARTS", GRID_PARTS), scores=osp.join(TYPED_ROOT, "ext_scores"),
                ckpt=osp.join(TYPED_ROOT, "ext_ckpt"), crash=osp.join(TYPED_ROOT, "ext_ckpt", "crash"))


def run_id_of(method, generator, seed, fv, tag=""):
    return f"{method}_{generator}_{fv}_s{seed}" + (f"_{tag}" if tag else "")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    """Import a file as a module under a private name (read only: no bytecode is written next to it)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_port(method):
    """The port's run_method.py (evalpaper/ext/<M>/run_method.py), imported read-only."""
    return load_module("typed_port_" + method.replace("-", "_"), osp.join(PORTS, method, "run_method.py"))


def load_harness():
    """harness/run_eval.py (read-only import, the same module name the ports use)."""
    return load_module("ep_harness_run_eval", osp.join(HARNESS, "run_eval.py"))


def load_metric():
    """UB-GOLD's GAOOD/metric/metric.py (ood_auc, ood_aupr, fpr95), as the harness and the ports use it."""
    return load_module("typed_ext_ubgold_metric", UBGOLD_METRIC)


def n_distinct_finite(s):
    s = np.asarray(s, dtype=np.float64)
    return int(np.unique(s[np.isfinite(s)]).size)


def skey(k):
    """npz-safe group key ('MS1:matched' -> 'MS1-matched')."""
    return k.replace(":", "-")


def versions(extra=()):
    import torch
    out = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    for mod in ("torch_geometric", "sklearn", "scipy", "networkx", "torch_scatter", "torch_sparse") + tuple(extra):
        try:
            m = sys.modules.get(mod) or __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------------------------------------------------
# RNG (copies of the harness helpers, so this module can be used before torch_geometric is imported)
# ----------------------------------------------------------------------------------------------------------------
def rng_state():
    import random
    import torch
    st = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    import random
    import torch
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def acquire_lock(path):
    """= harness acquire_lock (O_EXCL lock with the job id; a lock whose job is not RUNNING, or whose interactive pid
    on this host is dead, is stale and taken over)."""
    import subprocess
    me = f"{os.environ.get('SLURM_JOB_ID', 'nojob')} {os.getpid()} {os.uname().nodename}"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = open(path).read().split()
            except OSError:
                owner = []
            job = owner[0] if owner else "nojob"
            alive = False
            if job != "nojob":
                q = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
                alive = q.returncode == 0 and "RUNNING" in q.stdout
                if alive and job == os.environ.get("SLURM_JOB_ID") and len(owner) >= 3 \
                        and owner[2] == os.uname().nodename:
                    try:                     # same job: is the sibling process still alive?
                        os.kill(int(owner[1]), 0)
                    except (OSError, ValueError):
                        alive = False
                    else:
                        return False
            elif len(owner) >= 3 and owner[2] == os.uname().nodename:
                try:
                    os.kill(int(owner[1]), 0)
                    alive = True
                except (OSError, ValueError):
                    alive = False
            if alive and job != os.environ.get("SLURM_JOB_ID", "nojob-x"):
                return False
            try:
                os.remove(path)
            except OSError:
                pass
    return False


def peak_mem():
    import resource
    import torch
    out = dict(maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
    if torch.cuda.is_available():
        out["cuda_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        out["cuda_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3
    return out


def save_ckpt(path, payload):
    import torch
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------
# one typed run
# ----------------------------------------------------------------------------------------------------------------
class TypedRun:
    """Bookkeeping of one (method, generator, seed, feature variant) run; doubles as the ports' `C` context."""

    def __init__(self, method, generator, seed, fv=None, tag="", smoke=False, deadline=None, ckpt_every_s=300.0,
                 max_epochs=None, verbose=True):
        assert method in METHODS, method
        assert generator in ("GX", "MSV2"), f"{generator}: Phase B runs GX and MS-v2 only (MolSwap v1 retired, A28)"
        fv = fv or TD.FEATURE_VARIANTS[generator][0]
        assert (generator, int(seed), fv) in UNITS, (generator, seed, fv)
        if max_epochs is not None or tag:
            assert smoke and tag, "--max_epochs / --tag are for smoke runs only (needs --smoke and --tag)"
        self.method, self.generator, self.seed, self.fv, self.tag, self.smoke = method, generator, int(seed), fv, tag, smoke
        self.max_epochs, self.verbose, self.deadline, self.ckpt_every_s = max_epochs, verbose, deadline, ckpt_every_s
        self.t_start = time.time()
        self.R = roots(smoke)
        self.run_id = run_id_of(method, generator, self.seed, fv, tag)
        self.out_json = osp.join(self.R["parts"], self.run_id + ".json")
        self.ckdir = osp.join(self.R["ckpt"], method)
        self.ckpath = osp.join(self.ckdir, self.run_id + ".pt")
        self.lock = self.ckpath + ".lock"
        self.cache_dir = osp.join(self.R["ckpt"], method, "cache")      # label-free per-unit caches
        self.traj, self.train_loss, self.S = [], [], {}
        self.state = None
        self.t_train = self.t_eval = 0.0
        self.last_save = time.time()
        self.segments = []
        self.eval_epochs = set()
        self.meta = {}
        self.code_files = dict(COMMON_FILES)
        self._locked = False
        self.dry_n = 0

    # ---- data -------------------------------------------------------------------------------------------------
    def load_data(self):
        """TypedData (frozen hashes verified), group keys, labels, FROZEN cells. No RNG is consumed."""
        self.td = TD.load_typed(self.generator, self.seed, self.fv)
        self.dry_n = int(os.environ.get("TYPED_EXT_DRY_N", "0") or 0)
        if self.dry_n:
            # development-only code-path check (login node): truncated test sets; never a measurement or a result
            assert self.smoke and self.tag.startswith("dry"), "TYPED_EXT_DRY_N needs --smoke and a tag 'dry*'"
            self.td.test_normals = self.td.test_normals[:self.dry_n]
            from collections import OrderedDict
            self.td.groups = OrderedDict((k, v[:self.dry_n]) for k, v in self.td.groups.items())
        self.keys = list(self.td.groups)
        self.y = {k: self.td.test_set(k)[1] for k in self.keys}
        self.n_norm = len(self.td.test_normals)
        self.cells = {k: self.td.cell(k) for k in self.keys}
        return self.td

    # ---- lifecycle --------------------------------------------------------------------------------------------
    def begin(self):
        """-> 'done' (part exists), 'locked' (another job runs it) or None (go)."""
        os.makedirs(self.R["parts"], exist_ok=True)
        os.makedirs(self.ckdir, exist_ok=True)
        if osp.exists(self.out_json):
            print(f"[skip] {self.out_json} exists", flush=True)
            return "done"
        if not acquire_lock(self.lock):
            print(f"[locked] {self.run_id} is running in another job", flush=True)
            return "locked"
        self._locked = True
        return None

    def end(self):
        if self._locked:
            try:
                os.remove(self.lock)
            except OSError:
                pass
            self._locked = False

    def load_state(self, device_type, threads, first_epoch=1):
        """Checkpoint (if any) and the new segment record (start_epoch = first epoch run in this segment)."""
        import torch
        if osp.exists(self.ckpath):
            self.state = torch.load(self.ckpath, map_location="cpu", weights_only=False)
            print(f"[resume] {self.ckpath} at epoch {self.state['epoch']}", flush=True)
        seg = dict(device=device_type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
                   threads=threads, start_epoch=(self.state["epoch"] + 1 if self.state is not None else first_epoch),
                   t0=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.segments = (list(self.state.get("segments", [])) if self.state is not None else []) + [seg]
        self.last_save = time.time()
        return self.state

    def restore_common(self):
        """traj / train_loss / S / RNG from the checkpoint; returns the next epoch."""
        st = self.state
        self.traj, self.train_loss = st["traj"], st["train_loss"]
        self.S = st["S"]
        set_rng_state(st["rng"])
        return st["epoch"] + 1

    def maybe_stop(self, epoch, payload_fn, final_epoch=None):
        """Checkpoint every ckpt_every_s and at the deadline (or EP_TEST_STOP_EPOCH in resume tests). True = stop now.
        No checkpoint is written after the final epoch (the run is about to finish)."""
        now = time.time()
        first = self.segments[-1]["start_epoch"]
        stop = (self.deadline is not None and now > self.deadline) or (
            TEST_STOP_EPOCH > 0 and epoch == TEST_STOP_EPOCH and first <= epoch)
        if final_epoch is not None and epoch >= final_epoch:
            return False
        if stop or now - self.last_save > self.ckpt_every_s:
            p = payload_fn()
            p.update(epoch=epoch, traj=self.traj, train_loss=self.train_loss, rng=rng_state(), S=self.S,
                     segments=self.segments)
            save_ckpt(self.ckpath, p)
            self.last_save = now
        if stop:
            print(f"[{self.run_id}] deadline reached at epoch {epoch}; checkpoint saved", flush=True)
        return stop

    # ---- scoring ----------------------------------------------------------------------------------------------
    def score_groups(self, fn):
        """RUN_NOTES §0a: every group scored as its own test set; groups 2..K each from the same saved RNG state S0,
        then S0 restored and the first group scored last (the RNG-carrying set). -> {key: fn(key)}."""
        S0 = rng_state()
        out = {}
        for k in self.keys[1:]:
            set_rng_state(S0)
            out[k] = fn(k)
        set_rng_state(S0)
        out[self.keys[0]] = fn(self.keys[0])
        return {k: out[k] for k in self.keys}

    def record(self, epoch, group_scores, hold_scores, rec_extra=None, types_extra=None, extra_S=None, t_eval0=None,
               metric=None):
        """One evaluation point. group_scores[k]: anomaly scores (higher = more anomalous) of group k's test set in
        canonical order (test normals first). types_extra[k]: extra per-type fields (own-rule keys)."""
        metric = metric or load_metric()
        rec = dict(types={})
        for k in self.keys:
            s = np.asarray(group_scores[k], dtype=np.float64).ravel()
            assert s.shape[0] == self.y[k].shape[0], (k, s.shape, self.y[k].shape)
            m = eval_metrics(metric, self.y[k], s)
            m["n_distinct"] = n_distinct_finite(s)
            if types_extra and k in types_extra:
                m.update(types_extra[k])
            rec["types"][k] = m
            self.S.setdefault("test__" + skey(k), []).append(s)
        h = np.asarray(hold_scores, dtype=np.float64).ravel()
        rec["holdout_score"] = finite_mean(h)
        rec["holdout_loss"] = rec["holdout_score"]
        rec["n_hold_nonfinite"] = int((~np.isfinite(h)).sum())
        self.S.setdefault("hold", []).append(h)
        for k, v in (extra_S or {}).items():
            self.S.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
        rec.update(rec_extra or {})
        rec["epoch"] = int(epoch)
        rec["train_loss"] = float(self.train_loss[-1]) if self.train_loss else float("nan")
        rec["t"] = time.time() - self.t_start
        self.traj.append(rec)
        if t_eval0 is not None:
            self.t_eval += time.time() - t_eval0
        if self.verbose:   # training-side quantities only (blinding)
            nf = sum(int(m["nonfinite"] > 0) for m in rec["types"].values())
            print(f"[{self.run_id}] ep {epoch} train {rec['train_loss']:.5g} hold {rec['holdout_score']:.5g} "
                  f"types_nonfinite {nf} t={rec['t']:.0f}s", flush=True)
        return rec

    # ---- output -----------------------------------------------------------------------------------------------
    def base_meta(self, port_files, commit, config, config_line, hp_notes=(), device="cpu", flags=None,
                  extra_versions=()):
        import torch
        files = dict(self.code_files)
        files.update(port_files)
        self.meta = dict(
            method=self.method, generator=self.generator, dataset=self.generator, seed=self.seed,
            feature_variant=self.fv, run_id=self.run_id, tag=self.tag, smoke=bool(self.smoke),
            max_epochs_override=self.max_epochs, phase="B (E5/E4 typed, RUN_NOTES_ext.md)",
            torch_seed=3407 + self.seed, holdout_seed=TD.HOLDOUT_SEED + 1000 * self.seed, data_seed=self.seed,
            hp="ubgold", hp_line=config_line, hp_notes=list(hp_notes),
            hparams={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                     for k, v in sorted(config.items())},
            groups=self.cells, group_order=self.keys, carry_set=self.keys[0],
            n_test_normals=self.n_norm, n_train_normal=len(self.td.train), n_fit=len(self.td.fit),
            n_holdout=len(self.td.hold), num_feat=self.td.num_feat, max_nodes_num=self.td.max_nodes_num,
            data_provenance=self.td.provenance, third_party_commit=commit, device=str(device),
            threads=torch.get_num_threads(),
            cuda_name=(torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None),
            versions=versions(extra_versions), code_sha256={k: sha256_file(p) for k, p in files.items()},
            flags=dict(flags or {}, **({"dry_n": self.dry_n} if self.dry_n else {})), host=os.uname().nodename)
        return self.meta

    def finish(self, num_epoch, eval_freq, expected_epochs=None, extra_npz=None, meta_extra=None):
        """Write the per-graph score npz and the part JSON (atomic); remove the checkpoint."""
        import torch
        seg = self.segments[-1]
        seg["t1"] = time.strftime("%Y-%m-%d %H:%M:%S")
        epochs = [int(r["epoch"]) for r in self.traj]
        if expected_epochs is not None:
            assert epochs == list(expected_epochs), (epochs[:6], len(epochs), list(expected_epochs)[:6])
        for k, v in self.S.items():
            assert len(v) == len(epochs), (k, len(v), len(epochs))
        start = seg["start_epoch"]
        ep_seg = max(1, num_epoch - start + 1)
        n_ev_seg = max(1, sum(1 for r in self.traj if r["epoch"] >= start))
        self.meta["hparams"].update(num_epoch=int(num_epoch), eval_freq=int(eval_freq))
        self.meta.update(segments=self.segments,
                         device_all="+".join(sorted({x["device"] for x in self.segments})),
                         seconds=time.time() - self.t_start, seconds_train_this_segment=self.t_train,
                         seconds_eval_this_segment=self.t_eval, epochs_this_segment=ep_seg,
                         sec_per_epoch_train=self.t_train / ep_seg, sec_per_eval=self.t_eval / n_ev_seg,
                         n_eval=len(self.traj), memory=peak_mem(), completed=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.meta.update(meta_extra or {})
        npz = osp.join(self.R["scores"], self.method, self.run_id + ".npz")
        os.makedirs(osp.dirname(npz), exist_ok=True)
        arrs = dict(epochs=np.asarray(epochs, dtype=np.int64), num_epoch=np.int64(num_epoch),
                    eval_freq=np.int64(eval_freq), hold_pos=np.asarray(self.td.hold_pos, dtype=np.int64),
                    group_order=np.array(self.keys, dtype=str))
        for k in self.keys:
            arrs["y__" + skey(k)] = self.y[k]
        for k, v in self.S.items():
            arrs[k] = np.stack([np.asarray(r, dtype=np.float64) for r in v], 0) if len(v) else np.zeros((0, 0))
        arrs.update(extra_npz or {})
        tmp = npz + ".tmp.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, npz)
        self.meta["scores_npz"] = npz
        res = dict(meta=self.meta, traj=self.traj, train_loss=[float(x) for x in self.train_loss])
        tmp = self.out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, default=_json_default)
        os.replace(tmp, self.out_json)
        if osp.exists(self.ckpath):
            os.remove(self.ckpath)
        print(f"[done] {self.run_id} in {self.meta['seconds']:.0f}s -> {self.out_json}", flush=True)
        return "done"

    def record_crash(self, e):
        """A12 bookkeeping: crash record + checkpoint moved aside (the next attempt restarts from scratch)."""
        crash = self.R["crash"]
        os.makedirs(crash, exist_ok=True)
        k = 1 + sum(1 for f in os.listdir(crash) if f.startswith(self.run_id + ".attempt"))
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}"
        low = (msg + tb[-2000:]).lower()
        rec = dict(run_id=self.run_id, method=self.method, attempt=k, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                   job=os.environ.get("SLURM_JOB_ID"), host=platform.node(), error=msg[:2000], traceback=tb[-8000:],
                   nonfinite_suspected=any(w in low for w in ("nan", "inf", "non-finite", "nonfinite")))
        if osp.exists(self.ckpath):
            os.replace(self.ckpath, self.ckpath + f".crash{k}")
            rec["checkpoint_moved_to"] = self.ckpath + f".crash{k}"
        with open(osp.join(crash, f"{self.run_id}.attempt{k}.json"), "w") as f:
            json.dump(rec, f, indent=1)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def eval_metrics(metric, y, s):
    """= harness eval_metrics (UB-GOLD ood_auc / ood_aupr / fpr95; non-finite scores -> NaN metrics + count)."""
    y = np.asarray(y).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        return dict(auroc=float("nan"), auprc=float("nan"), fpr95=float("nan"), nonfinite=int((~np.isfinite(s)).sum()))
    return dict(auroc=float(metric.ood_auc(y, s)), auprc=float(metric.ood_aupr(y, s)),
                fpr95=float(metric.fpr95(y, s)), nonfinite=0)


def finite_mean(t):
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(t.mean()) if t.size and np.all(np.isfinite(t)) else float("nan")


class PrereqMissing(RuntimeError):
    """A label-free prerequisite (e.g. GLASS's text-embedding cache) is missing: exit 5, no crash record (not an A12
    attempt); the task generator does not schedule such runs until the prerequisite exists."""


# ----------------------------------------------------------------------------------------------------------------
# CLI shared by the adapters
# ----------------------------------------------------------------------------------------------------------------
def cli(method, extra=None):
    ap = argparse.ArgumentParser(description=f"Phase B typed adapter: {method}")
    ap.add_argument("--generator", required=True, choices=["GX", "MSV2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fv", default=None, help="GX: F2 (default) | F1; MSV2: MS")
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke/timing only (needs --smoke and --tag)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="write under <TYPED_ROOT>/smoke_ext/ (never collected)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TYPED_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    if extra:
        extra(ap)
    a = ap.parse_args()
    if (a.max_epochs is not None or a.tag) and not (a.smoke and a.tag):
        sys.exit("--max_epochs / --tag are for smoke runs: add --smoke and --tag (outputs outside typed/ext/parts)")
    return a


def main(method, run_fn, extra=None):
    """Exit codes (pool convention): 0 done, 3 checkpointed / locked, 4 crash (recorded under <root>/crash),
    5 prerequisite missing (no crash record)."""
    a = cli(method, extra)
    import torch
    torch.set_num_threads(a.threads)
    R = TypedRun(method, a.generator, a.seed, a.fv, tag=a.tag, smoke=a.smoke, deadline=a.deadline,
                 ckpt_every_s=a.ckpt_every_s, max_epochs=a.max_epochs)
    st = R.begin()
    if st is not None:
        sys.exit(0 if st == "done" else 3)
    try:
        st = run_fn(R, a)
    except PrereqMissing as e:
        print(f"[prereq] {R.run_id}: {e}", flush=True)
        R.end()
        sys.exit(5)
    except Exception as e:
        traceback.print_exc()
        R.record_crash(e)
        R.end()
        sys.exit(4)
    R.end()
    sys.exit(0 if st == "done" else 3)
