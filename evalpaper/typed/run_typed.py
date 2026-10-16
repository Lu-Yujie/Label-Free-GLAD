"""A9 typed runner: the 12 UB-GOLD deep detectors on the typed generators GX and MS (RUN_NOTES.md §0).

The training/evaluation loops are the main study's: harness/run_eval.py is imported READ-ONLY (never copied, never
edited; its sha256 goes into every run's meta) and its fit_* functions run unchanged. What this file adds (all marked
CHANGE-TYPED) is the typed data and a test "loader" that makes those loops score every anomaly group as its own test set:

  * data: typed/typed_data.py (GX = M2 exports, MS = frozen MolSwap), UB-GOLD TU-format PyG Data; held-out normals
    20% with RandomState(2026 + 1000*seed); set_seed(3407 + seed); modal UB-GOLD TU hyperparameters (M2 rule);
  * TypedTestLoader stands in for UB-GOLD's shuffled test DataLoader (RUN_NOTES §0a). Each iteration yields every
    group's test set (test normals + T), each one shuffled pass cut into batch_size_test batches exactly like torch's
    DataLoader (verified: same order and same post-iteration RNG state). Groups 2..K are scored first, each from the same
    saved RNG state S0; then S0 is restored and the first group (GX T1a, MS MS1:matched) is scored as the RNG-carrying
    set. So every group is scored as if it were the test set at that point, and the training stream is that of a UB-GOLD
    run whose test set is (test normals + first group). With batch_size_test == 1 (GLADC, GLocalKD) there is a single
    pass over all test graphs and per-type scores are slices of it (per-graph scoring: identical by construction);
  * Ctx.unscramble splits the scores of one pass by set (graph ids gid = set * 10^7 + position); harness.add_eval is
    replaced (in this process only) by typed_add_eval, which records per-type AUROC / AUPRC / FPR95 (UB-GOLD metric
    functions), the held-out-normal signal (A3, harness code), the training loss, and the per-graph scores. GraphDE:
    per-type metrics in both orientations (A5). CVTGAD: per-type metrics of the adaptive-normalised score for its ORACLE
    selection (A4). Nothing test-related is printed (blinding).

Outputs: <parts>/<run_id>.json (meta + trajectory), <root>/scores/<run_id>.npz (per-graph scores per set and point).
Resumable (harness checkpoints every 5 min and at the deadline; GLADC/GLocalKD replay-resume). A crash writes
<root>/crash/<run_id>.attempt<k>.json and moves the checkpoint aside, so a second attempt starts from epoch 1 (A12).

Usage (torch env; CPU nodes remap UB-GOLD's 'cuda' at runtime as the harness does):
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B run_typed.py \
      --method GLADC --generator MS --seed 0 [--fv MS|F2|F1] [--deadline UNIX] [--threads 2]
  smoke: add --max_epochs 8 --tag smoke --smoke   (all outputs under <root>/smoke/, never collected)
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
from collections import Counter

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
HARNESS = osp.join(EVALPAPER, "harness")
RUN_EVAL = osp.join(HARNESS, "run_eval.py")
for _p in (HERE, PRELIM, EVALPAPER):
    if _p not in sys.path:
        sys.path.append(_p)
import typed_data as TD  # noqa: E402

TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
GRID_PARTS = osp.join(HERE, "parts")
GID_OFF = 10_000_000
COMBINED = "ALL"
M2_RUN = osp.join(PRELIM, "m2_typed", "run_m2.py")
OWN_FILES = {"run_typed.py": osp.join(HERE, "run_typed.py"), "typed_data.py": osp.join(HERE, "typed_data.py"),
             "harness/run_eval.py": RUN_EVAL, "m2_typed/convert.py": osp.join(PRELIM, "m2_typed", "convert.py"),
             "m2_typed/m2io.py": osp.join(PRELIM, "m2_typed", "m2io.py"),
             "typed/molswap.py": osp.join(HERE, "molswap.py"),
             "typed/molswap_v2.py": osp.join(HERE, "molswap_v2.py")}   # CHANGE-A28.1: record MS-v2 generator hash


def roots(smoke):
    """(parts, scores, ckpt, crash). Smoke runs never touch the grid parts directory."""
    base = osp.join(TYPED_ROOT, "smoke") if smoke else TYPED_ROOT
    parts = os.environ.get("TYPED_PARTS", osp.join(base, "parts") if smoke else GRID_PARTS)
    if smoke:
        assert osp.realpath(parts) != osp.realpath(GRID_PARTS), "smoke outputs must stay outside typed/parts"
    return parts, osp.join(base, "scores"), osp.join(base, "ckpt"), osp.join(base, "crash")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_harness():
    """harness/run_eval.py as a module (read-only import; nothing is written next to it: dont_write_bytecode)."""
    if "ep_harness_run_eval" in sys.modules:
        return sys.modules["ep_harness_run_eval"]
    spec = importlib.util.spec_from_file_location("ep_harness_run_eval", RUN_EVAL)
    RE = importlib.util.module_from_spec(spec)
    sys.modules["ep_harness_run_eval"] = RE
    spec.loader.exec_module(RE)
    return RE


def m2_lines():
    """deep_rerun/M2 modal lines of the old 4 (run_m2.DR_MUTAG_LINES), read with ast (no import of run_m2)."""
    import ast
    tree = ast.parse(open(M2_RUN).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DR_MUTAG_LINES":
            return ast.literal_eval(n.value)
    raise KeyError("DR_MUTAG_LINES")


# ----------------------------------------------------------------------------------------------------------------
# hyperparameters: the M2 modal rule (RUN_NOTES §0 item 6)
# ----------------------------------------------------------------------------------------------------------------
def modal_hparams(RE, method, generator):
    """Per option: the most frequent value over the method's 11 UB-GOLD TU `-exp_type ad` lines (A2 fix applied first);
    ties -> mymain's default if it is tied, else the smallest value (m2_typed/run_m2.modal_line's rule)."""
    parser = RE.build_ubgold_parser()
    parsed = []
    for DS, (line, toks) in RE.sh_lines(method).items():
        if DS in RE.TU_DATASETS:
            t2, _ = RE.fix_known_defects(method, DS, list(toks))
            parsed.append(vars(parser.parse_args(t2)))
    assert len(parsed) == 11, (method, len(parsed))
    dtoks = ["-exp_type", "ad", "-DS", "X", "-model", method] + ([method] if method in RE.SUBPARSER else [])
    args = parser.parse_args(dtoks)
    default = dict(vars(args))
    assert all(set(p) == set(default) for p in parsed), method
    ties, changed = {}, {}
    for k in sorted(default):
        if k == "DS":
            continue
        cnt = Counter(p[k] for p in parsed)
        top = max(cnt.values())
        tied = [v for v, c in cnt.items() if c == top]
        v = default[k] if default[k] in tied else sorted(tied, key=lambda x: (str(type(x)), x))[0]
        if len(tied) > 1:
            ties[k] = dict(tied={str(t): cnt[t] for t in tied}, chosen=v, default=default[k])
        if v != default[k]:
            changed[k] = v
        setattr(args, k, v)
    args.DS = f"typed_{generator}"     # CHANGE-TYPED: only used by build_save_path (disabled) and Tox21 checks
    if method in RE.OLD4:              # identical to deep_rerun's / M2's modal lines (up to -DS)
        ref = vars(parser.parse_args(m2_lines()[method].split()[2:]))
        mine = vars(args)
        diff = {k: (mine[k], ref[k]) for k in ref if k != "DS" and mine[k] != ref[k]}
        assert not diff, (method, diff)
    line = (f"modal UB-GOLD TU config (11 lines, M2 rule): -model {method} "
            + " ".join(f"-{k} {v}" for k, v in sorted(changed.items())))
    return args, line, dict(ties=ties, non_default=changed)


# ----------------------------------------------------------------------------------------------------------------
# the typed test loader
# ----------------------------------------------------------------------------------------------------------------
class TypedTestLoader:
    """Iterable used as Ctx.dataloader_val (= UB-GOLD's test loader). RUN_NOTES §0a.

    own_set mode: every group's test set (normals + T) is one shuffled pass cut into batch_size_test batches. Groups
    2..K are scored first, each from the same saved RNG state S0; then S0 is restored and the first group (the
    RNG-carrying set) is scored, consuming the global RNG exactly as torch's shuffled DataLoader would.
    slice mode (batch_size_test == 1): one pass over all test graphs (per-graph scoring; per-type = slices)."""

    def __init__(self, RE, sets, combined, batch_size, mode):
        self.RE, self.sets, self.combined, self.bs, self.mode = RE, sets, combined, int(batch_size), mode
        self.dataset = combined if mode == "slice" else sets[0][1]   # (attribute parity; the loops never read it)

    @staticmethod
    def _draw_perm(n):
        torch.empty((), dtype=torch.int64).random_()                     # DataLoader iterator: _base_seed draw
        s = int(torch.empty((), dtype=torch.int64).random_().item())     # RandomSampler.__iter__: seed draw
        g = torch.Generator()
        g.manual_seed(s)
        return torch.randperm(n, generator=g).tolist()

    def _batches(self, lst, perm):
        from torch_geometric.data import Batch
        for i in range(0, len(perm), self.bs):
            yield Batch.from_data_list([lst[j] for j in perm[i:i + self.bs]])

    def __iter__(self):
        if self.mode == "slice":
            yield from self._batches(self.combined, self._draw_perm(len(self.combined)))
            return
        S0 = self.RE.rng_state()                        # python, numpy, torch CPU (+CUDA)
        for _, lst in self.sets[1:]:
            self.RE.set_rng_state(S0)                   # each group scored from the same state
            yield from self._batches(lst, self._draw_perm(len(lst)))
        self.RE.set_rng_state(S0)                       # the carrying set continues the training stream
        yield from self._batches(self.sets[0][1], self._draw_perm(len(self.sets[0][1])))

    def __len__(self):
        nb = lambda n: (n + self.bs - 1) // self.bs
        if self.mode == "slice":
            return nb(len(self.combined))
        return sum(nb(len(l)) for _, l in self.sets)


def skey(k):
    """npz-safe name of a group key ('MS1:matched' -> 'MS1-matched')."""
    return k.replace(":", "-")


def n_distinct_finite(s):
    s = np.asarray(s, dtype=np.float64)
    return int(np.unique(s[np.isfinite(s)]).size)


def typed_add_eval(C, epoch, test_main, hold_main, extra_test=None, extra_hold=None, rec_extra=None, t_eval0=None):
    """CHANGE-TYPED replacement of harness.add_eval (same call signature; installed only in this process)."""
    RE, T = C._RE, C._typed
    calls, C._calls = C._calls, []
    assert len(calls) == (2 if C.method == "CVTGAD" else 1), (C.method, len(calls))
    sign = -1.0 if C.method == "GraphDE" else 1.0          # harness passes -infer_e_gx for GraphDE (A5)
    test_main = np.asarray(test_main, dtype=np.float64)
    MAIN = T["main"]                                       # carrying set (own_set) or all test graphs (slice)
    assert np.array_equal(test_main, sign * calls[0][MAIN], equal_nan=True)

    def per_type(call, sgn):
        if T["mode"] == "slice":
            return {k: sgn * call[COMBINED][T["idx"][k]] for k in T["keys"]}
        return {k: sgn * call[k] for k in T["keys"]}

    S = per_type(calls[0], sign)
    rec = dict(types={}, carry_set=MAIN)
    if T["mode"] == "slice":     # all test graphs in one pass: their joint metrics are a diagnostic
        rec["ALL"] = RE.eval_metrics(C.metric, C.y_test, test_main)
    for k, s in S.items():
        m = RE.eval_metrics(C.metric, T["y"][k], s)
        m["n_distinct"] = n_distinct_finite(s)
        rec["types"][k] = m
    if C.method == "GraphDE":      # UB-GOLD orientation (reproduction only, A5)
        rec["types_ubgold"] = {k: RE.eval_metrics(C.metric, T["y"][k], -s) for k, s in S.items()}
    Snorm = None
    if C.method == "CVTGAD":       # adaptive-normalised score: CVTGAD's ORACLE selection key (A4)
        tn = np.asarray(extra_test["test_norm"], dtype=np.float64)
        assert np.array_equal(tn, calls[1][MAIN], equal_nan=True)
        Snorm = per_type(calls[1], 1.0)
        rec["types_sel"] = {k: RE.eval_metrics(C.metric, T["y"][k], s) for k, s in Snorm.items()}
    rec["holdout_score"] = RE.finite_mean(hold_main)      # VAL-NORMAL signal (A3), harness code
    rec["holdout_loss"] = rec["holdout_score"]
    for k, v in (rec_extra or {}).items():
        # metrics the harness loops compute on their (carrying / all-graphs) set: diagnostics only, prefixed carry_
        rec[("carry_" + k) if k.startswith(("auroc", "auprc", "fpr95")) else k] = v
    rec["epoch"] = epoch
    rec["train_loss"] = C.train_loss[-1] if C.train_loss else float("nan")
    rec["t"] = time.time() - C.t_start
    C.traj.append(rec)
    C.S.setdefault("hold", []).append(np.asarray(hold_main, dtype=np.float64).ravel())
    if T["mode"] == "slice":
        C.S.setdefault("test__" + COMBINED, []).append(test_main)
    for k, s in S.items():
        C.S.setdefault("test__" + skey(k), []).append(s)
    if Snorm is not None:
        for k, s in Snorm.items():
            C.S.setdefault("testnorm__" + skey(k), []).append(s)
    for k, v in (extra_hold or {}).items():
        C.S.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
    if t_eval0 is not None:
        C.t_eval += time.time() - t_eval0
    if C.verbose:   # no test metric is printed (blinding); only training-side quantities
        nf = sum(int(m["nonfinite"] > 0) for m in rec["types"].values())
        print(f"[{C.run_id}] ep {epoch} train {rec['train_loss']:.4f} hold {rec['holdout_score']:.4f} "
              f"types_nonfinite {nf} t={rec['t']:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------
def run_id_of(method, generator, seed, fv, tag=""):
    return f"{method}_{generator}_{fv}_s{seed}" + (f"_{tag}" if tag else "")


def run(method, generator, seed, fv=None, deadline=None, max_epochs=None, tag="", smoke=False, ckpt_every_s=300.0,
        verbose=True):
    RE = load_harness()
    method = RE.ALIASES.get(method, method)
    assert method in RE.ALL12, method
    fv = fv or TD.FEATURE_VARIANTS[generator][0]
    if max_epochs is not None:
        assert tag and smoke, "--max_epochs is for smoke/timing runs only (needs --tag and --smoke)"
    PARTS, SCORES, CKPT, CRASH = roots(smoke)
    t_start = time.time()
    rid = run_id_of(method, generator, seed, fv, tag)
    os.makedirs(PARTS, exist_ok=True)
    out_json = osp.join(PARTS, rid + ".json")
    if osp.exists(out_json):
        print(f"[skip] {out_json} exists")
        return "done"
    ckdir = osp.join(CKPT, method, generator)
    os.makedirs(ckdir, exist_ok=True)
    ckpath = osp.join(ckdir, rid + ".pt")
    lock = ckpath + ".lock"
    if not RE.acquire_lock(lock):
        print(f"[locked] {rid} is running in another job")
        return "locked"
    try:
        return _run(RE, method, generator, int(seed), fv, rid, out_json, ckpath, SCORES, deadline, max_epochs, tag,
                    ckpt_every_s, verbose, t_start)
    except Exception as e:
        record_crash(CRASH, rid, e, ckpath)
        raise
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def record_crash(crash_dir, rid, e, ckpath):
    """A12 bookkeeping: crash record + checkpoint moved aside (the next attempt restarts from epoch 1)."""
    os.makedirs(crash_dir, exist_ok=True)
    k = 1 + sum(1 for f in os.listdir(crash_dir) if f.startswith(rid + ".attempt"))
    tb = traceback.format_exc()
    msg = f"{type(e).__name__}: {e}"
    low = (msg + tb[-2000:]).lower()
    rec = dict(run_id=rid, attempt=k, time=time.strftime("%Y-%m-%d %H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"),
               host=platform.node(), error=msg[:2000], traceback=tb[-8000:],
               nonfinite_suspected=any(w in low for w in ("nan", "inf", "non-finite", "nonfinite")))
    if osp.exists(ckpath):
        os.replace(ckpath, ckpath + f".crash{k}")
        rec["checkpoint_moved_to"] = ckpath + f".crash{k}"
    with open(osp.join(crash_dir, f"{rid}.attempt{k}.json"), "w") as f:
        json.dump(rec, f, indent=1)


def versions():
    out = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    for mod in ("torch_geometric", "sklearn", "scipy", "networkx", "torch_scatter", "faiss", "GCL"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


def _run(RE, method, generator, seed, fv, rid, out_json, ckpath, SCORES, deadline, max_epochs, tag, ckpt_every_s,
         verbose, t_start):
    # Same import order as harness._run: torch_geometric must first be imported AFTER the overlay (torch_sparse,
    # torch_cluster, PyGCL) is on sys.path, so PyG's optional-backend detection is the harness's, and torch_sparse is
    # JIT-compiled before UB-GOLD's InfoGraph/GraphCL modules replace warnings.warn.
    assert "torch_geometric" not in sys.modules, "torch_geometric imported before the harness overlay"
    RE.setup_overlay()
    impl = [RE.install_torch_scatter_shim()]
    if method == "GOOD-D":
        impl.append(RE.install_faiss_gpu_fallback())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        impl.append(RE.install_cpu_device_remap())
    M = RE.load_ubgold(method)
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    metric = M["metric"]
    ns = RE.load_ubgold_data_functions()
    args, hp_line, hp_notes = modal_hparams(RE, method, generator)
    args.gpu = 0
    if max_epochs is not None:
        args.num_epoch = max_epochs

    # ---- CHANGE-TYPED: seed s -> set_seed(3407 + s) (M2 rule), then the typed data (no RNG consumed) ------------
    torch_seed = 3407 + seed
    ns["set_seed"](torch_seed)
    td = TD.load_typed(generator, seed, fv)
    tf = None
    if method in RE.NEEDS_STR_ENC:   # GOOD-D / CVTGAD read x_s: UB-GOLD's init_structural_encoding (verbatim)
        tf = lambda gs: ns["init_structural_encoding"](gs, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    P = td.pyg(transform=tf)
    fit_list, hold_list = P["fit"], P["hold"]
    keys = list(td.groups)
    dataloader = DataLoader(fit_list, batch_size=args.batch_size, shuffle=True)
    args.max_nodes_num = td.max_nodes_num
    args.dataset_num_features = td.num_feat
    args.n_train = len(fit_list)
    args.n_edge_feat = 0

    comb, y_comb, grp = P["combined"], np.asarray(P["y_combined"], np.int64), np.asarray(P["group_combined"])
    for i, d in enumerate(comb):
        d.gid = i
    mode = "slice" if int(args.batch_size_test) == 1 else "own_set"
    sets = []
    if mode == "own_set":
        for j, k in enumerate(keys):
            lst = P["test"][k]
            for i, d in enumerate(lst):
                d.gid = (j + 1) * GID_OFF + i
            sets.append((k, lst))
    idx = {k: np.where((grp == "normal") | (grp == k))[0] for k in keys}
    y_type = {k: np.asarray(P["y"][k], np.int64) for k in keys}
    for k in keys:
        assert np.array_equal(y_comb[idx[k]], y_type[k])
    loader = TypedTestLoader(RE, sets, comb, args.batch_size_test, mode)
    MAIN = COMBINED if mode == "slice" else keys[0]          # RUN_NOTES §0a: carrying set = first group

    C = RE.Ctx()
    C._RE = RE
    C.method, C.DS, C.fold, C.args, C.device, C.metric = method, generator, seed, args, device, metric
    C.dataloader, C.dataloader_val, C.fit_list, C.hold_list = dataloader, loader, fit_list, hold_list
    y_main = y_comb if mode == "slice" else y_type[MAIN]
    C.n_test, C.n_hold, C.y_test, C.run_id, C.verbose, C.t_start = len(y_main), len(hold_list), y_main, rid, verbose, t_start
    C.M = M
    C.traj, C.train_loss = [], []
    C.S = {}
    C.t_train, C.t_eval = 0.0, 0.0
    C.eval_epochs = set(range(args.eval_freq, args.num_epoch + 1, args.eval_freq)) | {args.num_epoch}
    C._calls = []
    C._typed = dict(keys=keys, idx=idx, y=y_type, mode=mode, main=MAIN)
    if mode == "slice":
        set_sizes, set_index = {COMBINED: len(comb)}, {0: COMBINED}
    else:
        set_sizes = {k: len(l) for k, l in sets}
        set_index = {j + 1: k for j, k in enumerate(keys)}
    set_y = {COMBINED: y_comb, **{k: y_type[k] for k in keys}}

    def unscramble(scores, gids, y_shuffled):
        """CHANGE-TYPED: one pass over all sets -> {set: canonical-order scores}; returns the carrying set's
        (own_set) or all test graphs' (slice), i.e. the array the harness loop treats as 'the test set'."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        g = np.asarray(gids, dtype=np.int64).ravel()
        yv = np.asarray(y_shuffled, dtype=np.int64).ravel()
        assert len(s) == len(g) == len(yv) == sum(set_sizes.values()), (len(s), len(g), len(yv), set_sizes)
        out = {}
        for j, name in set_index.items():
            sel = (g // GID_OFF) == j
            pos = g[sel] % GID_OFF
            n = set_sizes[name]
            assert sel.sum() == n and len(np.unique(pos)) == n, (name, int(sel.sum()), n)
            assert np.array_equal(set_y[name][pos], yv[sel]), name
            o = np.full(n, np.nan)
            o[pos] = s[sel]
            out[name] = o
        C._calls.append(out)
        return out[MAIN]
    C.unscramble = unscramble
    C.hold_batches = [Batch.from_data_list(c) for c in RE.chunks(hold_list, min(args.batch_size_test, len(hold_list)))]

    groups_meta = {k: td.cell(k) for k in keys}
    meta_out = dict(method=method, generator=generator, dataset=generator, seed=seed, feature_variant=fv, run_id=rid,
                    tag=tag, max_epochs_override=max_epochs, torch_seed=torch_seed, hp="modal",
                    hp_line=hp_line, hp_notes=hp_notes,
                    hparams={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in sorted(vars(args).items())},
                    groups=groups_meta, per_type_mode=mode, carry_set=MAIN, n_test_combined=len(comb),
                    n_test_normals=len(td.test_normals), n_train_normal=len(td.train), n_fit=len(fit_list),
                    n_holdout=len(hold_list), holdout_seed=TD.HOLDOUT_SEED + 1000 * seed,
                    num_feat=td.num_feat, max_nodes_num=td.max_nodes_num, data_provenance=td.provenance,
                    device=str(device), impl="; ".join(impl), threads=torch.get_num_threads(),
                    cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                    versions=versions(), code_sha256={k: sha256_file(p) for k, p in OWN_FILES.items()},
                    ubgold_commit=open(osp.join(HARNESS, "ubgold_src", "COMMIT")).read().strip(),
                    ubgold_src="prelim/evalpaper/harness/ubgold_src (unmodified copy)")

    C.state = None
    if osp.exists(ckpath):
        C.state = torch.load(ckpath, map_location="cpu", weights_only=False)
        print(f"[resume] {ckpath} at epoch {C.state['epoch']}", flush=True)
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(C.state["epoch"] + 1 if C.state is not None else 1),
               t0=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.segments = (list(C.state.get("segments", [])) if C.state is not None else []) + [seg]
    meta_out["segments"] = C.segments
    meta_out["device_all"] = "+".join(sorted({x["device"] for x in C.segments}))
    C.last_save = time.time()
    C.deadline, C.ckpt_every_s, C.ckpath = deadline, ckpt_every_s, ckpath

    RE.add_eval = typed_add_eval     # CHANGE-TYPED (in-process only; harness/run_eval.py itself is not modified)
    fn = {"SIGNET": RE.fit_signet, "OCGTL": RE.fit_ocg, "OCGIN": RE.fit_ocg, "GLocalKD": RE.fit_glocalkd,
          "GLADC": RE.fit_gladc, "CVTGAD": RE.fit_cvtgad, "GOOD-D": RE.fit_goodd,
          "GraphDE": RE.fit_graphde}.get(method, RE.fit_two_stage)
    st = fn(C)
    if st != "done":
        return st

    seg["t1"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ep_seg = args.num_epoch - seg["start_epoch"] + 1
    n_ev_seg = sum(1 for r in C.traj if r["epoch"] >= seg["start_epoch"])
    meta_out.update(seconds=time.time() - t_start, seconds_train_this_segment=C.t_train,
                    seconds_eval_this_segment=C.t_eval, epochs_this_segment=ep_seg,
                    sec_per_epoch_train=C.t_train / max(1, ep_seg), sec_per_eval=C.t_eval / max(1, n_ev_seg),
                    n_eval=len(C.traj), memory=RE.peak_mem(), completed=time.strftime("%Y-%m-%d %H:%M:%S"))
    epochs = [r["epoch"] for r in C.traj]
    assert epochs == sorted(C.eval_epochs), (epochs[:5], len(epochs))
    for k, v in C.S.items():
        assert len(v) == len(epochs), (k, len(v), len(epochs))
    npz = osp.join(SCORES, rid + ".npz")
    extra = dict(y_ALL=y_comb, group_ALL=np.array(grp, dtype=str), hold_n_nodes=np.array([int(d.num_nodes) for d in hold_list]),
                 test_n_nodes_ALL=np.array([int(d.num_nodes) for d in comb]),
                 num_epoch=np.int64(args.num_epoch), eval_freq=np.int64(args.eval_freq))
    for k in keys:
        extra["y__" + skey(k)] = y_type[k]
        extra["idx__" + skey(k)] = idx[k]
    RE.save_scores(npz, epochs, C.S, extra)
    meta_out["scores_npz"] = npz
    res = dict(meta=meta_out, traj=C.traj, train_loss=C.train_loss)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_json)
    if osp.exists(ckpath):
        os.remove(ckpath)
    print(f"[done] {rid} in {meta_out['seconds']:.0f}s -> {out_json}", flush=True)
    return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--generator", required=True, choices=list(TD.GENERATORS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fv", default=None, help="GX: F2 (default) | F1; MS: MS")
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke/timing only (needs --tag and --smoke)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="write under <TYPED_ROOT>/smoke/ (never collected)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TYPED_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    a = ap.parse_args()
    if (a.max_epochs is not None or a.tag) and not a.smoke:
        sys.exit("--max_epochs / --tag are for smoke runs: add --smoke (outputs outside typed/parts)")
    torch.set_num_threads(a.threads)
    try:
        st = run(a.method, a.generator, a.seed, a.fv, deadline=a.deadline, max_epochs=a.max_epochs, tag=a.tag,
                 smoke=a.smoke, ckpt_every_s=a.ckpt_every_s)
    except Exception:
        traceback.print_exc()
        sys.exit(4)          # crash (recorded under <root>/crash); 3 = checkpointed / locked; 0 = done
    sys.exit(0 if st == "done" else 3)
