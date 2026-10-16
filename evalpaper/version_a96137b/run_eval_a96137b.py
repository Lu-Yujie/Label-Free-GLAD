"""A17 version-sensitivity runner: GLocalKD and GLADC at UB-GOLD commit a96137b (NOTES.md §3).

harness/run_eval.py is imported READ-ONLY as a module (never copied, never edited; its sha256 goes into every run's
meta) and its own run()/_run() do everything except training: a96137b data/split code (ast-extracted from the a96137b
tree, asserted against the verified split export), seeds 3407+fold, held-out 20% normals (RandomState(2026+1000*fold)),
canonical test order, evaluation schedule (epoch % eval_freq == 0 plus the final epoch), add_eval (test metrics,
per-graph scores, held-out mean = VAL-NORMAL signal, A15 non-finite recording), score npz, checkpoint cadence, meta.
Overridden in THIS process only (module attributes): SRC -> the patched a96137b tree, PARTS/SCORES/CKPT_ROOT -> the A17
directories, fit_glocalkd / fit_gladc -> the a96137b-faithful loops below, _run -> a wrapper that writes A12 crash
records. Every deviation from UB-GOLD a96137b's fit() is marked "# CHANGE" (all inherited from the c940457 harness
except the epoch-boundary resume, marked "# CHANGE-A17").

a96137b behaviour reproduced (the difference to c940457 / PR #3):
  * every epoch iterates the shuffled train DataLoader and calls process_graph per batch (new composition each epoch);
  * every evaluation iterates the shuffled test DataLoader and calls process_graph per graph;
  * forward_model moves tensors with .to(self.device) (no hard-coded 'cuda').
Resume (no replay): checkpoints only at epoch boundaries (model, optimizer, StepLR, python/numpy/torch/CUDA RNG states,
trajectory, scores) plus a fingerprint of the next epoch's batch composition, recomputed and asserted on resume (the
DataLoader position is the epoch counter plus the restored global CPU RNG state; NOTES.md §3.5).
Logs never show test metrics (harness verbose off; own progress line: train loss, held-out mean, timing).

Usage (ubgold-gpu env):
  PYTHONDONTWRITEBYTECODE=1 python -B run_eval_a96137b.py --method GLADC --dataset BZR --fold 0 --require_device cuda
  smoke: add --out_root smoke --max_epochs 8 --tag smoke   (outputs under $GLAD_DATA_ROOT/evalpaper_a96137b/smoke)
Exit codes: 0 done, 3 checkpointed/locked, 5 device mismatch or failed precondition (no crash record), else crash.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
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
import types  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
PROJECT = PRELIM   # release: the repository root
HARNESS = osp.join(EVALPAPER, "harness")
RUN_EVAL = osp.join(HARNESS, "run_eval.py")
UBGOLD_COMMIT = "a96137b0f67dbb1f97cb1867c77849dd1bf5032f"
SRC_A96 = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_versions/src_a96137b")
DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_a96137b")
GRID_PARTS = osp.join(HERE, "parts")
METHODS = ("GLocalKD", "GLADC")
OWN_FILES = {"version_a96137b/run_eval_a96137b.py": osp.join(HERE, "run_eval_a96137b.py"),
             "version_a96137b/run_pool_a96137b.py": osp.join(HERE, "run_pool_a96137b.py"),
             "version_a96137b/make_src.py": osp.join(HERE, "make_src.py"),
             "harness/run_eval.py": RUN_EVAL}


def roots(out_root):
    """(parts, scores, ckpt, crash)."""
    if out_root == "grid":
        return GRID_PARTS, osp.join(DATA_ROOT, "scores"), osp.join(DATA_ROOT, "ckpt"), osp.join(DATA_ROOT, "crash")
    assert out_root == "smoke", out_root
    b = osp.join(DATA_ROOT, "smoke")
    return osp.join(b, "parts"), osp.join(b, "scores"), osp.join(b, "ckpt"), osp.join(b, "crash")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# source-tree verification and provenance (computed once per process)
# ----------------------------------------------------------------------------------------------------------------
_PROV = {}
_TREE_OK = []


def verify_tree():
    """COMMIT, PATCHES.json, patched-file hashes and the whole-tree digest (standard library only)."""
    if _TREE_OK:
        return _TREE_OK[0]
    commit = open(osp.join(SRC_A96, "COMMIT")).read().split()[0]
    assert commit == UBGOLD_COMMIT, commit
    P = json.load(open(osp.join(SRC_A96, "PATCHES.json")))
    assert P["commit"] == UBGOLD_COMMIT
    digest = tree_digest(SRC_A96)
    assert digest == P["tree_sha256_patched"], "a96137b tree differs from PATCHES.json (modified after make_src.py?)"
    for r in P["patched_files"]:
        assert sha256_file(osp.join(SRC_A96, r["file"])) == r["sha256_patched"], r["file"]
    _TREE_OK.append((P, digest))
    return P, digest


def tree_digest(root):
    """Same digest as make_src.py (every file except COMMIT / PATCHES.json)."""
    h = hashlib.sha256()
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = osp.join(dp, f)
            rel = osp.relpath(p, root)
            if rel in ("COMMIT", "PATCHES.json"):
                continue
            h.update(rel.encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def provenance():
    """Tree verification + code hashes + versions. Call only after the harness overlay is on sys.path (inside a fit
    function): torch_geometric must be imported first by the harness's _run, as in every harness run."""
    if _PROV:
        return _PROV
    assert "torch_geometric" in sys.modules, "provenance() called before the harness imported torch_geometric"
    P, digest = verify_tree()
    code = {k: sha256_file(v) for k, v in OWN_FILES.items() if osp.exists(v)}
    for dp, dn, fn in os.walk(SRC_A96):
        dn.sort()
        for f in sorted(fn):
            if f.endswith((".py", ".sh")):
                p = osp.join(dp, f)
                code["ubgold_a96137b/" + osp.relpath(p, SRC_A96)] = sha256_file(p)
    vers = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    for mod in ("torch_geometric", "scipy", "sklearn", "networkx", "torch_scatter"):
        try:
            vers[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception as e:   # pragma: no cover
            vers[mod] = f"unavailable: {type(e).__name__}"
    try:
        head = subprocess.run(["git", "-C", PROJECT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:
        head = None
    _PROV.update(ubgold_commit=UBGOLD_COMMIT, ubgold_src=SRC_A96, ubgold_tree_sha256=digest,
                 ubgold_patches=P["patched_files"], ubgold_tree_sha256_pristine=P["tree_sha256_pristine"],
                 code_sha256=code, versions=vers, project_git_head=head)
    return _PROV


# ----------------------------------------------------------------------------------------------------------------
# harness module (read-only import) and configuration
# ----------------------------------------------------------------------------------------------------------------
_RE = None
_ORIG_RUN_CODE = None
_CFG = dict(require_device=None, crash_dir=None)


def load_harness():
    if "ep_harness_run_eval" in sys.modules:
        return sys.modules["ep_harness_run_eval"]
    spec = importlib.util.spec_from_file_location("ep_harness_run_eval", RUN_EVAL)
    RE = importlib.util.module_from_spec(spec)
    sys.modules["ep_harness_run_eval"] = RE
    spec.loader.exec_module(RE)
    return RE


def configure(out_root="grid", require_device=None):
    """Point the imported harness at the a96137b tree and the A17 directories; install the A17 fit functions."""
    global _RE, _ORIG_RUN_CODE
    RE = load_harness()
    parts, scores, ckpt, crash = roots(out_root)
    RE.SRC = SRC_A96
    RE.PARTS, RE.SCORES, RE.CKPT_ROOT = parts, scores, ckpt
    RE.fit_glocalkd = fit_glocalkd_a96
    RE.fit_gladc = fit_gladc_a96
    if _ORIG_RUN_CODE is None:
        orig = RE._run
        _ORIG_RUN_CODE = orig.__code__

        def _run_with_crash_record(method, DS, fold, hp, run_id, out_json, ckpath, *rest):
            try:
                return orig(method, DS, fold, hp, run_id, out_json, ckpath, *rest)
            except Exception as e:
                record_crash(_CFG["crash_dir"], run_id, e, ckpath)
                raise
        RE._run = _run_with_crash_record
    _CFG.update(require_device=require_device, crash_dir=crash, out_root=out_root)
    _RE = RE
    return RE


def record_crash(crash_dir, rid, e, ckpath):
    """A12 bookkeeping (convention of typed/run_typed.py): crash record + checkpoint moved aside, so the next attempt
    starts from epoch 1."""
    os.makedirs(crash_dir, exist_ok=True)
    k = 1 + sum(1 for f in os.listdir(crash_dir) if f.startswith(rid + ".attempt"))
    tb = traceback.format_exc()
    msg = f"{type(e).__name__}: {e}"
    low = (msg + tb[-2000:]).lower()
    rec = dict(run_id=rid, attempt=k, time=time.strftime("%Y-%m-%d %H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"),
               host=platform.node(), ubgold_commit=UBGOLD_COMMIT, error=msg[:2000], traceback=tb[-8000:],
               nonfinite_suspected=any(w in low for w in ("nan", "inf", "non-finite", "nonfinite")))
    if osp.exists(ckpath):
        os.replace(ckpath, ckpath + f".crash{k}")
        rec["checkpoint_moved_to"] = ckpath + f".crash{k}"
    with open(osp.join(crash_dir, f"{rid}.attempt{k}.json"), "w") as f:
        json.dump(rec, f, indent=1)


def harness_meta_out():
    """The harness _run's own meta_out dict (same object; keys added here are written by _run)."""
    f = sys._getframe(1)
    while f is not None and f.f_code is not _ORIG_RUN_CODE:
        f = f.f_back
    assert f is not None, "fit function not called from harness _run"
    m = f.f_locals["meta_out"]
    assert isinstance(m, dict) and "run_id" in m
    return m


# ----------------------------------------------------------------------------------------------------------------
# shared A17 pieces
# ----------------------------------------------------------------------------------------------------------------
def loader_probe(loader):
    """Fingerprint of the NEXT iteration of the shuffled train DataLoader (fit-set positions per batch), computed inside
    rng_isolated(): no RNG state changes. Iterating draws exactly what the next epoch's iteration will draw."""
    with _RE.rng_isolated():
        order = [b.idx.view(-1).tolist() for b in loader]
    return hashlib.sha256(json.dumps(order).encode()).hexdigest()


def a17_start(C):
    """Meta injection, device checks, resume bookkeeping. Returns (meta_out, status or None)."""
    meta = harness_meta_out()
    meta.update(provenance())
    meta["a17"] = dict(planned_device=_CFG["require_device"], runner="prelim/evalpaper/version_a96137b/run_eval_a96137b.py",
                       loop="a96137b fit(): per-epoch shuffled train DataLoader + process_graph per batch; per-eval "
                            "shuffled test DataLoader + process_graph",
                       resume="epoch-boundary checkpoint + next-epoch loader fingerprint (no replay)",
                       probe_checks=[], out_root=_CFG.get("out_root"))
    if _CFG["require_device"] is not None and C.device.type != _CFG["require_device"]:
        print(f"[device] planned {_CFG['require_device']} but running on {C.device.type}; not started", flush=True)
        return meta, "device_mismatch"
    if C.state is not None:
        st = C.state
        if st.get("a17", {}).get("ubgold_commit") != UBGOLD_COMMIT:
            print("[resume] checkpoint is not an A17 a96137b checkpoint; refusing", flush=True)
            return meta, "device_mismatch"
        prev = {s["device"] for s in st.get("segments", [])}
        if prev and prev != {C.device.type}:
            print(f"[resume] checkpoint segments ran on {prev}, now {C.device.type}; refusing cross-device resume",
                  flush=True)
            return meta, "device_mismatch"
    return meta, None


def a17_resume_check(C, meta):
    """After the RNG restore: the next epoch's batch composition must equal the one recorded at the checkpoint."""
    got = loader_probe(C.dataloader)
    want = C.state["a17"]["probe_next"]
    ok = got == want
    meta["a17"]["probe_checks"].append(dict(resumed_after_epoch=int(C.state["epoch"]), ok=bool(ok)))
    C.segments[-1]["a17_loader_fingerprint_ok"] = bool(ok)   # kept in every later checkpoint and in meta.segments
    print(f"[resume] epoch {C.state['epoch']}: next-epoch loader fingerprint {'matches' if ok else 'MISMATCH'}",
          flush=True)
    assert ok, "DataLoader position after resume differs from the checkpoint (RNG restore incomplete?)"


def a17_payload(C, extra):
    """CHANGE-A17: checkpoint content = models/optimizer (extra) + the next-epoch loader fingerprint."""
    d = dict(extra)
    d["a17"] = dict(ubgold_commit=UBGOLD_COMMIT, probe_next=loader_probe(C.dataloader), device=C.device.type)
    return d


def progress(C, epoch, t_pg, extra=""):
    r = C.traj[-1]
    print(f"[{C.run_id}] ep {epoch} train {r['train_loss']:.6g} hold {r['holdout_score']:.6g} "
          f"pg_train {t_pg:.1f}s t={time.time() - C.t_start:.0f}s{extra}", flush=True)


# =============================================================================================
# GLocalKD at a96137b: GAOOD/detector/GLocalKD.py fit()
# =============================================================================================
def fit_glocalkd_a96(C):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch_geometric.data import Batch
    meta, st = a17_start(C)
    if st:
        return st
    RE = _RE
    args, device, M = C.args, C.device, C.M
    G = M["glocalkd"]
    GK = importlib.import_module("GAOOD.detector.GLocalKD").GLocalKD      # a96137b class (patched mybase)
    assert osp.realpath(sys.modules["GAOOD.detector.GLocalKD"].__file__).startswith(osp.realpath(SRC_A96))
    # a96137b init_model(**self.kwargs) with kwargs == {}: teacher, then student, .to(self.device)
    model_teacher = G.GcnEncoderGraph_teacher(input_dim=args.dataset_num_features, hidden_dim=args.hidden_dim,
                                              embedding_dim=args.output_dim, label_dim=2,
                                              num_layers=args.num_layer, bn=args.bn, args=args).to(device)
    model_student = G.GcnEncoderGraph_student(input_dim=args.dataset_num_features, hidden_dim=args.hidden_dim,
                                              embedding_dim=args.output_dim, label_dim=2,
                                              num_layers=args.num_layer, bn=args.bn, args=args).to(device)
    # stand-in `self` for the a96137b methods process_graph (max_nodes_num) and forward_model (device, models)
    selfobj = types.SimpleNamespace(max_nodes_num=args.max_nodes_num, device=device,
                                    model_teacher=model_teacher, model_student=model_student)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    preprocessed_hold = []   # CHANGE (harness): held-out normals, one graph at a time like the test set (no RNG)
    for g in C.hold_list:
        adj_matrixs, _, graph_appends, graph_label = GK.process_graph(selfobj, Batch.from_data_list([g]))
        preprocessed_hold.append(tuple([adj_matrixs, graph_appends, graph_label]))
    start_epoch = 1
    if C.state is not None:   # CHANGE-A17: epoch-boundary resume (no replay)
        model_student.load_state_dict(C.state["student"]); model_teacher.load_state_dict(C.state["teacher"])
        optimizer.load_state_dict(C.state["opt"]); scheduler.load_state_dict(C.state["sched"])
        start_epoch = RE.restore_common(C)
        RE.set_rng_state(C.state["rng"])
        a17_resume_check(C, meta)
    payload = lambda: a17_payload(C, dict(student=model_student.state_dict(), teacher=model_teacher.state_dict(),
                                          opt=optimizer.state_dict(), sched=scheduler.state_dict()))

    def score_one(adj_matrixs, graph_appends):
        # a96137b fit() evaluation block (per graph; batch_size_test == 1)
        adj = Variable(adj_matrixs.float(), requires_grad=False).to(selfobj.device)
        h0 = Variable(graph_appends.float(), requires_grad=False).to(selfobj.device)
        with torch.no_grad():   # CHANGE (harness): UB-GOLD scores with autograd on (identical values)
            embed_node, embed = model_student(h0, adj)
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
        loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1)
        loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
        loss_ = loss_graph + loss_node
        return np.array(loss_.cpu().detach())

    tm = dict(pg_train_s=0.0, pg_eval_s=0.0, epochs=0, evals=0)
    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        t_pg = 0.0
        total_loss = 0.0
        nb = 0
        model_student.train()
        for batch_idx, data in enumerate(C.dataloader):        # a96137b: reshuffled every epoch
            model_student.zero_grad()
            tp = time.time()
            adj, _, h0, _ = GK.process_graph(selfobj, data)    # a96137b: per batch, every epoch
            t_pg += time.time() - tp
            loss = GK.forward_model(selfobj, adj, h0)          # a96137b: .to(self.device)
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()   # CHANGE (harness): float accumulator (a96137b sums the tensor; same value)
            nb += 1
        C.train_loss.append(total_loss / max(nb, 1))
        C.t_train += time.time() - t0
        tm["pg_train_s"] += t_pg
        tm["epochs"] += 1
        if epoch in C.eval_epochs:   # UB-GOLD: epoch % eval_freq == 0 (+ CHANGE (harness): final epoch)
            t1 = time.time()
            model_student.eval()
            loss, y, gids = [], [], []
            tp = time.time()
            for batch_idx, data in enumerate(C.dataloader_val):   # a96137b: shuffled test loader, process_graph per graph
                adj_matrixs, _, graph_appends, graph_label = GK.process_graph(selfobj, data)
                loss.append(score_one(adj_matrixs, graph_appends))
                y.append(1 if int(graph_label) != 0 else 0)
                gids += data.gid.cpu().tolist()                  # CHANGE (harness): canonical test order
            tm["pg_eval_s"] += time.time() - tp
            tm["evals"] += 1
            test = C.unscramble(np.array(loss).ravel(), gids, y)
            with RE.rng_isolated():                               # CHANGE (harness): VAL-NORMAL signal
                sh = np.array([score_one(a, g) for a, g, _ in preprocessed_hold]).ravel()
            RE.add_eval(C, epoch, test, sh, t_eval0=t1)
            progress(C, epoch, t_pg)
        elif time.time() - t0 > 30:
            print(f"[{C.run_id}] ep {epoch} train {C.train_loss[-1]:.6g} pg_train {t_pg:.1f}s "
                  f"t={time.time() - C.t_start:.0f}s", flush=True)
        # CHANGE (harness): no early stopping and no torch.save of the best-test-AUROC model
        meta["a17_timing_this_segment"] = dict(tm)
        if RE.maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# =============================================================================================
# GLADC at a96137b: GAOOD/detector/GLADC.py fit()
# =============================================================================================
def fit_gladc_a96(C):
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch_geometric.data import Batch
    meta, st = a17_start(C)
    if st:
        return st
    RE = _RE
    args, device = C.args, C.device
    det = C.M["init_model"](args)                 # UB-GOLD benchmark/utils.py init_model (a96137b GLADC class)
    assert osp.realpath(sys.modules[type(det).__module__].__file__).startswith(osp.realpath(SRC_A96))
    det.device = device                           # fit(): torch.device('cuda:'+str(args.gpu) if cuda else 'cpu')
    det.NetGe, det.noise_NetG = det.init_model(**det.kwargs)
    optimizerG = torch.optim.Adam(det.NetGe.parameters(), lr=det.lr)
    preprocessed_hold = []   # CHANGE (harness): held-out normals, one graph at a time like the test set (no RNG)
    for g in C.hold_list:
        adj_matrixs, _, graph_appends, graph_label = det.process_graph(Batch.from_data_list([g]))
        preprocessed_hold.append(tuple([adj_matrixs, graph_appends, graph_label]))
    hold_obj_batches = None  # CHANGE (harness): secondary signal, training objective on held-out graphs (if cheap)
    if args.max_nodes_num <= 1000:
        hold_obj_batches = []
        for ch in RE.chunks(C.hold_list, det.batch_size):
            a, al, ga, _ = det.process_graph(Batch.from_data_list(ch))
            hold_obj_batches.append((a, al, ga))
    start_epoch = 1
    if C.state is not None:   # CHANGE-A17: epoch-boundary resume (no replay)
        det.NetGe.load_state_dict(C.state["NetGe"]); det.noise_NetG.load_state_dict(C.state["noise_NetG"])
        optimizerG.load_state_dict(C.state["opt"])
        start_epoch = RE.restore_common(C)
        RE.set_rng_state(C.state["rng"])
        a17_resume_check(C, meta)
    payload = lambda: a17_payload(C, dict(NetGe=det.NetGe.state_dict(), noise_NetG=det.noise_NetG.state_dict(),
                                          opt=optimizerG.state_dict()))

    def score_one(adj_matrixs, graph_appends):
        # a96137b fit() evaluation block (per graph; batch_size_test == 1)
        adj = Variable(adj_matrixs.float().float(), requires_grad=False).to(det.device)
        h0 = Variable(graph_appends.float(), requires_grad=False).to(det.device)
        with torch.no_grad():   # CHANGE (harness): UB-GOLD scores with autograd on (identical values)
            x1_r, Feat_0 = det.NetGe.shared_encoder(h0, adj)
            x_fake, s_fake, x2, Feat_1 = det.NetGe(x1_r, adj)
            loss_node = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
            loss_ = loss_node + loss_graph
        return np.array(loss_.cpu().detach())

    tm = dict(pg_train_s=0.0, pg_eval_s=0.0, epochs=0, evals=0)
    for epoch in range(start_epoch, det.num_epochs + 1):
        t0 = time.time()
        t_pg = 0.0
        total_lossG = 0.0
        nb = 0
        det.NetGe.train()
        for batch_idx, data in enumerate(C.dataloader):         # a96137b: reshuffled every epoch
            tp = time.time()
            adj_matrixs, adj_labels, graph_appends, _ = det.process_graph(data)   # a96137b: per batch, every epoch
            t_pg += time.time() - tp
            lossG = det.forward_model(adj_matrixs, adj_labels, graph_appends)
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            total_lossG += lossG.detach()   # CHANGE (harness): detach (a96137b keeps every graph alive; same value)
            nb += 1
        C.train_loss.append(float(total_lossG) / max(1, nb))
        C.t_train += time.time() - t0
        tm["pg_train_s"] += t_pg
        tm["epochs"] += 1
        if epoch in C.eval_epochs:   # UB-GOLD: (epoch) % args.eval_freq == 0 (+ CHANGE (harness): final epoch)
            t1 = time.time()
            det.NetGe.eval()
            loss, y, gids = [], [], []
            tp = time.time()
            for batch_idx, data in enumerate(C.dataloader_val):   # a96137b: shuffled test loader, process_graph per graph
                adj_matrixs, _, graph_appends, graph_label = det.process_graph(data)
                loss.append(score_one(adj_matrixs, graph_appends))
                y.append(1 if int(graph_label) != 0 else 0)
                gids += data.gid.cpu().tolist()                  # CHANGE (harness): canonical test order
            tm["pg_eval_s"] += time.time() - tp
            tm["evals"] += 1
            test = C.unscramble(np.array(loss).ravel(), gids, y)
            rec = {}
            with RE.rng_isolated():                               # CHANGE (harness): VAL-NORMAL signal (+ objective)
                sh = np.array([score_one(a, g) for a, g, _ in preprocessed_hold]).ravel()
                if hold_obj_batches is not None:
                    with torch.no_grad():
                        objs = [float(det.forward_model(a, al, ga)) * a.shape[0] for a, al, ga in hold_obj_batches]
                    rec["holdout_obj"] = sum(objs) / C.n_hold
            RE.add_eval(C, epoch, test, sh, rec_extra=rec, t_eval0=t1)
            progress(C, epoch, t_pg)
        elif time.time() - t0 > 30:
            print(f"[{C.run_id}] ep {epoch} train {C.train_loss[-1]:.6g} pg_train {t_pg:.1f}s "
                  f"t={time.time() - C.t_start:.0f}s", flush=True)
        # CHANGE (harness): no early stopping, no torch.save of the best-test-AUROC model
        meta["a17_timing_this_segment"] = dict(tm)
        if RE.maybe_stop(C, epoch, payload):
            return "incomplete"
    return "done"


# ----------------------------------------------------------------------------------------------------------------
def run(method, DS, fold, require_device, out_root="grid", deadline=None, max_epochs=None, ckpt_every_s=300.0,
        tag=""):
    assert method in METHODS, method
    if out_root == "grid" and (tag or max_epochs is not None):
        raise SystemExit("tagged / --max_epochs runs must use --out_root smoke (never the grid parts directory)")
    have = "cuda" if torch.cuda.is_available() else "cpu"
    if require_device is not None and have != require_device:
        print(f"[device] task planned on {require_device}, this process sees {have}; not started", flush=True)
        return "device_mismatch"
    verify_tree()                    # the a96137b tree must match PATCHES.json before anything runs
    RE = configure(out_root, require_device)
    return RE.run(method, DS, fold, hp="ubgold", deadline=deadline, max_epochs=max_epochs, ckpt_every_s=ckpt_every_s,
                  tag=tag, verbose=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--require_device", choices=["cpu", "cuda"], default=None,
                    help="planned device (A17 device plan); mismatch -> exit 5 before any training")
    ap.add_argument("--out_root", choices=["grid", "smoke"], default="grid")
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke tests only (with --tag, --out_root smoke)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EP_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    a = ap.parse_args()
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag (smoke runs must not be mistaken for real runs)")
    torch.set_num_threads(a.threads)
    st = run(a.method, a.dataset, a.fold, a.require_device, out_root=a.out_root, deadline=a.deadline,
             max_epochs=a.max_epochs, ckpt_every_s=a.ckpt_every_s, tag=a.tag)
    sys.exit(0 if st == "done" else (5 if st == "device_mismatch" else 3))
