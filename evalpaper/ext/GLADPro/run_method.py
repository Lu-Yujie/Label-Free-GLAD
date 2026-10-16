"""E5 extension (PREREG_AMENDMENTS A10): instrumented GLADPro (KDD'25) runs on UB-GOLD folds.

Same CLI and per-run JSON "part" schema as evalpaper/harness/run_eval.py:
  PYTHONDONTWRITEBYTECODE=1 python -B run_method.py --method GLADPro --dataset BZR --fold 0 [--hp hpfree]

* GLADPro's training and scoring code is the official repository at commit 1f811b9 (src/, byte-identical).
  The training loop below is a line-by-line copy of src/main.py:run_split(); every change is marked
  "# CHANGE" and listed in NOTES.md (D1-D7).
* Folds, features, seeds and the held-out normals come from the harness itself (its run_eval.py is imported,
  never modified): UB-GOLD get_ad_split_TU / get_ad_dataset_TU, asserted against
  $GLAD_DATA_ROOT/ubgold_splits, set_seed(3407+fold), RandomState(2026+1000*fold) carve-out.
* hp 'ubgold' = the authors' released README configuration (possibly test-tuned; modal line where none
  exists); hp 'hpfree' = repository argparse defaults (NOTES.md, pre-registered).
* GLADPro evaluates on the test graphs after every epoch, so every epoch is an evaluation point
  (meta.hparams.eval_freq = 1, num_epoch = epochs).

Outputs (EP_PARTS / EP_SCORES / EP_CKPT override):
  evalpaper/ext/parts/<run_id>.json                         meta + per-epoch metrics
  $GLAD_DATA_ROOT/ext_gladpro/scores/<run_id>.npz        per-graph test / held-out scores
  $GLAD_DATA_ROOT/ext_gladpro/ckpt/<DS>/<run_id>.pt      resumable state (removed when done)
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # never import from cwd

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import os.path as osp
import re
import shlex
import sys
import time
import types
from collections import Counter

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
GP_SRC = osp.join(HERE, "src")                                     # GLADPro @ 1f811b9 (unmodified)
GP_COMMIT = "1f811b9a5b7292d925bf09aa0ab93a2038c3542c"
HARNESS = osp.join(osp.dirname(osp.dirname(HERE)), "harness")      # evalpaper/harness
ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_gladpro")
CKPT_ROOT = os.environ.get("EP_CKPT", osp.join(ROOT, "ckpt"))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
SCORES = os.environ.get("EP_SCORES", osp.join(ROOT, "scores"))
METHOD = "GLADPro"

TU_LINES = ("MUTAG", "PROTEINS", "DD", "IMDB-BINARY")   # README lines for real TU datasets (modal rule)
README_DS_MAP = {"PROTEINS_full": "PROTEINS"}           # NOTES.md: same graphs + same 3 UB-GOLD features
FORCE_DELPROTO_EPOCH = int(os.environ.get("EP_TEST_FORCE_DELPROTO_EPOCH", "0"))   # resume tests only
# HP-FREE crash fallback (A10 §3 / A1 rule 3): options whose default crashes -> modal released value
HPFREE_CRASH_FALLBACK = {}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_harness():
    """Import evalpaper/harness/run_eval.py as a module (read-only use of its data/RNG/IO helpers)."""
    spec = importlib.util.spec_from_file_location("ep_harness_run_eval", osp.join(HARNESS, "run_eval.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    return H


def load_ubgold_metric(H):
    """UB-GOLD's GAOOD/metric functions, imported as the harness does (no GAOOD/__init__)."""
    if H.SRC not in sys.path:
        sys.path.append(H.SRC)            # appended: GLADPro's own top-level modules keep precedence
    H.stub_package("GAOOD", osp.join(H.SRC, "GAOOD"))
    importlib.import_module("GAOOD.metric")
    return importlib.import_module("GAOOD.metric.metric")


def load_gladpro():
    """Import GLADPro's utils/layers/model from src/ and apply the runtime compatibility overrides."""
    # CHANGE (D6): matplotlib is not installed; model.py imports it only for plot_fig() (never called)
    for name in ("matplotlib", "matplotlib.pyplot"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if GP_SRC not in sys.path:
        sys.path.insert(0, GP_SRC)
    utils = importlib.import_module("utils")
    layers = importlib.import_module("layers")
    model = importlib.import_module("model")
    for mod in (utils, layers, model):
        assert osp.realpath(mod.__file__).startswith(osp.realpath(GP_SRC)), mod.__file__
    from torch_geometric.nn.conv import MessagePassing

    # CHANGE (D1): PyG >= 2.5 compiles propagate() from the '# propagate_type: (x: OptPairTensor)' comment
    # GLADPro copied into GINConv.forward, which drops the edge_atten kwarg (TypeError). A class-level
    # propagate makes PyG skip compilation and use its generic MessagePassing.propagate (PyG 2.2's code path).
    def _generic_propagate(self, edge_index, size=None, **kwargs):
        return MessagePassing.propagate(self, edge_index, size=size, **kwargs)
    layers.GINConv.propagate = _generic_propagate

    nn = torch.nn

    # copy of model.py GAD_Explainer.prototype_distances
    def prototype_distances(self, x, x_2):

        #distance = torch.cdist(x,x_2,p=2)**2
        #similarity = torch.log( (distance + 1) / (distance + self.epsilon) )

        x1_abs = x.norm(dim=1)
        # CHANGE (D2): was `x1_abs[torch.nonzero(x1_abs == 0)] += self.epsilon` (in-place on the saved norm
        # output -> torch>=2 autograd error); out-of-place, identical values
        x1_abs = torch.where(x1_abs == 0, x1_abs + self.epsilon, x1_abs)

        x2_abs = x_2.norm(dim=1)
        x2_abs = torch.where(x2_abs == 0, x2_abs + self.epsilon, x2_abs)   # CHANGE (D2)

        similarity = torch.einsum('ik,jk->ij', x, x_2) / torch.einsum('i,j->ij', x1_abs, x2_abs)

        if self.training:
            return nn.Softmax(dim=-1)(similarity)
        else:
            return similarity
    model.GAD_Explainer.prototype_distances = prototype_distances
    return utils, layers, model


# ----------------------------------------------------------------------------------------------
# hyperparameters (NOTES.md, pre-registered 2026-10-13 01:18:23 EDT)
# ----------------------------------------------------------------------------------------------
def readme_lines():
    out = {}
    for raw in open(osp.join(GP_SRC, "README.md")):
        s = raw.strip()
        if s.startswith("python3 main.py"):
            toks = shlex.split(s)[2:]
            ds = toks[toks.index("--dataset") + 1] if "--dataset" in toks else "mutagen"
            out[ds] = (s, toks)
    return out


def parse_gladpro_args(utils, toks):
    """GLADPro's own argparse (utils.load_args), verbatim, on the given tokens."""
    saved = sys.argv
    try:
        sys.argv = ["main.py"] + list(toks)
        return utils.load_args()
    finally:
        sys.argv = saved


def modal_tokens(utils):
    """Per-option mode over the README lines of real TU datasets; ties -> smaller value (A1 rule)."""
    lines = readme_lines()
    dflt = vars(parse_gladpro_args(utils, []))
    parsed = [vars(parse_gladpro_args(utils, lines[d][1])) for d in TU_LINES]
    toks, chosen = [], {}
    for k, v0 in dflt.items():
        if k in ("dataset", "device"):
            continue
        cnt = Counter(p[k] for p in parsed)
        top = max(cnt.values())
        mv = sorted([v for v, c in cnt.items() if c == top])[0]
        if mv != v0:
            toks += [f"--{k}", str(mv)]
            chosen[k] = mv
    return toks, chosen


def resolve_hparams(utils, DS, hp):
    lines = readme_lines()
    notes = []
    if hp == "hpfree":
        args = parse_gladpro_args(utils, [])
        line = "HPFREE GLADPro argparse defaults (src/utils.py load_args)"
        for k in HPFREE_CRASH_FALLBACK.get(DS, []):
            raise NotImplementedError(k)
    else:
        assert hp == "ubgold", hp
        key = README_DS_MAP.get(DS, DS)
        if key in lines:
            line, toks = lines[key]
            toks = [t for i, t in enumerate(toks) if not (t == "--dataset" or (i > 0 and toks[i - 1] == "--dataset"))]
            if key != DS:
                notes.append(f"README '{key}' line used for {DS} (NOTES.md: same graphs and same UB-GOLD features)")
        else:
            toks, chosen = modal_tokens(utils)
            line = "MODAL README config over " + ",".join(TU_LINES) + ": python3 main.py " + " ".join(toks)
            notes.append(f"no README line for {DS}: modal released configuration {chosen}")
        args = parse_gladpro_args(utils, toks)
    args.dataset = DS
    return args, line, notes


# ----------------------------------------------------------------------------------------------
def load_fold(H, DS, fold, batch_size, batch_size_test, holdout_frac=0.2):
    """CHANGE (D3, D4): UB-GOLD trial `fold` exactly as the harness: set_seed(3407+fold), then UB-GOLD's own
    get_ad_split_TU / get_ad_dataset_TU (asserted against ubgold_splits), then the 20% held-out carve-out
    RandomState(2026+1000*fold). Returns the 80% fit list, held-out list and the test list (UB-GOLD order)."""
    ns = H.load_ubgold_data_functions()
    ns["set_seed"](3407 + fold)
    assert not DS.startswith("Tox21"), "Tox21 not supported by this adapter"
    uargs = types.SimpleNamespace(DS=DS, num_trial=5, batch_size=batch_size, batch_size_test=batch_size_test)
    cwd = os.getcwd()
    os.chdir(H.DATA_RUN)
    try:
        splits = ns["get_ad_split_TU"](uargs, fold=uargs.num_trial)
        out = ns["get_ad_dataset_TU"](uargs, splits[fold], need_str_enc=False)
    finally:
        os.chdir(cwd)
    (dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta) = out
    ref = json.load(open(osp.join(H.SPLITS, f"{DS}.json")))
    rf = ref["folds"][fold]
    assert list(map(int, splits[fold][0])) == list(map(int, rf["split_train_index"]))
    assert list(map(int, splits[fold][1])) == list(map(int, rf["split_test_index"]))
    train_all = list(dataloader.dataset)
    n_all = len(train_all)
    n_hold = int(round(holdout_frac * n_all))
    perm = np.random.RandomState(H.HOLDOUT_SEED + 1000 * fold).permutation(n_all)
    hold_pos = sorted(perm[:n_hold].tolist())
    fit_pos = sorted(perm[n_hold:].tolist())
    assert n_hold > 0
    test_list = list(dataloader_test.dataset)            # canonical (UB-GOLD split) order; y = 1 anomaly
    y_test = np.array([int(d.y) for d in test_list], dtype=np.int64)
    assert list(map(int, y_test)) == list(map(int, rf["y_test"])), "test labels differ from ubgold_splits"
    return dict(fit_list=[train_all[i] for i in fit_pos], hold_list=[train_all[i] for i in hold_pos],
                test_list=test_list, y_test=y_test, n_all=n_all, n_hold=n_hold, hold_pos=hold_pos, fit_pos=fit_pos,
                split_test_index=list(map(int, splits[fold][1])),
                split_train_index=list(map(int, splits[fold][0])), meta=meta)


class Ctx:
    pass


def run(DS, fold, hp="ubgold", holdout_frac=0.2, deadline=None, max_epochs=None, ckpt_every_s=300.0, tag="",
        verbose=True):
    t_start = time.time()
    run_id = f"{METHOD}_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + (f"_{tag}" if tag else "")
    os.makedirs(PARTS, exist_ok=True)
    out_json = osp.join(PARTS, run_id + ".json")
    if osp.exists(out_json):
        print(f"[skip] {out_json} exists")
        return "done"
    H = load_harness()
    ckdir = osp.join(CKPT_ROOT, DS)
    os.makedirs(ckdir, exist_ok=True)
    ckpath = osp.join(ckdir, run_id + ".pt")
    lock = ckpath + ".lock"
    if not H.acquire_lock(lock):
        print(f"[locked] {run_id} is running in another job")
        return "locked"
    try:
        return _run(H, DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s,
                    verbose, t_start)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _run(H, DS, fold, hp, run_id, out_json, ckpath, holdout_frac, deadline, max_epochs, ckpt_every_s, verbose,
         t_start):
    metric = load_ubgold_metric(H)
    utils, layers, model_mod = load_gladpro()
    deterministic = not torch.cuda.is_available()
    if deterministic:
        # CHANGE (D8): on CPU, the backward of node_bern[edge_index[...]] (index_put_ accumulate) uses parallel
        # atomic adds -> run-to-run float differences that GLADPro's argmax prototype assignment amplifies.
        # Deterministic mode only fixes the summation order (bit-reproducible runs and exact resume).
        torch.use_deterministic_algorithms(True, warn_only=True)
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn.models.autoencoder import GAE

    args, hp_line, hp_notes = resolve_hparams(utils, DS, hp)
    if max_epochs is not None:          # smoke / timing / resume tests only (tag required by the CLI)
        args.epochs = max_epochs
    device = args.device                # GLADPro: cuda:<device> if available else cpu
    seed = 3407 + fold
    F = load_fold(H, DS, fold, args.batch_size, args.batch_size_test, holdout_frac)
    train_graphs, hold_list, test_graphs, y_test = F["fit_list"], F["hold_list"], F["test_list"], F["y_test"]
    n_all, n_hold, n_test, meta = F["n_all"], F["n_hold"], len(test_graphs), F["meta"]
    split_test_index, hold_pos = F["split_test_index"], F["hold_pos"]

    meta_out = dict(method=METHOD, dataset=DS, fold=fold, hp=hp, seed=seed, run_id=run_id,
                    ubgold_sh_line=hp_line, hp_notes=hp_notes,
                    hp_meaning=("released README configuration (possibly test-tuned)" if hp == "ubgold"
                                else "repository argparse defaults"),
                    hparams=dict({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                  for k, v in sorted(vars(args).items())},
                                 num_epoch=int(args.epochs), eval_freq=1),
                    n_train_normal_ubgold=n_all, n_fit=len(train_graphs), n_holdout=n_hold,
                    holdout_seed=H.HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=int(y_test.sum()),
                    max_nodes_num=int(meta["max_nodes_num"]), num_feat=int(meta["num_feat"]), device=str(device),
                    torch=torch.__version__, cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                    threads=torch.get_num_threads(), max_epochs_override=max_epochs,
                    gladpro_commit=GP_COMMIT, harness_run_eval_sha256=sha256(osp.join(HARNESS, "run_eval.py")),
                    adapter_sha256=sha256(osp.abspath(__file__)),
                    deterministic_algorithms=deterministic,
                    impl="GLADPro src@1f811b9 + runtime overrides D1 (generic propagate), D2 (out-of-place norm eps), "
                         "D6 (matplotlib stub), D8 (deterministic algorithms on CPU); UB-GOLD folds via harness run_eval.py")

    # ================= copy of src/main.py: run_split() =================================================
    input_dim = train_graphs[0].num_node_features
    batch_train = DataLoader(train_graphs,batch_size=args.batch_size,shuffle=True)
    batch_test = DataLoader(test_graphs,batch_size=args.batch_size_test,shuffle=False)
    # CHANGE (D5): held-out normals, pre-collated in chunks of batch_size_test (fixed order, no RNG)
    batch_hold = DataLoader(hold_list, batch_size=args.batch_size_test, shuffle=False)

    model = GAE(model_mod.GAD_Explainer(args=args, input_dim=input_dim)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def train(batch_graphs,batch_num):
        optimizer.zero_grad()
        KL_loss,NCE_loss,similarity,_,_,_,embs,node_embs = model.encode(batch_graphs,batch_num)
        GAE_loss = model.recon_loss(node_embs, batch_graphs.edge_index)
        regular_loss = ( torch.norm(torch.sum(similarity,axis=0)) /similarity.shape[0] * np.sqrt(similarity.shape[1]) - 1 ) / (np.sqrt(similarity.shape[1])-1)
        loss =  args.kl * KL_loss + args.nce * NCE_loss +  args.regular * regular_loss + args.gae * GAE_loss
        loss.backward()
        optimizer.step()

        return loss, similarity, embs

    def test(batch_graphs):
        KL_loss,NCE_loss,similarity,node_bern,edge_bern,data_sim,embs,node_embs = model.encode(batch_graphs)
        logits = - torch.max(similarity, dim=-1).values

        return logits, similarity, node_bern, edge_bern, data_sim

    C = Ctx()
    C.traj, C.train_loss, C.S = [], [], {"test": [], "hold": []}
    C.t_train = C.t_eval = 0.0
    state = torch.load(ckpath, map_location="cpu", weights_only=False) if osp.exists(ckpath) else None
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(state["epoch"] + 1 if state is not None else 1),
               t0=time.strftime("%Y-%m-%d %H:%M:%S"))
    segments = (list(state.get("segments", [])) if state is not None else []) + [seg]
    meta_out["segments"] = segments
    meta_out["device_all"] = "+".join(sorted({x["device"] for x in segments}))

    local_optim = float("inf")
    count = None
    start_epoch = 1
    if state is None:
        model.encoder.init_prototypes(batch_train)
    else:
        # CHANGE (D7): resume. Shrink the prototype parameter to the saved shape (del_proto may have removed
        # some), then load model / optimizer / loop state / RNG; nothing is replayed.
        print(f"[resume] {ckpath} at epoch {state['epoch']}", flush=True)
        model.encoder.prototypes = torch.nn.Parameter(torch.empty(state["proto_shape"]).to(args.device))
        model.load_state_dict(state["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(state["opt"])
        model.encoder.assign_train = [a.to(args.device) for a in state["assign_train"]]
        local_optim, count = state["local_optim"], state["count"]
        C.traj, C.train_loss = state["traj"], state["train_loss"]
        C.S.update(state["S"])
        start_epoch = state["epoch"] + 1
        H.set_rng_state(state["rng"])
    last_save = time.time()

    def payload(epoch):
        return dict(model=model.state_dict(), proto_shape=tuple(model.encoder.prototypes.shape),
                    opt=optimizer.state_dict(), assign_train=[a.detach().cpu() for a in model.encoder.assign_train],
                    local_optim=local_optim, count=count, epoch=epoch, traj=C.traj, train_loss=C.train_loss,
                    rng=H.rng_state(), S=C.S, segments=segments)

    for epoch in range(start_epoch, args.epochs+1):

        model.train()
        epoch_loss = 0
        epoch_time = 0

        t0 = time.time()
        i = 0
        embs_lst = []
        for batch_graphs in batch_train:
            start_time = time.time()
            batch_graphs = batch_graphs.to(args.device)
            loss,sim,embs = train(batch_graphs,i)
            end_time = time.time()
            epoch_time += end_time - start_time
            epoch_loss += loss.item()
            i += 1
            embs_lst.append(embs)
        C.train_loss.append(epoch_loss / len(batch_train))
        C.t_train += time.time() - t0

        t1 = time.time()
        model.eval()
        i = 0
        with torch.no_grad():   # CHANGE (D5): GLADPro evaluates with autograd on (identical values)
            for batch_graphs in batch_test:
                batch_graphs = batch_graphs.to(args.device)
                logits, similarity, node_bern, edge_bern, data_sim = test(batch_graphs)
                if i == 0:
                    logits_ = logits
                    labels_ = batch_graphs.y
                else:
                    logits_ = torch.cat((logits_, logits), dim=0)
                    labels_ = torch.cat((labels_, batch_graphs.y), dim=0)
                i += 1

        auc = utils.compute_metrics(logits_, labels_)
        # CHANGE (D5): instrumentation -- canonical-order test scores, UB-GOLD metrics, held-out normals
        assert np.array_equal(labels_.cpu().numpy().astype(np.int64), y_test)
        test_scores = logits_.detach().cpu().numpy().astype(np.float64)
        hs = []
        with H.rng_isolated(), torch.no_grad():
            for hb in batch_hold:
                hl, *_ = test(hb.to(args.device))
                hs.append(hl.detach().cpu().numpy().astype(np.float64))
        hold_scores = np.concatenate(hs)
        rec = H.eval_metrics(metric, y_test, test_scores)
        C.S["test"].append(test_scores)
        C.S["hold"].append(hold_scores)
        rec["holdout_score"] = H.finite_mean(hold_scores)
        rec["holdout_loss"] = rec["holdout_score"]
        rec.update(epoch=epoch, train_loss=C.train_loss[-1], t=time.time() - t_start, auroc_gladpro=float(auc),
                   n_prot=int(model.encoder.prototypes.shape[0]), epoch_time_gladpro=epoch_time)
        C.traj.append(rec)
        C.t_eval += time.time() - t1
        if verbose:
            print(f'Fold_idx:{fold + 1}, Epoch: {epoch}, auc: {auc}, loss: {epoch_loss / len(batch_train)}, time: {epoch_time}s'
                  f' | [{run_id}] auroc {rec["auroc"]:.4f} hold {rec["holdout_score"]:.4f} n_prot {rec["n_prot"]}'
                  f' t={rec["t"]:.0f}s', flush=True)

        if epoch > 50 and (not args.nofilter):
            if epoch_loss <= local_optim:
                count = 0
                local_optim = epoch_loss
            else:
                count += 1
                if count >= 5:
                    count = 0
                    local_optim = float("inf")
                    # filter prototypes
                    train_embs = torch.cat(embs_lst)
                    model.encoder.del_proto(train_embs,len(train_graphs))   # (len(train_idx) in main.py)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if FORCE_DELPROTO_EPOCH > 0 and epoch == FORCE_DELPROTO_EPOCH:
            # TEST ONLY (verification runs, tag required; inert otherwise): force one prototype deletion
            # (theta = 2/1) so the resume test covers a shrunk prototype tensor
            assert max_epochs is not None
            model.encoder.del_proto(torch.cat(embs_lst), 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # CHANGE (D7): checkpoint every ckpt_every_s and at the deadline; resume tests stop at EP_TEST_STOP_EPOCH
        now = time.time()
        stop = (deadline is not None and now > deadline) or (H.TEST_STOP_EPOCH > 0 and epoch == H.TEST_STOP_EPOCH
                                                             and seg["start_epoch"] <= epoch)
        if (stop or now - last_save > ckpt_every_s) and epoch < args.epochs:
            H.save_ckpt(ckpath, payload(epoch))
            last_save = now
        if stop and epoch < args.epochs:
            print(f"[{run_id}] deadline reached at epoch {epoch}; checkpoint saved", flush=True)
            return "incomplete"

    print(f'Final Prototype Number is {model.encoder.prototypes.shape[0]}')
    # ================= end of run_split() copy ===========================================================

    seg["t1"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_out["seconds"] = time.time() - t_start
    meta_out["seconds_train_this_segment"] = C.t_train
    meta_out["seconds_eval_this_segment"] = C.t_eval
    meta_out["epochs_this_segment"] = args.epochs - seg["start_epoch"] + 1
    meta_out["sec_per_epoch_train"] = C.t_train / max(1, meta_out["epochs_this_segment"])
    meta_out["sec_per_eval"] = C.t_eval / max(1, meta_out["epochs_this_segment"])
    meta_out["n_eval"] = len(C.traj)
    meta_out["final_n_prot"] = int(model.encoder.prototypes.shape[0])
    meta_out["memory"] = H.peak_mem()
    meta_out["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    epochs = [r["epoch"] for r in C.traj]
    assert epochs == list(range(1, args.epochs + 1)), "trajectory must cover every epoch"
    npz = osp.join(SCORES, run_id + ".npz")
    H.save_scores(npz, epochs, C.S, dict(
        y_test=y_test, test_index=np.asarray(split_test_index, dtype=np.int64),
        hold_pos=np.asarray(hold_pos, dtype=np.int64),
        test_n_nodes=np.array([int(d.num_nodes) for d in test_graphs], dtype=np.int64),
        test_n_edges=np.array([int(d.edge_index.size(1)) // 2 for d in test_graphs], dtype=np.int64),
        hold_n_nodes=np.array([int(d.num_nodes) for d in hold_list], dtype=np.int64),
        hold_n_edges=np.array([int(d.edge_index.size(1)) // 2 for d in hold_list], dtype=np.int64),
        num_epoch=np.int64(args.epochs), eval_freq=np.int64(1)))
    meta_out["scores_npz"] = npz
    res = dict(meta=meta_out, traj=C.traj, train_loss=C.train_loss)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_json)
    if osp.exists(ckpath):
        os.remove(ckpath)
    print(f"[done] {run_id} in {meta_out['seconds']:.0f}s -> {out_json}", flush=True)
    return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--hp", default="ubgold", choices=["ubgold", "hpfree"])
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit after it")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke tests only (use with --tag)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EP_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    a = ap.parse_args()
    if a.method != METHOD:
        sys.exit(f"this adapter runs only {METHOD}")
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag (smoke runs must not be mistaken for real runs)")
    torch.set_num_threads(a.threads)
    st = run(a.dataset, a.fold, hp=a.hp, holdout_frac=a.holdout_frac, deadline=a.deadline, max_epochs=a.max_epochs,
             ckpt_every_s=a.ckpt_every_s, tag=a.tag)
    sys.exit(0 if st == "done" else 3)
