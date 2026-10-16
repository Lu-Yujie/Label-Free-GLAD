"""E5 adapter: one instrumented LGKDE run per (dataset, fold, hp) on the UB-GOLD folds of the evalpaper harness.

Same CLI as evalpaper/harness/run_eval.py; writes a part JSON in the harness schema:
  ext/parts/LGKDE_<DS>_fold<k>[_hpfree][_<tag>].json      meta + per-epoch trajectory
  $GLAD_DATA_ROOT/ext_lgkde/scores/<run_id>.npz per-graph test / held-out anomaly scores per epoch
Resumable (model, optimizer, RNG, trajectory) every --ckpt_every_s seconds and at --deadline.

LGKDE code: PyG port in lgkde_pyg.py / lgkde_trainer.py of github.com/MathAdventurer/LGKDE @1bba5ef (MIT).
Configurations and protocols: NOTES.md section 0 (pre-registered). Anomaly score = -KDE density.

Usage:
  PYTHONDONTWRITEBYTECODE=1 python -B run_method.py --method LGKDE --dataset MUTAG --fold 0 [--hp hpfree]
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]
_sys.path.insert(0, _HERE_)

import os as _os
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (GPU runs)

import argparse
import copy
import hashlib
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

OVERLAY = (os.environ.get("GLAD_ENVS", "envs") + "/ext-lgkde-overlay")     # pyyaml 6.0.2 (not in ubgold-gpu); appended
if OVERLAY not in sys.path:
    sys.path.append(OVERLAY)
import yaml

import lgkde_pyg as P
import lgkde_trainer as PT
import ubgold_data

HERE = _HERE_
METHOD = "LGKDE"
LGKDE_COMMIT = "1bba5ef00067230fa327a630d67cabbd53fdf523"
CONFIG_FILE = osp.join(HERE, "src", "configs", "MUTAG.yaml")
DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_lgkde")
PARTS = os.environ.get("EP_PARTS", osp.normpath(osp.join(HERE, "..", "parts")))
SCORES = os.environ.get("EP_SCORES", osp.join(DATA_ROOT, "scores"))
CKPT_ROOT = os.environ.get("EP_CKPT", osp.join(DATA_ROOT, "ckpt"))
NEGCACHE = os.environ.get("LGKDE_NEGCACHE", osp.join(DATA_ROOT, "negcache"))
TEST_STOP_EPOCH = int(os.environ.get("EP_TEST_STOP_EPOCH", "0"))   # resume tests only


def resolve_config(hp):
    """NOTES.md 0.2 / 0.3."""
    cfg = yaml.safe_load(open(CONFIG_FILE))
    notes = ["released config: src/configs/MUTAG.yaml (the only released per-dataset config; modal for all datasets)",
             "dataset.name/type/normal_class/train_ratio/mixed_anomaly_ratio -> UB-GOLD fold; base.seed -> 3407+fold"]
    if hp == "ubgold":
        return cfg, notes
    assert hp == "hpfree", hp
    old_ns, old_es = cfg["negative_sampling"], cfg["early_stopping"]
    cfg["negative_sampling"] = {}      # NegativeSampleGenerator keyword defaults
    del cfg["early_stopping"]           # Trainer: config.get('early_stopping', ...) -> .get defaults
    notes.append("hpfree: negative_sampling -> NegativeSampleGenerator defaults "
                 "(perturbation_methods ('feature_swap','feature_noise','structure'), feature_noise_std 0.1; "
                 f"was {old_ns})")
    notes.append(f"hpfree: early_stopping -> code defaults patience 10, min_delta 0.001, min_epochs 5 (was {old_es})")
    notes.append("hpfree: model.*, training.*, dataset.self_loop/batch_size have no code default (KeyError) "
                 "-> crash fallback = modal released value (MUTAG.yaml)")
    return cfg, notes


def own_rule_params(cfg):
    es = cfg.get('early_stopping', cfg['training'].get('early_stopping', {}))
    return dict(patience=es.get('patience', 10), min_delta=es.get('min_delta', 0.001),
                min_epochs=es.get('min_epochs', 5))


def flat_hparams(cfg):
    out = {}
    for sec, d in cfg.items():
        if isinstance(d, dict):
            for k, v in d.items():
                out[f"{sec}.{k}"] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    out["num_epoch"] = int(cfg["training"]["epochs"])
    out["eval_freq"] = 1
    return out


def neg_cache_path(DS, fold, ns_cfg, n_fit):
    gen = P.NegativeSampleGenerator(**ns_cfg)
    key = json.dumps(dict(t=gen.threshold, b=gen.big_energy_threshold, m=gen.mid_energy_threshold,
                          r=gen.max_ratio, s="structure" in gen.perturbation_methods, n=n_fit), sort_keys=True)
    return osp.join(NEGCACHE, f"{DS}_fold{fold}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.pt")


def edge_fingerprint(graphs):
    return [int(g["edge_index"].sum()) + 7 * int(g["n"]) for g in graphs]


def run(method, DS, fold, hp="ubgold", holdout_frac=0.2, deadline=None, max_epochs=None, ckpt_every_s=300.0,
        tag="", verbose=True):
    assert method == METHOD, method
    t_start = time.time()
    run_id = f"{METHOD}_{DS}_fold{fold}" + ("_hpfree" if hp == "hpfree" else "") + (f"_{tag}" if tag else "")
    os.makedirs(PARTS, exist_ok=True)
    out_json = osp.join(PARTS, run_id + ".json")
    if osp.exists(out_json):
        print(f"[skip] {out_json} exists")
        return "done"
    H = ubgold_data.load_harness()
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":   # deterministic index_add_ (GraphConv aggregation); warn instead of failing
        torch.use_deterministic_algorithms(True, warn_only=True)
    # vectorization block size (speed only): cache-sized blocks on CPU, large blocks on GPU
    P.DGMMD.block_elems = int(os.environ.get("LGKDE_BLOCK_ELEMS", 2 ** 25 if device.type == "cuda" else 2 ** 20))
    metric = _load_metric(H)
    cfg, hp_notes = resolve_config(hp)
    if max_epochs is not None:
        cfg["training"]["epochs"] = int(max_epochs)
    num_epoch = int(cfg["training"]["epochs"])
    fd = ubgold_data.load_fold(DS, fold, holdout_frac=holdout_frac, H=H)
    seed = fd["seed"]
    y_test = fd["y_test"]
    n_test, n_hold = len(fd["test"]), len(fd["hold"])

    T = PT.Trainer(cfg, fd["fit"], fd["test"], y_test, device, seed)   # set_global_seed(seed) + model init
    # held-out normals: one fixed batch (query graphs, like the test set); scoring never touches the RNG
    hold_graphs = [PT.build_graph(g) for g in fd["hold"]]
    hold_batch = P.batch_graphs([g["x"] for g in hold_graphs], [g["edge_index"] for g in hold_graphs],
                                [-10 ** 7 - i for i in range(n_hold)])
    # deterministic negative-graph structures: shared cache (same for both hp; SVD is label-free)
    ncp = neg_cache_path(DS, fold, cfg["negative_sampling"], len(fd["fit"]))
    fp = edge_fingerprint(fd["fit"])
    t_ns = time.time()
    if osp.exists(ncp):
        c = torch.load(ncp, weights_only=False)
        if c.get("fingerprint") == fp:
            T.neg_gen._struct_cache.update(c["cache"])
    if "structure" in T.neg_gen.perturbation_methods:
        # precompute the structure perturbation of every fit graph (SVD; no RNG, so the training stream is
        # unchanged), then share it through the cache file
        missing = [i for i in range(len(T.train_graphs)) if (i, False) not in T.neg_gen._struct_cache]
        for i in missing:
            g = T.train_graphs[i]
            T.neg_gen.structure_perturbation_svd(dict(x=g["x"], edge_index=g["edge_index"], gidx=i), False)
        if missing:
            os.makedirs(NEGCACHE, exist_ok=True)
            tmp = ncp + f".tmp{os.getpid()}"
            torch.save(dict(cache=T.neg_gen._struct_cache, fingerprint=fp), tmp)
            os.replace(tmp, ncp)
    seconds_negstruct = time.time() - t_ns

    meta_out = dict(method=METHOD, dataset=DS, fold=fold, hp=hp, seed=seed, run_id=run_id,
                    lgkde_commit=LGKDE_COMMIT, released_config=osp.relpath(CONFIG_FILE, HERE),
                    ubgold_sh_line="python main.py --config configs/MUTAG.yaml   # LGKDE released config "
                                   "(MUTAG.yaml, modal released config for every dataset; NOTES.md 0.2)",
                    config=cfg, hp_notes=hp_notes, hparams=flat_hparams(cfg), own_rule=own_rule_params(cfg),
                    n_train_normal_ubgold=fd["n_all"], n_fit=len(fd["fit"]), n_holdout=n_hold,
                    holdout_seed=H.HOLDOUT_SEED + 1000 * fold, n_test=n_test, n_test_anomalous=int(y_test.sum()),
                    num_feat=fd["num_feat"], device=str(device), torch=torch.__version__,
                    impl="LGKDE PyG port (lgkde_pyg.py, vectorized blocked MMD, CUDA copy semantics for the "
                         "negative-feature swap)", cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                    threads=torch.get_num_threads(), max_epochs_override=max_epochs,
                    harness_run_eval_sha256=ubgold_data.harness_sha256(),
                    port_sha256={f: hashlib.sha256(open(osp.join(HERE, f), "rb").read()).hexdigest()
                                 for f in ("lgkde_pyg.py", "lgkde_trainer.py", "run_method.py", "ubgold_data.py")},
                    block_elems=P.DGMMD.block_elems)

    state = None
    if osp.exists(ckpath):
        state = torch.load(ckpath, map_location="cpu", weights_only=False)
        print(f"[resume] {ckpath} at epoch {state['epoch']}", flush=True)
    seg = dict(device=device.type, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads=torch.get_num_threads(), start_epoch=(state["epoch"] + 1 if state is not None else 1),
               t0=time.strftime("%Y-%m-%d %H:%M:%S"))
    segments = (list(state.get("segments", [])) if state is not None else []) + [seg]
    meta_out["segments"] = segments
    meta_out["device_all"] = "+".join(sorted({x["device"] for x in segments}))

    traj, S = [], {"test": [], "hold": []}
    start_epoch = 1
    if state is not None:
        T.model.load_state_dict(state["model"])
        T.optimizer.load_state_dict(state["opt"])
        T.train_losses = list(state["train_losses"])
        traj, S = state["traj"], state["S"]
        start_epoch = state["epoch"] + 1
        H.set_rng_state(state["rng"])
    t_train = t_eval = 0.0
    last_save = time.time()
    for epoch in range(start_epoch, num_epoch + 1):
        t0 = time.time()
        avg_loss = T.train_epoch(epoch)
        t1 = time.time()
        t_train += t1 - t0
        # evaluation at every epoch (LGKDE's loop); held-out scoring inside rng_isolated()
        dens_test, (dens_hold,) = T.evaluate(extra_queries=(hold_batch,), extra_ctx=H.rng_isolated)
        test_score = -dens_test.detach().cpu().double().numpy()
        hold_score = -dens_hold.detach().cpu().double().numpy()
        rec = H.eval_metrics(metric, y_test, test_score)
        own = PT.metrics_lgkde(y_test, dens_test.detach().cpu().numpy()) if rec["nonfinite"] == 0 else \
            dict(lgkde_auroc=float("nan"), lgkde_aupr=float("nan"), lgkde_fpr95=float("nan"))
        rec.update(own)
        rec["own_sel"] = (own["lgkde_auroc"] + own["lgkde_aupr"]) / 2
        rec["holdout_score"] = H.finite_mean(hold_score)
        rec["holdout_loss"] = rec["holdout_score"]
        rec["holdout_density_mean"] = float(np.mean(-hold_score))
        rec["test_density_mean"] = float(np.mean(-test_score))
        rec["kde_alpha"] = [float(a) for a in torch.softmax(T.model.kde_logits.detach().cpu(), 0)]
        rec["epoch"] = epoch
        rec["train_loss"] = float(avg_loss)
        rec["t"] = time.time() - t_start
        traj.append(rec)
        S["test"].append(test_score)
        S["hold"].append(hold_score)
        t_eval += time.time() - t1
        if verbose:
            print(f"[{run_id}] ep {epoch} train {avg_loss:.5f} test_auc {rec['auroc']:.4f} own_sel {rec['own_sel']:.4f} "
                  f"hold {rec['holdout_score']:.5g} t={rec['t']:.0f}s", flush=True)
        now = time.time()
        stop = (deadline is not None and now > deadline) or (TEST_STOP_EPOCH > 0 and epoch == TEST_STOP_EPOCH
                                                             and segments[-1]["start_epoch"] <= epoch)
        if (stop or now - last_save > ckpt_every_s) and epoch < num_epoch:
            H.save_ckpt(ckpath, dict(model=T.model.state_dict(), opt=T.optimizer.state_dict(), epoch=epoch,
                                     traj=traj, S=S, train_losses=T.train_losses, rng=H.rng_state(),
                                     segments=segments))
            last_save = now
        if stop and epoch < num_epoch:
            print(f"[{run_id}] deadline reached at epoch {epoch}; checkpoint saved", flush=True)
            return "incomplete"

    seg["t1"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_out["seconds"] = time.time() - t_start
    meta_out["seconds_negstruct_this_segment"] = seconds_negstruct
    meta_out["seconds_train_this_segment"] = t_train
    meta_out["seconds_eval_this_segment"] = t_eval
    meta_out["epochs_this_segment"] = num_epoch - seg["start_epoch"] + 1
    meta_out["sec_per_epoch_train"] = t_train / max(1, meta_out["epochs_this_segment"])
    meta_out["sec_per_eval"] = t_eval / max(1, meta_out["epochs_this_segment"])
    meta_out["n_eval"] = len(traj)
    meta_out["memory"] = H.peak_mem()
    meta_out["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    epochs = [r["epoch"] for r in traj]
    assert epochs == list(range(1, num_epoch + 1)), epochs[:5]
    npz = osp.join(SCORES, run_id + ".npz")
    test_g, hold_g = fd["test"], fd["hold"]
    H.save_scores(npz, epochs, S, dict(
        y_test=y_test, test_index=fd["split_test_index"], hold_pos=fd["hold_pos"],
        test_n_nodes=np.array([g["n"] for g in test_g], dtype=np.int64),
        test_n_edges=np.array([int(g["edge_index"].size(1)) // 2 for g in test_g], dtype=np.int64),
        hold_n_nodes=np.array([g["n"] for g in hold_g], dtype=np.int64),
        hold_n_edges=np.array([int(g["edge_index"].size(1)) // 2 for g in hold_g], dtype=np.int64),
        num_epoch=np.int64(num_epoch), eval_freq=np.int64(1)))
    meta_out["scores_npz"] = npz
    res = dict(meta=meta_out, traj=traj, train_loss=[float(x) for x in T.train_losses])
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_json)
    if osp.exists(ckpath):
        os.remove(ckpath)
    print(f"[done] {run_id} in {meta_out['seconds']:.0f}s -> {out_json}", flush=True)
    return "done"


def _load_metric(H):
    """UB-GOLD's GAOOD/metric/metric.py (ood_auc, ood_aupr, fpr95), loaded the way the harness loads it."""
    H.stub_package("GAOOD", osp.join(H.SRC, "GAOOD"))
    if H.SRC not in sys.path:
        sys.path.insert(0, H.SRC)
    import importlib
    return importlib.import_module("GAOOD.metric.metric")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
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
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag (smoke runs must not be mistaken for real runs)")
    torch.set_num_threads(a.threads)
    st = run(a.method, a.dataset, a.fold, hp=a.hp, holdout_frac=a.holdout_frac, deadline=a.deadline,
             max_epochs=a.max_epochs, ckpt_every_s=a.ckpt_every_s, tag=a.tag)
    sys.exit(0 if st == "done" else 3)
