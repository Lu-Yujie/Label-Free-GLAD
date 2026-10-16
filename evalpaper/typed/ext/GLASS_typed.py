"""Phase B typed adapter: GLASS (E4; arXiv 2609.05253) on GX / MS-v2 (RUN_NOTES_ext.md §0).

The port evalpaper/ext/GLASS/run_method.py (GLASS src @3bbba39, unmodified; transformers overlay) is imported
READ-ONLY. Reused as is: load_glass, resolve_hparams / make_config (README single-domain line = the only released
configuration, A23 §3), make_text_encoder / text_embeddings (GLASS's own content-hashed precompute_text_embeddings,
Qwen3-Embedding-0.6B bf16, batches of 16 in dataset order), reference_rules / reference_scores (evaluate_reference.py's
24 rules, R = fit-set normals, k 5), holdout_train_scorers, make_evaluate_all_scorers (GLASS's in-loop scorers with score
capture), glass_summary. The port's training loop (copy of glass_training_core.train_single_run) is copied here with
# CHANGE-TYPED marks:
  * data: GLASSDataset over the unit's graph list [training normals (file order) | test normals | anomalies of each
    group in group order] (the generator's file order); GLASS's own _precompute_features (LTD, spectral sketch,
    GraphDP v1; the GraphDP text also summarises the node-feature matrix) once per unit in `--prep`, canonical input
    znorm([LTD; x]) with x = typed features. GLASS name: MS-v2 -> 'NCI1' (NCI1 molecules; domain 'molecular'),
    GX -> 'GX' (GLASS's default domain 'generic'). Caches under typed/ext_ckpt/GLASS/cache/ (smoke: smoke_ext/...);
  * set_seed(3407 + seed); fit set in RandomState(3407 + seed) order (port D2); held-out = typed held-out normals;
  * at every evaluation point (epoch 0, 1, 5, ..., 150) GLASS's in-loop evaluate_all_scorers runs once per group with that
    group's test set (it normalises over the scored set, grid-searches on its labels and re-initialises the vMF
    prototypes from the training graphs); groups 2..K from the saved RNG state S0, the first group last (RUN_NOTES §0a),
    so model state and training stream are those of a GLASS run whose test set is (test normals + first group);
  * reference-set scorer (all 24 rules) on the held-out set and on each group's test set as separate query sets,
    RNG-isolated (inductive: every query graph is scored against R only, A24 §1);
  * GLASS's online early-stop replay / own final evaluation is not needed: ORACLE-own (A23 §2) is replayed per group by
    the collector from the recorded in-loop ensemble AUROC (strict improvement, +5 per non-improving evaluation, stop at
    patience 40) and the 24 rules' per-group metrics at the selected checkpoint (the reference scores of the selected
    checkpoint ARE the recorded ones at that point); the secondary train_single best-of-6 row is not computed.
Label-free rows: default rule rcsm_full_test_cdf (A23 §1 / A24 §1); fallback fixed_test_raw reported as last_r0 /
valnormal_r0. CPU, 1 thread, deterministic algorithms (port D9); GLASS's eigsh start vectors draw OS entropy (A24 §2).
  --prep --generator G --seed s [--no_encode] [--smoke --smoke_random_emb]   (text embedding needs a GPU)
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B GLASS_typed.py --generator GX --seed 0
"""
import os.path as _osp, sys as _sys
_sys.dont_write_bytecode = True
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
if _HERE_ not in _sys.path:
    _sys.path.insert(0, _HERE_)

import json
import os
import os.path as osp
import pickle
import time

import numpy as np
import torch

import typed_ext_common as TC

METHOD = "GLASS"
GLASS_NAME = {"GX": "GX", "MSV2": "NCI1"}       # RUN_NOTES_ext §0.3


def cache_root(smoke):
    return osp.join(TC.roots(smoke)["ckpt"], METHOD, "cache")


def load_port_with_caches(smoke):
    """The port reads its cache dirs from the environment at import time: point them at the typed caches first."""
    root = cache_root(smoke)
    os.environ["EP_GLASS_EMB"] = osp.join(root, "embeddings")
    os.environ["EP_GLASS_DATA"] = osp.join(root, "features")
    M = TC.load_port(METHOD)
    assert M.EMB_CACHE == os.environ["EP_GLASS_EMB"] and M.DATA_ROOT == os.environ["EP_GLASS_DATA"]
    return M


def unit_graphs(td):
    """[training normals (file order) | test normals | anomalies of each group] as PyG Data (typed features)."""
    P = td.pyg()
    from torch_geometric.data import Data
    n_tr = len(td.train)
    train_data = [None] * n_tr
    for d, p in zip(P["fit"], td.fit_pos):
        train_data[p] = d
    for d, p in zip(P["hold"], td.hold_pos):
        train_data[p] = d
    keys = list(td.groups)
    n_norm = len(td.test_normals)
    full = list(train_data) + list(P["test"][keys[0]][:n_norm])
    test_idx = {}
    for k in keys:
        start = len(full)
        an = P["test"][k][n_norm:]
        full += list(an)
        test_idx[k] = np.r_[np.arange(n_tr, n_tr + n_norm), np.arange(start, start + len(an))].astype(np.int64)
    out = []
    for d in full:
        g = Data(x=d.x, edge_index=d.edge_index, y=d.y)
        g.num_nodes = int(d.num_nodes)
        g.edge_attr = None
        out.append(g)
    return out, test_idx


def feature_cache_path(smoke, generator, seed, fv):
    return osp.join(cache_root(smoke), "features", f"{generator}_{fv}_s{seed}_glass_features.pkl")


def build_dataset(M, G, config, generator, seed, fv, full, smoke, allow_build=False):
    """Copy of the port's build_glass_dataset (CHANGE-TYPED: typed graph list, per-(generator, seed) cache)."""
    ds = G.ds.GLASSDataset.__new__(G.ds.GLASSDataset)
    ds.config = config
    ds.dataset_name = config.dataset_name
    ds.data_root = config.data_root
    ds.raw_dataset = None
    ds.data_list = full
    ds.num_features = int(full[0].x.shape[1])
    ds.domain = G.gdp.DOMAIN_MAP.get(ds.dataset_name, 'generic')
    ds.graphdp_version = os.environ.get("GLASS_GRAPHDP_VERSION", "v1").lower()
    assert ds.graphdp_version == "v1"
    path = feature_cache_path(smoke, generator, seed, fv)
    sig = [int(d.num_nodes) for d in full], [int(d.edge_index.size(1)) for d in full], int(full[0].x.shape[1])
    if osp.exists(path):
        with open(path, "rb") as f:
            c = pickle.load(f)
        assert c["n_graphs"] == len(full) and c["generator"] == generator and c["seed"] == seed and c["fv"] == fv
        assert tuple(c["sig"]) == tuple(sig), "feature cache built for different graphs"
        ds.ltd_list, ds.spectral_list, ds.graph_dp_list = c["ltd"], c["spectral"], c["graph_dp"]
    elif allow_build:
        ds._precompute_features()                       # GLASS's own (LTD, spectral sketch, GraphDP v1)
        os.makedirs(osp.dirname(path), exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(dict(generator=generator, seed=seed, fv=fv, glass_name=ds.dataset_name, n_graphs=len(full), sig=sig,
                             ltd=ds.ltd_list, spectral=ds.spectral_list, graph_dp=ds.graph_dp_list,
                             glass_commit=M.GLASS_COMMIT, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             host=os.uname().nodename), f)
        os.replace(tmp, path)
    else:
        raise TC.PrereqMissing(f"{path} missing: run GLASS_typed.py --prep for {generator} {fv} seed {seed} first")
    assert len(ds.ltd_list) == len(full) and len(ds.graph_dp_list) == len(full)
    for i in (0, len(full) - 1):
        assert ds.ltd_list[i].shape[0] == full[i].num_nodes
    return ds


def prep(a):
    """GLASS precomputation for one (generator, seed): features (CPU) + Qwen3 text embeddings (GPU). Label-free."""
    import typed_data as TD
    M = load_port_with_caches(a.smoke)
    G = M.load_glass()
    fv = a.fv or TD.FEATURE_VARIANTS[a.generator][0]
    td = TD.load_typed(a.generator, a.seed, fv)
    full, _ = unit_graphs(td)
    args_g, _ = M.resolve_hparams(GLASS_NAME[a.generator], "ubgold")
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = M.make_config(G, args_g, dev_str)
    t0 = time.time()
    ds = build_dataset(M, G, config, a.generator, a.seed, fv, full, a.smoke, allow_build=True)
    t1 = time.time()
    out = dict(generator=a.generator, seed=a.seed, fv=fv, glass_name=config.dataset_name, n_graphs=len(full),
               feature_seconds=round(t1 - t0, 1), feature_cache=feature_cache_path(a.smoke, a.generator, a.seed, fv))
    inst = G.gdp.get_instruction(domain=G.gdp.DOMAIN_MAP.get(config.dataset_name, 'generic'), mode='train')
    if a.smoke_random_emb:
        assert a.smoke, "random text embeddings only in the smoke cache"
        gen = torch.Generator().manual_seed(0)
        emb = torch.randn(len(full), config.text_embed_dim, generator=gen)
        G.T.get_embedding_cache().put(config.dataset_name, ds.graph_dp_list, inst, emb)
        out.update(text="SMOKE: fixed random vectors (never used by grid runs)")
    elif not a.no_encode:
        if G.T.get_embedding_cache().get(config.dataset_name, ds.graph_dp_list, inst) is not None:
            out.update(text="cache hit")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen3 text embedding needs a GPU (port RUNTIME.md); run the prep job on a GPU node")
            device = torch.device(dev_str)
            _, raw = M.text_embeddings(G, ds, config, dev_str, device, allow_encode=True)
            out.update(text_seconds=round(time.time() - t1, 1), emb_shape=list(raw.shape),
                       emb_finite=bool(torch.isfinite(raw).all()), device=dev_str,
                       cuda_name=torch.cuda.get_device_name(0))
    out["emb_cache_dir"] = M.EMB_CACHE
    if out.get("text") in ("cache hit", "SMOKE: fixed random vectors (never used by grid runs)") or "emb_shape" in out:
        mk = prep_marker(a.smoke, a.generator, a.seed, fv)     # read by jobs/gen_tasks_ext.py
        os.makedirs(osp.dirname(mk), exist_ok=True)
        with open(mk, "w") as f:
            json.dump(dict(out, completed=time.strftime("%Y-%m-%d %H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"),
                           host=os.uname().nodename), f, indent=1)
    print("[prep]", json.dumps(out), flush=True)
    return out


def prep_marker(smoke, generator, seed, fv):
    return osp.join(cache_root(smoke), "prep_done", f"{generator}_{fv}_s{seed}.json")


def run(R, a):
    M = load_port_with_caches(R.smoke)
    H = TC.load_harness()
    metric = TC.load_metric()
    G = M.load_glass()
    T = G.T
    F = torch.nn.functional
    evaluate_all_scorers = M.make_evaluate_all_scorers(T)
    rules = M.reference_rules(G.R)
    rule_names = [r[0] for r in rules]
    assert rule_names[0] == M.R0_RULE and M.DEFAULT_RULE in rule_names and len(rule_names) == 24
    deterministic = not torch.cuda.is_available()
    if deterministic and os.environ.get("EP_GLASS_NONDET", "") != "1":
        torch.use_deterministic_algorithms(True, warn_only=True)     # port D9
    gname = GLASS_NAME[R.generator]
    a_g, hp_line = M.resolve_hparams(gname, "ubgold")
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = M.make_config(G, a_g, dev_str, R.max_epochs)
    text_device = dev_str
    seed = 3407 + R.seed

    # ---- CHANGE-TYPED: typed unit ------------------------------------------------------------------------------
    td = R.load_data()
    keys = R.keys
    full, test_idx = unit_graphs(td)
    if R.dry_n:
        raise RuntimeError("GLASS: dry mode unsupported (the feature cache covers the full unit)")

    # ================= copy of the port's train_single_run() ============================================
    G.ut.set_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(M, G, config, R.generator, R.seed, R.fv, full, R.smoke)     # CHANGE-TYPED
    T._batch_cache = None
    rng = np.random.RandomState(seed)                                             # port D2
    fit_orig = np.asarray(td.fit_pos, dtype=np.int64)
    train_indices = fit_orig[rng.permutation(len(fit_orig))]
    hold_indices = np.asarray(td.hold_pos, dtype=np.int64)
    labels_f = {k: R.y[k].astype(np.float64) for k in keys}

    sample_canon = dataset.get_canonicalized_features(0)
    canon_input_dim = sample_canon.shape[1]
    assert os.environ.get('GLASS_ABLATION', '') == ''
    try:
        text_encoder, raw_text_cache = M.text_embeddings(G, dataset, config, text_device, device, allow_encode=False)
    except FileNotFoundError as e:                 # port: cache miss -> needs the GPU prep
        raise TC.PrereqMissing(str(e))
    text_encoder.projections.to(device)

    model = G.gm.GLASSModel(config, canon_input_dim=canon_input_dim).to(device)
    param_groups = [
        {'params': model.canon_mlp.parameters(), 'lr': config.lr},
        {'params': model.graph_encoder.parameters(), 'lr': config.lr},
        {'params': model.graph_projections.parameters(), 'lr': config.lr},
        {'params': model.prototypes.log_kappas.parameters(), 'lr': config.lr * 0.5},
        {'params': [model.slice_weights], 'lr': config.lr * 0.1},
    ]
    if len(list(text_encoder.projections.parameters())) > 0:
        param_groups.append({'params': text_encoder.projections.parameters(), 'lr': config.text_proj_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    warmup_phase = 20

    R.base_meta({"ext/GLASS/run_method.py": osp.join(TC.PORTS, METHOD, "run_method.py"),
                 "ext/GLASS/src/glass_training_core.py": osp.join(M.GLASS_SRC, "glass_training_core.py"),
                 "ext/GLASS/src/evaluate_reference.py": osp.join(M.GLASS_SRC, "evaluate_reference.py"),
                 "ext/GLASS/src/src/dataset.py": osp.join(M.GLASS_SRC, "src", "dataset.py"),
                 "ext/GLASS/src/src/graph_dp.py": osp.join(M.GLASS_SRC, "src", "graph_dp.py"),
                 "typed/ext/GLASS_typed.py": osp.abspath(__file__)},
                M.GLASS_COMMIT, dict(vars(a_g), gin_layers=config.gin_layers, gin_hidden=config.gin_hidden,
                                     mrl_mode=config.mrl_mode, matryoshka_dims=str(config.matryoshka_dims),
                                     num_prototypes=config.num_prototypes, weight_decay=config.weight_decay,
                                     text_model=M.TEXT_MODEL, eval_epochs="0,1,5,10,..."),
                hp_line, [f"GLASS name {config.dataset_name} (domain {dataset.domain})"], device=str(device),
                flags=dict(default_rule=M.DEFAULT_RULE, r0_rule=M.R0_RULE, rules=rule_names,
                           oracle_own="A23 §2: GLASS test-selected checkpoint (in-loop ensemble AUROC, patience "
                                      f"{config.patience}, +5 per evaluation) then best of 24 rules by 5-seed mean",
                           patience=int(config.patience), glass_name=config.dataset_name, domain=dataset.domain,
                           deterministic_algorithms=deterministic, per_type_mode="own_set",
                           reproducibility="A24 §2: eigsh start vectors from fresh OS entropy (not patched)",
                           feature_cache=feature_cache_path(R.smoke, R.generator, R.seed, R.fv), emb_cache=M.EMB_CACHE,
                           canon_input_dim=int(canon_input_dim), n_graphs_unit=len(full)),
                extra_versions=("transformers", "tokenizers"))

    def eval_point(epoch_done, t_eval0):
        """CHANGE-TYPED: in-loop scorers per group (§0a), then the reference scorer and held-out scorers (isolated)."""
        caps = {}

        def one(k):
            cap = {}
            res = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset, train_indices, test_idx[k],
                                       labels_f[k], config, device, cap)
            caps[k] = cap
            return res
        res = R.score_groups(one)
        with H.rng_isolated():
            refs, rweights = M.reference_scores(G, rules, model, raw_text_cache, dataset, config, device, train_indices,
                                                dict(hold=hold_indices, **{f"g::{k}": test_idx[k] for k in keys}))
            htr = M.holdout_train_scorers(G, model, text_encoder, raw_text_cache, dataset, train_indices,
                                          hold_indices, config, device)
        tx, extra_S = {}, {}
        for k in keys:
            rr = refs[f"g::{k}"]
            rm = {}
            for r in rule_names:
                m = TC.eval_metrics(metric, R.y[k], rr[r])
                rm[r] = [m["auroc"], m["auprc"], m["fpr95"], m["nonfinite"], TC.n_distinct_finite(rr[r])]
                extra_S[f"r__{r}__{TC.skey(k)}"] = rr[r]
            tx[k] = dict(glass_ens_auroc=float(res[k]['ensemble']['auroc']),
                         glass_weight_graph=float(res[k]['ensemble']['weight_graph']),
                         glass_sms_best_alpha=float(res[k]['sms'].get('best_alpha', float('nan'))),
                         rules=rm, glass=M.glass_summary(res[k]))
            extra_S[f"ens__{TC.skey(k)}"] = np.asarray(caps[k]['ensemble'], dtype=np.float64)
        for r in rule_names:
            extra_S["h__" + r] = refs["hold"][r]
        for kk in M.HOLD_TRAIN_KEYS:
            extra_S["ht__" + kk] = htr[kk]
        R.record(epoch_done, {k: refs[f"g::{k}"][M.DEFAULT_RULE] for k in keys}, refs["hold"][M.DEFAULT_RULE],
                 rec_extra=dict(rule_holdout={r: TC.finite_mean(refs["hold"][r]) for r in rule_names},
                                rule_weights_default=rweights[M.DEFAULT_RULE]),
                 types_extra=tx, extra_S=extra_S, t_eval0=t_eval0, metric=metric)

    state = R.load_state(device.type, torch.get_num_threads(), first_epoch=0)
    if state is None:
        eval_point(0, time.time())                     # epoch-0 baseline (untrained model), A23 §4
        start_epoch = 0
    else:
        model.load_state_dict(state["model"])
        text_encoder.projections.load_state_dict(state["text_proj"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        start_epoch = R.restore_common() - 1           # checkpoints store the number of epochs done (1-based)

    def payload():
        return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    text_proj=text_encoder.projections.state_dict(), opt=optimizer.state_dict(),
                    sched=scheduler.state_dict())

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        model.train()
        text_encoder.projections.train()

        in_warmup = epoch < warmup_phase

        if len(train_indices) <= 256:
            batches = [train_indices]
        else:
            perm_idx = np.random.permutation(len(train_indices))
            shuffled = train_indices[perm_idx]
            batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]

        epoch_loss = 0
        n_batches = 0
        _epoch_t0 = time.time()

        for bi in batches:
            if len(bi) < 4:
                continue

            if T._batch_cache is not None:
                x, ei, b, sp = T._batch_cache.get_batch(bi, device)
            else:
                x, ei, b, sp = T.prepare_batch(dataset, bi, config, device)
            raw_emb = raw_text_cache[bi].to(device)
            text_slices = text_encoder.project_to_slices(raw_emb)
            z_slices, _ = model.encode_graph(x, ei, b, sp)

            # Loss 1: Soft cosine alignment (graph <-> text per pair)
            loss_align = torch.tensor(0.0, device=device)
            for z_s, e_s in zip(z_slices, text_slices):
                pos_sim = (z_s * e_s).sum(dim=-1)  # [B]
                loss_align = loss_align + (1 - pos_sim).mean()
            loss_align = loss_align / len(z_slices)

            # Loss 2: Prototype compactness + negatives (phase 2)
            loss_proto = torch.tensor(0.0, device=device)
            if not in_warmup:
                energy_normal = model.prototypes.energy(z_slices)
                loss_compact = torch.log(torch.cosh(energy_normal)).mean()

                loss_margin = torch.tensor(0.0, device=device)
                skip_neg = (T._batch_cache is None) and (not T.GPU_ACCEL) and len(dataset.data_list) > 4500  # Skip expensive negatives for large datasets
                if not skip_neg:
                 try:
                  if T._batch_cache is not None:
                    nx, nei, nb, nsp = T._batch_cache.get_negative_batch(bi, device)
                  else:
                    nx, nei, nb, nsp = T.prepare_negatives(dataset, bi, config, device)
                  z_neg, _ = model.encode_graph(nx, nei, nb, nsp)
                  energy_neg = model.prototypes.energy(z_neg)
                  loss_margin = F.relu(config.margin - (energy_neg - energy_normal)).mean()
                 except Exception:
                  pass

                loss_proto = loss_compact + loss_margin

            # Loss 3: Ortho
            loss_ortho = model.ortho_reg_fn(model.graph_projections, text_encoder.projections if config.mrl_mode == 'learned' else None)

            # Weighting
            proto_w = 0 if in_warmup else min(1.0, (epoch - warmup_phase + 1) / 20) * config.alpha
            total_loss = loss_align + proto_w * loss_proto + config.ortho_weight * loss_ortho

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(text_encoder.projections.parameters()), 1.0
            )
            optimizer.step()

            # EMA prototype update (phase 2)
            if not in_warmup:
                with torch.no_grad():
                    model.prototypes.update_prototypes_ema([z.detach() for z in z_slices])

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        R.train_loss.append(float(avg_loss))

        scheduler.step()

        # Initialize prototypes at start of phase 2
        if epoch == warmup_phase - 1:
            with torch.no_grad():
                if T._batch_cache is not None:
                    init_x, init_ei, init_b, init_sp = T._batch_cache.get_batch(train_indices[:min(256, len(train_indices))], device)
                else:
                    init_x, init_ei, init_b, init_sp = T.prepare_batch(
                        dataset, train_indices[:min(256, len(train_indices))], config, device
                    )
                z_init, _ = model.encode_graph(init_x, init_ei, init_b, init_sp)
                model.prototypes.init_from_data(z_init)
            print(f"  [Epoch {epoch+1}] Prototypes initialized [{time.time()-_epoch_t0:.0f}s]", flush=True)
        R.t_train += time.time() - t0

        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            eval_point(epoch + 1, time.time())         # CHANGE-TYPED (per-group in-loop + reference scorers)

        if R.maybe_stop(epoch + 1, payload, final_epoch=config.epochs):
            return "incomplete"
    # ================= end of the train_single_run() copy ===============================================

    return R.finish(config.epochs, 5, expected_epochs=[0, 1] + list(range(5, config.epochs + 1, 5)),
                    extra_npz=dict(fit_index_order=train_indices, hold_index=hold_indices,
                                   **{f"test_index__{TC.skey(k)}": test_idx[k] for k in keys}))



if __name__ == "__main__":
    if "--prep" in _sys.argv:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--prep", action="store_true")
        ap.add_argument("--generator", required=True, choices=["GX", "MSV2"])
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--fv", default=None, help="GX: F2 | F1 (GraphDP texts summarise the node features); MSV2: MS")
        ap.add_argument("--no_encode", action="store_true")
        ap.add_argument("--smoke", action="store_true")
        ap.add_argument("--smoke_random_emb", action="store_true")
        ap.add_argument("--threads", type=int, default=4)
        pa = ap.parse_args()
        torch.set_num_threads(pa.threads)
        prep(pa)
        _sys.exit(0)
    TC.main(METHOD, run)
