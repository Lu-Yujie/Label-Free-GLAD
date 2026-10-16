"""Phase B typed adapter: LGKDE (E5; ICML'26) on GX / MS-v2 (RUN_NOTES_ext.md §0).

The port evalpaper/ext/LGKDE/ (PyG port of github.com/MathAdventurer/LGKDE @1bba5ef: lgkde_pyg.py,
lgkde_trainer.py, run_method.py) is imported READ-ONLY. Reused as is: resolve_config ('ubgold' = configs/MUTAG.yaml for
every dataset, A18 §2), own_rule_params, flat_hparams, the Trainer (model init, negatives, training epoch: CUDA copy
semantics on every device, A18 §3), metrics_lgkde. The port's _run is copied here with # CHANGE-TYPED marks:
  * data = typed_data graph dicts (x = typed features, edge_index, n); set_seed(3407 + seed) (UB-GOLD) then the Trainer's
    set_global_seed(3407 + seed), as the port; reference set = all fit graphs;
  * the Trainer is built with group 1's test set; at every epoch each group's test set (test normals + T) is scored by a
    copy of Trainer.evaluate() with that group as the (non-shuffled, full-batch) test loader; groups 2..K from the saved
    RNG state S0, group 1 last (RUN_NOTES §0a; the RNG draws of evaluate() do not depend on the set's size);
  * held-out normals against the same reference set, RNG-isolated (A3);
  * the structure-perturbation SVD cache is per unit under typed/ext_ckpt/LGKDE/cache/ (label-free, deterministic).
Per type the selection key of LGKDE's own rule, own_sel = (sklearn AUROC + AP)/2, is recorded; the collector replays
LGKDE's EarlyStopping (MUTAG.yaml: patience 5, min_delta 0.001, min_epochs 10) on it (oracle_own).
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B LGKDE_typed.py --generator GX --seed 0
"""
import os.path as _osp, sys as _sys
_sys.dont_write_bytecode = True
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
if _HERE_ not in _sys.path:
    _sys.path.insert(0, _HERE_)

import hashlib
import json
import os
import os.path as osp
import time

import numpy as np
import torch

import typed_ext_common as TC

METHOD = "LGKDE"


def run(R, a):
    M = TC.load_port(METHOD)                        # ext/LGKDE/run_method.py (+ lgkde_pyg / lgkde_trainer), read only
    P, PT = M.P, M.PT
    H = TC.load_harness()
    metric = TC.load_metric()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True, warn_only=True)
    P.DGMMD.block_elems = int(os.environ.get("LGKDE_BLOCK_ELEMS", 2 ** 25 if device.type == "cuda" else 2 ** 20))
    cfg, hp_notes = M.resolve_config("ubgold")
    if R.max_epochs is not None:
        cfg["training"]["epochs"] = int(R.max_epochs)
    num_epoch = int(cfg["training"]["epochs"])
    seed = 3407 + R.seed

    # ---- CHANGE-TYPED: typed data as graph dicts (the port's ubgold_data.load_fold conv()) -----------------------
    ns = H.load_ubgold_data_functions()
    ns["set_seed"](seed)                            # = ubgold_data.load_fold's set_seed(3407 + fold)
    td = R.load_data()
    Pg = td.pyg()
    keys = R.keys

    def conv(d):
        return dict(x=d.x.to(torch.float32).contiguous(), edge_index=d.edge_index.long().contiguous(),
                    n=int(d.num_nodes))
    fit = [conv(d) for d in Pg["fit"]]
    hold = [conv(d) for d in Pg["hold"]]
    test = {k: [conv(d) for d in Pg["test"][k]] for k in keys}

    T = PT.Trainer(cfg, fit, test[keys[0]], R.y[keys[0]], device, seed)   # set_global_seed(seed) + model init
    # CHANGE-TYPED: one non-shuffled full-batch query loader per group (as Trainer.setup_experiment builds test_loader)
    DL = torch.utils.data.DataLoader
    loaders = {}
    for k in keys:
        gs = []
        for i, g in enumerate(test[k]):
            gg = PT.build_graph(g)
            gg["gidx"] = -1 - i
            gs.append(gg)
        ds = P.CustomDataset(gs, torch.as_tensor(np.asarray(R.y[k]), dtype=torch.long))
        loaders[k] = DL(ds, batch_size=len(ds), shuffle=False, collate_fn=P.collate)
    hold_graphs = [PT.build_graph(g) for g in hold]
    hold_batch = P.batch_graphs([g["x"] for g in hold_graphs], [g["edge_index"] for g in hold_graphs],
                                [-10 ** 7 - i for i in range(len(hold))])

    # structure-perturbation cache (copy of the port's block; CHANGE-TYPED: per-unit path under ext_ckpt)
    os.makedirs(R.cache_dir, exist_ok=True)
    gen = P.NegativeSampleGenerator(**cfg["negative_sampling"])
    key = json.dumps(dict(t=gen.threshold, b=gen.big_energy_threshold, m=gen.mid_energy_threshold, r=gen.max_ratio,
                          s="structure" in gen.perturbation_methods, n=len(fit)), sort_keys=True)
    ncp = osp.join(R.cache_dir, f"negstruct_{R.generator}_{R.fv}_s{R.seed}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
                                f"{'_smoke' if R.smoke else ''}.pt")
    fp = M.edge_fingerprint(fit)
    t_ns = time.time()
    if osp.exists(ncp):
        c = torch.load(ncp, weights_only=False)
        if c.get("fingerprint") == fp:
            T.neg_gen._struct_cache.update(c["cache"])
    if "structure" in T.neg_gen.perturbation_methods:
        missing = [i for i in range(len(T.train_graphs)) if (i, False) not in T.neg_gen._struct_cache]
        for i in missing:
            g = T.train_graphs[i]
            T.neg_gen.structure_perturbation_svd(dict(x=g["x"], edge_index=g["edge_index"], gidx=i), False)
        if missing:
            tmp = ncp + f".tmp{os.getpid()}"
            torch.save(dict(cache=T.neg_gen._struct_cache, fingerprint=fp), tmp)
            os.replace(tmp, ncp)
    seconds_negstruct = time.time() - t_ns

    own_rule = M.own_rule_params(cfg)
    R.base_meta({"ext/LGKDE/run_method.py": osp.join(TC.PORTS, METHOD, "run_method.py"),
                 "ext/LGKDE/lgkde_pyg.py": osp.join(TC.PORTS, METHOD, "lgkde_pyg.py"),
                 "ext/LGKDE/lgkde_trainer.py": osp.join(TC.PORTS, METHOD, "lgkde_trainer.py"),
                 "harness/run_eval.py": osp.join(TC.HARNESS, "run_eval.py"),
                 "typed/ext/LGKDE_typed.py": osp.abspath(__file__)},
                M.LGKDE_COMMIT, M.flat_hparams(cfg),
                "python main.py --config configs/MUTAG.yaml   # LGKDE released config (MUTAG.yaml for every dataset, A18 §2)",
                hp_notes, device=str(device),
                flags=dict(oracle_own="LGKDE EarlyStopping on (AUROC+AP)/2 " + json.dumps(own_rule),
                           own_rule=own_rule, reproducibility="A18 §4: reproduction criterion not met (1/4); port "
                           "bitwise identical to the original", device_semantics="CUDA copy semantics (A18 §3)",
                           block_elems=P.DGMMD.block_elems, per_type_mode="own_set"))

    def eval_group(k):
        """CHANGE-TYPED: copy of Trainer.evaluate() with group k's loader as the test loader."""
        T.model.eval()
        with torch.no_grad():
            for train_batched_graph, train_labels in T.train_loader_noshuffle:
                train_batched_graph = train_batched_graph.to(T.device)
            test_scores_list = []
            for test_batched_graph, test_labels in loaders[k]:
                test_batched_graph = test_batched_graph.to(T.device)
                query_scores = T.model.get_query_scores(test_batched_graph, train_batched_graph)
                test_scores_list.append(query_scores)
            test_scores = torch.cat(test_scores_list, dim=0)
        return test_scores

    state = R.load_state(device.type, torch.get_num_threads(), first_epoch=1)
    start_epoch = 1
    if state is not None:
        T.model.load_state_dict(state["model"])
        T.optimizer.load_state_dict(state["opt"])
        T.train_losses = list(state["train_losses"])
        start_epoch = R.restore_common()

    def payload():
        return dict(model=T.model.state_dict(), opt=T.optimizer.state_dict(), train_losses=T.train_losses)

    for epoch in range(start_epoch, num_epoch + 1):
        t0 = time.time()
        avg_loss = T.train_epoch(epoch)
        R.train_loss.append(float(avg_loss))
        t1 = time.time()
        R.t_train += t1 - t0
        dens = R.score_groups(eval_group)                       # CHANGE-TYPED: per-group evaluation (§0a)
        with H.rng_isolated():                                   # held-out normals vs the same reference set
            T.model.eval()
            with torch.no_grad():
                for ref, _ in T.train_loader_noshuffle:
                    ref = ref.to(T.device)
                dens_hold = T.model.get_query_scores(hold_batch.to(T.device), ref)
        scores = {k: -dens[k].detach().cpu().double().numpy() for k in keys}
        hold_score = -dens_hold.detach().cpu().double().numpy()
        tx = {}
        for k in keys:
            if np.all(np.isfinite(scores[k])):
                own = PT.metrics_lgkde(R.y[k], dens[k].detach().cpu().numpy())
            else:
                own = dict(lgkde_auroc=float("nan"), lgkde_aupr=float("nan"), lgkde_fpr95=float("nan"))
            own["own_sel"] = (own["lgkde_auroc"] + own["lgkde_aupr"]) / 2
            tx[k] = own
        R.record(epoch, scores, hold_score,
                 rec_extra=dict(holdout_density_mean=float(np.mean(-hold_score)),
                                kde_alpha=[float(x) for x in torch.softmax(T.model.kde_logits.detach().cpu(), 0)]),
                 types_extra=tx, t_eval0=t1, metric=metric)
        if R.maybe_stop(epoch, payload, final_epoch=num_epoch):
            return "incomplete"

    return R.finish(num_epoch, 1, expected_epochs=range(1, num_epoch + 1),
                    meta_extra=dict(seconds_negstruct_this_segment=seconds_negstruct, negstruct_cache=ncp,
                                    config=cfg))


if __name__ == "__main__":
    TC.main(METHOD, run)
