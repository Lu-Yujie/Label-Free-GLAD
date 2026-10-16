"""Phase B typed adapter: GLADPro (E5; KDD'25) on GX / MS-v2 (RUN_NOTES_ext.md §0).

The port evalpaper/ext/GLADPro/run_method.py (GLADPro src @1f811b9, unmodified, + the port's runtime overrides
D1, D2, D6) is imported READ-ONLY: load_gladpro and resolve_hparams are reused (a typed dataset has no README line ->
the modal README configuration = repository defaults with epochs 500, ratified in A15). The training loop (a line-by-line
copy of src/main.py:run_split() in the port) is copied here with # CHANGE-TYPED marks:
  * data = typed_data (fit / held-out / per-group test sets; no RNG consumed); set_seed(3407 + seed) before the model;
  * GLADPro evaluates after every epoch: each group's test set (test normals + T) is one unshuffled test loader of
    batch_size_test 9999 (one batch), scored by GLADPro's own test(); groups 2..K from the saved RNG state S0, the first
    group last (RUN_NOTES §0a; GLADPro's test pass consumes only the DataLoader's base-seed draw);
  * held-out normals: the port's chunks of batch_size_test, RNG-isolated (A3).
Own ORACLE (collector): common epoch with the max 5-seed mean test AUROC of that group (first max; A15.1: epochs finite in
all 5 seeds). CPU with deterministic algorithms (port D8).
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B GLADPro_typed.py --generator GX --seed 0
"""
import os.path as _osp, sys as _sys
_sys.dont_write_bytecode = True
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
if _HERE_ not in _sys.path:
    _sys.path.insert(0, _HERE_)

import os.path as osp
import time

import numpy as np
import torch

import typed_ext_common as TC

METHOD = "GLADPro"
FORCE_DELPROTO_EPOCH = int(TC.os.environ.get("EP_TEST_FORCE_DELPROTO_EPOCH", "0"))   # resume tests only (smoke)


def run(R, a):
    M = TC.load_port(METHOD)                        # ext/GLADPro/run_method.py (read only)
    H = TC.load_harness()                           # harness helpers (rng_isolated), read only
    metric = TC.load_metric()
    utils, layers, model_mod = M.load_gladpro()
    deterministic = not torch.cuda.is_available()
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)     # port D8 (CPU summation order)
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn.models.autoencoder import GAE

    # CHANGE-TYPED: typed dataset name -> no README line -> modal README configuration (A15)
    args, hp_line, hp_notes = M.resolve_hparams(utils, R.generator, "ubgold")
    assert hp_line.startswith("MODAL README config"), hp_line
    if R.max_epochs is not None:
        args.epochs = R.max_epochs
    device = args.device
    seed = 3407 + R.seed
    # ---- CHANGE-TYPED: typed data; set_seed(3407 + seed) (UB-GOLD set_seed, as the port's load_fold) -----------
    ns = H.load_ubgold_data_functions()
    ns["set_seed"](seed)
    td = R.load_data()
    P = td.pyg()
    train_graphs, hold_list = P["fit"], P["hold"]
    keys = R.keys

    R.base_meta({"ext/GLADPro/run_method.py": osp.join(TC.PORTS, METHOD, "run_method.py"),
                 "harness/run_eval.py": osp.join(TC.HARNESS, "run_eval.py"),
                 "typed/ext/GLADPro_typed.py": osp.abspath(__file__)},
                M.GP_COMMIT, dict(vars(args)), hp_line, hp_notes, device=str(device),
                flags=dict(oracle_own="common epoch, max 5-seed mean test AUROC (first max), finite in all seeds",
                           deterministic_algorithms=deterministic, per_type_mode="own_set"))

    # ================= copy of the port's run_split() loop ===============================================
    input_dim = train_graphs[0].num_node_features
    batch_train = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    # CHANGE-TYPED: one unshuffled test loader per group (test normals + T)
    batch_test = {k: DataLoader(P["test"][k], batch_size=args.batch_size_test, shuffle=False) for k in keys}
    batch_hold = DataLoader(hold_list, batch_size=args.batch_size_test, shuffle=False)

    model = GAE(model_mod.GAD_Explainer(args=args, input_dim=input_dim)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train(batch_graphs, batch_num):
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

    def score_group(k):
        """CHANGE-TYPED: GLADPro's test pass on group k's test set (main.py:72-88 under no_grad)."""
        i = 0
        with torch.no_grad():
            for batch_graphs in batch_test[k]:
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
        assert np.array_equal(labels_.cpu().numpy().astype(np.int64), R.y[k])
        return logits_.detach().cpu().numpy().astype(np.float64), float(auc)

    state = R.load_state(device.type, torch.get_num_threads(), first_epoch=1)
    local_optim = float("inf")
    count = None
    start_epoch = 1
    if state is None:
        model.encoder.init_prototypes(batch_train)
    else:
        model.encoder.prototypes = torch.nn.Parameter(torch.empty(state["proto_shape"]).to(args.device))
        model.load_state_dict(state["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(state["opt"])
        model.encoder.assign_train = [x.to(args.device) for x in state["assign_train"]]
        local_optim, count = state["local_optim"], state["count"]
        start_epoch = R.restore_common()

    def payload():
        return dict(model=model.state_dict(), proto_shape=tuple(model.encoder.prototypes.shape),
                    opt=optimizer.state_dict(), assign_train=[x.detach().cpu() for x in model.encoder.assign_train],
                    local_optim=local_optim, count=count)

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
        R.train_loss.append(epoch_loss / len(batch_train))
        R.t_train += time.time() - t0

        t1 = time.time()
        model.eval()
        res = R.score_groups(score_group)                      # CHANGE-TYPED: per-group test passes (§0a)
        hs = []
        with H.rng_isolated(), torch.no_grad():
            for hb in batch_hold:
                hl, *_ = test(hb.to(args.device))
                hs.append(hl.detach().cpu().numpy().astype(np.float64))
        R.record(epoch, {k: res[k][0] for k in keys}, np.concatenate(hs),
                 rec_extra=dict(n_prot=int(model.encoder.prototypes.shape[0]), epoch_time_gladpro=epoch_time),
                 types_extra={k: dict(auroc_gladpro=res[k][1]) for k in keys}, t_eval0=t1, metric=metric)

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

        if FORCE_DELPROTO_EPOCH > 0 and epoch == FORCE_DELPROTO_EPOCH:   # resume tests only (port's test hook)
            assert R.max_epochs is not None
            model.encoder.del_proto(torch.cat(embs_lst), 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if R.maybe_stop(epoch, payload, final_epoch=args.epochs):
            return "incomplete"
    # ================= end of the run_split() copy ======================================================

    return R.finish(args.epochs, 1, expected_epochs=range(1, args.epochs + 1),
                    meta_extra=dict(final_n_prot=int(model.encoder.prototypes.shape[0])))


if __name__ == "__main__":
    TC.main(METHOD, run)
