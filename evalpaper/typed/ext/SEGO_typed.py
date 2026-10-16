"""Phase B typed adapter: SEGO (E5; AAAI'25) on GX / MS-v2 (RUN_NOTES_ext.md §0).

The port evalpaper/ext/SEGO/run_method.py (SEGO src @45bdda7, unmodified on disk; the port's three asserted
PyG-2.6 text patches are applied in memory by its load_sego) is imported READ-ONLY. Reused as is: load_sego (HCL, HRN,
HRNEncoder, tree_utli.GraphTransform, init_structural_encoding, arg_parse, setup_seed), resolve_hparams ('ubgold' =
released-modal: alpha 0.5, batch_size_test 128, other treecl.py defaults; A16 §1), chunks, rng_isolated. The port's
training loop (line-by-line copy of treecl.py's __main__ loop) is copied here with # CHANGE-TYPED marks:
  * data = typed_data. Training and held-out graphs get SEGO's own coding-tree pre-transform
    GraphTransform(tree_depth 5, max_degree 10) applied to the typed Data(x, edge_index, y) (deg_x = typed x + one-hot
    degree, the analogue of "raw TU x + degree"); test graphs are plain (SEGO's own test loader uses pre=False); x_s from
    SEGO's init_structural_encoding (rw/dg 16) on every graph; follow_batch for the tree keys (port deviation 5);
  * setup_seed(3407 + seed) before the data and the model (port: before ensure_prepared/load_fold);
  * at every evaluation point each group's test set (test normals + T) is scored through SEGO's shuffled test
    DataLoader (batch_size_test 128), unscrambled by graph id; groups 2..K from the saved RNG state S0, the first group
    last as the RNG-carrying set (RUN_NOTES §0a); held-out normals in the port's fixed chunks, RNG-isolated (A3).
Own ORACLE: none (SEGO's own reporting rule is LAST, A16 §3); the collector reports UB-GOLD's rule.
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B SEGO_typed.py --generator GX --seed 0
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

METHOD = "SEGO"
TREE_MAX_DEGREE = 10       # get_ad_dataset_TU(pre=True): degreeDim = 10


def run(R, a):
    M = TC.load_port(METHOD)                        # ext/SEGO/run_method.py (read only)
    S = M.load_sego()
    from torch_geometric.data import Batch, Data
    DataLoader = S["DataLoader"]
    HCL, HRN, HRNEncoder = S["HCL"], S["HRN"], S["HRNEncoder"]
    metric = TC.load_metric()
    import sklearn.metrics as skm
    impl = list(S["patches"])

    args, sh_line, hp_notes = M.resolve_hparams(S, R.generator, "ubgold")
    assert "released-modal" in sh_line, sh_line
    if R.max_epochs is not None:
        args.num_epoch = R.max_epochs
    seed = 3407 + R.seed
    S["setup_seed"](seed)

    # ---- CHANGE-TYPED: typed data; SEGO's tree pre-transform (train/held-out) and structural encodings ----------
    td = R.load_data()
    t_prep = time.time()
    P = td.pyg()
    keys = R.keys
    gt = S["tree_utli"].GraphTransform(args.tree_depth, TREE_MAX_DEGREE).transform
    enc = lambda gs: S["init_structural_encoding"](gs, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    def plain(d):
        g = Data(x=d.x, edge_index=d.edge_index, y=d.y)
        g.num_nodes = int(d.num_nodes)
        g.edge_attr = None
        return g

    def tree(d, idx=None):
        g = gt(plain(d))
        if idx is not None:
            g["idx"] = idx
        return g
    fit_list = enc([tree(d, i) for i, d in enumerate(P["fit"])])
    hold_list = enc([tree(d) for d in P["hold"]])
    test_lists = {k: enc([plain(d) for d in P["test"][k]]) for k in keys}
    t_prep = time.time() - t_prep
    dataset_num_features = td.num_feat
    tree_input_dim = int(fit_list[0].deg_x.shape[1])
    fb_keys = sorted(k for k in fit_list[0].keys() if k.find('treePHLayer') >= 0)
    dataloader = DataLoader(fit_list, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)
    n_train = len(fit_list)
    tests = {}
    for k in keys:
        for i, d in enumerate(test_lists[k]):
            d.gid = i
        tests[k] = DataLoader(test_lists[k], batch_size=args.batch_size_test, shuffle=True)   # SEGO's test loader
    hold_batches = [Batch.from_data_list(c, follow_batch=fb_keys)
                    for c in M.chunks(hold_list, min(args.batch_size_test, len(hold_list)))]
    n_hold = len(hold_list)
    eval_epochs = set(range(args.eval_freq, args.num_epoch + 1, args.eval_freq)) | {args.num_epoch}

    # ---- treecl.py: model / tree encoder / optimizers (verbatim order, as the port) ----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    encoder = HRNEncoder(args.tree_depth, args.tree_pooling_type,
                         tree_input_dim, args.tree_hidden_dim,
                         args.hidden_dim*args.num_layer, args.tree_dropout,
                         args.tree_link_input, args.tree_drop_root,
                         device)
    treeM = HRN(encoder, args.hidden_dim*args.num_layer).to(device)
    treeOpt = torch.optim.Adam(treeM.parameters(), lr=args.tree_learning_rate)
    hbs = [b.to(device) for b in hold_batches]

    R.base_meta({"ext/SEGO/run_method.py": osp.join(TC.PORTS, METHOD, "run_method.py"),
                 "ext/SEGO/src/treecl.py": osp.join(M.SRC, "treecl.py"),
                 "ext/SEGO/src/model.py": osp.join(M.SRC, "model.py"),
                 "ext/SEGO/src/tree_utli.py": osp.join(M.SRC, "tree_utli.py"),
                 "ext/SEGO/src/codingTree.py": osp.join(M.SRC, "codingTree.py"),
                 "typed/ext/SEGO_typed.py": osp.abspath(__file__)},
                M.SEGO_COMMIT, dict(vars(args)), sh_line, hp_notes, device=str(device),
                flags=dict(oracle_own=None, label="SEGO ('ubgold' = released-modal, A16 §1)",
                           own_reporting_rule="LAST (A16 §3)", tree_input_dim=tree_input_dim, seconds_tree_prep=t_prep,
                           impl="; ".join(impl + [f"device {device.type}"]), per_type_mode="own_set"),
                extra_versions=("numba",))

    state = R.load_state(device.type, torch.get_num_threads(), first_epoch=1)
    start_epoch = 1
    mean_b = std_b = mean_g = std_g = mean_n = std_n = None
    if state is not None:
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["opt"])
        treeM.load_state_dict(state["treeM"]); treeOpt.load_state_dict(state["treeOpt"])
        mean_b, std_b, mean_g, std_g, mean_n, std_n = state["adaptive"]
        start_epoch = R.restore_common()
    payload = lambda: dict(model=model.state_dict(), opt=optimizer.state_dict(), treeM=treeM.state_dict(),
                           treeOpt=treeOpt.state_dict(), adaptive=(mean_b, std_b, mean_g, std_g, mean_n, std_n))

    def sego_score(m, data):
        """treecl.py's test-time scoring, verbatim (port's sego_score)."""
        b, g_f, g_s, n_f, n_s = m(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
        y_score_g = m.calc_loss_g(g_f, g_s)
        y_score_n = m.calc_loss_n(n_f, n_s, data.batch)
        y_score_t = m.calc_loss_g(g_f, g_s)

        if args.is_adaptive:
            y_score = (y_score_t - mean_b)/std_b + (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
        else:
            y_score = y_score_t + y_score_g + y_score_n
        return y_score, y_score_g, y_score_n, g_f

    def score_group(k):
        """CHANGE-TYPED: treecl.py's test loop on group k's test set (shuffled loader), canonical order by gid."""
        y_score_all, y_true_all, gid_all, g_all, n_all_ = [], [], [], [], []
        for data in tests[k]:
            data = data.to(device)
            with torch.no_grad():
                y_score, y_score_g, y_score_n, _ = sego_score(model, data)
            y_score_all = y_score_all + y_score.detach().cpu().tolist()
            y_true_all = y_true_all + data.y.detach().cpu().tolist()
            gid_all += data.gid.cpu().tolist()
            g_all += y_score_g.cpu().tolist()
            n_all_ += y_score_n.cpu().tolist()
        try:
            auc = float(skm.roc_auc_score(y_true_all, y_score_all))      # SEGO's own printed number
        except ValueError:
            auc = float("nan")
        n = len(R.y[k])
        gids = np.asarray(gid_all, dtype=np.int64)
        assert len(gids) == n and len(np.unique(gids)) == n
        assert np.array_equal(R.y[k][gids], np.asarray(y_true_all, dtype=np.int64))
        out = [np.full(n, np.nan) for _ in range(3)]
        for o, v in zip(out, (y_score_all, g_all, n_all_)):
            o[gids] = np.asarray(v, dtype=np.float64)
        return out[0], out[1], out[2], auc

    # ---- treecl.py training loop (port's copy; CHANGE-TYPED in the evaluation block) ---------------------------
    for epoch in range(start_epoch, args.num_epoch + 1):
        t0 = time.time()
        weight_b = weight_g = weight_n = 1
        if args.is_adaptive:
            if epoch == 1:
                weight_b, weight_g, weight_n = 1, 1, 1
            else:
                weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                weight_sum = (weight_b  + weight_g  + weight_n) / 3
                weight_b, weight_g, weight_n = weight_b/weight_sum, weight_g/weight_sum, weight_n/weight_sum

        model.train()
        treeM.train()

        loss_all = 0
        if args.is_adaptive:
            loss_t_all, loss_g_all, loss_n_all = [], [], []

        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            treeOpt.zero_grad()

            b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
            x_hrn = treeM(data)
            loss_g = model.calc_loss_g(g_f, g_s)
            loss_n = model.calc_loss_n(n_f, n_s, data.batch)
            loss_t = model.calc_loss_tree(x_hrn, g_f)

            if args.is_adaptive:
                loss = weight_b * loss_t.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                loss_t_all = loss_t_all + loss_t.detach().cpu().tolist()
                loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
            else:
                loss = loss_t.mean() + loss_g.mean() + loss_n.mean()

            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        R.train_loss.append(loss_all / n_train)

        if args.is_adaptive:
            mean_b, std_b = np.mean(loss_t_all), np.std(loss_t_all)
            mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
            mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
        R.t_train += time.time() - t0

        if epoch in eval_epochs:
            t1 = time.time()
            model.eval()
            treeM.eval()
            res = R.score_groups(score_group)                # CHANGE-TYPED: per-group test passes (§0a)
            hs, hg, hn, objs = [], [], [], []
            with M.rng_isolated(), torch.no_grad():
                for hb in hbs:
                    s_, sg_, sn_, g_f = sego_score(model, hb)
                    x_hrn = treeM(hb)
                    l_t = model.calc_loss_tree(x_hrn, g_f)
                    objs.append(float(weight_b * l_t.mean() + weight_g * sg_.mean() + weight_n * sn_.mean()) * hb.num_graphs)
                    hs.append(s_.cpu().numpy()); hg.append(sg_.cpu().numpy()); hn.append(sn_.cpu().numpy())
            rec = dict(holdout_obj=sum(objs) / n_hold,
                       mean_b=float(mean_b) if mean_b is not None else None, std_b=float(std_b) if std_b is not None else None,
                       mean_g=float(mean_g) if mean_g is not None else None, std_g=float(std_g) if std_g is not None else None,
                       mean_n=float(mean_n) if mean_n is not None else None, std_n=float(std_n) if std_n is not None else None,
                       weights=[float(weight_b), float(weight_g), float(weight_n)])
            R.record(epoch, {k: res[k][0] for k in keys}, np.concatenate(hs), rec_extra=rec,
                     types_extra={k: dict(auroc_sego=res[k][3]) for k in keys},
                     extra_S=dict(hold_g=np.concatenate(hg), hold_n=np.concatenate(hn),
                                  **{f"test_g__{TC.skey(k)}": res[k][1] for k in keys},
                                  **{f"test_n__{TC.skey(k)}": res[k][2] for k in keys}),
                     t_eval0=t1, metric=metric)
        if R.maybe_stop(epoch, payload, final_epoch=args.num_epoch):
            return "incomplete"

    return R.finish(args.num_epoch, args.eval_freq, expected_epochs=sorted(eval_epochs))


if __name__ == "__main__":
    TC.main(METHOD, run)
