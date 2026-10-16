"""Label-free test of the suspected hyperedge-batching defect (A11). No test labels, no test graphs.

For a dataset (ubgold configuration, fold 0, seed 3407) it takes the training normals of fold 0 and
  1. counts graphs with more hyperedges than nodes (#hyperedges > #nodes => id overlap with the next graph);
  2. collates them with PyG exactly as HC-GLAD's DataLoader does and counts hyperedge ids whose member nodes belong
     to >= 2 graphs (merged hyperedges), vs a correctly offset collation (row 1 shifted by cumulative #hyperedges);
  3. scores the same graphs (one batch, as the test loader does with batch_size_test 9999) in two different orders
     with the freshly initialised model, both with the released collation and with the corrected one, and reports
     the per-graph score change caused by the order (released) and by the defect itself (released vs corrected);
  4. reports whether any of this raises an error.
  python -B tests/hyperedge_defect.py DS [DS ...]
"""
import os.path as osp, sys, time
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, torch
from torch_geometric.data import Batch
import run_method as R
torch.set_num_threads(4)


def corrected(batch, graphs):
    """Same batch with hyperedge ids offset by the cumulative number of hyperedges (what an __inc__ override for
    'hyperedge_index' returning [[num_nodes], [num_hyperedges]] would give)."""
    rows, n_off, h_off = [], 0, 0
    for g in graphs:
        he = g.hyperedge_index
        rows.append(torch.stack([he[0] + n_off, he[1] + h_off]))
        n_off += int(g.num_nodes); h_off += int(he[1].max()) + 1
    b = batch.clone()
    b.hyperedge_index = torch.cat(rows, 1)
    return b


def merged_stats(batch, graph_of_node):
    he = batch.hyperedge_index
    gid = graph_of_node[he[0]]
    ids = torch.unique(he[1])
    n_multi = 0
    for i in ids.tolist():
        if torch.unique(gid[he[1] == i]).numel() > 1:
            n_multi += 1
    return n_multi, ids.numel()


if __name__ == "__main__":
  for DS in sys.argv[1:]:
        t0 = time.time()
        args, _, _ = R.resolve_hparams(DS, "ubgold")
        mns = R.load_main_functions(); dfn = R.load_data_functions(DS)
        HC, _, _ = R.import_src_modules()
        mns["setup_seed"](3407)
        splits, dl, dlt, meta = R.load_fold(args, DS, 0, dfn)
        graphs = list(dl.dataset)[: min(len(dl.dataset), 256, args.batch_size if args.batch_size < 128 else 256)]
        n = np.array([int(g.num_nodes) for g in graphs]); h = np.array([int(g.hyperedge_index[1].max()) + 1 for g in graphs])
        args.feat_dim = meta["num_feat"]
        model = HC(args); model.eval()
        order_b = list(range(len(graphs)))[::-1]
        res = {}
        with torch.no_grad():
            for name, gl in (("A", graphs), ("B", [graphs[i] for i in order_b])):
                b = Batch.from_data_list(gl)
                gon = b.batch
                nm, nh = merged_stats(b, gon)
                bc = corrected(b, gl)
                nmc, nhc = merged_stats(bc, gon)
                s_rel = model.encode(b); s_cor = model.encode(bc)
                def S(o, data_batch):
                    g_hBol, g_f, n_f, g_s, n_s, gh, gh_f, nh_f, gh_s, nh_s = o
                    return (args.a * (model.calc_loss_g(g_f, g_s) + model.calc_loss_n(n_f, n_s, data_batch))
                            + args.b * (model.calc_loss_g(gh_f, gh_s) + model.calc_loss_n(nh_f, nh_s, data_batch))).numpy()
                res[name] = dict(rel=S(s_rel, b.batch), cor=S(s_cor, bc.batch), merged=(nm, nh), merged_cor=(nmc, nhc))
        inv = np.argsort(order_b)
        relA, relB = res["A"]["rel"], res["B"]["rel"][inv]
        corA, corB = res["A"]["cor"], res["B"]["cor"][inv]
        print(f"{DS}: {len(graphs)} training graphs (fold 0); #hyperedges > #nodes in {np.mean(h > n):.2%} of graphs; "
              f"merged cross-graph hyperedges in one batch: released {res['A']['merged'][0]}/{res['A']['merged'][1]} (order A), "
              f"{res['B']['merged'][0]}/{res['B']['merged'][1]} (order B); corrected collation {res['A']['merged_cor'][0]}/{res['A']['merged_cor'][1]}", flush=True)
        print(f"   per-graph score S (fresh model): mean {relA.mean():.4f}; |order A - order B| released: max {np.abs(relA-relB).max():.4g} "
              f"mean {np.abs(relA-relB).mean():.4g}; corrected: max {np.abs(corA-corB).max():.3g}; "
              f"|released - corrected| (order A): max {np.abs(relA-corA).max():.4g} mean {np.abs(relA-corA).mean():.4g}; "
              f"Spearman(released, corrected) {np.corrcoef(np.argsort(np.argsort(relA)), np.argsort(np.argsort(corA)))[0,1]:.4f}; "
              f"no error raised; {time.time()-t0:.0f}s", flush=True)
