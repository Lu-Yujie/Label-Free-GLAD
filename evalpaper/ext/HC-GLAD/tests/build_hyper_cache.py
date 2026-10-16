"""Build the REDDIT-BINARY hyperedge cache with HC-GLAD's verbatim get_hyperedge_index (utils/data_loader.py:153-205),
one graph per call, in parallel. Pure function of each graph's edge_index (args unused). Output:
$GLAD_DATA_ROOT/ext_hcglad/hyperedges/<DS>.pt = {hyperedge_index: [LongTensor 2 x E_i], num_nodes, edge_index,
seconds, commit}. Graph order = TUDataset order (= the data_list order inside get_ad_dataset_TU).
  python -B tests/build_hyper_cache.py REDDIT-BINARY --procs 32
"""
import argparse, os, os.path as osp, sys, time
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import torch
import run_method as R

_F = None


def _init():
    global _F, _DS
    torch.set_num_threads(1)
    _F = R.load_data_functions(None)["get_hyperedge_index"]   # verbatim (no cache for DS=None)


def _one(item):
    i, ei, n = item
    from torch_geometric.data import Data
    d = Data(edge_index=ei, num_nodes=n)
    t0 = time.time()
    _F([d], None)
    return i, d.hyperedge_index.clone(), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("DS")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import Constant
    import multiprocessing as mp
    t0 = time.time()
    ds = TUDataset(osp.join(R.DATA_ROOT, "data", a.DS), name=a.DS, transform=Constant(1, cat=False))
    items = [(i, ds[i].edge_index.clone(), int(ds[i].num_nodes)) for i in range(len(ds))]
    items.sort(key=lambda x: -x[2])            # largest graphs first (no long tail)
    out = [None] * len(items); secs = [0.0] * len(items)
    with mp.get_context("fork").Pool(a.procs, initializer=_init) as pool:
        for k, (i, he, s) in enumerate(pool.imap_unordered(_one, items, chunksize=1)):
            out[i] = he; secs[i] = s
            if k % 200 == 0:
                print(f"{k}/{len(items)} done, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(R.HYPER_CACHE, exist_ok=True)
    order = sorted(items, key=lambda x: x[0])
    payload = dict(hyperedge_index=out, num_nodes=[x[2] for x in order], edge_index=[x[1] for x in order],
                   seconds=secs, commit=R.REPO_COMMIT, built=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = osp.join(R.HYPER_CACHE, f"{a.DS}.pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, osp.join(R.HYPER_CACHE, f"{a.DS}.pt"))
    print(f"{a.DS}: {len(out)} graphs, sum of per-graph seconds {sum(secs):.0f} (max {max(secs):.1f}), wall {time.time()-t0:.0f}s", flush=True)
