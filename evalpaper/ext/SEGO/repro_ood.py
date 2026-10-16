"""Port-fidelity check on SEGO's own OOD-detection protocol with an exactly released line (src/script/sego.sh),
e.g. `python treecl.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0.5`
(paper Table 1: BZR/COX2 96.66 ± 0.91). One process = one trial; the loop is treecl.py's, verbatim apart from
the PyG API patches of run_method.load_sego() and `dataset[0].keys` -> `.keys()` in get_ood_dataset.
Not part of the pre-registered protocols; results -> results/repro_ood.jsonl.

  python -B repro_ood.py --pair BZR+COX2 --trial 0 [--threads 8]
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]
import argparse
import ast
import json
import os
import os.path as osp
import shlex
import sys
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import run_method as rm  # noqa: E402


def released_line(pair):
    for t in rm.released_lines():
        if t[t.index("-DS_pair") + 1] == pair:
            return t
    raise KeyError(pair)


def load_get_ood_dataset(S):
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Constant
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import degree
    path = osp.join(rm.SRC, "data_loader.py")
    text = open(path).read()
    fn = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "get_ood_dataset"][0]
    code = ast.get_source_segment(text, fn)
    old = "for key in dataset[0].keys if"
    assert code.count(old) == 2
    code = code.replace(old, "for key in dataset[0].keys() if")   # CHANGE: PyG >= 2.4 Data.keys is a method
    ns = dict(os=os, osp=osp, torch=torch, np=np, DataLoader=DataLoader, Constant=Constant, TUDataset=TUDataset,
              degree=degree, init_structural_encoding=S["init_structural_encoding"],
              PygGraphPropPredDataset=None, __file__=rm.SEGO_FILE_ANCHOR)
    exec(compile(code, path + " [get_ood_dataset, keys()]", "exec"), ns)
    return ns["get_ood_dataset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BZR+COX2")
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    S = rm.load_sego()
    import sklearn.metrics as skm
    get_ood_dataset = load_get_ood_dataset(S)
    toks = released_line(a.pair)
    args = rm.sego_parse(S, toks)
    HCL, HRN, HRNEncoder = S["HCL"], S["HRN"], S["HRNEncoder"]
    t0 = time.time()
    trial = a.trial
    S["setup_seed"](trial + 1)                                   # treecl.py, verbatim from here
    dataloader, _, meta = get_ood_dataset(args, pre=True)
    _, dataloader_test, _ = get_ood_dataset(args)
    dataset_num_features = meta['num_feat']
    tree_input_dim = meta['deg_x']
    n_train = meta['num_train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    encoder = HRNEncoder(args.tree_depth, args.tree_pooling_type, tree_input_dim, args.tree_hidden_dim,
                         args.hidden_dim*args.num_layer, args.tree_dropout, args.tree_link_input,
                         args.tree_drop_root, device)
    treeM = HRN(encoder, args.hidden_dim*args.num_layer).to(device)
    treeOpt = torch.optim.Adam(treeM.parameters(), lr=args.tree_learning_rate)
    aucs = []
    for epoch in range(1, args.num_epoch + 1):
        if args.is_adaptive:
            if epoch == 1:
                weight_b, weight_g, weight_n = 1, 1, 1
            else:
                weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                weight_sum = (weight_b + weight_g + weight_n) / 3
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
        if args.is_adaptive:
            mean_b, std_b = np.mean(loss_t_all), np.std(loss_t_all)
            mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
            mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
        if epoch % args.eval_freq == 0:
            model.eval()
            treeM.eval()
            y_score_all = []
            y_true_all = []
            for data in dataloader_test:
                data = data.to(device)
                with torch.no_grad():     # CHANGE: values identical
                    b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                    y_score_g = model.calc_loss_g(g_f, g_s)
                    y_score_n = model.calc_loss_n(n_f, n_s, data.batch)
                    y_score_t = model.calc_loss_g(g_f, g_s)
                if args.is_adaptive:
                    y_score = (y_score_t - mean_b)/std_b + (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
                else:
                    y_score = y_score_t + y_score_g + y_score_n
                y_true = data.y
                y_score_all = y_score_all + y_score.detach().cpu().tolist()
                y_true_all = y_true_all + y_true.detach().cpu().tolist()
            auc = skm.roc_auc_score(y_true_all, y_score_all)
            aucs.append((epoch, float(auc)))
            print(f"[{a.pair} trial {trial}] epoch {epoch} AUC {auc:.4f} t={time.time()-t0:.0f}s", flush=True)
    rec = dict(time=time.strftime("%F %T"), job=os.environ.get("SLURM_JOB_ID"), pair=a.pair, trial=trial,
               line=" ".join(toks), last_auc=aucs[-1][1], aucs=aucs, n_train=n_train, seconds=time.time() - t0,
               threads=a.threads)
    os.makedirs(osp.join(HERE, "results"), exist_ok=True)
    with open(osp.join(HERE, "results", "repro_ood.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "aucs"}), flush=True)


if __name__ == "__main__":
    main()
