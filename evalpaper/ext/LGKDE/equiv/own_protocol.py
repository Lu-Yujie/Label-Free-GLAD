"""Reproduction under LGKDE's OWN protocol (investigation of reproduction misses; NOT part of the grid).

Step 1 (DGL env, login node): --prepare DS --seeds 42,43,44,45,46
  For each seed: set_global_seed(seed) and call the ORIGINAL src/dataprocessing.load_graph_dataset with the
  released config's dataset options (type GIN for MUTAG/PROTEINS/IMDB/NCI1/COLLAB as in the repo's
  test_all_datasets, TU otherwise; normal_class None = majority; train_ratio 0.8; self_loop True). Saves the
  train/test graphs (features as the original builds them, edges incl. self-loops), labels and the torch RNG
  state right after loading (the original's state when Trainer.init_components starts).
Step 2 (ubgold-gpu env, any device): --run DS --seed s
  Restores that RNG state, builds the port Trainer WITHOUT reseeding (as the original), trains with
  evaluation every epoch and the config's EarlyStopping on (AUROC+AUPR)/2, exactly like Trainer.train();
  reports the own-rule best epoch metrics. Output: equiv/own_protocol/<DS>_seed<s>.json
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
LG = osp.dirname(HERE)
SRC = osp.join(LG, "src")
OUTDIR = osp.join(HERE, "own_protocol")
DATADIR = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_lgkde/lgkde_own")
GIN_TYPE = {"MUTAG", "PROTEINS", "IMDBBINARY", "NCI1", "COLLAB"}
NAME = {"IMDB-BINARY": "IMDBBINARY", "PROTEINS_full": "PROTEINS"}


def prepare(ds, seeds):
    os.environ.setdefault("DGLBACKEND", "pytorch")
    sys.path.insert(0, SRC)
    import torch
    import yaml
    import dataprocessing as O_dp          # original, unmodified
    from utils import set_global_seed      # original
    cfg = yaml.safe_load(open(osp.join(SRC, "configs", "MUTAG.yaml")))
    name = NAME.get(ds, ds)
    dtype = "GIN" if name in GIN_TYPE else "TU"
    os.makedirs(DATADIR, exist_ok=True)
    for s in seeds:
        set_global_seed(s)
        tr, te, trn, full = O_dp.load_graph_dataset(dataset_name=name, dataset_type=dtype, normal_class=None,
                                                     train_ratio=cfg["dataset"]["train_ratio"], mixed_anomaly_ratio=0.0,
                                                     self_loop=cfg["dataset"]["self_loop"], batch_size=None)
        rng = torch.get_rng_state()

        def conv(dsx):
            gs, ys = [], []
            for i in range(len(dsx)):
                g, y = dsx[i]
                src, dst = g.edges()
                gs.append(dict(x=g.ndata["attr"].float().clone(), edge_index=torch.stack([src, dst]).long(), n=int(g.num_nodes())))
                ys.append(int(y))
            return gs, ys
        trg, try_ = conv(tr.dataset)
        teg, tey = conv(te.dataset)
        torch.save(dict(train=trg, test=teg, y_test=tey, y_train=try_, rng=rng, dataset_type=dtype, name=name, seed=s),
                   osp.join(DATADIR, f"{ds}_seed{s}.pt"))
        print(ds, s, dtype, "train", len(trg), "test", len(teg), "anomalies", sum(tey), "feat", trg[0]["x"].shape[1], flush=True)


def run(ds, seed, threads, max_epochs=None, lr=None):
    import numpy as np
    import torch
    sys.path.insert(0, LG)
    OVERLAY = (os.environ.get("GLAD_ENVS", "envs") + "/ext-lgkde-overlay")
    if OVERLAY not in sys.path:
        sys.path.append(OVERLAY)
    import yaml
    import lgkde_pyg as P
    import lgkde_trainer as PT
    torch.set_num_threads(threads)
    D = torch.load(osp.join(DATADIR, f"{ds}_seed{seed}.pt"), weights_only=False)
    cfg = yaml.safe_load(open(osp.join(SRC, "configs", "MUTAG.yaml")))
    if max_epochs:
        cfg["training"]["epochs"] = max_epochs
    tag = ""
    if lr is not None:   # post-hoc DIAGNOSTIC only (paper text says lr 0.001; released config 0.01)
        cfg["training"]["lr"] = lr
        tag = f"_lr{lr:g}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    P.DGMMD.block_elems = 2 ** 25 if device.type == "cuda" else 2 ** 20
    PT.build_graph = lambda g: dict(x=g["x"], edge_index=g["edge_index"])   # graphs already have self-loops

    class T(PT.Trainer):
        def __init__(self, config, train_graphs, test_graphs, test_labels, device):
            self.config, self.device = config, device
            torch.set_rng_state(D["rng"])         # the original's RNG state after load_graph_dataset
            self.setup_experiment(train_graphs, test_graphs, test_labels)
            self.init_components()

    t0 = time.time()
    tr = T(cfg, D["train"], D["test"], D["y_test"], device)
    es = PT.EarlyStopping(**{k: cfg["early_stopping"][k] for k in ("patience", "min_delta", "min_epochs")})
    y = np.asarray(D["y_test"])
    hist, best_i, best_perf, stop = [], None, 0, None
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss = tr.train_epoch(epoch)
        dens, _ = tr.evaluate()
        m = PT.metrics_lgkde(y, dens.detach().cpu().numpy())
        perf = (m["lgkde_auroc"] + m["lgkde_aupr"]) / 2
        hist.append(dict(epoch=epoch, loss=float(loss), **m, own_sel=perf))
        if perf > best_perf:
            best_perf, best_i = perf, len(hist) - 1
        print(f"[{ds} seed {seed}] ep {epoch} loss {loss:.4f} auroc {m['lgkde_auroc']:.4f} sel {perf:.4f}", flush=True)
        if es(epoch, perf):
            stop = epoch
            break
    out = dict(dataset=ds, seed=seed, lr=cfg["training"]["lr"], dataset_type=D["dataset_type"], device=str(device), seconds=time.time() - t0,
               n_train=len(D["train"]), n_test=len(D["test"]), n_test_anomalous=int(sum(D["y_test"])),
               stop_epoch=stop, best=hist[best_i], history=hist)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(osp.join(OUTDIR, f"{ds}_seed{seed}{tag}.json"), "w"), indent=1)
    print("BEST", ds, seed, json.dumps(hist[best_i]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare")
    ap.add_argument("--run")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="post-hoc diagnostic only")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.prepare, [int(s) for s in a.seeds.split(",")])
    if a.run:
        run(a.run, a.seed, a.threads, a.max_epochs, a.lr)
