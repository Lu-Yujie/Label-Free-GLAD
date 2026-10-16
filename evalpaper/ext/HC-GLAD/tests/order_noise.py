"""Diagnostic (uses test labels descriptively; never a protocol row, never used for any selection): how much does the
hyperedge-batching defect make the test AUROC depend on the order of the shuffled test loader?

For (DS, ubgold config, fold 0, seed 3407) it trains HC-GLAD with the verbatim main.py loop (on the 80% fit set, as the
grid runs) for E epochs, then scores the test set (one batch, batch_size_test 9999, as main.py) under K random
orders, (a) with PyG's released collation and (b) with a correctly offset hyperedge collation, and reports the
AUROC spread over orders. Output: one JSON line per dataset to stdout.
  python -B tests/order_noise.py DS EPOCHS K
"""
import json, os.path as osp, sys, time
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "tests"))
import numpy as np, torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import run_method as R
from hyperedge_defect import corrected

DS, E, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
metric = R.load_metric()
args, _, _ = R.resolve_hparams(DS, "ubgold")
mns = R.load_main_functions(); dfn = R.load_data_functions(DS)
HC, RiemannianSGD, _ = R.import_src_modules()
mns["setup_seed"](3407)
splits, dl, dlt, meta = R.load_fold(args, DS, 0, dfn)
train_all = list(dl.dataset); n_all = len(train_all); n_hold = int(round(0.2 * n_all))
perm = np.random.RandomState(2026).permutation(n_all)
fit = [train_all[i] for i in sorted(perm[n_hold:].tolist())]
dl = DataLoader(fit, batch_size=args.batch_size, shuffle=True)
args.feat_dim = meta["num_feat"]; args.lr_reduce_freq = args.lr_reduce_freq or args.num_epoch
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HC(args).to(dev)
opt = RiemannianSGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
for ep in range(E):
    model.train()
    for data in dl:
        data = data.to(dev); opt.zero_grad()
        g, gf, nf, gs, ns, gh, ghf, nhf, ghs, nhs = model.encode(data)
        loss = args.a * (model.calc_loss_g(gf, gs).mean() + model.calc_loss_n(nf, ns, data.batch).mean()) + \
            args.b * (model.calc_loss_g(ghf, ghs).mean() + model.calc_loss_n(nhf, nhs, data.batch).mean())
        loss.backward(); opt.step()
model.eval()
test = list(dlt.dataset); y = np.array([int(d.y) for d in test])
rs = np.random.RandomState(0)
out = {"released": [], "corrected": []}
with torch.no_grad():
    for k in range(K):
        p = rs.permutation(len(test)); gl = [test[i] for i in p]
        for name in ("released", "corrected"):
            b = Batch.from_data_list(gl)
            if name == "corrected":
                b = corrected(b, gl)
            b = b.to(dev)
            g, gf, nf, gs, ns, gh, ghf, nhf, ghs, nhs = model.encode(b)
            s = (args.a * (model.calc_loss_g(gf, gs) + model.calc_loss_n(nf, ns, b.batch))
                 + args.b * (model.calc_loss_g(ghf, ghs) + model.calc_loss_n(nhf, nhs, b.batch))).cpu().numpy()
            sc = np.empty(len(test)); sc[p] = s
            out[name].append(100 * metric.ood_auc(y, sc))
r = {k: np.array(v) for k, v in out.items()}
print(json.dumps(dict(dataset=DS, epochs=E, orders=K, device=str(dev),
                      released_mean=r["released"].mean(), released_sd=r["released"].std(), released_min=r["released"].min(),
                      released_max=r["released"].max(), corrected_mean=r["corrected"].mean(), corrected_sd=r["corrected"].std(),
                      seconds=time.time() - t0)), flush=True)
