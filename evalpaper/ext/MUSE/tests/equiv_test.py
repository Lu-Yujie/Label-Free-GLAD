"""Port-equivalence test (login node, < 1 min). Runs the adapter (tagged smoke run) and replays MUSE's own,
unmodified code on the same data and seeds:
 (a) snapshots of the adapter's instrumented Step-1 loop (with full rng-isolated evaluations in between) must
     equal the `parameters` returned by src.py MUSE_representation_learning.train();
 (b) the adapter's label-free MLP at the last snapshot must equal src.py train_MLP() run with an
     uninformative validation set (two copies of one graph, labels 0/1 -> constant AUROC 0.5, so
     EarlyStopper keeps updating and the last epoch's state is used), for the main cell (h=32, lr=1e-3):
     same test AUROC and same per-graph scores.
  EP_PARTS=... EP_SCORES=... EP_CKPT=... python -B tests/equiv_test.py DS FOLD HP
"""
import copy
import importlib.util
import json
import os
import os.path as osp
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

torch.set_num_threads(2)
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
spec = importlib.util.spec_from_file_location("rm", osp.join(HERE, "run_method.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)

DS, FOLD, HP = sys.argv[1], int(sys.argv[2]), sys.argv[3]
EPOCHS = 40
CAP = {}
orig = rm.muse_train


def wrapped(C, self, model, fh, eh, train_idxs, lr, wd, epochs, bs, seed, evaluate):
    snaps = []

    def ev(epoch, t1):
        snaps.append([copy.deepcopy(model.state_dict()), copy.deepcopy(fh.state_dict()), copy.deepcopy(eh.state_dict())])
        evaluate(epoch, t1)
    CAP.update(snaps=snaps, trainer=self, train_idxs=list(train_idxs), lr=lr, wd=wd, bs=bs, seed=seed,
               n_feat=model.convs[0].nn.layers[0].in_features)
    return orig(C, self, model, fh, eh, train_idxs, lr, wd, epochs, bs, seed, ev)


rm.muse_train = wrapped
tag = "eqtest"
assert rm.run("MUSE", DS, FOLD, hp=HP, max_epochs=EPOCHS, tag=tag, verbose=False) == "done"
run_id = f"MUSE_{DS}_fold{FOLD}" + ("_hpfree" if HP == "hpfree" else "") + f"_{tag}"
J = json.load(open(osp.join(rm.PARTS, run_id + ".json")))
Z = np.load(osp.join(rm.SCORES, run_id + ".npz"))

# ---- (a) MUSE's own train() on the same graphs/labels/seed -----------------------------------------------
G, S_ = rm.load_muse()["GNNs"], rm.load_muse()["src"]
H, _, _ = rm.resolve_hparams(DS, HP)
tr0 = CAP["trainer"]
graphs, adj, pw, device = tr0.datasets, tr0.labels, tr0.labels_pos_weights, tr0.device
seed, n_feat = CAP["seed"], CAP["n_feat"]
dim, n_layer = H["dim"], H["n_layers"]
torch.manual_seed(seed); torch.random.manual_seed(seed); np.random.seed(seed)
model = G.GIN(num_features=n_feat, num_classes=1, hidden_units=dim, num_layers=n_layer, dropout=0.3, mlp_layers=2,
              train_eps=False).to(device)
edge_encoder = G.MLP_Decoder(int(n_layer * dim), dim, dim).to(device)
feature_encoder = G.MLP_Decoder(int(n_layer * dim), dim, n_feat).to(device)
gnn_trainer = S_.MUSE_representation_learning(datasets=graphs, device=device, labels=adj, labels_pos_weights=pw)
loss, params = gnn_trainer.train(model=model, feature_head=feature_encoder, edge_head=edge_encoder, saving_interval=20,
                                 train_idxs=CAP["train_idxs"], lr=CAP["lr"], weight_decay=CAP["wd"], epochs=EPOCHS,
                                 batch_size=CAP["bs"], return_loss=True, seed=seed)
assert len(params) == len(CAP["snaps"]) == EPOCHS // 20
dmax = 0.0
for P_ref, P_ad in zip(params, CAP["snaps"]):
    for a, b in zip(P_ref, P_ad):
        for k in a:
            dmax = max(dmax, float((a[k].float() - b[k].float()).abs().max()))
tl = max(abs((l1 + l2) - t) for l1, l2, t in zip(loss[0], loss[1], J["train_loss"]))
print(f"(a) snapshots adapter vs MUSE train(): max|param diff| = {dmax}; max|train-loss diff| = {tl}")

# ---- (b) MUSE's own train_MLP with an uninformative validation set, last snapshot --------------------------
n_fit, n_hold = J["meta"]["n_fit"], J["meta"]["n_holdout"]
test_idx = list(range(n_fit + n_hold, len(graphs)))
y_test = Z["y_test"]
ae = S_.MUSE_oneclass_classification(model=model, feature_encoder=feature_encoder, edge_encoder=edge_encoder,
                                     datasets=graphs, device=device, B_size=CAP["bs"], labels=adj, pos_weights=pw)
torch.manual_seed(seed); torch.random.manual_seed(seed); np.random.seed(seed)
mlp = G.AutoEncoderOneclassClassifier(in_dim=4, hid_dim=32, n_layers=3, drop_p=0.0).to(device)
v = test_idx[0]
out = ae.train_MLP(encoder_param=params[-1], classifier=mlp, train_idxs=CAP["train_idxs"], valid_idxs=[v, v],
                   valid_labels=[0, 1], test_idxs=test_idx, test_labels=list(1 - y_test), lr=1e-3, epochs=500,
                   saving_interval=1, w_decay=1e-4, seed=seed)
stds = ae.return_variance(idxs=CAP["train_idxs"], classifier=mlp)
an_ref = -rm.muse_normality(ae, test_idx, mlp, stds)
r_last = J["traj"][-1]
print(f"(b) test AUROC: MUSE train_MLP {out[3]:.12f} vs adapter {r_last['auroc']:.12f}; "
      f"max|score diff| = {float(np.abs(an_ref - Z['test'][-1]).max())}")
print("PASS" if dmax == 0.0 and abs(out[3] - r_last["auroc"]) < 1e-12 and np.array_equal(an_ref, Z["test"][-1]) else "CHECK")
