"""Verification of the SEGO port (login node, short; REDDIT-BINARY parts in the debug job).

  python -B verify.py prep   DS [DS ...]       build processed files (SEGO_PREP_PROCS workers)
  python -B verify.py treeid DS                sequential verbatim pre-transform == parallel (_parallel_map)
  python -B verify.py probe_fb DS              does SEGO's released AD loader run the tree encoder?
  python -B verify.py folds  DS [DS ...]       splits / features / order == UB-GOLD's loaders (harness)
  python -B verify.py runs                     determinism + resume equivalence (BZR fold 1, 20 epochs, CPU)

Results are appended as JSON lines to logs/verify.jsonl. Smoke runs write to
$GLAD_DATA_ROOT/ext_sego/smoke/ (never to ext/parts).
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]

import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import run_method as rm  # noqa: E402

LOG = osp.join(HERE, "logs", "verify.jsonl")
UBG_DATA_RUN = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_run")
SMOKE = osp.join(rm.EXT_ROOT, "smoke")
PY = sys.executable


def log(rec):
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.uname().nodename,
               job=os.environ.get("SLURM_JOB_ID"), **rec)
    os.makedirs(osp.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def base_args(S, DS):
    a, _, _ = rm.resolve_hparams(S, DS, "ubgold")
    return a


def cmd_prep(dss):
    S = rm.load_sego()
    for DS in dss:
        t = time.time()
        st = rm.ensure_prepared(S, base_args(S, DS), DS)
        p_plain, p_tree = rm.processed_paths(DS)
        log(dict(check="prep", dataset=DS, status=st, seconds=round(time.time() - t, 1), procs=rm.PREP_PROCS,
                 tree_file_mb=round(osp.getsize(p_tree) / 2 ** 20, 2)))


def tensors_equal(a, b):
    ka, kb = sorted(a.keys()), sorted(b.keys())
    if ka != kb:
        return False, f"keys {ka} != {kb}"
    for k in ka:
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            if not (torch.is_tensor(vb) and va.shape == vb.shape and va.dtype == vb.dtype and torch.equal(va, vb)):
                return False, k
        elif va != vb:
            return False, k
    return True, ""


def cmd_treeid(DS):
    """Build the tree dataset sequentially (verbatim loop, only the API patch) in a temporary root and
    compare every graph with the parallel-built file used by the runs."""
    S = rm.load_sego()
    aug, tu = S["aug"], S["tree_utli"]
    args = base_args(S, DS)
    rm.ensure_prepared(S, args, DS)
    tmp = tempfile.mkdtemp(dir=rm.EXT_ROOT, prefix="treeid_")
    try:
        os.makedirs(osp.join(tmp, DS, "raw"))
        for f in os.listdir(osp.join(rm.EXT_ROOT, "data", DS, DS, "raw")):
            shutil.copy(osp.join(rm.EXT_ROOT, "data", DS, DS, "raw", f), osp.join(tmp, DS, "raw", f))
        old = rm.PREP_PROCS
        rm.PREP_PROCS = 1
        t = time.time()
        seq = aug.TUDataset_aug(tmp, name=DS, pre_transform=tu.GraphTransform(args.tree_depth, 10).transform,
                                processed_filename="data_tree%s.pt" % args.tree_depth)
        t_seq = time.time() - t
        rm.PREP_PROCS = old
        if os.environ.get("SEGO_TREEID_PARALLEL_FRESH") == "1":   # also rebuild with the current parallel code
            tmp2 = osp.join(tmp, "par")
            os.makedirs(osp.join(tmp2, DS, "raw"))
            for f in os.listdir(osp.join(tmp, DS, "raw")):
                shutil.copy(osp.join(tmp, DS, "raw", f), osp.join(tmp2, DS, "raw", f))
            par = aug.TUDataset_aug(tmp2, name=DS, pre_transform=tu.GraphTransform(args.tree_depth, 10).transform,
                                    processed_filename="data_tree%s.pt" % args.tree_depth)
            used = osp.join(rm.EXT_ROOT, "data", DS)
            ref = aug.TUDataset_aug(used, name=DS, pre_transform=tu.GraphTransform(args.tree_depth, 10).transform,
                                    processed_filename="data_tree%s.pt" % args.tree_depth)
            bad_ref = [i for i in range(len(ref)) if not tensors_equal(ref[i], par[i])[0]]
            log(dict(check="treeid_fresh_parallel_vs_file_in_use", dataset=DS, procs=rm.PREP_PROCS,
                     identical=not bad_ref, mismatches=bad_ref[:5]))
        else:
            par = aug.TUDataset_aug(osp.join(rm.EXT_ROOT, "data", DS), name=DS,
                                    pre_transform=tu.GraphTransform(args.tree_depth, 10).transform,
                                    processed_filename="data_tree%s.pt" % args.tree_depth)
        assert len(seq) == len(par)
        bad = []
        for i in range(len(seq)):
            ok, why = tensors_equal(seq[i], par[i])
            if not ok:
                bad.append((i, why))
        log(dict(check="treeid", dataset=DS, n_graphs=len(seq), identical=not bad, mismatches=bad[:5],
                 seconds_sequential=round(t_seq, 1)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_probe_fb(DS):
    """SEGO's released AD training loader (no follow_batch) -> one tree-encoder forward."""
    S = rm.load_sego()
    args = base_args(S, DS)
    rm.ensure_prepared(S, args, DS)
    splits = S["get_ad_split_TU"](args, fold=args.num_trial)
    dataloader, _, meta = S["get_ad_dataset_TU"](args, splits[0], pre=True)
    enc = S["HRNEncoder"](args.tree_depth, args.tree_pooling_type, meta["deg_x"], args.tree_hidden_dim,
                          args.hidden_dim * args.num_layer, args.tree_dropout, args.tree_link_input,
                          args.tree_drop_root, "cpu")
    treeM = S["HRN"](enc, args.hidden_dim * args.num_layer)
    data = next(iter(dataloader))
    try:
        out = treeM(data)
        res = dict(released_loader_runs=True, out_shape=list(out.shape), num_graphs=int(data.num_graphs))
    except Exception as e:  # noqa: BLE001
        res = dict(released_loader_runs=False, error=f"{type(e).__name__}: {e}")
    fb = sorted(k for k in dataloader.dataset[0].keys() if "treePHLayer" in k)
    dl2 = S["DataLoader"](dataloader.dataset, batch_size=args.batch_size, shuffle=False, follow_batch=fb)
    d2 = next(iter(dl2))
    out2 = treeM(d2)
    res.update(with_follow_batch_shape=list(out2.shape), with_follow_batch_num_graphs=int(d2.num_graphs))
    # independent reference for the tree message passing (PyG 2.0.0 semantics of SEGO's HRNConv with
    # flow='target_to_source'): parent p receives sum_{(p, c) in treeEdgeMatLayer} x[c]; output rows = #parents
    treeM.eval()
    x = d2.deg_x
    maxdiff = 0.0
    with torch.no_grad():
        for l in range(args.tree_depth):
            ei = d2["treeEdgeMatLayer%s" % (l + 1)]
            n_par = int(d2.treeNodeSize[:, l + 1].sum())
            size = d2.treeNodeSize[:, [l + 1, l]].sum(dim=0)
            conv = treeM.encoder.convs[l]
            got = conv(x, ei, size=size)
            ref = conv.nn(torch.zeros(n_par, x.size(1)).index_add_(0, ei[0], x[ei[1]]))
            maxdiff = max(maxdiff, float((got - ref).abs().max()))
            assert got.shape[0] == n_par
            x = got
        # pooled layer batches: every graph contributes exactly its own tree nodes
        for l in range(1, args.tree_depth + 1):
            b = d2["treePHLayer%s_batch" % l]
            assert torch.equal(torch.bincount(b, minlength=int(d2.num_graphs)), d2.treeNodeSize[:, l])
    res.update(tree_conv_vs_reference_maxdiff=maxdiff, layer_batches_consistent=True)
    log(dict(check="probe_follow_batch", dataset=DS, **res))


def ubgold_loader_fns():
    """UB-GOLD's get_ad_split_TU / get_ad_dataset_TU / init_structural_encoding, extracted from the
    harness copy exactly as harness/run_eval.py does."""
    import re
    import random
    import networkx as nx
    from scipy import sparse as sp
    from sklearn.model_selection import StratifiedKFold
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Constant
    from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
    ns = dict(os=os, osp=osp, re=re, sp=sp, torch=torch, np=np, random=random, nx=nx,
              DataLoader=DataLoader, Constant=Constant, TUDataset=TUDataset,
              to_scipy_sparse_matrix=to_scipy_sparse_matrix, degree=degree, from_networkx=from_networkx,
              StratifiedKFold=StratifiedKFold)
    rm.extract_functions(osp.join(rm.UBG_SRC, "dataloader", "data_loader.py"),
                         ["get_ad_split_TU", "get_ad_dataset_TU", "init_structural_encoding",
                          "read_graph_file", "get_ad_dataset_Tox21"], ns)
    return ns


def same_graph(a, b, keys=("x", "x_s", "edge_index")):
    for k in keys:
        va, vb = a[k], b[k]
        if va.shape != vb.shape or not torch.equal(va.to(vb.dtype) if va.dtype != vb.dtype else va, vb):
            return k
    if int(a.y) != int(b.y):
        return "y"
    return None


def cmd_folds(dss):
    S = rm.load_sego()
    U = ubgold_loader_fns()
    for DS in dss:
        args = base_args(S, DS)
        rm.ensure_prepared(S, args, DS)
        t = time.time()
        cwd = os.getcwd()
        os.chdir(UBG_DATA_RUN)
        try:
            usplits = U["get_ad_split_TU"](args, fold=args.num_trial)
        finally:
            os.chdir(cwd)
        out = []
        full = [int(x) for x in os.environ.get("SEGO_VERIFY_FULL_FOLDS", "0,1,2,3,4").split(",")]
        for fold in range(5):
            if fold not in full:     # split-only check (features do not depend on the fold: whole dataset encoded)
                ref = json.load(open(osp.join(rm.SPLITS, f"{DS}.json")))["folds"][fold]
                s_spl = S["get_ad_split_TU"](args, fold=args.num_trial)[fold]
                ok = (list(map(int, s_spl[0])) == list(map(int, usplits[fold][0])) == list(map(int, ref["split_train_index"]))
                      and list(map(int, s_spl[1])) == list(map(int, usplits[fold][1])) == list(map(int, ref["split_test_index"])))
                out.append(dict(fold=fold, split_only=True, ok=ok))
                continue
            splits, dl_pre, dl_test, meta = rm.load_fold(S, args, DS, fold)   # asserts vs ubgold_splits JSON
            same_split = (list(map(int, splits[fold][0])) == list(map(int, usplits[fold][0]))
                          and list(map(int, splits[fold][1])) == list(map(int, usplits[fold][1])))
            os.chdir(UBG_DATA_RUN)
            try:
                uo = U["get_ad_dataset_TU"](args, usplits[fold], need_str_enc=True)
            finally:
                os.chdir(cwd)
            u_train, u_test = list(uo[3].dataset), list(uo[5].dataset)
            s_train, s_test = list(dl_pre.dataset), list(dl_test.dataset)
            bad_tr = [i for i, (a, b) in enumerate(zip(s_train, u_train)) if same_graph(a, b)]
            bad_te = [i for i, (a, b) in enumerate(zip(s_test, u_test)) if same_graph(a, b)]
            # held-out carve-out: same RandomState and n as harness/run_eval.py; compare with a harness npz
            n_all = len(s_train)
            n_hold = int(round(0.2 * n_all))
            perm = np.random.RandomState(rm.HOLDOUT_SEED + 1000 * fold).permutation(n_all)
            hold_pos = sorted(perm[:n_hold].tolist())
            import glob
            hzs = sorted(glob.glob(osp.join((os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper/scores"), f"*_{DS}_fold{fold}.npz")))
            harness_hold = None
            if hzs:     # any harness run of this (dataset, fold): all share the carve-out
                harness_hold = bool(np.array_equal(np.load(hzs[0])["hold_pos"], np.asarray(hold_pos)))
            rec = dict(fold=fold, same_split_as_ubgold_code=same_split, n_train=len(s_train), n_train_ub=len(u_train),
                       n_test=len(s_test), n_test_ub=len(u_test), train_mismatch=bad_tr[:5], test_mismatch=bad_te[:5],
                       num_feat=meta["num_feat"], num_feat_ub=uo[6]["num_feat"], deg_x_dim=meta["deg_x"],
                       hold_pos_equals_harness=harness_hold)
            rec["ok"] = (same_split and len(s_train) == len(u_train) and len(s_test) == len(u_test) and not bad_tr
                         and not bad_te and meta["num_feat"] == uo[6]["num_feat"] and harness_hold is not False)
            out.append(rec)
        log(dict(check="folds_features", dataset=DS, all_ok=all(r["ok"] for r in out), folds=out,
                 seconds=round(time.time() - t, 1)))


def run_smoke(DS, fold, hp, tag, max_epochs, extra_env=None, threads=2):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", EP_PARTS=osp.join(SMOKE, "parts"),
               EP_SCORES=osp.join(SMOKE, "scores"), EP_CKPT=osp.join(SMOKE, "ckpt"), CUDA_VISIBLE_DEVICES="")
    env.update(extra_env or {})
    cmd = [PY, "-B", osp.join(HERE, "run_method.py"), "--method", "SEGO", "--dataset", DS, "--fold", str(fold),
           "--hp", hp, "--max_epochs", str(max_epochs), "--tag", tag, "--threads", str(threads)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd="/tmp")
    return p.returncode, p.stdout[-3000:] + p.stderr[-3000:]


def load_run(tag_run_id):
    J = json.load(open(osp.join(SMOKE, "parts", tag_run_id + ".json")))
    Z = np.load(osp.join(SMOKE, "scores", tag_run_id + ".npz"))
    return J, Z


def compare(rid_a, rid_b):
    Ja, Za = load_run(rid_a)
    Jb, Zb = load_run(rid_b)
    skip = {"t"}
    md = 0.0
    keys_bad = []
    for ra, rb in zip(Ja["traj"], Jb["traj"]):
        for k in ra:
            if k in skip:
                continue
            va, vb = ra[k], rb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = abs(va - vb) if (va == va and vb == vb) else (0.0 if (va != va and vb != vb) else float("inf"))
                md = max(md, d)
            elif va != vb:
                keys_bad.append(k)
    zd = 0.0
    for k in Za.files:
        if Za[k].dtype.kind in "fi" and Za[k].shape == Zb[k].shape:
            zd = max(zd, float(np.max(np.abs(Za[k].astype(np.float64) - Zb[k].astype(np.float64)))) if Za[k].size else 0.0)
        elif Za[k].shape != Zb[k].shape:
            keys_bad.append("npz:" + k)
    tl = float(np.max(np.abs(np.asarray(Ja["train_loss"]) - np.asarray(Jb["train_loss"]))))
    return dict(max_abs_diff_traj=md, max_abs_diff_scores=zd, max_abs_diff_train_loss=tl, n_eval=len(Ja["traj"]),
                same_n_eval=len(Ja["traj"]) == len(Jb["traj"]), nonnumeric_mismatch=keys_bad)


def cmd_runs():
    DS, fold, E = "BZR", 1, 20
    for p in ("parts", "scores", "ckpt"):
        os.makedirs(osp.join(SMOKE, p), exist_ok=True)
    for tag in ("vdetA", "vdetB", "vresume"):
        for d, ext in (("parts", ".json"), ("scores", ".npz")):
            f = osp.join(SMOKE, d, f"SEGO_{DS}_fold{fold}_{tag}{ext}")
            if osp.exists(f):
                os.remove(f)
    rcA, outA = run_smoke(DS, fold, "ubgold", "vdetA", E)
    rcB, outB = run_smoke(DS, fold, "ubgold", "vdetB", E)
    det = compare(f"SEGO_{DS}_fold{fold}_vdetA", f"SEGO_{DS}_fold{fold}_vdetB") if rcA == rcB == 0 else None
    log(dict(check="determinism", dataset=DS, fold=fold, epochs=E, rc=[rcA, rcB], result=det,
             tail=None if rcA == rcB == 0 else (outA + outB)[-2000:]))
    rc1, out1 = run_smoke(DS, fold, "ubgold", "vresume", E, extra_env=dict(EP_TEST_STOP_EPOCH="10"))
    ck = osp.join(SMOKE, "ckpt", "SEGO", DS, f"SEGO_{DS}_fold{fold}_vresume.pt")
    had_ckpt = osp.exists(ck)
    rc2, out2 = run_smoke(DS, fold, "ubgold", "vresume", E, extra_env=dict(EP_TEST_STOP_EPOCH="10"))
    res = compare(f"SEGO_{DS}_fold{fold}_vdetA", f"SEGO_{DS}_fold{fold}_vresume") if rc2 == 0 else None
    J = json.load(open(osp.join(SMOKE, "parts", f"SEGO_{DS}_fold{fold}_vresume.json"))) if rc2 == 0 else None
    log(dict(check="resume_equivalence", dataset=DS, fold=fold, epochs=E, stop_epoch=10, rc=[rc1, rc2],
             checkpoint_written=had_ckpt, segments=(J["meta"]["segments"] if J else None), result=res,
             tail=None if rc2 == 0 else (out1 + out2)[-2000:]))
    # hpfree path (full-batch test, batch_size_test 9999) on MUTAG
    for f in (osp.join(SMOKE, "parts", "SEGO_MUTAG_fold0_hpfree_vhpfree.json"),):
        if osp.exists(f):
            os.remove(f)
    rc3, out3 = run_smoke("MUTAG", 0, "hpfree", "vhpfree", 10)
    J = json.load(open(osp.join(SMOKE, "parts", "SEGO_MUTAG_fold0_hpfree_vhpfree.json"))) if rc3 == 0 else None
    log(dict(check="hpfree_smoke", dataset="MUTAG", rc=rc3,
             hparams=None if J is None else {k: J["meta"]["hparams"][k] for k in ("batch_size", "batch_size_test", "alpha", "num_epoch")},
             n_eval=None if J is None else len(J["traj"]), tail=None if rc3 == 0 else out3[-2000:]))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("EP_THREADS", "2")))
    c = sys.argv[1]
    if c == "prep":
        cmd_prep(sys.argv[2:])
    elif c == "treeid":
        cmd_treeid(sys.argv[2])
    elif c == "probe_fb":
        cmd_probe_fb(sys.argv[2])
    elif c == "folds":
        cmd_folds(sys.argv[2:])
    elif c == "runs":
        cmd_runs()
    else:
        sys.exit(__doc__)
