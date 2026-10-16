"""Write jobs/tasks_gpu.txt and jobs/tasks_cpu.txt: the HC-GLAD E4 grid (UB9 + MUTAG) x 5 folds x {ubgold, hpfree},
longest first, split by device (NOTES.md "Port and device criteria": GPU if the memory estimate <= 30 GB, else CPU).

Memory model (NOTES.md observation 6): eval peak ~ 4*d*N^2 bytes (largest test / held-out batch), train ~ 8*d*N^2
(largest training batch), d = hidden_dim * num_layers. Cost model for the ordering only (label-free node counts):
work ~ epochs * sum_b N_b^2 * d (training) + n_eval * (sum over test and held-out batches N_b^2 * d), with
sum_b N_b^2 ~ (total nodes)^2 / n_batches; if measured timings exist (finished parts or smoke parts with the same
config) they replace the model. Finished runs are skipped by the pool anyway.
    python3 -B jobs/gen_grid_tasks.py     (ubgold-gpu python)
"""
import glob, json, math, os, os.path as osp, sys
import numpy as np
HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
import run_method as R
DS10 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "MUTAG"]
GPU_LIMIT_GB = 30.0
SEC_PER_ELT = {"cpu": (2.3e-9, 1.5e-9), "cuda": (4e-11, 3e-11)}   # (train fwd+bwd, eval fwd) per N^2*d element, 1 call
OVERHEAD = {"cpu": 0.02, "cuda": 0.05}                          # s per batch (gc.collect, collation, small layers)


def node_stats(DS):
    from torch_geometric.datasets import TUDataset
    ds = TUDataset(osp.join(R.DATA_ROOT, "data", DS), name=DS)
    n = np.array([int(ds.get(i).num_nodes) for i in range(len(ds))]) if DS in ("IMDB-BINARY", "REDDIT-BINARY") \
        else np.diff(ds.slices["x"].numpy())
    y = ds.y.numpy()
    J = json.load(open(osp.join(R.SPLITS, f"{DS}.json")))
    out = []
    for f in J["folds"]:
        tr = np.array(f["split_train_index"]); te = np.array(f["split_test_index"])
        trn = tr[y[tr] != 0]
        n_all = len(trn); n_hold = int(round(0.2 * n_all))
        perm = np.random.RandomState(2026 + 1000 * f["fold"]).permutation(n_all)
        hold = trn[sorted(perm[:n_hold])]; fit = trn[sorted(perm[n_hold:])]
        out.append(dict(fit_n=n[fit], hold_n=n[hold], test_n=n[te]))
    return out


def batch_sq(nodes, bs, rs=None, reps=20):
    """mean of sum_b N_b^2 and max N_b over random batchings of size bs"""
    rs = rs or np.random.RandomState(0)
    tot, mx = [], 0
    for _ in range(reps):
        p = rs.permutation(len(nodes)) if len(nodes) > bs else np.arange(len(nodes))
        s = np.array([nodes[p[i:i + bs]].sum() for i in range(0, len(nodes), bs)], dtype=float)
        tot.append((s ** 2).sum()); mx = max(mx, s.max())
    return float(np.mean(tot)), float(mx)


def main():
    rows = []
    for DS in DS10:
        st = node_stats(DS)
        for hp in ("ubgold", "hpfree"):
            a, _, _ = R.resolve_hparams(DS, hp)
            d = a.hidden_dim * a.num_layers
            n_eval = a.num_epoch // a.eval_freq
            for f, s in enumerate(st):
                tr_sq, tr_mx = batch_sq(s["fit_n"], a.batch_size, reps=50)
                te_sq, te_mx = batch_sq(s["test_n"], a.batch_size_test, reps=50)
                ho_sq, ho_mx = batch_sq(s["hold_n"], a.batch_size_test, reps=1)      # fixed order (A3)
                hb_sq, hb_mx = batch_sq(s["hold_n"], a.batch_size, reps=1)
                gb_eval = 4 * d * max(te_mx, ho_mx, hb_mx) ** 2 / 1e9
                gb_train = 8 * d * tr_mx ** 2 / 1e9
                peak = max(gb_eval, gb_train)
                dev = "cuda" if peak <= GPU_LIMIT_GB else "cpu"
                if DS == "NCI1" and hp == "ubgold" and str(f) in os.environ.get("NCI1_GPU_FOLDS", "3,4").split(","):
                    dev = "cuda"   # NOTES Addendum 3: full test batch fits an exclusive A100 for these folds
                ct, ce = SEC_PER_ELT[dev]
                nb_tr = math.ceil(len(s["fit_n"]) / a.batch_size)
                nb_ev = math.ceil(len(s["test_n"]) / a.batch_size_test) + math.ceil(len(s["hold_n"]) / a.batch_size_test) \
                    + math.ceil(len(s["hold_n"]) / a.batch_size)
                est = a.num_epoch * (2 * ct * d * tr_sq + OVERHEAD[dev] * nb_tr) + \
                    n_eval * (2 * ce * d * (te_sq + ho_sq + hb_sq) + OVERHEAD[dev] * nb_ev) + 30
                rows.append(dict(ds=DS, hp=hp, fold=f, dev=dev, est=est, gb_eval=gb_eval, gb_train=gb_train, peak=peak, d=d))
    # measured timings (finished real parts) override the model per (ds, hp, dev)
    meas = {}
    for fn in glob.glob(osp.join(R.PARTS, "HC-GLAD_*.json")):
        m = json.load(open(fn))["meta"]
        if m.get("max_epochs_override") is None and len(m.get("segments", [])) == 1:
            meas.setdefault((m["dataset"], m["hp"], m["device"].split(":")[0]), []).append(m["seconds"])
    gpk = {}   # measured peak reserved GPU memory per (ds, d) from finished runs
    for fn in glob.glob(osp.join(R.PARTS, "HC-GLAD_*.json")):
        m = json.load(open(fn))["meta"]
        g = m.get("memory", {}).get("cuda_max_reserved_gb")
        if g:
            k = (m["dataset"], int(m["hparams"]["hidden_dim"]) * int(m["hparams"]["num_layers"]))
            gpk[k] = max(gpk.get(k, 0.0), g)
    for r in rows:
        k = (r["ds"], r["hp"], r["dev"])
        if k in meas:
            r["est"] = float(np.mean(meas[k])); r["measured"] = True
        r["gpk"] = gpk.get((r["ds"], r["d"]))
    rows.sort(key=lambda r: (-r["est"], r["ds"], r["hp"], r["fold"]))
    # NCI1 ubgold on GPU needs an empty A100: put it first so smaller tasks cannot starve it (longest remaining work)
    rows.sort(key=lambda r: 0 if (r["ds"] == "NCI1" and r["hp"] == "ubgold" and r["dev"] == "cuda") else 1)
    # measured GPU wall times (chunks 2-4) differ from the cost model; heaviest remaining cells first
    prio = [("NCI1", "ubgold"), ("DHFR", "ubgold"), ("NCI1", "hpfree"), ("ENZYMES", "ubgold"), ("PROTEINS_full", "hpfree"),
            ("DHFR", "hpfree"), ("ENZYMES", "hpfree"), ("AIDS", "hpfree"), ("REDDIT-BINARY", "ubgold")]
    rows.sort(key=lambda r: prio.index((r["ds"], r["hp"])) if (r["ds"], r["hp"]) in prio else len(prio))
    for dev, fn in (("cuda", "tasks_gpu.txt"), ("cpu", "tasks_cpu.txt")):
        sel = [r for r in rows if r["dev"] == dev]
        with open(osp.join(HERE, fn), "w") as fh:
            fh.write(f"# HC-GLAD E4 grid, {dev} tasks, longest first; generated by jobs/gen_grid_tasks.py; read at job start\n")
            for r in sel:
                if dev == "cuda":
                    th, mem, gmem = 2, 6, round(1.2 * r["peak"] + 1.5, 1)
                    if r.get("gpk"):   # measured peak of a finished run with the same dataset and width
                        gmem = round(min(36.0, 1.15 * r["gpk"] + 1.0), 1)
                    if r["ds"] == "NCI1" and r["hp"] == "ubgold":   # exclusive A100 (NOTES Addendum 3)
                        th, mem, gmem = 2, 10, 36.0
                    if r["ds"] == "REDDIT-BINARY":   # shuffled 16-graph test batches: worst observed 27 GB
                        th, mem, gmem = 2, 16, max(gmem, 28.0)
                    extra = f"gmem={gmem}"
                else:
                    th = int(os.environ.get("NCI1_THREADS", "12"))   # CPU NCI1 ubgold folds
                    mem, extra = round(1.15 * r["peak"] + 4, 1), "cpu"
                fh.write(f"HC-GLAD {r['ds']} {r['fold']} {r['hp']} {extra} mem={mem} threads={th}   "
                         f"# est {r['est']/60:.1f} min{' (measured)' if r.get('measured') else ''}; d={r['d']} peak eval {r['gb_eval']:.1f} GB train {r['gb_train']:.1f} GB\n")
        print(f"{fn}: {len(sel)} tasks; sum est {sum(r['est'] for r in sel)/3600:.1f} h; longest {sel[0]['est']/3600:.2f} h ({sel[0]['ds']} {sel[0]['hp']})" if sel else f"{fn}: 0 tasks")


if __name__ == "__main__":
    main()
