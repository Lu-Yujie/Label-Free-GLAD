"""Generate every M2 data set ONCE and store it as version-agnostic .npz (no pickle).

Run in the CPU venv ($GLAD_ENVS/grail-prelim) -- the environment that produced the GRAIL-0
synthetic results in results/synthetic_*.csv -- because the GPU venv has a different numpy
(1.26 vs 2.x) and networkx (3.6.1 vs 3.7), and gens.gx_graph draws its base graphs with
nx.barabasi_albert_graph. The GPU training code only reads the .npz files written here.

Generator: ./gens.py, a byte-identical copy of synthetic/gens.py (checked by cmp when copied);
parameters are the defaults used by synthetic/attrib.py, flip.py and bm.py:
  GX            gens.gx_set(seed)                      500 train normals; test 250 normals + 250 per type
  SC20-aX.XX    gens.syncom_set(20, 0.4, X.XX, seed)   500 train normals; test 250 + 250
  BM-MT/MN/MS   gens.bm_set(ds, seed)                  500 train normals; test 100 + 100
Seeds 0..4.

Fingerprint check against the meta JSONs written by the original runs
(results/synthetic_parts/{attrib,flip,bm}/*meta*.json): exact equality of the recorded
means (train n / m, anomaly m) and of the m-n set. Full identity is checked afterwards by
grail0_rerun.py, which re-scores GRAIL-0 on the exported sets and compares every AUROC.

Usage:  python -B export_data.py [--seeds 0 1 2 3 4]
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import gens as G  # noqa: E402  (copy of synthetic/gens.py)
from m2io import DATA as OUT, npz_path, load  # noqa: E402
SPARTS = osp.join(osp.dirname(HERE), "results", "synthetic_parts")
SEEDS = [0, 1, 2, 3, 4]
SYNCOM = [(20, 0.4, 0.6), (20, 0.4, 0.8), (20, 0.4, 0.2)]
BMS = ["BM-MT", "BM-MN", "BM-MS"]


def sc_id(n, tau0, taua):
    assert n == 20 and tau0 == 0.4
    return "SC20-a%.2f" % taua


def all_ids():
    return ["GX"] + [sc_id(*c) for c in SYNCOM] + BMS


def pack(graphs, pre):
    d = {}
    d[pre + "n"] = np.array([g["n"] for g in graphs], np.int64)
    d[pre + "m"] = np.array([len(g["edges"]) for g in graphs], np.int64)
    d[pre + "gid"] = np.array([g["gid"] for g in graphs], np.int64)
    d[pre + "edges"] = np.concatenate([np.asarray(g["edges"], np.int64).reshape(-1, 2) for g in graphs], 0)
    if all(g.get("x") is not None for g in graphs):
        d[pre + "x"] = np.concatenate([np.asarray(g["x"], np.float32) for g in graphs], 0)
    else:
        assert all(g.get("x") is None for g in graphs)
    if all(g.get("node_label") is not None for g in graphs):
        d[pre + "node_label"] = np.concatenate([np.asarray(g["node_label"], np.int64) for g in graphs], 0)
    else:
        assert all(g.get("node_label") is None for g in graphs)
    for k, dt in (("role", np.int64), ("gt_node", bool)):
        if all(k in g for g in graphs):
            d[pre + k] = np.concatenate([np.asarray(g[k], dt) for g in graphs], 0)
    return d


def generate(ds, seed):
    if ds == "GX":
        train, test_norm, anom = G.gx_set(seed)
        test = list(test_norm)
        group = [0] * len(test_norm)
        for k, t in enumerate(G.GX_TYPES, 1):
            test += anom[t]
            group += [k] * len(anom[t])
        names = ["normal"] + list(G.GX_TYPES)
    elif ds.startswith("SC20-a"):
        taua = float(ds.split("-a")[1])
        train, test, y = G.syncom_set(20, 0.4, taua, seed)
        group = list(map(int, y))
        names = ["normal", "taua%.2f" % taua]
    else:
        train, test, y = G.bm_set(ds, seed)
        group = list(map(int, y))
        names = ["normal", ds.replace("BM-", "")]
    return train, test, np.asarray(group, np.int64), names


def fingerprint_check(ds, seed, train, test, group):
    """Exact equality with what the original GRAIL-0 runs recorded about their generated sets."""
    if ds == "GX":
        meta = json.load(open(osp.join(SPARTS, "attrib", "s%d_meta.json" % seed)))
        got = {"n_mean": float(np.mean([g["n"] for g in train])),
               "m_minus_n": sorted(set(len(g["edges"]) - g["n"] for g in test))}
        ref = {"n_mean": meta["n_mean"], "m_minus_n": meta["m_minus_n"]}
    elif ds.startswith("SC20-a"):
        taua = float(ds.split("-a")[1])
        meta = json.load(open(osp.join(SPARTS, "flip", "main_n20_t0.40_a%.2f_s%d.json" % (taua, seed))))
        got = {"mean_m_train": float(np.mean([len(g["edges"]) for g in train])),
               "mean_m_test_anom": float(np.mean([len(g["edges"]) for g in test if g["gid"] >= 200000]))}
        ref = {k: meta[k] for k in got}
    else:
        meta = json.load(open(osp.join(SPARTS, "bm", "%s_s%d_meta.json" % (ds, seed))))
        got = {"n_train_mean": float(np.mean([g["n"] for g in train])),
               "m_train_mean": float(np.mean([len(g["edges"]) for g in train]))}
        ref = {k: meta[k] for k in got}
    ok = got == ref
    return ok, got, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--ds", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man_path = osp.join(HERE, "data_manifest.json")
    manifest = json.load(open(man_path)) if osp.exists(man_path) else {}
    bad = 0
    for ds in (a.ds or all_ids()):
        for seed in a.seeds:
            train, test, group, names = generate(ds, seed)
            assert all(g["gid"] < 100000 for g in train)
            assert np.all((group > 0) == np.array([g["gid"] >= 200000 for g in test]))
            ok, got, ref = fingerprint_check(ds, seed, train, test, group)
            bad += (not ok)
            d = pack(train, "tr_")
            d.update(pack(test, "te_"))
            d["te_group"] = group
            d["group_names"] = np.array(names)
            tmp = npz_path(ds, seed) + ".tmp.npz"
            np.savez_compressed(tmp, **d)
            os.replace(tmp, npz_path(ds, seed))
            # round trip
            tr2, te2, gr2, nm2 = load(ds, seed)
            for g1, g2 in zip(train + test, tr2 + te2):
                assert g1["n"] == g2["n"] and np.array_equal(np.asarray(g1["edges"]), g2["edges"])
                for k in ("x", "node_label", "role", "gt_node"):
                    v1, v2 = g1.get(k), g2.get(k)
                    assert (v1 is None and v2 is None) or np.array_equal(np.asarray(v1), v2), k
            assert np.array_equal(gr2, group) and nm2 == names
            h = hashlib.sha256()
            for k in sorted(d):
                h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
            manifest["%s_s%d" % (ds, seed)] = dict(
                file=npz_path(ds, seed), sha256_arrays=h.hexdigest(), n_train=len(train), n_test=len(test),
                test_counts={nm: int((group == k).sum()) for k, nm in enumerate(names)},
                n_nodes_max=int(max(g["n"] for g in train + test)),
                feature_keys=sorted(k for k in ("x", "node_label") if "tr_" + k in d),
                fingerprint_ok=bool(ok), fingerprint=got, fingerprint_ref=ref,
                numpy=np.__version__, networkx=__import__("networkx").__version__)
            print("%-12s s%d  train %d  test %s  max_n %d  fingerprint %s" % (
                ds, seed, len(train), manifest["%s_s%d" % (ds, seed)]["test_counts"],
                manifest["%s_s%d" % (ds, seed)]["n_nodes_max"], "OK" if ok else "MISMATCH %s vs %s" % (got, ref)),
                flush=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("fingerprint mismatches:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
