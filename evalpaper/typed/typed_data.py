"""Common typed-data module for the A9 confirmatory typed test (PREREG_AMENDMENTS A9, A9.1, A25).

Generators
  GX  GLAD-X-lite exactly as exported for M2 (/N/slate/yujlu/grail_data/m2_data/GX_s{seed}.npz, read with
      m2_typed/m2io.py; array hashes checked against m2_typed/data_manifest.json). Anomaly groups T1a, T3a, T3b
      (A9 §2). Feature variants F2 (A9 decision; [one-hot(label, 4), x 2-d]) and F1 (sensitivity; one-hot only),
      built with m2_typed/convert.py.
  MS  MolSwap (frozen; typed/molswap.py, A9.1). Six cells (type, variant): MS1/MS2/MS3 x matched/sizechg, loaded with
      load_ms(..., allow_infeasible=True) as A25 §3 authorises. File hashes checked against the frozen manifest, whose
      own sha256 must equal the A9.1 value. Feature variant 'MS' = UB-GOLD's one-hot of the 37 NCI1 atom labels.
      RETIRED by A28 (instrument failure): kept loadable for the record; no deep detector is run on it.
  MSV2  MS-v2 (A28; typed/molswap_v2.py, rules in typed/NOTES_MSv2.md, freeze typed/FROZEN_MSv2.md).   # CHANGE-MSv2
      Six cells MS1/MS2/MS3 x matched/sizechg loaded with load_ms_v2 (all feasible under A25). File hashes checked
      against its manifest, whose own sha256 must equal MSV2_MANIFEST_SHA256. Feature variant 'MS' (37-label one-hot).
      CSV label: dataset 'MS' (the A9 generator slot read by stats/glad_stats.py); runs record generator 'MSV2'.

Held-out normals (VAL-NORMAL, A3; harness / M2 rule with fold -> seed):
  n_hold = round(0.2 * n_train); perm = np.random.RandomState(2026 + 1000 * seed).permutation(n_train);
  hold = sorted(perm[:n_hold]), fit = sorted(perm[n_hold:]).   GX: 500 -> 400 fit + 100 held out;
  MS: 982 -> 786 + 196.  The torch/np/random training seed is UB-GOLD's set_seed(3407 + seed) (callers do this).

Interface (also meant for the Phase B adapters under prelim/evalpaper/ext/<M>/):

    import sys; sys.path.insert(0, "/geode3/home/u110/yujlu/BigRed200/research/GAD/Grail/prelim/evalpaper")
    from typed.typed_data import load_typed, UNITS, GROUPS

    td = load_typed("MS", 0)               # or load_typed("GX", 0, "F2"); data checked against the frozen hashes
    td.train / td.fit / td.hold            # graph dicts (prelim/common format: n, edges (m,2) i<j, x, node_label, gid)
    td.fit_pos, td.hold_pos                # positions into td.train
    td.test_normals                        # graph dicts
    td.groups                              # OrderedDict: group key -> anomaly graph dicts
                                           #   GX keys 'T1a','T3a','T3b'; MS keys 'MS1:matched', 'MS1:sizechg', ...
    graphs, y = td.test_set(key)           # test normals + that group, canonical order (normals first), y 1 = anomaly
    graphs, y, grp = td.combined()         # test normals + every group (grp[i] = group key or 'normal')
    td.cell(key)                           # FROZEN.md columns: dataset, anomaly_type, size_variant, feature_variant,
                                           #   n_anomalies, n_normals
    td.num_feat, td.max_nodes_num          # UB-GOLD meta (num_feat: one-hot width; max over the generator's whole data)
    td.node_features(g)                    # float32 (n, num_feat) node features of one graph
    P = td.pyg(transform=None)             # fresh UB-GOLD TU-format torch_geometric Data lists:
        P['fit'] (y=0, idx = position in fit), P['hold'] (y=0), P['test'][key] / P['y'][key] (per group),
        P['combined'] / P['y_combined'] / P['group_combined'], P['meta'] (num_feat, max_nodes_num, num_train, ...)
        transform: optional callable(list_of_Data) -> list_of_Data applied to every list (e.g. UB-GOLD's
        init_structural_encoding for GOOD-D / CVTGAD).

Every data object returned is fresh per call (no state leaks between callers). Nothing here uses node labels to choose
anything. Run `python -B typed_data.py selftest` for the integrity checks (no detector involved).
"""
import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
import os.path as osp
from collections import OrderedDict

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
M2DIR = osp.join(PRELIM, "m2_typed")
for _p in (PRELIM, EVALPAPER, M2DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GENERATORS = ("GX", "MS", "MSV2")                                                   # CHANGE-MSv2: + MSV2
SEEDS = (0, 1, 2, 3, 4)
FEATURE_VARIANTS = {"GX": ("F2", "F1"), "MS": ("MS",), "MSV2": ("MS",)}             # CHANGE-MSv2: + MSV2
GX_TYPES = ("T1a", "T3a", "T3b")
MS_CELLS = (("MS1", "matched"), ("MS1", "sizechg"), ("MS2", "matched"), ("MS2", "sizechg"),
            ("MS3", "matched"), ("MS3", "sizechg"))
MSV2_CELLS = (("MS1", "matched"), ("MS1", "sizechg"), ("MS2", "matched"), ("MS2", "sizechg"),   # CHANGE-MSv2
              ("MS3", "matched"), ("MS3", "sizechg"))
GROUPS = {"GX": list(GX_TYPES), "MS": [f"{t}:{v}" for t, v in MS_CELLS],
          "MSV2": [f"{t}:{v}" for t, v in MSV2_CELLS]}                               # CHANGE-MSv2: + MSV2
UNITS = ([("GX", s, "F2") for s in SEEDS] + [("GX", s, "F1") for s in SEEDS] + [("MS", s, "MS") for s in SEEDS])
# CHANGE-MSv2: the A28 training units (MS-v2 replaces the retired MolSwap v1). UNITS is left unchanged for v1 tooling.
UNITS_A28 = ([("GX", s, "F2") for s in SEEDS] + [("GX", s, "F1") for s in SEEDS] + [("MSV2", s, "MS") for s in SEEDS])

HOLDOUT_SEED = 2026
HOLDOUT_FRAC = 0.2
N_LABELS_GX = 4
MS_ROOT = "/N/slate/yujlu/grail_data/molswap"
MS_MANIFEST_SHA256 = "123f0cda46c0b343285938071a10d73c7a6dda1d9e80914fef2cb2621c5aa6ea"   # A9.1
M2_MANIFEST = osp.join(M2DIR, "data_manifest.json")
MSV2_ROOT = "/N/slate/yujlu/grail_data/molswap_v2"                                   # CHANGE-MSv2
MSV2_MANIFEST_SHA256 = "7d7ddf76749425a74bed8d7e5c667b510318f6ef7b755a1224ee8d8dd73f2658"   # CHANGE-MSv2 (FROZEN_MSv2)
SIZE_VARIANT_CSV = {"matched": "matched", "sizechg": "changing"}   # stats/FROZEN.md typed schema


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def npz_array_hash(path):
    """export_data.py's sha256_arrays: sha256 over (key bytes, array bytes) for the sorted keys of the npz."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as Z:
        for k in sorted(Z.files):
            h.update(k.encode())
            h.update(np.ascontiguousarray(Z[k]).tobytes())
    return h.hexdigest()


def holdout_split(n_train, seed, frac=HOLDOUT_FRAC):
    """(fit_pos, hold_pos): the harness / M2 carve-out with fold -> seed."""
    n_hold = int(round(frac * n_train))
    perm = np.random.RandomState(HOLDOUT_SEED + 1000 * int(seed)).permutation(n_train)
    return sorted(perm[n_hold:].tolist()), sorted(perm[:n_hold].tolist())


def csv_cell(generator, key, feature_variant):
    """Group key -> the FROZEN.md typed-schema columns (dataset, anomaly_type, size_variant, feature_variant)."""
    if generator == "GX":
        assert key in GX_TYPES, key
        return dict(dataset="GX", anomaly_type=key, size_variant="matched", feature_variant=feature_variant)
    t, v = key.split(":")   # MS and MSV2 (CHANGE-MSv2: MSV2 rows carry dataset 'MS', the A9 generator slot)
    return dict(dataset="MS", anomaly_type=t, size_variant=SIZE_VARIANT_CSV[v], feature_variant="MS")


def _copy_graph(g):
    return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in g.items()}


class TypedData:
    """One (generator, seed, feature variant): training normals, held-out split and the per-group test sets."""

    def __init__(self, generator, seed, feature_variant, train, test_normals, groups, max_nodes_num, provenance):
        self.generator, self.seed, self.feature_variant = generator, int(seed), feature_variant
        self.train = train
        self.fit_pos, self.hold_pos = holdout_split(len(train), seed)
        self.fit = [train[i] for i in self.fit_pos]
        self.hold = [train[i] for i in self.hold_pos]
        self.test_normals = test_normals
        self.groups = groups
        self.max_nodes_num = int(max_nodes_num)
        self.num_feat = {"F1": N_LABELS_GX, "F2": N_LABELS_GX + 2, "MS": provenance["node_label_dim"]}[feature_variant]
        self.provenance = provenance

    # ------------------------------------------------------------------ graph-dict views
    def test_set(self, key):
        an = self.groups[key]
        graphs = [_copy_graph(g) for g in self.test_normals] + [_copy_graph(g) for g in an]
        y = np.r_[np.zeros(len(self.test_normals), np.int64), np.ones(len(an), np.int64)]
        return graphs, y

    def combined(self):
        graphs = [_copy_graph(g) for g in self.test_normals]
        grp = ["normal"] * len(self.test_normals)
        for k, an in self.groups.items():
            graphs += [_copy_graph(g) for g in an]
            grp += [k] * len(an)
        y = np.array([0 if x == "normal" else 1 for x in grp], np.int64)
        return graphs, y, grp

    def cell(self, key):
        d = csv_cell(self.generator, key, self.feature_variant)
        d.update(n_anomalies=len(self.groups[key]), n_normals=len(self.test_normals))
        return d

    # ------------------------------------------------------------------ features / PyG
    def node_features(self, g):
        if self.feature_variant in ("F1", "F2"):
            import convert   # m2_typed/convert.py (unchanged): F1 / F2 exactly as M2
            return convert.node_features(g, self.feature_variant)
        L = self.num_feat
        lab = np.asarray(g["node_label"], np.int64)
        assert lab.min() >= 0 and lab.max() < L, (lab.min(), lab.max(), L)
        oh = np.zeros((g["n"], L), np.float32)
        oh[np.arange(g["n"]), lab] = 1.0
        return oh

    def to_data(self, g, y, **extra):
        """One graph -> PyG Data as UB-GOLD's TUDataset / get_ad_dataset_TU hands it to the detectors
        (x float32 one-hot [+ x], edge_index both directions coalesced, edge_attr None, y int64 (1,)).
        Same construction as m2_typed/convert.to_data."""
        import torch
        from torch_geometric.data import Data
        from torch_geometric.utils import coalesce
        e = np.asarray(g["edges"], np.int64).reshape(-1, 2)
        ei = torch.from_numpy(np.concatenate([e, e[:, ::-1]], 0).T.copy())
        ei = coalesce(ei, num_nodes=g["n"])
        d = Data(x=torch.from_numpy(self.node_features(g)), edge_index=ei, y=torch.tensor([int(y)], dtype=torch.long))
        d.num_nodes = g["n"]
        for k, v in extra.items():
            d[k] = torch.tensor([int(v)], dtype=torch.long)
        return d

    def pyg(self, transform=None):
        tf = transform or (lambda lst: lst)
        fit = tf([self.to_data(g, 0, idx=i) for i, g in enumerate(self.fit)])
        hold = tf([self.to_data(g, 0, idx=p) for g, p in zip(self.hold, self.hold_pos)])
        test, ys = OrderedDict(), OrderedDict()
        for k in self.groups:
            graphs, y = self.test_set(k)
            test[k] = tf([self.to_data(g, yy) for g, yy in zip(graphs, y)])
            ys[k] = y
        graphs, y, grp = self.combined()
        comb = tf([self.to_data(g, yy) for g, yy in zip(graphs, y)])
        meta = dict(num_feat=self.num_feat, max_nodes_num=self.max_nodes_num, num_train=len(fit), num_edge_feat=0,
                    n_train_normal=len(self.train), n_fit=len(fit), n_hold=len(hold),
                    holdout_seed=HOLDOUT_SEED + 1000 * self.seed, generator=self.generator, seed=self.seed,
                    feature_variant=self.feature_variant, groups=list(self.groups))
        return dict(fit=fit, hold=hold, test=test, y=ys, combined=comb, y_combined=y, group_combined=grp, meta=meta)

    def summary(self):
        return dict(generator=self.generator, seed=self.seed, feature_variant=self.feature_variant,
                    n_train=len(self.train), n_fit=len(self.fit), n_hold=len(self.hold),
                    n_test_normals=len(self.test_normals), groups={k: len(v) for k, v in self.groups.items()},
                    num_feat=self.num_feat, max_nodes_num=self.max_nodes_num)


# ---------------------------------------------------------------------------------------------- loaders
def _load_gx(seed, feature_variant, verify=True):
    import m2io   # m2_typed/m2io.py (unchanged)
    assert feature_variant in FEATURE_VARIANTS["GX"], feature_variant
    path = m2io.npz_path("GX", seed)
    man = json.load(open(M2_MANIFEST))[f"GX_s{seed}"]
    prov = dict(source=path, m2_manifest=M2_MANIFEST, m2_manifest_sha256=sha256_file(M2_MANIFEST),
                sha256_arrays_expected=man["sha256_arrays"], node_label_dim=N_LABELS_GX)
    if verify:
        got = npz_array_hash(path)
        if got != man["sha256_arrays"]:
            raise RuntimeError(f"GX seed {seed}: array hash {got} != data_manifest {man['sha256_arrays']}")
        prov["sha256_arrays_checked"] = got
    train, test, group, names = m2io.load("GX", seed)
    assert len(train) == man["n_train"] and len(test) == man["n_test"]
    normals = [g for g, k in zip(test, group) if k == 0]
    groups = OrderedDict()
    for t in GX_TYPES:
        k = names.index(t)
        groups[t] = [g for g, kk in zip(test, group) if kk == k]
        assert len(groups[t]) == man["test_counts"][t]
    assert len(normals) == man["test_counts"]["normal"]
    max_nodes = max(g["n"] for g in list(train) + list(test))      # whole exported file (incl. T1b/T2) = M2's value
    assert max_nodes == man["n_nodes_max"]
    prov["max_nodes_scope"] = "whole exported GX file (train + all 5 test groups), as M2"
    return TypedData("GX", seed, feature_variant, train, normals, groups, max_nodes, prov)


def _ms_manifest(verify=True):
    path = osp.join(MS_ROOT, "manifest.json")
    if verify:
        got = sha256_file(path)
        if got != MS_MANIFEST_SHA256:
            raise RuntimeError(f"MolSwap manifest sha256 {got} != A9.1 {MS_MANIFEST_SHA256}")
    return json.load(open(path))


def _load_ms(seed, verify=True):
    from typed.molswap import load_ms   # frozen loader (A9.1)
    man = _ms_manifest(verify)
    files = man["seeds"][str(seed)]["files"]
    prov = dict(source=MS_ROOT, manifest_sha256=MS_MANIFEST_SHA256, node_label_dim=int(man["node_label_dim"]),
                load="load_ms(seed, type, variant, allow_infeasible=True) (A25 §3)", files_checked={})
    if verify:
        for rel, d in files.items():
            got = sha256_file(osp.join(MS_ROOT, rel))
            if got != d["sha256"]:
                raise RuntimeError(f"MolSwap {rel}: sha256 {got} != manifest {d['sha256']}")
            prov["files_checked"][rel] = got
    train = normals = None
    groups = OrderedDict()
    for t, v in MS_CELLS:
        tr, te, y = load_ms(seed, t, v, allow_infeasible=True)
        n_norm = int((y == 0).sum())
        assert np.all(y[:n_norm] == 0) and np.all(y[n_norm:] == 1)
        if train is None:
            train, normals = tr, te[:n_norm]
        else:   # the same training normals and test normals for every cell
            assert [g["gid"] for g in tr] == [g["gid"] for g in train]
            assert [g["gid"] for g in te[:n_norm]] == [g["gid"] for g in normals]
        groups[f"{t}:{v}"] = te[n_norm:]
        assert len(groups[f"{t}:{v}"]) == man["seeds"][str(seed)]["counts"][f"anom_{t}_{v}"]
    assert len(train) == files[f"seed{seed}/train.pkl"]["n_graphs"]
    assert len(normals) == files[f"seed{seed}/test_normal.pkl"]["n_graphs"]
    max_nodes = max(g["n"] for g in list(train) + list(normals) + [g for a in groups.values() for g in a])
    prov["max_nodes_scope"] = "all MolSwap graphs of this seed (train + test normals + six anomaly files)"
    return TypedData("MS", seed, "MS", train, normals, groups, max_nodes, prov)


def _msv2_manifest(verify=True):                                                     # CHANGE-MSv2
    path = osp.join(MSV2_ROOT, "manifest.json")
    if verify:
        got = sha256_file(path)
        if got != MSV2_MANIFEST_SHA256:
            raise RuntimeError(f"MS-v2 manifest sha256 {got} != frozen {MSV2_MANIFEST_SHA256}")
    return json.load(open(path))


def _load_msv2(seed, verify=True):                                                   # CHANGE-MSv2
    from typed.molswap_v2 import load_ms_v2   # frozen loader (FROZEN_MSv2.md)
    man = _msv2_manifest(verify)
    files = man["seeds"][str(seed)]["files"]
    prov = dict(source=MSV2_ROOT, generator_version="MS-v2 (A28)", manifest_sha256=MSV2_MANIFEST_SHA256,
                node_label_dim=int(man["node_label_dim"]), load="load_ms_v2(seed, type, variant)", files_checked={})
    if verify:
        for rel, d in files.items():
            got = sha256_file(osp.join(MSV2_ROOT, rel))
            if got != d["sha256"]:
                raise RuntimeError(f"MS-v2 {rel}: sha256 {got} != manifest {d['sha256']}")
            prov["files_checked"][rel] = got
    train = normals = None
    groups = OrderedDict()
    for t, v in MSV2_CELLS:
        tr, te, y = load_ms_v2(seed, t, v, root=MSV2_ROOT)
        n_norm = int((y == 0).sum())
        assert np.all(y[:n_norm] == 0) and np.all(y[n_norm:] == 1)
        if train is None:
            train, normals = tr, te[:n_norm]
        else:   # the same training normals and test normals for every cell
            assert [g["gid"] for g in tr] == [g["gid"] for g in train]
            assert [g["gid"] for g in te[:n_norm]] == [g["gid"] for g in normals]
        groups[f"{t}:{v}"] = te[n_norm:]
        assert len(groups[f"{t}:{v}"]) == man["seeds"][str(seed)]["counts"][f"anom_{t}_{v}"]
    assert len(train) == files[f"seed{seed}/train.pkl"]["n_graphs"]
    assert len(normals) == files[f"seed{seed}/test_normal.pkl"]["n_graphs"]
    max_nodes = max(g["n"] for g in list(train) + list(normals) + [g for a in groups.values() for g in a])
    prov["max_nodes_scope"] = "all MS-v2 graphs of this seed (train + test normals + six anomaly files)"
    return TypedData("MSV2", seed, "MS", train, normals, groups, max_nodes, prov)


def load_typed(generator, seed, feature_variant=None, verify=True):
    """-> TypedData for (generator, seed, feature variant). feature_variant defaults to the A9 decision variant
    (GX F2; MS 'MS')."""
    assert generator in GENERATORS, generator
    assert int(seed) in SEEDS, seed
    fv = feature_variant or FEATURE_VARIANTS[generator][0]
    assert fv in FEATURE_VARIANTS[generator], (generator, fv)
    if generator == "MSV2":                                                          # CHANGE-MSv2
        return _load_msv2(int(seed), verify)
    return _load_gx(int(seed), fv, verify) if generator == "GX" else _load_ms(int(seed), verify)


# ---------------------------------------------------------------------------------------------- self-test
def selftest(with_pyg=False):
    """Integrity checks only (no detector): hashes, counts, split sizes and disjointness, feature widths; with_pyg also
    checks that the GX PyG objects equal m2_typed/convert.build's and the MS one-hot equals node_label."""
    out = []
    for gen, s, fv in UNITS + [u for u in UNITS_A28 if u[0] == "MSV2"]:              # CHANGE-MSv2: + MSV2 units
        td = load_typed(gen, s, fv)
        S = td.summary()
        assert len(set(td.fit_pos) & set(td.hold_pos)) == 0 and len(td.fit_pos) + len(td.hold_pos) == len(td.train)
        gids_tr = {g["gid"] for g in td.train}
        gids_te = {g["gid"] for g in td.test_normals}
        assert not gids_tr & gids_te
        for k, an in td.groups.items():
            assert not ({g["gid"] for g in an} & (gids_tr | gids_te)), k
        if with_pyg and s == 0:
            import torch
            P = td.pyg()
            assert all(d.x.shape[1] == td.num_feat for d in P["fit"][:50])
            if gen == "GX":
                import convert
                import m2io
                tr, te, group, names = m2io.load("GX", s)
                dtr, dte, meta = convert.build(tr, te, group, fv)
                assert meta["max_nodes_num"] == td.max_nodes_num and meta["num_feat"] == td.num_feat
                ref = {i: d for i, d in enumerate(dtr)}
                for j, (d, p) in enumerate(zip(P["fit"], td.fit_pos)):
                    assert torch.equal(d.x, ref[p].x) and torch.equal(d.edge_index, ref[p].edge_index)
            else:
                for d, g in zip(P["combined"], td.combined()[0]):
                    assert torch.equal(d.x.argmax(1), torch.from_numpy(np.asarray(g["node_label"], np.int64)))
            S["pyg_checked"] = True
        out.append(S)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        for r in selftest(with_pyg="--pyg" in sys.argv):
            print(json.dumps(r))
        print("selftest OK")
