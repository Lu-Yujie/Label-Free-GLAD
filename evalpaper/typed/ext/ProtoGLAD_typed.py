"""Phase B typed adapter: ProtoGLAD (E4; non-deep, checkpoint-free) on GX / MS-v2 (RUN_NOTES_ext.md §0).

The port evalpaper/ext/ProtoGLAD/run_method.py (ProtoGLAD src @f7d8ce9, unmodified, + the port's D1 CPU fallback
of IK_inne_gpu) is imported READ-ONLY. Reused as is: load_protoglad, to_protoglad (adjacency from edge_index, features
as float64), stage1 (UB-GOLD set_seed(3407 + seed), Isolation Kernel on the fit nodes, WL embeddings), stage2 (prototype
detector fitted on the fit graphs, anomaly score = -max prototype similarity, held-out scored RNG-isolated), grid_cells
(the repository's 144-cell search space). # CHANGE-TYPED: typed data; one run evaluates the whole grid on the unit:
  * stage 1 once per psi1 (h = 2), RNG state saved after it and restored before every (psi2, tau) cell, exactly as the
    port's grid replay (bit-identical to a from-scratch run of the cell, A20 §3);
  * the test graphs of stage 1 are the union of the test normals and all anomaly groups; every graph is mapped and
    scored on its own (IK transform and prototype similarity are per node / per graph and draw no random numbers), so
    a group's test set (test normals + T) is the slice of that pass; `--check_slice` (smoke) recomputes the README cell
    with one group's test set alone and reports the max |difference|.
Main configuration (rows oracle = last = valnormal; checkpoint-free, A20): the README eval-mode line h 2, psi1 64,
psi2 2, tau 0.01 (A20 §1). Own ORACLE (collector): A11/A20 test-grid-best: the first cell in product order whose 5-seed
mean test AUROC of that group is strictly larger than the running best (from 0.0); every seed reports that cell.
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B ProtoGLAD_typed.py --generator GX --seed 0
"""
import os.path as _osp, sys as _sys
_sys.dont_write_bytecode = True
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
if _HERE_ not in _sys.path:
    _sys.path.insert(0, _HERE_)

import os
import os.path as osp
import time

import numpy as np
import torch

import typed_ext_common as TC

METHOD = "ProtoGLAD"


def cell_name(c):
    h, psi1, psi2, tau = c
    return f"h{int(h)}_psi1_{int(psi1)}_psi2_{int(psi2)}_tau{float(tau):g}"


def run(R, a):
    os.environ.setdefault("TQDM_DISABLE", "1")      # port D5
    M = TC.load_port(METHOD)                        # ext/ProtoGLAD/run_method.py (read only)
    H = TC.load_harness()
    metric = TC.load_metric()
    P = M.load_protoglad()
    readme = dict(M.README_MUTAG)                   # README eval-mode line (A20 §1)
    readme_cell = (readme["h"], readme["psi1"], readme["psi2"], readme["tau"])
    cells = M.grid_cells()
    assert readme_cell in cells and len(cells) == 144
    fixed = dict(t1=P.cfg.DEFAULT_T1, t2=P.cfg.DEFAULT_T2, rho=P.cfg.DEFAULT_RHO, top_k=P.cfg.DEFAULT_TOP_K)
    seed = 3407 + R.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- CHANGE-TYPED: typed data (no RNG consumed) ------------------------------------------------------------
    td = R.load_data()
    Pg = td.pyg()
    keys, n_norm = R.keys, R.n_norm
    fit = [M.to_protoglad(d) for d in Pg["fit"]]
    hold = [M.to_protoglad(d) for d in Pg["hold"]]
    union = list(Pg["test"][keys[0]][:n_norm])
    pos = {}
    for k in keys:
        start = len(union)
        an = Pg["test"][k][n_norm:]
        union += list(an)
        pos[k] = np.r_[np.arange(n_norm), np.arange(start, start + len(an))].astype(np.int64)
    test = [M.to_protoglad(d) for d in union]
    y_union = np.r_[np.zeros(n_norm, np.int64), np.ones(len(union) - n_norm, np.int64)]
    for k in keys:
        assert np.array_equal(y_union[pos[k]], R.y[k])

    R.base_meta({"ext/ProtoGLAD/run_method.py": osp.join(TC.PORTS, METHOD, "run_method.py"),
                 "ext/ProtoGLAD/src/protoglad/detector.py": osp.join(M.PG_SRC, "protoglad", "detector.py"),
                 "ext/ProtoGLAD/src/protoglad/embedding.py": osp.join(M.PG_SRC, "protoglad", "embedding.py"),
                 "ext/ProtoGLAD/src/protoglad/ik_inne_gpu.py": osp.join(M.PG_SRC, "protoglad", "ik_inne_gpu.py"),
                 "harness/run_eval.py": osp.join(TC.HARNESS, "run_eval.py"),
                 "typed/ext/ProtoGLAD_typed.py": osp.abspath(__file__)},
                M.PG_COMMIT, dict(readme, **fixed),
                "ProtoGLAD: README eval-mode line h 2 psi1 64 psi2 2 tau 0.01 (the only released per-dataset choice, "
                "A20 §1) | grid: 144 cells (run_detection.py l.311-316) for the own rule",
                ["A11 'tuned' test-grid replay is used only as the own reporting rule (oracle_own), never as the main "
                 "configuration"], device=device,
                flags=dict(checkpoint_free=True, oracle_own="test-grid-best over 144 cells, 5-seed mean test AUROC, "
                           "first strictly larger (run_detection.py l.184, 259)", per_type_mode="union_slice",
                           grid_cells=[cell_name(c) for c in cells], readme_cell=cell_name(readme_cell)))
    R.load_state(device, torch.get_num_threads(), first_epoch=1)
    R.eval_epochs = {1}
    t0 = time.time()
    grid, grid_test, grid_hold, readme_res, slice_check = {}, [], [], None, {}
    if a.smoke_psi1:                                # smoke/timing only: a subset of the psi1 values
        assert R.smoke, "--smoke_psi1 is for smoke runs only"
        keep = {int(x) for x in a.smoke_psi1.split(",")} | {readme_cell[1]}
        cells = [c for c in cells if c[1] in keep]
        R.meta["flags"]["grid_cells"] = [cell_name(c) for c in cells]
        R.meta["flags"]["smoke_psi1"] = sorted(keep)
    by_psi1 = {}
    for c in cells:
        by_psi1.setdefault((c[0], c[1]), []).append(c)
    timings = {}
    for (h, psi1), cs in by_psi1.items():
        X_train, X_test, X_hold, tim1 = M.stage1(P, H, fit, hold, test, psi1, h, seed)
        timings[f"stage1_psi1_{psi1}"] = sum(tim1.values())
        t_s2 = time.time()
        st = H.rng_state()
        for c in cs:
            H.set_rng_state(st)
            res = M.stage2(P, H, X_train, X_test, X_hold, y_union, c[2], c[3])
            assert res is not None
            ts, hs = res["test_scores"], res["hold_scores"]
            ent = {"hold": TC.finite_mean(hs), "n_prototypes": int(res["n_prototypes"])}
            for k in keys:
                s = ts[pos[k]]
                m = TC.eval_metrics(metric, R.y[k], s)
                ent[k] = [m["auroc"], m["auprc"], m["fpr95"], m["nonfinite"], TC.n_distinct_finite(s)]
            grid[cell_name(c)] = ent
            grid_test.append(ts)
            grid_hold.append(hs)
            if c == readme_cell:
                readme_res = res
        timings[f"stage2_psi1_{psi1}"] = time.time() - t_s2
        print(f"[{R.run_id}] psi1 {psi1}: stage1 {timings[f'stage1_psi1_{psi1}']:.1f}s, {len(cs)} stage-2 cells "
              f"{timings[f'stage2_psi1_{psi1}']:.1f}s", flush=True)
        if a.check_slice and psi1 == readme_cell[1] and not slice_check:   # smoke: one group alone
            k = keys[-1]
            sub = [test[i] for i in pos[k]]
            Xtr, Xs, Xh, _ = M.stage1(P, H, fit, hold, sub, psi1, h, seed)    # same seed -> same IK fit
            r2 = M.stage2(P, H, Xtr, Xs, Xh, R.y[k], readme_cell[2], readme_cell[3])
            d = np.abs(r2["test_scores"] - readme_res["test_scores"][pos[k]]) if readme_res is not None else None
            slice_check.update(group=k, max_abs_diff=(float(np.max(d)) if d is not None else None))
    R.t_eval = time.time() - t0
    rec_extra = dict(grid=grid, n_prototypes=int(readme_res["n_prototypes"]), timings=timings,
                     auroc_protoglad_union=float(readme_res["auc"]))
    R.record(1, {k: readme_res["test_scores"][pos[k]] for k in keys}, readme_res["hold_scores"],
             rec_extra=rec_extra, metric=metric)
    return R.finish(1, 1, expected_epochs=[1],
                    extra_npz=dict(grid_test=np.stack(grid_test), grid_hold=np.stack(grid_hold),
                                   grid_cells=np.asarray(cells, dtype=np.float64), y_union=y_union,
                                   **{f"pos__{TC.skey(k)}": pos[k] for k in keys}),
                    meta_extra=dict(slice_check=slice_check or None))


if __name__ == "__main__":
    def _extra(ap):
        ap.add_argument("--check_slice", action="store_true", help="smoke: README cell with one group's test set alone")
        ap.add_argument("--smoke_psi1", default="", help="smoke/timing only: comma-separated psi1 subset (+ README's 64)")
    TC.main(METHOD, run, extra=_extra)
