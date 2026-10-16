"""A33.2 structural-encoding ablation runner: CVTGAD and GOOD-D with x_s replaced by zeros of the same shape, in
training and in every scoring pass. NOTES.md §1-4.

Two paths, both importing the existing runners READ-ONLY (a33_common.load_harness / load_typed_runner):
  --path ub9    harness/run_eval.py run(): UB9 x 5 folds, UB-GOLD per-dataset lines ('ubgold');
  --path typed  typed/run_typed.py run(): GX (fv F2) and MSV2 (fv MS), seeds 0-4, modal UB-GOLD TU config.
Overridden in THIS process only (module attributes, # CHANGE-A33):
  * load_ubgold_data_functions -> harness function + init_structural_encoding replaced by a zeroing version
    (x_s = zeros(num_nodes, rw_dim + dg_dim), float32; shape/dtype asserted against UB-GOLD's function on the first
    graphs of the run);
  * load_ubgold -> harness function + forward pre-hook on every encoder_str (CVTGAD, GOOD_D, HCL) asserting an
    all-zero input of width rw_dim + dg_dim on every call (count -> meta.a33.xs_zero_checks);
  * fit_cvtgad / fit_goodd -> wrappers: meta.a33 + code hashes + planned-device check (loops unchanged);
  * output roots (harness PARTS/SCORES/CKPT_ROOT, typed roots()), checkpoint marker, crash records (A12).
Usage (ubgold-gpu env):
  PYTHONDONTWRITEBYTECODE=1 python -B run_ablation.py --path ub9 --method CVTGAD --dataset BZR --fold 0 --require_device cuda
  PYTHONDONTWRITEBYTECODE=1 python -B run_ablation.py --path typed --method GOOD-D --dataset GX:F2 --fold 3 --require_device cuda
  smoke: add --out_root smoke --max_epochs 4 --tag smoke
Exit codes: 0 done, 3 checkpointed/locked, 5 device mismatch / precondition, 4 crash (recorded).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import traceback  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
import a33_common as A  # noqa: E402

import torch  # noqa: E402

METHODS = ("CVTGAD", "GOOD-D")
UB9 = ("AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY")
TYPED_UNITS = {"GX:F2": ("GX", "F2"), "MSV2:MS": ("MSV2", "MS")}
ROOT = osp.join(A.DATA_ROOT, "ablation_xs")
OWN_FILES = {"followups/ablation_xs/run_ablation.py": osp.join(HERE, "run_ablation.py")}
VARIANTS = {"zeros": "x_s=0", "ones": "x_s=1"}   # 'zeros' = A33 as written (default); 'ones' = NOTES.md §7 option
_CFG = dict(xs_checks=0, xs_shape_check=None, xs_width=None, xs_variant="zeros")
_RE = []


def roots(path, out_root, xs_variant="zeros"):
    """(parts, scores, ckpt, crash) for path 'ub9' | 'typed' and x_s variant 'zeros' | 'ones'."""
    assert xs_variant in VARIANTS, xs_variant
    if out_root == "grid":
        b = osp.join(ROOT, xs_variant, path)
        return (osp.join(HERE, f"parts_{path}_{xs_variant}"), osp.join(b, "scores"), osp.join(b, "ckpt"),
                osp.join(b, "crash"))
    assert out_root == "smoke", out_root
    b = osp.join(ROOT, "smoke", xs_variant, path)
    return osp.join(b, "parts"), osp.join(b, "scores"), osp.join(b, "ckpt"), osp.join(b, "crash")


def zero_structural_encoding(orig):
    """CHANGE-A33: UB-GOLD's init_structural_encoding replaced by zeros (or, option 'ones', ones) of the same shape
    and dtype."""
    fill = torch.zeros if _CFG["xs_variant"] == "zeros" else torch.ones
    def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
        gs = list(gs) if not isinstance(gs, list) else gs
        if _CFG["xs_shape_check"] is None and len(gs):
            import copy
            probe = [copy.copy(g) for g in gs[:4]]
            ref = orig(probe, rw_dim=rw_dim, dg_dim=dg_dim)          # UB-GOLD's own x_s on 4 graphs
            shapes = [tuple(g["x_s"].shape) for g in ref]
            dtypes = {str(g["x_s"].dtype) for g in ref}
        else:
            shapes = dtypes = None
        for g in gs:
            g["x_s"] = fill(int(g.num_nodes), rw_dim + dg_dim, dtype=torch.float32)   # CHANGE-A33
        if shapes is not None:
            assert shapes == [tuple(g["x_s"].shape) for g in gs[:4]], (shapes, "x_s shape")
            assert dtypes == {"torch.float32"}, dtypes
            _CFG["xs_shape_check"] = dict(graphs=len(shapes), shapes_equal=True, dtype="torch.float32",
                                          width=rw_dim + dg_dim)
        _CFG["xs_width"] = rw_dim + dg_dim
        return gs
    return init_structural_encoding


def xs_pre_hook(mod, args):
    """Forward pre-hook on encoder_str: its input (x_s) must be all zero (all one for 'ones'), width rw_dim + dg_dim."""
    x = args[0]
    assert x.dim() == 2 and x.shape[1] == _CFG["xs_width"], ("x_s width", tuple(x.shape), _CFG["xs_width"])
    if _CFG["xs_variant"] == "zeros":
        assert not bool(x.any()), "A33 ablation: non-zero x_s reached encoder_str"
    else:
        assert bool((x == 1).all()), "A33 ablation: x_s != 1 reached encoder_str"
    _CFG["xs_checks"] += 1


def patch_model_classes():
    """Register xs_pre_hook on encoder_str whenever a CVTGAD / GOOD_D / HCL module is built."""
    for modname, clsnames in (("GAOOD.nn.cvtgad", ("CVTGAD",)), ("GAOOD.nn.good_d", ("GOOD_D", "HCL"))):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for cn in clsnames:
            cls = getattr(mod, cn)
            if getattr(cls.__init__, "_a33", False):
                continue
            orig_init = cls.__init__

            def __init__(self, *a, _orig=orig_init, **k):
                _orig(self, *a, **k)
                self.encoder_str.register_forward_pre_hook(xs_pre_hook)    # CHANGE-A33 (check only)
            __init__._a33 = True
            cls.__init__ = __init__


def configure(path, out_root, require_device, xs_variant="zeros"):
    if _RE:
        return _RE[0]
    RE = A.load_harness()
    parts, scores, ckpt, crash = roots(path, out_root, xs_variant)
    _CFG.update(path=path, out_root=out_root, require_device=require_device, crash_dir=crash, xs_variant=xs_variant)
    VARIANT = VARIANTS[xs_variant]
    orig_ldf = RE.load_ubgold_data_functions

    def load_ubgold_data_functions():                     # CHANGE-A33
        ns = orig_ldf()
        ns["init_structural_encoding"] = zero_structural_encoding(ns["init_structural_encoding"])
        return ns
    RE.load_ubgold_data_functions = load_ubgold_data_functions

    orig_lu = RE.load_ubgold

    def load_ubgold(method):                              # CHANGE-A33 (forward check)
        M = orig_lu(method)
        patch_model_classes()
        return M
    RE.load_ubgold = load_ubgold

    codes = [RE._run.__code__]
    if path == "ub9":
        RE.PARTS, RE.SCORES, RE.CKPT_ROOT = parts, scores, ckpt
        orig_run = RE._run

        def _run_with_crash_record(method, DS, fold, hp, run_id, out_json, ckpath, *rest):
            try:
                return orig_run(method, DS, fold, hp, run_id, out_json, ckpath, *rest)
            except Exception as e:
                A.record_crash(crash, run_id, e, ckpath, dict(a33="A33.2 x_s ablation (ub9)"))
                raise
        RE._run = _run_with_crash_record
        A.Progress.install(RE)
    else:
        RT = A.load_typed_runner()
        assert RT.load_harness() is RE
        RT.roots = lambda smoke: (parts, scores, ckpt, crash)     # CHANGE-A33: typed outputs -> A33 dirs
        codes.append(RT._run.__code__)
        _CFG["RT"] = RT
    _CFG["codes"] = tuple(codes)
    for name in ("fit_cvtgad", "fit_goodd"):
        setattr(RE, name, wrap_fit(RE, getattr(RE, name)))
    A.install_ckpt_marker(RE, dict(experiment="A33.2 x_s ablation", variant=VARIANT, path=path))
    _CFG["VARIANT"] = VARIANT
    _RE.append(RE)
    return RE


def wrap_fit(RE, fn):
    def fit(C):
        meta = A.find_meta_out(_CFG["codes"])
        meta.update(A.provenance(OWN_FILES))
        VARIANT = _CFG["VARIANT"]
        a33 = dict(experiment="A33.2 structural-encoding ablation", amendment=A.AMENDMENT, variant=VARIANT,
                   xs_variant=_CFG["xs_variant"],
                   change=f"x_s replaced by {_CFG['xs_variant']}(num_nodes, rw_dim + dg_dim) float32 at UB-GOLD's "
                          "init_structural_encoding (training, held-out and test scoring)",
                   path=_CFG["path"], planned_device=_CFG["require_device"], xs_shape_check=_CFG["xs_shape_check"],
                   xs_width=_CFG["xs_width"], runner="prelim/evalpaper/followups/ablation_xs/run_ablation.py",
                   out_root=_CFG["out_root"])
        meta["a33"] = a33
        assert _CFG["xs_shape_check"] is not None, "zeroing init_structural_encoding did not run"
        st = A.device_check(C, _CFG["require_device"], meta)
        if st:
            return st
        if C.state is not None and C.state.get("a33", {}).get("variant") != VARIANT:
            print("[resume] checkpoint variant mismatch; refusing", flush=True)
            return "device_mismatch"
        n0 = _CFG["xs_checks"]
        out = fn(C)
        a33["xs_zero_checks"] = _CFG["xs_checks"] - n0
        if out == "done":
            assert a33["xs_zero_checks"] > 0, "encoder_str was never called with the checked input"
        return out
    fit.__name__ = fn.__name__
    return fit


def run(path, method, unit, fold, require_device, out_root="grid", deadline=None, max_epochs=None,
        ckpt_every_s=300.0, tag="", xs_variant="zeros"):
    assert method in METHODS, method
    if out_root == "grid" and (tag or max_epochs is not None):
        raise SystemExit("tagged / --max_epochs runs must use --out_root smoke")
    have = "cuda" if torch.cuda.is_available() else "cpu"
    if require_device is not None and have != require_device:
        print(f"[device] task planned on {require_device}, this process sees {have}; not started", flush=True)
        return "device_mismatch"
    RE = configure(path, out_root, require_device, xs_variant)
    if path == "ub9":
        assert unit in UB9, unit
        return RE.run(method, unit, fold, hp="ubgold", deadline=deadline, max_epochs=max_epochs,
                      ckpt_every_s=ckpt_every_s, tag=tag, verbose=False)
    gen, fv = TYPED_UNITS[unit]
    RT = _CFG["RT"]
    # typed run(): smoke flag only marks max_epochs runs; outputs go to roots() (patched above) in both cases
    return RT.run(method, gen, fold, fv, deadline=deadline, max_epochs=max_epochs, tag=tag,
                  smoke=(out_root == "smoke"), ckpt_every_s=ckpt_every_s, verbose=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, choices=["ub9", "typed"])
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True, help="UB9 dataset (ub9) or GX:F2 / MSV2:MS (typed)")
    ap.add_argument("--fold", type=int, required=True, help="fold (ub9) or seed (typed)")
    ap.add_argument("--require_device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--out_root", choices=["grid", "smoke"], default="grid")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EP_THREADS", "2")))
    ap.add_argument("--ckpt_every_s", type=float, default=300.0)
    ap.add_argument("--xs_variant", choices=sorted(VARIANTS), default="zeros",
                    help="zeros = A33 as written (default); ones = NOTES.md §7 option (needs ratification)")
    a = ap.parse_args()
    if a.max_epochs is not None and not a.tag:
        sys.exit("--max_epochs requires --tag")
    torch.set_num_threads(a.threads)
    try:
        st = run(a.path, a.method, a.dataset, a.fold, a.require_device, out_root=a.out_root, deadline=a.deadline,
                 max_epochs=a.max_epochs, ckpt_every_s=a.ckpt_every_s, tag=a.tag, xs_variant=a.xs_variant)
    except Exception:
        traceback.print_exc()
        sys.exit(4)
    sys.exit(0 if st == "done" else (5 if st == "device_mismatch" else 3))
