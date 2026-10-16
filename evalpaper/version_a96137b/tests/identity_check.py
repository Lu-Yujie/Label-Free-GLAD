"""A17 smoke S3 (NOTES.md §4): one tagged run of the c940457 harness OR of the a96137b runner, same seed, CPU, with
hooks that record (a) the initial model state, (b) every training-forward input tensor and the global CPU RNG state at
that call. Optional --sync_from: at the FIRST training forward, set the RNG state to the one recorded by a c940457 run
(S3c). Test metrics stay in the smoke part JSON (never read or printed here).

  python -B tests/identity_check.py --which c940457|a96137b --method GLADC --dataset BZR --fold 0 --epochs 3 \
      --out <json> [--sync_from <c940457 json>]
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os.path as osp  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
A17 = osp.dirname(HERE)
RUN_EVAL = osp.join(osp.dirname(A17), "harness", "run_eval.py")
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_a96137b/smoke")
REC = dict(calls=[], init={})
SYNC = []


def th(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode() + t.numpy().tobytes()).hexdigest()


def sd_hash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode() + th(v).encode())
    return h.hexdigest()


def at_first_call(init):
    REC["init"].update(init)
    REC["rng_first_natural_hex"] = torch.get_rng_state().numpy().tobytes().hex()
    if SYNC:
        torch.set_rng_state(torch.from_numpy(np.frombuffer(bytes.fromhex(SYNC[0]), dtype=np.uint8).copy()))
    REC["rng_first_hex"] = torch.get_rng_state().numpy().tobytes().hex()


def rng_hash():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def install_hooks(RE, method):
    orig_load = RE.load_ubgold

    def load_ubgold(m):
        M = orig_load(m)
        if method == "GLADC":
            cls = M["cls"]
            f0 = cls.forward_model

            def fm(self, adj_matrixs, adj_labels, graph_appends, args=None):
                if torch.is_grad_enabled():   # training calls (the held-out objective runs under no_grad)
                    if not REC["calls"]:
                        at_first_call(dict(NetGe=sd_hash(self.NetGe), noise_NetG=sd_hash(self.noise_NetG)))
                    REC["calls"].append(dict(adj=th(adj_matrixs), adj_label=th(adj_labels), h0=th(graph_appends),
                                             rng=rng_hash()))
                return f0(self, adj_matrixs, adj_labels, graph_appends, args)
            cls.forward_model = fm
        else:
            G = M["glocalkd"]
            s0, t0 = G.GcnEncoderGraph_student.forward, G.GcnEncoderGraph_teacher.forward

            def sf(self, x, adj, *a, **k):
                if self.training:            # training forward (the student is in eval mode when scoring)
                    if not REC["calls"]:
                        at_first_call(dict(student=sd_hash(self)))
                    REC["calls"].append(dict(adj=th(adj), h0=th(x), rng=rng_hash()))
                return s0(self, x, adj, *a, **k)

            def tf(self, x, adj, *a, **k):
                if "teacher" not in REC["init"]:   # first teacher call = first training forward (never trained)
                    REC["init"]["teacher"] = sd_hash(self)
                return t0(self, x, adj, *a, **k)
            G.GcnEncoderGraph_student.forward = sf
            G.GcnEncoderGraph_teacher.forward = tf
        return M
    RE.load_ubgold = load_ubgold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["c940457", "a96137b"], required=True)
    ap.add_argument("--method", choices=["GLADC", "GLocalKD"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sync_from", default=None)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    assert not torch.cuda.is_available(), "identity check is a CPU test (CUDA_VISIBLE_DEVICES='')"
    if a.sync_from:
        SYNC.append(json.load(open(a.sync_from))["rng_first_hex"])
    tag = {"c940457": "idc94", "a96137b": "ida96"}[a.which] + ("sync" if a.sync_from else "")
    if a.which == "c940457":
        spec = importlib.util.spec_from_file_location("ep_harness_run_eval", RUN_EVAL)
        RE = importlib.util.module_from_spec(spec)
        sys.modules["ep_harness_run_eval"] = RE
        spec.loader.exec_module(RE)
        RE.PARTS, RE.SCORES, RE.CKPT_ROOT = (osp.join(SMOKE, "parts"), osp.join(SMOKE, "scores"),
                                             osp.join(SMOKE, "ckpt"))
        install_hooks(RE, a.method)
        st = RE.run(a.method, a.dataset, a.fold, hp="ubgold", max_epochs=a.epochs, tag=tag, verbose=False)
    else:
        spec = importlib.util.spec_from_file_location("a17_runner", osp.join(A17, "run_eval_a96137b.py"))
        R = importlib.util.module_from_spec(spec)
        sys.modules["a17_runner"] = R
        spec.loader.exec_module(R)
        RE = R.configure("smoke", None)
        install_hooks(RE, a.method)
        st = R.run(a.method, a.dataset, a.fold, None, out_root="smoke", max_epochs=a.epochs, tag=tag)
        if a.method == "GLocalKD":   # S0: a96137b process_graph == the harness's verbatim c940457 copy (ast, no docstring)
            import ast
            import inspect
            import textwrap

            def body(fn):
                f = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
                b = f.body[1:] if isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant) else f.body
                b = [x for x in b if not isinstance(x, (ast.Import, ast.ImportFrom))]   # harness copy imports sp locally
                return ast.dump(ast.Module(body=b, type_ignores=[]))
            GK = sys.modules["GAOOD.detector.GLocalKD"].GLocalKD
            REC["process_graph_ast_equal_to_harness_copy"] = body(GK.process_graph) == body(RE.glocalkd_process_graph)
    rid = f"{a.method}_{a.dataset}_fold{a.fold}_{tag}"
    REC.update(which=a.which, method=a.method, dataset=a.dataset, fold=a.fold, epochs=a.epochs, status=st, run_id=rid,
               part=osp.join(SMOKE, "parts", rid + ".json"), npz=osp.join(SMOKE, "scores", rid + ".npz"),
               synced=bool(a.sync_from))
    with open(a.out, "w") as f:
        json.dump(REC, f)
    print(f"[identity] {a.which} {a.method} {a.dataset} fold {a.fold}: status {st}, {len(REC['calls'])} training "
          f"forwards recorded -> {a.out}", flush=True)
    sys.exit(0 if st == "done" else 3)


if __name__ == "__main__":
    main()
