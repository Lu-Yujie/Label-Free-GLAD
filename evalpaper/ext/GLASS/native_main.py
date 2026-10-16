"""DIAGNOSTIC ONLY (A10 §4 reproduction investigation; never a protocol row): run GLASS's OWN single-domain
pipeline -- train_single.py (README line: --epochs 150 --patience 40 --batch_size 64, seeds 42-46) and then
evaluate_reference.py -- on GLASS's own split (minority class anomalous, 80% of normals train) and its own features
(TUDataset use_node_attr=True), to check whether the released code regenerates the paper table.

Runtime overrides are PATHS ONLY (# CHANGE): data root, checkpoint / result roots, the text-embedding cache directory
and TrainingLogger's output directory, so that nothing is written into src/. Seeds run as separate processes
(--num_seeds 1 --seed_start S); train_single_run() re-seeds at its start, so this equals the 5-seed loop.

  python -B native_main.py prep  --dataset AIDS              # GLASS features + text embeddings (GPU)
  python -B native_main.py train --dataset AIDS --seed 42    # train_single.py, one seed
  python -B native_main.py eval  --datasets AIDS MUTAG       # evaluate_reference.py over seeds 42-46
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]

import argparse
import importlib
import json
import os
import os.path as osp
import sys
import time
from datetime import datetime

HERE = osp.dirname(osp.abspath(__file__))
GLASS_SRC = osp.join(HERE, "src")
OVERLAY = (os.environ.get("GLAD_ENVS", "envs") + "/ext-glass")
ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_glass")
NATIVE = osp.join(ROOT, "native")
NATIVE_DATA = osp.join(ROOT, "data_native")          # <DS>/raw copied from the UB-GOLD TU raw files
EMB_CACHE = osp.join(ROOT, "embedding_cache")        # content-hashed; shared with the port when texts are identical
TEXT_MODEL = osp.join(ROOT, "models", "Qwen3-Embedding-0.6B")


def setup():
    for k in ("GLASS_GPU_ACCEL", "GLASS_ABLATION", "GLASS_GRAPHDP_VERSION"):
        assert not os.environ.get(k), k
    if OVERLAY not in sys.path:
        sys.path.append(OVERLAY)
    if GLASS_SRC not in sys.path:
        sys.path.insert(0, GLASS_SRC)
    ec = importlib.import_module("src.embedding_cache")
    ec._global_cache = None
    ec.get_embedding_cache(cache_dir=EMB_CACHE)          # CHANGE (path): GLASS default is src/embedding_cache
    ut = importlib.import_module("src.utils")
    log_dir = osp.join(NATIVE, "training_logs")

    def save(self):                                        # copy of TrainingLogger.save; CHANGE (path) only
        os.makedirs(log_dir, exist_ok=True)
        tag = f"_{self.extra_tag}" if self.extra_tag else ""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{self.dataset}_{self.text_model_tag}{tag}_seed{self.seed}_{timestamp}.json"
        filepath = os.path.join(log_dir, fname)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  [LOG] Saved training log: {filepath}")
        return filepath
    ut.TrainingLogger.save = save
    return ec, ut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "train", "eval"])
    ap.add_argument("--dataset")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    setup()
    os.makedirs(NATIVE, exist_ok=True)
    t0 = time.time()
    if a.mode == "prep":
        import torch
        cfg = importlib.import_module("src.config")
        dsm = importlib.import_module("src.dataset")
        tem = importlib.import_module("src.text_encoder")
        T = importlib.import_module("glass_training_core")
        config = cfg.GLASSConfig(dataset_name=a.dataset, data_root=NATIVE_DATA, device="cuda:0", epochs=150,
                                 num_seeds=1, batch_size=64, lr=5e-5, alpha=0.3, patience=40, num_prototypes=8)
        config.gin_layers = 3
        config.mrl_mode = "native"
        config.text_model_path = TEXT_MODEL
        config.text_embed_dim = 1024
        dataset = dsm.GLASSDataset(config, precompute=True)        # GLASS's own loader (use_node_attr=True)
        te = tem.TextEncoder(config, text_device="cuda:0")
        te.to_text_device()
        raw = T.precompute_text_embeddings(dataset, te, config, "cuda:0", torch.device("cuda:0"))
        print("[native prep]", a.dataset, tuple(raw.shape), f"{time.time() - t0:.0f}s", flush=True)
    elif a.mode == "train":
        ts = importlib.import_module("train_single")
        sys.argv = ["train_single.py", "--dataset", a.dataset, "--device", "cuda:0", "--text_device", "cuda:0",
                    "--text_model", TEXT_MODEL, "--data_root", NATIVE_DATA,
                    "--num_seeds", "1", "--seed_start", str(a.seed),
                    "--epochs", "150", "--patience", "40", "--batch_size", "64",          # README single-domain line
                    "--ckpt_root", osp.join(NATIVE, "ckpt"),
                    "--result_root", osp.join(NATIVE, "results", f"seed{a.seed}")]
        ts.main()
        print("[native train]", a.dataset, a.seed, f"{time.time() - t0:.0f}s", flush=True)
    else:
        os.environ.setdefault("GLASS_EVAL_DEVICE", "cuda:0")
        ev = importlib.import_module("evaluate_reference")
        sys.argv = ["evaluate_reference.py", "--ckpt_root", osp.join(NATIVE, "ckpt"),
                    "--result_root", osp.join(NATIVE, "results", "reference"),
                    "--out_name", "reference_scorers_" + "_".join(a.datasets) + ".json",
                    "--text_model", TEXT_MODEL, "--data_root", NATIVE_DATA, "--num_seeds", "5",
                    "--datasets"] + list(a.datasets)
        ev.main()
        print("[native eval]", a.datasets, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
