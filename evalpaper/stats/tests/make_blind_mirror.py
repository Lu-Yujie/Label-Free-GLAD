"""Blind smoke-test mirror (development tool; never part of a result).

Copies every run JSON of harness/parts and ext/parts into <dst>/{harness_parts,ext_parts}/ with EVERY finite float
value replaced by an independent U(0,1) draw (trajectories, nested per-rule/per-cell metrics, meta['own'],
train_loss). Integers (epochs, counts), strings, NaN/inf positions and the whole file structure are kept, so the
adapters' row logic, the A15 audit (non-finite positions) and the statistics pipeline run end-to-end on the real
structure while no real performance number enters any computation. meta['scores_npz'] is kept, so the A21 distinct-
score counts use the real per-graph score files (no labels are read).

  python -B make_blind_mirror.py <dst>
Then build <dst>/res: eval_folds.csv from the scrambled harness parts via harness/collect.py (EP_PARTS/EP_RES), plus
links to the already-inspected inputs (kernels, references, M1, M2) in prelim/results.
"""
import glob
import json
import math
import os
import os.path as osp
import sys
import zlib

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(osp.dirname(HERE))
KEEP_FLOAT_KEYS = {"epoch"}


def scramble(x, rng):
    if isinstance(x, bool) or isinstance(x, int) or x is None or isinstance(x, str):
        return x
    if isinstance(x, float):
        return float(rng.uniform()) if math.isfinite(x) else x
    if isinstance(x, list):
        return [scramble(v, rng) for v in x]
    if isinstance(x, dict):
        return {k: (v if k in KEEP_FLOAT_KEYS else scramble(v, rng)) for k, v in x.items()}
    return x


def main(dst):
    for sub, src in (("harness_parts", osp.join(EVALPAPER, "harness", "parts")),
                     ("ext_parts", osp.join(EVALPAPER, "ext", "parts"))):
        out = osp.join(dst, sub)
        os.makedirs(out, exist_ok=True)
        n = 0
        for fn in sorted(glob.glob(osp.join(src, "*.json"))):
            J = json.load(open(fn))
            rng = np.random.default_rng(zlib.crc32(osp.basename(fn).encode()))
            J["traj"] = scramble(J["traj"], rng)
            if "train_loss" in J:
                J["train_loss"] = scramble(J["train_loss"], rng)
            if isinstance(J.get("meta", {}).get("own"), dict):
                J["meta"]["own"] = scramble(J["meta"]["own"], rng)
            with open(osp.join(out, osp.basename(fn)), "w") as fh:
                json.dump(J, fh)
            n += 1
        print(f"{sub}: {n} scrambled files")


if __name__ == "__main__":
    main(sys.argv[1])
