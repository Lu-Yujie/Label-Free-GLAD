"""Runtime table from finished HC-GLAD parts (real grid runs; single-segment runs give clean timings, resumed runs
are summed over segments only for the wall total). Prints a markdown table (mean over folds per dataset/hp/device).
  python -B tests/timing_table.py [PARTS_DIR]
"""
import glob, json, os.path as osp, sys
from collections import defaultdict
import numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "..", "parts")
G = defaultdict(list)
for fn in sorted(glob.glob(osp.join(P, "HC-GLAD_*.json"))):
    m = json.load(open(fn))["meta"]
    if m.get("max_epochs_override") is not None:
        continue
    dev = m.get("device_all", m["device"])
    G[(m["dataset"], m["hp"], dev)].append(m)
print("| dataset | hp | device | runs (single-segment) | threads | s/epoch train | s/eval | startup s | epochs | evals | wall min (single-segment runs; else estimate) | peak RSS GB | peak GPU GB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (ds, hp, dev), ms in sorted(G.items()):
    one = [m for m in ms if len(m["segments"]) == 1]
    h = ms[0]["hparams"]
    # rates from the last segment of every run (resumed runs: the last segment only)
    spe = np.mean([m["sec_per_epoch_train"] for m in ms])
    spv = np.mean([m["seconds_eval_this_segment"] / max(1, sum(1 for e in range(m["segments"][-1]["start_epoch"], h["num_epoch"] + 1) if e % h["eval_freq"] == 0)) for m in ms])
    st = np.mean([m["seconds"] - m["seconds_train_this_segment"] - m["seconds_eval_this_segment"] for m in one]) if one else np.mean([m.get("seconds_data_load", 0) for m in ms])
    wall = np.mean([m["seconds"] for m in one]) / 60 if one else (st + spe * h["num_epoch"] + spv * ms[0]["n_eval"]) / 60
    rss = max(m["memory"]["maxrss_gb"] for m in ms)
    gpu = max(m["memory"].get("cuda_max_alloc_gb", float("nan")) for m in ms)
    print(f"| {ds} | {hp} | {dev} | {len(ms)} ({len(one)}) | {ms[0]['threads']} | {spe:.3f} | {spv:.2f} | {st:.0f} | {h['num_epoch']} | {ms[0]['n_eval']} | {wall:.1f} | {rss:.1f} | {gpu:.1f} |")
