"""Runtime / memory table for GLASS (RUNTIME.md source): grid parts (CPU), tagged GPU timing runs (smoke dir), the
prep logs (features + Qwen3 embeddings) and the native-diagnostic logs.
  $GLAD_ENVS/ubgold-gpu/bin/python -B evalpaper/ext/GLASS/runtime.py   -> runtime_table.csv (+ print)"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import glob
import json
import os.path as osp
import re

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = osp.join(osp.dirname(HERE), "parts")
SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_glass/smoke/parts")


def part_rows():
    rows = []
    for f in sorted(glob.glob(osp.join(PARTS, "GLASS_*.json")) + glob.glob(osp.join(SMOKE, "GLASS_*_tgpu.json"))):
        m = json.load(open(f))["meta"]
        mem = m.get("memory", {})
        rows.append(dict(kind="grid" if m.get("max_epochs_override") is None else "timing", run_id=m["run_id"],
                         dataset=m["dataset"], hp=m["hp"], device=m["device_all"], threads=m["threads"],
                         segments=len(m["segments"]), epochs=m["hparams"]["num_epoch"],
                         n_fit=m["n_fit"], n_test=m["n_test"],
                         sec_total_last_segment=round(m["seconds"], 1),
                         sec_per_epoch_train=round(m["sec_per_epoch_train"], 2),
                         sec_eval_per_point=round(m["seconds_eval_this_segment"] / max(1, m["n_eval"]), 2),
                         maxrss_gb=round(mem.get("maxrss_gb", float("nan")), 2),
                         cuda_max_alloc_gb=round(mem.get("cuda_max_alloc_gb", float("nan")), 2)))
    return rows


def prep_rows():
    rows = []
    for f in sorted(glob.glob(osp.join(HERE, "logs", "prep", "prep_*.log"))):
        txt = open(f).read()
        m = re.search(r"^\[prep\] (\{.*\})$", txt, re.M)
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        if m:
            d = json.loads(m.group(1))
            d.update(kind="prep", log=osp.basename(f), maxrss_gb=round(int(rss.group(1)) / 1024 ** 2, 2) if rss else None)
            rows.append(d)
    return rows


def main():
    R = pd.DataFrame(part_rows())
    P = pd.DataFrame(prep_rows())
    R.to_csv(osp.join(HERE, "runtime_table.csv"), index=False)
    if len(P):
        P.to_csv(osp.join(HERE, "runtime_prep.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        if len(R):
            g = R[R.kind == "grid"].groupby(["dataset", "device"]).agg(
                runs=("run_id", "count"), sec_per_epoch=("sec_per_epoch_train", "median"),
                sec_eval=("sec_eval_per_point", "median"), sec_total=("sec_total_last_segment", "median"),
                rss_gb=("maxrss_gb", "max"), segments=("segments", "max"))
            print(g.to_string())
            print(R[R.kind == "timing"].to_string(index=False))
        if len(P):
            print(P.to_string(index=False))


if __name__ == "__main__":
    main()
