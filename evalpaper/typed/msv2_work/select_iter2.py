"""Apply the pre-declared iteration-2 selection rule (NOTES_MSv2.md §5) to the grid outputs. Prints the full table."""
import sys

sys.dont_write_bytecode = True
import glob
import json
import os.path as osp

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
rows = []
for p in sorted(glob.glob(osp.join(HERE, "grid_iter2_P*.json"))):
    if "_test" in p:
        continue
    rows += json.load(open(p))["rows"]
d = pd.DataFrame(rows)
assert not d.duplicated(["pool", "family", "k", "seed"]).any()
g = d.groupby(["pool", "family", "k"])
T = g.agg(seeds=("seed", "nunique"), n_pool=("n_pool", "first"), parents=("n_parents", "min"),
          min_matched=("n_matched", "min"), min_sizechg=("n_sizechg", "min"), mean_edits=("mean_edits", "mean"),
          gc=("matched_gc_lw", "mean"), gc_min=("matched_gc_lw", "min"), gc_max=("matched_gc_lw", "max"),
          size=("matched_size_lw", "mean"), elp=("matched_elp_lw", "mean"), null_gc=("null_gc_lw", "mean"),
          null_size=("null_size_lw", "mean"), null_elp=("null_elp_lw", "mean"),
          sc_gc=("sizechg_gc_lw", "mean"), sc_size=("sizechg_size_lw", "mean"),
          sc_elp=("sizechg_elp_lw", "mean")).reset_index()
T["feasible"] = (T.min_matched >= 100) & (T.min_sizechg >= 100) & (T.seeds == 5)
T["PASS"] = (T.gc >= 85) & (T["size"] <= 60) & (T.elp < 85) & (T.null_gc <= 60) & T.feasible
korder = {"1": 0, "2": 1, "all": 2}
T["kord"] = T.k.map(korder)
T = T.sort_values(["pool", "family", "kord"])
pd.set_option("display.width", 250)
print(T.drop(columns="kord").round(1).to_string(index=False))
# iteration-2 rule (NOTES_MSv2.md §5): pool fixed P0; MS1 = C k=1, MS2 = D k=2 (iteration 1); MS3 = first passing
# family in priority E, F at its smallest passing k.
x = T[(T.pool == "P0") & T.PASS]
ms3 = None
for fam in "EF":
    y = x[x.family == fam].sort_values("kord")
    if len(y):
        ms3 = (fam, y.k.iloc[0])
        break
final = [("MS1", "C", "1"), ("MS2", "D", "2")] + ([("MS3",) + ms3] if ms3 else [])
print("iteration-2 MS3:", ms3, "final design (pool P0):", final)
json.dump(dict(table=T.drop(columns="kord").to_dict(orient="records"), ms3=ms3, pool="P0", final=final),
          open(osp.join(HERE, "select_iter2.json"), "w"), indent=1)
