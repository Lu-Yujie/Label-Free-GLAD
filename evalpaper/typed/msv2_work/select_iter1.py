"""Apply the pre-declared iteration-1 selection rule (NOTES_MSv2.md §3.5) to the grid outputs. Prints the full table."""
import sys

sys.dont_write_bytecode = True
import glob
import json
import os.path as osp

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
rows = []
for p in sorted(glob.glob(osp.join(HERE, "grid_iter1_P*.json"))):
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
sel = {}
for pool in ("P0", "P56"):
    fams = {}
    for fam in "ABCD":
        x = T[(T.pool == pool) & (T.family == fam) & T.PASS]
        if len(x):
            fams[fam] = x.sort_values("kord").k.iloc[0]
    sel[pool] = fams
print("passing families (smallest passing k):", sel)
npass = {p: len(v) for p, v in sel.items()}
pool = "P0" if npass["P0"] >= npass["P56"] else "P56"
final = [(f"MS{i + 1}", f, sel[pool][f]) for i, f in enumerate([f for f in "ABCD" if f in sel[pool]][:3])]
print("selected pool:", pool, "final types:", final)
json.dump(dict(table=T.drop(columns="kord").to_dict(orient="records"), passing=sel, pool=pool, final=final),
          open(osp.join(HERE, "select_iter1.json"), "w"), indent=1)
