"""D6: iteration-1 family A (k=all) and B (k=all), seeds 0-1: graphlet-count+LW AUROC by parent ring-size class
(global model; anomalies grouped by their parent's class vs all test normals) and by number of edits. No detector."""
import sys
sys.dont_write_bytecode = True
import os.path as osp, json
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ctrllib as CL
import molswap_v2 as M
import numpy as np
from collections import defaultdict
out = []
for pool_name in ("P0", "P56"):
    pool = M.load_pool(pool_name)
    for s in (0, 1):
        tr, tn, pa = M.split_pool(pool, s)
        for fam in ("A", "B"):
            an, cls, ned = [], [], []
            for P in pa:
                seq = M.edit_sequence(P, fam, s, M.KCAP)
                r, w = M.finalize(P, seq, "all")
                if r is None:
                    continue
                an.append({"n": P["n"], "edges": M.edges_array(r["E"]), "x": None, "node_label": P["node_label"], "gid": -1})
                h = set(M.ring_hist(P["n"], M.eset_of(P["edges"])))
                cls.append("6only" if h == {6} else ("56only" if h == {5, 6} else ("has5+6+other" if {5, 6} <= h else "other")))
                ned.append(min(r["n_steps"], 4))
            S = CL.scores(tr, list(tn) + an, 37, ("gc_lw",))["gc_lw"]
            sn, sa = S[:len(tn)], S[len(tn):]
            cls, ned = np.array(cls), np.array(ned)
            rec = dict(pool=pool_name, seed=s, fam=fam, all=CL.auroc(sn, sa), n=len(an))
            for c in sorted(set(cls)):
                m = cls == c
                if m.sum() >= 10:
                    rec[f"cls_{c}"] = (round(CL.auroc(sn, sa[m]), 1), int(m.sum()))
            for k in sorted(set(ned)):
                m = ned == k
                if m.sum() >= 10:
                    rec[f"edits_{k}"] = (round(CL.auroc(sn, sa[m]), 1), int(m.sum()))
            out.append(rec)
            print(rec, flush=True)
json.dump(out, open(osp.join(osp.dirname(osp.abspath(__file__)), "diag_iter1_A.json"), "w"), indent=1)
