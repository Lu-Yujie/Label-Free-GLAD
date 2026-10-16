"""UB-GOLD's published KernelGLAD AUROCs vs chance and vs our runs.

  python -I published.py <path to downloaded arxiv.org/html/2406.15523v2>   (run from this directory
  with PYTHONPATH unset; the HTML is untrusted input and is only parsed as text)

Reads UB-GOLD Table 3 ("Comparison in terms of AUROC") from the arXiv HTML v2, keeps the four
graph-kernel columns (PK-SVM, PK-iF, WL-SVM, WL-iF = our PK-OCSVM, PK-IF, WL-OCSVM, WL-IF) for the 13
datasets we run, and writes results/eval_kernels_published.csv with
  * the published 5-trial mean AUROC,
  * the chance band of a 5-fold mean when scores and labels are paired at random (as in UB-GOLD's
    released KernelGLAD): fold AUROC under random pairing has mean 50 and sd
    100*sqrt((n0+n1+1)/(12*n0*n1)) (Mann-Whitney null), using each fold's test counts (n0 normals,
    n1 anomalies) from our fold exports; the 5-fold mean has sd sqrt(sum sd_i^2)/5;
  * z = (published - 50) / sd_mean,
  * our UB-GOLD-code-path numbers (fit on the test set, default config, correct labels) and our
    train-fit default numbers, from eval_kernels_summary.csv when present.
"""
import csv
import hashlib
import json
import os.path as osp
import sys
import glob
from html.parser import HTMLParser

HERE = osp.dirname(osp.abspath(__file__))
RES = osp.join(osp.dirname(osp.dirname(HERE)), "results")
URL = "https://arxiv.org/html/2406.15523v2"
NAMES = {"p53": "Tox21_p53", "MMP": "Tox21_MMP", "COLLAB": "COLLAB", "IMDB-B": "IMDB-BINARY",
         "REDDIT-B": "REDDIT-BINARY", "ENZYMES": "ENZYMES", "PROTEINS": "PROTEINS_full", "DD": "DD",
         "BZR": "BZR", "AIDS": "AIDS", "COX2": "COX2", "NCI1": "NCI1", "DHFR": "DHFR"}
COLS = {"PK-SVM": "PK-OCSVM", "PK-iF": "PK-IF", "WL-SVM": "WL-OCSVM", "WL-iF": "WL-IF"}


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.cur, self.row, self.cell, self.cap, self.incap = [], None, None, None, "", False

    def handle_starttag(self, t, a):
        if t == "table":
            self.cur = {"rows": []}
        elif t == "tr" and self.cur is not None:
            self.row = []
        elif t in ("td", "th") and self.row is not None:
            self.cell = ""
        elif t == "figcaption":
            self.incap, self.cap = True, ""

    def handle_endtag(self, t):
        if t == "table" and self.cur is not None:
            self.cur["caption"] = self.cap
            self.tables.append(self.cur)
            self.cur = None
        elif t == "tr" and self.row is not None and self.cur is not None:
            self.cur["rows"].append(self.row)
            self.row = None
        elif t in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join(self.cell.split()))
            self.cell = None
        elif t == "figcaption":
            self.incap = False

    def handle_data(self, d):
        if self.cell is not None:
            self.cell += d
        if self.incap:
            self.cap += d


def fold_counts():
    """(n0 normals, n1 anomalies) per dataset and fold from the unit JSONs (test-mode WL h5 units)."""
    out = {}
    for p in glob.glob(osp.join(HERE, "parts", "*__WL__test__h5.json")):
        d = json.load(open(p))
        y = d["y_test"]
        out.setdefault(d["dataset"], {})[d["fold"]] = (len(y) - sum(y), sum(y))
    return out


def main():
    raw = open(sys.argv[1], "rb").read()
    p = P()
    p.feed(raw.decode("utf-8", errors="replace"))
    tab = [t for t in p.tables if t["caption"].strip().startswith("Table 3") and
           any("WL-iF" in c for r in t["rows"] for c in r)]
    assert len(tab) == 1, len(tab)
    rows = tab[0]["rows"]
    head = [r for r in rows if "PK-SVM" in r][0]
    idx = {c: head.index(c) for c in COLS}
    off = len(rows[[i for i, r in enumerate(rows) if r is head][0] + 1]) - len(head)  # data rows carry the dataset name
    counts = fold_counts()
    ours = {}
    sp = osp.join(RES, "eval_kernels_summary.csv")
    if osp.exists(sp):
        for r in csv.DictReader(open(sp)):
            if r["protocol"] == "default":
                ours[(r["method"], r["dataset"])] = float(r["auroc_mean"])
    out = []
    for r in rows:
        if not r or r[0] not in NAMES:
            continue
        ds = NAMES[r[0]]
        for c, m in COLS.items():
            v = float(r[idx[c] + off])
            fc = counts.get(ds, {})
            sd = None
            if len(fc) == 5:
                var = sum(((n0 + n1 + 1) / (12.0 * n0 * n1)) * 1e4 for n0, n1 in fc.values())
                sd = var ** 0.5 / 5
            out.append({"dataset": ds, "method": m, "published_auroc": v,
                        "chance_sd_of_5fold_mean": round(sd, 3) if sd else "",
                        "z_vs_chance": round((v - 50) / sd, 2) if sd else "",
                        "ours_testfit_default": round(ours.get((m + "@testfit", ds), float("nan")), 2),
                        "ours_trainfit_default": round(ours.get((m, ds), float("nan")), 2),
                        "source": f"{URL} Table 3 (AUROC), sha256 {hashlib.sha256(raw).hexdigest()[:16]}"})
    with open(osp.join(RES, "eval_kernels_published.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows ->", osp.join(RES, "eval_kernels_published.csv"))
    for o in out:
        print(o["dataset"], o["method"], o["published_auroc"], o["z_vs_chance"], o["ours_testfit_default"], o["ours_trainfit_default"])


if __name__ == "__main__":
    main()
