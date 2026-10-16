"""Refresh the generated tables in NOTES.md (blocks between <!-- BEGIN:KEY --> and <!-- END:KEY -->)
from make_tables.py and compare.py. Run after aggregate.py."""
import os.path as osp
import re
import subprocess
import sys

HERE = osp.dirname(osp.abspath(__file__))
run = lambda s: subprocess.run([sys.executable, "-B", osp.join(HERE, s)], capture_output=True, text=True,
                               cwd=HERE, check=True).stdout
t1, t2 = run("make_tables.py"), run("compare.py")
sec = {}
for block in t1.split("\n#### ")[1:]:
    title, body = block.split("\n", 1)
    sec[title.strip()] = body.strip()
blocks = {
    "SIGNET": sec["SIGNET"], "OCGTL": sec["OCGTL"], "OCGIN": sec["OCGIN"], "GLocalKD": sec["GLocalKD"],
    "AVERAGES": [v for k, v in sec.items() if k.startswith("Averages")][0],
    "FOLDLEVEL": [v for k, v in sec.items() if k.startswith("Fold-level")][0],
    "FULLDATA": [v for k, v in sec.items() if k.startswith("SIGNET on UB-GOLD")][0],
}
paired, ranking = t2.split("\nRanking on")
keep = [l for l in paired.strip().splitlines()
        if l.startswith("| deep method") or l.startswith("|---") or "| GRAIL-0 |" in l or "| size-only |" in l]
blocks["PAIRED"] = "\n".join(keep)
blocks["RANKING"] = "Ranking on" + ranking.rstrip()
p = osp.join(HERE, "NOTES.md")
s = open(p).read()
for k, v in blocks.items():
    pat = re.compile(rf"(<!-- BEGIN:{k} -->\n).*?(<!-- END:{k} -->)", re.S)
    assert pat.search(s), k
    s = pat.sub(lambda m: m.group(1) + v + "\n" + m.group(2), s)
open(p, "w").write(s)
print("filled:", ", ".join(blocks))
