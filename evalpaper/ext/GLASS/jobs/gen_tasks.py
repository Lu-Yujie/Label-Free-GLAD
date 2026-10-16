"""Write jobs/tasks_grid.txt: unfinished GLASS grid runs (UB9 + MUTAG x 5 folds), longest first; all 'ubgold'
runs before all 'hpfree' runs (NOTES.md Part 1). All tasks run on CPU (deterministic kernels; the cost is
CPU-bound: GLASS recomputes LTD of perturbed negatives with networkx every batch).
  python3 -B jobs/gen_tasks.py [--out jobs/tasks_grid.txt]"""
import os, os.path as osp, sys
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
ORDER = ["REDDIT-BINARY", "NCI1", "AIDS", "PROTEINS_full", "DHFR", "ENZYMES", "IMDB-BINARY", "COX2", "BZR", "MUTAG"]
MEM = {"REDDIT-BINARY": 5, "NCI1": 5, "AIDS": 4}
THR = {}   # 1 thread per task (the cost is python-bound); fixed across resume segments
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else osp.join(HERE, "jobs", "tasks_grid.txt")
FAILED = osp.join(HERE, "logs", "failed_tasks.txt")      # crashed runs (rc not in {0,3}); investigate, never auto-retry
failed = set(open(FAILED).read().split()) if osp.exists(FAILED) else set()
lines, n_done = [], 0
for hp in ("ubgold", "hpfree"):
    for DS in ORDER:
        for k in range(5):
            rid = f"GLASS_{DS}_fold{k}" + ("_hpfree" if hp == "hpfree" else "")
            if osp.exists(osp.join(PARTS, rid + ".json")):
                n_done += 1
                continue
            if rid in failed:
                continue
            lines.append(f"GLASS {DS} {k} {hp} cpu mem={MEM.get(DS, 3)} threads={THR.get(DS, 1)}")
with open(out, "w") as f:
    f.write("# GLASS grid tasks (generated %s); %d done, %d to run\n" % (__import__("time").strftime("%F %T"), n_done, len(lines)))
    f.write("\n".join(lines) + "\n")
print(f"{out}: {len(lines)} tasks to run, {n_done} done, {len(failed)} failed (excluded)")
