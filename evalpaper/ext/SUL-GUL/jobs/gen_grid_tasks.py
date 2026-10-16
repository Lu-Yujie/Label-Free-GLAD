"""Build the SUL/GUL task list (longest first): grid = UB9 + MUTAG x 5 folds x hp {ubgold, hpfree} = 100 runs, plus
25 NATIVE port-fidelity tasks (GUL's own main.py code on its own splits; NOTES.md "Reproduction targets").

  python3 -B jobs/gen_grid_tasks.py        (login node) -> jobs/tasks_grid.txt

Estimates: login-node timing (fold 0, 3 epochs, 4 threads, 2026-10-13 ~13:10 EDT; smoke parts *_tlogin):
s/epoch incl. evaluation = REDDIT-B 1.45, NCI1 0.61, AIDS 0.21, PROTEINS_full 0.15; the others are smaller (MUTAG
0.3 incl. ~7 s setup over 5 epochs). x 500 epochs x 1.5 (node contention). Both hp of one (dataset, fold) use the same
thread count (bit-identity expected, NOTES.md).
"""
import os.path as osp
import time

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
S_EPOCH = {"REDDIT-BINARY": 1.45, "NCI1": 0.61, "AIDS": 0.21, "PROTEINS_full": 0.15, "DHFR": 0.15, "IMDB-BINARY": 0.15,
           "ENZYMES": 0.12, "COX2": 0.05, "BZR": 0.05, "MUTAG": 0.05}
THREADS = {"REDDIT-BINARY": 8}
items = []
for DS, s in S_EPOCH.items():
    est = (20 + 500 * s * 1.5) / 60.0
    for k in range(5):
        for hp in ("ubgold", "hpfree"):
            items.append((est, f"SUL-GUL {DS} {k} {hp} threads={THREADS.get(DS, 4)} mem=4"))
# native diagnostic: GUL's own loader/splits; train sizes 100 (MUTAG poll 0) .. 1280 (AIDS poll 0)
for DS, polls, s in (("AIDS", ("0",), 0.25), ("IMDB-BINARY", ("0", "1"), 0.2), ("MUTAG", ("0", "1"), 0.06)):
    for p in polls:
        for k in range(5):
            items.append(((20 + 500 * s * 1.5) / 60.0, f"NATIVE {DS} {k} poll={p} threads=4 mem=4"))
items.sort(key=lambda x: -x[0])
with open(osp.join(HERE, "jobs", "tasks_grid.txt"), "w") as f:
    f.write(f"# SUL/GUL grid + NATIVE diagnostic, longest first (generated {time.strftime('%F %T')} by "
            f"jobs/gen_grid_tasks.py); est. minutes per task in the trailing comment\n")
    for est, line in items:
        f.write(f"{line}  # {est:.1f} min\n")
print(len(items), "tasks; est. task-minutes", round(sum(e for e, _ in items)),
      "core-hours", round(sum(e * (8 if 'REDDIT' in l and 'NATIVE' not in l else 4) for e, l in items) / 60, 1))
