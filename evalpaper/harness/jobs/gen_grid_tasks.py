"""Build the runtime/memory table (RUNTIME.md + runtime_table.csv) from the debug timing runs and write
the grid task lists (longest first):
  jobs/tasks_cpu.txt   regular `general` pools: CPU-placed tasks, then CPU-feasible GPU tasks (fallback)
  jobs/tasks_gpu.txt   regular `gpu` pools: big-memory CPU tasks ('cpu' flag), GPU-placed tasks, then all
                       CPU-placed tasks as fallback (tasks whose GPU footprint is > 12 GB keep the 'cpu' flag)
Timing source: $GLAD_DATA_ROOT/evalpaper/smoke/parts/*_t{g,c}pu.json (fold 0, 1-4 epochs;
CPU runs used 4 threads, GPU runs shared an A100 with up to 3 other runs).
Usage (login node): PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/ubgold-gpu/bin/python -B jobs/gen_grid_tasks.py
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import csv
import glob
import json
import os.path as osp
import sys
import time

J = osp.dirname(osp.abspath(__file__))
H = osp.dirname(J)
sys.path.insert(0, H)
import run_eval as R  # noqa: E402  (only resolve_hparams is used; no training)

SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper/smoke/parts")
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SEC = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP"]
NEW8 = list(R.NEW8)
OLD4 = list(R.OLD4)
CPU_2THREAD_PENALTY = 1.4    # grid CPU tasks mostly run with 2 threads; timings were measured with 4
MAX_CPU_HOURS = 24.0         # CPU placement only if the run takes <= 24 h on CPU (resumable anyway)
# known big-memory cases (host GB): measured peak RSS on CPU (GLADC REDDIT-B 229 ubgold / 216 hpfree,
# GLocalKD DD 187) + margin; GLADC DD (not measurable in the debug job) and GLocalKD REDDIT-B (deep_rerun ~140) estimated
BIG = {("GLADC", "REDDIT-BINARY"): 250, ("GLADC", "DD"): 440, ("GLocalKD", "DD"): 210, ("GLocalKD", "REDDIT-BINARY"): 150}
# no timing run possible in the debug job (140 GB host each): from earlier measurements
MANUAL = {("GLocalKD", "REDDIT-BINARY", "hpfree"): (791 * 400 / 150, 5 * 3600,
          "deep_rerun GPU 791 s per 150-epoch run, scaled to 400 epochs",
          "m1 CPU 26-31 s/epoch incl. eval at 32 threads; ~5 h assumed at 16 threads")}
SIZE = {"AIDS": (1280, 15.7), "IMDB-BINARY": (320, 19.8), "REDDIT-BINARY": (640, 429.6), "DD": (312, 284.3),
        "COLLAB": (1536, 74.5), "Tox21_p53": (6470, 17.4), "Tox21_MMP": (4936, 16.7)}
ANALOG = {"DD": "REDDIT-BINARY", "COLLAB": "IMDB-BINARY", "Tox21_p53": "AIDS", "Tox21_MMP": "AIDS"}


def timings():
    T = {}
    for fn in glob.glob(osp.join(SMOKE, "*_t[cg]pu.json")):
        d = json.load(open(fn))
        m = d["meta"]
        dev = "gpu" if m["run_id"].endswith("_tgpu") else "cpu"
        ep = max(1, m.get("epochs_this_segment") or 1)
        ne = max(1, m.get("n_eval") or 1)
        tr = m.get("seconds_train_this_segment", 0.0)
        ev = m.get("seconds_eval_this_segment", 0.0)
        T[(m["method"], m["dataset"], m.get("hp", "ubgold"), dev)] = dict(
            spe=tr / ep, spv=ev / ne, setup=max(0.0, m["seconds"] - tr - ev), mem=m["memory"].get("maxrss_gb", 0),
            gmem=m["memory"].get("cuda_max_reserved_gb", 0), threads=m.get("threads"))
    return T


def est(T, m, ds, hp, dev, ne, nv):
    """(seconds, timing record, source) for a full run on dev; hp fallback, then analog dataset."""
    for h2, src in ((hp, "measured"), ("ubgold" if hp == "hpfree" else "hpfree", "measured(other hp)")):
        k = (m, ds, h2, dev)
        if k in T:
            t = T[k]
            return t["setup"] + ne * t["spe"] + nv * t["spv"], t, src
    if ds in ANALOG:
        a = ANALOG[ds]
        s, t, _ = est(T, m, a, hp, dev, ne, nv)
        if s is not None:
            f = (SIZE[ds][0] * SIZE[ds][1]) / (SIZE[a][0] * SIZE[a][1])
            return s * f, t, f"extrapolated from {a} x{f:.1f}"
    return None, None, "unknown"


def old4_cpu_ratio():
    """CPU/GPU wall-time ratio per (old method, dataset) from m1_selection (CPU, 2 threads) vs
    deep_rerun (GPU) runs of the same UB-GOLD configuration (single-segment runs only)."""
    import collections
    PRE = osp.dirname(osp.dirname(H))
    cpu, gpu = collections.defaultdict(list), collections.defaultdict(list)
    for fn in glob.glob(osp.join(PRE, "m1_selection", "parts", "*.json")):
        m = json.load(open(fn))["meta"]
        if m.get("device_all") == "cpu" and len(m.get("segments", [])) == 1:
            cpu[(m["method"], m["dataset"])].append(m["seconds"])
    for fn in glob.glob(osp.join(PRE, "results", "deep_rerun_parts", "*.json")):
        m = json.load(open(fn))["meta"]
        if m["run_id"] == f"{m['method']}_{m['dataset']}_fold{m['fold']}":
            gpu[(m["method"], m["dataset"])].append(m["seconds"])
    r = {k: (sum(cpu[k]) / len(cpu[k])) / (sum(gpu[k]) / len(gpu[k])) for k in cpu if k in gpu}
    per_m = {}
    for (mm, _), v in r.items():
        per_m.setdefault(mm, []).append(v)
    return r, {mm: sorted(v)[len(v) // 2] for mm, v in per_m.items()}


def main():
    T = timings()
    RAT, RAT_M = old4_cpu_ratio()
    rows = []
    for grid, hp, dsets, meths in (("P1", "ubgold", UB9, NEW8), ("P2", "hpfree", UB9, NEW8 + OLD4),
                                    ("P3", "ubgold", SEC, NEW8 + OLD4), ("P3h", "hpfree", SEC, NEW8 + OLD4)):
        for m in meths:
            for ds in dsets:
                a, _, _ = R.resolve_hparams(m, ds, hp)
                ne, ef = int(a.num_epoch), int(a.eval_freq)
                nv = ne // ef + (1 if ne % ef else 0)
                g, tg, sg = est(T, m, ds, hp, "gpu", ne, nv)
                c, tc, sc = est(T, m, ds, hp, "cpu", ne, nv)
                if c is not None and sc == "measured(other hp)" and (m, ds, hp, "gpu") in T and \
                        (m, ds, "ubgold" if hp == "hpfree" else "hpfree", "gpu") in T:
                    go, _, _ = est(T, m, ds, "ubgold" if hp == "hpfree" else "hpfree", "gpu", ne, nv)
                    c = c * g / max(go, 1e-9)
                    sc += " scaled by GPU hp ratio"
                if c is None and g is not None and m in OLD4:   # old 4: m1 CPU vs deep_rerun GPU ratio
                    rat = RAT.get((m, ds), RAT_M.get(m))
                    if rat is not None:
                        c = g * rat / CPU_2THREAD_PENALTY            # m1 ran 2 threads; penalty re-applied below
                        sc = f"GPU est x m1/deep_rerun CPU:GPU ratio {rat:.1f}" + ("" if (m, ds) in RAT else " (method median)")
                if (m, ds, hp) in MANUAL:
                    g, c, sg, sc = MANUAL[(m, ds, hp)]
                hostmem = max([t["mem"] for t in (tg, tc) if t] or [0])
                mem = BIG.get((m, ds), max(3, int(round(1.3 * hostmem + 1))))
                if sc.startswith("extrapolated") or sg.startswith("extrapolated"):
                    mem = BIG.get((m, ds), max(mem, 16 if ds in ("DD", "COLLAB") else 6))
                rows.append(dict(grid=grid, method=m, dataset=ds, hp=hp, num_epoch=ne, eval_freq=ef, n_eval=nv,
                                 gpu_s_per_epoch=(tg or {}).get("spe"), gpu_s_per_eval=(tg or {}).get("spv"),
                                 cpu_s_per_epoch=(tc or {}).get("spe"), cpu_s_per_eval=(tc or {}).get("spv"),
                                 gpu_total_s=g, cpu_total_s=c, gpu_src=sg, cpu_src=sc,
                                 gpu_mem_gb=(tg or {}).get("gmem"), host_mem_gb=hostmem, mem=mem))
    for r in rows:
        g, c = r["gpu_total_s"], r["cpu_total_s"]
        cc = None if c is None else c * CPU_2THREAD_PENALTY
        key = (r["method"], r["dataset"])
        if r["grid"] == "P3h" and key in (("GLADC", "DD"), ("GLocalKD", "DD")):
            r["place"] = "skip"            # optional grid: no dense 5748-node runs
        elif key in (("GLADC", "REDDIT-BINARY"), ("GLADC", "DD"), ("GLocalKD", "DD")):
            r["place"] = "bigcpu"          # host RAM > general-node budget (GLocalKD DD 187 GB RSS measured); GPU impossible
        elif key == ("GLocalKD", "REDDIT-BINARY"):
            r["place"] = "cpu"             # 150 GB host; ~30-40 s/epoch on CPU; GPU pools take it if still free
        elif cc is None:
            r["place"] = "gpu"
        elif cc <= MAX_CPU_HOURS * 3600:
            r["place"] = "cpu"
        else:
            r["place"] = "gpu"
        r["threads"] = 24 if r["place"] == "bigcpu" else (16 if key in (("GLocalKD", "REDDIT-BINARY"), ("GLocalKD", "DD")) else
                                                         (4 if (cc or 0) > 3 * 3600 else 2))
        r["est_h"] = ((c if r["threads"] >= 4 else cc) if r["place"] in ("cpu", "bigcpu") else g or 0) / 3600 \
            if (c is not None or g is not None) else None
    with open(osp.join(H, "runtime_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def line(r, fo, dev):
        flag = " cpu" if r["place"] == "bigcpu" or (dev == "gpu" and r["place"] == "cpu" and (r["gpu_mem_gb"] or 0) > 12) else ""
        return f"{r['method']} {r['dataset']} {fo} {r['hp']}{flag} mem={r['mem']} threads={r['threads']}"

    order = lambda r: -(r["est_h"] if r["est_h"] is not None else 99)
    opt = [r for r in rows if r["grid"] == "P3h"]
    rows_main = [r for r in rows if r["grid"] != "P3h"]
    opt_cpu = sorted([r for r in opt if r["place"] == "cpu"], key=order)            # optional, last
    opt_gpu = sorted([r for r in opt if r["place"] == "gpu"], key=order)
    cpu = sorted([r for r in rows_main if r["place"] == "cpu"], key=order)
    gpu = sorted([r for r in rows_main if r["place"] == "gpu"], key=order)
    big = sorted([r for r in rows_main if r["place"] == "bigcpu"], key=lambda r: (r["dataset"] == "DD", order(r)))
    with open(osp.join(J, "tasks_cpu.txt"), "w") as f:
        f.write(f"# general pools ({time.strftime('%F %T')}, gen_grid_tasks.py): CPU-placed tasks longest first, "
                "then GPU-placed tasks (fallback)\n")
        for r in cpu:
            for fo in range(5):
                f.write(line(r, fo, "cpu") + "\n")
        f.write("# --- fallback: GPU-placed tasks\n")
        for r in gpu:
            for fo in range(5):
                f.write(line(r, fo, "cpu") + "\n")
        f.write("# --- optional P3 HP-FREE (PREREG_AMENDMENTS A8; dense GLADC/GLocalKD DD excluded)\n")
        for r in opt_cpu + opt_gpu:
            for fo in range(5):
                f.write(line(r, fo, "cpu") + "\n")
    with open(osp.join(J, "tasks_gpu.txt"), "w") as f:
        f.write(f"# gpu pools ({time.strftime('%F %T')}, gen_grid_tasks.py): big-memory CPU tasks, GPU-placed tasks, "
                "then CPU-placed tasks (fallback)\n")
        for r in big + gpu + cpu:
            for fo in range(5):
                f.write(line(r, fo, "gpu") + "\n")
        f.write("# --- optional P3 HP-FREE (PREREG_AMENDMENTS A8; dense GLADC/GLocalKD DD excluded)\n")
        for r in opt_gpu + opt_cpu:
            for fo in range(5):
                f.write(line(r, fo, "gpu") + "\n")
    # RUNTIME.md
    fmt = lambda x, d=1: "" if x is None else (f"{x:.{d}f}")
    L = ["# Runtime and memory table (evalpaper week-1 grids)", "",
         f"Generated {time.strftime('%F %T')} by `jobs/gen_grid_tasks.py` from the debug timing runs. Machine-readable "
         "version: `runtime_table.csv`.", "",
         "How the numbers were measured:",
         "- Source: fold 0, 1–4 epochs, tags `tgpu`/`tcpu`, in `$GLAD_DATA_ROOT/evalpaper/smoke/parts`.",
         "- GPU runs used an A100-40GB that up to 3 other timing runs were sharing.",
         "- CPU runs used 4 threads on the same node (AMD EPYC, gpu-debug), with other jobs running.",
         "- s/epoch counts training only; s/eval is one evaluation point: test plus held-out scoring (GOOD-D also "
         "runs its k-means).",
         "", "How the estimates and placement are built:",
         "- Estimate = setup + num_epoch × s/epoch + n_eval × s/eval.",
         f"- CPU tasks in the grid mostly use 2 threads, so the CPU estimate is multiplied by {CPU_2THREAD_PENALTY} "
         "unless the task gets ≥ 4 threads.",
         "- `measured(other hp)`: the timing of the other hyperparameter set, scaled by the GPU ratio where both "
         "exist.",
         "- `extrapolated`: secondary datasets without a timing run, scaled by #graphs × mean nodes from an analog "
         "dataset. Treat these as rough.",
         f"- Placement: `cpu` if the CPU estimate is ≤ {MAX_CPU_HOURS:.0f} h. The general queue (~6–10 h) is much "
         "shorter than the gpu queue (~1.5–3 days), and every run is resumable.",
         "- `bigcpu`: the host-RAM need exceeds a general node, so the task runs on a 512 GB gpu-partition node, "
         "on CPU.", "- `gpu`: everything else.", "",
         "| grid | method | dataset | hp | epochs (eval_freq) | GPU s/epoch | GPU s/eval | CPU s/epoch | CPU s/eval | "
         "GPU est h | CPU est h | GPU GB | host GB | place | threads | est h (placed) | source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['grid']} | {r['method']} | {r['dataset']} | {r['hp']} | {r['num_epoch']} ({r['eval_freq']}) | "
                 f"{fmt(r['gpu_s_per_epoch'], 2)} | {fmt(r['gpu_s_per_eval'], 2)} | {fmt(r['cpu_s_per_epoch'], 2)} | "
                 f"{fmt(r['cpu_s_per_eval'], 2)} | {fmt(None if r['gpu_total_s'] is None else r['gpu_total_s'] / 3600, 2)} | "
                 f"{fmt(None if r['cpu_total_s'] is None else r['cpu_total_s'] / 3600, 2)} | {fmt(r['gpu_mem_gb'])} | "
                 f"{fmt(r['host_mem_gb'])} | {r['place']} | {r['threads']} | {fmt(r['est_h'], 2)} | "
                 f"gpu: {r['gpu_src']}; cpu: {r['cpu_src']} |")
    tot = {}
    for r in rows:
        k = r["place"] + ("(optional P3h)" if r["grid"] == "P3h" else "")
        tot[k] = tot.get(k, 0.0) + 5 * (r["est_h"] or 0) * (r["threads"] if k != "gpu" else 1)
    L += ["", "## Totals (5 folds each)", "",
          f"- CPU-placed: {tot.get('cpu', 0):.0f} core-hours (threads × hours). Two general nodes provide 256 cores.",
          f"- bigcpu (512 GB gpu-partition nodes, CPU compute): {tot.get('bigcpu', 0):.0f} core-hours.",
          f"- GPU-placed: {tot.get('gpu', 0):.1f} GPU-run-hours, with up to 4 runs sharing one A100.",
          f"- Optional P3 HP-FREE (appended at the end of both lists): CPU {tot.get('cpu(optional P3h)', 0):.0f} "
          f"core-hours, GPU {tot.get('gpu(optional P3h)', 0):.1f} run-hours. The dense DD runs of GLADC and GLocalKD "
          "are excluded.",
          "", "Unknown timings (no timing run and no analog): " +
          ", ".join(f"{r['method']}/{r['dataset']}/{r['hp']}" for r in rows if r["est_h"] is None)]
    open(osp.join(H, "RUNTIME.md"), "w").write("\n".join(L) + "\n")
    print({k: round(v) for k, v in tot.items()}, "tasks:", {p: 5 * sum(1 for r in rows if r["place"] == p) for p in ("cpu", "gpu", "bigcpu")})


if __name__ == "__main__":
    main()
