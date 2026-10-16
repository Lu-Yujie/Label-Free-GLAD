"""A17 task list and device plan (NOTES.md §3.4). Standard library only; reads META blocks only (no metric).

  python3 -B jobs/gen_tasks.py [--runtime tests/runtime_smoke.json]

Device plan: for every (method, UB9 dataset, fold) the device TYPE of the c940457 run that stats/glad_stats.py pairs
with, found exactly as harness/collect.py finds it: the untagged harness part <M>_<DS>_fold<k>.json if it exists (and
has no max_epochs_override), else results/deep_rerun_parts/<M>_<DS>_fold<k>.json. CPU threads = that run's
threads (deep_rerun / GPU runs: 2).
Order: longest first, from the smoke timing table (est_full_h per method x dataset x device) when given, else from the
c940457 run's seconds. mem / gmem (GB) from the smoke peaks with a margin when available, else conservative defaults.
Writes jobs/tasks_all.txt (90 tasks) and jobs/device_plan.json.
"""
import argparse
import json
import os.path as osp
import time

HERE = osp.dirname(osp.abspath(__file__))
A17 = osp.dirname(HERE)
EVALPAPER = osp.dirname(A17)
PRELIM = osp.dirname(EVALPAPER)
HPARTS = osp.join(EVALPAPER, "harness", "parts")
DRPARTS = osp.join(PRELIM, "results", "deep_rerun_parts")
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
METHODS = ["GLocalKD", "GLADC"]
# conservative defaults (GB) before smoke measurements; REDDIT-B: one dense 300-graph batch is 17 GB adj (float32)
# + 34 GB adj_label (float64) on the host, rebuilt every epoch at a96137b
DEF_MEM = {"REDDIT-BINARY": {"GLADC": 200.0, "GLocalKD": 130.0}}
DEF_GMEM = {"REDDIT-BINARY": {"GLADC": 38.0, "GLocalKD": 30.0}}


def c940457_part(m, ds, k):
    hp = osp.join(HPARTS, f"{m}_{ds}_fold{k}.json")
    if osp.exists(hp):
        J = json.load(open(hp))
        if J["meta"]["run_id"] == f"{m}_{ds}_fold{k}" and J["meta"].get("max_epochs_override") is None:
            return hp, J["meta"], "evalpaper_harness"
    dp = osp.join(DRPARTS, f"{m}_{ds}_fold{k}.json")
    J = json.load(open(dp))
    assert J["meta"]["run_id"] == f"{m}_{ds}_fold{k}"
    return dp, J["meta"], "deep_rerun"


def device_type(meta):
    d = meta.get("device_all") or meta.get("device", "")
    s = sorted({"cuda" if "cuda" in x else "cpu" for x in d.split("+") if x})
    assert len(s) == 1, (meta["run_id"], d)   # a mixed-device c940457 run would need a ruling
    return s[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=osp.join(A17, "tests", "runtime_smoke.json"))
    a = ap.parse_args()
    rt = {}
    if osp.exists(a.runtime):
        for r in json.load(open(a.runtime)):
            if r.get("est_full_h") is not None:
                key = (r["method"], r["dataset"], r["device"])
                old = rt.get(key)
                if old is None or r["epochs"] > old["epochs"]:
                    rt[key] = r
    plan, tasks = {}, []
    for m in METHODS:
        for ds in UB9:
            devs, srcs, thr = set(), set(), set()
            for k in range(5):
                path, meta, src = c940457_part(m, ds, k)
                dev = device_type(meta)
                th = meta.get("threads") if dev == "cpu" else 2
                th = int(th or 2)
                devs.add(dev)
                srcs.add(src)
                thr.add(th)
                r = rt.get((m, ds, dev))
                est_h = r["est_full_h"] if r else meta.get("seconds", 0) / 3600.0
                mem = DEF_MEM.get(ds, {}).get(m, 8.0)
                gmem = DEF_GMEM.get(ds, {}).get(m, 4.0) if dev == "cuda" else 0.0
                if r:
                    mem = round(max(2.0, 1.3 * r["maxrss_gb"] + 2.0), 0)
                    if dev == "cuda":
                        gmem = round(max(1.0, 1.25 * r["cuda_max_reserved_gb"] + 1.0), 0)
                tasks.append(dict(method=m, ds=ds, fold=k, dev=dev, threads=th, mem=mem, gmem=gmem, est_h=est_h,
                                  est_source="smoke timing" if r else "c940457 seconds (no a96137b timing yet)",
                                  c940457_part=osp.relpath(path, PRELIM), c940457_source=src,
                                  c940457_device=meta.get("device"), c940457_gpu=meta.get("cuda_name")))
            assert len(devs) == 1, (m, ds, devs)   # one device type per (method, dataset)
            plan[f"{m} {ds}"] = dict(device=devs.pop(), c940457_source="+".join(sorted(srcs)), threads=sorted(thr))
    tasks.sort(key=lambda t: -t["est_h"])
    lines = [f"# A17 a96137b grid: {len(tasks)} tasks, longest first; generated {time.strftime('%F %T')} by "
             f"jobs/gen_tasks.py (estimates: {'smoke timing' if rt else 'c940457 seconds'}).",
             "# METHOD DATASET FOLD dev=cpu|cuda mem=GB gmem=GB threads=N   # est_h, c940457 source"]
    for t in tasks:
        g = f" gmem={t['gmem']:g}" if t["dev"] == "cuda" else ""
        lines.append(f"{t['method']} {t['ds']} {t['fold']} dev={t['dev']} mem={t['mem']:g}{g} threads={t['threads']}"
                     f"   # est {t['est_h']:.2f} h ({t['est_source']}); c940457 {t['c940457_source']} {t['c940457_device']}")
    open(osp.join(HERE, "tasks_all.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(generated=time.strftime("%F %T"), rule="device type of the paired c940457 run (NOTES.md §3.4)",
                   plan=plan, tasks=tasks), open(osp.join(HERE, "device_plan.json"), "w"), indent=1)
    n = {d: sum(t["dev"] == d for t in tasks) for d in ("cpu", "cuda")}
    print(f"{len(tasks)} tasks ({n['cpu']} cpu, {n['cuda']} cuda) -> jobs/tasks_all.txt, jobs/device_plan.json")
    for k, v in plan.items():
        print(f"  {k:24s} {v['device']:5s} threads {v['threads']}  (c940457 {v['c940457_source']})")
    tot = {d: sum(t["est_h"] * t["threads"] for t in tasks if t["dev"] == d) for d in ("cpu", "cuda")}
    print(f"  estimated task-hours x threads: cpu {tot['cpu']:.1f}, cuda tasks {sum(t['est_h'] for t in tasks if t['dev'] == 'cuda'):.1f} h")


if __name__ == "__main__":
    main()
