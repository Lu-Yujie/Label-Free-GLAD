"""Print the timing table from smoke timing parts (tags tcpu*, tgpu)."""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import glob, json, os.path as osp, sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import run_method as R
rows = []
for fn in sorted(glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_gladmamba/smoke/parts/GLADMamba_*_t*.json"))):
    J = json.load(open(fn)); m = J["meta"]
    tag = m["run_id"].rsplit("_", 1)[1]
    a, _, _ = R.resolve_hparams(m["dataset"], m["hp"])
    ne = a.num_epoch // a.eval_freq
    spe = m["sec_per_epoch_train"]; spv = m["seconds_eval_this_segment"] / max(1, m["n_eval"])
    startup = m["seconds"] - m["seconds_train_this_segment"] - m["seconds_eval_this_segment"]
    est = startup + spe * a.num_epoch + spv * ne
    mem = m["memory"]
    print(f"| {m['dataset']} | {m['hp']} | {tag} | {m['threads']} | {spe:.2f} | {spv:.2f} | {startup:.0f} | {a.num_epoch} | {ne} | {est/60:.1f} | {mem['maxrss_gb']:.1f} | {mem.get('cuda_max_alloc_gb', float('nan')):.2f} |")
