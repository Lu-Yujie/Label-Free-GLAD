"""A17 smoke S3 verdicts from identity_check.py outputs (NOTES.md §4). Prints booleans only (no metric values).

  python -B tests/compare_identity.py <dir with {M}_{DS}_{c94,a96,a96sync}.json> [--out verdict.json]

S3a  initial model state equal; every epoch-1 training-batch tensor equal; test/held-out sets equal (npz index arrays)
S3b  c940457 RNG state at the first training forward == a96137b's advanced by exactly the loader draws PR #3 moved
     (GLADC: one test-loader iteration = 2 draws; GLocalKD: one test-loader + one train-loader iteration = 4 draws)
S3c  with that RNG state synchronised, the epoch-1 training loss is bit-identical
S3d  c940457 reuses its epoch-1 batches in epoch 2+; a96137b's epoch-2 batches differ (reshuffle)
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os.path as osp  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

DRAWS = {"GLADC": 2, "GLocalKD": 4}


def advance(hexstate, k):
    torch.set_rng_state(torch.from_numpy(np.frombuffer(bytes.fromhex(hexstate), dtype=np.uint8).copy()))
    for _ in range(k):
        torch.empty((), dtype=torch.int64).random_()
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    for f94 in sorted(glob.glob(osp.join(a.dir, "*_c94.json"))):
        key = osp.basename(f94)[:-len("_c94.json")]
        m = key.split("_")[0]
        A = {s: json.load(open(osp.join(a.dir, f"{key}_{s}.json"))) for s in ("c94", "a96", "a96sync")
             if osp.exists(osp.join(a.dir, f"{key}_{s}.json"))}
        c, d = A["c94"], A["a96"]
        E = c["epochs"]
        nb = len(c["calls"]) // E
        assert nb * E == len(c["calls"]) == len(d["calls"]), (key, len(c["calls"]), len(d["calls"]))
        keys = [k for k in c["calls"][0] if k != "rng"]
        r = dict(case=key, batches_per_epoch=nb, epochs=E)
        r["S3a_init_equal"] = c["init"] == d["init"] and len(c["init"]) == (2 if m == "GLADC" else 2)
        r["S3a_epoch1_batches_equal"] = all(c["calls"][i][k] == d["calls"][i][k] for i in range(nb) for k in keys)
        zc, zd = np.load(c["npz"]), np.load(d["npz"])
        r["S3a_test_holdout_sets_equal"] = all(np.array_equal(zc[k], zd[k]) for k in
                                               ("y_test", "test_index", "hold_pos", "test_n_nodes", "hold_n_nodes"))
        c_rng = hashlib.sha256(bytes.fromhex(c["rng_first_hex"])).hexdigest()
        r["S3b_rng_first_forward_equal_without_offset"] = c_rng == hashlib.sha256(bytes.fromhex(d["rng_first_hex"])).hexdigest()
        r["S3b_rng_equal_after_moved_draws"] = advance(d["rng_first_hex"], DRAWS[m]) == c_rng
        Jc, Jd = json.load(open(c["part"])), json.load(open(d["part"]))
        r["epoch1_train_loss_equal_unsynced"] = Jc["train_loss"][0] == Jd["train_loss"][0]
        if "a96sync" in A:
            s = A["a96sync"]
            Js = json.load(open(s["part"]))
            r["S3c_sync_epoch1_batches_equal"] = all(c["calls"][i][k] == s["calls"][i][k] for i in range(nb) for k in keys)
            r["S3c_sync_epoch1_train_loss_equal"] = Jc["train_loss"][0] == Js["train_loss"][0]
            r["S3c_sync_epoch2_train_loss_equal"] = Jc["train_loss"][1] == Js["train_loss"][1]
        r["S3d_c940457_epoch2_reuses_epoch1_batches"] = all(c["calls"][nb + i][k] == c["calls"][i][k]
                                                            for i in range(nb) for k in keys)
        r["S3d_a96137b_epoch2_batches_differ"] = all(d["calls"][nb + i]["adj"] != d["calls"][i]["adj"] for i in range(nb))
        r["S0_glocalkd_process_graph_same_code"] = d.get("process_graph_ast_equal_to_harness_copy")
        ok = (r["S3a_init_equal"] and r["S3a_epoch1_batches_equal"] and r["S3a_test_holdout_sets_equal"]
              and r["S3b_rng_equal_after_moved_draws"] and r.get("S3c_sync_epoch1_train_loss_equal", False)
              and r["S3d_c940457_epoch2_reuses_epoch1_batches"] and r["S3d_a96137b_epoch2_batches_differ"])
        r["PASS"] = bool(ok)
        out.append(r)
        print(json.dumps(r))
    print(f"S3 identity: {sum(r['PASS'] for r in out)}/{len(out)} cases PASS")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
