"""Point the post hoc analysis cache at your copy of the run records.

The cache files (one JSON per run, in $GLAD_DATA_ROOT/posthoc_cache/) store the path of the run record and of the
score file they were computed from, written as $REPO/... and $GLAD_DATA_ROOT/.... evalpaper/stats/posthoc_robustness.py
compares them with the absolute paths of your copy, so run this once after unpacking the archives:

  python -B scripts/localize_records.py            # uses $GLAD_DATA_ROOT and this repository's location
"""
import json
import os
import os.path as osp
import sys

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))


def main():
    root = os.environ.get("GLAD_DATA_ROOT")
    if not root:
        sys.exit("set GLAD_DATA_ROOT first")
    root = osp.abspath(root)
    d = osp.join(root, "posthoc_cache")
    n = 0
    for dp, _, fn in os.walk(d):
        for f in fn:
            if not f.endswith(".json"):
                continue
            p = osp.join(dp, f)
            obj = json.load(open(p))
            changed = False
            for k in ("part", "npz", "fold_cache"):
                v = obj.get(k)
                if isinstance(v, str):
                    w = v.replace("$REPO/", REPO + "/").replace("$GLAD_DATA_ROOT/", root + "/")
                    if w != v:
                        obj[k], changed = w, True
            if changed:
                with open(p, "w") as fh:
                    json.dump(obj, fh, separators=(",", ":"))
                    fh.write("\n")
                n += 1
    print(f"localized {n} cache files in {d}")


if __name__ == "__main__":
    main()
