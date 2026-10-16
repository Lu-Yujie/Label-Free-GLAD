"""Check that StratifiedKFold folds from scikit-learn 1.0.2 (UB-GOLD's pinned version)
equal those from the installed scikit-learn, for every cached TU dataset.

The method ``StratifiedKFold._make_test_folds`` is extracted from the official 1.0.2
source (downloaded to $GLAD_DATA_ROOT/ext/sklearn102/_split_1.0.2.py) and run
standalone; numpy's legacy RandomState stream is version-stable.
Output: $GLAD_DATA_ROOT/ubgold_splits/sklearn_1.0.2_check.json
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import ast
import json
import os.path as osp
import sys
import warnings

PRELIM = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PRELIM)

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state, column_or_1d
from sklearn.utils.multiclass import type_of_target

from common.data import list_datasets, load_raw_cache

SRC = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext/sklearn102/_split_1.0.2.py")


def load_102():
    tree = ast.parse(open(SRC).read())
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StratifiedKFold"][0]
    fn = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_make_test_folds"][0]
    ns = dict(np=np, check_random_state=check_random_state, column_or_1d=column_or_1d,
              type_of_target=type_of_target, warnings=warnings)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), ns)

    class Old:
        n_splits, shuffle, random_state = 5, True, 0
    old = Old()
    return lambda y: ns["_make_test_folds"](old, np.zeros(len(y)), y)


def main():
    mk = load_102()
    out = {"sklearn_installed": sklearn.__version__, "datasets": {}}
    for DS in list_datasets(include_tox21=False):
        obj = load_raw_cache(DS)
        y = obj["labels_orig"]
        tf_old = mk(y)
        tf_new = np.full(len(y), -1)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
            tf_new[te] = k
        tf_cache = np.full(len(y), -1)
        for f in obj["folds"]:
            tf_cache[f["test_idx"]] = f["fold"]
        out["datasets"][DS] = {"same_102_vs_installed": bool(np.array_equal(tf_old, tf_new)),
                               "same_102_vs_cache": bool(np.array_equal(tf_old, tf_cache))}
        print(DS, out["datasets"][DS], flush=True)
    with open((os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_splits/sklearn_1.0.2_check.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
