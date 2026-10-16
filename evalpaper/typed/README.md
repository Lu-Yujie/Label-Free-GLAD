# Typed-anomaly suite

The typed suite tests whether a detector responds to specific, controlled structural changes. It has two generators
and five seeds each. Every seed has 500 or more normal training graphs, of which 20% are held out as in the main
evaluation, a set of test normals, and one test set per anomaly type. A detector is trained once per (generator,
seed, feature variant) and scored on every type.

## GX (synthetic)

`m2_typed/gens.py` (`gx_set`), exported once to `data/m2_data/GX_s<seed>.npz` by `m2_typed/export_data.py`. A normal
graph is a preferential-attachment tree with 20 to 30 nodes and ten attached motifs: two stars and eight five-node
rings, each ring a house or a bowtie with equal probability. Every graph of every type has the same number of nodes
and edges. Each seed has 500 training normals and 250 test normals, plus 250 anomalies per type.

| Type | Change |
| --- | --- |
| T1a | two of the eight rings are K2,3, a motif that never occurs in normals |
| T3a | houses are over-used (90% instead of 50%) |
| T3b | houses are under-used (10%) |

Feature variants: F2 (one-hot node label of four values plus a two-dimensional attribute) and F1 (one-hot label only).

## MS-v2 (molecular)

`molswap_v2.py` edits real NCI1 molecules of the normal class. Each seed has 982 training normals and 327 test
normals. Every edit keeps each atom's label and only uses (label, degree) pairs that occur in real normal molecules.
Each type comes in two variants. In the size-matched variant (`matched`) the anomaly keeps the molecule's size. In the
size-changing variant (`sizechg`) the anomaly is joined by one edge to an unmodified donor molecule.

| Type | Edit |
| --- | --- |
| MS1 | a six-membered ring becomes a four-membered ring with a two-atom chain |
| MS2 | one ring is opened and a new ring of four to six atoms is closed in an acyclic part (two edits) |
| MS3 | a six-membered ring becomes a three-membered ring with a chain |

The generated data are in `data/molswap_v2/`. `manifest.json` (sha256
`7d7ddf76749425a74bed8d7e5c667b510318f6ef7b755a1224ee8d8dd73f2658`) lists the sha256 of every data file, and
`typed_data.py` checks both before any use. `validation_msv2.json` and `negcheck_msv2.json` are the generator's
validity checks and negative controls, and `msv2_work/` holds the design grid from which the three types were chosen
by a rule fixed in advance.

The first molecular generator (`molswap.py`, data in `data/molswap/`, manifest sha256
`123f0cda46c0b343285938071a10d73c7a6dda1d9e80914fef2cb2621c5aa6ea`) was retired because the reference control
(graphlet counts) stayed below the validity threshold of 85 AUROC on all three types, so none of them was a valid
test. It is kept as a record, and no full deep run was made on it.

## Running

`typed_data.py` loads any (generator, seed, feature variant) with the held-out split and the test sets per type, and
`controls_typed.py` scores the training-free controls. Both files are frozen with their machine paths, so run them, and
code that imports them, through `scripts/portable_run.py`:

```bash
python -B scripts/portable_run.py evalpaper/typed/typed_data.py selftest
python -B scripts/portable_run.py evalpaper/typed/run_typed.py --help      # the 12 UB-GOLD detectors
python -B scripts/portable_run.py evalpaper/typed/ext/run_pool_ext.py --help   # the nine recent methods
```

`collect_typed.py` and `ext/collect_typed_ext.py` write `results/eval_typed.csv` and `results/eval_typed_ext.csv`.
