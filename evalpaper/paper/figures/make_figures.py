"""Paper figures from the frozen-stats outputs (results/eval_leaderboard_v2_*.csv).

    $GLAD_ENVS/paper-figs/bin/python -B evalpaper/paper/figures/make_figures.py

Writes fig1_dumbbell.pdf, fig2_inflation.pdf, fig3_hpleak.pdf (and .png previews) next to this file.
Palette: validated categorical slots 1-3 (blue = UB-GOLD detectors, orange = recent methods, aqua = training-free
references); marker shape is a second encoding (circle / square / diamond) for greyscale print and CVD readers.
"""
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
RES = osp.abspath(osp.join(HERE, "..", "..", "..", "results"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
GROUP = {"frozen12": ("UB-GOLD detectors", BLUE, "o"), "E5": ("Recent methods", ORANGE, "s"),
         "E4": ("Recent methods", ORANGE, "s")}
NAMES = {"GraphCL_IF": "GraphCL-IF", "GraphCL_OCSVM": "GraphCL-OCSVM", "InfoGraph_IF": "InfoGraph-IF",
         "InfoGraph_OCSVM": "InfoGraph-OCSVM", "SUL-GUL": "SUL/GUL", "GOOD-D": "GOOD-D"}
REFS = {"GRAIL-0": "GRAIL-0", "graphlet-count+LW": "Graphlet count", "size-only": "Size only"}

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0, "pdf.fonttype": 42, "ps.fonttype": 42})


def name(m):
    return NAMES.get(m, m)


def style_axes(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(osp.join(HERE, f"{stem}.{ext}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def fig1_dumbbell():
    """UB9 mean AUROC per detector: test-label selection (ORACLE, hollow) vs label-free (VAL-NORMAL, filled)."""
    lb = pd.read_csv(osp.join(RES, "eval_leaderboard_v2_leaderboard.csv"))
    lb = lb[(lb.set == "UB9") & (lb.pool == "all")]
    det = lb[lb.stratum.isin(GROUP)]
    wide = det.pivot_table(index=["method", "stratum"], columns="protocol", values="mean_auroc").reset_index()
    wide = wide.sort_values("valnormal").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.25, 3.9))
    for i, r in wide.iterrows():
        lab, col, mk = GROUP[r.stratum]
        ax.plot([r.valnormal, r.oracle], [i, i], color=col, alpha=0.45, linewidth=1.6, zorder=2,
                solid_capstyle="round")
        ax.scatter(r.oracle, i, s=16, facecolor="white", edgecolor=col, linewidth=1.0, marker=mk, zorder=3)
        ax.scatter(r.valnormal, i, s=16, color=col, marker=mk, zorder=4, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(wide)))
    ax.set_yticklabels([name(m) for m in wide.method])
    refs = lb[lb.method.isin(REFS) & (lb.protocol == "valnormal")]
    for _, r in refs.iterrows():
        ax.axvline(r.mean_auroc, color=AQUA, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    # label the references at the top, staggered to avoid overlap
    for _, r in refs.iterrows():
        ax.text(r.mean_auroc - 0.25, len(wide) + 0.1, REFS[r.method], color=INK, fontsize=5.5, ha="right",
                va="bottom", rotation=90)
    ax.set_xlabel("Mean AUROC on UB9 (9 datasets $\\times$ 5 folds)")
    ax.set_xlim(43, 78)
    ax.set_ylim(-0.8, len(wide) + 3.6)
    style_axes(ax)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=4, label="UB-GOLD detector"),
         Line2D([], [], marker="s", color=ORANGE, linestyle="", markersize=4, label="Recent method"),
         Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=MUTED, linestyle="", markersize=4,
                label="Test-label selection"),
         Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=4, label="Label-free checkpoint (Val-Normal)"),
         Line2D([], [], color=AQUA, linestyle=(0, (3, 2)), linewidth=0.8, label="Training-free reference")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=2, frameon=False, fontsize=6,
              handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig1_dumbbell")
    return wide


def fig2_inflation():
    """Per-method inflation under both label-free selectors (UB-GOLD-rule ORACLE for every method; A26)."""
    inf = pd.read_csv(osp.join(RES, "eval_leaderboard_v2_inflation.csv"))
    inf = inf[(inf.set == "UB9") & inf.stratum.isin(GROUP)]
    comps = {"checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL": "vs. Val-Normal",
             "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST": "vs. Last"}
    inf = inf[inf.comparison.isin(comps) & (inf.method != "ProtoGLAD")]
    order = (inf[inf.comparison.str.endswith("VAL-NORMAL")].sort_values("mean_diff").method.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), sharey=True)
    for ax, (comp, title) in zip(axes, comps.items()):
        d = inf[inf.comparison == comp].set_index("method").loc[order].reset_index()
        for i, r in d.iterrows():
            lab, col, mk = GROUP[r.stratum]
            ax.plot([r.ds_ci_lo, r.ds_ci_hi], [i, i], color=col, alpha=0.5, linewidth=1.2, zorder=2,
                    solid_capstyle="round")
            ax.scatter(r.mean_diff, i, s=16, color=col, marker=mk, zorder=3, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.axvline(2, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
        ax.set_title(f"Test-label selection {title}", fontsize=7, color=INK)
        ax.set_xlabel("Checkpoint leakage (AUROC points)")
        style_axes(ax)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([name(m) for m in order])
    save(fig, "fig2_inflation")


def fig3_hpleak():
    """Hyperparameter leak: VAL-NORMAL(tuned) - VAL-NORMAL(defaults), UB9 mean, dataset-bootstrap CI."""
    hp = pd.read_csv(osp.join(RES, "eval_leaderboard_v2_hpleak.csv"))
    hp = hp[hp.stratum.isin(GROUP) & hp.excluded_from_pooled.isna()].sort_values("ub9_mean_hp_leak")
    fig, ax = plt.subplots(figsize=(3.25, 2.75))
    for i, (_, r) in enumerate(hp.iterrows()):
        lab, col, mk = GROUP[r.stratum]
        ax.barh(i, r.ub9_mean_hp_leak, color=col, height=0.62, zorder=2)
        ax.plot([r.ds_ci_lo, r.ds_ci_hi], [i, i], color=INK, linewidth=0.7, zorder=3)
    ax.set_yticks(range(len(hp)))
    ax.set_yticklabels([name(m) for m in hp.method])
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.set_xlabel("Gain from test-tuned hyperparameters (AUROC points)")
    style_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="UB-GOLD detector"), Patch(color=ORANGE, label="Recent method")],
              loc="lower right", frameon=False, fontsize=6)
    save(fig, "fig3_hpleak")


if __name__ == "__main__":
    w = fig1_dumbbell()
    fig2_inflation()
    fig3_hpleak()
    print("figures written to", HERE, "| detectors in fig1:", len(w))
