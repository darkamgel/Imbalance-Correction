"""
Robustness check: is r=0.50 (the balanced control condition) driving the
paper's headline Correction Harm Rate / Correction Gain / PSG numbers?

r=0.50 is a *balanced* training set (before contamination); it is one of the
four tested imbalance ratios but is not itself an imbalanced condition, so a
reviewer can reasonably ask whether pooling it into the headline numbers
alongside r in {0.20, 0.10, 0.05} muddies a paper about imbalance correction.

This script does NOT rerun any experiment or touch any existing results file.
It re-applies the exact same contribution-metric and statistical-test
functions already defined and used in backup.ipynb (Sections 10.9, 10.14,
10.15, 10.16, 10.16b) to three views of the same already-collected
raw_runs.csv:
  - "pooled"      : all four r values (== the existing, unchanged results/*.csv)
  - "excl_r050"   : r in {0.20, 0.10, 0.05} only -- the actual-imbalance subset
  - "r050_only"   : r == 0.50 only -- the balanced-control condition, reported
                    separately rather than folded into the headline

Outputs go to results/sensitivity_r050/ and figures/sensitivity_r050/, kept
fully separate from the original results/ and figures/ so every existing,
already-verified number in the paper remains reproducible unchanged.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "sensitivity_r050"
FIG_DIR = PROJECT_ROOT / "figures" / "sensitivity_r050"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIRECTIONS = ["majority_to_minority", "minority_to_majority"]

# ---- exact functions from backup.ipynb Section 10.9, verbatim ----

def correction_gain(score_corrected, score_uncorrected, lower_is_better=False):
    gain = score_corrected - score_uncorrected
    return -gain if lower_is_better else gain

def correction_harm_rate(gains, epsilon=0.0):
    gains = np.asarray(gains)
    return float((gains < -epsilon).mean())

def break_even_point(c_values, gains_by_c):
    for c, gain in zip(c_values, gains_by_c):
        if c > 0 and gain <= 0:
            return c
    return f">{max(c_values):.2f}"

def prior_sensitivity_gap(score_oracle_prior, score_observed_prior):
    return score_oracle_prior - score_observed_prior

BASELINE_MAP = {
    "xgb_weighted": "xgb_none", "xgb_ros": "xgb_none", "xgb_smote": "xgb_none",
    "tabpfn_threshold": "tabpfn_none", "tabpfn_distpfn": "tabpfn_none",
    "tabpfn_distpfn_t": "tabpfn_none", "tabpfn_downsample": "tabpfn_none",
}
MERGE_KEYS = ["dataset", "r", "c", "direction"]
PRIOR_METHODS = ["tabpfn_threshold", "tabpfn_distpfn", "tabpfn_distpfn_t"]
XGB_METHODS = ["xgb_weighted", "xgb_ros", "xgb_smote"]
TAB_METHODS = ["tabpfn_threshold", "tabpfn_distpfn", "tabpfn_distpfn_t", "tabpfn_downsample"]
ALL_METHODS = XGB_METHODS + TAB_METHODS
FAMILIES = {
    "xgboost": ["xgb_none", "xgb_weighted", "xgb_ros", "xgb_smote"],
    "tabpfn": ["tabpfn_none", "tabpfn_threshold", "tabpfn_distpfn", "tabpfn_distpfn_t", "tabpfn_downsample"],
}
METRIC_COLS = ["balanced_accuracy", "worst_class_recall", "auprc", "macro_f1",
               "minority_precision", "minority_recall", "auroc", "brier", "log_loss", "ece"]
GROUP_COLS = ["dataset", "method", "prior_mode", "r", "c", "direction"]


def build_summary(raw, r_filter, label):
    """Mirrors 10.14 exactly, applied to a filtered slice of raw_runs.csv."""
    df = raw[r_filter(raw["r"])].copy()
    agg_spec = {col: ["mean", "std"] for col in METRIC_COLS}
    agg_spec["realized_c"] = "mean"
    agg_spec["seed"] = "count"
    summary = df.groupby(GROUP_COLS, dropna=False).agg(agg_spec)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index().rename(columns={"seed_count": "n_seeds"})
    summary.to_csv(OUT_DIR / f"summary_{label}.csv", index=False)
    return summary


def build_gains(summary, label):
    """Mirrors 10.15 exactly."""
    baseline_xgb = summary[summary["method"] == "xgb_none"][MERGE_KEYS + ["balanced_accuracy_mean", "brier_mean"]].rename(
        columns={"balanced_accuracy_mean": "baseline_ba", "brier_mean": "baseline_brier"})
    baseline_tabpfn = summary[summary["method"] == "tabpfn_none"][MERGE_KEYS + ["balanced_accuracy_mean", "brier_mean"]].rename(
        columns={"balanced_accuracy_mean": "baseline_ba", "brier_mean": "baseline_brier"})

    gain_rows = []
    for method, baseline_method in BASELINE_MAP.items():
        baseline_df = baseline_xgb if baseline_method == "xgb_none" else baseline_tabpfn
        sub = summary[summary["method"] == method].merge(baseline_df, on=MERGE_KEYS, how="left")
        sub["balanced_accuracy_gain"] = correction_gain(sub["balanced_accuracy_mean"], sub["baseline_ba"], lower_is_better=False)
        sub["brier_gain"] = correction_gain(sub["brier_mean"], sub["baseline_brier"], lower_is_better=True)
        gain_rows.append(sub)
    gains_view = pd.concat(gain_rows, ignore_index=True)
    gains_view.to_csv(OUT_DIR / f"correction_gains_{label}.csv", index=False)

    deployable_mask = gains_view["prior_mode"].isin(["observed"]) | gains_view["prior_mode"].isna()
    noisy = gains_view[(gains_view["c"] > 0) & deployable_mask]

    chr_table = noisy.groupby(["method", "direction"])["balanced_accuracy_gain"].apply(correction_harm_rate).reset_index(name="CHR")
    mean_cg_table = noisy.groupby(["method", "direction"])["balanced_accuracy_gain"].mean().reset_index(name="mean_CG")
    n_table = noisy.groupby(["method", "direction"])["balanced_accuracy_gain"].size().reset_index(name="n_cells")
    chr_cg = chr_table.merge(mean_cg_table, on=["method", "direction"]).merge(n_table, on=["method", "direction"])
    chr_cg.to_csv(OUT_DIR / f"chr_cg_table_{label}.csv", index=False)

    be_rows = []
    for (method, dataset, r, direction), g in noisy.groupby(["method", "dataset", "r", "direction"]):
        g = g.sort_values("c")
        be = break_even_point(g["c"].tolist(), g["balanced_accuracy_gain"].tolist())
        be_rows.append(dict(method=method, dataset=dataset, r=r, direction=direction, break_even_c=be))
    break_even_table = pd.DataFrame(be_rows)
    break_even_table.to_csv(OUT_DIR / f"break_even_table_{label}.csv", index=False)

    be_share_rows = []
    for (method, direction), g in break_even_table.groupby(["method", "direction"]):
        never = g["break_even_c"].apply(lambda x: isinstance(x, str)).sum()
        be_share_rows.append(dict(method=method, direction=direction, n_pairs=len(g),
                                   pct_never_broke_even=round(100 * never / len(g), 1)))
    be_share = pd.DataFrame(be_share_rows)
    be_share.to_csv(OUT_DIR / f"break_even_share_{label}.csv", index=False)

    psg_rows = []
    for method in PRIOR_METHODS:
        obs = summary[(summary["method"] == method) & (summary["prior_mode"] == "observed")]
        oracle = summary[(summary["method"] == method) & (summary["prior_mode"] == "oracle")]
        merged = obs.merge(oracle, on=MERGE_KEYS, suffixes=("_observed", "_oracle"))
        merged["method"] = method
        merged["psg"] = prior_sensitivity_gap(merged["balanced_accuracy_mean_oracle"], merged["balanced_accuracy_mean_observed"])
        psg_rows.append(merged[MERGE_KEYS + ["method", "psg"]])
    psg_table = pd.concat(psg_rows, ignore_index=True)
    psg_table = psg_table[psg_table["c"] > 0]
    psg_table.to_csv(OUT_DIR / f"psg_table_{label}.csv", index=False)
    psg_mean = psg_table.groupby(["method", "direction"])["psg"].mean().reset_index(name="mean_PSG")
    psg_mean.to_csv(OUT_DIR / f"psg_mean_{label}.csv", index=False)

    return chr_cg, be_share, psg_mean


def friedman_wilcoxon(summary, label):
    """Mirrors 10.16 (condition-level, block=(dataset,r,c)) and 10.16b (dataset-level)."""
    rows = []
    for family, methods in FAMILIES.items():
        for direction in DIRECTIONS:
            sub = summary[(summary["method"].isin(methods)) & (summary["direction"] == direction)]
            sub = sub[sub["prior_mode"].isin(["observed"]) | sub["prior_mode"].isna()]
            pivot = sub.pivot_table(index=["dataset", "r", "c"], columns="method", values="balanced_accuracy_mean")
            pivot_complete = pivot[methods].dropna()
            if len(pivot_complete) < 10:
                continue
            stat, p = friedmanchisquare(*[pivot_complete[m] for m in methods])
            rows.append(dict(family=family, direction=direction, test="friedman", pair=None, stat=stat, p=p,
                              p_holm=None, significant=bool(p < 0.05), n_blocks=len(pivot_complete), block_level="condition"))
            if p < 0.05:
                pvals, pairs = [], []
                for a, b in combinations(methods, 2):
                    _, pv = wilcoxon(pivot_complete[a], pivot_complete[b])
                    pvals.append(pv); pairs.append((a, b))
                reject, pvals_holm, _, _ = multipletests(pvals, method="holm")
                for (a, b), pv, pc, rej in zip(pairs, pvals, pvals_holm, reject):
                    rows.append(dict(family=family, direction=direction, test="wilcoxon", pair=f"{a} vs {b}",
                                      stat=None, p=pv, p_holm=pc, significant=bool(rej), n_blocks=len(pivot_complete),
                                      block_level="condition"))

            # dataset-level (stricter), pooling every tested r,c within one dataset
            dataset_means = sub.groupby(["dataset", "method"])["balanced_accuracy_mean"].mean().reset_index()
            dpivot = dataset_means.pivot_table(index="dataset", columns="method", values="balanced_accuracy_mean")
            dpivot_complete = dpivot[methods].dropna()
            if len(dpivot_complete) < 10:
                continue
            dstat, dp = friedmanchisquare(*[dpivot_complete[m] for m in methods])
            rows.append(dict(family=family, direction=direction, test="friedman", pair=None, stat=dstat, p=dp,
                              p_holm=None, significant=bool(dp < 0.05), n_blocks=len(dpivot_complete), block_level="dataset"))
            if dp < 0.05:
                pvals, pairs = [], []
                for a, b in combinations(methods, 2):
                    _, pv = wilcoxon(dpivot_complete[a], dpivot_complete[b])
                    pvals.append(pv); pairs.append((a, b))
                reject, pvals_holm, _, _ = multipletests(pvals, method="holm")
                for (a, b), pv, pc, rej in zip(pairs, pvals, pvals_holm, reject):
                    rows.append(dict(family=family, direction=direction, test="wilcoxon", pair=f"{a} vs {b}",
                                      stat=None, p=pv, p_holm=pc, significant=bool(rej), n_blocks=len(dpivot_complete),
                                      block_level="dataset"))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / f"statistical_tests_{label}.csv", index=False)
    return table


# ---- run for the three views ----
raw = pd.read_csv(RESULTS_DIR / "raw_runs.csv")

views = {
    "pooled": lambda r: r.notna(),                 # all 4 ratios (== existing results/*.csv, recomputed here only to diff-check)
    "excl_r050": lambda r: r != 0.5,                # the actual-imbalance subset r in {0.20,0.10,0.05}
    "r050_only": lambda r: r == 0.5,                # balanced control, reported separately
}

all_chr_cg, all_be, all_psg, all_stats = {}, {}, {}, {}
for label, filt in views.items():
    summary = build_summary(raw, filt, label)
    chr_cg, be_share, psg_mean = build_gains(summary, label)
    stats_table = friedman_wilcoxon(summary, label)
    all_chr_cg[label] = chr_cg
    all_be[label] = be_share
    all_psg[label] = psg_mean
    all_stats[label] = stats_table

# ---- combined comparison table (the one the paper table will be built from) ----
comp = None
for label in views:
    t = all_chr_cg[label].rename(columns={"CHR": f"CHR_{label}", "mean_CG": f"mean_CG_{label}", "n_cells": f"n_{label}"})
    comp = t if comp is None else comp.merge(t, on=["method", "direction"])
comp = comp[comp["method"].isin(ALL_METHODS)]
comp.to_csv(OUT_DIR / "comparison_chr_cg_pooled_vs_subsets.csv", index=False)

be_comp = None
for label in views:
    t = all_be[label].rename(columns={"pct_never_broke_even": f"pct_never_broke_even_{label}", "n_pairs": f"n_pairs_{label}"})
    be_comp = t if be_comp is None else be_comp.merge(t, on=["method", "direction"])
be_comp.to_csv(OUT_DIR / "comparison_breakeven_pooled_vs_subsets.csv", index=False)

psg_comp = None
for label in views:
    t = all_psg[label].rename(columns={"mean_PSG": f"mean_PSG_{label}"})
    psg_comp = t if psg_comp is None else psg_comp.merge(t, on=["method", "direction"])
psg_comp.to_csv(OUT_DIR / "comparison_psg_pooled_vs_subsets.csv", index=False)

friedman_comp = pd.concat([all_stats[l][all_stats[l]["test"] == "friedman"].assign(view=l) for l in views], ignore_index=True)
friedman_comp.to_csv(OUT_DIR / "comparison_friedman_pooled_vs_subsets.csv", index=False)

print("=== CHR + mean CG: pooled vs excl_r050 vs r050_only ===")
print(comp.round(4).to_string(index=False))
print("\n=== Break-even share (%): pooled vs excl_r050 vs r050_only ===")
print(be_comp.to_string(index=False))
print("\n=== mean PSG: pooled vs excl_r050 vs r050_only ===")
print(psg_comp.round(4).to_string(index=False))
print("\n=== Friedman omnibus: pooled vs excl_r050 ===")
print(friedman_comp[["family", "direction", "view", "n_blocks", "block_level", "stat", "p", "significant"]].round(6).to_string(index=False))

# sanity check: pooled recomputation here must match the paper's existing chr_table.csv exactly
existing_chr = pd.read_csv(RESULTS_DIR / "chr_table.csv")
check = comp[["method", "direction", "CHR_pooled"]].merge(
    existing_chr.rename(columns={"CHR": "CHR_existing"}), on=["method", "direction"])
check["diff"] = (check["CHR_pooled"] - check["CHR_existing"]).abs()
max_diff = check["diff"].max()
print(f"\nsanity check vs existing results/chr_table.csv: max abs diff = {max_diff:.6f} "
      f"({'PASS' if max_diff < 1e-9 else 'FAIL -- INVESTIGATE'})")

# ---- comparison figure: CHR, pooled vs excl_r050 vs r050_only, grouped bars per method/direction ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
width = 0.25
colors = {"pooled": "#8a8880", "excl_r050": "#2a78d6", "r050_only": "#e34948"}
labels_nice = {"pooled": "pooled (all r)", "excl_r050": "r ≠ 0.50 (actual imbalance)", "r050_only": "r = 0.50 only (balanced control)"}
for ax, direction in zip(axes, DIRECTIONS):
    sub = comp[comp["direction"] == direction].set_index("method").loc[ALL_METHODS]
    x = np.arange(len(ALL_METHODS))
    for i, label in enumerate(["pooled", "excl_r050", "r050_only"]):
        ax.bar(x + (i - 1) * width, sub[f"CHR_{label}"], width, color=colors[label], label=labels_nice[label])
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_METHODS, rotation=30, ha="right")
    ax.set_title(direction)
    ax.set_ylabel("Correction Harm Rate")
axes[1].legend(fontsize=8)
fig.suptitle("Sensitivity check: Correction Harm Rate pooled vs. isolated by r=0.50")
fig.tight_layout()
fig.savefig(FIG_DIR / "figureS1_chr_r050_sensitivity.png", bbox_inches="tight", dpi=150)
print(f"\nsaved -> {FIG_DIR / 'figureS1_chr_r050_sensitivity.png'}")
