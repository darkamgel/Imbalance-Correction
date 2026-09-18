"""
Backfill the 4 stress cells left incomplete after the full matrix run finished
(Sep 17 2026): 3 in `mushroom` that the resume-skip guard treated as already
complete (it only checks that all 10 seeds are present for a dataset, not that
every cell within each seed succeeded) and 1 in `aps_failure` that is a genuine
compute failure logged in failures.csv (SVD did not converge). All 4 are the
same method, tabpfn_downsample.

Why this replays the whole seed instead of just sampling the one target cell:
sample_controlled_set / majority_to_minority_contamination / minority_to_majority_loss
share one np.random.default_rng(seed) per (dataset, seed) that is consumed in the
exact order the main matrix loop visits cells (every r, then c=0 clean, then every
c x direction). Jumping straight to the target cell would draw from the wrong point
in that sequence and reconstruct a *different* corrupted training set than the one
every other method in that cell was already scored against, which would break the
matched-cell comparison the paper's paired tests rely on. So this replays every
earlier cell for that seed in the same order first -- but only the sampling/corruption
step (cheap, no model fitting), since fit_preprocessing/apply_preprocessing and every
xgb_*/tabpfn_* function take a plain seed:int, not the shared rng, and so cannot
perturb it. Only at the actual target cell does it run tabpfn_downsample_predict for
real. Same technique as the 10.13d recovery cell in backup.ipynb, generalized to
r != 0.50 (10.13d could rely on r=0.50 being R_VALUES[0], i.e. the rng's fresh state;
these 4 targets are not all at r=0.50, so this version replays the full prior sequence
instead of assuming a fresh rng).

Safe to re-run: skips any target already present in raw_runs.csv.

Run from the project root with the notebook venv active (same as run_with_retry.sh):
    source .venv_notebook/bin/activate
    python3 backfill_missing_cells.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()

# Load only the setup / function-definition portion of run_full_matrix.py (everything
# before cell 46, the full-26-dataset matrix driver) so DATASETS and every helper
# function are available without re-triggering the full run.
_src = (PROJECT_ROOT / "run_full_matrix.py").read_text()
_marker = "# ===== cell 46 ====="
_cutoff = _src.index(_marker)
_ns = {"__name__": "run_full_matrix_setup", "__file__": str(PROJECT_ROOT / "run_full_matrix.py")}
print("Loading dataset + function definitions from run_full_matrix.py (cells 1-45)...")
exec(compile(_src[:_cutoff], "run_full_matrix.py", "exec"), _ns)
print("Done loading.\n")

sample_controlled_set = _ns["sample_controlled_set"]
majority_to_minority_contamination = _ns["majority_to_minority_contamination"]
minority_to_majority_loss = _ns["minority_to_majority_loss"]
fit_preprocessing = _ns["fit_preprocessing"]
apply_preprocessing = _ns["apply_preprocessing"]
tabpfn_downsample_predict = _ns["tabpfn_downsample_predict"]
compute_all_metrics = _ns["compute_all_metrics"]
stratified_split = _ns["stratified_split"]
compute_common_training_size = _ns["compute_common_training_size"]
DATASETS = _ns["DATASETS"]
R_VALUES = _ns["R_VALUES"]
C_VALUES = _ns["C_VALUES"]
DIRECTIONS = _ns["DIRECTIONS"]
RESULTS_DIR = _ns["RESULTS_DIR"]

METHOD = "tabpfn_downsample"

TARGETS = [
    dict(dataset="mushroom", seed=5, r=0.05, c=0.10, direction="minority_to_majority"),
    dict(dataset="mushroom", seed=11, r=0.05, c=0.20, direction="minority_to_majority"),
    dict(dataset="mushroom", seed=31, r=0.05, c=0.30, direction="majority_to_minority"),
    dict(dataset="aps_failure", seed=19, r=0.10, c=0.05, direction="majority_to_minority"),
]

raw_runs = pd.read_csv(RESULTS_DIR / "raw_runs.csv")
failures = pd.read_csv(RESULTS_DIR / "failures.csv")

already_logged = set(zip(raw_runs["dataset"], raw_runs["seed"], raw_runs["r"],
                          raw_runs["c"], raw_runs["direction"], raw_runs["method"]))

recovered_rows, still_failed_rows = [], []

for dataset_slug, seed in sorted({(t["dataset"], t["seed"]) for t in TARGETS}):
    seed_targets = {(t["r"], t["c"], t["direction"]) for t in TARGETS
                     if t["dataset"] == dataset_slug and t["seed"] == seed}
    pending = {key for key in seed_targets
               if (dataset_slug, seed, key[0], key[1], key[2], METHOD) not in already_logged}

    if not pending:
        print(f"[{dataset_slug} seed={seed}] all target cell(s) already present -- skipping")
        continue

    print(f"[{dataset_slug} seed={seed}] replaying rng to reach {len(pending)} target cell(s): {sorted(pending)}")

    X_full = DATASETS[dataset_slug]["X"].reset_index(drop=True)
    y_full = DATASETS[dataset_slug]["y"]
    train_idx, test_idx = stratified_split(X_full, y_full, seed=seed)
    X_train_full = X_full.loc[train_idx].reset_index(drop=True)
    y_train_full = y_full[train_idx]
    X_test = X_full.loc[test_idx].reset_index(drop=True)
    y_test = y_full[test_idx]

    min_idx = np.flatnonzero(y_train_full == 1)
    maj_idx = np.flatnonzero(y_train_full == 0)
    n_common = compute_common_training_size(R_VALUES, len(min_idx), len(maj_idx))
    rng = np.random.default_rng(seed)

    def maybe_recover(r, c, direction_label, y_ctrl_input, realized, X_ctrl):
        key = (r, c, direction_label)
        if key not in pending:
            return
        try:
            fitted = fit_preprocessing(X_ctrl)
            X_ctrl_proc = apply_preprocessing(X_ctrl, fitted)
            X_test_proc = apply_preprocessing(X_test, fitted)
            y_pred_ds, p_ds = tabpfn_downsample_predict(X_ctrl_proc, y_ctrl_input, X_test_proc, seed)
            m = compute_all_metrics(y_test, y_pred_ds, p_ds)
            recovered_rows.append(dict(dataset=dataset_slug, seed=seed, r=r, c=c,
                                        direction=direction_label, realized_c=realized,
                                        method=METHOD, prior_mode="n/a", **m))
            print(f"  recovered r={r} c={c} dir={direction_label}")
        except Exception as e:
            still_failed_rows.append(dict(dataset=dataset_slug, seed=seed, r=r, c=c,
                                           direction=direction_label, method=METHOD, error=str(e)))
            print(f"  STILL FAILING r={r} c={c} dir={direction_label}: {e}")

    for r in R_VALUES:
        sel = sample_controlled_set(r, n_common, min_idx, maj_idx, rng)
        X_ctrl = X_train_full.loc[sel].reset_index(drop=True)
        y_ctrl_clean = y_train_full[sel]
        maybe_recover(r, 0.0, "clean", y_ctrl_clean, 0.0, X_ctrl)

        for c in C_VALUES[1:]:
            for direction in DIRECTIONS:
                sel = sample_controlled_set(r, n_common, min_idx, maj_idx, rng)
                X_ctrl = X_train_full.loc[sel].reset_index(drop=True)
                y_ctrl_clean = y_train_full[sel]
                noise_fn = (majority_to_minority_contamination if direction == "majority_to_minority"
                            else minority_to_majority_loss)
                y_noisy, _, realized = noise_fn(y_ctrl_clean, c, rng)
                maybe_recover(r, c, direction, y_noisy, realized, X_ctrl)

print(f"\nrecovered: {len(recovered_rows)}   still failing: {len(still_failed_rows)}")

if recovered_rows:
    raw_runs_updated = pd.concat([raw_runs, pd.DataFrame(recovered_rows)], ignore_index=True)
    raw_runs_updated.to_csv(RESULTS_DIR / "raw_runs.csv", index=False)

    recovered_key_set = {(r_["dataset"], r_["seed"], r_["r"], r_["c"], r_["direction"], r_["method"])
                          for r_ in recovered_rows}
    failures_updated = failures[~failures.apply(
        lambda row: (row["dataset"], row["seed"], row["r"], row["c"], row["direction"], row["method"]) in recovered_key_set,
        axis=1
    )]
    if still_failed_rows:
        failures_updated = pd.concat([failures_updated, pd.DataFrame(still_failed_rows)], ignore_index=True)
    failures_updated.to_csv(RESULTS_DIR / "failures.csv", index=False)

    print(f"raw_runs.csv: {len(raw_runs)} -> {len(raw_runs_updated)} rows")
    print(f"failures.csv: {len(failures)} -> {len(failures_updated)} rows")
else:
    print("nothing recovered, files left unchanged")
