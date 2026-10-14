# When Imbalance Correction Backfires

Robustness of tabular imbalance mitigation under minority-label contamination.

The study lives in [`backup.ipynb`](backup.ipynb): data preparation, the stress-cell pipeline, the metric definitions, and the analysis that turns the evaluation log into the paper tables and figures. [`run_full_matrix.py`](run_full_matrix.py) is that same matrix loop copied out of the notebook so the 26-dataset run could be resumed from the shell. The published log, `results/raw_runs.csv`, was produced by the script.

This repository also holds the processed UCI datasets, the archived evaluations, and the manuscript (`paper/main.tex`, with an ACM copy in `paper/ACM/`). Reviewers can check every reported table against the files below. A full refit is optional and is not required to verify the paper.

## Verify the tables without rerunning

The original evaluation log is [`results/raw_runs.csv`](results/raw_runs.csv):

| | |
|---|---|
| Rows | 112,319 successful evaluations (112,320 planned) |
| Datasets | 26 |
| Seeds | 13, 37, 73, 5, 11, 17, 19, 23, 29, 31 |
| SHA-256 | `5365dc6bfbeb4365940e80648cfd91389070cfbc33a3c16a0d14bf213d638d0e` |
| Size | 28,063,835 bytes |

One evaluation did not finish: `tabpfn_downsample` on APS Failure, seed 19, \(r = 0.10\), \(c = 0.05\), Direction A (`majority_to_minority`), with a singular-value decomposition non-convergence error. That cell is recorded in [`results/failures.csv`](results/failures.csv) and is left out of the tables. The CSV lists the same failure twice; it is one cell.

Direction A is `majority_to_minority`. Direction B is `minority_to_majority`.

### Where each manuscript table lives

| Paper result | Exact values |
|---|---|
| Correction Harm Rate (headline) | [`results/chr_table.csv`](results/chr_table.csv) |
| CHR and mean Correction Gain, including the \(r = 0.50\) split | [`tables/table6_r050_sensitivity_chr_cg.csv`](tables/table6_r050_sensitivity_chr_cg.csv) |
| Break-even share | [`tables/table6b_r050_sensitivity_breakeven.csv`](tables/table6b_r050_sensitivity_breakeven.csv) |
| Per-(dataset, \(r\)) break-even contamination | [`results/break_even_table.csv`](results/break_even_table.csv) |
| Mean Prior Sensitivity Gap | [`tables/table5b_psg_summary.csv`](tables/table5b_psg_summary.csv), cell-level gaps in [`results/psg_table.csv`](results/psg_table.csv) |
| PSG on the \(r = 0.50\) split | [`tables/table6c_r050_sensitivity_psg.csv`](tables/table6c_r050_sensitivity_psg.csv) |
| Friedman and Wilcoxon tests | [`results/statistical_tests.csv`](results/statistical_tests.csv), [`tables/table5_statistical_comparisons.csv`](tables/table5_statistical_comparisons.csv) |
| Friedman on each \(r\) view | [`tables/table6d_r050_sensitivity_friedman.csv`](tables/table6d_r050_sensitivity_friedman.csv) |
| Bootstrap intervals for CHR and mean CG | [`tables/table7_chr_cg_bootstrap_ci.csv`](tables/table7_chr_cg_bootstrap_ci.csv) |
| Clean (\(c = 0\)) vs severe (\(c = 0.30\)) balanced accuracy | [`tables/table3_clean_vs_severe.csv`](tables/table3_clean_vs_severe.csv) |
| Dataset characteristics | [`tables/table1_dataset_characteristics.csv`](tables/table1_dataset_characteristics.csv) |
| Seed-averaged metrics | [`results/summary.csv`](results/summary.csv) |
| Seed-averaged correction gains | [`results/correction_gains.csv`](results/correction_gains.csv) |

`results/raw_runs_pilot.csv` is an earlier pilot log. The manuscript uses `results/raw_runs.csv`.

`results/sensitivity_r050/` is an earlier pass with 160 cells per method and direction (10 datasets). The manuscript's 26-dataset sensitivity numbers are the `tables/table6*.csv` files (416 pooled cells, 312 with \(r \neq 0.50\), 104 at \(r = 0.50\)).

### Headline robustness

CHR and mean Correction Gain on balanced accuracy, over noisy blocks only (\(c > 0\)), seed-averaged, then pooled across 26 datasets and four ratios. \(n = 416\) cells per method and direction. These are the values behind Table I of the results section, rounded to three decimals. Exact CHR is in `results/chr_table.csv`. Exact mean CG is `mean_CG_pooled` in `tables/table6_r050_sensitivity_chr_cg.csv`.

| Method | CHR A | CHR B | mean CG A | mean CG B |
|---|---:|---:|---:|---:|
| xgb_weighted | 0.130 | 0.060 | 0.026 | 0.034 |
| xgb_ros | 0.147 | 0.067 | 0.026 | 0.035 |
| xgb_smote | 0.135 | 0.072 | 0.023 | 0.033 |
| tabpfn_threshold | 0.048 | 0.024 | 0.072 | 0.093 |
| tabpfn_distpfn | 0.351 | 0.438 | 0.009 | −0.004 |
| tabpfn_distpfn_t | 0.339 | 0.474 | 0.008 | −0.009 |
| tabpfn_downsample | 0.204 | 0.077 | 0.054 | 0.073 |

The same quantities with the balanced control \(r = 0.50\) removed (\(n = 312\)):

| Method | CHR A | CHR B | mean CG A | mean CG B |
|---|---:|---:|---:|---:|
| xgb_weighted | 0.115 | 0.032 | 0.028 | 0.043 |
| xgb_ros | 0.115 | 0.026 | 0.029 | 0.044 |
| xgb_smote | 0.106 | 0.029 | 0.026 | 0.041 |
| tabpfn_threshold | 0.032 | 0.003 | 0.084 | 0.119 |
| tabpfn_distpfn | 0.349 | 0.298 | 0.009 | 0.013 |
| tabpfn_distpfn_t | 0.359 | 0.378 | 0.004 | −0.004 |
| tabpfn_downsample | 0.186 | 0.051 | 0.061 | 0.094 |

### Break-even share

Share of 104 (dataset, \(r\)) pairs whose seed-averaged Correction Gain stays positive through \(c = 0.30\) (\(c^* > 0.30\)):

| Method | Dir. A | Dir. B |
|---|---:|---:|
| xgb_weighted | 69.2% | 80.8% |
| xgb_ros | 64.4% | 80.8% |
| xgb_smote | 62.5% | 78.8% |
| tabpfn_threshold | 85.6% | 87.5% |
| tabpfn_distpfn | 40.4% | 37.5% |
| tabpfn_distpfn_t | 41.3% | 32.7% |
| tabpfn_downsample | 62.5% | 74.0% |

### Prior Sensitivity Gap

Mean PSG (oracle balanced accuracy minus observed-prior balanced accuracy). Negative means the oracle prior did not beat the observed prior.

| Method | Dir. | pooled | \(c = 0.30\) |
|---|---|---:|---:|
| tabpfn_threshold | A | −0.012 | −0.037 |
| tabpfn_threshold | B | −0.008 | −0.019 |
| tabpfn_distpfn | A | −0.007 | −0.033 |
| tabpfn_distpfn | B | −0.020 | −0.043 |
| tabpfn_distpfn_t | A | −0.008 | −0.036 |
| tabpfn_distpfn_t | B | −0.014 | −0.033 |

### Friedman tests

Blocked by (dataset, \(r\), \(c\)), \(n = 416\). All four are significant.

| Family | Direction | \(\chi^2\) | \(p\) |
|---|---|---:|---:|
| XGBoost | A | 400.0 | \(2.2 \times 10^{-86}\) |
| XGBoost | B | 563.7 | \(7.3 \times 10^{-122}\) |
| TabPFN | A | 681.7 | \(3.1 \times 10^{-146}\) |
| TabPFN | B | 933.0 | \(1.2 \times 10^{-200}\) |

Pairwise Wilcoxon tests, Holm-corrected, are in `results/statistical_tests.csv`.

### Recompute the headline table from the archived log

From the repository root, with pandas and numpy available:

```python
import pandas as pd

raw = pd.read_csv("results/raw_runs.csv")
baseline = {
    "xgb_weighted": "xgb_none", "xgb_ros": "xgb_none", "xgb_smote": "xgb_none",
    "tabpfn_threshold": "tabpfn_none", "tabpfn_distpfn": "tabpfn_none",
    "tabpfn_distpfn_t": "tabpfn_none", "tabpfn_downsample": "tabpfn_none",
}
keys = ["dataset", "r", "c", "direction"]
grouped = (
    raw.groupby(["dataset", "method", "prior_mode", "r", "c", "direction"], dropna=False)
       ["balanced_accuracy"].mean().reset_index()
)
base_xgb = grouped.loc[grouped.method == "xgb_none", keys + ["balanced_accuracy"]]
base_tab = grouped.loc[grouped.method == "tabpfn_none", keys + ["balanced_accuracy"]]
parts = []
for method, base_name in baseline.items():
    base = base_xgb if base_name == "xgb_none" else base_tab
    part = grouped.loc[grouped.method == method].merge(
        base, on=keys, suffixes=("", "_base"))
    part["gain"] = part["balanced_accuracy"] - part["balanced_accuracy_base"]
    parts.append(part)
gains = pd.concat(parts, ignore_index=True)
deployable = gains["prior_mode"].isin(["observed", "n/a"]) | gains["prior_mode"].isna()
noisy = gains[(gains["c"] > 0) & deployable]
summary = noisy.groupby(["method", "direction"])["gain"].agg(
    CHR=lambda s: (s < 0).mean(),
    mean_CG="mean",
    n="size",
)
print(summary.round(6))
```

`n` is 416 for every method and direction. CHR matches `results/chr_table.csv`. For the three prior-dependent TabPFN methods the deployable row is `prior_mode = observed`. The `oracle` rows are diagnostics and stay out of CHR and mean CG.

## Experiment design

Each stress cell is one dataset, one seed, one imbalance ratio, and one contamination setting.

- Imbalance ratios \(r \in \{0.50, 0.20, 0.10, 0.05\}\). \(r\) is the minority share of the controlled training sample. \(r = 0.50\) is the balanced control.
- Contamination rates \(c \in \{0.00, 0.05, 0.10, 0.20, 0.30\}\). \(c = 0\) is clean and is shared by both directions.
- Direction A (`majority_to_minority`) relabels majority training rows as minority. Direction B (`minority_to_majority`) relabels minority training rows as majority.
- An 80/20 stratified split is drawn per seed. The test partition keeps the dataset's natural class ratio and is never corrupted.
- Training size is held fixed across ratios at \(N_{\text{common}}\), capped at 2,622 rows.
- Datasets with more than 100 raw features (SECOM, Arrhythmia, Internet Advertisements, APS Failure) keep a deterministic 100-column subset (`numpy` Generator seed 0).

Seven corrections are scored against two uncorrected baselines:

| Method | What it does |
|---|---|
| `xgb_none` | XGBoost, no imbalance handling |
| `xgb_weighted` | `scale_pos_weight` from the observed class counts |
| `xgb_ros` | random oversampling to balance the observed labels |
| `xgb_smote` | SMOTE, with `k_neighbors` capped at `min(5, n_minority - 1)` |
| `tabpfn_none` | TabPFN, decision threshold 0.5 |
| `tabpfn_threshold` | shift TabPFN's threshold to the class prior |
| `tabpfn_distpfn`, `tabpfn_distpfn_t` | DistPFN and temperature-scaled DistPFN |
| `tabpfn_downsample` | downsample the larger observed class, then fit TabPFN |

XGBoost uses one frozen configuration for every variant: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric=logloss`. TabPFN uses library defaults plus the study seed. `tabpfn_threshold`, `tabpfn_distpfn`, and `tabpfn_distpfn_t` are also scored with the true pre-contamination prior (`prior_mode=oracle`). Those rows are diagnostics.

The 26 datasets are the processed files named in [`data/manifest.json`](data/manifest.json). Each `processed_full` CSV has features, a `target` column (1 = minority), and `target_raw`. The runner reads those files. It does not download from UCI.

## Reproduce the model runs

The archived CSV is the verification artifact. A fresh run is only needed to regenerate `results/raw_runs.csv`.

Recorded environment (manuscript configuration table):

| Package | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.0 |
| XGBoost | 3.4.1 |
| imbalanced-learn | 0.14.2 |
| statsmodels | 0.14.6 |
| TabPFN | 8.3.0 |

The fitting environment is NumPy 2.5.2. `run_with_retry.sh` launches `run_full_matrix.py` inside `.venv_notebook`, and the 15 September 2026 continuation log printed `numpy: 2.5.2` with the other versions in the table above. That virtualenv still has NumPy 2.5.2, installed 16 August 2026, before those September runs. The manuscript configuration table matches this environment.

`tables/table2_configurations.csv` originally listed NumPy 2.3.0. That value came from the Jupyter kernel, Homebrew Python 3.13, which still reports NumPy 2.3.0. The notebook's environment cell on 11 September 2026 printed 2.3.0, and the CSV copied it. Every other row in that CSV already matched the fitting virtualenv. The NumPy row is now 2.5.2 so the file matches the environment that produced the model runs. The surviving shell log begins after 19 datasets were already complete, so it does not by itself print the version used for those earlier datasets. The virtualenv that the runner activates has been NumPy 2.5.2 since before that continuation.

The published run used macOS 26.5.1 (arm64), the TabPFN checkpoint `tabpfn-v3-classifier-v3_default.ckpt` (SHA-256 prefix `d0d865d5`, suffix `8f3988`), and Apple's Metal backend. OpenMP and BLAS were pinned to one thread inside `run_full_matrix.py`, because XGBoost and PyTorch otherwise deadlock in the same process. Scores can shift on CPU or on another GPU. Compare a new run to `results/raw_runs.csv` rather than expecting bitwise equality.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TabPFN needs a Prior Labs license token. Put it in a gitignored `.env` file at the repository root:

```
TABPFN_TOKEN=your_token_here
```

Move the archived log aside before starting, then run from the repository root. If `results/raw_runs.csv` is still in place, the script treats every dataset that already has all ten seeds as finished and skips it.

```bash
mv results/raw_runs.csv results/raw_runs.archived.csv
mv results/failures.csv results/failures.archived.csv
python3 run_full_matrix.py
```

The script checkpoints both CSVs after each dataset, so an interrupted run can be continued by executing the same command again. `run_with_retry.sh` loops that command until all 26 datasets have ten seeds. It changes to its own directory and activates `.venv_notebook` when that environment exists, otherwise `.venv`.

The APS Failure `tabpfn_downsample` cell above fails the same way on a repeat. Leave it as a logged failure. `backfill_missing_cells.py` replays four specific `tabpfn_downsample` cells against an existing log, preserving the shared random stream so the corrupted training set matches the other methods in that cell. The archived `raw_runs.csv` already contains the three cells that completed. The fourth is the SVD failure.

`results/sensitivity_r050/build_sensitivity_r050.py` recomputes CHR, Correction Gain, break-even, PSG, and the Friedman–Wilcoxon tests on three slices of `raw_runs.csv` (all ratios, \(r \neq 0.50\), and \(r = 0.50\) only). It locates the repository from its own file path and writes under `results/sensitivity_r050/` and `figures/sensitivity_r050/`. Run it from any working directory with `python3 results/sensitivity_r050/build_sensitivity_r050.py`.

## Repository layout

| Path | Contents |
|---|---|
| `backup.ipynb` | Full study notebook: pipeline, metrics, and table/figure analysis |
| `run_full_matrix.py` | Notebook matrix loop, extracted for the resumable 26-dataset run |
| `backfill_missing_cells.py` | Replay of four incomplete `tabpfn_downsample` cells |
| `data/manifest.json` | Dataset audit: sizes, class meaning, file paths |
| `data/processed/` | Matrices the runner loads |
| `data/raw/` | Original UCI files kept for traceability |
| `results/raw_runs.csv` | Archived evaluations |
| `results/` | Derived summaries used by the manuscript |
| `tables/` | CSV copies of the paper tables |
| `figures/` | Figure images |
| `paper/` | IEEE manuscript |
| `paper/ACM/` | ACM manuscript, including `main_anonymous.tex` |
| `requirements.txt` | Pinned fitting environment |
| `LICENSE` | MIT for code and manuscript source; UCI data stays CC BY 4.0 |

## License

Code, notebooks, figures, and manuscript source are under the MIT license in [`LICENSE`](LICENSE).

`data/raw/` and `data/processed/` are UCI Machine Learning Repository datasets and tables derived from them. UCI distributes those datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the repository and the dataset entries in `paper/references.bib`. The license badge on each page linked from `data/manifest.json` governs that file.

## TKDD double-anonymous review

ACM TKDD requires double-anonymous review for regular submissions. Author names, affiliations, and funding must not appear in the submission, and a paper that reveals the authors' identity can be rejected. The current guidelines are at <https://dl.acm.org/journal/tkdd/author-guidelines>.

This repository identifies the authors: `paper/main.tex` and `paper/ACM/main.tex` name them, and a public GitHub account does too. Do not put this repository's URL in an anonymized TKDD manuscript or cover letter. Submit `paper/ACM/main_anonymous.tex` for review. If reviewers need the code during review, upload it as anonymous supplementary material or through an anonymized host, and add the public URL after acceptance. Conference-extension papers are the exception stated in the TKDD guidelines.
