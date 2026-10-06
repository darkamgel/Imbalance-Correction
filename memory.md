# Project memory: Imbalance correction paper

Copy of the Claude Code memory saved for this project. The live copies are in `~/.claude/projects/-Users-sanju-Desktop-AI-new-papoer/memory/`.

## Index
- [Imbalance correction paper](#imbalance-correction-paper): design, repo layout, results, status, gotchas
- [Sanjeev K. Khatri](#user-sanjeev-khatri): user is a DSU researcher and corresponding author

---

## Imbalance correction paper

Paper: **"When Imbalance Correction Backfires: Robustness of Tabular Imbalance Mitigation Under Minority-Label Contamination"**. Authors: Sanjeev K. Khatri (corresponding), Zainab Albujasim, Youssef Harrath (Dakota State University).

**Research question:** Are imbalance corrections (resampling, cost-sensitive weighting, TabPFN posterior rescaling) still beneficial when minority labels are corrupted? Two directed label-error mechanisms are tested: Direction A (majority rows mislabeled as minority) and Direction B (minority rows lost to majority).

**Experiment design (from `run_full_matrix.py`):**
- 26 UCI datasets (`data/manifest.json`, processed CSVs in `data/processed/`).
- 10 seeds: 13, 37, 73, 5, 11, 17, 19, 23, 29, 31. The first three are the originals.
- Imbalance ratios r: 0.50, 0.20, 0.10, 0.05. Contamination rates c: 0.00, 0.05, 0.10, 0.20, 0.30.
- Total training size held fixed across r (N_common capped at 2622).
- Seven corrections: TabPFN `threshold`, `distpfn`, `distpfn_t`, `downsample`; XGBoost `weighted`, `ros`, `smote`.
- Metrics: Correction Gain, Correction Harm Rate (CHR), Regret, break-even contamination rate, Prior Sensitivity Gap, plus balanced accuracy, AUPRC, macro-F1, AUROC, Brier, log loss, ECE.
- Stats: dataset-level Friedman tests, pairwise Wilcoxon tests, bootstrap CIs.

**Key results (from the abstract):**
- 112,319 successful evaluations. One further evaluation failed to converge (`linalg.svd`, aps_failure, `results/failures.csv`).
- Every correction underperformed its uncorrected baseline in some condition. CHR ranges from 2.4% to 47.4%, and 37.8% once the r=0.50 control is excluded.
- `tabpfn_threshold` is the most robust method. DistPFN and DistPFN-T have the highest CHR but are the best calibrated.
- Corrections can become non-beneficial at only 5% contamination.

**Repo layout:**
- `paper/main.tex` (IEEEtran), `paper/main_anonymous.tex`, `paper/sections/00..07`, `paper/references.bib`, `paper/ACM/` (ACM version).
- `paper/*.zip` and `ACM_overleaf.zip` are Overleaf packages.
- Experiments: `run_full_matrix.py` (main runner, resume-safe), `run_with_retry.sh` (relaunches until 26 datasets have 10 seeds), `backfill_missing_cells.py`. Venv: `.venv_notebook/`.
- Results: `results/` (`raw_runs.csv`, `summary.csv`, `correction_gains.csv`, `chr_table.csv`, `break_even_table.csv`, `statistical_tests.csv`, `failures.csv`, `sensitivity_r050/`). Paper tables: `tables/`. Figures: `figures/`, `paper/figures/`.
- Explainers: `Project_Explainer.pdf`, `Study_Explainer.pdf`, `results_comparison.pdf`.

**Git history:** IEEE draft complete (3095af2), then ACM paper added (9abbcb6, 2026-09-17), then "updated to ACM format" (209239c, 2026-09-18, HEAD of master).

**Status as of 2026-10-06:** Working tree was clean at session start. Drafted in IEEE and ACM formats. Future work (from the conclusion): cleaning mislabeled rows from TabPFN's in-context prompt, a matched-prior test-set evaluation, other tabular foundation models, multi-class imbalance, instance-dependent or two-direction noise.

**Not verified:** The run log's last lines show mushroom still running, but `raw_runs.csv` already has the 112,319 rows the abstract reports. It is not confirmed that all 26 datasets finished.

**Gotchas:**
- `run_full_matrix.py` expects to run from the project root (`DATA_DIR` and `.env` are resolved from the working directory).
- Threading env vars are forced to 1 (`OMP_NUM_THREADS` etc.) because XGBoost and TabPFN/torch clash. Don't remove them.
- `.env` holds a TABPFN_TOKEN. It is gitignored and must never be copied into memory or shared files.
- The banknote `positive_class` meaning is marked UNVERIFIED in the manifest.

---

## User: Sanjeev K. Khatri

Researcher at The Beacom College of Computer and Cyber Sciences, Dakota State University (Madison, SD). Corresponding author on the imbalance-correction paper, with co-authors Zainab Albujasim and Youssef Harrath.

Works on ML for tabular data in Python (scikit-learn, XGBoost, TabPFN, imbalanced-learn). Writes LaTeX in both IEEE and ACM templates, using Overleaf. Git identity: `sanjeev`.

**How to apply:** Assume an academic research context and a bar of reproducible, reviewer-proof work. Use the DSU institutional address for author details.
