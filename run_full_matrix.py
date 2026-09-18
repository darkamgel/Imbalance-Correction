
# ===== cell 17 =====

"""
Section 10.1 — Environment setup.

Three things happen here:
1. Import every library the study needs and confirm they're actually installed.
2. Define the constants that stay fixed for the entire study (project paths, the
   3 random seeds, the 4 imbalance ratios, the 5 noise levels, the 2 label-error
   directions) so every later cell references the same frozen values instead of
   re-typing magic numbers.
3. Print the exact package versions in use. This is not decorative — the blueprint
   (Step 4) requires the software environment to be recorded and frozen *before*
   any comparative result is generated, so a later reviewer can reproduce the run.
"""

# CRITICAL: must run before numpy/xgboost/tabpfn(torch) are imported anywhere in this
# kernel. XGBoost and PyTorch (which TabPFN depends on) each bundle their own OpenMP/BLAS
# runtime; running an XGBoost fit followed by a TabPFN fit in the SAME process without this
# would hang indefinitely (or in the reverse import order, segfault) on this machine — a
# real binary conflict between the two libraries' threading runtimes, reproduced and
# confirmed via isolated testing outside this notebook. Forcing every backend to a single
# thread avoids the conflict entirely; for this study's dataset sizes (hundreds to a few
# thousand rows per fit) the performance cost of single-threading is negligible.
import os
for _var in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import xgboost
import imblearn
import tabpfn

# PROJECT_ROOT assumes the notebook's working directory is the project root
# (where the data/ folder lives). DATA_DIR points at the processed CSVs and
# the manifest built in the earlier data-preparation step.
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

# Load TABPFN_TOKEN from a local, untracked .env file rather than hardcoding the
# secret into this notebook — the key never gets written into a file that might
# later be shared/published/committed. os.environ.setdefault() means a token
# already exported in the shell always takes priority over the .env file.
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# --- Frozen experimental design constants (blueprint Section 5.1) ---
SEEDS = [13, 37, 73, 5, 11, 17, 19, 23, 29, 31]   # 10 random seeds -> every stress cell repeated 10x
# Extended from the original 3 (13, 37, 73) to 10 in Session 18, per an external
# reviewer's note that 3 repetitions give limited precision for quantities that
# hinge on a sign crossing zero (Correction Harm Rate, the break-even rate) --
# already flagged as a limitation in Threats to Validity (Section VI-A) before
# this fix. The original 3 seeds are kept first and unchanged, so every already-
# reported result computed from just those 3 remains exactly reproducible; the 7
# new seeds (5, 11, 17, 19, 23, 29, 31, continuing the same "small prime" style
# as 13/37/73) are additive, not a replacement.
R_VALUES = [0.50, 0.20, 0.10, 0.05]                # true minority prevalence, balanced -> severe
MAX_RAW_FEATURES = 100                            # ceiling on raw feature count (Session 18) --
# see the feature-capping note in 10.2. Chosen to comfortably clear the original 15 datasets'
# actual max (spambase, 57 features) while capping internet_advertisements (1,558),
# secom (590), arrhythmia (279), and aps_failure (170), the four datasets whose raw feature
# count is otherwise far outside anything this study's TabPFN timing was ever validated at,
# and plausible contributors to a kernel crash observed in this same session.
N_COMMON_CAP = 2622                               # ceiling on N_common (Session 18, see
# compute_common_training_size's docstring in 10.3): htru2's own N_common under the original
# 10 datasets, kept as the ceiling so the original 10 datasets' N_common values are completely
# unaffected (none of them were ever close to this), while the largest of the 16 datasets added
# in 10.1c (which would otherwise reach N_common ~18,700 for adult) stay within the scale this
# study's TabPFN timing was ever actually measured at.
C_VALUES = [0.00, 0.05, 0.10, 0.20, 0.30]          # directed label-error rate; c=0 is the shared clean case
DIRECTIONS = ["majority_to_minority", "minority_to_majority"]  # the two corruption mechanisms (Direction A / B)

# ENV_INFO is what gets logged/printed for reproducibility. Anyone re-running this
# notebook later can diff their own printed versions against this to see if the
# environment drifted.
ENV_INFO = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "xgboost": xgboost.__version__,
    "imbalanced-learn": imblearn.__version__,
    "statsmodels": statsmodels.__version__,
    "tabpfn": tabpfn.__version__,
    "seeds": SEEDS,
}
for k, v in ENV_INFO.items():
    print(f"{k:18s}: {v}")

if os.environ.get("TABPFN_TOKEN"):
    print(f"{'TABPFN_TOKEN':18s}: set")
else:
    print(f"{'TABPFN_TOKEN':18s}: NOT SET — TabPFN model calls (Phase 6) will fail until the license is accepted")

# Frozen checkpoint identity (blueprint Step 4: "record model/checkpoint identifier
# ... do not change checkpoint after inspecting results"). This reads whatever is
# already cached locally rather than forcing a fresh download/fit here — the first
# real .fit() call in 10.5 is what actually downloads it if it isn't cached yet.
try:
    from tabpfn.model_loading import get_cache_dir
    tabpfn_cache_dir = get_cache_dir()
    ckpt_files = sorted(tabpfn_cache_dir.glob("*.ckpt")) if tabpfn_cache_dir.exists() else []
    if ckpt_files:
        for ckpt in ckpt_files:
            sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            print(f"{'tabpfn checkpoint':18s}: {ckpt.name}")
            print(f"{'checkpoint sha256':18s}: {sha256}")
    else:
        print(f"{'tabpfn checkpoint':18s}: not yet downloaded (fetched on first .fit() call in 10.5)")
except Exception as e:
    print(f"{'tabpfn checkpoint':18s}: could not resolve cache dir ({e})")




# ===== cell 23 =====

"""
Section 10.2 — Dataset loading and audit (blueprint Section 4).

Loads all 5 UCI datasets from the already-prepared data/manifest.json + data/processed/
CSVs (built in the earlier data-preparation pass) rather than re-downloading from UCI
every time the notebook runs. For each dataset this:
  1. Reads the manifest entry, which records the audited row/feature counts, the raw
     class counts *before* any encoding, and — critically — which raw value was chosen
     as the "positive"/minority class and why (Section 4.1: this must come from a cited
     source, never be guessed).
  2. Loads the processed "_full.csv", which already has a clean 0/1 `target` column
     (minority = 1) plus the original `target_raw` value for traceability.
  3. Splits that into X (features only) and y (the encoded target as a numpy array).
  4. Stores both in the DATASETS dict, keyed by a human-readable slug, so later cells
     can do `DATASETS["banknote_authentication"]["X"]` instead of juggling numeric UCI IDs.
"""

with open(DATA_DIR / "manifest.json") as f:
    MANIFEST = json.load(f)

DATASETS = {}
for entry in MANIFEST:
    slug = entry["slug"]
    full = pd.read_csv(PROJECT_ROOT / entry["processed_full"])
    y = full["target"].to_numpy()                       # encoded target: 1 = minority, 0 = majority
    X = full.drop(columns=["target", "target_raw"])      # feature matrix only

    # Session 18 feature cap (see MAX_RAW_FEATURES in 10.1): a deterministic random
    # subsample of columns, not a truncation by column order, so no systematic bias
    # toward whichever features happened to be listed first in the raw source. The
    # seed is fixed (0) so this is exactly reproducible, and which columns were kept
    # is recorded on the manifest entry itself, not just printed and forgotten.
    feature_cap_applied = False
    if X.shape[1] > MAX_RAW_FEATURES:
        kept_cols = np.random.default_rng(0).choice(X.columns, size=MAX_RAW_FEATURES, replace=False)
        kept_cols = sorted(kept_cols.tolist())  # sorted only for a stable, readable column order
        X = X[kept_cols]
        entry["feature_cap_applied"] = True
        entry["feature_cap_kept_columns"] = kept_cols
        feature_cap_applied = True

    DATASETS[slug] = {"X": X, "y": y, "meta": entry}
    cap_note = f"  [CAPPED {entry['n_features']}->{MAX_RAW_FEATURES}]" if feature_cap_applied else ""
    print(f"{slug:28s} n={entry['n_rows']:5d}  features={entry['n_features']:3d}  "
          f"minority={entry['encoded_minority_count']:4d}  majority={entry['encoded_majority_count']:4d}  "
          f"positive={entry['positive_class']['meaning']}{cap_note}")

# South German Credit note: not importable via ucimlrepo — data/raw/south_german_credit/
# was fetched directly from the UCI static zip and parsed from SouthGermanCredit.asc.
# Banknote Authentication note: UCI metadata does not document the 0/1 class mapping;
# class 1 = forged/counterfeit follows community convention, not an official UCI source.




# ===== cell 25 =====

"""
Section 10.3 — Constant training size + controlled imbalance (blueprint Section 5.2).

Why this exists: if you just sampled "20% minority" vs "5% minority" directly from a
fixed pool, the 5% condition would end up with a *smaller total training set* (fewer
minority rows available -> fewer total rows possible), so any performance difference
between imbalance levels could be explained by sample size instead of imbalance itself.
These three functions remove that confound by picking one common total size N_common
that is achievable at every tested r, then sampling exactly N_common rows each time.
"""


def feasible_n(r: float, n_min_available: int, n_maj_available: int) -> float:
    """Largest total sample size achievable at minority ratio r given the two class pools.

    At ratio r, a total size N needs r*N minority rows and (1-r)*N majority rows.
    So N is capped by whichever class runs out first: N <= n_min_available / r
    and N <= n_maj_available / (1-r). This returns the tighter (smaller) of those two caps.

    Args:
        r: target minority prevalence (e.g. 0.20 for 20% minority).
        n_min_available: minority rows available to sample from.
        n_maj_available: majority rows available to sample from.

    Returns:
        The largest N (as a float) for which both class requirements are satisfiable.
    """
    return min(n_min_available / r, n_maj_available / (1.0 - r))


def compute_common_training_size(r_values, n_min_available: int, n_maj_available: int,
                                  max_n: int = None) -> int:
    """The single training size usable across every r in r_values without exceeding
    either class pool at any of them (blueprint Section 5.2).

    Computes feasible_n() at every requested r, then takes the minimum across all of
    them (the tightest constraint wins) and floors it to a whole number of rows.

    Args:
        r_values: list of minority prevalences that will all need to share one N.
        n_min_available: minority rows available (from the training split, pre-imbalance).
        n_maj_available: majority rows available (from the training split, pre-imbalance).
        max_n: optional ceiling on the returned value, on top of the feasibility
            constraint above. None (the default) preserves the original, uncapped
            behavior exactly -- every call site in 10.12/10.13 that generated this
            paper's actual reported results omits this argument, so none of their
            already-computed N_common values change. Added in Session 18: several of
            the 16 datasets added in 10.1c have a much larger minority pool than any
            of the original 10 (e.g. adult: ~18,700 uncapped vs. htru2's 2,622, the
            previous maximum), and a single TabPFN stress cell at that scale measured
            over 10 minutes without completing, vs. ~20s for the previous largest
            (Spambase, N_common=2,347) -- TabPFN's cost does not scale linearly with
            context size. Passing max_n keeps every dataset's actual sampled training
            size within the range this study's TabPFN timing was ever actually
            validated at, at the cost of not fully using the largest new datasets'
            extra rows -- a deliberate, disclosed tradeoff, not a silent one.
    Returns:
        N_common: an integer row count safe to sample at every r in r_values, capped
            at max_n if given.
    """
    feasible = [feasible_n(r, n_min_available, n_maj_available) for r in r_values]
    n_common = math.floor(min(feasible))
    return min(n_common, max_n) if max_n is not None else n_common


def sample_controlled_set(r: float, n_common: int, min_idx: np.ndarray, maj_idx: np.ndarray, rng: np.random.Generator):
    """Draw exactly n_common rows at minority prevalence r from the available index pools.

    Splits n_common into round(r * n_common) minority rows and the remainder majority
    rows, samples each class without replacement, concatenates, and shuffles so the
    two classes aren't sitting in two contiguous blocks.

    Args:
        r: target minority prevalence for this sample.
        n_common: total rows to draw (from compute_common_training_size()).
        min_idx: array of row indices belonging to the minority class.
        maj_idx: array of row indices belonging to the majority class.
        rng: numpy random Generator, so sampling is seeded/reproducible by the caller.

    Returns:
        A shuffled numpy array of n_common row indices meeting the target ratio.

    Raises:
        ValueError: if either class pool doesn't have enough rows to satisfy r at
            this n_common (should not happen if n_common came from
            compute_common_training_size() using the same pools).
    """
    n_min = int(round(r * n_common))
    n_maj = n_common - n_min
    if n_min > len(min_idx) or n_maj > len(maj_idx):
        raise ValueError(
            f"r={r}: need n_min={n_min}/n_maj={n_maj} but pools have "
            f"{len(min_idx)}/{len(maj_idx)} available"
        )
    sel_min = rng.choice(min_idx, size=n_min, replace=False)
    sel_maj = rng.choice(maj_idx, size=n_maj, replace=False)
    selected = np.concatenate([sel_min, sel_maj])
    rng.shuffle(selected)  # avoid a block of all-minority followed by all-majority rows
    return selected




# ===== cell 27 =====

"""
Section 10.4 — Leakage-safe preprocessing & stratified split (Phase 5, blueprint Section 8).

Two leakage risks this guards against:
1. Split leakage: if you preprocess (impute/encode) before splitting train/test, the
   imputer's median or the encoder's category list is influenced by test rows the model
   should never see. stratified_split() always runs first, before any fitting.
2. Fit leakage within a stress condition: fit_preprocessing() must only ever be fit on
   the *current* training condition (after imbalance + label-error injection), and
   apply_preprocessing() is what turns the held-out test set through that same fitted
   transform — never the other way around.
"""

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def stratified_split(X: pd.DataFrame, y: np.ndarray, seed: int, train_fraction: float = 0.80):
    """One stratified 80/20 clean train/test split, created before any controlled
    imbalance or directed label error is introduced (blueprint Step 8).

    "Stratified" means the train and test sets each preserve the original minority/
    majority ratio of the full dataset — otherwise a random split could accidentally
    put too few (or zero) minority rows in the test set.

    Args:
        X: full feature matrix for one dataset.
        y: full encoded target array (aligned row-for-row with X).
        seed: one of the 3 study seeds (13/37/73); the split is deterministic per seed.
        train_fraction: fraction of rows to place in the training partition.

    Returns:
        (train_idx, test_idx): two numpy arrays of row-position indices into X/y.
        test_idx and its labels must never change after this point for this seed —
        every later r/c/direction/method condition reuses the same test rows.
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), train_size=train_fraction, stratify=y, random_state=seed,
    )
    return train_idx, test_idx


def fit_preprocessing(X_train: pd.DataFrame):
    """Fit imputers/encoders on the current training condition only (Section 8, rules 4-5).

    Splits columns into numeric vs categorical, then:
      - numeric columns: fit a median imputer (fills missing values with the training
        column's median — trees/TabPFN don't need scaling, so no scaler is fit).
      - categorical columns: fit a most-frequent-value imputer, then a one-hot encoder
        with handle_unknown="ignore" so a category never seen in training doesn't crash
        or (worse) get quietly learned from test data later.

    Args:
        X_train: the training feature matrix for the *current* stress condition
            (i.e. after controlled imbalance and label-error injection have already
            picked which rows are in play — this must be called fresh for every
            r/c/direction cell, never fit once and reused).

    Returns:
        A dict bundling the fitted transformers and the column-name lists needed to
        apply them consistently later via apply_preprocessing().
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_imputer = SimpleImputer(strategy="median")
    if numeric_cols:
        numeric_imputer.fit(X_train[numeric_cols])

    cat_imputer = None
    encoder = None
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer.fit(X_train[categorical_cols])
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(cat_imputer.transform(X_train[categorical_cols]))

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_imputer": numeric_imputer,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
    }


def apply_preprocessing(X: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Apply an already-fitted preprocessing bundle to any feature matrix (train or test).

    Unknown test-time categories are ignored (encoded as all-zeros for that column's
    one-hot block), never learned from test data — this is what handle_unknown="ignore"
    guarantees at fit time.

    Args:
        X: feature matrix to transform. Can be the same X_train used to fit, or the
            held-out X_test — the transform logic is identical either way, which is
            the whole point (no separate "test path").
        fitted: the dict returned by fit_preprocessing().

    Returns:
        A new DataFrame: imputed numeric columns unchanged in name, categorical
        columns replaced by their one-hot encoded columns.
    """
    parts = []
    if fitted["numeric_cols"]:
        num = fitted["numeric_imputer"].transform(X[fitted["numeric_cols"]])
        parts.append(pd.DataFrame(num, columns=fitted["numeric_cols"], index=X.index))
    if fitted["categorical_cols"]:
        cat_imputed = fitted["cat_imputer"].transform(X[fitted["categorical_cols"]])
        cat_encoded = fitted["encoder"].transform(cat_imputed)
        enc_cols = fitted["encoder"].get_feature_names_out(fitted["categorical_cols"])
        parts.append(pd.DataFrame(cat_encoded, columns=enc_cols, index=X.index))
    return pd.concat(parts, axis=1)




# ===== cell 29 =====

"""
Section 10.5 — Directed label-error functions (blueprint Section 6 / 13.4).

These two functions ARE the study's central manipulation. Everything else in this
notebook exists to measure what happens when one of these runs. They take a set of
clean (trustworthy) labels and corrupt a controlled fraction of them in one of two
directions, then hand back the corrupted labels plus a record of exactly what changed
so the corruption is fully auditable.
"""


def majority_to_minority_contamination(y_clean: np.ndarray, c: float, rng: np.random.Generator):
    """Direction A: flip true-majority (0) rows to minority (1).

    Models the failure mode where majority examples get mislabeled as the rare class
    (e.g. an annotator wrongly flags a normal transaction as fraud). c is defined
    relative to the *observed* minority pool, not the clean one: c is the fraction of
    labels that will *look* minority after corruption but are actually majority.

    Derivation of k (how many majority rows to flip):
        We want:  k / (n_min + k) = c        (k fake-minority rows among the observed pool)
        Solving for k:  k = c * (n_min + k)  ->  k - c*k = c*n_min  ->  k = (c / (1-c)) * n_min

    Args:
        y_clean: original trustworthy 0/1 label array (not modified in place — a copy
            is returned).
        c: target contamination rate in [0, 1). c=0 means no corruption at all.
        rng: numpy random Generator controlling which majority rows get flipped
            (pass a seeded Generator for reproducibility).

    Returns:
        y: the corrupted label array (same length as y_clean).
        flip_idx: array of positions that were flipped (empty array if c=0).
        realized: the *actual* achieved contamination rate k/(n_min+k), which can
            differ slightly from the requested c due to integer rounding of k.

    Raises:
        ValueError: if the requested c would need more majority rows to flip (k) than
            actually exist in y_clean.
    """
    y = y_clean.copy()
    n_min = int((y_clean == 1).sum())
    k = int(round((c / (1.0 - c)) * n_min)) if c > 0 else 0

    src = np.flatnonzero(y_clean == 0)  # candidate majority rows eligible to be flipped
    if k > len(src):
        raise ValueError(f"requested k={k} contamination flips exceeds available majority rows={len(src)}")

    flip_idx = rng.choice(src, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    y[flip_idx] = 1                                   # relabel: majority -> "minority"
    realized = k / max(1, n_min + k)                  # observed contamination rate after rounding
    return y, flip_idx, realized


def minority_to_majority_loss(y_clean: np.ndarray, c: float, rng: np.random.Generator):
    """Direction B: flip true-minority (1) rows to majority (0).

    Models the complementary failure mode where genuine rare-class examples get missed
    during annotation and are recorded as majority instead (e.g. a real fraud case
    slips through review as "normal"). Here c is simpler: it's directly the fraction of
    true minority rows that get lost.

    Derivation of k (how many minority rows to flip):
        We want:  k / n_min = c   (k lost minority rows out of the true minority pool)
        Solving for k:  k = c * n_min   -- no rearranging needed, unlike Direction A.

    Args:
        y_clean: original trustworthy 0/1 label array (not modified in place).
        c: fraction of true minority labels to omit, in [0, 1).
        rng: numpy random Generator controlling which minority rows get flipped.

    Returns:
        y: the corrupted label array (same length as y_clean).
        flip_idx: array of positions that were flipped (empty array if c=0).
        realized: the actual achieved loss rate k/n_min after integer rounding of k.

    Raises:
        ValueError: if the requested c would need more minority rows to flip (k) than
            actually exist in y_clean (cannot happen since k <= n_min by construction,
            but the check documents that invariant explicitly).
    """
    y = y_clean.copy()
    n_min = int((y_clean == 1).sum())
    k = int(round(c * n_min)) if c > 0 else 0

    src = np.flatnonzero(y_clean == 1)  # candidate minority rows eligible to be flipped
    if k > len(src):
        raise ValueError(f"requested k={k} label-loss flips exceeds available minority rows={len(src)}")

    flip_idx = rng.choice(src, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    y[flip_idx] = 0                                   # relabel: minority -> "majority" (i.e. lost)
    realized = k / max(1, n_min)                      # observed loss rate after rounding
    return y, flip_idx, realized




# ===== cell 31 =====

"""
Section 10.6 — Correction methods: TabPFN family (blueprint Section 13.5).

tabpfn is installed (pulled in torch as a dependency). One external, one-time step is
still required before any of these can actually fit a model: TabPFN's pretrained
weights are gated behind a license acceptance on Prior Labs' site (see the note
printed in 10.1 above). Until TABPFN_TOKEN is set, tabpfn_predict_proba() will raise
tabpfn.errors.TabPFNLicenseError with the exact steps needed — that's expected, not a
bug, and the pure-math correction functions below (threshold/DistPFN/DistPFN-T) can
still be defined, unit-tested, and reasoned about independently of that being done.
"""

from tabpfn import TabPFNClassifier


def threshold_correction(p_minority: np.ndarray, tau: float) -> np.ndarray:
    """McDowell-style threshold correction.

    A plain classifier trained on imbalanced data tends to under-predict the rare
    class because 0.5 is the wrong decision boundary when the classes aren't 50/50.
    This shifts the boundary down to the *observed* minority prior tau: predict
    minority whenever the model's probability clears that lower bar, instead of 0.5.

    Args:
        p_minority: TabPFN's P(y=1) for each test row.
        tau: decision threshold. The deployable variant uses tau=pi_obs (the
            training set's observed minority fraction); the oracle diagnostic
            variant (not deployable, for research use only) uses tau=pi_true.

    Returns:
        0/1 predicted labels, one per row of p_minority.
    """
    return (p_minority > tau).astype(int)


def distpfn_adjust(p_minority: np.ndarray, pi_prior: float) -> np.ndarray:
    """DistPFN posterior adjustment -- matches the OFFICIAL implementation exactly
    (github.com/seunghan96/DistPFN, README usage snippet + src/evals/eval_TABPFN_shift_O.py,
    verified by executing their exact code against realistic shapes on 2026-08-18).

    This does NOT match the blueprint's paper-derived description
    (p_tilde(y) = Norm[p_hat(y)^2 / p_train(y)], i.e. squaring each row's own
    prediction) -- that description turned out to be a mistranscription of the
    actual algorithm. The authors' real formula is:

        P_test_avg = mean(p_hat) over the WHOLE test batch for this stress cell
        p_tilde(y) = Norm[p_hat(y) * P_test_avg(y) / p_train(y)]

    i.e. every row is scaled by the SAME batch-level average prediction, not by
    squaring itself. Step by step:
      1. Turn each row's P(y=1) into a full 2-class vector [P(y=0), P(y=1)].
      2. Average those vectors across every row in the batch -> P_test_avg (one
         shared 2-vector, an estimate of the test-time class distribution).
      3. Multiply each row's prediction by P_test_avg (not by itself) and divide by
         the training prior.
      4. Renormalize each row back to a valid probability.

    Args:
        p_minority: TabPFN's raw P(y=1) for each test row in one stress cell's
            fixed test set. P_test_avg is computed across all of these rows, so
            passing a subset would change every row's output, not just the subset's.
        pi_prior: the class-1 prior to correct against. Pass pi_obs (the corrupted,
            observed training prior) for the real deployable method, or pi_true (the
            clean pre-contamination prior) for the oracle diagnostic that isolates how
            much of any failure is caused by prior distortion specifically.

    Returns:
        Adjusted P(y=1) for each row.
    """
    p2 = np.column_stack([1 - p_minority, p_minority])
    P_test_avg = p2.mean(axis=0)  # shared across the whole batch -- the official formula's P_test_avg
    prior = np.array([1 - pi_prior, pi_prior])
    adj = (p2 * P_test_avg) / np.clip(prior, 1e-8, None)
    p_dist = adj / adj.sum(axis=1, keepdims=True)
    return p_dist[:, 1]


def distpfn_t_adjust(p_minority: np.ndarray, pi_prior: float) -> np.ndarray:
    """DistPFN-T -- matches the official algorithm's INTENT, with one deliberate
    correction to a genuine bug in the authors' own published reference driver.

    Their src/evals/eval_TABPFN_shift_O.py calls a generic cross_entropy(p, q) helper
    (written for a batched (N,C) target array reduced with axis=1) on P_test_avg,
    which is only a single shared (C,) vector for the whole test batch -- not one row
    per example. Executing their exact code with realistic shapes raises
    numpy.exceptions.AxisError, which IS a subclass of ValueError and gets silently
    swallowed by the script's own `except ValueError as ve: continue`, meaning their
    published driver cannot produce a single successful DistPFN-T score for binary
    classification as literally written (verified by running their exact
    cross_entropy()/softmax_temperature() functions on 2026-08-18).

    The algorithm's intent is unambiguous from their code once the shape bug is
    corrected: ONE shared tau (a scalar cross-entropy between P_test_avg and the
    prior, not per-row), used to temperature-scale that same shared P_test_avg
    before the DistPFN-style adjustment:

        P_test_avg = mean(p_hat) over the whole test batch
        tau = CE(P_test_avg, p_train)                    -- one scalar, not per-row
        P_test_avg_T = softmax(P_test_avg / tau)          -- one shared 2-vector
        p_tilde(y) = Norm[p_hat(y) * P_test_avg_T(y) / p_train(y)]

    Args:
        p_minority: TabPFN's raw P(y=1) for each test row (same batch-dependence
            caveat as distpfn_adjust() -- P_test_avg/tau are shared across the batch).
        pi_prior: class-1 prior to correct against (pi_obs deployable / pi_true oracle,
            same convention as distpfn_adjust()).

    Returns:
        Adjusted P(y=1) for each row.
    """
    p2 = np.column_stack([1 - p_minority, p_minority])
    P_test_avg = p2.mean(axis=0)
    prior = np.array([1 - pi_prior, pi_prior])

    eps = 1e-12
    tau = -np.sum(P_test_avg * np.log(np.clip(prior, eps, None)))  # single scalar, not per-row
    tau = max(tau, eps)

    scaled = P_test_avg / tau
    scaled = scaled - scaled.max()          # numerically stable softmax
    exp_scaled = np.exp(scaled)
    P_test_avg_T = exp_scaled / exp_scaled.sum()  # one shared, temperature-scaled 2-vector

    adj = (p2 * P_test_avg_T) / np.clip(prior, eps, None)
    p_dist = adj / adj.sum(axis=1, keepdims=True)
    return p_dist[:, 1]


def tabpfn_predict_proba(X_train, y_train, X_test, seed):
    """Fit TabPFN on the current training condition and return raw P(y=1) on the test set.

    This is the one expensive call per stress cell. Per blueprint Step 17, this raw
    output should be cached and reused by threshold_correction()/distpfn_adjust()/
    distpfn_t_adjust() rather than re-fitting TabPFN separately for each post-hoc
    correction variant (observed vs oracle prior) — they all operate on the same p.

    Args:
        X_train: preprocessed training features for the current stress condition.
        y_train: (possibly noisy) training labels for the current stress condition.
        X_test: preprocessed, fixed, clean test features.
        seed: study seed, passed through as TabPFN's random_state for reproducibility.

    Returns:
        1D array of P(y=1) for each row in X_test.

    Raises:
        tabpfn.errors.TabPFNLicenseError: if TABPFN_TOKEN hasn't been set yet
            (see the note in 10.1 / at the top of this cell for the one-time setup).
    """
    model = TabPFNClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def tabpfn_none(X_train, y_train, X_test, seed: int):
    """Uncorrected TabPFN baseline.

    Matches the xgb_*() interface pattern — (X_train, y_train, X_test, seed) ->
    (y_pred, y_prob) — so it drops directly into the same method-comparison loop
    used for the XGBoost family in the smoke test / full matrix runner.
    """
    p = tabpfn_predict_proba(X_train, y_train, X_test, seed)
    y_pred = (p > 0.5).astype(int)
    return y_pred, p


def tabpfn_downsample(X_train, y_train, X_test, seed, rng):
    """TabPFN data-level correction: downsample whichever class has more observed
    rows to match the observed count of the smaller class, then fit TabPFN on the
    balanced subset.

    Identifies the larger and smaller classes from the OBSERVED (possibly noisy)
    labels directly, rather than assuming encoded class 0 stays the numerically
    larger one. An earlier version of this function assumed that and downsampled
    class 0 unconditionally, which crashed under Direction A contamination at
    r=0.50, where relabeling majority rows as minority pushes the observed
    minority count above the observed majority count for every tested c > 0 (an
    algebraic certainty of that corruption mechanism at r=0.50, not a rare edge
    case). Downsampling whichever class the data actually shows as larger removes
    that failure entirely, since sampling fewer rows than exist never fails.

    Args:
        X_train: preprocessed training features (full, imbalanced, current condition).
        y_train: (possibly noisy) training labels, same rows as X_train.
        X_test: preprocessed, fixed, clean test features.
        seed: study seed, passed to TabPFN for reproducibility.
        rng: numpy random Generator controlling which rows of the larger class
            survive downsampling.

    Returns:
        1D array of P(y=1) for each row in X_test, from TabPFN fit on the
        downsampled (class-balanced) training subset.
    """
    class0_idx = np.flatnonzero(y_train == 0)
    class1_idx = np.flatnonzero(y_train == 1)
    if len(class0_idx) >= len(class1_idx):
        larger_idx, smaller_idx = class0_idx, class1_idx
    else:
        larger_idx, smaller_idx = class1_idx, class0_idx
    larger_sample = rng.choice(larger_idx, size=len(smaller_idx), replace=False)
    keep = np.concatenate([smaller_idx, larger_sample])
    return tabpfn_predict_proba(X_train.iloc[keep], y_train[keep], X_test, seed)


def tabpfn_downsample_predict(X_train, y_train, X_test, seed: int):
    """Pipeline-interface wrapper around tabpfn_downsample(), matching the
    (X_train, y_train, X_test, seed) -> (y_pred, y_prob) shape used everywhere else,
    so it can be dropped into the same comparison loop as xgb_*() / tabpfn_none()."""
    rng = np.random.default_rng(seed)
    p = tabpfn_downsample(X_train, y_train, X_test, seed, rng)
    y_pred = (p > 0.5).astype(int)
    return y_pred, p




# ===== cell 33 =====

"""
Section 10.7 — Correction methods: XGBoost family (blueprint Section 13.6).

The classical control group. All four variants share one FROZEN_XGB_CONFIG — no
per-condition hyperparameter tuning — so any performance difference between them is
attributable to the imbalance-correction technique itself, not to one variant
secretly getting a better-tuned model.
"""

from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier

# Frozen once and reused for every XGBoost variant and every stress condition
# (blueprint Section 7.2). Never adjusted after seeing results.
FROZEN_XGB_CONFIG = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)


def _fit_predict(X_train, y_train, X_test, seed: int):
    """Shared fit/predict routine used by every XGBoost variant below, so the
    training and prediction logic only needs to be written (and trusted) once.

    Args:
        X_train: preprocessed training features (already resampled if applicable).
        y_train: training labels matching X_train.
        X_test: preprocessed, fixed test features.
        seed: study seed, used as XGBoost's random_state.

    Returns:
        (y_pred, y_prob): 0/1 predictions at the default 0.5 threshold, and the
        raw P(y=1) probabilities.
    """
    model = XGBClassifier(**FROZEN_XGB_CONFIG, random_state=seed)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred, y_prob


def xgb_none(X_train, y_train, X_test, seed: int):
    """Uncorrected XGBoost baseline — no imbalance handling at all. This is the
    reference every correction_gain() calculation below is measured against."""
    return _fit_predict(X_train, y_train, X_test, seed)


def xgb_weighted(X_train, y_train, X_test, seed: int):
    """Cost-sensitive correction: upweight minority-class errors during training via
    XGBoost's scale_pos_weight, set from the *observed* (possibly noisy) class counts.

    Because this trusts the observed counts rather than the true clean counts, it's
    directly exposed to whatever the current label-error direction did to the prior —
    that sensitivity is exactly what the study wants to measure.

    Args:
        X_train, y_train, X_test, seed: same as xgb_none().

    Returns:
        (y_pred, y_prob), same shape/meaning as _fit_predict().
    """
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    ratio_obs = n_neg / n_pos  # how many majority rows per minority row -> weight applied to minority-class loss
    model = XGBClassifier(**FROZEN_XGB_CONFIG, random_state=seed, scale_pos_weight=ratio_obs)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred, y_prob


def xgb_ros(X_train, y_train, X_test, seed: int):
    """Random oversampling correction: duplicate observed minority-labeled rows until
    the training set is class-balanced, then train normally.

    Under Direction A this can duplicate *false* minority rows (amplifying the
    contamination); under Direction B it duplicates from an already-shrunken genuine
    minority pool. Both are exactly the failure modes the study is probing.
    """
    X_res, y_res = RandomOverSampler(random_state=seed).fit_resample(X_train, y_train)
    return _fit_predict(X_res, y_res, X_test, seed)


def xgb_smote(X_train, y_train, X_test, seed: int):
    """SMOTE correction: synthesize new minority-class rows by interpolating between
    real minority rows' feature vectors (training data only, never touches test rows).

    k_neighbors is capped at min(5, n_min-1) because SMOTE needs at least k_neighbors
    real minority rows to interpolate between; with very few minority rows (extreme
    r + heavy Direction B loss), the default k=5 would error, so this shrinks k
    deterministically instead of silently failing. If even k=1 isn't possible
    (n_min<2), the ValueError below should be caught by the caller and the cell
    marked failed per blueprint Step 26, rather than the design being changed ad hoc.
    """
    n_min = int((y_train == 1).sum())
    if n_min < 2:
        raise ValueError(f"SMOTE infeasible: only {n_min} minority-labeled training rows")
    k_neighbors = min(5, n_min - 1)
    X_res, y_res = SMOTE(random_state=seed, k_neighbors=k_neighbors).fit_resample(X_train, y_train)
    return _fit_predict(X_res, y_res, X_test, seed)




# ===== cell 35 =====

"""
Section 10.8 — Metrics module (blueprint Section 9).

One function per metric family, plus compute_all_metrics() that runs every metric
in a single call so every method/condition in the study gets scored identically.
Metrics split into three groups: discrimination (does it rank/classify correctly),
a diagnostic, and reliability (are the *probabilities* trustworthy, not just the
final yes/no call) — reliability metrics matter because H5 predicts that correction
harm often shows up in calibration before it shows up in accuracy.
"""

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def worst_class_recall(y_true, y_pred) -> float:
    """The lower of the two per-class recalls.

    Balanced accuracy can look fine even if one class's recall has collapsed to near
    zero, as long as the other class is doing very well — this metric exists
    specifically to catch that collapse, which balanced accuracy can hide.

    Args:
        y_true: true 0/1 labels for the test set.
        y_pred: predicted 0/1 labels for the test set.

    Returns:
        min(recall_class_0, recall_class_1) as a single float.
    """
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return float(np.min(recalls))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: how far predicted probabilities are from actual observed frequencies.

    Splits predictions into n_bins equal-width buckets by predicted probability
    (e.g. all rows with P(y=1) in [0.7, 0.8)), then for each bucket compares the
    average predicted probability to the actual fraction of positives in that bucket.
    A well-calibrated model's "70% confident" predictions should be right about 70%
    of the time — ECE is the size-weighted average gap between confidence and
    reality across all buckets.

    Args:
        y_true: true 0/1 labels for the test set.
        y_prob: predicted P(y=1) for the test set.
        n_bins: number of equal-width probability buckets (documented per blueprint
            Section 9 — must be fixed and reported, not tuned).

    Returns:
        ECE as a single float in [0, 1]; 0 means perfectly calibrated.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        # size-weighted gap between "how confident" (mean predicted prob) and
        # "how often actually right" (mean true label) within this bucket
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def compute_all_metrics(y_true, y_pred, y_prob, ece_bins: int = 10) -> dict:
    """Run every study metric in one call so every method/condition is scored identically.

    Args:
        y_true: true 0/1 labels for the (fixed, clean) test set.
        y_pred: predicted 0/1 labels from the method being evaluated.
        y_prob: predicted P(y=1) from the method being evaluated.
        ece_bins: bin count passed through to expected_calibration_error().

    Returns:
        A dict with keys: balanced_accuracy, worst_class_recall, auprc, macro_f1,
        minority_precision, minority_recall, auroc, brier, log_loss, ece.
        auroc is NaN in the (should-never-happen-for-this-study) edge case where
        y_true only contains one class, since AUROC is undefined there.
    """
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "worst_class_recall": worst_class_recall(y_true, y_pred),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "minority_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "minority_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=ece_bins),
    }




# ===== cell 37 =====

"""
Section 10.9 — Contribution metrics: gain / harm / break-even / PSG (blueprint Section 10).

Where 10.7's metrics score one method on one condition, these functions compare a
*corrected* method against its matching *uncorrected* baseline on the exact same
dataset/seed/r/c/direction cell. This comparison — not raw accuracy — is the study's
actual contribution: it answers "did the correction help, and by how much" rather
than just "how good was this one number."
"""


def correction_gain(score_corrected: float, score_uncorrected: float, lower_is_better: bool = False) -> float:
    """CG_m(r,c,d) = S_corrected - S_uncorrected.

    Positive CG always means "the correction helped," by construction. For a metric
    where higher is naturally better (balanced accuracy, AUPRC), that's just the raw
    difference. For a loss-type metric where *lower* is better (Brier score, log loss),
    the sign is flipped so the convention stays consistent everywhere else that reads CG.

    Args:
        score_corrected: the metric value for the corrected method.
        score_uncorrected: the metric value for the matching uncorrected baseline,
            same dataset/seed/r/c/direction cell.
        lower_is_better: True for loss-type metrics (Brier, log loss); False for
            metrics where higher is better (balanced accuracy, AUPRC, etc.).

    Returns:
        The gain, sign-adjusted so positive always means improvement.
    """
    gain = score_corrected - score_uncorrected
    return -gain if lower_is_better else gain


def correction_harm_rate(gains, epsilon: float = 0.0) -> float:
    """CHR_m(d) = fraction of evaluated cells where the correction made things worse.

    Counts how often a method's gain across many stress cells falls below -epsilon,
    then divides by the total number of cells. epsilon=0 is the strict "any harm
    counts" definition used as the headline number; a small positive epsilon (e.g.
    0.01 balanced-accuracy points) can be used as a secondary "meaningful harm only"
    sensitivity check — always state which one is being reported.

    Args:
        gains: array-like of correction_gain() values across many stress cells for
            one method/direction.
        epsilon: harm threshold; a gain below -epsilon counts as "harmful."

    Returns:
        Fraction of cells counted as harmful, in [0, 1].
    """
    gains = np.asarray(gains)
    return float((gains < -epsilon).mean())


def correction_regret(score_uncorrected: float, score_corrected: float, lower_is_better: bool = False) -> float:
    """Regret = how badly a correction hurt, when it did hurt (0 if it didn't).

    CHR answers "how often does this correction backfire"; regret answers "how bad is
    it when it does." A method with low CHR but huge regret on the rare occasions it
    fails is a very different practical risk than one with high CHR but tiny regret.

    Args:
        score_uncorrected: baseline metric value for this cell.
        score_corrected: corrected-method metric value for the same cell.
        lower_is_better: same convention as correction_gain().

    Returns:
        max(0, -gain): zero if the correction helped or was neutral, otherwise the
        magnitude of the harm.
    """
    gain = correction_gain(score_corrected, score_uncorrected, lower_is_better)
    return max(0.0, -gain)


def break_even_point(c_values, gains_by_c):
    """c*_m(r,d): the smallest nonzero noise level c at which a method's gain first
    drops to zero or below — i.e. the point past which the correction stops being
    worth using at all.

    Args:
        c_values: the noise levels tested, e.g. [0.00, 0.05, 0.10, 0.20, 0.30].
        gains_by_c: correction_gain() values in the same order as c_values (already
            averaged across seeds for this dataset/method/r/direction, per blueprint
            Section 10.3, so a single noisy seed can't manufacture a false break-even).

    Returns:
        The break-even c value (a float), or the string ">{max(c_values)}" if the
        correction never crossed zero within the tested range — the blueprint
        explicitly forbids extrapolating past what was actually tested.
    """
    for c, gain in zip(c_values, gains_by_c):
        if c > 0 and gain <= 0:
            return c
    return f">{max(c_values):.2f}"


def prior_sensitivity_gap(score_oracle_prior: float, score_observed_prior: float) -> float:
    """PSG = S_oracle_prior - S_observed_prior.

    For prior-dependent corrections (threshold, DistPFN, DistPFN-T), this compares
    the real deployable version (which trusts the corrupted observed prior pi_obs)
    against a diagnostic-only oracle version (which is handed the true clean prior
    pi_true — information not available in real deployment). A large positive PSG
    means most of the correction's failure comes specifically from the prior being
    wrong; a small PSG despite poor overall performance means the failure is coming
    from somewhere else (e.g. the underlying model degrading), not the prior.

    Args:
        score_oracle_prior: metric value using the oracle (true, pre-contamination) prior.
        score_observed_prior: metric value using the corrupted observed prior.

    Returns:
        PSG as a float; positive implicates prior distortion in the failure.
    """
    return score_oracle_prior - score_observed_prior




# ===== cell 43 =====

def run_one_stress_cell(X_train_full, y_train_full, X_test, y_test, dataset_slug, seed, r, c, direction,
                         n_common, min_idx, maj_idx, rng):
    """Runs every core method (+ oracle diagnostics for prior-dependent TabPFN corrections)
    on one dataset/seed/r/c/direction stress cell.

    Reuses sample_controlled_set, majority_to_minority_contamination/minority_to_majority_loss,
    fit_preprocessing/apply_preprocessing, xgb_*/tabpfn_* pipelines, and compute_all_metrics —
    all defined in earlier cells (10.3-10.8). This function only adds the orchestration:
    sample -> corrupt -> preprocess -> run every method -> log result or log failure.

    Returns:
        (result_rows, failure_rows): two lists of dicts, appended to the pilot-wide
        accumulators by the caller. A method failing never stops the other methods in
        the same cell, and never stops later cells (blueprint Step 35: record failures,
        do not erase them).
    """
    result_rows, failure_rows = [], []

    sel = sample_controlled_set(r, n_common, min_idx, maj_idx, rng)
    X_ctrl = X_train_full.loc[sel].reset_index(drop=True)
    y_ctrl_clean = y_train_full[sel]
    pi_true = y_ctrl_clean.mean()  # true pre-contamination prior, used only by oracle diagnostics

    if c == 0.0:
        y_noisy, realized, direction_label = y_ctrl_clean.copy(), 0.0, "clean"
    else:
        noise_fn = majority_to_minority_contamination if direction == "majority_to_minority" else minority_to_majority_loss
        y_noisy, _, realized = noise_fn(y_ctrl_clean, c, rng)
        direction_label = direction

    pi_obs = y_noisy.mean()
    fitted = fit_preprocessing(X_ctrl)
    X_ctrl_proc = apply_preprocessing(X_ctrl, fitted)
    X_test_proc = apply_preprocessing(X_test, fitted)

    def log_result(method, prior_mode, y_pred, y_prob):
        m = compute_all_metrics(y_test, y_pred, y_prob)
        result_rows.append(dict(dataset=dataset_slug, seed=seed, r=r, c=c, direction=direction_label,
                                 realized_c=realized, method=method, prior_mode=prior_mode, **m))

    def log_failure(method, error):
        failure_rows.append(dict(dataset=dataset_slug, seed=seed, r=r, c=c, direction=direction_label,
                                  method=method, error=str(error)))

    for name, fn in [("xgb_none", xgb_none), ("xgb_weighted", xgb_weighted),
                      ("xgb_ros", xgb_ros), ("xgb_smote", xgb_smote)]:
        try:
            y_pred, y_prob = fn(X_ctrl_proc, y_noisy, X_test_proc, seed)
            log_result(name, "n/a", y_pred, y_prob)
        except Exception as e:
            log_failure(name, e)

    try:
        p_raw = tabpfn_predict_proba(X_ctrl_proc, y_noisy, X_test_proc, seed)
        log_result("tabpfn_none", "n/a", (p_raw > 0.5).astype(int), p_raw)
        log_result("tabpfn_threshold", "observed", threshold_correction(p_raw, pi_obs), p_raw)
        log_result("tabpfn_threshold", "oracle", threshold_correction(p_raw, pi_true), p_raw)
        p_do = distpfn_adjust(p_raw, pi_obs)
        log_result("tabpfn_distpfn", "observed", (p_do > 0.5).astype(int), p_do)
        p_dor = distpfn_adjust(p_raw, pi_true)
        log_result("tabpfn_distpfn", "oracle", (p_dor > 0.5).astype(int), p_dor)
        p_to = distpfn_t_adjust(p_raw, pi_obs)
        log_result("tabpfn_distpfn_t", "observed", (p_to > 0.5).astype(int), p_to)
        p_tor = distpfn_t_adjust(p_raw, pi_true)
        log_result("tabpfn_distpfn_t", "oracle", (p_tor > 0.5).astype(int), p_tor)
    except Exception as e:
        log_failure("tabpfn_family", e)

    try:
        # tabpfn_downsample (10.6) identifies which class is observed-larger
        # dynamically, so it no longer fails when Direction A contamination at
        # r=0.50 pushes the observed minority count above the observed majority
        # count, it just downsamples whichever one is larger. An earlier,
        # fixed-class-assignment version of this function did fail on exactly
        # that case (120 stress cells, every dataset, every nonzero c, every
        # seed, at r=0.50 under Direction A); those were recovered by re-running
        # just those cells with the fixed function, replaying the shared rng
        # identically so the recovered result uses the same corrupted training
        # set every other method in that cell already used (10.13d).
        y_pred_ds, p_ds = tabpfn_downsample_predict(X_ctrl_proc, y_noisy, X_test_proc, seed)
        log_result("tabpfn_downsample", "n/a", y_pred_ds, p_ds)
    except Exception as e:
        log_failure("tabpfn_downsample", e)

    return result_rows, failure_rows




# ===== cell 46 =====

# Full core matrix runner (blueprint Step 34). Loops over every dataset in DATASETS,
# reusing run_one_stress_cell() from 10.12 without modification. Checkpoints after
# each dataset completes so partial progress always survives an interruption.
#
# Resume-skip guard (Session 18): if results/raw_runs.csv already exists (e.g. from a
# kernel that died partway through, as happened in this session), its rows are loaded
# first and any dataset that already has a complete row for every seed in SEEDS is
# skipped entirely rather than recomputed. A dataset counts as "complete" only if
# every seed in SEEDS is present for it -- a dataset that was only partially through
# its seeds when the kernel died does NOT count as complete, and is redone from
# scratch for every seed (simpler and safer than trying to resume mid-dataset, and
# cheap relative to the per-dataset cost this guard exists to avoid repeating).

import time

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

all_results, all_failures = [], []
already_complete_datasets = set()

_existing_path = RESULTS_DIR / "raw_runs.csv"
if _existing_path.exists():
    _existing = pd.read_csv(_existing_path)
    all_results = _existing.to_dict("records")
    _seeds_needed = set(SEEDS)
    for _slug, _grp in _existing.groupby("dataset"):
        if set(_grp["seed"].unique()) >= _seeds_needed:
            already_complete_datasets.add(_slug)
    print(f"resume-skip guard: loaded {len(all_results)} existing rows from {_existing_path.name}, "
          f"{len(already_complete_datasets)} dataset(s) already complete for all {len(SEEDS)} seeds: "
          f"{sorted(already_complete_datasets)}")

_existing_failures_path = RESULTS_DIR / "failures.csv"
if _existing_failures_path.exists():
    try:
        _existing_f = pd.read_csv(_existing_failures_path)
        if len(_existing_f):
            all_failures = _existing_f.to_dict("records")
    except pd.errors.EmptyDataError:
        # A 0-failure run writes pd.DataFrame([]).to_csv(...), which is a file
        # with no header and no rows -- pandas can't parse that as a table, but
        # it's not an error condition here, it just means nothing has failed yet.
        pass

dataset_timings = {}
t_matrix_start = time.time()

n_cells_total_per_dataset = 36
for dataset_slug in DATASETS.keys():
    if dataset_slug in already_complete_datasets:
        print(f"[{dataset_slug}] already complete for all {len(SEEDS)} seeds -- skipping "
              f"({time.time()-t_matrix_start:.0f}s elapsed total)", flush=True)
        continue

    t_dataset_start = time.time()
    X_full = DATASETS[dataset_slug]["X"].reset_index(drop=True)
    y_full = DATASETS[dataset_slug]["y"]

    dataset_results, dataset_failures = [], []
    cells_done_this_dataset = 0

    for seed in SEEDS:
        train_idx, test_idx = stratified_split(X_full, y_full, seed=seed)
        X_train_full = X_full.loc[train_idx].reset_index(drop=True)
        y_train_full = y_full[train_idx]
        X_test = X_full.loc[test_idx].reset_index(drop=True)
        y_test = y_full[test_idx]

        min_idx = np.flatnonzero(y_train_full == 1)
        maj_idx = np.flatnonzero(y_train_full == 0)
        n_common = compute_common_training_size(R_VALUES, len(min_idx), len(maj_idx), max_n=N_COMMON_CAP)
        rng = np.random.default_rng(seed)

        for r in R_VALUES:
            t_cell = time.time()
            rows, fails = run_one_stress_cell(X_train_full, y_train_full, X_test, y_test, dataset_slug,
                                               seed, r, 0.0, "clean", n_common, min_idx, maj_idx, rng)
            dataset_results.extend(rows); dataset_failures.extend(fails)
            cells_done_this_dataset += 1
            print(f"  [{dataset_slug}] seed={seed} r={r} c=0.0 clean -- cell {cells_done_this_dataset}/"
                  f"{n_cells_total_per_dataset} done in {time.time()-t_cell:.1f}s, "
                  f"{time.time()-t_matrix_start:.0f}s total elapsed", flush=True)

            for c in C_VALUES[1:]:
                for direction in DIRECTIONS:
                    t_cell = time.time()
                    rows, fails = run_one_stress_cell(X_train_full, y_train_full, X_test, y_test, dataset_slug,
                                                       seed, r, c, direction, n_common, min_idx, maj_idx, rng)
                    dataset_results.extend(rows); dataset_failures.extend(fails)
                    cells_done_this_dataset += 1
                    print(f"  [{dataset_slug}] seed={seed} r={r} c={c} dir={direction} -- cell "
                          f"{cells_done_this_dataset}/{n_cells_total_per_dataset} done in "
                          f"{time.time()-t_cell:.1f}s, {time.time()-t_matrix_start:.0f}s total elapsed", flush=True)

    all_results.extend(dataset_results)
    all_failures.extend(dataset_failures)
    dataset_timings[dataset_slug] = time.time() - t_dataset_start

    # Checkpoint after every dataset -- protects the run against losing everything
    # already computed if a later dataset crashes or the kernel is interrupted.
    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "raw_runs.csv", index=False)
    pd.DataFrame(all_failures).to_csv(RESULTS_DIR / "failures.csv", index=False)

    print(f"[{dataset_slug}] done in {dataset_timings[dataset_slug]:.0f}s "
          f"({len(dataset_results)} rows, {len(dataset_failures)} failures) "
          f"-- checkpoint saved, {time.time()-t_matrix_start:.0f}s elapsed total", flush=True)

total_time = time.time() - t_matrix_start
raw_runs = pd.DataFrame(all_results)
failures = pd.DataFrame(all_failures)

print(f"\nFULL CORE MATRIX COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"total: {len(raw_runs)} result rows, {len(failures)} failures across {len(DATASETS)} datasets")
print("\nper-dataset timing:")
for slug, t in dataset_timings.items():
    print(f"  {slug:28s} {t:7.0f}s")
if len(failures):
    print("\nfailure breakdown by dataset x method:")
    print(failures.groupby(["dataset", "method"]).size())


