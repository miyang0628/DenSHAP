# DenSHAP

**DenSHAP: Density-Aware Background Dataset Construction for Counterfactual Shapley Explanations**

DenSHAP improves upon CF-SHAP (Albini et al., FAccT '22) by replacing plain distance-based KNN background selection with a LOF-weighted KNN approach that jointly considers proximity and local data density. This yields higher Background Density Scores (BDS) in Hard Cases (sparse regions), providing more realistic explanation contexts.

---

## Method Overview

```
[Phase 1] Compute LOF(x) → α(x) ∈ [0,1]  (difficulty estimation)
[Phase 2] LOF-weighted KNN background selection
          w(x_i) ∝ exp(-d(x,x_i)/σ) × (1/LOF(x_i))
          Filtered by model predictions (Validity guaranteed)
[Phase 3] SHAP(background = LOF-weighted KNN set)
```

**Key difference from CF-SHAP:**

| Method   | KNN selection criterion |
|----------|------------------------|
| CF-SHAP  | Distance only          |
| DenSHAP  | Distance + Density (inverse LOF) |

---

## Installation

```bash
pip install shap xgboost scikit-learn pandas numpy optuna
```

---

## File Structure

```
├── denshap.py          # DenSHAP pipeline (main method)
├── cfshap_baseline.py  # CF-SHAP + 3 baselines reimplemented
├── dhace.py            # LOFDifficultyEstimator, shared utilities
├── data_loader.py      # Dataset loaders (HELOC, LendingClub, Wine)
├── run_experiment.py   # Full experiment script
└── data/
    ├── heloc.csv
    ├── lendingclub.csv
    └── wine.csv
```

---

## Usage

```bash
# Run all datasets
python run_experiment.py

# Single dataset
python run_experiment.py --dataset heloc
python run_experiment.py --dataset wine
python run_experiment.py --dataset lendingclub

# Limit evaluation samples (for speed)
python run_experiment.py --dataset heloc --eval_n 200
```

Results are saved to `results/`:
- `{dataset}_cfshap.csv`   — per-sample CF-SHAP baseline results
- `{dataset}_denshap.csv`  — per-sample DenSHAP results
- `{dataset}_summary.csv`  — method-level aggregated summary
- `{dataset}_group.csv`    — LOF group-level breakdown (Easy/Medium/Hard)

---

## Datasets

| Dataset     | N         | Features | Source  |
|-------------|-----------|----------|---------|
| HELOC       | 9,871     | 23       | FICO    |
| LendingClub | 1,373,324 | 20       | Kaggle  |
| Wine Quality| 6,497     | 11       | UCI     |

Place CSV files in `data/` directory before running.

---

## Configuration

Edit `CONFIG` in `run_experiment.py`:

```python
CONFIG = {
    'eval_n':                 999999,  # evaluation samples (999999 = full test set)
    'optuna_n_trials':        30,      # Optuna tuning trials
    'k_total':                100,     # background dataset size
    'k_lof':                  20,      # LOF neighborhood size
    'p_low':                  25.0,    # α=0 percentile (Easy boundary)
    'p_high':                 75.0,    # α=1 percentile (Hard boundary)
    'n_candidates_multiplier': 5,      # candidate pool = k_total × 5
    'lendingclub_sample_n':   20000,   # LendingClub training sample size
}
```

---

## Baseline Methods

| Method      | Background Dataset         |
|-------------|---------------------------|
| SHAP_TRAIN  | Random sample from all training data |
| SHAP_D_LAB  | Random sample from opposite-label training data |
| SHAP_D_PRED | Random sample from opposite-predicted training data |
| CF_SHAP     | IQR-normalized KNN in opposite-label data |
| **DenSHAP** | **LOF-weighted KNN (distance + density)** |

---

## Novel Metric: BDS (Background Density Score)

$$\text{BDS}(B) = \frac{1}{|B|} \sum_{x_i \in B} \frac{1}{\text{LOF}(x_i)}$$

Higher BDS indicates the background dataset is drawn from denser, more realistic regions of the feature space. DenSHAP achieves consistently higher BDS in Hard Cases across all three datasets.

---

## Citation

```bibtex
@article{denshap2025,
  title   = {DenSHAP: Density-Aware Background Dataset Construction 
             for Counterfactual Shapley Explanations},
  author  = {[Author]},
  year    = {2025}
}
```

---

## Baseline Reference

```bibtex
@inproceedings{Albini2022,
  title     = {Counterfactual Shapley Additive Explanations},
  booktitle = {FAccT '22},
  author    = {Albini, Emanuele and Long, Jason and Dervovic, Danial and Magazzeni, Daniele},
  year      = {2022},
  doi       = {10.1145/3531146.3533168}
}
```
