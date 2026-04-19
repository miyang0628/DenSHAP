# DenSHAP

**DenSHAP: Density-Aware Background Dataset Construction for Counterfactual Shapley Explanations**

DenSHAP improves upon CF-SHAP (Albini et al., FAccT '22) by replacing plain distance-based KNN
background selection with a LOF-weighted KNN approach that jointly considers proximity and local
data density. This yields higher Background Density Scores (BDS) in Hard Cases (query instances
in sparse feature-space regions), providing more representative explanation contexts while
maintaining Validity at 100%.

---

## Method Overview

```
[Phase 1] Compute LOF(x) → α(x) ∈ [0,1]   (difficulty estimation)
[Phase 2] LOF-weighted KNN background selection
          w(x_i) ∝ exp(-d(x,x_i)/σ) × (1/LOF(x_i))
          Filtered by model predictions   (Validity structurally guaranteed)
[Phase 3] SHAP(background = LOF-weighted KNN set)
```

**Key difference from CF-SHAP:**

| Method | KNN selection criterion |
|--------|------------------------|
| CF-SHAP | Distance only |
| DenSHAP | Distance + Density (inverse LOF) |

**Difficulty score:**  α(x) = clip((LOF(x) − Q₂₅) / (Q₇₅ − Q₂₅), 0, 1)
- Easy: α ≤ 0.33 · Medium: 0.33 < α ≤ 0.67 · Hard: α > 0.67

---

## Installation

```bash
pip install shap xgboost scikit-learn pandas numpy optuna
```

---

## File Structure

```
├── denshap.py                  # DenSHAP pipeline (main method)
├── cfshap_baseline.py          # CF-SHAP + 3 baselines reimplemented
├── dhace.py                    # LOFDifficultyEstimator, shared utilities
├── data_loader.py              # Dataset loaders (HELOC, LendingClub, Wine)
├── run_experiment.py           # Full experiment script (Optuna + k_lof sensitivity)
├── DenSHAP_demo.ipynb          # Results viewer (loads results/ CSVs)
├── DenSHAP_figures_en.ipynb    # Publication figures (Fig 1–6, 600 DPI)
├── DenSHAP_sensitivity.ipynb   # σ sensitivity & appendix analysis
├── data/
│   ├── heloc.csv
│   ├── lendingclub.csv
│   └── wine.csv
├── results/                    # Auto-generated after run_experiment.py
│   ├── {dataset}_cfshap.csv
│   ├── {dataset}_denshap.csv
│   ├── {dataset}_summary.csv
│   ├── {dataset}_group.csv
│   ├── {dataset}_bds_by_group.csv
│   ├── {dataset}_model.joblib        # cached XGBoost model
│   └── {dataset}_best_params.joblib  # cached Optuna best params
└── figures/                    # Auto-generated after figure notebooks
    ├── fig1_overall_ca_bds.png
    ├── fig2_bds_by_group.png
    ├── fig3_lof_distribution.png
    ├── fig4_bds_delta_violin.png
    ├── fig5_alpha_vs_delta_bds.png
    ├── fig6_plausibility.png
    ├── fig_appendix_A_gamma_sensitivity.png
    └── fig_appendix_B_lambda_interpolation.png
```

---

## Usage

```bash
# Run all datasets (full test set, k_lof=20)
python run_experiment.py

# Single dataset
python run_experiment.py --dataset heloc
python run_experiment.py --dataset wine
python run_experiment.py --dataset lendingclub

# Limit evaluation samples (for speed)
python run_experiment.py --dataset heloc --eval_n 200

# k_LOF sensitivity analysis (models cached — fast)
python run_experiment.py --k_lof 10 --output results_klof10
python run_experiment.py --k_lof 30 --output results_klof30
python run_experiment.py --k_lof 50 --output results_klof50
```

After the experiment, generate figures by running the notebooks in order:

```
1. DenSHAP_demo.ipynb           → Load and inspect results/ CSVs
2. DenSHAP_figures_en.ipynb     → Fig 1–6 (main paper)
3. DenSHAP_sensitivity.ipynb    → Appendix figures & tables
```

---

## Datasets

| Dataset | N (train sample) | Features | Source |
|---------|-----------------|----------|--------|
| HELOC | 9,871 | 23 | FICO |
| LendingClub | 20,000 (stratified from 1,373,324) | 20 | Kaggle |
| Wine Quality | 6,497 | 11 | UCI |

Place CSV files in `data/` before running.

---

## Configuration

Key parameters in `run_experiment.py` CONFIG (all overridable via CLI):

```python
CONFIG = {
    'eval_n'                 : 999999, # evaluation samples (999999 = full test set)
    'random_state'           : 42,     # global random seed
    'optuna_n_trials'        : 30,     # Optuna tuning trials per dataset
    'optuna_cv'              : 3,      # stratified CV folds for Optuna
    'k_total'                : 100,    # background dataset size
    'k_lof'                  : 20,     # LOF neighbourhood size (default)
    'p_low'                  : 25.0,   # α=0 percentile (Easy boundary)
    'p_high'                 : 75.0,   # α=1 percentile (Hard boundary)
    'n_candidates_multiplier': 5,      # candidate pool = k_total × 5
    'lendingclub_sample_n'   : 20000,  # LendingClub stratified training sample
}
```

**Practical guideline for k_lof:**
- Start with `k_lof=20` (default).
- Compute the LOF IQR of your dataset: if IQR < 0.1, density signal is weak and gains will be modest; if IQR ≥ 0.1, meaningful BDS improvements are expected.
- For training sets > ~100,000 instances, consider approximate LOF or parallelised implementations.

---

## Software Environment

| Library | Version |
|---------|---------|
| Python | 3.10.15 |
| scikit-learn | 1.5.2 |
| XGBoost | 1.7.5 |
| SHAP | 0.46.0 |
| Optuna | 4.5.0 |
| pandas | 2.3.3 |
| NumPy | 1.26.4 |

Hardware: Intel Core (Family 6, Model 191), 127.7 GB RAM, no GPU.
Total wall-clock time for main experiment + 4× k_lof sensitivity runs: ~18 hours.

---

## Baseline Methods

| Method | Background Dataset |
|--------|-------------------|
| SHAP_TRAIN | Random sample from all training data |
| SHAP_D_LAB | Random sample from opposite-label training data |
| SHAP_D_PRED | Random sample from opposite-predicted training data |
| CF_SHAP | IQR-normalised KNN from opposite-predicted data |
| **DenSHAP** | **LOF-weighted KNN (distance + density)** |

---

## Novel Metric: BDS (Background Density Score)

$$\text{BDS}(B) = \frac{1}{|B|} \sum_{x_i \in B} \frac{1}{\text{LOF}(x_i)}$$

Higher BDS indicates the background dataset is drawn from denser, more representative regions
of the feature space. DenSHAP achieves the highest BDS across all three datasets and all
difficulty groups, with statistically significant improvement in Hard Cases
(all p < 0.001, one-sided Wilcoxon signed-rank test).

---

## Key Results (k_lof = 20, Optuna-tuned XGBoost)

### Hard-group BDS (primary claim)

| Dataset | N | CF-SHAP BDS | DenSHAP BDS | Δ | p-value |
|---------|---|-------------|-------------|---|---------|
| HELOC | 708 | 0.9385 | **0.9641** | +2.72% | 4.19×10⁻⁵² |
| Wine Quality | 318† | 0.9633 | **0.9767** | +1.39% | 4.74×10⁻⁷ |
| LendingClub | 1595 | 0.9651 | **0.9755** | +1.08% | 2.13×10⁻³³ |

† 3 instances with undefined BDS excluded (N_valid = 318 of 321).

Validity is guaranteed at 100% across all datasets via model-prediction-based filtering.

### Model performance (Optuna-tuned XGBoost)

| Dataset | Test Accuracy | Test AUC |
|---------|--------------|----------|
| HELOC | 0.7241 | 0.8013 |
| Wine Quality | 0.8846 | 0.9111 |
| LendingClub | 0.6520 | 0.7207 |

### k_LOF Sensitivity (Hard-group BDS, DenSHAP vs CF-SHAP)

| k_lof | HELOC Δ | Wine Δ | LendingClub Δ |
|-------|---------|--------|---------------|
| 10 | +2.56% | +1.18% | +0.72% |
| **20** | **+2.72%** | **+1.39%** | **+1.08%** |
| 30 | +3.10% | +1.52% | +1.39% |
| 50 | +3.23% | +1.89% | +1.98% |

DenSHAP consistently outperforms CF-SHAP across all k_lof values and datasets.

---

## Reproducibility

```bash
# Reproduce from scratch (deletes cached models)
rm results/*.joblib
python run_experiment.py

# Re-use cached models (skip Optuna, faster)
python run_experiment.py   # models already in results/
```

Random seed: 42 (applied globally to all stochastic components).

---

## Citation

```bibtex
@article{denshap2026,
  title  = {DenSHAP: Density-Aware Background Dataset Construction
            for Counterfactual Shapley Explanations},
  author = {[Author]},
  year   = {2026}
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
