# DenSHAP

**DenSHAP: Density-Aware Background Dataset Construction for Counterfactual Shapley Explanations**

DenSHAP improves upon CF-SHAP (Albini et al., FAccT '22) by replacing plain distance-based KNN
background selection with a LOF-weighted KNN approach that jointly considers proximity and local
data density. This yields higher Background Density Scores (BDS) in Hard Cases (sparse regions),
providing more realistic explanation contexts while maintaining Validity at 100%.

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
├── run_experiment.py           # Full experiment script (Optuna included)
├── DenSHAP_demo.ipynb          # Reproducible demo notebook
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
│   └── {dataset}_bds_by_group.csv
└── figures/                    # Auto-generated after figure notebooks
    ├── fig1_overall_ca_bds.pdf / .png
    ├── fig2_bds_by_group.pdf / .png
    ├── fig3_lof_distribution.pdf / .png
    ├── fig4_bds_delta_violin.pdf / .png
    ├── fig5_alpha_vs_delta_bds.pdf / .png
    ├── fig6_plausibility.pdf / .png
    ├── fig_appendix_A_gamma_sensitivity.pdf / .png
    └── fig_appendix_B_lambda_interpolation.pdf / .png
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

After the experiment, generate figures by running the notebooks in order:

```
1. DenSHAP_figures_en.ipynb     → Fig 1–6 (main paper)
2. DenSHAP_sensitivity.ipynb    → Appendix figures & tables
```

---

## Datasets

| Dataset | N | Features | Source |
|---------|---|----------|--------|
| HELOC | 9,871 | 23 | FICO |
| LendingClub | 1,373,324 | 20 | Kaggle |
| Wine Quality | 6,497 | 11 | UCI |

Place CSV files in `data/` before running.

---

## Configuration

Edit `CONFIG` in `run_experiment.py`:

```python
CONFIG = {
    'eval_n'                 : 999999, # evaluation samples (999999 = full test set)
    'optuna_n_trials'        : 30,     # Optuna tuning trials
    'k_total'                : 100,    # background dataset size
    'k_lof'                  : 20,     # LOF neighbourhood size
    'p_low'                  : 25.0,   # α=0 percentile (Easy boundary)
    'p_high'                 : 75.0,   # α=1 percentile (Hard boundary)
    'n_candidates_multiplier': 5,      # candidate pool = k_total × 5
    'lendingclub_sample_n'   : 20000,  # LendingClub training sample size
}
```

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
difficulty groups, with statistically significant improvement in Hard Cases (p < 0.001,
Wilcoxon signed-rank test).

---

## Key Results

| Dataset | Group | CF-SHAP BDS | DenSHAP BDS | Δ | p-value |
|---------|-------|-------------|-------------|---|---------|
| HELOC | Hard | 0.9386 | **0.9650** | +2.8% | 2.60e-56 |
| Wine | Hard | 0.9644 | **0.9763** | +1.2% | 3.05e-06 |
| LendingClub | Hard | 0.9652 | **0.9744** | +1.0% | 5.49e-24 |

Validity is guaranteed at 100% across all datasets via model-prediction-based filtering.

---

## Citation

```bibtex
@article{denshap2025,
  title  = {DenSHAP: Density-Aware Background Dataset Construction
            for Counterfactual Shapley Explanations},
  author = {[Author]},
  year   = {2025}
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
