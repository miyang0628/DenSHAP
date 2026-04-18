"""
DenSHAP — LOF-Weighted KNN Selection
================================

Key changes (v2 → v3):
    v2: KNN-first + DiCE supplement
        Issue: KNN 100% success → DiCE contaminates background data

    v3: DiCE fully removed; replace KNN selection with LOF-weighted selection
        - Pure distance KNN → jointly consider distance + density
        - w(x_i) ∝ exp(-d(x,x_i)/σ) × (1/LOF(x_i))
        - Background always from actual opposite-label data (Validity 100% structurally guaranteed)
        - Prefers denser neighbors in Hard region → improved Plausibility

Redefined Contributions:
    CF-SHAP: KNN = distance only
    DenSHAP:  KNN = distance + density (inverse LOF) jointly
    → Improved background data quality in Hard Cases
    → Hard improvement > Easy improvement (core paper claim)

Design principles:
    - DiCE dependency fully removed (significant speed-up)
    - Identical interface to CFSHAPEvaluator
    - IQR-normalized distance (removes feature scale differences)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

from cfshap_baseline import (
    compute_shap_values, counterfactual_ability,
    plausibility, compute_iqr, iqr_scale,
    background_density_score
)
from dhace import LOFDifficultyEstimator


# ─────────────────────────────────────────────
# Core function: LOF-weighted KNN selection
# ─────────────────────────────────────────────

def lof_weighted_knn(
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_label: int,
    lof_estimator: LOFDifficultyEstimator,
    iqr: np.ndarray,
    k: int,
    n_candidates: int = None,
    temperature: float = 1.0,
    y_pred_train: np.ndarray = None
) -> tuple:
    """
    LOF-weighted KNN background data selection.

    Key difference from CF-SHAP (plain distance KNN):
        CF-SHAP: argmin_k d(x, x_i)              ← distance only
        DenSHAP:  w(x_i) ∝ exp(-d/σ) × (1/LOF(x_i))  ← distance + density

    Prefers denser neighbors (lower LOF):
        - Easy: Distance weight dominates → similar to CF-SHAP (expected)
        - Hard: LOF weight active → prefers denser neighbors → improved Plausibility

    Parameters
    ----------
    x             : query input to explain
    X_train       : training data
    y_train       : training labels
    query_label   : predicted label of x
    lof_estimator : fitted LOFDifficultyEstimator
    iqr           : per-feature IQR (for distance normalization)
    k             : final number of background samples to select
    n_candidates  : candidate pool size for LOF-weighted selection (None = k*5)
    temperature   : softness of distance weighting (higher = more uniform)
    y_pred_train  : model predicted labels (None = use training labels)
                    Filter by model predictions → structural Validity guarantee

    Returns
    -------
    background : selected background dataset (k × n_features)
    success    : True if k samples were successfully selected
    """
    # Opposite-label pool: model prediction basis (Validity guarantee) or training label (fallback)
    filter_label  = y_pred_train if y_pred_train is not None else y_train
    opposite_mask = (filter_label != query_label)
    X_opp = X_train[opposite_mask]

    if len(X_opp) == 0:
        return np.empty((0, X_train.shape[1])), False

    # Determine candidate pool size
    if n_candidates is None:
        n_candidates = min(k * 5, len(X_opp))
    else:
        n_candidates = min(n_candidates, len(X_opp))

    k_final = min(k, len(X_opp))

    # Step 1: Select top n_candidates by IQR distance
    nn = NearestNeighbors(n_neighbors=n_candidates, metric='euclidean')
    nn.fit(iqr_scale(X_opp, iqr))
    distances, indices = nn.kneighbors(iqr_scale(x.reshape(1, -1), iqr))
    distances = distances[0]
    indices   = indices[0]

    candidates     = X_opp[indices]       # candidate pool
    cand_distances = distances            # IQR-normalized distances

    # Step 2: Compute LOF weights (for candidate pool only)
    lof_scores = np.array([lof_estimator.score(c) for c in candidates])
    lof_w = 1.0 / np.maximum(lof_scores, 1e-6)  # Inverse LOF: higher for denser points

    # Step 3: Compute distance weights
    sigma  = np.median(cand_distances) + 1e-6
    dist_w = np.exp(-cand_distances / (temperature * sigma))

    # Step 4: Combined weights → probabilistic selection (diversity preserved)
    combined = lof_w * dist_w
    combined /= combined.sum()

    selected_idx = np.random.choice(
        len(candidates),
        size=k_final,
        replace=False,
        p=combined
    )

    background = candidates[selected_idx]
    success    = len(background) >= k

    return background, success


# ─────────────────────────────────────────────
# DenSHAP Pipeline
# ─────────────────────────────────────────────

class DenSHAPPipeline:
    """
    DenSHAP — LOF-Weighted KNN Selection pipeline.

    [Phase 1] LOF → α(x) → difficulty group classification
    [Phase 2] LOF-weighted KNN selection of opposite-label background data
    [Phase 3] SHAP(background = LOF-weighted KNN)

    No DiCE dependency → significant speed-up.
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list,
        k_total: int = 100,
        k_lof: int = 20,
        p_low: float = 25.0,
        p_high: float = 75.0,
        n_candidates_multiplier: int = 5,
        temperature: float = 1.0,
        model_type: str = 'tree',
        top_k_list: list = [1, 2, 3, 5],
        failure_penalty: float = 10.0,
        random_state: int = 42
    ):
        self.model        = model
        self.X_train      = X_train
        self.y_train      = y_train
        self.feature_names = feature_names
        self.k_total      = k_total
        self.n_cand_mult  = n_candidates_multiplier
        self.temperature  = temperature
        self.model_type   = model_type
        self.top_k_list   = top_k_list
        self.failure_penalty = failure_penalty
        self.iqr          = compute_iqr(X_train)
        self.y_pred_train = model.predict(X_train)  # For model-prediction-based filtering
        np.random.seed(random_state)

        # Pre-fit LOF estimator
        print('[Init] Fitting LOF estimator...')
        self.lof_est = LOFDifficultyEstimator(
            k_lof=k_lof, p_low=p_low, p_high=p_high
        )
        self.lof_est.fit(X_train)
        print(f'         Q{int(p_low)}: {self.lof_est._q_low:.4f} | '
              f'Q{int(p_high)}: {self.lof_est._q_high:.4f}')
        print('[Init] Done. (DiCE-free — LOF-weighted KNN only)\n')

    # ── Phase 1 ──────────────────────────────
    def _phase1(self, x: np.ndarray) -> tuple:
        lof_score = self.lof_est.score(x)
        alpha     = self.lof_est.alpha(x)
        group     = self.lof_est.difficulty_group(x)
        return lof_score, alpha, group

    # ── Phase 2: LOF-weighted KNN ───────────────
    def _phase2_lof_knn(
        self,
        x: np.ndarray,
        query_label: int
    ) -> tuple:
        return lof_weighted_knn(
            x=x,
            X_train=self.X_train,
            y_train=self.y_train,
            query_label=query_label,
            lof_estimator=self.lof_est,
            iqr=self.iqr,
            k=self.k_total,
            n_candidates=self.k_total * self.n_cand_mult,
            temperature=self.temperature,
            y_pred_train=self.y_pred_train
        )

    # ── Single-sample full pipeline ─────────────
    def _run_single(self, x: np.ndarray, query_label: int) -> dict:
        row = {}

        # Phase 1: Difficulty estimation
        lof_score, alpha, group = self._phase1(x)
        row.update({
            'lof_score':        lof_score,
            'alpha':            alpha,
            'difficulty_group': group,
        })

        # Phase 2: LOF-weighted KNN
        background, success = self._phase2_lof_knn(x, query_label)
        row['knn_success']    = success
        row['background_size'] = len(background)

        if len(background) == 0:
            for k in self.top_k_list:
                row[f'DenSHAP_CA_top{k}'] = self.failure_penalty
            row['DenSHAP_plausibility'] = np.nan
            row['DenSHAP_validity']     = 0.0
            return row

        # Check Validity (structurally guaranteed at 100%)
        bg_preds = self.model.predict(background)
        row['DenSHAP_validity'] = float(np.mean(bg_preds != query_label))

        # Phase 3: SHAP
        try:
            sv = compute_shap_values(
                self.model, x, background, self.model_type
            )
        except Exception:
            for k in self.top_k_list:
                row[f'DenSHAP_CA_top{k}'] = self.failure_penalty
            row['DenSHAP_plausibility'] = np.nan
            row['DenSHAP_validity']     = row.get('DenSHAP_validity', 0.0)
            return row

        # CA
        ca = counterfactual_ability(
            x, sv,
            self.X_train, self.y_train,
            self.model, query_label,
            self.iqr, self.top_k_list, self.failure_penalty
        )
        for k, v in ca.items():
            row[f'DenSHAP_CA_top{k}'] = v

        # Plausibility (IQR normalized)
        pl_scores = [
            plausibility(bg_row, self.X_train, iqr=self.iqr)
            for bg_row in background[:min(10, len(background))]
        ]
        row['DenSHAP_plausibility'] = float(np.mean(pl_scores))

        # BDS (Background Density Score) — novel metric proposed by DenSHAP
        row['DenSHAP_bds'] = background_density_score(
            background[:min(10, len(background))], self.lof_est
        )

        return row

    # ── Full evaluation ────────────────────────────
    def evaluate(
        self,
        X_eval: np.ndarray,
        y_eval_pred: np.ndarray,
        verbose: bool = True
    ) -> pd.DataFrame:
        records = []
        n = len(X_eval)
        for i, (x, label) in enumerate(zip(X_eval, y_eval_pred)):
            if verbose and i % 50 == 0:
                lof, alpha, group = self._phase1(x)
                print(f'  [{i+1:>5}/{n}] group={group:<6} '
                      f'α={alpha:.2f} LOF={lof:.3f}', end='\r')
            row = self._run_single(x, int(label))
            records.append(row)
        if verbose:
            print(f'  [{n}/{n}] done.                              ')
        return pd.DataFrame(records)

    # ── Summary table ──────────────────────────
    def summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for group in ['Total', 'Easy', 'Medium', 'Hard']:
            df = results_df if group == 'Total' \
                 else results_df[results_df['difficulty_group'] == group]
            if len(df) == 0:
                continue

            row = {
                'Group':    group,
                'N':        len(df),
                'α_mean':   df['alpha'].mean(),
                'LOF_mean': df['lof_score'].mean(),
            }
            for k in self.top_k_list:
                col = f'DenSHAP_CA_top{k}'
                if col in df.columns:
                    row[f'CA_top{k}'] = df[col].mean()
            if 'DenSHAP_plausibility' in df.columns:
                row['Plausibility'] = df['DenSHAP_plausibility'].mean()
            if 'DenSHAP_validity' in df.columns:
                row['Validity'] = df['DenSHAP_validity'].mean()
            if 'knn_success' in df.columns:
                row['KNN_success'] = df['knn_success'].mean()
            if 'DenSHAP_bds' in df.columns:
                row['BDS'] = df['DenSHAP_bds'].mean()
            rows.append(row)

        return pd.DataFrame(rows).set_index('Group')