"""
DenSHAP (Dynamic Hybrid Actionable Counterfactual Explanations)
================================================================
Three key improvements over CF-SHAP:

    C1. LOF-based difficulty-aware continuous mixing function α(x)
        - Replaces binary P≈0.5 classification with LOF-based continuous function
        - θ defined as percentile of LOF distribution (dataset-adaptive)

    C2. Density-Weighted Stochastic Projection
        - Instead of projecting DiCE candidates to single nearest point (FACE)
          uses probabilistic selection weighted by distance × inverse LOF
        - DiCE loss function itself is NOT modified

    C3. Hard Case focused analysis framework
        - Stratified evaluation by LOF group (Easy/Medium/Hard)
        - Improvement in Hard group is the key evidence for the paper's claim
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import shap
import warnings
warnings.filterwarnings('ignore')

from cfshap_baseline import (
    compute_shap_values, counterfactual_ability, plausibility, compute_iqr
)


# ─────────────────────────────────────────────
# C1. LOF-based difficulty estimation and α(x) computation
# ─────────────────────────────────────────────

class LOFDifficultyEstimator:
    """
    Fits LOF distribution on training data and maps
    CF generation difficulty of new input x to [0, 1].

    Key design decision:
        - θ defined as the p-th percentile of training LOF distribution
        - Resolves scale inconsistency of LOF across different datasets
        - α = 0 → KNN 100% (Easy), α = 1 → DiCE 100% (Hard)
    """

    def __init__(self, k_lof: int = 20, p_low: float = 25, p_high: float = 75):
        """
        Parameters
        ----------
        k_lof   : number of neighbors for LOF (k_LOF, separate from k_CF)
        p_low   : percentile for α=0 boundary (default: 25th → Easy lower bound)
        p_high  : percentile for α=1 boundary (default: 75th → Hard upper bound)
        """
        self.k_lof = k_lof
        self.p_low = p_low
        self.p_high = p_high
        self._lof_model = None
        self._q_low = None
        self._q_high = None

    def fit(self, X_train: np.ndarray):
        """Fit LOF distribution on training data."""
        # novelty=True: build LOF distribution on training data at fit time,
        # estimate LOF for new points at predict time
        self._lof_model = LocalOutlierFactor(
            n_neighbors=self.k_lof,
            novelty=True,
            metric='euclidean'
        )
        self._lof_model.fit(X_train)

        # Compute percentiles of training LOF distribution
        # When novelty=True, training LOF is accessed via negative_outlier_factor_
        train_lof = -self._lof_model.negative_outlier_factor_  # convert to positive values
        self._q_low = np.percentile(train_lof, self.p_low)
        self._q_high = np.percentile(train_lof, self.p_high)

        return self

    def score(self, x: np.ndarray) -> float:
        """
        Return LOF score for a single input x.
        LOF ≈ 1.0 → Dense region (Easy)
        LOF >> 1.0 → Sparse region (Hard)
        """
        lof = -self._lof_model.score_samples(x.reshape(1, -1))[0]
        return float(lof)

    def alpha(self, x: np.ndarray) -> float:
        """
        Map LOF → α(x) ∈ [0, 1].

        α(x) = clip((LOF(x) - Q_low) / (Q_high - Q_low), 0, 1)

        α = 0 → KNN 100% (Easy, dense region)
        α = 1 → DiCE 100% (Hard, sparse region)
        """
        lof = self.score(x)
        denom = self._q_high - self._q_low
        if denom == 0:
            return 0.5
        alpha = (lof - self._q_low) / denom
        return float(np.clip(alpha, 0.0, 1.0))

    def difficulty_group(self, x: np.ndarray) -> str:
        """Classify into Easy / Medium / Hard groups."""
        a = self.alpha(x)
        if a < 0.33:
            return 'Easy'
        elif a < 0.67:
            return 'Medium'
        else:
            return 'Hard'


# ─────────────────────────────────────────────
# C2. Density-Weighted Stochastic Projection
# ─────────────────────────────────────────────

def density_weighted_projection(
    cf_candidates: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_label: int,
    lof_estimator: LOFDifficultyEstimator,
    n_proj_neighbors: int = 20,
    temperature: float = 1.0,
    dedup: bool = True
) -> np.ndarray:
    """
    Density-Weighted Stochastic Projection.

    Key difference from FACE:
        - FACE: argmin d(c', x_i) → deterministic projection to single nearest point
        - DenSHAP: w(x_i) ∝ exp(-d/σ) × (1/LOF(x_i)) → probabilistic selection considering density

    This ensures:
        ① Even if c' is OOD, density weighting corrects toward realistic regions
        ② Dense points preferred (low LOF) → lower expected Plausibility
        ③ Probabilistic selection partially preserves diversity
        ④ Deduplication prevents convergence to the same point

    Parameters
    ----------
    cf_candidates   : CF candidates generated by DiCE (n_cf x n_features)
    X_train         : training data
    y_train         : training labels
    query_label     : predicted label of the query x
    lof_estimator   : already fitted LOFDifficultyEstimator
    n_proj_neighbors: number of neighbors to consider during projection
    temperature     : softness of distance weighting (higher = more uniform)
    dedup           : whether to prevent duplicate projected points

    Returns
    -------
    projected_cfs : projected CF set (n_cf x n_features)
    """
    # Use only opposite-label training data as candidates
    opposite_mask = (y_train != query_label)
    X_opp = X_train[opposite_mask]

    if len(X_opp) == 0:
        return cf_candidates

    # Pre-compute LOF scores for opposite-label data
    lof_scores_opp = np.array([
        lof_estimator.score(xi) for xi in X_opp
    ])
    # LOF weight: higher for denser points (lower LOF)
    lof_weights = 1.0 / np.maximum(lof_scores_opp, 1e-6)

    # IQR normalized Distance-based neighbor search
    from cfshap_baseline import compute_iqr, iqr_scale
    iqr = compute_iqr(X_train)
    n_neighbors = min(n_proj_neighbors, len(X_opp))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(iqr_scale(X_opp, iqr))

    projected = []
    used_indices = set()

    for c_prime in cf_candidates:
        distances, indices = nn.kneighbors(iqr_scale(c_prime.reshape(1, -1), iqr))
        distances = distances[0]
        indices = indices[0]

        # Distance weight: higher for closer points
        sigma = np.mean(distances) + 1e-6
        dist_weights = np.exp(-distances / (temperature * sigma))

        # Combined weight = distance weight × inverse LOF weight
        combined_weights = dist_weights * lof_weights[indices]
        combined_weights /= combined_weights.sum()  # normalized

        if dedup:
            # Exclude already selected indices
            available = [i for i, idx in enumerate(indices) if idx not in used_indices]
            if len(available) == 0:
                # If all used, sample with original weights
                chosen_pos = np.random.choice(len(indices), p=combined_weights)
            else:
                # Re-normalize and sample from available candidates
                avail_weights = combined_weights[available]
                avail_weights /= avail_weights.sum()
                chosen_pos = available[np.random.choice(len(available), p=avail_weights)]
        else:
            chosen_pos = np.random.choice(len(indices), p=combined_weights)

        chosen_idx = indices[chosen_pos]
        used_indices.add(chosen_idx)
        projected.append(X_opp[chosen_idx])

    return np.array(projected)


# ─────────────────────────────────────────────
# DenSHAP background dataset construction
# ─────────────────────────────────────────────

def get_background_dhace(
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    query_label: int,
    lof_estimator: LOFDifficultyEstimator,
    k_total: int = 100,
    k_lof_neighbors: int = 20,
    temperature: float = 1.0
) -> tuple:
    """
    DenSHAP background dataset construction.

    "DiCE-free" version that works without DiCE:
        Easy region: KNN only
        Hard region: density-weighted sampling from all opposite-label data

    Full DiCE integration handled in dhace_with_dice().

    Returns
    -------
    background    : final background dataset
    alpha         : computed α(x) value
    diff_group    : 'Easy' / 'Medium' / 'Hard'
    knn_success   : KNN search success flag
    """
    alpha = lof_estimator.alpha(x)
    diff_group = lof_estimator.difficulty_group(x)

    k_knn = int(round((1 - alpha) * k_total))
    k_dice = k_total - k_knn  # replaced by density-weighted sampling in DiCE-free mode

    # Opposite-label data
    opposite_mask = (y_train != query_label)
    X_opp = X_train[opposite_mask]

    if len(X_opp) == 0:
        return np.random.choice(X_train, size=k_total, replace=False), alpha, diff_group, False

    # KNN background (Easy component)
    knn_bg = []
    knn_success = True
    if k_knn > 0:
        n_knn = min(k_knn, len(X_opp))
        nn = NearestNeighbors(n_neighbors=n_knn, metric='euclidean')
        nn.fit(X_opp)
        _, indices = nn.kneighbors(x.reshape(1, -1))
        knn_bg = X_opp[indices[0]]
        knn_success = len(knn_bg) >= k_knn

    # Density-weighted sampling background (Hard component) — DiCE-free substitute
    density_bg = []
    if k_dice > 0:
        lof_scores = np.array([lof_estimator.score(xi) for xi in X_opp])
        lof_weights = 1.0 / np.maximum(lof_scores, 1e-6)

        # Combined distance and LOF weights
        from cfshap_baseline import compute_iqr, iqr_scale
        iqr_d = compute_iqr(X_train)
        dists = np.linalg.norm(iqr_scale(X_opp, iqr_d) - iqr_scale(x, iqr_d), axis=1)
        sigma = np.median(dists) + 1e-6
        dist_weights = np.exp(-dists / (temperature * sigma))
        combined = dist_weights * lof_weights
        combined /= combined.sum()

        n_sample = min(k_dice, len(X_opp))
        sampled_idx = np.random.choice(len(X_opp), size=n_sample, replace=False, p=combined)
        density_bg = X_opp[sampled_idx]

    # combined
    parts = [p for p in [knn_bg, density_bg] if len(p) > 0]
    if len(parts) == 0:
        background = X_opp[:k_total]
    elif len(parts) == 1:
        background = parts[0]
    else:
        background = np.vstack(parts)

    return background, alpha, diff_group, knn_success


# ─────────────────────────────────────────────
# DenSHAP unified evaluation class
# ─────────────────────────────────────────────

class DenSHAPEvaluator:
    """
    DenSHAP evaluation class.
    Provides identical interface to CFSHAPEvaluator for direct comparison.
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: int = 100,
        k_lof: int = 20,
        p_low: float = 25,
        p_high: float = 75,
        model_type: str = 'tree',
        top_k_list: list = [1, 2, 3, 5],
        failure_penalty: float = 10.0,
        temperature: float = 1.0,
        random_state: int = 42
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k_neighbors
        self.model_type = model_type
        self.top_k_list = top_k_list
        self.failure_penalty = failure_penalty
        self.iqr = compute_iqr(X_train)
        self.temperature = temperature
        np.random.seed(random_state)

        # Fit LOF estimator
        print('Fitting LOF estimator...')
        self.lof_estimator = LOFDifficultyEstimator(
            k_lof=k_lof, p_low=p_low, p_high=p_high
        )
        self.lof_estimator.fit(X_train)
        print(f'  LOF Q{p_low}: {self.lof_estimator._q_low:.3f}, '
              f'Q{p_high}: {self.lof_estimator._q_high:.3f}')

    def _evaluate_single(self, x: np.ndarray, query_label: int) -> dict:
        row = {}

        background, alpha, diff_group, knn_success = get_background_dhace(
            x, self.X_train, self.y_train, self.model,
            query_label, self.lof_estimator, self.k, temperature=self.temperature
        )

        row['alpha'] = alpha
        row['difficulty_group'] = diff_group
        row['knn_success'] = knn_success
        row['lof_score'] = self.lof_estimator.score(x)

        try:
            sv = compute_shap_values(self.model, x, background, self.model_type)
        except Exception as e:
            for k in self.top_k_list:
                row[f'DenSHAP_CA_top{k}'] = self.failure_penalty
            row['DenSHAP_plausibility'] = np.nan
            return row

        ca = counterfactual_ability(
            x, sv, self.X_train, self.y_train, self.model,
            query_label, self.iqr, self.top_k_list, self.failure_penalty
        )
        for k, v in ca.items():
            row[f'DenSHAP_CA_top{k}'] = v

        pl_scores = [plausibility(bg_row, self.X_train) for bg_row in background[:10]]
        row['DenSHAP_plausibility'] = float(np.mean(pl_scores))

        return row

    def evaluate(
        self,
        X_eval: np.ndarray,
        y_eval_pred: np.ndarray,
        verbose: bool = True
    ) -> pd.DataFrame:
        records = []
        n = len(X_eval)
        for i, (x, label) in enumerate(zip(X_eval, y_eval_pred)):
            if verbose and i % 10 == 0:
                print(f'  [{i+1}/{n}] evaluating...')
            row = self._evaluate_single(x, int(label))
            records.append(row)
        return pd.DataFrame(records)

    def summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results overall and by LOF difficulty group."""
        rows = []
        for group in ['Total', 'Easy', 'Medium', 'Hard']:
            if group == 'Total':
                df = results_df
            else:
                df = results_df[results_df['difficulty_group'] == group]

            if len(df) == 0:
                continue

            row = {'Group': group, 'N': len(df)}
            for k in self.top_k_list:
                col = f'DenSHAP_CA_top{k}'
                if col in df.columns:
                    row[f'CA_top{k}'] = df[col].mean()
            if 'DenSHAP_plausibility' in df.columns:
                row['Plausibility'] = df['DenSHAP_plausibility'].mean()
            row['α_mean'] = df['alpha'].mean()
            row['KNN_success'] = df['knn_success'].mean()
            rows.append(row)

        return pd.DataFrame(rows).set_index('Group')