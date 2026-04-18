"""
CF-SHAP Baseline Reimplementation
===================================
Albini et al. (2022), FAccT '22
"Counterfactual Shapley Additive Explanations"

Core logic of Algorithm 1:
    1. Search KNN in opposite-label training data for input x
    2. Use found KNN neighbors as SHAP background dataset
    3. Compute marginal (interventional) SHAP values
    4. Evaluate CA (Counterfactual-Ability) and Plausibility

Also implements 4 comparison baselines:
    - SHAP_TRAIN : Full training data as background
    - SHAP_D_LAB : All opposite-label data as background
    - SHAP_D_PRED: All opposite-predicted data as background
    - CF_SHAP    : KNN-based CF set as background (proposed in paper)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import shap
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Utility functions
# ─────────────────────────────────────────────

def compute_iqr(X_train: np.ndarray) -> np.ndarray:
    """Compute IQR per feature for CA normalization."""
    q75 = np.percentile(X_train, 75, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0  # avoid division by zero
    return iqr


def iqr_scale(X: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Scale X by IQR to normalize feature distances."""
    return X / np.maximum(iqr, 1e-8)


def background_density_score(background: np.ndarray, lof_estimator) -> float:
    """
    Background Density Score (BDS) — novel metric proposed by DenSHAP.

    BDS(B) = (1/|B|) * sum(1 / LOF(x_i))

    Higher BDS indicates background data selected from denser regions,
    reflecting higher plausibility of the explanation context.
    """
    if len(background) == 0:
        return 0.0
    lof_scores = np.array([lof_estimator.score(xi) for xi in background])
    return float(np.mean(1.0 / np.maximum(lof_scores, 1e-6)))


# ─────────────────────────────────────────────
# 2. Background dataset construction
# ─────────────────────────────────────────────

def get_background_train(X_train: np.ndarray, max_samples: int = 100) -> np.ndarray:
    """SHAP_TRAIN: random sample from full training data."""
    idx = np.random.choice(len(X_train), size=min(max_samples, len(X_train)), replace=False)
    return X_train[idx]


def get_background_d_lab(
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_label: int,
    max_samples: int = 100
) -> np.ndarray:
    """SHAP_D_LAB: random sample from all opposite-label training data."""
    opposite_mask = (y_train != query_label)
    X_opp = X_train[opposite_mask]
    idx = np.random.choice(len(X_opp), size=min(max_samples, len(X_opp)), replace=False)
    return X_opp[idx]


def get_background_d_pred(
    X_train: np.ndarray,
    y_pred_train: np.ndarray,
    query_label: int,
    max_samples: int = 100
) -> np.ndarray:
    """SHAP_D_PRED: random sample from all opposite-predicted training data."""
    opposite_mask = (y_pred_train != query_label)
    X_opp = X_train[opposite_mask]
    if len(X_opp) == 0:
        return get_background_train(X_train, max_samples)
    idx = np.random.choice(len(X_opp), size=min(max_samples, len(X_opp)), replace=False)
    return X_opp[idx]


def get_background_cf_shap(
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_label: int,
    k: int = 100,
    fallback_to_d_lab: bool = True
) -> tuple:
    """
    CF-SHAP (Algorithm 1): IQR-normalized KNN search in opposite-label data.

    Parameters
    ----------
    x                : query input (1D array)
    X_train          : training data
    y_train          : training labels
    query_label      : predicted label of x
    k                : number of neighbors
    fallback_to_d_lab: fall back to D_LAB if KNN fails

    Returns
    -------
    background : selected background data (np.ndarray)
    success    : True if k neighbors were found
    """
    opposite_mask = (y_train != query_label)
    X_opp = X_train[opposite_mask]

    if len(X_opp) == 0:
        if fallback_to_d_lab:
            return get_background_d_lab(X_train, y_train, query_label, k), False
        return None, False

    iqr = compute_iqr(X_train)
    n_neighbors = min(k, len(X_opp))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(iqr_scale(X_opp, iqr))
    _, indices = nn.kneighbors(iqr_scale(x.reshape(1, -1), iqr))

    background = X_opp[indices[0]]
    success = len(background) >= k
    return background, success


# ─────────────────────────────────────────────
# 3. SHAP value computation
# ─────────────────────────────────────────────

def compute_shap_values(
    model,
    x: np.ndarray,
    background: np.ndarray,
    model_type: str = 'tree'
) -> np.ndarray:
    """
    Compute marginal (interventional) SHAP values.

    Parameters
    ----------
    model      : trained classifier
    x          : query input (1D array)
    background : background dataset
    model_type : 'tree' or 'kernel'

    Returns
    -------
    shap_values : per-feature SHAP values (1D array)
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation='interventional'
        )
    else:
        explainer = shap.KernelExplainer(model.predict_proba, background)

    sv = explainer.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]  # binary classification: use class-1 SHAP values
    return sv.flatten()


# ─────────────────────────────────────────────
# 4. Evaluation metrics
# ─────────────────────────────────────────────

def counterfactual_ability(
    x: np.ndarray,
    shap_values: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    query_label: int,
    iqr: np.ndarray,
    top_k_list: list = [1, 2, 3, 5],
    failure_penalty: float = 10.0
) -> dict:
    """
    Compute CA (Counterfactual-Ability).

    Replaces top-k SHAP-ranked features with nearest opposite-label neighbor values
    and measures the minimum IQR-normalized L1 cost to cross the decision boundary.

    Formula:
        CA = min_{x' in A_k, F(x') != F(x)} c_L1(x, x')
        c_L1(x, x') = sum_i |x_i - x'_i| / IQR(X_i)
    """
    results = {}
    feature_order = np.argsort(np.abs(shap_values))[::-1]

    opposite_mask = (y_train != query_label)
    X_opp = X_train[opposite_mask]

    if len(X_opp) == 0:
        return {k: failure_penalty for k in top_k_list}

    iqr_ca = compute_iqr(X_train)
    nn = NearestNeighbors(n_neighbors=min(5, len(X_opp)), metric='euclidean')
    nn.fit(iqr_scale(X_opp, iqr_ca))
    _, idx = nn.kneighbors(iqr_scale(x.reshape(1, -1), iqr_ca))
    reference = X_opp[idx[0][0]]

    for k in top_k_list:
        x_modified = x.copy()
        x_modified[feature_order[:k]] = reference[feature_order[:k]]
        pred = model.predict(x_modified.reshape(1, -1))[0]
        if pred != query_label:
            results[k] = float(np.sum(np.abs(x - x_modified) / iqr))
        else:
            results[k] = failure_penalty

    return results


def plausibility(
    x_cf: np.ndarray,
    X_train: np.ndarray,
    n_neighbors: int = 5,
    iqr: np.ndarray = None
) -> float:
    """
    Compute Plausibility using IQR-normalized distance.

    Formula:
        Plausibility(x') = (1/5) * sum_{j=1}^{5} d_q(x', NN_j(x'))
    Lower is better (closer to real data distribution).
    """
    if iqr is None:
        iqr = compute_iqr(X_train)
    X_scaled    = iqr_scale(X_train, iqr)
    x_cf_scaled = iqr_scale(x_cf.reshape(1, -1), iqr)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_train)), metric='euclidean')
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(x_cf_scaled)
    return float(np.mean(distances[0]))


# ─────────────────────────────────────────────
# 5. Main evaluator class
# ─────────────────────────────────────────────

class CFSHAPEvaluator:
    """
    Unified evaluator for CF-SHAP and 4 comparison baselines.
    Accepts an optional LOF estimator to compute BDS for comparison with DenSHAP.
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: int = 100,
        model_type: str = 'tree',
        top_k_list: list = [1, 2, 3, 5],
        failure_penalty: float = 10.0,
        random_state: int = 42,
        lof_estimator=None
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k_neighbors
        self.model_type = model_type
        self.top_k_list = top_k_list
        self.failure_penalty = failure_penalty
        self.iqr = compute_iqr(X_train)
        self.lof_estimator = lof_estimator
        self.y_pred_train = model.predict(X_train)
        np.random.seed(random_state)
        self._build_opposite_index()

    def _build_opposite_index(self):
        """Pre-build per-class opposite-label index for speed."""
        self._opposite_index = {}
        for label in np.unique(self.y_train):
            mask = self.y_train != label
            self._opposite_index[label] = self.X_train[mask]

    def _evaluate_single(self, x: np.ndarray, query_label: int) -> dict:
        """Evaluate all 4 baselines for a single sample."""
        row = {}

        methods = {
            'SHAP_TRAIN' : get_background_train(self.X_train, self.k),
            'SHAP_D_LAB' : get_background_d_lab(self.X_train, self.y_train, query_label, self.k),
            'SHAP_D_PRED': get_background_d_pred(self.X_train, self.y_pred_train, query_label, self.k),
        }
        bg_cfshap, knn_success = get_background_cf_shap(
            x, self.X_train, self.y_train, query_label, self.k
        )
        methods['CF_SHAP'] = bg_cfshap
        row['CF_SHAP_knn_success'] = knn_success

        for method_name, background in methods.items():
            if background is None or len(background) == 0:
                for k in self.top_k_list:
                    row[f'{method_name}_CA_top{k}'] = self.failure_penalty
                row[f'{method_name}_plausibility'] = np.nan
                continue

            try:
                sv = compute_shap_values(self.model, x, background, self.model_type)
            except Exception:
                for k in self.top_k_list:
                    row[f'{method_name}_CA_top{k}'] = self.failure_penalty
                row[f'{method_name}_plausibility'] = np.nan
                continue

            ca = counterfactual_ability(
                x, sv, self.X_train, self.y_train, self.model,
                query_label, self.iqr, self.top_k_list, self.failure_penalty
            )
            for k, v in ca.items():
                row[f'{method_name}_CA_top{k}'] = v

            pl_scores = [plausibility(bg_row, self.X_train, iqr=self.iqr)
                         for bg_row in background[:10]]
            row[f'{method_name}_plausibility'] = float(np.mean(pl_scores))

            if self.lof_estimator is not None:
                row[f'{method_name}_bds'] = background_density_score(
                    background[:10], self.lof_estimator
                )

        return row

    def evaluate(
        self,
        X_eval: np.ndarray,
        y_eval_pred: np.ndarray,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Run evaluation on the full evaluation set."""
        records = []
        n = len(X_eval)
        for i, (x, label) in enumerate(zip(X_eval, y_eval_pred)):
            if verbose and i % 10 == 0:
                print(f'  [{i+1}/{n}] evaluating...')
            records.append(self._evaluate_single(x, int(label)))
        return pd.DataFrame(records)

    def summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results by method mean."""
        methods = ['SHAP_TRAIN', 'SHAP_D_LAB', 'SHAP_D_PRED', 'CF_SHAP']
        rows = []
        for m in methods:
            row = {'Method': m}
            for k in self.top_k_list:
                col = f'{m}_CA_top{k}'
                if col in results_df.columns:
                    row[f'CA_top{k}'] = results_df[col].mean()
            if f'{m}_plausibility' in results_df.columns:
                row['Plausibility'] = results_df[f'{m}_plausibility'].mean()
            if f'{m}_bds' in results_df.columns:
                row['BDS'] = results_df[f'{m}_bds'].mean()
            if m == 'CF_SHAP' and 'CF_SHAP_knn_success' in results_df.columns:
                row['KNN_success_rate'] = results_df['CF_SHAP_knn_success'].mean()
            rows.append(row)
        return pd.DataFrame(rows).set_index('Method')
