"""
InstanceSHAP Supplementary Experiment
======================================
Adds INSTANCE_SHAP baseline to existing XGBoost experiment results.

- Loads cached XGBoost model (results/cache/{dataset}_xgboost_model.joblib)
  or falls back to legacy cache (results/{dataset}_model.joblib)
- Loads existing cfshap CSV to recover eval sample order
- Computes InstanceSHAP background + SHAP values + metrics
- Merges INSTANCE_SHAP columns into existing CSVs
- Does NOT rerun XGBoost, CF-SHAP, or DenSHAP

Usage:
    python run_instance_shap_only.py                  # all datasets, k_lof=20
    python run_instance_shap_only.py --dataset heloc
    python run_instance_shap_only.py --k_lof 20       # single k_lof
    python run_instance_shap_only.py --k_lof 10 20 30 50  # all k_lof values
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset
from cfshap_baseline import (
    get_background_instance_shap,
    compute_shap_values,
    counterfactual_ability,
    plausibility,
    compute_iqr,
    background_density_score,
)
from dhace import LOFDifficultyEstimator

DATASETS         = ['heloc', 'wine', 'lendingclub']
DEFAULT_K_LOF    = [10, 20, 30, 50]
LENDINGCLUB_N    = 20000
RANDOM_STATE     = 42
K_NEIGHBORS      = 100
TOP_K_LIST       = [1, 2, 3, 5]
FAILURE_PENALTY  = 10.0


# ── Helpers ───────────────────────────────────────────────────

def find_model_cache(dataset_name: str) -> str:
    """Locate model cache — supports new and legacy naming."""
    candidates = [
        os.path.join('results', 'cache', f'{dataset_name}_xgboost_model.joblib'),
        os.path.join('results', f'{dataset_name}_model.joblib'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No cached model found for {dataset_name}. '
        f'Checked: {candidates}'
    )


def find_cfshap_csv(dataset_name: str, k_lof: int, base_dir: str) -> str:
    """Locate existing cfshap CSV — supports new and legacy directory layout."""
    candidates = [
        os.path.join(base_dir, 'xgboost', f'klof{k_lof}', f'{dataset_name}_cfshap.csv'),
        os.path.join(base_dir, f'klof{k_lof}', f'{dataset_name}_cfshap.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No cfshap CSV found for {dataset_name} k_lof={k_lof}. '
        f'Checked: {candidates}'
    )


def find_denshap_csv(dataset_name: str, k_lof: int, base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, 'xgboost', f'klof{k_lof}', f'{dataset_name}_denshap.csv'),
        os.path.join(base_dir, f'klof{k_lof}', f'{dataset_name}_denshap.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No denshap CSV found for {dataset_name} k_lof={k_lof}. '
        f'Checked: {candidates}'
    )


def output_dir_for(dataset_name: str, k_lof: int, base_dir: str) -> str:
    """Use new layout; create if missing."""
    d = os.path.join(base_dir, 'xgboost', f'klof{k_lof}')
    os.makedirs(d, exist_ok=True)
    return d


# ── Per-sample InstanceSHAP evaluation ───────────────────────

def evaluate_instance_shap_single(
    x: np.ndarray,
    query_label: int,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    iqr: np.ndarray,
    lof_estimator,
    k: int = K_NEIGHBORS,
) -> dict:
    row = {}
    background = get_background_instance_shap(x, X_train, k=k, iqr=iqr)

    if len(background) == 0:
        for kk in TOP_K_LIST:
            row[f'INSTANCE_SHAP_CA_top{kk}'] = FAILURE_PENALTY
        row['INSTANCE_SHAP_plausibility'] = np.nan
        row['INSTANCE_SHAP_bds']          = np.nan
        return row

    try:
        sv = compute_shap_values(model, x, background, model_type='tree')
    except Exception:
        for kk in TOP_K_LIST:
            row[f'INSTANCE_SHAP_CA_top{kk}'] = FAILURE_PENALTY
        row['INSTANCE_SHAP_plausibility'] = np.nan
        row['INSTANCE_SHAP_bds']          = np.nan
        return row

    ca = counterfactual_ability(
        x, sv, X_train, y_train, model,
        query_label, iqr, TOP_K_LIST, FAILURE_PENALTY,
    )
    for kk, v in ca.items():
        row[f'INSTANCE_SHAP_CA_top{kk}'] = v

    pl_scores = [plausibility(bg_row, X_train, iqr=iqr)
                 for bg_row in background[:10]]
    row['INSTANCE_SHAP_plausibility'] = float(np.mean(pl_scores))
    row['INSTANCE_SHAP_bds']          = background_density_score(
        background[:10], lof_estimator
    )
    return row


# ── Single dataset runner ─────────────────────────────────────

def run_instance_shap(
    dataset_name: str,
    k_lof: int,
    base_dir: str,
    log_lines: list,
):
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  Dataset : {dataset_name.upper()}  |  k_lof={k_lof}')
    print(f'{sep}')

    # ── Load data ─────────────────────────────────────────────
    if dataset_name == 'lendingclub':
        from data_loader import load_lendingclub
        X_train, X_test, y_train, y_test, feature_names = load_lendingclub(
            'data/lendingclub.csv', sample_n=LENDINGCLUB_N,
        )
    else:
        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name)

    iqr = compute_iqr(X_train)

    # ── Load cached model ─────────────────────────────────────
    model_path = find_model_cache(dataset_name)
    print(f'  Loading model: {model_path}')
    model = joblib.load(model_path)

    # ── Load existing cfshap CSV (to recover eval sample order) ──
    cfshap_path = find_cfshap_csv(dataset_name, k_lof, base_dir)
    print(f'  Loading cfshap CSV: {cfshap_path}')
    cfshap_df = pd.read_csv(cfshap_path)
    n = len(cfshap_df)
    print(f'  Eval samples: {n}')

    # ── Reconstruct eval set from test set ────────────────────
    # Same stratified sampling as original run
    np.random.seed(RANDOM_STATE)
    y_pred_test = model.predict(X_test)
    idx_0 = np.where(y_pred_test == 0)[0]
    idx_1 = np.where(y_pred_test == 1)[0]
    n0    = min(n // 2, len(idx_0))
    n1    = min(n - n0, len(idx_1))
    sel   = np.concatenate([
        np.random.choice(idx_0, n0, replace=False),
        np.random.choice(idx_1, n1, replace=False),
    ])
    np.random.shuffle(sel)
    X_eval = X_test[sel]
    y_eval = y_pred_test[sel]

    if len(X_eval) != n:
        print(f'  [Warning] Eval size mismatch: CSV={n}, reconstructed={len(X_eval)}. '
              f'Using reconstructed set.')

    # ── Fit LOF (reuse same config as original run) ───────────
    print(f'  Fitting LOF (k_lof={k_lof})...')
    lof_est = LOFDifficultyEstimator(k_lof=k_lof, p_low=25.0, p_high=75.0)
    lof_est.fit(X_train)

    # ── Compute InstanceSHAP for each eval sample ─────────────
    print(f'  Computing InstanceSHAP...')
    t0 = time.time()
    records = []
    for i, (x, label) in enumerate(zip(X_eval, y_eval)):
        if i % 50 == 0:
            print(f'  [{i+1:>5}/{len(X_eval)}]', end='\r')
        records.append(
            evaluate_instance_shap_single(
                x, int(label), model, X_train, y_train,
                iqr, lof_est,
            )
        )
    print(f'  [{len(X_eval)}/{len(X_eval)}] done.  ({time.time()-t0:.1f}s)')

    inst_df = pd.DataFrame(records)

    # ── Merge into existing CSVs ──────────────────────────────
    out_dir = output_dir_for(dataset_name, k_lof, base_dir)

    # cfshap CSV: add INSTANCE_SHAP columns
    inst_cols = [c for c in inst_df.columns if c.startswith('INSTANCE_SHAP')]
    for col in inst_cols:
        cfshap_df[col] = inst_df[col].values
    cfshap_out = os.path.join(out_dir, f'{dataset_name}_cfshap.csv')
    cfshap_df.to_csv(cfshap_out, index=False)
    print(f'  Updated cfshap CSV: {cfshap_out}')

    # denshap CSV: add difficulty_group column for group-level BDS
    try:
        denshap_path = find_denshap_csv(dataset_name, k_lof, base_dir)
        denshap_df   = pd.read_csv(denshap_path)

        # BDS by group comparison
        print(f'\n  BDS by Group — CF-SHAP vs InstanceSHAP vs DenSHAP')
        print(f'  {"Group":<8} {"N":>5}  {"CF-SHAP":>8}  {"InstSHAP":>9}  {"DenSHAP":>8}  '
              f'{"vs CF":>7}  {"vs Inst":>8}')
        print(f'  {"-"*60}')

        bds_rows = []
        for group in ['Easy', 'Medium', 'Hard']:
            mask   = denshap_df['difficulty_group'] == group
            idx_g  = denshap_df[mask].index
            if len(idx_g) == 0:
                continue
            ds_bds   = denshap_df.loc[mask, 'DenSHAP_bds'].mean() \
                       if 'DenSHAP_bds' in denshap_df.columns else np.nan
            cf_bds   = cfshap_df.loc[idx_g, 'CF_SHAP_bds'].mean() \
                       if 'CF_SHAP_bds' in cfshap_df.columns else np.nan
            inst_bds = inst_df.loc[idx_g, 'INSTANCE_SHAP_bds'].mean() \
                       if 'INSTANCE_SHAP_bds' in inst_df.columns else np.nan

            delta_cf   = (ds_bds - cf_bds)   / cf_bds   * 100 if cf_bds   > 0 else np.nan
            delta_inst = (ds_bds - inst_bds) / inst_bds * 100 if inst_bds > 0 else np.nan

            print(f'  {group:<8} {len(idx_g):>5}  {cf_bds:>8.4f}  {inst_bds:>9.4f}  '
                  f'{ds_bds:>8.4f}  {delta_cf:>+7.2f}%  {delta_inst:>+8.2f}%')

            bds_rows.append({
                'Dataset':       dataset_name.upper(),
                'k_lof':         k_lof,
                'Group':         group,
                'N':             len(idx_g),
                'CF_SHAP_BDS':   round(cf_bds,   4),
                'INST_SHAP_BDS': round(inst_bds, 4),
                'DenSHAP_BDS':   round(ds_bds,   4),
                'Delta_vs_CF':   round(delta_cf,   2) if not np.isnan(delta_cf)   else np.nan,
                'Delta_vs_Inst': round(delta_inst, 2) if not np.isnan(delta_inst) else np.nan,
            })

        bds_out = os.path.join(out_dir, f'{dataset_name}_bds_by_group.csv')
        pd.DataFrame(bds_rows).to_csv(bds_out, index=False)
        print(f'\n  BDS table saved: {bds_out}')

    except FileNotFoundError as e:
        print(f'  [Warning] Could not load denshap CSV for group comparison: {e}')

    log_lines.append(
        f'  [{dataset_name.upper()}] k_lof={k_lof} InstanceSHAP done '
        f'({time.time()-t0:.1f}s) → {out_dir}/'
    )
    print(f'  ✅ Done → {out_dir}/')


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='InstanceSHAP supplementary experiment')
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all'] + DATASETS,
    )
    parser.add_argument(
        '--k_lof', type=int, nargs='+', default=DEFAULT_K_LOF,
        help='k_lof value(s) matching original experiment (default: 10 20 30 50)',
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Base results directory (default: results)',
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    log_lines = [
        'InstanceSHAP Supplementary Experiment Log',
        f'datasets = {datasets}',
        f'k_lof    = {args.k_lof}',
        '',
    ]

    t_total = time.time()
    for k_lof in args.k_lof:
        for ds in datasets:
            try:
                run_instance_shap(ds, k_lof, args.output, log_lines)
            except FileNotFoundError as e:
                print(f'\n[ERROR] {e}')
                log_lines.append(f'[ERROR] {ds} k_lof={k_lof}: {e}')
            except Exception as e:
                print(f'\n[ERROR] {ds} k_lof={k_lof}: {e}')
                import traceback; traceback.print_exc()
                log_lines.append(f'[ERROR] {ds} k_lof={k_lof}: {e}')

    elapsed = time.time() - t_total
    log_lines.append(f'\nTotal time: {elapsed/3600:.2f} h')
    print(f'\nTotal time: {elapsed:.1f}s')

    log_path = os.path.join(args.output, 'instance_shap_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f'Log saved: {log_path}')


if __name__ == '__main__':
    main()