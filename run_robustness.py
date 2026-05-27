"""
DenSHAP Robustness Experiment — LightGBM & Random Forest
==========================================================
Appendix: "Robustness Across Model Families"

Runs DenSHAP vs CF-SHAP vs InstanceSHAP on LightGBM and RF.
XGBoost results are loaded from existing CSVs for comparison table.

Usage:
    python run_robustness.py                         # LightGBM + RF, all datasets
    python run_robustness.py --model lightgbm        # LightGBM only
    python run_robustness.py --dataset heloc         # one dataset
    python run_robustness.py --k_lof 20              # single k_lof (recommended for Appendix)

Outputs saved to results/{model}/klof{k}/:
    {dataset}_cfshap.csv
    {dataset}_denshap.csv
    {dataset}_bds_by_group.csv
    {dataset}_summary.csv
    robustness_summary.csv   ← cross-model BDS comparison table
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset
from cfshap_baseline import CFSHAPEvaluator
from denshap import DenSHAPPipeline

DATASETS        = ['heloc', 'wine', 'lendingclub']
LENDINGCLUB_N   = 20000
RANDOM_STATE    = 42
K_NEIGHBORS     = 100
K_TOTAL         = 100
TOP_K_LIST      = [1, 2, 3, 5]
FAILURE_PENALTY = 10.0
OPTUNA_TRIALS   = 30
OPTUNA_CV       = 3


# ── Model tuning ──────────────────────────────────────────────

def tune_lightgbm(X_train, y_train):
    from lightgbm import LGBMClassifier

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 100, 500),
            max_depth         = trial.suggest_int('max_depth', 3, 8),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample         = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            min_child_samples = trial.suggest_int('min_child_samples', 5, 50),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            random_state      = RANDOM_STATE,
            verbosity         = -1,
        )
        m   = LGBMClassifier(**params)
        cv  = StratifiedKFold(n_splits=OPTUNA_CV, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(m, X_train, y_train, cv=cv,
                               scoring='roc_auc', n_jobs=-1).mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'random_state': RANDOM_STATE, 'verbosity': -1})
    print(f'  [LightGBM] Best CV AUC: {study.best_value:.4f}')
    return best, study.best_value


def tune_rf(X_train, y_train):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 100, 500),
            max_depth         = trial.suggest_int('max_depth', 3, 20),
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10),
            max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        )
        m  = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=OPTUNA_CV, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(m, X_train, y_train, cv=cv,
                               scoring='roc_auc', n_jobs=-1).mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
    print(f'  [RF] Best CV AUC: {study.best_value:.4f}')
    return best, study.best_value


def build_and_cache_model(model_name, dataset_name, X_train, y_train):
    cache_dir  = os.path.join('results', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, f'{dataset_name}_{model_name}_model.joblib')
    param_path = os.path.join(cache_dir, f'{dataset_name}_{model_name}_params.joblib')

    if os.path.exists(model_path):
        print(f'  Loading cached {model_name} model: {model_path}')
        return joblib.load(model_path), joblib.load(param_path)

    print(f'  Tuning {model_name} ({OPTUNA_TRIALS} trials)...')
    tune_fn = {'lightgbm': tune_lightgbm, 'rf': tune_rf}[model_name]
    best_params, best_auc = tune_fn(X_train, y_train)

    if model_name == 'lightgbm':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**best_params)
    else:
        model = RandomForestClassifier(**best_params)

    model.fit(X_train, y_train)
    joblib.dump(model,       model_path)
    joblib.dump(best_params, param_path)
    print(f'  Cached: {model_path}')
    return model, best_params


# ── Single dataset runner ─────────────────────────────────────

def run_single(dataset_name, model_name, k_lof, base_dir, log_lines):
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  Dataset : {dataset_name.upper()}  |  Model: {model_name.upper()}  |  k_lof={k_lof}')
    print(f'{sep}')

    # ── Data ──────────────────────────────────────────────────
    if dataset_name == 'lendingclub':
        from data_loader import load_lendingclub
        X_train, X_test, y_train, y_test, feature_names = load_lendingclub(
            'data/lendingclub.csv', sample_n=LENDINGCLUB_N,
        )
    else:
        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name)

    # ── Model ─────────────────────────────────────────────────
    model, best_params = build_and_cache_model(model_name, dataset_name, X_train, y_train)

    y_pred_test = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    acc = (y_pred_test == y_test).mean()
    print(f'  Test AUC: {auc:.4f}  Acc: {acc:.4f}')
    log_lines.append(f'  [{dataset_name.upper()}|{model_name}] AUC={auc:.4f} Acc={acc:.4f}')

    # ── Eval sample (same seed as original) ───────────────────
    np.random.seed(RANDOM_STATE)
    n_total = len(X_test)
    idx_0   = np.where(y_pred_test == 0)[0]
    idx_1   = np.where(y_pred_test == 1)[0]
    n0      = min(n_total // 2, len(idx_0))
    n1      = min(n_total - n0, len(idx_1))
    sel     = np.concatenate([
        np.random.choice(idx_0, n0, replace=False),
        np.random.choice(idx_1, n1, replace=False),
    ])
    np.random.shuffle(sel)
    X_eval = X_test[sel]
    y_eval = y_pred_test[sel]
    print(f'  Eval: {len(X_eval)} samples')

    out_dir = os.path.join(base_dir, model_name, f'klof{k_lof}')
    os.makedirs(out_dir, exist_ok=True)

    # ── DenSHAP pipeline ──────────────────────────────────────
    denshap_pipe = DenSHAPPipeline(
        model=model,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        k_total=K_TOTAL,
        k_lof=k_lof,
        p_low=25.0,
        p_high=75.0,
        temperature=1.0,
        model_type='tree',
        top_k_list=TOP_K_LIST,
        failure_penalty=FAILURE_PENALTY,
        random_state=RANDOM_STATE,
    )

    # ── Baselines (CF-SHAP + InstanceSHAP) ───────────────────
    cfshap_eval = CFSHAPEvaluator(
        model=model,
        X_train=X_train,
        y_train=y_train,
        k_neighbors=K_NEIGHBORS,
        model_type='tree',
        top_k_list=TOP_K_LIST,
        failure_penalty=FAILURE_PENALTY,
        random_state=RANDOM_STATE,
        lof_estimator=denshap_pipe.lof_est,
    )

    t0 = time.time()
    cfshap_results  = cfshap_eval.evaluate(X_eval, y_eval, verbose=True)
    denshap_results = denshap_pipe.evaluate(X_eval, y_eval, verbose=True)
    elapsed = time.time() - t0

    # ── BDS by group ──────────────────────────────────────────
    print(f'\n  BDS by Group — CF-SHAP vs InstanceSHAP vs DenSHAP')
    print(f'  {"Group":<8} {"N":>5}  {"CF-SHAP":>8}  {"InstSHAP":>9}  '
          f'{"DenSHAP":>8}  {"vs CF":>7}  {"vs Inst":>8}')
    print(f'  {"-"*62}')

    bds_rows = []
    for group in ['Easy', 'Medium', 'Hard']:
        mask     = denshap_results['difficulty_group'] == group
        idx_g    = denshap_results[mask].index
        if len(idx_g) == 0:
            continue
        ds_bds   = denshap_results.loc[mask, 'DenSHAP_bds'].mean() \
                   if 'DenSHAP_bds' in denshap_results.columns else np.nan
        cf_bds   = cfshap_results.loc[idx_g, 'CF_SHAP_bds'].mean() \
                   if 'CF_SHAP_bds' in cfshap_results.columns else np.nan
        inst_bds = cfshap_results.loc[idx_g, 'INSTANCE_SHAP_bds'].mean() \
                   if 'INSTANCE_SHAP_bds' in cfshap_results.columns else np.nan

        delta_cf   = (ds_bds - cf_bds)   / cf_bds   * 100 if cf_bds   > 0 else np.nan
        delta_inst = (ds_bds - inst_bds) / inst_bds * 100 if inst_bds > 0 else np.nan

        print(f'  {group:<8} {len(idx_g):>5}  {cf_bds:>8.4f}  {inst_bds:>9.4f}  '
              f'{ds_bds:>8.4f}  {delta_cf:>+7.2f}%  {delta_inst:>+8.2f}%')

        bds_rows.append({
            'Dataset':       dataset_name.upper(),
            'Model':         model_name,
            'k_lof':         k_lof,
            'Group':         group,
            'N':             len(idx_g),
            'Test_AUC':      round(auc, 4),
            'CF_SHAP_BDS':   round(cf_bds,   4),
            'INST_SHAP_BDS': round(inst_bds, 4),
            'DenSHAP_BDS':   round(ds_bds,   4),
            'Delta_vs_CF':   round(delta_cf,   2) if not np.isnan(delta_cf)   else np.nan,
            'Delta_vs_Inst': round(delta_inst, 2) if not np.isnan(delta_inst) else np.nan,
        })

    # ── Save ──────────────────────────────────────────────────
    cfshap_results.to_csv(f'{out_dir}/{dataset_name}_cfshap.csv',   index=False)
    denshap_results.to_csv(f'{out_dir}/{dataset_name}_denshap.csv', index=False)
    pd.DataFrame(bds_rows).to_csv(f'{out_dir}/{dataset_name}_bds_by_group.csv', index=False)

    log_lines.append(
        f'  [{dataset_name.upper()}|{model_name}] k_lof={k_lof} '
        f'elapsed={elapsed:.1f}s → {out_dir}/'
    )
    print(f'  ✅ Saved → {out_dir}/')
    return bds_rows


# ── Cross-model summary table ─────────────────────────────────

def build_robustness_summary(all_bds_rows: list, base_dir: str):
    """Combine XGBoost (existing) + new model results into one Appendix table."""
    df_new = pd.DataFrame(all_bds_rows)

    # Try to load existing XGBoost BDS results
    xgb_frames = []
    for ds in DATASETS:
        for k_lof in df_new['k_lof'].unique():
            candidates = [
                os.path.join(base_dir, 'xgboost', f'klof{k_lof}', f'{ds}_bds_by_group.csv'),
                os.path.join(base_dir, f'klof{k_lof}', f'{ds}_bds_by_group.csv'),
            ]
            for p in candidates:
                if os.path.exists(p):
                    tmp = pd.read_csv(p)
                    tmp['Model'] = 'xgboost'
                    xgb_frames.append(tmp)
                    break

    frames = ([pd.concat(xgb_frames, ignore_index=True)] if xgb_frames else []) + [df_new]
    summary = pd.concat(frames, ignore_index=True)

    out_path = os.path.join(base_dir, 'robustness_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f'\n  Robustness summary saved: {out_path}')

    # Print pivot for Hard group only
    hard = summary[summary['Group'] == 'Hard'].copy()
    if not hard.empty:
        pivot = hard.pivot_table(
            index=['Dataset', 'Group'],
            columns='Model',
            values='DenSHAP_BDS',
            aggfunc='mean',
        ).round(4)
        print('\n  Hard-group DenSHAP BDS across models:')
        print(pivot.to_string())


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DenSHAP robustness experiment (Appendix)')
    parser.add_argument(
        '--model', type=str, nargs='+', default=['lightgbm', 'rf'],
        choices=['lightgbm', 'rf'],
        help='Models to evaluate (default: lightgbm rf)',
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all'] + DATASETS,
    )
    parser.add_argument(
        '--k_lof', type=int, nargs='+', default=[20],
        help='k_lof value(s) — recommend single value (20) for Appendix (default: 20)',
    )
    parser.add_argument(
        '--output', type=str, default='results',
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    log_lines = [
        'DenSHAP Robustness Experiment Log',
        f'models   = {args.model}',
        f'datasets = {datasets}',
        f'k_lof    = {args.k_lof}',
        '',
    ]

    t_total  = time.time()
    all_rows = []

    for model_name in args.model:
        for k_lof in args.k_lof:
            for ds in datasets:
                try:
                    rows = run_single(ds, model_name, k_lof, args.output, log_lines)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f'\n[ERROR] {ds} {model_name} k_lof={k_lof}: {e}')
                    import traceback; traceback.print_exc()
                    log_lines.append(f'[ERROR] {ds} {model_name} k_lof={k_lof}: {e}')

    if all_rows:
        build_robustness_summary(all_rows, args.output)

    elapsed = time.time() - t_total
    log_lines.append(f'\nTotal time: {elapsed/3600:.2f} h')
    print(f'\nTotal time: {elapsed/3600:.2f} h')

    log_path = os.path.join(args.output, 'robustness_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f'Log saved: {log_path}')


if __name__ == '__main__':
    main()