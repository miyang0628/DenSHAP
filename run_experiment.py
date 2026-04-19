"""
DenSHAP vs CF-SHAP — Main Experiment Script on Real Datasets
=================================================
Usage (py files and data/ folder must be in the same directory):

    python run_experiment.py                     # all 3 datasets, k_lof=[10,20,30,50]
    python run_experiment.py --dataset heloc     # specific dataset only
    python run_experiment.py --dataset wine --eval_n 200

k_LOF sensitivity analysis:
    python run_experiment.py --k_lof 10         # single k_lof value
    python run_experiment.py --k_lof 10 20 30 50  # multiple (default behaviour)

Outputs (saved to --output/{k_lof}/ subdirectory per k_lof value):
    results/
    ├── klof10/
    │   ├── {dataset}_cfshap.csv
    │   ├── {dataset}_denshap.csv
    │   ├── {dataset}_summary.csv
    │   ├── {dataset}_group.csv
    │   ├── {dataset}_bds_by_group.csv
    │   └── experiment_log.txt
    ├── klof20/  ...
    ├── klof30/  ...
    ├── klof50/  ...
    └── (model cache: results/{dataset}_model.joblib  ← shared across all k_lof)
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset
from cfshap_baseline import CFSHAPEvaluator
from denshap import DenSHAPPipeline

# ── Experiment Configuration ──────────────────────────────────
CONFIG = {
    # General
    'random_state':          42,
    'eval_n':                999999,   # 999999 = full test set
    'top_k_list':            [1, 2, 3, 5],
    'failure_penalty':       10.0,
    'lendingclub_sample_n':  20000,    # stratified sample from ~1.37M records

    # Model — Optuna hyperparameter tuning
    'optuna_n_trials': 30,
    'optuna_cv':       3,
    'xgb_params': dict(           # fallback defaults (used only if Optuna fails)
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    ),

    # CF-SHAP baseline
    'k_neighbors': 100,

    # DenSHAP
    'k_total':                100,
    'k_lof':                  20,     # default single value (overridden by CLI)
    'p_low':                  25.0,
    'p_high':                 75.0,
    'n_candidates_multiplier': 5,
    'temperature':            1.0,
}

# ── Default k_lof values for sensitivity analysis ─────────────
DEFAULT_K_LOF_LIST = [10, 20, 30, 50]

DATASETS = ['heloc', 'wine', 'lendingclub']


# ── Optuna XGBoost Tuning ─────────────────────────────────────
def tune_xgboost(X_train, y_train, n_trials=30, cv=3, random_state=42):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 100, 500),
            max_depth        = trial.suggest_int('max_depth', 3, 8),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
            gamma            = trial.suggest_float('gamma', 0.0, 1.0),
            reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            eval_metric      = 'logloss',
            random_state     = random_state,
            verbosity        = 0,
        )
        model  = XGBClassifier(**params)
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv_obj, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({'eval_metric': 'logloss',
                        'random_state': random_state,
                        'verbosity': 0})

    print(f'  Optuna done | Best CV AUC: {study.best_value:.4f}')
    print(f'  Best params: {best_params}')
    return best_params, study.best_value


# ── Single Dataset Runner ─────────────────────────────────────
def run_single_dataset(dataset_name: str,
                       eval_n: int,
                       output_dir: str,
                       k_lof: int,
                       log_lines: list):

    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  Dataset : {dataset_name.upper()}')
    print(f'  k_lof   : {k_lof}')
    print(f'  output  : {output_dir}/')
    print(f'{sep}')

    os.makedirs(output_dir, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────
    if dataset_name == 'lendingclub':
        from data_loader import load_lendingclub
        X_train, X_test, y_train, y_test, feature_names = load_lendingclub(
            'data/lendingclub.csv',
            sample_n=CONFIG['lendingclub_sample_n'],
        )
    else:
        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name)

    # ── Model: load cache or run Optuna tuning ────────────────
    # Model cache lives in results/ (shared across k_lof runs)
    base_dir    = 'results'
    os.makedirs(base_dir, exist_ok=True)
    model_path  = os.path.join(base_dir, f'{dataset_name}_model.joblib')
    params_path = os.path.join(base_dir, f'{dataset_name}_best_params.joblib')

    if os.path.exists(model_path) and os.path.exists(params_path):
        print(f'\n[Model] Loading cached model: {model_path}')
        model       = joblib.load(model_path)
        best_params = joblib.load(params_path)
        print(f'  Cached params: {best_params}')
        log_lines.append(f'  [{dataset_name.upper()}] Loaded cached model')
    else:
        print(f'\n[Model] Running Optuna tuning ({CONFIG["optuna_n_trials"]} trials) ...')
        t_tune = time.time()
        try:
            best_params, best_auc = tune_xgboost(
                X_train, y_train,
                n_trials=CONFIG['optuna_n_trials'],
                cv=CONFIG['optuna_cv'],
                random_state=CONFIG['random_state'],
            )
            log_lines.append(
                f'  [{dataset_name.upper()}] Optuna Best AUC: {best_auc:.4f} '
                f'| params: {best_params}'
            )
            print(f'  Tuning time: {time.time() - t_tune:.1f}s')
        except Exception as e:
            print(f'  [Warning] Optuna failed ({e}), using fallback defaults')
            best_params = CONFIG['xgb_params'].copy()

        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        joblib.dump(model,       model_path)
        joblib.dump(best_params, params_path)
        print(f'  Model cached: {model_path}')
        log_lines.append(f'  [{dataset_name.upper()}] Model cached: {model_path}')

    # Test-set evaluation
    y_pred_test = model.predict(X_test)
    acc = (y_pred_test == y_test).mean()
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'\n  Test Accuracy: {acc:.4f} | AUC: {auc:.4f}')
    print(classification_report(y_test, y_pred_test, digits=3))
    log_lines.append(f'  [{dataset_name.upper()}] Accuracy: {acc:.4f} | AUC: {auc:.4f}')

    # ── Evaluation sample selection (stratified) ──────────────
    np.random.seed(CONFIG['random_state'])
    n      = min(eval_n, len(X_test))
    idx_0  = np.where(y_pred_test == 0)[0]
    idx_1  = np.where(y_pred_test == 1)[0]
    n0     = min(n // 2, len(idx_0))
    n1     = min(n - n0, len(idx_1))
    sel    = np.concatenate([
        np.random.choice(idx_0, n0, replace=False),
        np.random.choice(idx_1, n1, replace=False),
    ])
    np.random.shuffle(sel)
    X_eval = X_test[sel]
    y_eval = y_pred_test[sel]
    print(f'\n  Evaluation: {len(X_eval)} samples '
          f'(class 0: {n0}, class 1: {n1})')

    # ── DenSHAP pipeline ─────────────────────────────────────
    print(f'\n{"─"*40}')
    print(f'DenSHAP  (k_lof={k_lof})')
    print(f'{"─"*40}')

    denshap_pipe = DenSHAPPipeline(
        model=model,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        k_total=CONFIG['k_total'],
        k_lof=k_lof,
        p_low=CONFIG['p_low'],
        p_high=CONFIG['p_high'],
        temperature=CONFIG['temperature'],
        model_type='tree',
        top_k_list=CONFIG['top_k_list'],
        failure_penalty=CONFIG['failure_penalty'],
        random_state=CONFIG['random_state'],
    )

    # ── CF-SHAP baselines (shares LOF estimator) ──────────────
    print(f'\n{"─"*40}')
    print('CF-SHAP Baselines (4 methods)')
    print(f'{"─"*40}')
    t_cf0 = time.time()

    cfshap_eval = CFSHAPEvaluator(
        model=model,
        X_train=X_train,
        y_train=y_train,
        k_neighbors=CONFIG['k_neighbors'],
        model_type='tree',
        top_k_list=CONFIG['top_k_list'],
        failure_penalty=CONFIG['failure_penalty'],
        random_state=CONFIG['random_state'],
        lof_estimator=denshap_pipe.lof_est,
    )
    cfshap_results = cfshap_eval.evaluate(X_eval, y_eval, verbose=True)
    cfshap_summary = cfshap_eval.summary(cfshap_results)
    t_cfshap = time.time() - t_cf0
    print(f'\n[CF-SHAP] ({t_cfshap:.1f}s)')
    print(cfshap_summary.round(4).to_string())

    # ── DenSHAP evaluation ────────────────────────────────────
    print(f'\n{"─"*40}')
    print('DenSHAP Evaluation')
    print(f'{"─"*40}')
    t_ds0 = time.time()
    denshap_results = denshap_pipe.evaluate(X_eval, y_eval, verbose=True)
    denshap_summary = denshap_pipe.summary(denshap_results)
    t_denshap = time.time() - t_ds0
    print(f'\n[DenSHAP by LOF Group] ({t_denshap:.1f}s)')
    print(denshap_summary.round(4).to_string())

    log_lines.append(
        f'  [{dataset_name.upper()}] k_lof={k_lof} | '
        f'CF-SHAP: {t_cfshap:.1f}s | DenSHAP: {t_denshap:.1f}s'
    )

    # ── Final comparison table ────────────────────────────────
    print(f'\n{"─"*40}')
    print('Final Comparison Table')
    print(f'{"─"*40}')
    rows = []
    for method in ['SHAP_TRAIN', 'SHAP_D_LAB', 'SHAP_D_PRED', 'CF_SHAP']:
        row = {'Method': method}
        for k in CONFIG['top_k_list']:
            col = f'{method}_CA_top{k}'
            row[f'CA_top{k}'] = (cfshap_results[col].mean()
                                 if col in cfshap_results else np.nan)
        pl_col = f'{method}_plausibility'
        row['Plausibility'] = (cfshap_results[pl_col].mean()
                               if pl_col in cfshap_results else np.nan)
        bds_col = f'{method}_bds'
        row['BDS'] = (cfshap_results[bds_col].mean()
                      if bds_col in cfshap_results else np.nan)
        rows.append(row)

    denshap_row = {'Method': 'DenSHAP (ours)'}
    for k in CONFIG['top_k_list']:
        col = f'DenSHAP_CA_top{k}'
        denshap_row[f'CA_top{k}'] = (denshap_results[col].mean()
                                      if col in denshap_results else np.nan)
    denshap_row['Plausibility'] = denshap_results['DenSHAP_plausibility'].mean()
    denshap_row['BDS'] = (denshap_results['DenSHAP_bds'].mean()
                          if 'DenSHAP_bds' in denshap_results else np.nan)
    rows.append(denshap_row)

    comparison_df = pd.DataFrame(rows).set_index('Method')
    print(comparison_df.round(4).to_string())

    # ── LOF group comparison ──────────────────────────────────
    print(f'\n{"─"*40}')
    print('DenSHAP vs CF-SHAP by LOF Group')
    print(f'{"─"*40}')
    for group in ['Easy', 'Medium', 'Hard']:
        df_g = denshap_results[denshap_results['difficulty_group'] == group]
        if len(df_g) == 0:
            continue
        ds_ca1  = df_g['DenSHAP_CA_top1'].mean()
        cf_ca1  = cfshap_results.loc[df_g.index, 'CF_SHAP_CA_top1'].mean()
        ds_bds  = (df_g['DenSHAP_bds'].mean()
                   if 'DenSHAP_bds' in df_g.columns else float('nan'))
        cf_bds  = (cfshap_results.loc[df_g.index, 'CF_SHAP_bds'].mean()
                   if 'CF_SHAP_bds' in cfshap_results.columns else float('nan'))
        bds_imp = (ds_bds - cf_bds) / cf_bds * 100 if cf_bds > 0 else float('nan')
        print(f'\n  [{group}] N={len(df_g)}, α_mean={df_g["alpha"].mean():.3f}')
        print(f'    CA_top1 : CF-SHAP {cf_ca1:.4f} → DenSHAP {ds_ca1:.4f}')
        print(f'    BDS     : CF-SHAP {cf_bds:.4f} → DenSHAP {ds_bds:.4f} '
              f'({"↑" if bds_imp > 0 else "↓"}{abs(bds_imp):.2f}%)')

    # ── Validity / KNN success ────────────────────────────────
    print(f'\n{"─"*40}')
    print('Validity / KNN Success Rate by LOF Group')
    print(f'{"─"*40}')
    for group in ['Easy', 'Medium', 'Hard']:
        df_g = denshap_results[denshap_results['difficulty_group'] == group]
        if len(df_g) == 0:
            continue
        knn  = (df_g['knn_success'].mean()
                if 'knn_success' in df_g.columns else float('nan'))
        val  = (df_g['DenSHAP_validity'].mean()
                if 'DenSHAP_validity' in df_g.columns else float('nan'))
        lof  = (df_g['lof_score'].mean()
                if 'lof_score' in df_g.columns else float('nan'))
        print(f'  {group:<6}: KNN {knn:.1%} | Validity {val:.3f} | LOF {lof:.4f}')

    # ── BDS summary table ─────────────────────────────────────
    print(f'\n{"─"*40}')
    print('BDS (Background Density Score) — All Methods')
    print(f'{"─"*40}')
    bds_method_rows = []
    for method in ['SHAP_TRAIN', 'SHAP_D_LAB', 'SHAP_D_PRED', 'CF_SHAP']:
        col = f'{method}_bds'
        if col in cfshap_results.columns:
            bds_method_rows.append({
                'Method': method,
                'BDS': cfshap_results[col].mean(),
            })
    if 'DenSHAP_bds' in denshap_results.columns:
        bds_method_rows.append({
            'Method': 'DenSHAP (ours)',
            'BDS': denshap_results['DenSHAP_bds'].mean(),
        })
    if bds_method_rows:
        print(pd.DataFrame(bds_method_rows).set_index('Method').round(4).to_string())

    # ── Save results ──────────────────────────────────────────
    cfshap_results.to_csv(
        f'{output_dir}/{dataset_name}_cfshap.csv', index=False)
    denshap_results.to_csv(
        f'{output_dir}/{dataset_name}_denshap.csv', index=False)

    comparison_df.to_csv(f'{output_dir}/{dataset_name}_summary.csv')
    denshap_summary.to_csv(f'{output_dir}/{dataset_name}_group.csv')

    bds_rows = []
    for group in ['Easy', 'Medium', 'Hard']:
        df_g   = denshap_results[denshap_results['difficulty_group'] == group]
        if len(df_g) == 0:
            continue
        ds_bds = (df_g['DenSHAP_bds'].mean()
                  if 'DenSHAP_bds' in df_g.columns else np.nan)
        cf_bds = (cfshap_results.loc[df_g.index, 'CF_SHAP_bds'].mean()
                  if 'CF_SHAP_bds' in cfshap_results.columns else np.nan)
        delta  = (ds_bds - cf_bds) / cf_bds * 100 if cf_bds > 0 else np.nan
        bds_rows.append({
            'Dataset':      dataset_name.upper(),
            'Group':        group,
            'N':            len(df_g),
            'CF_SHAP_BDS':  round(cf_bds, 4),
            'DenSHAP_BDS':  round(ds_bds, 4),
            'Delta_pct':    round(delta, 4) if not np.isnan(delta) else np.nan,
            'Direction':    (f'{"↑" if delta > 0 else "↓"}{abs(delta):.2f}%'
                             if not np.isnan(delta) else 'N/A'),
        })
    pd.DataFrame(bds_rows).to_csv(
        f'{output_dir}/{dataset_name}_bds_by_group.csv', index=False)

    print(f'\n✅ [{dataset_name.upper()}] k_lof={k_lof} Results saved → {output_dir}/')
    log_lines.append(f'  [{dataset_name.upper()}] k_lof={k_lof} Saved → {output_dir}/')

    return comparison_df, denshap_summary


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='DenSHAP experiment runner')
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all', 'heloc', 'wine', 'lendingclub'],
        help='Dataset to run (default: all)',
    )
    parser.add_argument(
        '--eval_n', type=int, default=CONFIG['eval_n'],
        help=f'Number of evaluation samples (default: {CONFIG["eval_n"]})',
    )
    parser.add_argument(
        '--k_lof', type=int, nargs='+', default=DEFAULT_K_LOF_LIST,
        help=f'LOF neighbourhood size(s) (default: {DEFAULT_K_LOF_LIST}). '
             f'Pass one or more values, e.g. --k_lof 10 20 30 50',
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Base output directory (default: results). '
             'Each k_lof value gets its own subdirectory: results/klof{k}/',
    )
    args = parser.parse_args()

    k_lof_list = args.k_lof   # always a list thanks to nargs='+'
    datasets   = DATASETS if args.dataset == 'all' else [args.dataset]

    log_lines = [
        'DenSHAP Experiment Log',
        f'eval_n     = {args.eval_n}',
        f'k_lof list = {k_lof_list}',
        f'base output= {args.output}',
        '',
    ]

    t_total = time.time()
    all_summaries = {}   # { (dataset, k_lof): comparison_df }

    # ── Outer loop: k_lof values ──────────────────────────────
    for k_lof in k_lof_list:
        output_dir = os.path.join(args.output, f'klof{k_lof}')
        CONFIG['k_lof'] = k_lof

        print(f'\n{"#"*65}')
        print(f'  k_lof sweep: {k_lof}  →  saving to {output_dir}/')
        print(f'{"#"*65}')
        log_lines.append(f'\n=== k_lof={k_lof} | output={output_dir} ===')

        # ── Inner loop: datasets ───────────────────────────────
        for ds in datasets:
            try:
                comparison, group_summary = run_single_dataset(
                    dataset_name=ds,
                    eval_n=args.eval_n,
                    output_dir=output_dir,
                    k_lof=k_lof,
                    log_lines=log_lines,
                )
                all_summaries[(ds, k_lof)] = comparison
            except FileNotFoundError as e:
                print(f'\n[ERROR] Data file not found for {ds}: {e}')
                log_lines.append(f'[ERROR] {ds} k_lof={k_lof}: {e}')
            except Exception as e:
                print(f'\n[ERROR] {ds} k_lof={k_lof} failed: {e}')
                import traceback; traceback.print_exc()
                log_lines.append(f'[ERROR] {ds} k_lof={k_lof}: {e}')

    # ── Cross-k_lof BDS summary ───────────────────────────────
    if all_summaries:
        print(f'\n{"="*65}')
        print('All experiments complete — BDS summary across k_lof values')
        print(f'{"="*65}')
        for (ds, k_lof), df in all_summaries.items():
            print(f'\n  [{ds.upper()}] k_lof={k_lof}')
            if 'BDS' in df.columns:
                print(df['BDS'].to_string())

    total_time = time.time() - t_total
    log_lines.append(f'\nTotal wall-clock time: {total_time/3600:.2f} h')
    print(f'\nTotal time: {total_time/3600:.2f} h')

    # ── Save log ──────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, 'experiment_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f'Log saved: {log_path}')


if __name__ == '__main__':
    main()