"""
DenSHAP vs CF-SHAP — Main Experiment Script on Real Datasets
=================================================
Usage (py files and data/ folder must be in the same directory):

    python run_experiment.py                     # all 3 datasets
    python run_experiment.py --dataset heloc     # specific dataset only
    python run_experiment.py --dataset wine --eval_n 50

Outputs:
    results/
        {dataset}_cfshap.csv     : per-sample results for 4 CF-SHAP baselines
        {dataset}_dhace.csv      : per-sample DenSHAP results
        {dataset}_summary.csv    : method-level aggregated summary (paper Table format)
        {dataset}_group.csv      : LOF group-level aggregation (evidence for paper claim)
        experiment_log.txt       : full experiment log
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
from denshap import DenSHAPPipeline  # v3: LOF-weighted KNN only

# ── Experiment Configuration ──────────────────────────
CONFIG = {
    # General
    'random_state':  42,
    'eval_n':        50,       # number of evaluation samples (999999 = full test set)
    'top_k_list':    [1, 2, 3, 5],
    'failure_penalty': 10.0,
    'lendingclub_sample_n': None,  # None = use all / integer = stratified sample size

    # Model (auto-optimized by Optuna — xgb_params overwritten after tuning)
    'optuna_n_trials': 30,   # increase to 50~100 if time allows
    'optuna_cv':       3,    # number of cross-validation folds
    'xgb_params': dict(      # fallback defaults if Optuna fails
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
    'k_total':       100,
    'k_lof':         20,
    'p_low':         25.0,
    'p_high':        75.0,
    # DiCE removed (v3) — LOF-weighted KNN only
    'n_candidates_multiplier': 5,  # Candidate pool = k_total × 5
    # 'proj_neighbors': 20,  # removed in v3
    'temperature':   1.0,
}

def tune_xgboost(X_train, y_train, n_trials=30, cv=3, random_state=42):
    """
    Optimize XGBoost hyperparameters using Optuna.
    Objective: Stratified K-Fold AUC (handles class imbalance).

    Paper description:
        "To isolate the effect of background dataset construction from model performance,
         using Optuna (Akiba et al., 2019) for each dataset,
         with individually optimized XGBoost hyperparameters."
    """
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
        model = XGBClassifier(**params)
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv_obj, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({'eval_metric': 'logloss',
                        'random_state': random_state,
                        'verbosity': 0})

    print(f'  Optuna done | Best AUC: {study.best_value:.4f}')
    print(f'  Best params: {best_params}')
    return best_params, study.best_value


DATASETS = ['heloc', 'wine', 'lendingclub']


def run_single_dataset(dataset_name: str, eval_n: int, output_dir: str, log_lines: list):

    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  Dataset: {dataset_name.upper()}')
    print(f'{sep}')

    # ── Data loading ──────────────────────────
    # LendingClub uses separate sample_n (None = use all)
    if dataset_name == 'lendingclub':
        from data_loader import load_lendingclub
        sample_n = CONFIG.get('lendingclub_sample_n', 50000)
        X_train, X_test, y_train, y_test, feature_names = load_lendingclub(
            'data/lendingclub.csv', sample_n=sample_n
        )
    else:
        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name)

    # ── Model training (load saved model first, otherwise run Optuna tuning) ──
    os.makedirs(output_dir, exist_ok=True)
    model_path  = os.path.join(output_dir, f'{dataset_name}_model.joblib')
    params_path = os.path.join(output_dir, f'{dataset_name}_best_params.joblib')

    if os.path.exists(model_path) and os.path.exists(params_path):
        print(f'\n[Model] Loading saved model: {model_path}')
        model       = joblib.load(model_path)
        best_params = joblib.load(params_path)
        print(f'  Load complete | params: {best_params}')
        log_lines.append(f'  [{dataset_name.upper()}] Loaded saved model')
    else:
        print(f'\n[Model] Optuna XGBoost tuning ({CONFIG["optuna_n_trials"]} trials)...')
        t_tune = time.time()
        try:
            best_params, best_auc = tune_xgboost(
                X_train, y_train,
                n_trials=CONFIG['optuna_n_trials'],
                cv=CONFIG['optuna_cv'],
                random_state=CONFIG['random_state']
            )
            model = XGBClassifier(**best_params)
            print(f'  Tuning time: {time.time()-t_tune:.1f}s | Best CV AUC: {best_auc:.4f}')
            log_lines.append(f'  Optuna Best AUC: {best_auc:.4f} | params: {best_params}')
        except Exception as e:
            print(f'  [Warning] Optuna failed ({e}), using default parameters')
            best_params = CONFIG['xgb_params']
            model = XGBClassifier(**best_params)

        model.fit(X_train, y_train)

        # Save model and parameters
        joblib.dump(model,       model_path)
        joblib.dump(best_params, params_path)
        print(f'  Model saved: {model_path}')
        log_lines.append(f'  Model saved: {model_path}')

    y_pred_test = model.predict(X_test)
    acc = (y_pred_test == y_test).mean()
    print(f'  Test accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred_test, digits=3))

    log_lines.append(f'\n[{dataset_name.upper()}] Accuracy: {acc:.4f}')

    # Select evaluation samples:
    # Include minority class samples to ensure Hard Case coverage
    np.random.seed(CONFIG['random_state'])
    eval_n = min(eval_n, len(X_test))

    # Stratified sampling by class
    idx_0 = np.where(y_pred_test == 0)[0]
    idx_1 = np.where(y_pred_test == 1)[0]
    n0 = min(eval_n // 2, len(idx_0))
    n1 = min(eval_n - n0, len(idx_1))
    sel_0 = np.random.choice(idx_0, size=n0, replace=False)
    sel_1 = np.random.choice(idx_1, size=n1, replace=False)
    sel_idx = np.concatenate([sel_0, sel_1])
    np.random.shuffle(sel_idx)

    X_eval      = X_test[sel_idx]
    y_eval_pred = y_pred_test[sel_idx]
    print(f'\nEvaluation samples: {len(X_eval)} '
          f'(class 0: {n0}, class 1: {n1})')

    # ── DenSHAP initialization (LOF estimator shared with CF-SHAP) ──
    print(f'\n{"─"*40}')
    print('DenSHAP Pipeline Evaluation')
    print(f'{"─"*40}')
    t0 = time.time()

    denshap_pipe = DenSHAPPipeline(
        model=model,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        k_total=CONFIG['k_total'],
        k_lof=CONFIG['k_lof'],
        p_low=CONFIG['p_low'],
        p_high=CONFIG['p_high'],
    # DiCE removed (v3) — LOF-weighted KNN only
        temperature=CONFIG['temperature'],
        model_type='tree',
        top_k_list=CONFIG['top_k_list'],
        failure_penalty=CONFIG['failure_penalty'],
        random_state=CONFIG['random_state'],
    )
    # ── CF-SHAP Baseline (sharing LOF estimator with DenSHAP) ──
    print(f'\n{"─"*40}')
    print('CF-SHAP Baseline Evaluation (4 methods)')
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
        lof_estimator=denshap_pipe.lof_est,  # Using same LOF estimator as DenSHAP
    )
    cfshap_results = cfshap_eval.evaluate(X_eval, y_eval_pred, verbose=True)
    cfshap_summary = cfshap_eval.summary(cfshap_results)
    t_cfshap = time.time() - t_cf0

    print(f'\n[CF-SHAP Results] ({t_cfshap:.1f}s)')
    print(cfshap_summary.round(4).to_string())

    print(f'\n{"─"*40}')
    print('Run DenSHAP evaluation')
    print(f'{"─"*40}')
    denshap_results = denshap_pipe.evaluate(X_eval, y_eval_pred, verbose=True)
    denshap_summary = denshap_pipe.summary(denshap_results)
    t_dhace = time.time() - t0

    print(f'\n[DenSHAP Results by LOF Group] ({t_dhace:.1f}s)')
    print(denshap_summary.round(4).to_string())

    # ── Final comparison table ──────────────────
    print(f'\n{"─"*40}')
    print('Final Comparison Table (paper format)')
    print(f'{"─"*40}')

    rows = []
    for method in ['SHAP_TRAIN', 'SHAP_D_LAB', 'SHAP_D_PRED', 'CF_SHAP']:
        row = {'Method': method}
        for k in CONFIG['top_k_list']:
            col = f'{method}_CA_top{k}'
            row[f'CA_top{k}'] = cfshap_results[col].mean() if col in cfshap_results else np.nan
        pl_col = f'{method}_plausibility'
        row['Plausibility'] = cfshap_results[pl_col].mean() if pl_col in cfshap_results else np.nan
        rows.append(row)

    # DenSHAP overall average
    dhace_row = {'Method': 'DenSHAP (ours)'}
    for k in CONFIG['top_k_list']:
        col = f'DenSHAP_CA_top{k}'
        dhace_row[f'CA_top{k}'] = denshap_results[col].mean() if col in denshap_results else np.nan
    dhace_row['Plausibility'] = denshap_results['DenSHAP_plausibility'].mean()
    rows.append(dhace_row)

    comparison_df = pd.DataFrame(rows).set_index('Method')
    print(comparison_df.round(4).to_string())

    # ── LOF group comparison (evidence for paper claim) ──
    print(f'\n{"─"*40}')
    print('DenSHAP vs CF-SHAP comparison by LOF group')
    print('(Larger Hard group improvement strengthens the paper claim)')
    print(f'{"─"*40}')

    for group in ['Easy', 'Medium', 'Hard']:
        df_g = denshap_results[denshap_results['difficulty_group'] == group]
        if len(df_g) == 0:
            continue

        dhace_ca1  = df_g['DenSHAP_CA_top1'].mean()
        dhace_pl   = df_g['DenSHAP_plausibility'].mean()
        cfshap_ca1 = cfshap_results.loc[df_g.index, 'CF_SHAP_CA_top1'].mean()
        cfshap_pl  = cfshap_results.loc[df_g.index, 'CF_SHAP_plausibility'].mean()

        ca_improvement = (cfshap_ca1 - dhace_ca1) / cfshap_ca1 * 100
        pl_improvement = (cfshap_pl - dhace_pl) / cfshap_pl * 100

        print(f'\n  [{group}] N={len(df_g)}, α_mean={df_g["alpha"].mean():.3f}')
        print(f'    CA_top1   : CF-SHAP {cfshap_ca1:.4f} → DenSHAP {dhace_ca1:.4f} '
              f'({"↓" if ca_improvement > 0 else "↑"}{abs(ca_improvement):.1f}%)')
        print(f'    Plausibility: CF-SHAP {cfshap_pl:.4f} → DenSHAP {dhace_pl:.4f} '
              f'({"↓" if pl_improvement > 0 else "↑"}{abs(pl_improvement):.1f}%)')


    # ── KNN Success Rate / Validity Summary (v3) ──────
    print(f'\n{"─"*40}')
    print('KNN Success Rate / Validity (by group)')
    print(f'{"─"*40}')
    for group in ['Easy', 'Medium', 'Hard']:
        df_g = denshap_results[denshap_results['difficulty_group'] == group]
        if len(df_g) == 0:
            continue
        knn_ok   = df_g['knn_success'].mean() if 'knn_success' in df_g.columns else float('nan')
        validity = df_g['DenSHAP_validity'].mean() if 'DenSHAP_validity' in df_g.columns else float('nan')
        lof_mean = df_g['lof_score'].mean() if 'lof_score' in df_g.columns else float('nan')
        print(f'  {group:<6}: KNN {knn_ok:.1%} | Validity {validity:.3f} | LOF {lof_mean:.4f}')


    # ── BDS comparison (novel metric proposed by DenSHAP) ──────
    print(f'\n{"─"*40}')
    print('BDS (Background Density Score) Comparison')
    print('Higher = background data from denser regions — DenSHAP structural advantage')
    print(f'{"─"*40}')
    bds_rows = []
    for method in ['SHAP_TRAIN', 'SHAP_D_LAB', 'SHAP_D_PRED', 'CF_SHAP']:
        col = f'{method}_bds'
        if col in cfshap_results.columns:
            bds_rows.append({'Method': method, 'BDS': cfshap_results[col].mean()})
    if 'DenSHAP_bds' in denshap_results.columns:
        bds_rows.append({'Method': 'DenSHAP (ours)', 'BDS': denshap_results['DenSHAP_bds'].mean()})
    if bds_rows:
        bds_df = pd.DataFrame(bds_rows).set_index('Method')
        print(bds_df.round(4).to_string())
    print(f'\n  [BDS by group]')
    for group in ['Easy', 'Medium', 'Hard']:
        df_g = denshap_results[denshap_results['difficulty_group'] == group]
        if len(df_g) == 0:
            continue
        dhace_bds  = df_g['DenSHAP_bds'].mean() if 'DenSHAP_bds' in df_g.columns else float('nan')
        cfshap_col = 'CF_SHAP_bds'
        cfshap_bds = cfshap_results.loc[df_g.index, cfshap_col].mean() if cfshap_col in cfshap_results.columns else float('nan')
        improvement = (dhace_bds - cfshap_bds) / cfshap_bds * 100 if cfshap_bds > 0 else float('nan')
        sign = '↑' if improvement > 0 else '↓'
        print(f'  {group:<6}: CF-SHAP {cfshap_bds:.4f} → DenSHAP {dhace_bds:.4f} ({sign}{abs(improvement):.1f}%)')

    # ── Save results ────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    cfshap_results.to_csv(f'{output_dir}/{dataset_name}_cfshap.csv', index=False)
    denshap_results.to_csv(f'{output_dir}/{dataset_name}_dhace.csv', index=False)
    comparison_df.to_csv(f'{output_dir}/{dataset_name}_summary.csv')
    denshap_summary.to_csv(f'{output_dir}/{dataset_name}_group.csv')

    log_lines.append(f'  CF-SHAP time: {t_cfshap:.1f}s | DenSHAP time: {t_dhace:.1f}s')
    log_lines.append(f'  Save results: {output_dir}/{dataset_name}_*.csv')

    print(f'\n✅ [{dataset_name.upper()}] done. Results → {output_dir}/')
    return comparison_df, denshap_summary


# ── Main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DenSHAP experiment runner')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'heloc', 'lendingclub', 'wine'],
                        help='dataset to run (default: all)')
    parser.add_argument('--eval_n', type=int, default=CONFIG['eval_n'],
                        help=f'number of evaluation samples (default: {CONFIG["eval_n"]})')
    parser.add_argument('--output', type=str, default='results',
                        help='output directory for results (default: results)')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    log_lines = [f'DenSHAP Experiment Log', f'eval_n={args.eval_n}', '']

    all_summaries = {}
    for ds in datasets:
        try:
            comparison, group_summary = run_single_dataset(
                ds, args.eval_n, args.output, log_lines
            )
            all_summaries[ds] = comparison
        except FileNotFoundError as e:
            print(f'\n[ERROR] Data file not found for {ds}: {e}')
            log_lines.append(f'[ERROR] {ds}: {e}')
            continue
        except Exception as e:
            print(f'\n[ERROR] {ds} Experiment failed: {e}')
            import traceback; traceback.print_exc()
            log_lines.append(f'[ERROR] {ds}: {e}')
            continue

    # Print overall summary
    if all_summaries:
        print(f'\n{"="*65}')
        print('All experiments completed — CA_top1 summary by dataset')
        print(f'{"="*65}')
        for ds, df in all_summaries.items():
            print(f'\n  [{ds.upper()}]')
            if 'CA_top1' in df.columns:
                print(df['CA_top1'].to_string())

    # Save log
    os.makedirs(args.output, exist_ok=True)
    with open(f'{args.output}/experiment_log.txt', 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'\nLog saved: {args.output}/experiment_log.txt')


if __name__ == '__main__':
    main()