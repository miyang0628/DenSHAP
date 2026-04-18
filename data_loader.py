"""
Data loading and preprocessing module (rewritten based on actual column structure)
=============================================================
Actual data structure:
    Wine        : includes target column; drops quality/type; uses remaining 11 features
    LendingClub : includes target column; uses all numeric features as-is
    HELOC       : includes target column; uses remaining 23 features

General principles:
    - Use 'target' column as y (already encoded)
    - feature_names always returned as pure str list
    - Column names: spaces → '_' (DiCE compatibility)
    - No StandardScaler (raw values preserved for DiCE speed)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def _sanitize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names: replace spaces/special chars with '_' for DiCE compatibility."""
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    return df


def _split_and_return(X, y, feature_names, test_size=TEST_SIZE):
    feature_names = [str(f) for f in feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f'  Train: {X_train.shape} | Test: {X_test.shape}')
    unique, counts = np.unique(y_train, return_counts=True)
    print(f'  Label distribution (train): {dict(zip(unique.tolist(), counts.tolist()))}') 
    print(f'  Minority class ratio: {y_train.mean():.3f}')
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────
# Wine Quality
# ─────────────────────────────────────────────

def load_wine(path: str = 'data/wine.csv'):
    """
    Actual columns: fixed acidity, volatile acidity, citric acid,
               residual sugar, chlorides, free sulfur dioxide,
               total sulfur dioxide, density, pH, sulphates,
               alcohol, quality, type, target

    Processing:
        - target column → y
        - Drop quality, type columns (redundant / categorical)
        - Use remaining 11 numeric features
        - No scaling (raw values preserved)
    """
    print('[Wine Quality] loading...')
    df = pd.read_csv(path, sep=None, engine='python')
    df = _sanitize_colnames(df)
    print(f'  Columns: {list(df.columns)}')

    # Extract target column
    y = df['target'].values.astype(int)

    # Columns to drop: target, quality, type (if present)
    drop_cols = [c for c in ['target', 'quality', 'type'] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].values.astype(np.float64)

    print(f'  Features used ({len(feat_cols)} features): {feat_cols}')
    print(f'  Total samples: {len(df)}')
    return _split_and_return(X, y, feat_cols)


# ─────────────────────────────────────────────
# LendingClub
# ─────────────────────────────────────────────

def load_lendingclub(path: str = 'data/lendingclub.csv', sample_n: int = 50000):
    """
    Actual columns: annual_inc, application_type, dti, earliest_cr_line,
               emp_length, grade, home_ownership, installment, int_rate,
               loan_amnt, mort_acc, open_acc, pub_rec, pub_rec_bankruptcies,
               revol_bal, revol_util, sub_grade, term, total_acc,
               verification_status, target

    Processing:
        - target column → y
        - Use all remaining features (already numerically encoded)
        - Sample sample_n instances for large datasets
        - No StandardScaler (raw values preserved)
    """
    print('[LendingClub] loading...')
    df = pd.read_csv(path, low_memory=False)
    df = _sanitize_colnames(df)
    print(f'  Original size: {df.shape}')

    # Extract target column
    if 'target' not in df.columns:
        raise ValueError("'target' column not found. Available columns: " + str(list(df.columns)))
    y = df['target'].values.astype(int)

    feat_cols = [c for c in df.columns if c != 'target']

    # Handle missing values
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))

    # Balanced 50:50 sampling to handle class imbalance
    # LendingClub imbalance (1:3.7) causes too few minority samples
    # Prevents KNN opposite-label search failure
    if sample_n and len(df) > sample_n:
        n_per_class = sample_n // 2
        df_min = df[df['target'] == df['target'].value_counts().idxmin()]
        df_maj = df[df['target'] == df['target'].value_counts().idxmax()]
        n_min = min(n_per_class, len(df_min))
        n_maj = min(n_per_class, len(df_maj))
        df_sampled = pd.concat([
            df_min.sample(n=n_min, random_state=RANDOM_STATE),
            df_maj.sample(n=n_maj, random_state=RANDOM_STATE)
        ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        y = df_sampled['target'].values.astype(int)
        X = df_sampled[feat_cols].values.astype(np.float64)
        print(f'  After balanced sampling: {df_sampled.shape} '
              f'(class 0: {n_maj}, class 1: {n_min})')
    else:
        X = df[feat_cols].values.astype(np.float64)


    print(f'  Features used ({len(feat_cols)} features): {feat_cols}')
    return _split_and_return(X, y, feat_cols)


# ─────────────────────────────────────────────
# HELOC
# ─────────────────────────────────────────────

def load_heloc(path: str = 'data/heloc.csv'):
    """
    Actual columns: AverageMInFile, ExternalRiskEstimate, ...
               (23 features) + target

    Processing:
        - target column → y
        - Special values -7/-8/-9 → NaN → replaced with median
        - Use remaining 23 features
        - No StandardScaler (raw values preserved)
    """
    print('[HELOC] loading...')
    df = pd.read_csv(path)
    df = _sanitize_colnames(df)
    print(f'  Number of columns: {len(df.columns)}')

    # Extract target column
    if 'target' not in df.columns:
        # Handle RiskPerformance column if present
        if 'RiskPerformance' in df.columns:
            df['target'] = (df['RiskPerformance'] == 'Good').astype(int)
            df = df.drop(columns=['RiskPerformance'])
        else:
            raise ValueError("'target' column not found. Columns: " + str(list(df.columns)))

    y = df['target'].values.astype(int)
    feat_cols = [c for c in df.columns if c != 'target']

    # Handle special missing values
    for col in feat_cols:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col] = df[col].replace([-7, -8, -9], np.nan)
            df[col] = df[col].fillna(df[col].median())

    X = df[feat_cols].values.astype(np.float64)

    print(f'  Features used ({len(feat_cols)} features): {feat_cols[:5]} ...')
    print(f'  Total samples: {len(df)}')
    return _split_and_return(X, y, feat_cols)


# ─────────────────────────────────────────────
# Unified dataset loader
# ─────────────────────────────────────────────

def load_dataset(name: str, data_dir: str = 'data'):
    loaders = {
        'heloc':       (load_heloc,       f'{data_dir}/heloc.csv'),
        'lendingclub': (load_lendingclub,  f'{data_dir}/lendingclub.csv'),
        'wine':        (load_wine,         f'{data_dir}/wine.csv'),
    }
    if name not in loaders:
        raise ValueError(f"Unsupported dataset: {name}. Choose from: {list(loaders.keys())}")
    fn, path = loaders[name]
    return fn(path)