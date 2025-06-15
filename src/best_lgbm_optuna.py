import os
import json
from math import sqrt
from typing import Any, Dict, List, Tuple
from datetime import datetime
import config

import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log.
    """
    ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def load_and_sort(path: str) -> pd.DataFrame:
    """
    Load CSV data, parse timestamps, and sort by station and timestamp.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Sorted DataFrame with datetime index reset.
    """
    log(f"[INFO] Loading data from '{path}'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at '{path}'")
    df: pd.DataFrame = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    sorted_df: pd.DataFrame = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    log(f"[OK] Data loaded and sorted: {sorted_df.shape[0]} rows")
    return sorted_df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation and test sets based on calendar months.

    Hold out the last month as the test set.

    Args:
        df (pd.DataFrame): Full DataFrame with 'timestamp'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_val and test DataFrames.
    """
    log("[INFO] Splitting data into train/val and test sets")

    is_may_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 5)
    )
    test: pd.DataFrame = df[is_may_2025].copy()
    train_val: pd.DataFrame = df[~is_may_2025].copy()
    log(f"[OK] Split complete: train_val={train_val.shape[0]} rows, test={test.shape[0]} rows")
    return train_val, test


def sample_stations(
    df: pd.DataFrame,
    n_stations: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Randomly sample a subset of stations for tuning.

    Args:
        df (pd.DataFrame): DataFrame containing station data.
        n_stations (int): Number of unique stations to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Subset DataFrame for sampled stations.
    """
    log(f"[INFO] Sampling {n_stations} stations with seed={seed}")
    np.random.seed(seed)
    all_ids: np.ndarray = df['id'].unique()
    selected: np.ndarray = np.random.choice(all_ids, size=n_stations, replace=False)
    sampled_df: pd.DataFrame = df[df['id'].isin(selected)].copy()
    log(f"[OK] Sampled data contains {sampled_df.shape[0]} rows across {n_stations} stations")
    return sampled_df


def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """
    Identify feature columns, excluding timestamp and target.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target variable.

    Returns:
        List[str]: Names of feature columns.
    """
    ignore: List[str] = ['timestamp', target]
    features: List[str] = [col for col in df.columns if col not in ignore]
    log(f"[OK] Identified {len(features)} features")
    return features


def objective(
    trial: Trial,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series
) -> float:
    """
    Optuna objective: minimize mean CV RMSE using GroupKFold.

    Args:
        trial (Trial): Optuna trial object.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        groups (pd.Series): Group labels for cross-validation.

    Returns:
        float: Mean RMSE across folds.
    """
    params: Dict[str, Any] = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    rmse_list: List[float] = []
    cv = GroupKFold(n_splits=5)
    for train_idx, valid_idx in cv.split(X, y, groups):
        model = LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds: np.ndarray = model.predict(X.iloc[valid_idx])
        rmse: float = sqrt(mean_squared_error(y.iloc[valid_idx], preds))
        rmse_list.append(rmse)
    mean_rmse: float = float(np.mean(rmse_list))
    return mean_rmse


def main() -> None:
    """
    Main entry point: run hyperparameter tuning and save best parameters.
    """
    # Load and prepare data
    data_path: str = f"{config.DATA_PATH}/dataset.csv"
    df: pd.DataFrame = load_and_sort(data_path)
    train_val, _ = split_train_test(df)

    # Sample stations for tuning
    sampled: pd.DataFrame = sample_stations(train_val)

    # Prepare training inputs
    TARGET: str = 'bikes_available'
    FEATURES: List[str] = get_feature_columns(sampled, TARGET)
    X_tv: pd.DataFrame = sampled[FEATURES]
    y_tv: pd.Series = sampled[TARGET]
    groups_tv: pd.Series = sampled['id']

    log("[INFO] Setting up Optuna study")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    log("[INFO] Starting optimization (15 trials)")
    study.optimize(
        lambda trial: objective(trial, X_tv, y_tv, groups_tv),
        n_trials=15,
        show_progress_bar=True,
    )
    log(f"[OK] Optimization complete: Best RMSE={study.best_value:.4f}")

    # Save best parameters
    model_dir: str = config.MODEL_PATH
    os.makedirs(model_dir, exist_ok=True)
    best_params: Dict[str, Any] = study.best_params
    params_path: str = os.path.join(model_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    log(f"[OK] Best parameters saved to '{params_path}'")


if __name__ == '__main__':
    main()