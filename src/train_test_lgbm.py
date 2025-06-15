import os
import json
from math import sqrt
from typing import Any, Dict, List, Tuple
import config

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    Load CSV, parse timestamps, and sort by station and time.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Sorted DataFrame with a datetime timestamp column.
    """
    log(f"[INFO] Loading data from '{path}'")
    df: pd.DataFrame = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    sorted_df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    log(f"[OK] Data loaded and sorted: {sorted_df.shape[0]} rows")
    return sorted_df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test based on calendar months.

    Hold out the last month as test set.

    Args:
        df (pd.DataFrame): Full dataset with 'timestamp' field.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    log("[INFO] Splitting data into train and test sets")

    is_may_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 5)
    )

    test: pd.DataFrame = df[is_may_2025].copy()
    train: pd.DataFrame = df[~is_may_2025].copy()
    log(f"[OK] Split complete: train={train.shape[0]} rows, test={test.shape[0]} rows")
    return train, test


def get_feature_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    """
    Return feature column names, ignoring timestamp and target.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target variable.

    Returns:
        List[str]: List of feature column names.
    """
    ignore: Tuple[str, str] = ('timestamp', target_column)
    features: List[str] = [col for col in df.columns if col not in ignore]
    log(f"[OK] Identified {len(features)} feature columns")
    return features


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true (pd.Series): True target values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: Dictionary with MAE, MSE, RMSE, and R2.
    """
    mae: float = mean_absolute_error(y_true, y_pred)
    mse: float = mean_squared_error(y_true, y_pred)
    rmse: float = sqrt(mse)
    r2: float = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def main() -> None:
    """
    Main entry point: orchestrates model training, evaluation, and saving.
    """
    # Paths and constants
    data_path: str = f"{config.DATA_PATH}/dataset.csv"
    model_dir: str = f"{config.MODEL_PATH}"
    os.makedirs(model_dir, exist_ok=True)

    # Load and split data
    df: pd.DataFrame = load_and_sort(data_path)
    train_df, test_df = split_train_test(df)
    TARGET: str = 'bikes_available'
    FEATURES: List[str] = get_feature_columns(train_df, TARGET)

    # Load best parameters
    params_path: str = os.path.join(model_dir, 'best_params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Best parameters file not found at '{params_path}'")
    log(f"[INFO] Loading best parameters from '{params_path}'")
    with open(params_path, 'r') as f:
        best_params: Dict[str, Any] = json.load(f)
    log(f"[OK] Best parameters loaded: {best_params}")

    # Train and evaluate on test set
    log("[INFO] Training model on train set")
    model: LGBMRegressor = LGBMRegressor(
        **best_params,
        objective='regression',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    model.fit(X_train, y_train)
    log("[OK] Model training complete")

    log("[INFO] Evaluating model on test set")
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    preds: np.ndarray = model.predict(X_test)
    metrics: Dict[str, float] = evaluate_model(y_test, preds)
    log(f"[OK] Test metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # Save retrained model
    retrained_path: str = os.path.join(model_dir, 'retrained_model.pkl')
    joblib.dump(model, retrained_path)
    log(f"[OK] Retrained model saved to '{retrained_path}'")

    # Train on full dataset for final model
    log("[INFO] Training final model on full dataset")
    full_X, full_y = df[FEATURES], df[TARGET]
    final_model: LGBMRegressor = LGBMRegressor(
        **best_params,
        objective='regression',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(full_X, full_y)
    final_path: str = os.path.join(model_dir, 'final_model.pkl')
    joblib.dump(final_model, final_path)
    log(f"[OK] Final model saved to '{final_path}'")


if __name__ == '__main__':
    main()