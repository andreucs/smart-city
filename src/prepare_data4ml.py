import os
import glob
from typing import Dict, Tuple, List
import config

import pandas as pd
import numpy as np
from datetime import datetime


def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log.
    """
    ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def process_bike_files(folder: str) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Read and preprocess bike availability CSV files from a folder.

    Steps:
        1. Read all files matching 'valenbici_*.csv' in the folder.
        2. Concatenate and sort by station id and timestamp.
        3. For each station:
            a. Reindex to a complete 15-minute interval range.
            b. Count missing intervals.
            c. Interpolate missing bike availability values in time.
            d. Forward/backward fill static fields (lat, lon).
        4. Return the filled DataFrame and missing counts per station.

    Args:
        folder (str): Path to the directory containing bike CSV files.

    Returns:
        Tuple[pd.DataFrame, Dict[int, int]]:
            - bike_df: DataFrame with complete 15-minute intervals and imputed values.
            - missing_counts: Dict mapping station id to number of originally missing intervals.
    """
    log(f"[INFO] Reading bike files from '{folder}'")
    pattern: str = os.path.join(folder, "valenbici_*.csv")
    file_paths: List[str] = glob.glob(pattern)
    if not file_paths:
        raise FileNotFoundError(
            f"No bike files in {folder} matching 'valenbici_*.csv'."
        )
    log(f"[OK] Found {len(file_paths)} bike files")

    # Read and combine CSV files into a single DataFrame
    dfs: List[pd.DataFrame] = [pd.read_csv(fp, parse_dates=["timestamp"]) for fp in file_paths]
    combined: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["id", "timestamp"], inplace=True)
    log("[OK] Combined and sorted bike data")

    missing_counts: Dict[int, int] = {}
    filled_list: List[pd.DataFrame] = []

    # Process each station separately
    for station_id, group in combined.groupby("id"):
        # Ensure proper indexing by timestamp
        group = group.set_index("timestamp").sort_index()
        # Create complete 15-min interval index
        idx: pd.DatetimeIndex = pd.date_range(
            start=group.index.min(), end=group.index.max(), freq="15min"
        )
        reindexed: pd.DataFrame = group.reindex(idx)

        # Count original missing intervals
        missing_counts[station_id] = int(reindexed["bikes_available"].isna().sum())

        # Interpolate bikes_available over time, then round to integer
        reindexed["bikes_available"] = (
            reindexed["bikes_available"]
            .interpolate(method="time")
            .round()
            .astype(int)
        )

        # Fill static station information
        reindexed["id"] = station_id
        reindexed[["lat", "lon"]] = (
            reindexed[["lat", "lon"]].ffill().bfill()
        )

        # Reset index to restore timestamp column
        reindexed.reset_index(inplace=True)
        reindexed.rename(columns={"index": "timestamp"}, inplace=True)
        filled_list.append(reindexed)

    # Concatenate all stations back together
    bike_df: pd.DataFrame = pd.concat(filled_list, ignore_index=True)
    bike_df.sort_values(["id", "timestamp"], inplace=True)
    bike_df.reset_index(drop=True, inplace=True)

    log("[OK] Bike data processing complete\n")
    return bike_df, missing_counts


def process_weather_files(folder: str) -> pd.DataFrame:
    """
    Read and preprocess weather CSV files from a folder.

    Steps:
        1. Read all files matching 'weather-*.csv'.
        2. Concatenate and sort by year, month, day, hour.
        3. Construct a timestamp at the start of each hour.

    Args:
        folder (str): Path to the directory containing weather CSV files.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', weather variables...].
    """
    log(f"[INFO] Reading weather files from '{folder}'")
    pattern: str = os.path.join(folder, "weather-*.csv")
    file_paths: List[str] = glob.glob(pattern)
    if not file_paths:
        raise FileNotFoundError(
            f"No weather files in {folder} matching 'weather-*.csv'."
        )
    log(f"[OK] Found {len(file_paths)} weather files")

    dfs: List[pd.DataFrame] = [pd.read_csv(fp) for fp in file_paths]
    combined: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["year", "month", "day", "hour"], inplace=True)
    log("[OK] Combined and sorted weather data")

    # Build hourly timestamp
    combined["timestamp"] = pd.to_datetime(
        dict(
            year=combined["year"],
            month=combined["month"],
            day=combined["day"],
            hour=combined["hour"],
        )
    )
    log("[OK] Weather data processing complete\n")

    # Return only relevant columns
    return combined[
        ["timestamp", "temperature_celsius", "precipitation_mm", 
         "humidity_percent", "wind_speed_kmh"]
    ]


def merge_and_impute(
    bike_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge bike availability and weather data on timestamp and impute missing weather values.

    Args:
        bike_df (pd.DataFrame): Preprocessed bike DataFrame with 'timestamp'.
        weather_df (pd.DataFrame): Preprocessed weather DataFrame with 'timestamp'.

    Returns:
        pd.DataFrame: Merged DataFrame with no missing weather values per station.
    """
    log("[INFO] Merging bike and weather data")
    # Left merge on timestamp; weather data is hourly
    merged: pd.DataFrame = pd.merge(
        bike_df, weather_df, on="timestamp", how="left"
    ).set_index("timestamp")

    # List of weather variables to fill
    weather_vars: List[str] = [
        "temperature_celsius",
        "precipitation_mm",
        "humidity_percent",
        "wind_speed_kmh",
    ]

    # Time-based interpolation per station
    for var in weather_vars:
        merged[var] = (
            merged.groupby("id")[var]
            .transform(lambda x: x.interpolate(method="time").ffill().bfill())
        )
    log("[OK] Weather imputation complete\n")

    merged.reset_index(inplace=True)
    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and lag-based features for modeling.

    Features added:
        - weekday (0=Mon, 6=Sun)
        - is_weekend (binary)
        - month_sin, month_cos (cyclical encoding)
        - lag_k (availability k*15min ago)
        - roll_mean_k (rolling mean over past k intervals)
    """
    log("[INFO] Adding time and lag features")
    
    # 1) Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 2) Recompute month from timestamp (garantiza valores completos)
    df["month"] = df["timestamp"].dt.month
    
    df["hour"] = df["timestamp"].dt.hour
    
    # 3) Weekday and weekend flag
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    
    # 4) Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # 5) Drop raw time columns
    df.drop(columns=["year", "month", "day", 
                     #"hour", 
                     "minute"], errors="ignore", inplace=True)
    
    # 6) Sort and group for lags
    df.sort_values(["id", "timestamp"], inplace=True)
    grp = df.groupby("id")
    
    # 7) Lag features at 15m, 1h, 2h, 3h, 6h
    lags = [1, 4, 8, 12, 24]
    for lag in lags:
        df[f"lag_{lag}"] = grp["bikes_available"].shift(lag)
    log(f"[OK] Generated lag features: {lags}")
    
    # 8) Rolling means 1h, 2h
    windows = [4, 8]
    for win in windows:
        df[f"roll_mean_{win}"] = (
            grp["bikes_available"]
            .shift(1)
            .rolling(win)
            .mean()
            .reset_index(level=0, drop=True)
        )
    log(f"[OK] Generated rolling mean features: {windows}")
    
    # 9) Drop rows with any NaN in lag or rolling mean features
    df.dropna(subset=[f"lag_{lag}" for lag in lags]
                      + [f"roll_mean_{w}" for w in windows]
                      + ["month_sin", "month_cos"],
              inplace=True)
    log("[OK] Feature engineering complete\n")
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate that data intervals are consistent and no missing/duplicate rows exist.

    Checks:
        - All timestamp diffs per station equal 15 minutes.
        - No missing values across DataFrame.
        - No duplicate (id, timestamp) pairs.

    Args:
        df (pd.DataFrame): DataFrame to validate.
    """
    log("[INFO] Validating data integrity")

    # Check time interval consistency
    df_sorted: pd.DataFrame = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    diffs: pd.Series = df_sorted.groupby("id")["timestamp"].diff().dropna()
    invalid_intervals: int = int((diffs != pd.Timedelta(minutes=15)).sum())

    # Check for any missing values
    missing_vals: int = int(df.isna().sum().sum())

    # Check for duplicates
    duplicates: int = int(df.duplicated(subset=["id", "timestamp"]).sum())

    log(f"[OK] Intervals != 15min: {invalid_intervals}")
    log(f"[OK] Missing values: {missing_vals}")
    log(f"[OK] Duplicate rows: {duplicates}")

    if invalid_intervals + missing_vals + duplicates == 0:
        log("[OK] Validation passed: data is clean and regular\n")
    else:
        log("[FAIL] Validation failed: please review above issues\n")


def main() -> None:
    """
    Main entry point: orchestrates data processing, feature engineering, validation, and saving.
    """
    bike_folder: str = config.DATA_VALENBICI_PATH
    weather_folder: str = config.DATA_WEATHER_PATH

    # Process bike and weather datasets
    bike_df, _ = process_bike_files(bike_folder)
    weather_df = process_weather_files(weather_folder)

    # Merge and impute
    merged: pd.DataFrame = merge_and_impute(bike_df, weather_df)

    # Feature engineering
    featured: pd.DataFrame = add_features(merged)

    # Data validation
    validate_data(featured)

    # Save final DataFrame
    log("[INFO] Saving final DataFrame with features")
    output_path: str = f"{config.DATA_PATH}/dataset.csv"
    featured.to_csv(output_path, index=False)
    log(f"[OK] Features saved to '{output_path}'")


if __name__ == "__main__":
    main()