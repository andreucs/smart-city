import os
import io
import calendar
import argparse
import time
from datetime import datetime, date, timedelta
from typing import List, Tuple
import config

import requests
import zipfile
import pandas as pd

BASE_URL: str = "https://raw.githubusercontent.com/ceferra/valenbici/master"


def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log.
    """
    ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def process_month(
    year: int,
    month: int,
    output_folder: str
) -> None:
    """
    Download, verify, process, and save Valenbici data for a specific month.

    Steps:
      1. Download daily ZIP archives from GitHub.
      2. Verify each day contains 96 CSV snapshots; track missing/incomplete.
      3. Extract, clean, and enrich each CSV into DataFrames.
      4. Save a missing days report if applicable.
      5. Concatenate all data and write a monthly CSV.

    Args:
        year (int): Year of the data.
        month (int): Month of the data (1-12).
        output_folder (str): Directory to save outputs.
    """
    os.makedirs(output_folder, exist_ok=True)
    log(f"[INFO] Processing data for {year}-{month:02d}")

    data_frames: List[pd.DataFrame] = []
    missing_days: List[Tuple[date, str]] = []

    total_days: int = calendar.monthrange(year, month)[1]

    for day in range(1, total_days + 1):
        current_date: date = date(year, month, day)
        zip_date: date = current_date + timedelta(days=1)
        zip_name: str = f"{zip_date.day:02d}-{zip_date.month:02d}-{zip_date.year}.zip"
        url: str = f"{BASE_URL}/{zip_name}"

        log(f"[INFO] Downloading {zip_name} for {current_date.isoformat()}")
        try:
            response = requests.get(url, stream=True)
        except Exception as e:
            log(f"[FAIL] Download error for {zip_name}: {e}")
            missing_days.append((current_date, "download_error"))
            continue

        if response.status_code != 200:
            log(f"[FAIL] HTTP {response.status_code} for {zip_name}")
            missing_days.append((current_date, f"http_{response.status_code}"))
            continue

        try:
            archive = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            log(f"[FAIL] Invalid ZIP for {zip_name}")
            missing_days.append((current_date, "bad_zip"))
            continue

        csv_files = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(csv_files) != 96:
            log(f"[WARN] Incomplete day {current_date.isoformat()}: {len(csv_files)}/96 files")
            missing_days.append((current_date, f"{len(csv_files)}_files"))
        else:
            log(f"[OK] Complete day {current_date.isoformat()} (96/96 CSV files)")

        for csv_name in csv_files:
            base, _ = os.path.splitext(csv_name)
            parts: List[str] = base.split('_')
            if len(parts) == 2:
                date_str, time_str = parts
                fmt: str = "%d-%m-%Y_%H-%M"
            elif len(parts) == 3 and parts[0] == 'valenbici':
                _, date_str, time_str = parts
                fmt = "%d-%m-%Y_%H-%M-%S"
            else:
                log(f"[INFO] Skipping unrecognized file name {csv_name}")
                continue

            try:
                dt: datetime = datetime.strptime(f"{date_str}_{time_str}", fmt).replace(second=0)
            except ValueError:
                log(f"[INFO] Timestamp parse failed for {csv_name}")
                continue

            with archive.open(csv_name) as fh:
                try:
                    df: pd.DataFrame = pd.read_csv(fh, sep=';')
                except Exception as e:
                    log(f"[FAIL] Read error in {csv_name}: {e}")
                    continue

            # Clean and transform
            df = df.drop(columns=[
                'Direccion', 'Activo', 'Espacios_libres',
                'Espacios_totales', 'ticket', 'geo_shape', 'update_jcd'
            ], errors='ignore')
            df = df.rename(columns={
                'Numero': 'id',
                'Bicis_disponibles': 'bikes_available'
            })
            
            # Split geo_point_2d
            if 'geo_point_2d' in df.columns:
                if df['geo_point_2d'].notna().any():
                    coords = df['geo_point_2d'].str.split(',', expand=True)
                    if coords.shape[1] == 2:
                        try:
                            df['lat'] = coords[0].astype(float)
                            df['lon'] = coords[1].astype(float)
                            df = df.drop(columns='geo_point_2d')
                        except Exception as e: continue
                    else:
                        sample = df['geo_point_2d'].dropna().head(3).tolist()
                        continue
                else: continue

            # Override update timestamp
            df = df.drop(columns=['fecha_actualizacion'], errors='ignore')
            
            # Timestamp columns
            df['timestamp'] = dt
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute

            data_frames.append(df)

    # Save missing days report
    report_path: str = os.path.join(output_folder, f"missing_days_{year:04d}-{month:02d}.txt")
    if missing_days:
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(f"Missing or incomplete days for {year:04d}-{month:02d}\n")
            for d, reason in missing_days:
                report_file.write(f"{d.isoformat()} -> {reason}\n")
        log(f"[OK] Missing days report saved to '{report_path}'")
    else:
        log("[OK] All days complete; no report generated")

    if not data_frames:
        log("[FAIL] No data available for the requested month. Exiting.")
        return

    # Concatenate and save monthly CSV
    log("[INFO] Concatenating daily data frames")
    monthly_df: pd.DataFrame = pd.concat(data_frames, ignore_index=True)
    output_file: str = os.path.join(output_folder, f"valenbici_{year:04d}-{month:02d}.csv")
    monthly_df.to_csv(output_file, index=False)
    log(f"[OK] Monthly CSV saved to '{output_file}' ({monthly_df.shape[0]} rows)")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for year, month, and output folder.

    Returns:
        argparse.Namespace: Parsed arguments with attributes 'year', 'month', 'output'.
    """
    parser = argparse.ArgumentParser(
        description="Download and process Valenbici data for a given year and month."
    )
    parser.add_argument(
        '--year', type=int, required=True,
        help='Year to process (e.g., 2025)'
    )
    parser.add_argument(
        '--month', type=int, required=True, choices=list(range(1, 13)),
        help='Month to process (1-12)'
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function: orchestrates argument parsing and month processing.
    """
    start_time = time.monotonic()
    args = parse_arguments()
    year: int = args.year
    month: int = args.month
    output_folder: str = config.DATA_VALENBICI_PATH

    process_month(year, month, output_folder)

    elapsed = timedelta(seconds=int(time.monotonic() - start_time))
    log(f"[OK] Finished processing in {elapsed}")


if __name__ == '__main__':
    main()
