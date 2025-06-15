import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Months to process: from December 2024 to May 2025 inclusive
MONTHS: List[Tuple[str, str]] = [
    ("2024", "12"),
    ("2025", "01"),
    ("2025", "02"),
    ("2025", "03"),
    ("2025", "04"),
    ("2025", "05"),
]

def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log.
    """
    ts: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def run_command(command: List[str]) -> None:
    """
    Run a subprocess command and exit if it fails.

    Args:
        command (List[str]): The command and arguments to run.
    """
    log(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        log(f"Command succeeded: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        log(f"Command failed ({e.returncode}): {' '.join(command)}")
        sys.exit(e.returncode)


def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.

    Args:
        path (Path): Directory path to ensure.
    """
    if not path.exists():
        log(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        log(f"Directory created: {path}")


def main() -> None:
    """
    Main setup script to prepare directories and execute
    data collection and machine learning pipeline steps.
    """
    # Ensure base directories exist
    ensure_directory(Path("data"))
    ensure_directory(Path("model"))

    # Check and fetch weather data files
    weather_dir: Path = Path("data/weather")
    ensure_directory(weather_dir)
    for year, month in MONTHS:
        weather_file = weather_dir / f"weather-{year}-{month}.csv"
        if not weather_file.exists():
            log(f"Missing weather file: {weather_file}")
            run_command([
                "uv", "run", "src/weather_scraper.py",
                "-y", year, "-m", month, "--report"
            ])
        else:
            log(f"Found weather file: {weather_file}")

    # Check and fetch valenbici data files
    valenbici_dir: Path = Path("data/valenbici")
    ensure_directory(valenbici_dir)
    for year, month in MONTHS:
        vb_file = valenbici_dir / f"valenbici_{year}-{month}.csv"
        if not vb_file.exists():
            log(f"Missing valenbici file: {vb_file}")
            run_command([
                "uv", "run", "src/extract_data_from_ceferra_repo.py",
                "--year", year, "--month", month
            ])
        else:
            log(f"Found valenbici file: {vb_file}")

    # Verify dataset and prepare data if needed
    dataset_file: Path = Path("data/dataset.csv")
    if dataset_file.exists():
        log(f"Dataset file already exists, skipping prepare_data4ml: {dataset_file}")
    else:
        log(f"Dataset file not found, running prepare_data4ml")
        run_command(["uv", "run", "src/prepare_data4ml.py"]);

    # Optional best parameter search
    best_params_file: Path = Path("model/best_params.json")
    if not best_params_file.exists():
        answer = input(
            "best_params.json not found. Run best_lgbm_optuna? (y/n): "
        ).strip().lower()
        if answer == "y":
            run_command(["uv", "run", "src/best_lgbm_optuna.py"])
        else:
            log("Skipping best_lgbm_optuna.")
    else:
        answer = input(
            "best_params.json already exists. Re-run best_lgbm_optuna? (y/n): "
        ).strip().lower()
        if answer == "y":
            run_command(["uv", "run", "src/best_lgbm_optuna.py"])
        else:
            log("Skipping best_lgbm_optuna.")

    # Ask before running train/test
    run_train: str = input(
        "Run train_test_lgbm? (y/n): "
    ).strip().lower()
    if run_train == "y":
        run_command(["uv", "run", "src/train_test_lgbm.py"])
    else:
        log("Skipping train_test_lgbm.")


if __name__ == "__main__":
    main()
