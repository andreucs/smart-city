import argparse
import calendar
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import config

import polars as pl
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log.
    """
    ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def start_driver() -> ChromeDriver:
    """
    Initialize and return a headless Chrome WebDriver with typical options.

    Returns:
        ChromeDriver: Configured Selenium Chrome WebDriver.
    """
    log("[INFO] Starting Chrome WebDriver")
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(options=options)
    log("[OK] WebDriver started")
    return driver


def accept_cookies(driver: ChromeDriver) -> None:
    """
    Attempt to click the cookie acceptance button if present.

    Args:
        driver (ChromeDriver): Selenium WebDriver instance.
    """
    log("[INFO] Accepting cookies if prompt exists")
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH,
                '/html/body/div[14]/div[2]/div[2]/div[2]/div[2]/button[1]'))
        )
        btn.click()
        log("[OK] Cookies accepted")
    except Exception:
        log("[INFO] No cookie prompt found or click failed")


def select_year(driver: ChromeDriver, year: int) -> None:
    """
    Click the button to select a specific year in the UI.

    Args:
        driver (ChromeDriver): Selenium WebDriver instance.
        year (int): Year to select on the page.
    """
    log(f"[INFO] Selecting year {year}")
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, f"button.gray-button[data-year=\"{year}\"]")
            )
        )
        driver.execute_script("arguments[0].scrollIntoView();", btn)
        btn.click()
        time.sleep(0.25)
        log(f"[OK] Year {year} selected")
    except Exception:
        log(f"[INFO] Year button for {year} not found or click failed")


def extract_data(
    driver: ChromeDriver,
    day: int,
    month: int,
    year: int,
    results: List[Dict[str, Any]],
) -> None:
    """
    Navigate to the weather page for a given day and extract the hourly table.

    Args:
        driver (ChromeDriver): Selenium WebDriver.
        day (int): Day of month.
        month (int): Month number.
        year (int): Year number.
        results (List[Dict[str, Any]]): Accumulated output list to append to.
    """
    url = (
        f"https://www.tiempo3.com/europe/spain/comunidad-valenciana/valencia"
        f"?page=past-weather#day={day}&month={month}"
    )
    log(f"[INFO] Loading URL: {url}")
    driver.get(url)
    driver.execute_script("document.body.style.zoom='0.1'")
    select_year(driver, year)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".day_table"))
        )
        rows = driver.find_elements(By.CSS_SELECTOR, ".day_table tbody tr")
        data: Dict[str, Any] = {"Day": day, "Month": month, "Year": year}
        for row in rows:
            key = row.find_element(By.TAG_NAME, "th").text
            vals = [cell.text for cell in row.find_elements(By.TAG_NAME, "td")]
            data[key] = vals
        if rows:
            results.append(data)
            log(f"[OK] Extracted data for {year}-{month:02d}-{day:02d}")
        else:
            log(f"[INFO] No data rows found for {year}-{month:02d}-{day:02d}")
    except Exception:
        log(f"[INFO] Failed to extract data for {year}-{month:02d}-{day:02d}")


def flatten_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Optional[str]]]:
    """
    Convert nested daily lists into flat hourly records.

    Args:
        results (List[Dict[str, Any]]): Raw extracted data.

    Returns:
        List[Dict[str, Optional[str]]]: Flattened list of hourly dicts.
    """
    flattened: List[Dict[str, Optional[str]]] = []
    for item in results:
        day = item.get("Day")
        month = item.get("Month")
        year = item.get("Year")
        # find number of hours
        n = next((len(v) for k, v in item.items() if isinstance(v, list)), 0)
        if n <= 0:
            continue
        for i in range(n):
            rec: Dict[str, Optional[str]] = {
                "Year": str(year),
                "Month": str(month),
                "Day": str(day),
                "Hour": str(i),
            }
            for k, v in item.items():
                if isinstance(v, list):
                    rec[k] = v[i] if i < len(v) else None
            flattened.append(rec)
    log(f"[OK] Flattened results to {len(flattened)} hourly records")
    return flattened


def save_csv(
    records: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Clean, convert types and write records to CSV via Polars.

    Args:
        records (List[Dict[str, Any]]): Flat hourly data.
        output_path (str): Path to save the CSV.
    """
    if not records:
        log("[FAIL] No data to save.")
        return
    flat = flatten_results(records)
    if not flat:
        log("[FAIL] Flattened data empty.")
        return
    df = pl.from_records(flat)
    df_clean = df.select([
        pl.col("Year").cast(pl.Int32).alias("year"),
        pl.col("Month").cast(pl.Int32).alias("month"),
        pl.col("Day").cast(pl.Int32).alias("day"),
        pl.col("Hour").cast(pl.Int32).alias("hour"),
        pl.col("Temperatura").str.replace_all(",", ".").str.replace_all(r"[^\d\.]", "").cast(pl.Float64).alias("temperature_celsius"),
        pl.col("Precipitaciones").str.replace_all(r"[^\d\.]", "").cast(pl.Float64).alias("precipitation_mm"),
        pl.col("Humedad").str.replace_all("%", "").cast(pl.Float64).alias("humidity_percent"),
        pl.col("Velocidad del viento").str.replace_all(r"[^\d\.]", "").cast(pl.Float64).alias("wind_speed_kmh"),
    ])
    df_clean.write_csv(output_path)
    log(f"[OK] CSV saved to '{output_path}' ({df_clean.height} rows)")


def generate_report(
    records: List[Dict[str, Any]],
    year: int,
    month: int,
    directory: str,
) -> bool:
    """
    Generate a text report of missing days or hours and save it.

    Args:
        records (List[Dict[str, Any]]): Raw extracted data.
        year (int): Year number.
        month (int): Month number.
        directory (str): Directory to save report.

    Returns:
        bool: True if report created, False if no missing data.
    """
    log("[INFO] Generating missing data report")
    days = calendar.monthrange(year, month)[1]
    today = datetime.now()
    missing_days: List[int] = []
    for d in range(1, days+1):
        date = datetime(year, month, d)
        if date > today:
            break
        if not any(r.get("Day") == d for r in records):
            missing_days.append(d)
    missing_hours: Dict[int, List[str]] = {}
    for item in records:
        d = item.get("Day")
        n = next((len(v) for k, v in item.items() if isinstance(v, list)), 0)
        if n <= 0:
            continue
        hours = [str(i) for i in range(n)]
        miss: List[str] = []
        for k, v in item.items():
            if isinstance(v, list):
                miss.extend(
                    hours[i] for i, val in enumerate(v) if not val or val.strip() in ('', '-')
                )
        if miss:
            missing_hours[d] = sorted(set(miss))
    if not missing_days and not missing_hours:
        log("[OK] No missing data detected")
        return False
    report_file = os.path.join(directory, f"missing_data-{year}-{month:02d}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        for d in missing_days:
            f.write(f"Day {d}: no data\n")
        for d, hrs in missing_hours.items():
            f.write(f"Day {d}: missing hours: {', '.join(hrs)}\n")
    log(f"[OK] Missing data report saved to '{report_file}'")
    return True


def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI arguments for year, month, output directory and report flag.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Scrape historical weather data for a given year and month."
    )
    parser.add_argument('-y', '--year', type=int, required=True, help='Year to scrape')
    parser.add_argument('-m', '--month', type=int, required=True, choices=range(1,13), help='Month to scrape')
    parser.add_argument('-r', '--report', action='store_true', help='Generate missing data report')
    args = parser.parse_args()
    return args


def format_elapsed(elapsed: timedelta) -> str:
    """
    Format a timedelta into a human-readable elapsed time string.

    Args:
        elapsed (timedelta): Duration to format.

    Returns:
        str: Formatted string like '1 hours, 2 minutes, 3 seconds'.
    """
    secs = int(elapsed.total_seconds())
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    secs = secs % 60
    parts: List[str] = []
    if hrs:
        parts.append(f"{hrs} hours")
    if mins:
        parts.append(f"{mins} minutes")
    parts.append(f"{secs} seconds")
    return ", ".join(parts)


def main() -> None:
    """
    Main orchestration: parse args, scrape each day, save CSV and optional report.
    """
    start = time.monotonic()
    args = parse_arguments()
    year: int = args.year
    month: int = args.month
    directory: str = config.DATA_WEATHER_PATH
    do_report: bool = args.report

    log(f"[INFO] Starting scrape for {year}-{month:02d}")
    os.makedirs(directory, exist_ok=True)

    driver = start_driver()
    driver.get("https://www.tiempo3.com/europe/spain/comunidad-valenciana/valencia?page=past-weather")
    accept_cookies(driver)

    today = datetime.now()
    days = calendar.monthrange(year, month)[1]
    results: List[Dict[str, Any]] = []
    for d in range(1, days+1):
        date = datetime(year, month, d)
        if date > today:
            break
        log(f"[INFO] Processing date {date.strftime('%Y-%m-%d')}")
        extract_data(driver, d, month, year, results)

    driver.quit()
    log("[OK] WebDriver closed")

    output_file = os.path.join(directory, f"weather-{year}-{month:02d}.csv")
    save_csv(results, output_file)

    if do_report:
        had = generate_report(results, year, month, directory)
        if not had:
            log("[OK] No missing data report needed")

    elapsed = timedelta(seconds=int(time.monotonic() - start))
    log(f"[OK] Finished in {format_elapsed(elapsed)}")


if __name__ == '__main__':
    main()
