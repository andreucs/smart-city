import os
import sys
import time
import math
from typing import List, Tuple
from datetime import datetime

import pandas as pd
from openrouteservice import Client
from dotenv import dotenv_values

# Maximum number of elements per request (free plan limit)
MAX_ELEMENTS: int = 3500


def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log.
    """
    ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def chunk_indices(n: int, chunk_size: int) -> List[range]:
    """
    Split range(n) into subranges of length <= chunk_size.

    Args:
        n (int): Total number of elements.
        chunk_size (int): Maximum size of each chunk.

    Returns:
        List[range]: List of range objects covering 0..n-1.
    """
    return [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def compute_matrices(
    coordinates: List[Tuple[float, float]],
    api_key: str
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute distance and duration matrices in batches using ORS API.

    Args:
        coordinates (List[Tuple[float, float]]): List of (lon, lat) pairs.
        api_key (str): OpenRouteService API key.

    Returns:
        Tuple[List[List[float]], List[List[float]]]:
            - distance_matrix: NxN matrix in meters.
            - duration_matrix: NxN matrix in seconds.
    """
    log("[INFO] Initializing ORS client and preparing batches")
    client = Client(key=api_key)
    n: int = len(coordinates)
    chunk_size: int = min(n, int(math.floor(math.sqrt(MAX_ELEMENTS))))
    src_ranges = chunk_indices(n, chunk_size)
    dst_ranges = src_ranges

    distance_matrix: List[List[float]] = [[0.0]*n for _ in range(n)]
    duration_matrix: List[List[float]] = [[0.0]*n for _ in range(n)]

    chunk_pairs = [(s, d) for s in src_ranges for d in dst_ranges]
    total: int = len(chunk_pairs)
    log(f"[INFO] Starting matrix computation in {total} batches")

    for idx, (src_range, dst_range) in enumerate(chunk_pairs, start=1):
        log(
            f"[INFO] Batch {idx}/{total}: src {src_range.start}-{src_range.stop-1}, "
            f"dst {dst_range.start}-{dst_range.stop-1}"
        )
        resp = client.distance_matrix(
            locations=coordinates,
            sources=list(src_range),
            destinations=list(dst_range),
            metrics=["distance", "duration"],
            profile="cycling-regular",
        )
        for i, si in enumerate(src_range):
            for j, dj in enumerate(dst_range):
                distance_matrix[si][dj] = resp["distances"][i][j]
                duration_matrix[si][dj] = resp["durations"][i][j]
        time.sleep(1)
    log("[OK] Matrix computation complete")
    return distance_matrix, duration_matrix


def compute_walking_durations(
    coordinates: List[Tuple[float, float]],
    api_key: str
) -> List[List[float]]:
    """
    Compute duration matrix (walking time) using ORS API.

    Args:
        coordinates (List[Tuple[float, float]]): List of (lon, lat) pairs.
        api_key (str): OpenRouteService API key.

    Returns:
        List[List[float]]: NxN matrix in seconds.
    """
    client = Client(key=api_key)
    n = len(coordinates)
    chunk_size = min(n, int(math.floor(math.sqrt(MAX_ELEMENTS))))
    src_ranges = chunk_indices(n, chunk_size)
    dst_ranges = src_ranges

    duration_matrix = [[0.0] * n for _ in range(n)]

    for src_range in src_ranges:
        for dst_range in dst_ranges:
            response = client.distance_matrix(
                locations=coordinates,
                sources=list(src_range),
                destinations=list(dst_range),
                metrics=["duration"],
                profile="foot-walking",
            )
            for i, si in enumerate(src_range):
                for j, dj in enumerate(dst_range):
                    duration_matrix[si][dj] = response["durations"][i][j]
            time.sleep(1)  # evitar rate limit

    return duration_matrix

def main() -> None:
    """
    Main orchestration: load stations, compute matrices, and save to CSV.
    """
    log("[INFO] Loading environment variables")
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    config = dotenv_values(env_path)
    api_key = config.get("ORS_API_KEY")
    if not api_key:
        log(f"[FAIL] ORS_API_KEY not found in {env_path}")
        sys.exit(1)
    log("[OK] API key loaded")

    log("[INFO] Reading station data from 'bike_stations.csv'")
    df = pd.read_csv("./data/bike_stations.csv", dtype={"id": str, "lat": float, "lon": float})
    df = df.sort_values("id").reset_index(drop=True)
    station_ids = df["id"].tolist()
    coords = list(zip(df["lon"].tolist(), df["lat"].tolist()))
    n = len(coords)
    log(f"[OK] Loaded {n} stations")

    distances, durations = compute_matrices(coords, api_key)

    log("[INFO] Saving distance matrix to 'distance_matrix.csv'")
    dist_df = pd.DataFrame(distances, index=station_ids, columns=station_ids)
    dist_df.index.name = "id"
    dist_df.reset_index().to_csv("distance_matrix.csv", index=False)
    log("[OK] distance_matrix.csv saved")

    log("[INFO] Saving duration matrix to 'duration_matrix.csv'")
    dur_df = pd.DataFrame(durations, index=station_ids, columns=station_ids)
    dur_df.index.name = "id"
    dur_df.reset_index().to_csv("duration_matrix.csv", index=False)
    log("[OK] duration_matrix.csv saved")

    durations_w = compute_walking_durations(coords, api_key)
    log("[INFO] Saving duration matrix to 'duration_matrix.csv'")
    dur_w_df = pd.DataFrame(durations_w, index=station_ids, columns=station_ids)
    dur_w_df.index.name = "id"
    dur_w_df.reset_index().to_csv("duration_w_matrix.csv", index=False)
    log("[OK] duration_w_matrix.csv saved")


    log("[OK] All matrices computed and saved successfully")


if __name__ == '__main__':
    main()