import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar
from typing import List
import src.config #este
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib import rcParams
from tqdm import tqdm
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Roboto', 'Helvetica', 'Arial']


def plot_target_time_series(
    df: pd.DataFrame,
    month: int,
    station_ids: List[int]
) -> None:
    """
    Plot stacked time series of bikes available for specified stations in a given month,
    using Roboto font, annotating the single-step maximum increase, 
    and saving the figure in high resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - 'timestamp' (datetime or parsable string)
        - 'id' (station identifier)
        - 'bikes_available' (int/float)
    month : int
        Month number (1 = January … 12 = December) to filter by.
    station_ids : List[int]
        Exactly three station IDs to plot.

    Returns
    -------
    None
        Displays a Matplotlib figure with three stacked subplots and
        saves it as a PNG at 300 DPI.
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Filter data
    df = df.loc[
        (df['timestamp'].dt.month == month) &
        df['id'].isin(station_ids)
    ]

    # Set up figure
    n = len(station_ids)
    fig, axes = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(12, 3 * n),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Loop through each station
    for ax, sid in zip(axes, station_ids):
        station_df = df[df['id'] == sid].sort_values('timestamp')
        x = station_df['timestamp']
        y = station_df['bikes_available']
        ax.plot(x, y, linewidth=1.5, label=f"Station {sid}")

        ax.set_title(f"Station {sid}", loc='left', fontweight='bold', fontsize=12)
        ax.set_ylabel("Bikes Available", fontsize=10)

    # Format x-axis on bottom subplot
    bottom = axes[-1]
    bottom.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(bottom.get_xticklabels(), rotation=0, ha='center')

    # Super-title
    month_name = calendar.month_name[month]
    fig.suptitle(
        f"Valenbici: Bike Availability Trends in {month_name}",
        fontsize=16, fontweight='bold'
    )

    # Save high-res figure
    filename = f"../figures/valenbici_bikes_{month_name.lower()}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_wheather_time_series(
    df: pd.DataFrame,
    month: int
) -> None:
    """
    Plot stacked time series of key weather variables for a given month,
    and save the figure in high resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - 'timestamp' (datetime or parsable string)
        - 'temperature_celsius'
        - 'precipitation_mm'
        - 'humidity_percent'
        - 'wind_speed_kmh'
    month : int
        Month number (1 = January … 12 = December) to filter by.

    Returns
    -------
    None
        Displays a Matplotlib figure with four stacked subplots and
        saves it as a PNG at 300 DPI.
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Filter data for the given month
    df_filtered = df.loc[df['timestamp'].dt.month == month]

    # List of weather variables to plot
    variables = [
        "temperature_celsius",
        "precipitation_mm",
        "humidity_percent",
        "wind_speed_kmh"
    ]

    # Prepare figure with one subplot per variable
    n = len(variables)
    fig, axes = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(12, 3 * n),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Loop through each variable and plot
    for ax, var in zip(axes, variables):
        series = df_filtered.sort_values('timestamp')[['timestamp', var]]
        ax.plot(
            series['timestamp'],
            series[var],
            linewidth=1.5,
            label=var.replace('_', ' ').title()
        )
        ax.set_title(
            var.replace('_', ' ').title(),
            loc='left',
            fontweight='bold',
            fontsize=12
        )
        ax.set_ylabel(var.replace('_', ' ').title(), fontsize=10)

    # Format x-axis on the bottom subplot
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(bottom_ax.get_xticklabels(), rotation=0, ha='center')

    # Super-title
    month_name = calendar.month_name[month]
    fig.suptitle(
        f"Valenbici: Weather Time Series in {month_name}",
        fontsize=16,
        fontweight='bold'
    )

    # Save high-res figure
    filename = f"../figures/valenbici_weather_{month_name.lower()}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_test_time_series(
    df: pd.DataFrame,
    station_ids: List[int]
) -> None:
    """
    Plot actual vs. predicted bikes_available for specified stations during May 2025,
    and save the figure in high resolution.

    This function:
      1. Filters the DataFrame to May 2025.
      2. Loads a LightGBM regressor from 'model/retrained_model.pkl'.
      3. Runs inference on the test set.
      4. Creates one stacked subplot per station ID, showing real vs. predicted series.
      5. Shows the legend only on the first subplot.
      6. Uses a matplotlib qualitative colormap for two distinct, professional colors.
      7. Saves the resulting figure as a PNG at 300 DPI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'timestamp' (datetime or parsable string)
          - 'id' (station identifier)
          - 'bikes_available' (target)
          - all feature columns used by the model, including 'id'
    station_ids : List[int]
        List of station IDs to plot.

    Returns
    -------
    None
        Displays the figure and saves it to disk.
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Filter to May 2025
    is_may_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 5)
    )
    mayo = df.loc[is_may_2025].copy()

    # Load the retrained model
    # model_path = os.path.join('model', 'retrained_model.pkl')
    model = joblib.load(f"../model/retrained_model.pkl")

    # Prepare features and run inference
    TARGET = 'bikes_available'
    feature_cols = [c for c in mayo.columns if c not in ('timestamp', TARGET)]
    X_test = mayo[feature_cols]
    mayo['predicted'] = model.predict(X_test)

    # Choose two professional colors from a matplotlib qualitative colormap
    cmap = plt.get_cmap('Set1')
    actual_color    = cmap(0)
    predicted_color = cmap(1)

    # Set up the figure
    n = len(station_ids)
    fig, axes = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(12, 3 * n),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Plot each station
    for idx, (ax, sid) in enumerate(zip(axes, station_ids)):
        sd = mayo[mayo['id'] == sid].sort_values('timestamp')
        x = sd['timestamp']
        y_true = sd[TARGET]
        y_pred = sd['predicted']

        ax.plot(x, y_true,   label='Actual',   linewidth=1.5, color=actual_color)
        ax.plot(x, y_pred,   label='Predicted', linestyle='--', linewidth=1.5, color=predicted_color)

        # Title
        ax.set_title(f"Station {sid}", loc='left', fontweight='bold', fontsize=12)

        # Legend only on first subplot
        if idx == 0:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.10),
                ncol=2,
                frameon=False,
                fontsize=9
            )

        ax.set_ylabel("Bikes Available", fontsize=10)

    # Format x-axis on bottom subplot
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(bottom_ax.get_xticklabels(), rotation=0, ha='center')

    # Super-title
    fig.suptitle(
        "Valenbici: Actual vs Predicted Bikes Available — May 2025",
        fontsize=16,
        fontweight='bold'
    )

    # Save high-res figure
    station_str = "_".join(str(s) for s in station_ids)
    filename = f"../figures/valenbici_test_may2025_stations_{station_str}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    # Show on screen
    plt.show()


def plot_metrics_by_hour(df: pd.DataFrame) -> None:
    """
    Compute and plot hourly error metrics (MAE, MSE, RMSE) for May 2025,
    and save the figure in high resolution.

    This function:
      1. Filters the DataFrame to May 2025.
      2. Loads a LightGBM regressor from 'model/retrained_model.pkl'.
      3. Runs inference on the test set.
      4. Groups predictions by hour of day and computes MAE, MSE, RMSE.
      5. Displays three bar charts: MAE, MSE, RMSE by hour, labeling only every 4 hours.
      6. Saves the resulting figure as a PNG at 300 DPI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'timestamp' (datetime or parsable string)
          - 'bikes_available' (target)
          - all feature columns used by the model, including 'id'
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Filter to May 2025
    mask = (df['timestamp'].dt.year == 2025) & (df['timestamp'].dt.month == 5)
    mayo = df.loc[mask].copy()

    # Load the retrained model
    # model_path = os.path.join('model', 'retrained_model.pkl')
    model = joblib.load(f"../model/retrained_model.pkl")

    # Prepare features and predict
    TARGET = 'bikes_available'
    feature_cols = [c for c in mayo.columns if c not in ('timestamp', TARGET)]
    X_test = mayo[feature_cols]
    mayo['predicted'] = model.predict(X_test)

    # Extract hour of day
    mayo['hour'] = mayo['timestamp'].dt.hour

    # Compute metrics by hour
    metrics = {'MAE': [], 'MSE': [], 'RMSE': []}
    hours = list(range(24))
    for hr in hours:
        grp = mayo[mayo['hour'] == hr]
        y_true = grp[TARGET]
        y_pred = grp['predicted']
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)

    # Choose colors from a qualitative colormap
    cmap = plt.get_cmap('Set1')
    colors = {'MAE': cmap(1), 'MSE': cmap(2), 'RMSE': cmap(3)}

    # Plot bar charts
    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(12, 10),
        constrained_layout=True
    )
    for ax, metric_name in zip(axes, ['MAE', 'MSE', 'RMSE']):
        ax.bar(hours, metrics[metric_name], color=colors[metric_name], width=0.8)
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(f"{metric_name} by Hour of Day", loc='left', fontsize=12, fontweight='bold')
        # ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Format x-axis on bottom subplot: label every 4 hours only
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(MultipleLocator(4))
    bottom_ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val):02d}:00"))
    bottom_ax.set_xlabel("Hour of Day", fontsize=10, fontweight='bold')

    # Super-title
    fig.suptitle(
        "Valenbici: Hourly Error Metrics — May 2025",
        fontsize=16,
        fontweight='bold'
    )

    # Save high-res figure
    filename = f"../figures/valenbici_hourly_metrics_may2025.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    # Show on screen
    plt.show()


def create_time_columns(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['timestamp'].dt.day
    df['minute'] = df['timestamp'].dt.minute
    df['year'] = df['timestamp'].dt.year
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    return df

def plot_naive_forecast(
    df: pd.DataFrame,
    station_ids: List[int]
) -> None:
    """
    Plot actual vs. predicted bikes_available for specified stations during May 2025,
    and save the figure in high resolution.

    This function:
      1. Filters the DataFrame to May 2025.
      2. Prepares a naive forecast based on April 2025 data.
      3. Runs inference on the test set.
      4. Creates one stacked subplot per station ID, showing real vs. predicted series.
      5. Shows the legend only on the first subplot.
      6. Uses a matplotlib qualitative colormap for two distinct, professional colors.
      7. Saves the resulting figure as a PNG at 300 DPI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'timestamp' (datetime or parsable string)
          - 'id' (station identifier)
          - 'bikes_available' (target)
          - all feature columns used by the model, including 'id'
    station_ids : List[int]
        List of station IDs to plot.

    Returns
    -------
    None
        Displays the figure and saves it to disk.
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df = create_time_columns(df)
    
    # Filter to April 2025
    is_april_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 4)
    )

    is_may_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 5)
    )

    abril = df.loc[is_april_2025].copy()
    mayo = df.loc[is_may_2025].copy()

    # Change the name of the bikes column in April so it doesn't repeat
    abril = abril.rename(columns={'bikes_available': 'bikes_april'})
    abril_dict = abril.set_index(['id', 'day', 'hour', 'minute'])['bikes_april'].to_dict()

    # Progress bar with tqdm
    tqdm.pandas() 
    def lookup_forecast(row):
        key = (row['id'], row['day'], row['hour'], row['minute'])
        return abril_dict.get(key, None)

    # Prediction
    mayo['forecast_naive'] = mayo.progress_apply(lookup_forecast, axis=1)

    # Choose two professional colors from a matplotlib qualitative colormap
    cmap = plt.get_cmap('Set1')
    actual_color    = cmap(0)
    predicted_color = cmap(1)

    # Set up the figure
    n = len(station_ids)
    fig, axes = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(12, 3 * n),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Plot each station
    for idx, (ax, sid) in enumerate(zip(axes, station_ids)):
        sd = mayo[mayo['id'] == sid].sort_values('timestamp')
        x = sd['timestamp']
        y_true = sd['bikes_available']
        y_pred = sd['forecast_naive']

        ax.plot(x, y_true,   label='Actual',   linewidth=1.5, color=actual_color)
        ax.plot(x, y_pred,   label='Predicted', linestyle='--', linewidth=1.5, color=predicted_color)

        # Title
        ax.set_title(f"Station {sid}", loc='left', fontweight='bold', fontsize=12)

        # Legend only on first subplot
        if idx == 0:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.10),
                ncol=2,
                frameon=False,
                fontsize=9
            )

        ax.set_ylabel("Bikes Available", fontsize=10)

    # Format x-axis on bottom subplot
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(bottom_ax.get_xticklabels(), rotation=0, ha='center')

    # Super-title
    fig.suptitle(
        "Valenbici: Actual vs Predicted Bikes Available — May 2025",
        fontsize=16,
        fontweight='bold'
    )

    # Save high-res figure
    station_str = "_".join(str(s) for s in station_ids)
    filename = f"../figures/valenbici_test_may2025_stations_naive_forecast{station_str}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    # Show on screen
    plt.show()



def plot_metrics_by_hour_naive_forecast(df: pd.DataFrame) -> None:
    """
    Compute and plot hourly error metrics (MAE, MSE, RMSE) for May 2025,
    and save the figure in high resolution.

    This function:
      1. Filters the DataFrame to May 2025.
      2. Prepares a naive forecast based on April 2025 data.
      3. Runs inference on the test set.
      4. Groups predictions by hour of day and computes MAE, MSE, RMSE.
      5. Displays three bar charts: MAE, MSE, RMSE by hour, labeling only every 4 hours.
      6. Saves the resulting figure as a PNG at 300 DPI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'timestamp' (datetime or parsable string)
          - 'bikes_available' (target)
          - all feature columns used by the model, including 'id'
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')


    df = create_time_columns(df)
    
    # Filter to April 2025
    is_april_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 4)
    )

    is_may_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 5)
    )

    abril = df.loc[is_april_2025].copy()
    mayo = df.loc[is_may_2025].copy()

    # Change the name of the bikes column in April so it doesn't repeat
    abril = abril.rename(columns={'bikes_available': 'bikes_april'})
    abril_dict = abril.set_index(['id', 'day', 'hour', 'minute'])['bikes_april'].to_dict()

    # Progress bar with tqdm
    tqdm.pandas()  
    def lookup_forecast(row):
        key = (row['id'], row['day'], row['hour'], row['minute'])
        return abril_dict.get(key, None)

    # Prediction
    mayo['forecast_naive'] = mayo.progress_apply(lookup_forecast, axis=1)
    mayo_filtered = mayo[mayo['day'] != 31]


    # Compute metrics by hour
    metrics = {'MAE': [], 'MSE': [], 'RMSE': []}
    hours = list(range(24))
    for hr in hours:
        grp = mayo_filtered[mayo_filtered['hour'] == hr]
        y_true = grp['bikes_available']
        y_pred = grp['forecast_naive']
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)

    # Choose colors from a qualitative colormap
    cmap = plt.get_cmap('Set1')
    colors = {'MAE': cmap(1), 'MSE': cmap(2), 'RMSE': cmap(3)}

    # Plot bar charts
    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(12, 10),
        constrained_layout=True
    )
    for ax, metric_name in zip(axes, ['MAE', 'MSE', 'RMSE']):
        ax.bar(hours, metrics[metric_name], color=colors[metric_name], width=0.8)
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(f"{metric_name} by Hour of Day", loc='left', fontsize=12, fontweight='bold')
        # ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Format x-axis on bottom subplot: label every 4 hours only
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(MultipleLocator(4))
    bottom_ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val):02d}:00"))
    bottom_ax.set_xlabel("Hour of Day", fontsize=10, fontweight='bold')

    # Super-title
    fig.suptitle(
        "Valenbici: Hourly Error Metrics — May 2025",
        fontsize=16,
        fontweight='bold'
    )

    # Save high-res figure
    filename = f"../figures/valenbici_hourly_metrics_may2025_naive_forecast.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    # Show on screen
    plt.show()



def plot_pdp_or_ice(
    df: pd.DataFrame,
    station_ids: list = [83, 178, 112],
    choice: str = "pdp",
    ft: list = ["temperature_celsius", "precipitation_mm", "weekday", "wind_speed_kmh", "hour", "is_weekend"]
    #station_ids: List[int]
) -> None:
    """
    Plot actual vs. predicted bikes_available for specified stations during May 2025,
    and save the figure in high resolution.

    This function:
      1. Filters the DataFrame to May 2025.
      2. Loads a LightGBM regressor from 'model/retrained_model.pkl'.
      3. Runs inference on the test set.
      4. Creates the partial dependece or ICE plots for the specified features.
      5. Saves the resulting figure as a PNG at 300 DPI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'timestamp' (datetime or parsable string)
          - 'id' (station identifier)
          - 'bikes_available' (target)
          - all feature columns used by the model, including 'id'
    station_ids : List[int]
        List of station IDs to plot.

    Returns
    -------
    None
        Displays the figure and saves it to disk.
    """

    if choice == "pdp":
        choice_n = 'average'
    elif choice == "ice":
        choice_n = 'individual'
    else:
        choice_n = 'both'

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Filter to May 2025
    is_may_2025 = (
        (df['timestamp'].dt.year  == 2025) &
        (df['timestamp'].dt.month == 5)
    )

    resto = df.loc[~is_may_2025].copy()
    resto2 = resto[resto["id"].isin(station_ids)]
    # Load the retrained model
    model = joblib.load(f"../model/retrained_model.pkl")

    # Prepare features and run inference
    y = 'bikes_available'
    feature_cols = [c for c in resto2.columns if c not in ('timestamp', y)]
    X_train = resto2[feature_cols]

    # Config characteristics of the plot
    common_params = {
    "subsample": 500, #uses a subsample of the data to speed up the computation
    "grid_resolution": 50,
    "n_jobs": -1,  # uses all available cores
    "random_state": 0,
    "grid_resolution": 100 
}
    categorical_features = ['weekday', 'is_weekend']
    ft_c = [f for f in ft if isinstance(f, str) and f in categorical_features]
    # ft_c = [f for f in feature_cols if f in categorical_features and len(f) == 1]  # selecciona las variables categóricas

    if len(ft_c) == 0 or choice_n != 'average':
        ft_c = None

    features_info = {
    'features': ft,
    'categorical_features':ft_c
}
    # Define the number of columns and rows for the subplots
    if len(ft) == 1:
        cols = 1
        rows = 1
    elif len(ft) == 2:
        cols = 2
        rows = 1
    elif len(ft) == 3:
        cols = 3
        rows = 1
    elif len(ft) == 4:
        cols = 2
        rows = 2
    elif len(ft) in [5, 6]:
        cols = 3
        rows = 2

    # Define indicator so that the plots are only centered if the choice is 'both' or 'ice'
    c = False
    if choice_n != 'average':
        c = True

    # Plot
    _, ax = plt.subplots(ncols=cols, nrows=rows, figsize=(12, 10), constrained_layout=True)
    
    display = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    **features_info,
    centered=c,
    **common_params,
    ax=ax,
    kind= choice_n,
    ice_lines_kw= {"color":"gray", "alpha":0.6}, 
    pd_line_kw = {"color": "black", "lw" : 0.8} 
)

    if choice_n == 'both':
        # Delete the internal legends of the display
        for idx, ax in enumerate( display.axes_.ravel()):
            ax.legend_.remove() 

            if idx ==  1:
                ax.legend(
                    loc='upper center',              # legend situated in the upper center
                    bbox_to_anchor=(0.5, 1.10),      # pushes it even further up, outside the axis area
                    ncol=2,                          # shows the legend in 2 columns
                    frameon=False,                   # no border around
                    fontsize=9                       # text size
                )

    fig = display.figure_  

    # Details of the figure
    fig.suptitle(
        (
            f"Partial dependence of the number of bike rentals in stations {station_ids}\n"
            "for the bike rental dataset with lightgbm regressor"
        ),
        fontsize=16,
        fontweight='bold'
    )


    # Save high-res figure
    station_str = "_".join(str(s) for s in station_ids)
    filename = f"../figures/{choice}_for_{station_str}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')





