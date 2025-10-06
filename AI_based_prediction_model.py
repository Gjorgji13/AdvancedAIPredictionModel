# pip install pytorch-lightning
# pip install pytorch-forecasting
# pip install optuna  # for hyperparameter optimization
# pip install xlsxwriter

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
import re
from pathlib import Path
from io import BytesIO
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pandas import ExcelWriter
from pytorch_forecasting import TimeSeriesDataSet
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout
from IPython.display import display, FileLink, clear_output
from scipy.stats import wasserstein_distance
import hashlib, re
import math
import io

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Global paths and setup ---
BASE_DIR = Path("/content/created_dataset")
TRAINED_MODELS_DIR = Path("/content/Trained_Models")
TRAINED_MODELS_DIR.mkdir(exist_ok=True, parents=True)
BASE_DIR.mkdir(exist_ok=True, parents=True)

# --- Data and Widget State Variables ---
df = None
encoders = {}
output = widgets.Output()
current_model = None

def load_preview(file_path, nrows=5):
    """Load only first few rows for preview to avoid memory issues."""
    return pd.read_csv(file_path, nrows=nrows, dtype=str)

def load_full_dataset(file_path):
    """Load entire dataset with automatic dtype inference."""
    return pd.read_csv(file_path, low_memory=False)

def analyze_dataset(df, max_rows=10000000000):

    # Take only a sample of the dataset
    sample_df = df.head(max_rows)

    numeric_cols = sample_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = [col for col in sample_df.columns if col not in numeric_cols]

    # Minimal statistics for speed
    stats = {col: sample_df[col].agg(['min','max','mean','std']).to_dict() for col in numeric_cols}

    # Lightweight hash using numeric columns only (or all if none)
    hash_data = sample_df[numeric_cols] if numeric_cols else sample_df
    df_hash = hashlib.md5(pd.util.hash_pandas_object(hash_data).values).hexdigest()

    summary_dict = {
        "columns": list(sample_df.columns),
        "dtypes": {col: str(sample_df[col].dtype) for col in sample_df.columns},
        "stats": stats,
        "hash": df_hash
    }

    return summary_dict, numeric_cols, categorical_cols

# --- Utility Functions ---
def get_class_options():
    """Fetches names of all previously processed dataset classes from disk."""
    base = Path("/content/created_dataset")
    class_names = []
    if base.exists():
        for dataset_folder in base.iterdir():
            if dataset_folder.is_dir():
                for subfolder in dataset_folder.iterdir():
                    if subfolder.is_dir() and subfolder.name.startswith("class_"):
                        class_name = subfolder.name.replace("class_", "")
                        class_names.append(class_name)
    return sorted(set(class_names))

def get_base_class_name(folder_name):
    """Extracts the base class name from a full path or folder name."""
    if 'class_' in folder_name:
        return folder_name.split('class_')[-1]
    return folder_name

def generate_smart_name(df_hash, target, include_date=True):
    """Generates a unique name for a dataset/model based on key attributes."""
    base_str = f"{target}_{df_hash}"
    if include_date:
        base_str += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return base_str

def save_metadata_and_data(folder_path, data_df):
    """Saves a DataFrame and its metadata to a specified folder."""
    folder_path.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(folder_path / "data.csv", index=False)
    with open(folder_path / "headers.json", "w") as f:
        json.dump(list(data_df.columns), f, indent=4)
    dtypes = {col: str(dtype) for col, dtype in data_df.dtypes.items()}
    with open(folder_path / "types.json", "w") as f:
        json.dump(dtypes, f, indent=4)
    data_df.head(100).to_csv(folder_path / "preview.csv", index=False)

def safe_load_metadata(base_class, base_models_dir=TRAINED_MODELS_DIR):
    """Loads model metadata safely."""
    metadata_path = base_models_dir / f"{base_class}_metadata.json"
    if not metadata_path.exists():
        return None, None, None
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        saved_target = metadata.get("target")
        saved_features = metadata.get("features", [])
        return metadata, saved_target, saved_features
    except Exception as e:
        return None, None, None

def load_model_for_class(base_class):
    """Loads a pre-trained model for a given class name."""
    model_path = TRAINED_MODELS_DIR / f"{base_class}_model.pkl"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

def find_similar_model(new_summary, metadata_dir="models_metadata"):
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    best_match = None
    best_score = 0.0

    for meta_file in os.listdir(metadata_dir):
        if meta_file.endswith(".json"):
            with open(os.path.join(metadata_dir, meta_file), 'r') as f:
                old_summary = json.load(f)
            score = compare_datasets(new_summary, old_summary)
            if score > best_score:
                best_score, best_match = score, meta_file

    return best_match if best_score > 0.9 else None

def load_model_metadata(base_class):
    """Loads a model's metadata."""
    metadata_path = TRAINED_MODELS_DIR / f"{base_class}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def auto_detect_targets(data_df):
    """Automatically detects potential target columns based on keywords and variability."""
    candidates = []
    for col in data_df.select_dtypes(include=np.number).columns:
        lower = col.lower()
        if any(k in lower for k in ['sales', 'price', 'qty', 'volume', 'amount', 'demand', 'load']):
            candidates.append(col)
    if not candidates:
        stds = data_df.select_dtypes(include=np.number).std()
        candidates = stds[stds > 0].index.tolist()
    return candidates

def parse_promo_per_target(text):
    """Parses promotional uplift from a text string."""
    result = {}
    try:
        for pair in text.split(','):
            if ':' in pair:
                key, val = pair.split(':')
                key = key.strip()
                val = float(val.strip())
                if val < 0:
                    val = 0
                result[key] = val
    except Exception:
        pass
    return result

def remove_highly_correlated_features(data_df, targets, threshold=0.95):
    """Removes features that are highly correlated with each other or with targets."""
    numeric_df = data_df.select_dtypes(include=[np.number])
    if numeric_df.empty: return []
    corr = numeric_df.corr().abs()
    to_remove = set()
    for tcol in targets:
        if tcol in corr.columns:
            high_corr = corr.index[(corr[tcol] > threshold) & (corr.index != tcol)].tolist()
            to_remove.update(high_corr)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    for col in upper.columns:
        correlated_cols = upper.index[upper[col] > threshold].tolist()
        if correlated_cols:
            to_remove.add(col)
    to_remove = to_remove - set(targets)
    return list(to_remove)

def build_lags_rollings(data_df, targets, max_lag=3, rolling_windows=[3, 6]):
    """Creates lag and rolling mean features for forecasting."""
    data_df = data_df.copy()
    for tcol in targets:
        for lag in range(1, max_lag + 1):
            data_df[f"{tcol}_lag{lag}"] = data_df[tcol].shift(lag)
        for window in rolling_windows:
            data_df[f"{tcol}_roll{window}"] = data_df[tcol].rolling(window).mean()
    return data_df

def detect_time_col(df):
    """Detect a likely time column."""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        return datetime_cols[0]  # pick the first datetime column
    # fallback: check common names
    for col in df.columns:
        if col.lower() in ["date", "time", "month", "year", "period"]:
            return col
    return None


def add_cyclical_time_features(data_df, time_col='Month'):
    """Adds cyclical sine and cosine features for time-based data."""
    if time_col in data_df.columns:
        data_df[f'{time_col}_sin'] = np.sin(2 * np.pi * data_df[time_col] / 12)
        data_df[f'{time_col}_cos'] = np.cos(2 * np.pi * data_df[time_col] / 12)
    return data_df

def save_model_metadata(base_class, target, features, base_models_dir=TRAINED_MODELS_DIR):
    """Saves metadata for a trained model."""
    metadata = {"target": target, "features": features}
    os.makedirs(base_models_dir, exist_ok=True)
    metadata_path = os.path.join(base_models_dir, f"{base_class}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def compare_datasets(new_summary, old_summary):
    matching_columns = len(set(new_summary["columns"]) & set(old_summary["columns"]))
    total_columns = max(len(new_summary["columns"]), len(old_summary["columns"]))
    return matching_columns / total_columns

def save_metadata(summary, file_path):

    file_path = Path(file_path)
    metadata_path = file_path.with_suffix('.metadata.json')

    with open(metadata_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Metadata saved to {metadata_path}")

from pytorch_forecasting import TimeSeriesDataSet

def prepare_tft_dataset(df, target_col, time_col, group_col=None, max_encoder_length=24, max_prediction_length=12):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['time_idx'] = (df[time_col] - df[time_col].min()).dt.days

    if group_col is None:
        df['group'] = 'single_series'
        group_col = 'group'

    # Define dataset
    tft_dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=[col for col in df.columns if col not in [target_col, group_col, time_col]],
        time_varying_unknown_reals=[target_col],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return tft_dataset

from pandas import ExcelWriter

def process_dataset(file_path):
    df = pd.read_csv(file_path)
    summary = analyze_dataset(df)
    model_file = find_similar_model(summary)

    # Run forecast and get results
    if model_file:
        print(f"Using existing model: {model_file}")
        backtest_results_df, forecast_df = run_forecast(
            df, model_path=f"models/{model_file.replace('.json', '.pkl')}"
        )
    else:
        print("No similar model found. Training new model")
        backtest_results_df, forecast_df = run_forecast(df, model_path=None)
        save_metadata(summary, file_path)

    # Call helper function to save forecast output
    save_forecast_output(backtest_results_df, forecast_df, "my_dataset.csv")

def compare_with_existing_classes(df, base_dir=BASE_DIR, threshold=0.8):
    summary = analyze_dataset(df)
    numeric_cols = [col for col, dtype in summary['dtypes'].items() if 'float' in dtype or 'int' in dtype]
    categorical_cols = [col for col in summary['columns'] if col not in numeric_cols]

    for dataset_folder in base_dir.iterdir():
        if not dataset_folder.is_dir():
            continue
        class_name = dataset_folder.name
        try:
            existing_df = pd.read_csv(dataset_folder / "data.csv")
            existing_summary = analyze_dataset(existing_df)
            existing_numeric = [col for col, dtype in existing_summary['dtypes'].items() if 'float' in dtype or 'int' in dtype]
            existing_categorical = [col for col in existing_summary['columns'] if col not in existing_numeric]

            num_overlap = len(set(numeric_cols) & set(existing_numeric)) / max(len(numeric_cols), 1)
            cat_overlap = len(set(categorical_cols) & set(existing_categorical)) / max(len(categorical_cols), 1)

            if num_overlap >= threshold and cat_overlap >= threshold:
                return class_name
        except Exception:
            continue
    return None

def create_train_test_split(df, test_size=0.2, random_state=42, base_folder=None):
    """Split dataset into train/test and save CSVs."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    if base_folder:
        train_folder = base_folder / "train"
        test_folder = base_folder / "test"
        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_folder / "train.csv", index=False)
        test_df.to_csv(test_folder / "test.csv", index=False)
    return train_df, test_df

def load_or_train_model(df, targets, features, class_name, base_models_dir=TRAINED_MODELS_DIR):

    model_path = base_models_dir / f"{class_name}_model.pkl"
    metadata_path = base_models_dir / f"{class_name}_metadata.json"

    # Compute dataset hash to detect changes
    df_hash = hashlib.md5(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]

    # Check if metadata exists
    retrain = True
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if metadata.get("df_hash") == df_hash:
            retrain = False

    if retrain:
        print(f"ℹ️ Training new model for class '{class_name}'...")
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
        model.fit(df[features], df[targets])
        joblib.dump(model, model_path)
        # Save metadata
        metadata = {"features": features, "targets": targets, "df_hash": df_hash}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print("✅ Model trained and saved.")
    else:
        print(f"ℹ️ Loading existing model for class '{class_name}'...")
        model = joblib.load(model_path)
    return model

def get_features_targets(df, auto_detect=True, manual_targets=None):
    """Determine features and targets for modeling."""
    if auto_detect:
        targets = auto_detect_targets(df)
        features = [col for col in df.columns if col not in targets]
    else:
        targets = manual_targets
        features = [col for col in df.columns if col not in targets]
    return features, targets

def smart_forecast_pipeline(df, class_name=None, manual_targets=None):

    """Full end-to-end pipeline with class detection, train/test split, and model."""
    # Detect existing class
    if not class_name:
        matched_class = compare_with_existing_classes(df)
        if matched_class:
            print(f"✅ Dataset matches existing class: {matched_class}")
            class_name = matched_class
        else:
            class_name = f"class_{generate_smart_name(hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:8], 'dataset')}"
            print(f"ℹ️ Creating new class: {class_name}")

    class_folder = BASE_DIR / class_name
    class_folder.mkdir(parents=True, exist_ok=True)

    # Split train/test
    train_df, test_df = create_train_test_split(df, base_folder=class_folder)

    # Detect features and targets
    features, targets = get_features_targets(df, auto_detect=(manual_targets is None), manual_targets=manual_targets)

    # Train or load model
    model = load_or_train_model(train_df, targets, features, class_name)

    return model, features, targets, train_df, test_df, class_name

def plot_dynamic_forecast(df_history, forecast_df, targets):

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))

    # Determine x-axis
    if 'Forecast_Date' in forecast_df.columns:
        x_axis = 'Forecast_Date'
    elif 'Period' in forecast_df.columns:
        x_axis = 'Period'
    else:
        x_axis = forecast_df.index

    for t in targets:
        forecast_col = f"Forecasted_{t}"
        if forecast_col not in forecast_df.columns:
            continue

        # Plot historical (last n points)
        if t in df_history.columns:
            plt.plot(df_history.index[-20:], df_history[t].iloc[-20:],
                     label=f"Actual {t}", linestyle='--', marker='o', alpha=0.7)

        # Plot forecast
        plt.plot(forecast_df[x_axis], forecast_df[forecast_col],
                 label=f"Forecast {t}", linewidth=2, marker='o')

    plt.title("Forecast Visualization")
    plt.xlabel(x_axis)
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    def get_ci(y_actual, y_forecast, ci_level=0.95):
        # Approximate CI using RMSE
        residuals = y_actual[-len(y_forecast):] - y_forecast[:len(y_actual[-len(y_forecast):])]
        rmse = np.sqrt(np.mean(residuals**2)) if len(residuals) > 0 else 0
        z = 1.96  # 95% CI
        return y_forecast - z*rmse, y_forecast + z*rmse

    # Prepare x-axis for historical and forecast
    hist_x = original_df[time_col] if time_col else np.arange(len(original_df))
    forecast_x = forecast_df[time_col] if time_col in forecast_df else np.arange(len(original_df), len(original_df)+len(forecast_df))

    if plot_type == "line_single":
        t = targets[0]

        # Plot historical
        plt.plot(hist_x, original_df[t], label=f'Actual {t}', marker='o')

        # Plot forecast
        plt.plot(forecast_x, forecast_df[f'Forecasted_{t}'], label=f'Forecasted {t}', linestyle='--', marker='x')

        # Confidence interval
        if show_ci:
            lower, upper = get_ci(original_df[t].values, forecast_df[f'Forecasted_{t}'].values)
            plt.fill_between(forecast_x, lower, upper, color='orange', alpha=0.2, label='95% CI')

        # Annotate extremes
        if annotate_extremes:
            top_idx = forecast_df[f'Forecasted_{t}'].idxmax()
            bottom_idx = forecast_df[f'Forecasted_{t}'].idxmin()
            plt.annotate(f"Max: {forecast_df[f'Forecasted_{t}'][top_idx]:.2f}",
                         xy=(forecast_x[top_idx], forecast_df[f'Forecasted_{t}'][top_idx]),
                         xytext=(forecast_x[top_idx], forecast_df[f'Forecasted_{t}'][top_idx]*1.05),
                         arrowprops=dict(facecolor='green', arrowstyle='->'))
            plt.annotate(f"Min: {forecast_df[f'Forecasted_{t}'][bottom_idx]:.2f}",
                         xy=(forecast_x[bottom_idx], forecast_df[f'Forecasted_{t}'][bottom_idx]),
                         xytext=(forecast_x[bottom_idx], forecast_df[f'Forecasted_{t}'][bottom_idx]*0.95),
                         arrowprops=dict(facecolor='red', arrowstyle='->'))

        plt.xlabel(time_col if time_col else "Period")
        plt.ylabel("Value")
        plt.title(f"Forecast for {t}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    elif plot_type == "subplots":
        fig, axs = plt.subplots(n_targets, 1, figsize=(max(12, n_periods/2), 3 * n_targets), squeeze=False)
        for i, t in enumerate(targets):
            ax = axs[i, 0]

            # Historical
            ax.plot(hist_x, original_df[t], label=f'Actual {t}', marker='o')
            # Forecast
            ax.plot(forecast_x, forecast_df[f'Forecasted_{t}'], label=f'Forecasted {t}', linestyle='--', marker='x')

            # CI
            if show_ci:
                lower, upper = get_ci(original_df[t].values, forecast_df[f'Forecasted_{t}'].values)
                ax.fill_between(forecast_x, lower, upper, color='orange', alpha=0.2, label='95% CI')

            # Extremes
            if annotate_extremes:
                top_idx = forecast_df[f'Forecasted_{t}'].idxmax()
                bottom_idx = forecast_df[f'Forecasted_{t}'].idxmin()
                ax.annotate(f"Max: {forecast_df[f'Forecasted_{t}'][top_idx]:.2f}",
                            xy=(forecast_x[top_idx], forecast_df[f'Forecasted_{t}'][top_idx]),
                            xytext=(forecast_x[top_idx], forecast_df[f'Forecasted_{t}'][top_idx]*1.05),
                            arrowprops=dict(facecolor='green', arrowstyle='->'))
                ax.annotate(f"Min: {forecast_df[f'Forecasted_{t}'][bottom_idx]:.2f}",
                            xy=(forecast_x[bottom_idx], forecast_df[f'Forecasted_{t}'][bottom_idx]),
                            xytext=(forecast_x[bottom_idx], forecast_df[f'Forecasted_{t}'][bottom_idx]*0.95),
                            arrowprops=dict(facecolor='red', arrowstyle='->'))

            ax.set_xlabel(time_col if time_col else "Period")
            ax.set_ylabel("Value")
            ax.set_title(f"Forecast: {t}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    # Bar and heatmap types remain unchanged
    elif plot_type == "bar":
        x = np.arange(n_periods)
        width = 0.7 / n_targets
        for i, t in enumerate(targets):
            plt.bar(x + i * width, forecast_df[f'Forecasted_{t}'], width=width, label=f'Forecasted {t}')
        plt.xlabel("Forecast Period")
        plt.ylabel("Value")
        plt.title("Forecast Bar Plot")
        plt.xticks(x + width * (n_targets-1)/2, forecast_df.get('Period', x), rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif plot_type == "subplots":
        fig, axs = plt.subplots(n_targets, 1, figsize=(max(12, n_periods/2), 3 * n_targets), squeeze=False)
        for i, t in enumerate(targets):
            ax = axs[i, 0]
            x_actual = original_df[time_col] if time_col else forecast_df['Period']
            ax.plot(x_actual, original_df[t], label=f'Actual {t}', marker='o')
            ax.plot(forecast_df['Period'], forecast_df[f'Forecasted_{t}'], label=f'Forecasted {t}', linestyle='--', marker='x')
            ax.set_xlabel(time_col if time_col else "Forecast Period")
            ax.set_ylabel("Value")
            ax.set_title(f"Forecast: {t}")
            ax.legend()
            ax.grid(True)
            if annotate_extremes:
                top_idx = forecast_df[f'Forecasted_{t}'].idxmax()
                bottom_idx = forecast_df[f'Forecasted_{t}'].idxmin()
                ax.annotate(f"Max: {forecast_df[f'Forecasted_{t}'][top_idx]:.2f}",
                            xy=(top_idx, forecast_df[f'Forecasted_{t}'][top_idx]),
                            xytext=(top_idx, forecast_df[f'Forecasted_{t}'][top_idx]*1.05),
                            arrowprops=dict(facecolor='green', arrowstyle='->'))
                ax.annotate(f"Min: {forecast_df[f'Forecasted_{t}'][bottom_idx]:.2f}",
                            xy=(bottom_idx, forecast_df[f'Forecasted_{t}'][bottom_idx]),
                            xytext=(bottom_idx, forecast_df[f'Forecasted_{t}'][bottom_idx]*0.95),
                            arrowprops=dict(facecolor='red', arrowstyle='->'))
        plt.tight_layout()
        plt.show()

    elif plot_type == "heatmap":
        data = forecast_df[[f'Forecasted_{t}' for t in targets]]
        sns.heatmap(data.T, cmap='viridis', cbar=True, annot=False)
        plt.xlabel("Forecast Period")
        plt.ylabel("Targets")
        plt.title("Forecast Heatmap")
        plt.tight_layout()
        plt.show()


# --- Widget Definitions ---
style = {'description_width': '180px'}
layout = Layout(width='90%')
upload = widgets.FileUpload(accept='.csv, .xlsx', multiple=False)
process_button = widgets.Button(description="Upload & Process Dataset", button_style='primary')
# preview_button = widgets.Button(description="Preview First 5 Rows", button_style='info')

use_manual_targets = widgets.Checkbox(
    value=False,
    description="Manual Target/Feature Selection",
    tooltip="Check to select targets/features manually"
)

mode_select = widgets.Dropdown(
    options=[
        "Auto Select Best Model",
        "Manual Select Algorithm",
        "TFT (Deep Learning)"  # new deep learning option
    ],
    description="Mode:",
    style=style,
    layout=layout,
    disabled=False  # allow changing mode
)

algo_dropdown = widgets.Dropdown(
    options=["RandomForest", "XGBoost", "TFT (Deep Learning)"],  # add TFT
    description="Algorithm:",
    style=style,
    layout=layout,
    disabled=False
)

target_col = widgets.SelectMultiple(
    options=[], description="Targets:",
    layout=widgets.Layout(width='50%')
)
features_select = widgets.SelectMultiple(
    options=[], description="Features:",
    layout=widgets.Layout(width='50%')
)

class_select = widgets.Dropdown(
    options=[''] + get_class_options(),
    description="Select Class:",
    style=style, layout=layout
)

drop_na_chk = widgets.Checkbox(value=True, description="Drop missing values")
fill_na_chk = widgets.Checkbox(value=False, description="Fill missing with 0")
forecast_slider = widgets.IntSlider(value=5, min=1, max=12, step=1, description="Forecast periods")
promo_slider = widgets.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.01, description="Global Promo Uplift %")
promo_per_target = widgets.Text(
    value='',
    placeholder='e.g. target1:0.1, target2:0.05',
    description='Promo uplift per target:',
    layout=Layout(width='90%'),
    style={'description_width': '200px'}
)
run_button = widgets.Button(description="Run Forecast", button_style='success')


class DatasetChunkProcessor:

    def __init__(self, train_path, test_path, test_size=0.2, random_state=42, sample_size=10000):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.test_size = float(test_size)
        self.rng = np.random.RandomState(random_state)
        self.sample_size = int(sample_size)

        # write-header flags
        self._train_header_written = False
        self._test_header_written = False

        # accumulators
        self._sampled_frames = [] # list of DataFrame samples
        self._sampled_rows = 0
        self.total_rows = 0

        # detection placeholders
        self.numeric_cols = None
        self.auto_targets = None

    def _maybe_detect(self, df_chunk):
        # detect numeric cols and auto-detected targets using your existing helper
        if self.numeric_cols is None:
            self.numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns.tolist()
        if self.auto_targets is None:
            at = auto_detect_targets(df_chunk)
            if at:
                self.auto_targets = at
            else:
                # fallback to first numeric column if exists
                if self.numeric_cols:
                    self.auto_targets = [self.numeric_cols[0]]

    def process_chunk(self, chunk_df):

        if chunk_df.empty:
            return

        # detect on first useful chunk
        self._maybe_detect(chunk_df)

        # if still no targets, skip train/test sampling and just append to train as fallback
        targets = self.auto_targets or []

        # coerce target columns to numeric to avoid surprises
        for t in targets:
            if t in chunk_df.columns:
                chunk_df[t] = pd.to_numeric(chunk_df[t], errors='coerce')

        # drop rows without target values (mimic your pipeline)
        if targets:
            chunk_df = chunk_df.dropna(subset=targets)
            if chunk_df.empty:
                return

        # sample rows (for summary/hashing) up to sample_size
        if self._sampled_rows < self.sample_size:
            need = self.sample_size - self._sampled_rows
            # if chunk smaller, take up to need
            sample_n = min(need, len(chunk_df))
            if sample_n > 0:
                self._sampled_frames.append(chunk_df.sample(n=sample_n, random_state=self.rng))
                self._sampled_rows += sample_n

        r = self.rng.rand(len(chunk_df))
        test_mask = r < self.test_size
        train_mask = ~test_mask

        train_rows = chunk_df.loc[train_mask]
        test_rows = chunk_df.loc[test_mask]

        # append to CSVs (write header only once)
        if not train_rows.empty:
            train_rows.to_csv(self.train_path, mode='a', index=False, header=not self._train_header_written)
            self._train_header_written = True
        if not test_rows.empty:
            test_rows.to_csv(self.test_path, mode='a', index=False, header=not self._test_header_written)
            self._test_header_written = True

        # update counters
        self.total_rows += len(chunk_df)

    def finalize(self):
        """Return summary info computed from sampled rows (or None if nothing)."""
        summary = None
        sampled_df = pd.concat(self._sampled_frames, ignore_index=True) if self._sampled_frames else pd.DataFrame()
        if not sampled_df.empty:
            # numeric summary on sampled rows (fast)
            numeric_cols = sampled_df.select_dtypes(include=[np.number]).columns.tolist()
            stats = {c: sampled_df[c].agg(['min', 'max', 'mean', 'std']).to_dict() for c in numeric_cols}
            df_hash = hashlib.md5(pd.util.hash_pandas_object(sampled_df[numeric_cols] if numeric_cols else sampled_df).values).hexdigest()
            summary = {
                "columns": list(sampled_df.columns),
                "dtypes": {col: str(sampled_df[col].dtype) for col in sampled_df.columns},
                "stats": stats,
                "hash": df_hash
            }
        return {
            "total_rows": self.total_rows,
            "numeric_cols": self.numeric_cols or [],
            "auto_targets": self.auto_targets or [],
            "sample_summary": summary
        }

def smart_load_file(file_content=None, path=None, sheet=None):
    if file_content:
        file_name = file_content['metadata']['name']
        content = file_content['content']
        ext = Path(file_name).suffix.lower()

        if ext in [".xls", ".xlsx"]:
            return pd.read_excel(BytesIO(content), sheet_name=sheet)
        else:
            try:
                return pd.read_csv(BytesIO(content), encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                return pd.read_csv(BytesIO(content), encoding="latin1", low_memory=False)

    elif path:
        path = Path(path)
        ext = path.suffix.lower()
        if ext in [".xls", ".xlsx"]:
            return pd.read_excel(path, sheet_name=sheet)
        else:
            try:
                return pd.read_csv(path, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="latin1", low_memory=False)
    else:
        raise ValueError("Either file_content or path must be provided.")

upload = widgets.FileUpload(accept=".csv,.xls,.xlsx", multiple=False)
sheet_dropdown = widgets.Dropdown(description="Sheet:", options=[], disabled=True)

preview_button = widgets.Button(description="Preview", button_style="info")
remove_button = widgets.Button(description="Remove", button_style="danger")

target_col = widgets.SelectMultiple(description="Target(s):")
features_select = widgets.SelectMultiple(description="Features:")
use_manual_targets = widgets.Checkbox(description="Manual target selection", value=False)

# Optional advanced section
mode_dropdown = widgets.Dropdown(description="Mode:", options=["Auto Select Best Model", "Manual"])
algorithm_dropdown = widgets.Dropdown(description="Algorithm:", options=["RandomForest", "XGBoost", "LinearRegression"])

output = widgets.Output()

# --- Define Handlers First ---
def on_upload_changed(change):
    """Populate sheet dropdown and keep lightweight preview info available."""
    sheet_dropdown.options = []
    sheet_dropdown.disabled = True

    if not upload.value:
        # no file selected
        with output:
            clear_output()
            print("🛈 Upload cleared.")
        return

    # Pull first file (FileUpload stores a dict)
    file_content = next(iter(upload.value.values()))
    file_name = file_content.get('metadata', {}).get('name', '') or file_content.get('name', '')

    # detect Excel sheets
    try:
        if file_name.lower().endswith(('.xls', '.xlsx')):
            excel_file = pd.ExcelFile(BytesIO(file_content['content']))
            sheets = excel_file.sheet_names
            if len(sheets) > 1:
                sheet_dropdown.options = sheets
                sheet_dropdown.value = sheets[0]
                sheet_dropdown.disabled = False
            else:
                sheet_dropdown.options = []
                sheet_dropdown.disabled = True
    except Exception as e:
        with output:
            clear_output()
            print(f"❌ Failed to inspect file: {e}")
            sheet_dropdown.options = []
            sheet_dropdown.disabled = True

    with output:
        clear_output()
        print(f"✅ File ready: {file_name} — click Preview to inspect (loads preview then full file).")


def on_remove_clicked(b):
    """Clear upload and reset dependent widgets."""
    global df
    df = None
    # Recreate upload widget to fully clear it
    recreate_upload()
    sheet_dropdown.options = []
    sheet_dropdown.disabled = True
    target_col.options = []
    target_col.value = tuple()
    features_select.options = []
    features_select.value = tuple()
    with output:
        clear_output()
        print("🗑️ Upload removed and UI reset.")

def on_preview_clicked(b):
    """Load a small preview to show immediately, then load the full dataset into df and populate selectors."""
    global df
    with output:
        clear_output()
        if not upload.value:
            print("❌ No file uploaded.")
            return

        file_content = next(iter(upload.value.values()))
        file_name = file_content.get('metadata', {}).get('name', '') or file_content.get('name', '')
        sheet = sheet_dropdown.value if not sheet_dropdown.disabled else None

        try:
            # Lightweight preview first
            if sheet:
                df_preview = pd.read_excel(BytesIO(file_content['content']), sheet_name=sheet, nrows=10)
            else:
                df_preview = smart_read_file(file_content, nrows=10)
            print(f"📂 Preview ({'Sheet: ' + sheet if sheet else 'CSV'})")
            display(df_preview)

            # Now load the full dataset (may be large)
            df = smart_load_file(file_content=file_content, sheet=sheet)

            # Populate selectors
            numeric_targets = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()

            # set options
            target_col.options = numeric_targets or all_cols
            # auto-target selection
            if numeric_targets:
                auto_targets = auto_detect_targets(df) or [numeric_targets[-1]]
                if not use_manual_targets.value:
                    target_col.value = tuple(auto_targets)
                    features_select.options = [c for c in all_cols if c not in auto_targets]
                    features_select.value = tuple([c for c in all_cols if c not in auto_targets])
                else:
                    features_select.options = all_cols
            else:
                # no numeric -> let user pick
                target_col.options = all_cols
                features_select.options = all_cols

            print(f"✅ Loaded full dataset ({len(df)} rows). Targets/options updated.")
        except Exception as e:
            print(f"❌ ERROR during preview/load: {e}")

# --- Attach handlers (only once) ---
upload.observe(on_upload_changed, names="value")
preview_button.on_click(on_preview_clicked)
remove_button.on_click(on_remove_clicked)

# --- Layout & display once ---
file_box = widgets.HBox([upload, remove_button, preview_button])
display(widgets.VBox([file_box, sheet_dropdown, widgets.HBox([target_col, features_select]), use_manual_targets, output]))

def clean_data(df):

    import numpy as np

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])  # most frequent
        else:
            df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

    # Cap outliers (z-score > 3)
    for col in df.select_dtypes(include=np.number).columns:
        z = (df[col] - df[col].mean()) / df[col].std()
        df.loc[z > 3, col] = df[col].mean() + 3 * df[col].std()
        df.loc[z < -3, col] = df[col].mean() - 3 * df[col].std()

    return df

def smart_read_file(file_content, nrows=None):

    file_name = file_content['metadata']['name']
    content = file_content['content']
    ext = Path(file_name).suffix.lower()

    try:
        if ext in [".xls", ".xlsx"]:
            # Excel file
            return pd.read_excel(BytesIO(content), nrows=nrows)
        else:
            # CSV with encoding fallbacks
            try:
                return pd.read_csv(BytesIO(content), nrows=nrows, low_memory=False, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(BytesIO(content), nrows=nrows, low_memory=False, encoding="latin1")
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_name}: {e}")

# --- Widgets ---
def remove_file_clicked(b):
    global upload, df
    df = None
    with output:
        output.clear_output()
        print("🗑️ File removed. Please upload a new one.")

# --- Target/feature selectors (placeholders for now) ---
target_col = widgets.SelectMultiple(description="Target(s):")
features_select = widgets.SelectMultiple(description="Features:")
use_manual_targets = widgets.Checkbox(description="Manual target selection", value=False)

def create_upload_widget():
    up = widgets.FileUpload(accept='.csv,.xls,.xlsx', multiple=False)
    up.observe(on_upload_changed, names='value')
    return up

upload = create_upload_widget()
remove_btn = widgets.Button(description="Remove", button_style="danger")

file_box = widgets.HBox([upload, remove_btn, preview_button])
remove_btn.on_click(remove_file_clicked)

sheet_dropdown = widgets.Dropdown(options=[], description="Sheet:", disabled=True)


# 1️⃣ Define the handler first
def on_sheet_changed(change):
    """When user picks a different sheet, show preview of that sheet (if file present)."""
    global df
    if not upload.value:
        return
    file_content = next(iter(upload.value.values()))
    sheet = sheet_dropdown.value
    try:
        df_preview = pd.read_excel(BytesIO(file_content['content']), sheet_name=sheet, nrows=10)
        with output:
            clear_output()
            print(f"📂 Preview Sheet: {sheet}")
            display(df_preview)
    except Exception as e:
        with output:
            clear_output()
            print(f"❌ Failed to load sheet '{sheet}': {e}")

# 2️⃣ Then attach it
sheet_dropdown.observe(on_sheet_changed, names='value')


def process_dataset_in_chunks(
    file_path,
    dataset_folder,
    train_folder,
    test_folder,
    chunksize=50000,
    test_size=0.2,
    random_state=42,
    sample_size=10000,
    low_memory=False
):

    file_path = Path(file_path)
    dataset_folder = Path(dataset_folder)
    train_folder = Path(train_folder)
    test_folder = Path(test_folder)

    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    train_path = train_folder / "train.csv"
    test_path = test_folder / "test.csv"

    # remove any existing partial files (optional)
    try:
        if train_path.exists(): train_path.unlink()
        if test_path.exists(): test_path.unlink()
    except Exception:
        pass

    processor = DatasetChunkProcessor(train_path, test_path, test_size=test_size, random_state=random_state, sample_size=sample_size)

    # iterate chunks
    reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=low_memory)
    for chunk in reader:
        try:
            processor.process_chunk(chunk)
        except Exception as e:
            # best-effort: continue on chunk errors
            print(f"⚠️ Warning: error processing a chunk: {e}")
            continue

    results = processor.finalize()
    return results, str(train_path), str(test_path)

    time_col = detect_time_col(df)
    print(f"Using time_col = {time_col}")
    df = generate_features(df, time_col)


def generate_features(df, time_col=None):
    df = df.copy()

    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df['Month'] = df[time_col].dt.month
        df['Quarter'] = df[time_col].dt.quarter
        df['DayOfWeek'] = df[time_col].dt.dayofweek

    # Lag + rolling stats
    for col in df.select_dtypes(include=np.number).columns:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_rolling_mean3'] = df[col].rolling(window=3).mean()
        df[f'{col}_rolling_std3'] = df[col].rolling(window=3).std()

    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def on_plot_clicked(b):
    with plot_output:
        plot_output.clear_output()
        x_col = x_dropdown.value
        y_col = None if y_dropdown.value == 'None' else y_dropdown.value
        if x_col:
            smart_graph(df, x_col, y_col)

def smart_graph_auto(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import combinations

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Single-column distributions
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

    for col in categorical_cols:
        plt.figure(figsize=(7, 4))
        df[col].value_counts().head(20).plot(kind="bar")
        plt.title(f"Top categories of {col}")
        plt.show()

    for col in datetime_cols:
        plt.figure(figsize=(7, 4))
        df[col].value_counts().sort_index().plot()
        plt.title(f"Timeline of {col}")
        plt.show()

    # Two-column combinations
    all_cols = numeric_cols + categorical_cols + datetime_cols
    for x_col, y_col in combinations(all_cols, 2):
        try:
            plt.figure(figsize=(7, 4))
            if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                sns.scatterplot(x=df[x_col], y=df[y_col])
                plt.title(f"Scatter: {x_col} vs {y_col}")

            elif (pd.api.types.is_datetime64_any_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])) or \
                 (pd.api.types.is_datetime64_any_dtype(df[y_col]) and pd.api.types.is_numeric_dtype(df[x_col])):
                # Time series line
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    sns.lineplot(x=df[x_col], y=df[y_col])
                    plt.title(f"Time series: {y_col} over {x_col}")
                else:
                    sns.lineplot(x=df[y_col], y=df[x_col])
                    plt.title(f"Time series: {x_col} over {y_col}")

            elif (x_col in categorical_cols or pd.api.types.is_categorical_dtype(df[x_col])) and y_col in numeric_cols:
                sns.boxplot(x=df[x_col], y=df[y_col])
                plt.title(f"Boxplot: {y_col} by {x_col}")

            elif (y_col in categorical_cols or pd.api.types.is_categorical_dtype(df[y_col])) and x_col in numeric_cols:
                sns.boxplot(x=df[y_col], y=df[x_col])
                plt.title(f"Boxplot: {x_col} by {y_col}")

            else:
                # fallback bar plot for other cases
                df.groupby(x_col)[y_col].mean().plot(kind="bar")
                plt.title(f"Bar plot: {y_col} by {x_col}")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not plot {x_col} vs {y_col}: {e}")

# --- Integrate into on_process_clicked ---
def on_process_clicked(b):
    """Handler for processing uploaded file with Excel sheet support and automatic smart graphs."""
    global df, current_model
    output.clear_output()
    with output:
        if not upload.value:
            print("❌ No file uploaded. Please upload a CSV/Excel file first.")
            return

        try:
            # Reset states
            df = None
            current_model = None

            # Read upload (get first file)
            file_content = next(iter(upload.value.values()))
            file_name = file_content['metadata']['name']
            print(f"✅ File '{file_name}' uploaded successfully.")

            preview_rows = 10000000

            # --- Excel sheet handling ---
            if file_name.lower().endswith((".xlsx", ".xls")):
                file_bytes = io.BytesIO(file_content['content'])
                xls = pd.ExcelFile(file_bytes)
                available_sheets = xls.sheet_names
                sheet_dropdown.options = available_sheets
                if sheet_dropdown.value not in available_sheets:
                    sheet_dropdown.value = available_sheets[0]
                chosen_sheet = sheet_dropdown.value
                print(f"📑 Using sheet: {chosen_sheet}")
                df_preview = pd.read_excel(file_bytes, sheet_name=chosen_sheet, nrows=preview_rows)
            else:
                df_preview = smart_read_file(file_content, nrows=preview_rows)

            nrows_preview = len(df_preview)
            print(f"Preview loaded with {nrows_preview} rows for fast processing.")
            df = df_preview if nrows_preview < preview_rows else df_preview

            # --- Dataset hashing, folder setup ---
            numeric_sample = df.select_dtypes(include=np.number)
            df_hash = hashlib.md5(
                numeric_sample.head(10_000).to_csv(index=False).encode('utf-8')
            ).hexdigest()[:8]
            base_stem = re.sub(r'[^0-9A-Za-z_\-]', '_', Path(file_name).stem)
            smart_name = generate_smart_name(df_hash, base_stem, include_date=False)

            dataset_folder = BASE_DIR / smart_name
            class_folder = dataset_folder / f"class_{smart_name}"
            train_folder = class_folder / "train"
            test_folder = class_folder / "test"
            class_folder.mkdir(parents=True, exist_ok=True)
            train_folder.mkdir(parents=True, exist_ok=True)
            test_folder.mkdir(parents=True, exist_ok=True)

            save_metadata_and_data(dataset_folder, df)
            print(f"✅ Dataset saved under '{dataset_folder}'")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            auto_targets = auto_detect_targets(df)
            auto_features = [c for c in df.columns if c not in auto_targets]
            if not auto_targets and numeric_cols:
                auto_targets = [numeric_cols[0]]

            # --- Train/Test Split ---
            if auto_targets:
                raw_features = [c for c in df.columns if c not in auto_targets]
                X_raw = df[raw_features]
                y_raw = df[auto_targets]
                X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
                pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)\
                  .to_csv(train_folder / "train.csv", index=False)
                pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)\
                  .to_csv(test_folder / "test.csv", index=False)
                print(f"✅ Train/Test saved (train: {train_folder/'train.csv'}, test: {test_folder/'test.csv'})")
            else:
                df.to_csv(train_folder / "train.csv", index=False)
                print("⚠️ No numeric targets detected. Saved original dataset to train folder. Choose targets manually.")

            class_select.options = [''] + get_class_options()
            class_select.value = smart_name if smart_name in class_select.options else class_select.options[-1]

            target_col.options = numeric_cols
            features_select.options = df.columns.tolist()
            if not use_manual_targets.value:
                target_col.value = tuple(auto_targets)
                features_select.value = tuple(auto_features)
                print(f"Auto-selected targets: {auto_targets}")

            print(f"Dataset processed. Smart name: {smart_name}")

            # --- Run forecast if automatic ---
            if not use_manual_targets.value:
                run_forecast(None)

            # --- Auto smart graphs ---
            print("\n📊 Generating all smart graphs automatically...")
            smart_graph_auto(df)

        except Exception as e:
            print(f"❌ An ERROR occurred during processing: {e}")

# --- Populate sheet dropdown when Excel uploaded ---
def update_sheet_dropdown(change):
    if upload.value:
        file_content = next(iter(upload.value.values()))
        name = file_content["metadata"]["name"]
        if name.endswith((".xls", ".xlsx")):
            try:
                sheets = pd.ExcelFile(BytesIO(file_content["content"])).sheet_names
                sheet_dropdown.options = sheets
                sheet_dropdown.value = sheets[0] if sheets else None
                sheet_dropdown.disabled = False
            except Exception as e:
                with output:
                    print(f"❌ Failed to read sheets: {e}")
                sheet_dropdown.options = []
                sheet_dropdown.disabled = True
        else:
            sheet_dropdown.options = []
            sheet_dropdown.disabled = True

upload.observe(update_sheet_dropdown, names="value")

# --- Preview handler ---
def diagnostics_report(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.stattools import adfuller
    import numpy as np

    print("🔎 DATA DIAGNOSTICS REPORT")
    print("="*40)

    # Missing values
    missing = df.isnull().mean() * 100
    print("📊 Missing values (%):")
    print(missing[missing > 0].round(2))
    print("="*40)

    # Outliers (z-score > 3)
    for col in df.select_dtypes(include=np.number).columns:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = (z > 3).sum()
        if outliers > 0:
            print(f"⚠️ Column {col}: {outliers} outliers detected")

    print("="*40)

    # Correlation heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Stationarity check (ADF test)
    for col in df.select_dtypes(include=np.number).columns:
        result = adfuller(df[col].dropna())
        print(f"📉 ADF Test for {col}: p-value = {result[1]:.4f}")
        if result[1] < 0.05:
            print(f"✅ {col} is stationary.")
        else:
            print(f"❌ {col} is non-stationary (consider differencing).")

def on_manual_select_change(change):
    if change['new']:
        mode_select.disabled = False
        algo_dropdown.disabled = (mode_select.value == "Auto Select Best Model")
    else:
        mode_select.disabled = True
        algo_dropdown.disabled = True

def on_mode_select_change(change):
    if change['new'] == "Manual Select Algorithm":
        algo_dropdown.disabled = False
    else:
        algo_dropdown.disabled = True

# --- Auto target detection ---
def auto_detect_targets(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [numeric_cols[-1]] if numeric_cols else []

def on_class_select_change(change):
    global df, current_model, encoders
    output.clear_output()
    with output:
        selected_class = change['new']
        if not selected_class:
            df = None
            target_col.options = []
            features_select.options = []
            return

        # Use the provided selected_class as the base class name
        base_class = get_base_class_name(selected_class)
        print(f"Loading data from class: {base_class}")

        try:
            # Find the dataset folder whose class_{...} subfolder matches
            matching_folders = []
            for p in BASE_DIR.iterdir():
                if not p.is_dir():
                    continue
                # look for class_ subfolders under p
                if any(sub.is_dir() and sub.name == f"class_{base_class}" for sub in p.iterdir()):
                    matching_folders.append(p)
                # also accept folder named exactly smart_name
                if p.name == base_class or p.name.endswith(base_class):
                    matching_folders.append(p)

            if not matching_folders:
                raise FileNotFoundError(f"No folder found for class '{selected_class}'")

            # prefer folder that contains train/train.csv
            data_folder = None
            for cand in matching_folders:
                if (cand / f"class_{base_class}" / "train" / "train.csv").exists():
                    data_folder = cand
                    break
            if data_folder is None:
                data_folder = matching_folders[0]

            # Prefer to load train.csv (clean split) when present; else data.csv
            train_file = data_folder / f"class_{base_class}" / "train" / "train.csv"
            data_file = data_folder / "data.csv"
            if train_file.exists():
                df = pd.read_csv(train_file)
                print(f"Loaded train.csv for class '{base_class}'.")
            elif data_file.exists():
                df = pd.read_csv(data_file)
                print(f"Loaded data.csv for class '{base_class}'.")
            else:
                raise FileNotFoundError(f"No data file found for class '{base_class}' (expected train.csv or data.csv)")

            # load saved metadata if available
            metadata, saved_target, saved_features = safe_load_metadata(base_class)

            # update selectors
            target_col.options = df.columns.tolist()
            features_select.options = df.columns.tolist()

            if metadata:
                # prefer the 'target' key saved earlier
                if saved_target:
                    target_col.value = tuple([saved_target]) if saved_target in df.columns else tuple()
                else:
                    target_col.value = tuple(auto_detect_targets(df))
                if saved_features:
                    # filter saved_features to those present in df
                    valid_saved_features = [f for f in saved_features if f in df.columns]
                    features_select.value = tuple(valid_saved_features) if valid_saved_features else tuple(df.columns.tolist())
                else:
                    features_select.value = tuple(df.columns.tolist())

                # Attempt to load model
                current_model = load_model_for_class(base_class)
                if current_model:
                    print("✅ Model and metadata loaded successfully.")
                else:
                    print("⚠️ Metadata present but model could not be loaded. A new model will be trained on run.")
                    current_model = None
            else:
                # no metadata -> auto-detect
                auto_targets = auto_detect_targets(df)
                auto_features = [c for c in df.columns if c not in auto_targets]
                target_col.value = tuple(auto_targets)
                features_select.value = tuple(auto_features)
                current_model = None
                print("⚠️ No saved model or metadata found. Auto-selected targets/features.")

            print(f"DataFrame loaded. Shape: {df.shape}")

        except Exception as e:
            print(f"❌ An ERROR occurred loading class data: {e}")
            df = None
            target_col.options = []
            features_select.options = []

def _save_train_test_split(base_class_name, df_full, final_features, targets, X_train, X_test, y_train, y_test):

    try:
        # Determine class folder: find a dataset folder in BASE_DIR that contains class_{base_class_name}
        matching = [p for p in BASE_DIR.iterdir() if p.is_dir() and f"class_{base_class_name}" in [s.name for s in p.iterdir() if p.exists()]]
        # If not found by complicated check, fallback to direct path assumption:
        dataset_folder = BASE_DIR / base_class_name
        # Prefer an existing dataset that contains class_{base_class_name}
        for p in BASE_DIR.iterdir():
            if p.is_dir() and f"class_{base_class_name}" in "".join([x.name for x in p.iterdir()]):
                dataset_folder = p
                break

        class_folder = dataset_folder / f"class_{base_class_name}"
        train_folder = class_folder / "train"
        test_folder = class_folder / "test"
        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        # Compose DataFrames
        # X_train / X_test might be numpy arrays if user used .values — ensure DataFrames with correct columns
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=final_features)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=final_features)

        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train_df = pd.DataFrame(y_train).reset_index(drop=True)
        else:
            y_train_df = pd.DataFrame(y_train, columns=targets)

        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
        else:
            y_test_df = pd.DataFrame(y_test, columns=targets)

        train_combined = pd.concat([X_train.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
        test_combined = pd.concat([X_test.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)

        train_combined.to_csv(train_folder / "train.csv", index=False)
        test_combined.to_csv(test_folder / "test.csv", index=False)

    except Exception as e:
        # best-effort: if saving fails, do not crash the main flow
        print(f"⚠️ Warning: could not save train/test split: {e}")

def calculate_feature_similarity(new_df, meta_path):

    if not os.path.exists(meta_path):
        return 0.0  # No metadata found → train new model

    with open(meta_path, 'r') as f:
        old_meta = json.load(f)

    similarities = []
    for col in new_df.columns:
        if col in old_meta["feature_stats"]:
            # Compare means & std via Wasserstein distance (distribution similarity)
            old_stats = old_meta["feature_stats"][col]
            new_mean, new_std = new_df[col].mean(), new_df[col].std()
            dist_mean = wasserstein_distance([old_stats["mean"]], [new_mean])
            dist_std = wasserstein_distance([old_stats["std"]], [new_std])

            sim = 1 - (dist_mean + dist_std) / (abs(new_mean) + abs(new_std) + 1e-5)
            similarities.append(max(0, min(sim, 1)))  # Clamp to [0,1]
        else:
            similarities.append(0.0)

    return np.mean(similarities) if similarities else 0.0

def adaptive_model_training(new_df, model_path, meta_path, train_func, retrain_func):

    similarity_score = calculate_feature_similarity(new_df, meta_path)
    print(f"🔍 Similarity Score: {similarity_score:.2f}")

    if similarity_score >= 0.9 and os.path.exists(model_path):
        print("♻️ Reusing existing model – no retraining needed.")
        model = joblib.load(model_path)

    elif similarity_score >= 0.7 and os.path.exists(model_path):
        print("🛠️ Partial retraining model with new dataset...")
        model = retrain_func(model_path, new_df)
        joblib.dump(model, model_path)

    else:
        print("🆕 Training new model from scratch...")
        model = train_func(new_df)
        joblib.dump(model, model_path)

    return model

from google.colab import files
import os

def save_and_download_excel(backtest_results_df, forecast_df, dataset_path):
    # Extract dataset base name (without extension)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_filename = f"{dataset_name}_forecast_output.xlsx"

    # Save to Excel
    with ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        backtest_results_df.to_excel(writer, sheet_name='Backtest', index=False)
        forecast_df.to_excel(writer, sheet_name='Forecast', index=False)

    print(f"✅ Excel forecast output saved as '{output_filename}'")
    files.download(output_filename)  # Direct download

from pathlib import Path
from pandas import ExcelWriter
import os

def plot_history_forecast(history, forecast):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # Plot history at indices 0..len(history)-1
    plt.plot(range(len(history)), history, label="History")

    # Plot forecast starting right after history
    start_idx = len(history)
    forecast_x = list(range(start_idx, start_idx + len(forecast)))
    plt.plot(forecast_x, forecast, label="Forecast")

    plt.legend()
    plt.show()

# --- Scenario Simulation Example ---
def simulate_scenario(input_df, modifications: dict):

    df_mod = input_df.copy()
    for k, v in modifications.items():
        if k in df_mod.columns:
            df_mod[k] = v
    return current_model.predict(df_mod)

# --- Helper metrics ---
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return np.mean(diff) * 100

def theils_u(y_true, y_pred):
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return num / denom

def naive_forecast(y_test):
    # Simple naive forecast: last observed value
    last_val = y_test.shift(1).fillna(method='bfill')
    return last_val.values

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_forecast(original_df, forecast_df, targets):
    results = {}
    for t in targets:
        y_true = original_df[t].values[-len(forecast_df):]
        y_pred = forecast_df[f'Forecasted_{t}'].values[:len(y_true)]
        results[t] = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "R2": r2_score(y_true, y_pred)
        }
    return results

def get_dataset_hash(df, features, targets):
    """Generate a short hash based on features, targets, and numeric sample."""
    numeric_sample = df[features + targets].select_dtypes(include=np.number).head(10_000)
    df_bytes = numeric_sample.to_csv(index=False).encode('utf-8')
    return hashlib.md5(df_bytes).hexdigest()[:8]

def train_models(X_train, y_train):


    models = {}

    # Classical: ARIMA
    if y_train.shape[1] == 1:
        try:
            arima = ARIMA(y_train.iloc[:, 0], order=(2,1,2)).fit()
            models["ARIMA"] = arima
        except Exception as e:
            print(f"⚠️ ARIMA failed: {e}")

    # Classical: ETS (Exponential Smoothing)
    try:
        ets = ExponentialSmoothing(y_train, seasonal="add", seasonal_periods=12).fit()
        models["ETS"] = ets
    except Exception as e:
        print(f"⚠️ ETS failed: {e}")

    # ML: Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # ML: Gradient Boosted Trees
    gbt = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    gbt.fit(X_train, y_train)
    models["GBT"] = gbt

    # DL: Simple MLP (fast alternative to LSTM in notebook environment)
    mlp = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    models["MLP"] = mlp

    return models

def evaluate_models(models, X_test, y_test):

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    results = {}
    naive_forecast = np.full_like(y_test, y_test.iloc[-1])  # naive baseline

    for name, model in models.items():
        try:
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:  # ARIMA, ETS
                y_pred = model.forecast(len(y_test))

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
            smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-9))

            # Theil's U
            num = np.sqrt(np.mean((y_pred - y_test) ** 2))
            den = np.sqrt(np.mean((naive_forecast - y_test) ** 2))
            theils_u = num / den if den != 0 else np.nan

            results[name] = {
                "MSE": mse, "MAE": mae, "R²": r2, "MAPE": mape, "SMAPE": smape, "Theil's U": theils_u
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results

def ensemble_predictions(models, X_test, method="mean"):

    preds = []

    for name, model in models.items():
        try:
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.forecast(len(X_test))
            preds.append(y_pred)
        except Exception:
            continue

    preds = np.array(preds)
    if method == "mean":
        return np.mean(preds, axis=0)
    elif method == "median":
        return np.median(preds, axis=0)
    else:
        raise ValueError("Unknown ensemble method")

def prediction_intervals(y_pred, error_std, confidence=0.95):

    import scipy.stats as st
    z = st.norm.ppf((1 + confidence) / 2)  # 95% -> 1.96
    lower = y_pred - z * error_std
    upper = y_pred + z * error_std
    return lower, upper

import json, hashlib, os
META_DIR = Path("dataset_metadata")
META_DIR.mkdir(exist_ok=True)

def save_dataset_meta(df, targets, features, dataset_name):
    meta = {
        "columns": df.columns.tolist(),
        "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
        "targets": targets,
        "features": features,
        "hash": hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    }
    with open(META_DIR/f"{dataset_name}.json","w") as f:
        json.dump(meta,f)

def load_all_dataset_meta():
    metas = []
    for f in META_DIR.glob("*.json"):
        with open(f) as ff:
            metas.append(json.load(ff))
    return metas

# --- Updated run_forecast ---
def run_forecast(b):
    global df, encoders, current_model
    output.clear_output()
    with output:
        try:
            print("🔄 Running forecast...")

            # --- Initial Checks ---
            targets = list(target_col.value)
            features = list(features_select.value)

            if df is None:
                print("❌ No data available. Please upload and process a file or select a class.")
                return
            if not targets:
                print("❌ Please select at least one target column.")
                return
            if mode_select.value != "Manual" and 'auto_suggest_chk' in globals() and auto_suggest_chk.value:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                flag_cols = [c for c in numeric_cols if any(x in c.lower() for x in ['is_', 'flag', 'promo'])]
                candidate_cols = [c for c in numeric_cols if c not in flag_cols]

                if candidate_cols:
                    # --- Step 3a: Historical AI knowledge ---
                    previous_meta = load_all_dataset_meta()
                    target_scores = {}
                    for meta in previous_meta:
                        for t in meta.get("targets", []):
                            target_scores[t] = target_scores.get(t,0)+1

                    # Normalize scores
                    max_score = max(target_scores.values()) if target_scores else 1
                    for k in target_scores:
                        target_scores[k] /= max_score

                    # --- Step 3b: Correlation & autocorr scoring ---
                    correlations = df[candidate_cols].corr().abs()
                    mean_corr = correlations.mean().sort_values(ascending=False)
                    variances = df[candidate_cols].var()
                    low_var_cols = variances[variances < 1e-5].index.tolist()
                    ranked_cols = [c for c in mean_corr.index if c not in low_var_cols]

                    autocorr_scores = {}
                    for col in ranked_cols:
                        series = df[col].dropna()
                        autocorr_scores[col] = series.autocorr(lag=1) if len(series)>1 else 0

                    # --- Step 3c: Combine historical + current scores ---
                    combined_score = {}
                    for c in ranked_cols:
                        hist_score = target_scores.get(c, 0)
                        combined_score[c] = 0.5*hist_score + 0.5*(abs(autocorr_scores[c]))

                    sorted_targets = sorted(combined_score, key=lambda k: combined_score[k], reverse=True)

                    # Multi-target redundancy check
                    final_targets = []
                    used_cols = set()
                    for t in sorted_targets:
                        if all(correlations[t].get(ft,0) < 0.95 for ft in used_cols):
                            final_targets.append(t)
                            used_cols.add(t)
                        if len(final_targets) >= min(3, len(sorted_targets)):
                            break

                    # Update UI selections
                    target_col.options = candidate_cols
                    target_col.value = tuple(final_targets)
                    features_select.options = [c for c in df.columns if c not in final_targets]
                    features_select.value = tuple(features_select.options)

                    print(f"💡 AI-suggested targets (historical + autocorr): {final_targets}")
                    targets = list(target_col.value)
                    features = list(features_select.value)

                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

                # Exclude typical "flag/indicator" columns from automatic targets
                flag_cols = [c for c in numeric_cols if any(x in c.lower() for x in ['is_', 'flag', 'promo'])]
                candidate_cols = [c for c in numeric_cols if c not in flag_cols]

                if candidate_cols:
                    # --- 1. Correlation & Variance ---
                    correlations = df[candidate_cols].corr().abs()
                    mean_corr = correlations.mean().sort_values(ascending=False)
                    variances = df[candidate_cols].var()
                    low_var_cols = variances[variances < 1e-5].index.tolist()
                    ranked_cols = [c for c in mean_corr.index if c not in low_var_cols]

                    # --- 2. Time Series Predictability ---
                    # Compute basic autocorrelation (lag-1) as a measure of predictability
                    autocorr_scores = {}
                    for col in ranked_cols:
                        series = df[col].dropna()
                        if len(series) > 1:
                            autocorr_scores[col] = series.autocorr(lag=1)
                        else:
                            autocorr_scores[col] = 0
                    # Combine mean correlation + autocorr as final score
                    combined_score = {c: mean_corr[c]*0.5 + abs(autocorr_scores[c])*0.5 for c in ranked_cols}
                    sorted_targets = sorted(combined_score, key=lambda k: combined_score[k], reverse=True)

                    # --- 3. Multi-target redundancy check ---
                    final_targets = []
                    used_cols = set()
                    for t in sorted_targets:
                        # Skip if highly correlated with already selected target
                        if all(correlations[t].get(ft,0) < 0.95 for ft in used_cols):
                            final_targets.append(t)
                            used_cols.add(t)
                        if len(final_targets) >= min(3, len(sorted_targets)):
                            break

                    # --- 4. Feature selection ---
                    target_col.options = candidate_cols
                    target_col.value = tuple(final_targets)
                    features_select.options = [c for c in df.columns if c not in final_targets]
                    features_select.value = tuple(features_select.options)

                    print(f"💡 AI-suggested targets (smarter): {final_targets}")
                    targets = list(target_col.value)
                    features = list(features_select.value)
            # --- Work on copy & preprocessing ---
            df_copy = df.copy()
            for t in targets:
                df_copy[t] = pd.to_numeric(df_copy[t], errors='coerce')
            df_copy.dropna(subset=targets, inplace=True)

            if drop_na_chk.value:
                df_copy.dropna(inplace=True)
            elif fill_na_chk.value:
                df_copy.fillna(0, inplace=True)

            df_copy = build_lags_rollings(df_copy, targets, max_lag=3, rolling_windows=[3,6])

            date_col = 'Date' if 'Date' in df_copy.columns else None
            if date_col:
                try:
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                except Exception:
                    date_col = None

            time_col_candidates = ['Month','month','Period','period']
            time_col = None
            for col in time_col_candidates:
                if col in df_copy.columns:
                    time_col = col
                    df_copy = add_cyclical_time_features(df_copy, time_col)
                    break

            df_copy.dropna(inplace=True)
            to_remove = remove_highly_correlated_features(df_copy, targets, threshold=0.95)
            if to_remove:
                print(f"ℹ️ Removed highly correlated features: {to_remove}")

            user_features = [f for f in features if f not in targets and f not in to_remove and f != date_col]
            lag_roll_features = [f"{tcol}_lag{lag}" for tcol in targets for lag in range(1,4)] + \
                                [f"{tcol}_roll3" for tcol in targets] + [f"{tcol}_roll6" for tcol in targets]
            cyclical_cols = [f"{time_col}_sin", f"{time_col}_cos"] if time_col else []
            final_features = list(OrderedDict.fromkeys(user_features + lag_roll_features + cyclical_cols))
            final_features = [f for f in final_features if f in df_copy.columns]
            if not final_features:
                print("❌ No valid features available after preprocessing. Aborting.")
                return

            X = df_copy[final_features].copy()
            y = df_copy[targets].copy()

            # Save metadata for AI training
            dataset_name = class_select.value or f"{targets[0]}_{abs(hash(frozenset(final_features))) % (10**8)}"
            save_dataset_meta(df_copy, targets, final_features, dataset_name)

            encoders.clear()
            for col in X.select_dtypes(include='object').columns.tolist():
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            # --- Model Loading / Training with caching ---
            dataset_hash = get_dataset_hash(df_copy, final_features, targets)
            model_filename = TRAINED_MODELS_DIR / f"model_{dataset_hash}.pkl"

            best_model = None

            if model_filename.exists():
                # Load cached model
                best_model = joblib.load(model_filename)
                print(f"✅ Reusing cached model for this dataset (hash={dataset_hash}).")
            else:
                # Train new model
                print("🛠️ Training a new model...")
                if algo_dropdown.value == "RandomForest":
                    base_est = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
                    best_model = MultiOutputRegressor(base_est)
                elif algo_dropdown.value == "XGBoost":
                    base_est = XGBRegressor(
                        n_estimators=200, max_depth=6, random_state=42,
                        objective='reg:squarederror', n_jobs=-1
                    )
                    best_model = MultiOutputRegressor(base_est)
                elif algo_dropdown.value == "TFT (Deep Learning)":
                    print("🧠 Using TFT / placeholder")
                    base_est = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
                    best_model = MultiOutputRegressor(base_est)

                # Fit model
                best_model.fit(X, y)
                joblib.dump(best_model, model_filename)
                print(f"✅ Model training complete and cached (hash={dataset_hash}).")

            current_model = best_model

            # --- Evaluation & Validation ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            preds = best_model.predict(X_test)
            for i, tcol in enumerate(targets):
                mae = mean_absolute_error(y_test.iloc[:,i], preds[:,i])
                rmse = np.sqrt(mean_squared_error(y_test.iloc[:,i], preds[:,i]))
                r2 = r2_score(y_test.iloc[:,i], preds[:,i])
                smp = smape(y_test.iloc[:,i].values, preds[:,i])
                tu = theils_u(y_test.iloc[:,i].values, preds[:,i])
                naive_preds = naive_forecast(y_test.iloc[:,i])
                print(f"[{tcol}] MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.3f} | SMAPE={smp:.2f}% | Theil's U={tu:.3f}")

            # --- Residual Plots ---
            max_points_plot = 20000
            for i, tcol in enumerate(targets):
                resid = y_test.iloc[:, i].values - preds[:, i]
                if len(resid) > max_points_plot:
                    idx = np.random.default_rng(42).choice(len(resid), size=max_points_plot, replace=False)
                    x_plot = preds[:,i][idx]
                    y_plot = resid[idx]
                else:
                    x_plot = preds.iloc[:,i] if hasattr(preds,'iloc') else preds[:,i]
                    y_plot = resid
                min_len = min(len(x_plot), len(y_plot))
                if min_len>1:
                    x_plot = x_plot[:min_len]
                    y_plot = y_plot[:min_len]
                    plt.figure()
                    sns.residplot(x=x_plot, y=y_plot, lowess=True)
                    plt.title(f"Residual Plot: {tcol}")
                    plt.xlabel("Predicted")
                    plt.ylabel("Residual")
                    plt.grid(True)
                    plt.show()
                else:
                    print(f"⚠️ Not enough data points to plot residuals for {tcol}.")

            # Save train/test splits
            save_name = class_select.value or f"{targets[0]}_{abs(hash(frozenset(final_features))) % (10**8)}"
            _save_train_test_split(save_name, df_copy, final_features, targets, X_train, X_test, y_train, y_test)

            # --- Backtest Results display ---
            backtest_results_df = pd.DataFrame(y_test.index.values, columns=['Index'])
            for i, tcol in enumerate(targets):
                backtest_results_df[f'Original_{tcol}'] = y_test.iloc[:, i].values
                backtest_results_df[f'Predicted_{tcol}'] = preds[:, i]
                backtest_results_df[f'Residual_{tcol}'] = backtest_results_df[f'Original_{tcol}'] - backtest_results_df[f'Predicted_{tcol}']

            core_features_to_display = [f for f in ['PromoFlag', 'IsWeekend', 'IsHoliday', 'CompetitorPrice'] if f in df_copy.columns]
            for feature in core_features_to_display:
                if feature in X_test.columns:
                    backtest_results_df[feature] = df_copy.loc[y_test.index, feature].values

            display(backtest_results_df.head(10))

            # --- Forecasting Future Periods ---
            f_periods = int(forecast_slider.value)
            forecast_data_list = []
            promo_dict = parse_promo_per_target(promo_per_target.value)
            promo_dict = {k: float(v) for k, v in promo_dict.items()} if promo_dict else {}
            global_promo = float(promo_slider.value) if promo_slider.value is not None else 0.0

            context_cols = [c for c in df.columns if c not in targets + final_features]
            history = {t: list(df_copy[t].astype(float).iloc[-6:]) for t in targets}
            last_known_state = df_copy[final_features + targets].iloc[-1].to_dict()

            print("🔮 Generating future forecast...")
            for step in range(f_periods):
                input_data = {}
                for feature in final_features:
                    input_data[feature] = last_known_state.get(feature, 0)
                for tcol in targets:
                    input_data[f"{tcol}_roll3"] = float(np.mean(history[tcol][-3:])) if len(history[tcol]) >= 3 else 0.0
                    input_data[f"{tcol}_roll6"] = float(np.mean(history[tcol][-6:])) if len(history[tcol]) >= 6 else 0.0
                    for lag in range(1, 4):
                        input_data[f"{tcol}_lag{lag}"] = float(history[tcol][-lag]) if len(history[tcol]) >= lag else 0.0

                if time_col and time_col in last_known_state:
                    next_period = float(last_known_state.get(time_col, 0)) + 1.0
                    input_data[time_col] = next_period
                    denom = (df[time_col].max() if time_col in df.columns else (len(df) + f_periods))
                    input_data[f"{time_col}_sin"] = float(np.sin(2 * np.pi * next_period / (denom + f_periods)))
                    input_data[f"{time_col}_cos"] = float(np.cos(2 * np.pi * next_period / (denom + f_periods)))
                    last_known_state[time_col] = next_period

                input_df = pd.DataFrame([input_data]).reindex(columns=final_features, fill_value=0)
                for col in input_df.select_dtypes(include='object').columns:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                    else:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

                forecast_vals = best_model.predict(input_df)[0]
                forecast_vals = [float(v) for v in np.array(forecast_vals).reshape(-1)]

                for i, tcol in enumerate(targets):
                    history[tcol].append(forecast_vals[i])
                    last_known_state[tcol] = forecast_vals[i]

                forecast_dict = {}
                for i, t in enumerate(targets):
                    uplift = promo_dict.get(t, global_promo)
                    forecast_dict[f"Forecasted_{t}"] = float(forecast_vals[i] * (1.0 + float(uplift)))

                forecast_dict['Period'] = len(df_copy) + step + 1
                if date_col:
                    try:
                        last_date = pd.to_datetime(df[date_col].iloc[-1])
                        forecast_dict['Forecast_Date'] = last_date + pd.DateOffset(days=step + 1)
                    except Exception:
                        pass

                for col in context_cols:
                    forecast_dict[col] = df[col].iloc[-1]

                for cat_col, le in encoders.items():
                    try:
                        last_label = str(df[cat_col].iloc[-1])
                    except Exception:
                        last_label = ""
                    forecast_dict[f"{cat_col}_label"] = last_label

                forecast_data_list.append(forecast_dict)

            forecast_df = pd.DataFrame(forecast_data_list)
            print("\n📈 Visualizing Forecast:")
            plot_dynamic_forecast(df, forecast_df, targets)

            # --- Save outputs to Excel ---
            import os
            from google.colab import files
            from pathlib import Path

            def _resolve_dataset_name():
                if 'file_path' in globals():
                    try:
                        return Path(globals()['file_path']).stem
                    except Exception:
                        pass
                try:
                    if upload.value:
                        fc = next(iter(upload.value.values()))
                        return Path(fc['metadata']['name']).stem
                except Exception:
                    pass
                if class_select.value:  # ✅ use class_select instead of base_class_name
                    return class_select.value
                ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                return f"forecast_{ts}"  # ✅ safer, doesn’t rely on targets[0]

            dataset_name = _resolve_dataset_name()
            for col in [c for c in forecast_df.columns if c.endswith('_label')] + context_cols:
                if col in forecast_df.columns:
                    forecast_df[col] = forecast_df[col].astype(str)

            from google.colab import drive
            drive.mount('/content/drive')

            output_filename = f"/content/drive/My Drive/{dataset_name}_forecast_output.xlsx"

            with ExcelWriter(output_filename, engine='xlsxwriter') as writer:
                backtest_results_df.to_excel(writer, sheet_name='Backtest', index=False)
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)

            print(f"✅ Excel forecast output saved to Google Drive: {output_filename}")


            print(f"✅ Excel forecast output saved as '{output_filename}'")
            files.download(output_filename)

        except Exception as e:
            print(f"❌ ERROR: {e}")

# --- Widget Connections ---
process_button.on_click(on_process_clicked)
preview_button.on_click(on_preview_clicked)
use_manual_targets.observe(on_manual_select_change, names='value')
mode_select.observe(on_mode_select_change, names='value')
class_select.observe(on_class_select_change, names='value')
run_button.on_click(run_forecast)

# --- Display UI ---
display(widgets.VBox([
    HBox([file_box, class_select]),  # only file_box, not a separate upload
    HBox([process_button, preview_button]),
    use_manual_targets,
    mode_select,
    algo_dropdown,
    HBox([target_col, features_select]),
    drop_na_chk, fill_na_chk, forecast_slider, promo_slider, promo_per_target,
    run_button
]))

display(output)