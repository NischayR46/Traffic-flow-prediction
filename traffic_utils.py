"""Data loading, cleaning, features, windowing, baselines and forecasting helpers.

Nothing in this file needs TensorFlow, so it can be tested on its own.
"""
import numpy as np
import pandas as pd

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00492/"
           "Metro_Interstate_Traffic_Volume.csv.gz")
TARGET = "traffic_volume"
TIME_COLS = ["is_weekend", "Day_sin", "Day_cos", "Week_sin", "Week_cos",
             "Year_sin", "Year_cos"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(source):
    """Read the Metro Interstate Traffic Volume CSV (local path or URL, .gz is fine)."""
    return pd.read_csv(source, parse_dates=["date_time"], index_col="date_time")


def make_synthetic_data(days=730, seed=42):
    """Fake hourly traffic with the same columns as the UCI file.

    Only for testing the pipeline when the real dataset is not available.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=days * 24, freq="h")
    hour, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    rush = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2) + np.exp(-0.5 * ((hour - 17) / 2.0) ** 2)
    weekday = dow < 5
    base = 1200 + 3600 * rush * np.where(weekday, 1.0, 0.45)
    base += 800 * np.exp(-0.5 * ((hour - 13) / 3) ** 2) * np.where(weekday, 0.3, 1.0)
    base *= 0.9 + 0.1 * np.sin(2 * np.pi * doy / 365.25)
    weather = rng.choice(["Clear", "Clouds", "Rain", "Snow", "Mist"], size=len(idx),
                         p=[0.4, 0.3, 0.15, 0.05, 0.1])
    base = base * np.where(weather == "Rain", 0.92, 1.0) * np.where(weather == "Snow", 0.8, 1.0)
    traffic = np.clip(base + rng.normal(0, 250, len(idx)), 0, None).round()
    df = pd.DataFrame({
        "holiday": np.full(len(idx), "None"),
        "temp": 283 + 12 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2, len(idx)),
        "rain_1h": np.where(weather == "Rain", rng.exponential(1.0, len(idx)), 0.0),
        "snow_1h": np.where(weather == "Snow", rng.exponential(0.1, len(idx)), 0.0),
        "clouds_all": rng.integers(0, 101, len(idx)),
        "weather_main": weather,
        "weather_description": weather,
        "traffic_volume": traffic,
    }, index=idx)
    df.index.name = "date_time"
    return df


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def time_features(index):
    """Weekend flag plus cyclical (sin/cos) encodings of hour, weekday and day of year."""
    hour = index.hour.values + index.minute.values / 60
    dow = index.dayofweek.values
    doy = index.dayofyear.values
    return pd.DataFrame({
        "is_weekend": (dow >= 5).astype(float),
        "Day_sin": np.sin(2 * np.pi * hour / 24),
        "Day_cos": np.cos(2 * np.pi * hour / 24),
        "Week_sin": np.sin(2 * np.pi * (dow + hour / 24) / 7),
        "Week_cos": np.cos(2 * np.pi * (dow + hour / 24) / 7),
        "Year_sin": np.sin(2 * np.pi * doy / 365.25),
        "Year_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)


def build_hourly(raw):
    """Clean the raw data and return one row per hour, with traffic_volume as column 0."""
    df = raw.copy()
    # Sensor errors in the raw file: 0 K temperatures and one absurd rainfall spike.
    df["temp"] = df["temp"].where(df["temp"] > 0)
    df["rain_1h"] = df["rain_1h"].where(df["rain_1h"] < 500)

    df["is_holiday"] = (df["holiday"].astype(str) != "None").astype(float)
    weather = pd.get_dummies(df["weather_main"], prefix="weather").astype(float)
    numeric = df[[TARGET, "temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday"]].astype(float)
    df = pd.concat([numeric, weather], axis=1)

    # The raw file has duplicate timestamps and gaps; averaging per hour and
    # interpolating gives a regular hourly series.
    df = df.sort_index().resample("1h").mean().interpolate(limit_direction="both")
    df["is_holiday"] = (df["is_holiday"] >= 0.5).astype(float)
    return pd.concat([df, time_features(df.index)], axis=1)


# --------------------------------------------------------------------------
# Windowing and splitting
# --------------------------------------------------------------------------
def make_windows(values, width):
    """X[i] = rows i..i+width-1, y[i] = traffic (column 0) at row i+width."""
    swv = np.lib.stride_tricks.sliding_window_view(values, width, axis=0)  # (n, feat, width)
    X = swv[:-1].transpose(0, 2, 1).astype("float32")
    y = values[width:, 0].astype("float32")
    return X, y


def split_masks(n_rows, width, train=0.70, val=0.15):
    """Chronological split of the window targets (no shuffling, no leakage from the future)."""
    positions = np.arange(width, n_rows)
    n_train, n_val = int(n_rows * train), int(n_rows * val)
    return (positions, positions < n_train,
            (positions >= n_train) & (positions < n_train + n_val),
            positions >= n_train + n_val)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mse = float(np.mean(err ** 2))
    return {"MAE": float(np.mean(np.abs(err))), "MSE": mse, "RMSE": float(np.sqrt(mse))}


def baseline_predictions(traffic, positions):
    """Simple forecasts the LSTM has to beat."""
    return {
        "Persistence (same as 1 h ago)": traffic[positions - 1],
        "Seasonal naive (same hour yesterday)": traffic[positions - 24],
    }


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
def forecast_next(df, predict_fn, scaler, width, steps=24):
    """Recursive multi-step forecast of traffic volume.

    predict_fn takes a scaled window of shape (1, width, n_features) and returns the
    scaled next-hour traffic. Calendar features of future hours are computed exactly;
    weather is unknown, so the last observed weather is held constant.
    """
    lo, span = scaler.data_min_[0], scaler.data_range_[0]
    hist = df.iloc[-width:].copy()
    times, values = [], []
    for _ in range(steps):
        window = scaler.transform(hist.values)[None].astype("float32")
        pred_scaled = float(np.ravel(predict_fn(window))[0])
        value = max(pred_scaled * span + lo, 0.0)

        ts = hist.index[-1] + pd.Timedelta(hours=1)
        row = hist.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex([ts])
        row[TARGET] = value
        row[TIME_COLS] = time_features(row.index)[TIME_COLS].values
        hist = pd.concat([hist.iloc[1:], row])
        times.append(ts)
        values.append(value)
    return pd.Series(values, index=pd.DatetimeIndex(times), name="forecast")
