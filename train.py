"""Train an LSTM to forecast hourly traffic volume.

Examples
--------
python train.py                                   # downloads the UCI dataset
python train.py --data Metro_Interstate_Traffic_Volume.csv.gz
python train.py --synthetic --epochs 5            # quick pipeline check, no download
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import traffic_utils as tu


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=tu.UCI_URL, help="path or URL of the traffic CSV (.csv or .csv.gz)")
    p.add_argument("--synthetic", action="store_true", help="use generated data instead of the real dataset")
    p.add_argument("--days", type=int, default=0, help="use only the last N days (0 = all data)")
    p.add_argument("--window", type=int, default=24, help="hours of history fed to the LSTM")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="outputs", help="folder for plots, metrics and the saved model")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(42)

    # 1. Data ---------------------------------------------------------------
    raw = tu.make_synthetic_data() if args.synthetic else tu.load_raw(args.data)
    df = tu.build_hourly(raw)
    if args.days > 0:
        df = df.iloc[-args.days * 24:]
    print(f"Hourly rows: {len(df)}  ({df.index[0]} -> {df.index[-1]})  features: {df.shape[1]}")

    # 2. Chronological split; the scaler only ever sees training data ---------
    positions, tr, va, te = tu.split_masks(len(df), args.window)
    n_train = int(len(df) * 0.70)
    scaler = MinMaxScaler().fit(df.values[:n_train])
    scaled = scaler.transform(df.values)
    X, y = tu.make_windows(scaled, args.window)
    X_tr, y_tr, X_va, y_va, X_te = X[tr], y[tr], X[va], y[va], X[te]
    print(f"Windows -> train {len(X_tr)}, validation {len(X_va)}, test {len(X_te)}")

    # 3. Model ----------------------------------------------------------------
    import tensorflow as tf
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(args.window, X.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                    restore_best_weights=True)],
    )
    model.save(os.path.join(args.out, "model.keras"))

    # 4. Evaluation on the held-out test period (original units) --------------
    lo, span = scaler.data_min_[0], scaler.data_range_[0]
    traffic = df[tu.TARGET].values
    pos_te = positions[te]
    y_true = traffic[pos_te]
    lstm_pred = np.ravel(model.predict(X_te, batch_size=256, verbose=0)) * span + lo

    results = {"LSTM": tu.regression_metrics(y_true, lstm_pred)}
    baselines = tu.baseline_predictions(traffic, pos_te)
    for name, pred in baselines.items():
        results[name] = tu.regression_metrics(y_true, pred)

    print("\nTest-set results (vehicles per hour)")
    print(f"{'Model':40s} {'MAE':>9s} {'RMSE':>9s} {'MSE':>12s}")
    for name, m in results.items():
        print(f"{name:40s} {m['MAE']:9.1f} {m['RMSE']:9.1f} {m['MSE']:12.0f}")

    summary = {
        "rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
        "window_hours": args.window, "epochs_run": len(history.history["loss"]),
        "test_windows": int(te.sum()), "results": results,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Plots ----------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], marker="o", label="train")
    plt.plot(history.history["val_loss"], marker="o", label="validation")
    plt.title("Training loss vs epochs"); plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "training_loss.png"), dpi=150); plt.close()

    n_show = min(24 * 7, len(y_true))
    idx = df.index[pos_te][:n_show]
    plt.figure(figsize=(11, 4))
    plt.plot(idx, y_true[:n_show], label="actual")
    plt.plot(idx, lstm_pred[:n_show], label="LSTM")
    plt.plot(idx, baselines["Seasonal naive (same hour yesterday)"][:n_show],
             label="same hour yesterday", alpha=0.6)
    plt.title("Test period: actual vs predicted traffic volume (first 7 days)")
    plt.ylabel("Vehicles per hour"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "test_predictions.png"), dpi=150); plt.close()

    future = tu.forecast_next(df, lambda w: model.predict(w, verbose=0), scaler, args.window, steps=24)
    recent = df[tu.TARGET].iloc[-72:]
    plt.figure(figsize=(11, 4))
    plt.plot(recent.index, recent.values, label="last 72 h (actual)")
    plt.plot(future.index, future.values, marker="o", label="next 24 h (forecast)")
    plt.title("Traffic volume forecast"); plt.ylabel("Vehicles per hour")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "next_24_hours.png"), dpi=150); plt.close()
    future.to_csv(os.path.join(args.out, "forecast_next_24h.csv"))
    print(f"\nSaved model, metrics.json, 3 plots and the forecast to {args.out}/")


if __name__ == "__main__":
    main()
