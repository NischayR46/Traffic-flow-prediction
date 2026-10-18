# AI-Based Vehicle Traffic Flow Prediction (LSTM)

Forecasts hourly traffic volume on a highway using an LSTM neural network trained on past traffic and weather data. Final-year B.E. project, Computer Science & Engineering, Bangalore Institute of Technology (2025-26).

## Dataset
[Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (UCI Machine Learning Repository): hourly westbound traffic volume on I-94 with weather and holiday information.

## Approach
1. **Cleaning** - fix impossible sensor values (0 K temperature, a rainfall spike), merge duplicate timestamps, fill gaps to get one row per hour
2. **Features** - traffic volume, temperature, rain, snow, cloud cover, one-hot weather type, holiday and weekend flags, and cyclical (sin/cos) encodings of hour, weekday and day of year
3. **Windowing** - the previous 24 hours are used to predict the next hour
4. **Model** - LSTM(64) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1), trained with Adam on MSE, with early stopping on a validation set
5. **Evaluation** - chronological 70/15/15 train/validation/test split (no shuffling), scaler fitted on training data only; MAE, MSE and RMSE reported in vehicles per hour and compared with two simple baselines (same as one hour ago, same hour yesterday)
6. **Forecast** - the next 24 hours are predicted recursively

## Results
Run `python train.py`, then copy your numbers from `outputs/metrics.json`:

| Model | MAE | RMSE |
|---|---|---|
| LSTM | _fill in_ | _fill in_ |
| Persistence (1 h ago) | _fill in_ | _fill in_ |
| Seasonal naive (same hour yesterday) | _fill in_ | _fill in_ |

Plots are saved in `outputs/`: `training_loss.png`, `test_predictions.png`, `next_24_hours.png`.

## Project structure
- `train.py` - full pipeline: load, train, evaluate, plot, forecast
- `traffic_utils.py` - cleaning, features, windowing, baselines, forecasting helpers
- `requirements.txt` - dependencies

## How to run
```bash
pip install -r requirements.txt
python train.py                                  # downloads the dataset automatically
python train.py --data Metro_Interstate_Traffic_Volume.csv.gz   # or use a local copy
python train.py --synthetic --epochs 5           # quick check with generated data
```
Useful options: `--window 24`, `--epochs 30`, `--batch-size 64`, `--days N` (use only the last N days).

## Limitations
- Future weather is unknown, so the 24-hour forecast holds the last observed weather constant.
- One road segment only; the model has not been tested on other roads or cities.
- Sudden events (accidents, road closures) are not in the data.

## My contribution
Final-year group project; my part:
- Data preprocessing and feature engineering: cleaned the raw sensor data, built hourly series, weather/holiday features and cyclical time encodings
- Built and trained the LSTM forecasting model (24-hour input windows, early stopping)
- Evaluated it on a time-ordered held-out test set with MAE / MSE / RMSE and compared it with simple baselines
- Produced the plots and the 24-hour traffic forecast
