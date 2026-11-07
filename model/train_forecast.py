"""Train models that forecast NYC yellow taxi trips and average fares by the hour.

Two gradient-boosted tree models learn from taxi.hourly_trips (every trip, not
a sample) from 2023 on:
  * trips picked up per hour
  * average metered fare per hour
Their inputs are calendar features: hour, day of week, month, day of year,
US federal holidays, and a trend term. Daily, weekly and monthly forecasts
are the hourly forecasts added up.

Testing never lets a model see the future:
  * The last 3 months are held out as the test set.
  * Settings are tuned with 3-fold time-ordered cross-validation on the
    months before that: each fold trains on everything up to a 3-month block,
    then forecasts that block.
  * The tuned models are compared on the test set against a "same week last
    year" baseline.
How wrong the models were on those held-out blocks sets the forecast ranges
for each granularity. The saved models are then refit on every month, so
forecasts start from the latest data.

The models are saved into taxi_explorer/ so `adk deploy` packages them with
the agent.

Usage:
    python model/train_forecast.py

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements-model.txt
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from google.cloud import bigquery
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
# Load forecast_features.py on its own: importing the taxi_explorer package
# would also load the agent.
sys.path.insert(0, str(ROOT / "taxi_explorer"))
from forecast_features import (  # noqa: E402
    FEATURES,
    GRANULARITIES,
    TRAIN_START,
    aggregate,
    build_calendar_features,
)

PROJECT_ID = "traffic-513918"
HOURLY_REF = f"{PROJECT_ID}.taxi.hourly_trips"
MODEL_PATH = ROOT / "taxi_explorer" / "forecast_model.joblib"
SEED = 1337  # used for every random step

TEST_MONTHS = 3
CV_FOLDS = 3
CV_BLOCK_MONTHS = 3
# 2 x 2 = 4 settings, each fit on 3 folds, for each of the 2 models.
PARAM_GRID = [
    {"learning_rate": rate, "max_leaf_nodes": leaves}
    for rate in (0.05, 0.1)
    for leaves in (31, 63)
]
# The trend may only shift the overall level, never combine with the calendar.
# Otherwise trees learn rules like "latest period x month", and the latest
# period only covers some months, so forecasts for the others go wrong (a
# January after the data ended came out a third too low).
TREND = FEATURES.index("years_since_start")
BASE_PARAMS = dict(
    max_iter=300,
    early_stopping=False,
    random_state=SEED,
    interaction_cst=[[i for i in range(len(FEATURES)) if i != TREND], [TREND]],
)
# Forecast ranges cover the middle 80% of how far off the held-out forecasts were.
RANGE_QUANTILES = (0.1, 0.9)
# Same weekday one year earlier, for the baseline.
SEASONAL_LAG = pd.Timedelta(days=364)


def load_hours() -> pd.DataFrame:
    """Every hour from TRAIN_START to the latest loaded month; empty hours get zeros."""
    query = f"""
        SELECT pickup_hour, trips, fare_trips, fare_sum
        FROM `{HOURLY_REF}`
        WHERE pickup_hour >= DATETIME(@start)
        ORDER BY pickup_hour
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start", "DATE", TRAIN_START.date())
        ]
    )
    rows = bigquery.Client(project=PROJECT_ID).query(query, job_config=config).to_dataframe()
    rows["pickup_hour"] = pd.to_datetime(rows["pickup_hour"])
    # Some sources repeat the hour that occurs twice when clocks go back.
    rows = rows.groupby("pickup_hour").sum()

    last_month = rows.index.max().to_period("M")
    end = (last_month + 1).to_timestamp() - pd.Timedelta(hours=1)
    hours = pd.date_range(TRAIN_START, end, freq="h")
    hourly = rows.reindex(hours, fill_value=0).astype(float)
    hourly["avg_fare"] = hourly["fare_sum"] / hourly["fare_trips"].replace(0, np.nan)
    print(
        f"{len(hourly):,} hours from {hours[0]:%Y-%m-%d} to {hours[-1]:%Y-%m-%d}, "
        f"{hourly['trips'].sum():,.0f} trips"
    )
    return hourly


def month_offset(hourly: pd.DataFrame, months_back: int) -> pd.Timestamp:
    """The first hour of the month `months_back` months before the data ends."""
    last_month = hourly.index.max().to_period("M")
    return (last_month - months_back + 1).to_timestamp()


def fit(hourly: pd.DataFrame, trips_params: dict, fare_params: dict) -> dict:
    X = build_calendar_features(hourly.index)
    trips_model = HistGradientBoostingRegressor(
        loss="poisson", **BASE_PARAMS, **trips_params
    ).fit(X, hourly["trips"])

    # Average fare is only defined for hours with fares; busier hours count more.
    has_fares = hourly["fare_trips"] > 0
    fare_model = HistGradientBoostingRegressor(**BASE_PARAMS, **fare_params).fit(
        X[has_fares],
        hourly.loc[has_fares, "avg_fare"],
        sample_weight=hourly.loc[has_fares, "fare_trips"],
    )
    return {"trips": trips_model, "fare": fare_model}


def forecast(models: dict, hours: pd.DatetimeIndex) -> pd.DataFrame:
    X = build_calendar_features(hours)
    predicted = pd.DataFrame(index=hours)
    predicted["trips"] = models["trips"].predict(X)
    predicted["avg_fare"] = models["fare"].predict(X)
    predicted["fare_total"] = predicted["trips"] * predicted["avg_fare"]
    return predicted


def compare(actual: pd.DataFrame, predicted: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Actual and forecast trips and average fare per period."""
    a = aggregate(actual, granularity, "fare_sum", "fare_trips")
    p = aggregate(predicted, granularity, "fare_total", "trips")
    return a.add_suffix("_actual").join(p.add_suffix("_forecast"))


def daily_scores(both: pd.DataFrame) -> dict:
    """Daily trip error as a percentage, and daily average fare error in dollars."""
    trips_pct = (both["trips_forecast"] - both["trips_actual"]).abs() / both["trips_actual"]
    fare_err = (both["avg_fare_forecast"] - both["avg_fare_actual"]).abs()
    return {"trips_mape": trips_pct.mean(), "fare_mae": fare_err.mean()}


def cv_blocks(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Time-ordered folds: train on everything before a block, test on the block."""
    blocks = []
    for fold in range(CV_FOLDS, 0, -1):
        start = month_offset(train, fold * CV_BLOCK_MONTHS)
        end = month_offset(train, (fold - 1) * CV_BLOCK_MONTHS) if fold > 1 else None
        test = train[train.index >= start] if end is None else train[
            (train.index >= start) & (train.index < end)
        ]
        blocks.append((train[train.index < start], test))
    return blocks


def baseline(hourly: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast each hour as the same hour on the same weekday a year earlier."""
    earlier = hourly.reindex(hours - SEASONAL_LAG)
    predicted = pd.DataFrame(index=hours)
    predicted["trips"] = earlier["trips"].to_numpy()
    predicted["avg_fare"] = earlier["avg_fare"].to_numpy()
    predicted["fare_total"] = (predicted["trips"] * predicted["avg_fare"]).fillna(0)
    return predicted


def main() -> None:
    hourly = load_hours()
    test_start = month_offset(hourly, TEST_MONTHS)
    train, test = hourly[hourly.index < test_start], hourly[hourly.index >= test_start]
    blocks = cv_blocks(train)
    print(
        f"Test set: {test_start:%Y-%m} to {hourly.index.max():%Y-%m}. "
        f"CV blocks: " + ", ".join(f"{b.index.min():%Y-%m}" for _, b in blocks)
    )

    print(f"\nTuning: {len(PARAM_GRID)} settings x {CV_FOLDS} folds, for both models")
    started = time.perf_counter()
    results = []
    for params in PARAM_GRID:
        fold_scores = []
        for fold_train, fold_test in blocks:
            models = fit(fold_train, params, params)
            both = compare(fold_test, forecast(models, fold_test.index), "day")
            fold_scores.append(daily_scores(both))
        results.append({**params, **pd.DataFrame(fold_scores).mean().to_dict()})
    results = pd.DataFrame(results)
    print(f"Search finished in {time.perf_counter() - started:.0f}s")

    print("\nCross-validation, daily error (lower is better):")
    print(f"  {'learn_rate':>10}{'leaves':>8}{'trips %':>10}{'fare $':>9}")
    for _, row in results.iterrows():
        print(
            f"  {row['learning_rate']:>10}{row['max_leaf_nodes']:>8.0f}"
            f"{row['trips_mape']:>10.1%}{row['fare_mae']:>9.2f}"
        )
    # Each model gets the settings that suit it best.
    best = {
        "trips": results.loc[results["trips_mape"].idxmin(), ["learning_rate", "max_leaf_nodes"]],
        "fare": results.loc[results["fare_mae"].idxmin(), ["learning_rate", "max_leaf_nodes"]],
    }
    best = {
        name: {"learning_rate": float(p["learning_rate"]), "max_leaf_nodes": int(p["max_leaf_nodes"])}
        for name, p in best.items()
    }
    print(f"Best for trips: {best['trips']}")
    print(f"Best for fare:  {best['fare']}")

    # Held-out forecasts from the CV blocks and the test set set the ranges.
    held_out = []
    for fold_train, fold_test in [*blocks, (train, test)]:
        models = fit(fold_train, best["trips"], best["fare"])
        held_out.append((fold_test, forecast(models, fold_test.index)))

    test_actual, test_forecast = held_out[-1]
    model_scores = daily_scores(compare(test_actual, test_forecast, "day"))
    base_scores = daily_scores(compare(test, baseline(hourly, test.index), "day"))
    print(f"\nTest set ({test_start:%Y-%m} on), daily error:")
    print(f"  {'':<28}{'trips %':>10}{'fare $':>9}")
    for name, s in [("Same week last year", base_scores), ("Gradient boosting", model_scores)]:
        print(f"  {name:<28}{s['trips_mape']:>10.1%}{s['fare_mae']:>9.2f}")

    ranges = {}
    print(f"\nForecast ranges (middle {RANGE_QUANTILES[1] - RANGE_QUANTILES[0]:.0%} "
          "of held-out results, as actual / forecast):")
    print(f"  {'':<8}{'periods':>8}{'trips':>18}{'avg fare':>18}")
    for granularity in GRANULARITIES:
        both = pd.concat(
            compare(actual, predicted, granularity) for actual, predicted in held_out
        ).dropna()
        trips_ratio = both["trips_actual"] / both["trips_forecast"]
        fare_ratio = both["avg_fare_actual"] / both["avg_fare_forecast"]
        ranges[granularity] = {
            "trips": [float(trips_ratio.quantile(q)) for q in RANGE_QUANTILES],
            "avg_fare": [float(fare_ratio.quantile(q)) for q in RANGE_QUANTILES],
            "periods": len(both),
        }
        t, f = ranges[granularity]["trips"], ranges[granularity]["avg_fare"]
        print(
            f"  {granularity:<8}{len(both):>8}"
            f"{f'{t[0]:.2f} to {t[1]:.2f}':>18}{f'{f[0]:.2f} to {f[1]:.2f}':>18}"
        )

    print("\nRefitting on every month for the saved models...")
    models = fit(hourly, best["trips"], best["fare"])
    bundle = {
        "trips_model": models["trips"],
        "fare_model": models["fare"],
        "features": FEATURES,
        "train_start": TRAIN_START.isoformat(),
        "data_end": hourly.index.max().isoformat(),
        "ranges": ranges,
        "test_scores": {"model": model_scores, "baseline": base_scores},
        "params": best,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(
        f"Saved models to {MODEL_PATH.relative_to(ROOT)} "
        f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB), data through "
        f"{hourly.index.max():%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
