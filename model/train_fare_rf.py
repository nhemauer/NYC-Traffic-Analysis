"""Train a random forest that predicts the metered fare of a NYC yellow taxi trip.

Features: year, trip distance, passenger count, pickup hour / day of week /
month, and trip duration (dropoff minus pickup). Target: fare_amount, the
metered fare before tips, tolls and surcharges.

The trips come from the `taxi.yellow_trips_sample` view in BigQuery. The first
run saves them to data/ so later runs don't query BigQuery again; use
--refresh after loading new TLC months.

Tree depth and leaf size are tuned with a small grid search, scoring each
setting by 3-fold cross-validation on the training trips. The held-out test
trips are only used once, to score the final model.

The model is saved into taxi_explorer/ so `adk deploy` packages it with the
agent. The grid only tries tree sizes that fit in the Cloud Run service's
512 MiB.

Usage:
    python model/train_fare_rf.py                    # tune and train on all trips
    python model/train_fare_rf.py --sample 200000    # quicker run on a subset
    python model/train_fare_rf.py --refresh          # re-download the trips first

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements-model.txt
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pickle

import joblib
import numpy as np
import pandas as pd
import sklearn
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
# Load features.py on its own: importing the taxi_explorer package would also
# load the agent.
sys.path.insert(0, str(ROOT / "taxi_explorer"))
from features import FEATURES, build_features  # noqa: E402

PROJECT_ID = "traffic-513918"
# View joining the 2011 - Nov 2022 sample with the TLC months loaded since.
TABLE_REF = f"{PROJECT_ID}.taxi.yellow_trips_sample"
CACHE_PATH = ROOT / "data" / "yellow_trips_sample.parquet"
MODEL_PATH = ROOT / "taxi_explorer" / "fare_model.joblib"
TARGET = "fare_amount"
SEED = 1337  # used for every random step: sampling, splits, folds and trees

# Trips outside these ranges are almost always recording errors.
MAX_FARE = 500
MAX_DISTANCE_MILES = 100
MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 180
MAX_PASSENGERS = 6

# Settings shared by every forest in the search.
FOREST_PARAMS = dict(
    n_estimators=100,
    max_samples=0.3,  # each tree sees 30% of the training rows
    n_jobs=-1,
    random_state=SEED,
)
# Settings the search tries: 2 x 3 = 6 combinations, each fit 3 times.
# Smaller leaves and deeper trees make bigger models, so the grid stops at
# sizes that still load in the agent's 512 MiB Cloud Run service.
PARAM_GRID = {
    "max_depth": [15, 25],
    "min_samples_leaf": [30, 50, 100],
}
CV_FOLDS = 3
# The loaded model must leave room for the agent itself in 512 MiB.
MODEL_MEMORY_WARNING_MB = 250


def load_trips(refresh: bool) -> pd.DataFrame:
    """Read the trips from the local cache, or from BigQuery on the first run."""
    if CACHE_PATH.exists() and not refresh:
        print(f"Reading cached trips from {CACHE_PATH.relative_to(ROOT)}")
        return pd.read_parquet(CACHE_PATH)

    print(f"Downloading trips from {TABLE_REF} (scans about 100 MB)")
    query = f"""
        SELECT pickup_datetime, dropoff_datetime, passenger_count,
               trip_distance, {TARGET}
        FROM `{TABLE_REF}`
    """
    trips = bigquery.Client(project=PROJECT_ID).query(query).to_dataframe()
    CACHE_PATH.parent.mkdir(exist_ok=True)
    trips.to_parquet(CACHE_PATH, index=False)
    return trips


def clean(trips: pd.DataFrame) -> pd.DataFrame:
    """Drop trips with missing or impossible values, reporting what each rule removes."""
    features = build_features(trips)
    rules = {
        "missing values": features.notna().all(axis=1) & trips[TARGET].notna(),
        f"fare not in ($0, ${MAX_FARE}]": trips[TARGET].between(
            0, MAX_FARE, inclusive="right"
        ),
        f"distance not in (0, {MAX_DISTANCE_MILES}] miles": features[
            "trip_distance"
        ].between(0, MAX_DISTANCE_MILES, inclusive="right"),
        f"duration not in [{MIN_DURATION_MIN}, {MAX_DURATION_MIN}] minutes": features[
            "duration_min"
        ].between(MIN_DURATION_MIN, MAX_DURATION_MIN),
        f"passengers not in [1, {MAX_PASSENGERS}]": features["passenger_count"].between(
            1, MAX_PASSENGERS
        ),
    }

    print(f"\n{len(trips):,} trips before cleaning")
    keep = pd.Series(True, index=trips.index)
    for name, passes in rules.items():
        removed = (keep & ~passes).sum()
        keep &= passes
        print(f"  - {removed:>9,} removed: {name}")
    print(f"{keep.sum():,} trips kept ({keep.mean():.1%})")

    cleaned = features[keep].copy()
    cleaned[TARGET] = trips.loc[keep, TARGET].astype(float)
    return cleaned


def scores(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "r2": r2_score(actual, predicted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sample", type=int, help="train on this many randomly chosen trips"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="download the trips from BigQuery again"
    )
    args = parser.parse_args()

    trips = clean(load_trips(args.refresh))
    if args.sample and args.sample < len(trips):
        trips = trips.sample(args.sample, random_state=SEED)
        print(f"Using a random {args.sample:,} of them")

    train, test = train_test_split(trips, test_size=0.2, random_state=SEED)
    print(f"\nTraining on {len(train):,} trips, testing on {len(test):,}")

    # Baseline: fare from distance alone. The forest should beat this clearly.
    baseline = LinearRegression().fit(train[["trip_distance"]], train[TARGET])
    baseline_scores = scores(test[TARGET], baseline.predict(test[["trip_distance"]]))

    combos = len(PARAM_GRID["max_depth"]) * len(PARAM_GRID["min_samples_leaf"])
    print(
        f"\nTuning: {combos} settings x {CV_FOLDS} folds = {combos * CV_FOLDS} fits, "
        "then one final fit on all training trips"
    )
    search = GridSearchCV(
        RandomForestRegressor(**FOREST_PARAMS),
        PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=1,  # each forest already uses every core
        refit=True,
    )
    started = time.perf_counter()
    search.fit(train[FEATURES], train[TARGET])
    print(f"Search finished in {time.perf_counter() - started:.0f}s")

    print(f"\n{CV_FOLDS}-fold cross-validation MAE in dollars (lower is better):")
    print(f"  {'max_depth':>10}{'min_leaf':>10}{'mean MAE':>10}{'± std':>8}")
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    for _, row in results.iterrows():
        marker = "  <- best" if row["rank_test_score"] == 1 else ""
        print(
            f"  {row['param_max_depth']:>10}{row['param_min_samples_leaf']:>10}"
            f"{-row['mean_test_score']:>10.3f}{row['std_test_score']:>8.3f}{marker}"
        )

    forest = search.best_estimator_
    forest_scores = scores(test[TARGET], forest.predict(test[FEATURES]))

    print(
        f"\nBest: max_depth={search.best_params_['max_depth']}, "
        f"min_samples_leaf={search.best_params_['min_samples_leaf']}"
    )
    print("\nTest set results (MAE and RMSE in dollars):")
    print(f"  {'':<28}{'MAE':>8}{'RMSE':>8}{'R²':>8}")
    for name, s in [
        ("Baseline (distance only)", baseline_scores),
        ("Random forest (tuned)", forest_scores),
    ]:
        print(f"  {name:<28}{s['mae']:>8.2f}{s['rmse']:>8.2f}{s['r2']:>8.3f}")

    print("\nFeature importance:")
    importances = pd.Series(forest.feature_importances_, index=FEATURES)
    for name, value in importances.sort_values(ascending=False).items():
        print(f"  {name:<18}{value:>6.1%}")

    bundle = {
        "model": forest,
        "features": FEATURES,
        "target": TARGET,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": len(train),
        # The forest can't extrapolate: later years are priced like the last one.
        "year_range": [int(train["year"].min()), int(train["year"].max())],
        "test_scores": forest_scores,
        "cv_mae": -search.best_score_,
        "params": forest.get_params(),
        "seed": SEED,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    disk_mb = MODEL_PATH.stat().st_size / 1e6
    # The uncompressed size is roughly what the model takes up once loaded.
    memory_mb = len(pickle.dumps(forest)) / 1e6
    print(
        f"\nSaved model to {MODEL_PATH.relative_to(ROOT)} "
        f"({disk_mb:.0f} MB on disk, about {memory_mb:.0f} MB once loaded)"
    )
    print(f"Load it with scikit-learn {sklearn.__version__}; other versions may fail.")
    if memory_mb > MODEL_MEMORY_WARNING_MB:
        print(
            f"Warning: over {MODEL_MEMORY_WARNING_MB} MB once loaded, which may not "
            "fit in the agent's 512 MiB. Raise min_samples_leaf or lower max_samples."
        )


if __name__ == "__main__":
    main()
