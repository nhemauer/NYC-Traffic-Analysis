"""Turn raw taxi trip columns into the inputs the fare model expects.

The training script and the agent both use build_features, so a trip is
described to the model the same way in both places.

The TLC timestamps are New York local times. BigQuery labels them UTC, but they
were never converted, so the hour of day is used as stored.
"""

import pandas as pd

# Column order matters: the model expects exactly these columns, in this order.
FEATURES = [
    "year",
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "duration_min",
]


def _naive(timestamps: pd.Series) -> pd.Series:
    """Parse to datetimes and drop any time zone label without shifting the clock."""
    parsed = pd.to_datetime(timestamps)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def build_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Build the model inputs from trips.

    Needs the columns pickup_datetime, dropoff_datetime, trip_distance and
    passenger_count. The year comes from the pickup time, which always matches
    source_year in the sample table.
    """
    pickup = _naive(trips["pickup_datetime"])
    dropoff = _naive(trips["dropoff_datetime"])

    features = pd.DataFrame(index=trips.index)
    features["year"] = pickup.dt.year
    features["trip_distance"] = trips["trip_distance"].astype(float)
    features["passenger_count"] = trips["passenger_count"].astype(float)
    features["pickup_hour"] = pickup.dt.hour
    features["pickup_dayofweek"] = pickup.dt.dayofweek  # Monday = 0
    features["pickup_month"] = pickup.dt.month
    features["duration_min"] = (dropoff - pickup).dt.total_seconds() / 60
    return features[FEATURES]
