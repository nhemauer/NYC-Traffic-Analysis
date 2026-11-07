"""Calendar inputs and totals for the trip forecast models.

The training script and the agent both use these, so the models see hours
described the same way, and forecasts are added up the same way.

Hours are New York local clock times, like the pickup_hour column in
taxi.hourly_trips.
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# The models only learn from trips after the COVID collapse and the TLC fare
# increase of December 2022; earlier volumes and prices no longer apply.
TRAIN_START = pd.Timestamp("2023-01-01")

# Column order matters: the models expect exactly these, in this order.
FEATURES = [
    "hour",
    "dayofweek",  # Monday = 0
    "month",
    "dayofyear",
    "is_holiday",
    # Lets the models follow the trend in the training years. Trees can't
    # extrapolate, so future hours get the level of the latest data.
    "years_since_start",
]

GRANULARITIES = ("hour", "day", "week", "month")


def build_calendar_features(hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Describe each hour by its place in the day, week and year."""
    holidays = USFederalHolidayCalendar().holidays(
        hours.min().normalize(), hours.max().normalize()
    )
    features = pd.DataFrame(index=hours)
    features["hour"] = hours.hour
    features["dayofweek"] = hours.dayofweek
    features["month"] = hours.month
    features["dayofyear"] = hours.dayofyear
    features["is_holiday"] = hours.normalize().isin(holidays).astype(int)
    features["years_since_start"] = (hours - TRAIN_START) / pd.Timedelta(days=365.25)
    return features[FEATURES]


def period_start(hours: pd.DatetimeIndex, granularity: str) -> pd.DatetimeIndex:
    """The first hour of the day, week (from Monday) or month each hour is in."""
    if granularity == "hour":
        return hours
    if granularity == "day":
        return hours.normalize()
    if granularity == "week":
        return hours.normalize() - pd.to_timedelta(hours.dayofweek, unit="D")
    if granularity == "month":
        return hours.to_period("M").to_timestamp()
    raise ValueError(f"granularity must be one of {GRANULARITIES}")


def aggregate(
    hourly: pd.DataFrame, granularity: str, fare_sum: str, fare_weight: str
) -> pd.DataFrame:
    """Add hourly rows up to the given granularity.

    Trips are summed. The average fare is fare_sum / fare_weight over each
    period, so busy hours count for more than quiet ones. For actual data pass
    the fare_sum and fare_trips columns; for forecasts, predicted fare x
    predicted trips and predicted trips.
    """
    keys = period_start(hourly.index, granularity)
    # For forecasts the fare weight is the trips column itself, so drop repeats.
    columns = list(dict.fromkeys(["trips", fare_sum, fare_weight]))
    totals = hourly[columns].groupby(keys).sum()
    totals["avg_fare"] = totals[fare_sum] / totals[fare_weight]
    totals.index.name = "period_start"
    return totals[["trips", "avg_fare"]]
