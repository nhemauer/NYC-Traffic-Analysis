"""Ask questions in plain English about the NYC taxi data in BigQuery.

The agent can also estimate a trip's fare with the random forest in
fare_model.joblib. Build that file first with `python model/train_fare_rf.py`.
Each estimate is saved to the `taxi.fare_predictions` table, which
`python model/create_predictions_table.py` sets up.

It forecasts future trips and average fares with the models in
forecast_model.joblib, built by `python model/train_forecast.py`. Each
forecast is saved to the `taxi.forecast_requests` table, which the same
create_predictions_table.py script sets up.

Usage:
    python taxi_explorer/agent.py                       # interactive chat
    python taxi_explorer/agent.py "Busiest year?"       # ask one question and exit
    adk web                                            # browser UI (run from the project root)

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements-agent.txt
"""

import asyncio
import functools
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import google.auth
import joblib
import pandas as pd
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext
from google.adk.integrations.bigquery import (
    BigQueryCredentialsConfig,
    BigQueryToolset,
)
from google.adk.integrations.bigquery.config import BigQueryToolConfig, WriteMode
from google.adk.runners import InMemoryRunner
from google.cloud import bigquery
from google.genai import types

# A relative import works when ADK loads this as a package; running the file
# directly needs the plain import instead.
try:
    from .features import build_features
    from .forecast_features import aggregate, build_calendar_features, period_start
except ImportError:
    from features import build_features
    from forecast_features import aggregate, build_calendar_features, period_start

PROJECT_ID = "traffic-513918"
DATASET_ID = "taxi"
TABLE_ID = "yellow_trips_sample"  # view: 2011 to the latest TLC month
HOURLY_TABLE_ID = "hourly_trips"
BIGQUERY_LOCATION = "US"
MODEL = "gemini-3.5-flash"
# Queries that would scan more than 250 MB fail. All the taxi tables together
# are about 115 MB (Nov 2026), so this only stops queries against other, much
# larger public datasets, which would be billed to this project.
MAX_BYTES_PER_QUERY = 250_000_000
FARE_MODEL_PATH = Path(__file__).resolve().parent / "fare_model.joblib"
FORECAST_MODEL_PATH = Path(__file__).resolve().parent / "forecast_model.joblib"
FORECAST_GRANULARITIES = ("hour", "day", "week", "month", "hour_of_day")
MAX_FORECAST_DAYS = 730  # how far past the latest data forecasts may reach
LONG_FORECAST_DAYS = 365  # beyond this, warn that the forecast is a rough guide
MAX_HOURLY_DAYS = 7  # keeps hour-by-hour answers to a readable size
PREDICTIONS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.fare_predictions"
FORECASTS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.forecast_requests"
# Cloud Run sets K_SERVICE, which keeps local test rows apart from real ones.
ENVIRONMENT = os.environ.get("K_SERVICE", "local")
NEW_YORK = ZoneInfo("America/New_York")

log = logging.getLogger(__name__)

# Call Gemini through Vertex AI, using the same Google login as BigQuery.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
# Assigned, not defaulted: `adk deploy` sets this to the deploy region, but the
# model is called through the global endpoint.
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

INSTRUCTION = f"""
You answer questions about NYC yellow taxi trips by querying BigQuery.

There are two tables. Check a table's columns with get_table_info before
writing SQL, then answer from the results of execute_sql.

`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` holds individual trips: a sample of
about 1 in every 1,000 yellow taxi trips from 2011 to the latest month the TLC
has published. Use it for questions about trip details, such as distances,
tips or payment types. Averages and percentages can be reported as they are.
Counts and totals describe the sample, so say so, and multiply by 1,000 if the
user asks for an estimate of the real figure.

`{PROJECT_ID}.{DATASET_ID}.{HOURLY_TABLE_ID}` has one row per pickup hour,
counted from every trip, not a sample. Prefer it for exact numbers of trips
per hour, day, month or year, and for average fares over time. Average fare
over any period is SUM(fare_sum) / SUM(fare_trips), never the average of
hourly averages.

Pickup times in both tables are New York local time.

Say plainly when the data can't answer a question. Never guess at a number.

To estimate what a particular trip would cost, use predict_fare rather than
SQL. It needs the distance in miles, the pickup date and time, the trip
duration in minutes, and the number of passengers. Ask for any of these the
user hasn't given, except passengers, which you can assume is 1 (say so). The
estimate is the metered fare only, without tips, tolls or surcharges, so say
that, and pass on any warnings the tool returns.

Every estimate predict_fare makes is saved in `{PREDICTIONS_TABLE}`, one row
per estimate, with its inputs, the estimated fare and when it was made. Query
that table if the user asks about earlier estimates. Rows whose environment is
'local' are tests run from a laptop.

For questions about the future, such as how many trips there will be next
month, what fares will average, or how busy certain hours will be, use
forecast_trips. Forecasts only start after the latest data; for dates before
that, answer from {HOURLY_TABLE_ID} instead. Choose the granularity that fits
the question: "hour_of_day" for busiest times of day, "hour" for a few
specific days, otherwise "day", "week" or "month". Give the range along with
the forecast, mention when a period is only partly covered, and pass on any
warnings. Trip counts in forecasts are real totals, not sample counts.

Every forecast is saved in `{FORECASTS_TABLE}`, one row per forecast, with
the request, each period's forecast and range in the repeated `periods`
field (use UNNEST to read them), the totals, and which model made it. Query it
if the user asks about earlier forecasts, or to compare a past forecast with
what actually happened in {HOURLY_TABLE_ID}. For hour_of_day forecasts,
periods have hour_of_day set and trips is the average per hour.
"""


def instruction_with_today(context: ReadonlyContext) -> str:
    """The instructions plus today's date, worked out fresh for every request.

    Without the date, the agent reads "next month" as the month after the
    latest data rather than after today.
    """
    today = datetime.now(NEW_YORK).date()
    return INSTRUCTION + f"""
Today is {today:%A, %B} {today.day}, {today:%Y} in New York. Work out
relative dates such as "next week", "next month" or "this winter" from today,
and say which dates you used. The TLC publishes trip data about two months
late, so recent weeks may have no actual data yet. If the user asks about a
period that has already happened but has no data, say so, and offer a
forecast for it instead.
"""


@functools.cache
def _load_fare_model() -> dict:
    """Load the model on first use, so the agent starts quickly."""
    return joblib.load(FARE_MODEL_PATH)


@functools.cache
def _load_forecast_models() -> dict:
    return joblib.load(FORECAST_MODEL_PATH)


def _with_range(value: float, ratios: list[float], digits: int) -> tuple:
    """A forecast value and its likely low and high, from held-out error ratios."""
    return tuple(round(v, digits) for v in (value, value * ratios[0], value * ratios[1]))


def _forecast_period_row(period: dict) -> dict:
    """One period of a forecast, flattened for the forecast_requests table."""
    trips = period.get("trips") or period["avg_trips"]
    fare = period["avg_fare"]
    row = {
        "trips": trips[0],
        "trips_low": trips[1],
        "trips_high": trips[2],
        "avg_fare": fare[0],
        "avg_fare_low": fare[1],
        "avg_fare_high": fare[2],
        "partly_covered": period.get("partly_covered", False),
    }
    if "hour_of_day" in period:
        row["hour_of_day"] = period["hour_of_day"]
    else:
        row["period_start"] = pd.Timestamp(period["period_start"]).isoformat()
    return row


def forecast_trips(
    start_date: str,
    end_date: str,
    granularity: str,
    day_of_week: str,
    tool_context: ToolContext,
) -> dict:
    """Forecast NYC yellow taxi trips and the average metered fare for future dates.

    Each period gets a forecast and a likely range (80% of past forecasts
    fell inside ranges like it). Trip counts are real totals, not samples.

    Args:
        start_date: First day to forecast, as "YYYY-MM-DD".
        end_date: Last day to forecast, included, as "YYYY-MM-DD".
        granularity: "hour", "day", "week" (Monday to Sunday), "month", or
            "hour_of_day" for the average trips in each of the 24 hours of the
            day across the whole period.
        day_of_week: Only forecast this weekday, such as "Friday". Pass an
            empty string to include every day.

    Returns:
        A row per period with trips and average fare and their ranges, a
        total for the whole period, and any warnings.
    """
    if not FORECAST_MODEL_PATH.exists():
        return {"error": "The forecast models haven't been trained or deployed yet."}
    if granularity not in FORECAST_GRANULARITIES:
        return {"error": f"granularity must be one of {', '.join(FORECAST_GRANULARITIES)}."}
    try:
        start = pd.Timestamp(datetime.fromisoformat(start_date.strip()).date())
        end = pd.Timestamp(datetime.fromisoformat(end_date.strip()).date())
    except ValueError:
        return {"error": 'Dates must look like "2026-12-01".'}
    if end < start:
        return {"error": "end_date is before start_date."}

    bundle = _load_forecast_models()
    data_end = pd.Timestamp(bundle["data_end"])
    first_forecast_day = data_end.normalize() + pd.Timedelta(days=1)
    warnings = []
    if end < first_forecast_day:
        return {
            "error": f"Actual data exists up to {data_end:%Y-%m-%d}, so these dates "
            f"don't need a forecast. Query {HOURLY_TABLE_ID} instead."
        }
    if start < first_forecast_day:
        warnings.append(
            f"Actual data exists up to {data_end:%Y-%m-%d}, so the forecast starts "
            f"on {first_forecast_day:%Y-%m-%d}. Query {HOURLY_TABLE_ID} for earlier days."
        )
        start = first_forecast_day
    days_ahead = (end - data_end).days
    if days_ahead > MAX_FORECAST_DAYS:
        return {
            "error": f"Forecasts reach at most {MAX_FORECAST_DAYS} days past the latest "
            f"data ({data_end:%Y-%m-%d})."
        }
    if days_ahead > LONG_FORECAST_DAYS:
        warnings.append(
            "This reaches over a year past the latest data, so treat it as a rough "
            "guide: the models assume recent patterns continue."
        )

    hours = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h")
    if day_of_week.strip():
        names = [d.lower() for d in pd.date_range("2024-01-01", periods=7).day_name()]
        if day_of_week.strip().lower() not in names:
            return {"error": 'day_of_week must be a weekday name like "Friday", or "".'}
        hours = hours[hours.dayofweek == names.index(day_of_week.strip().lower())]
        if hours.empty:
            return {"error": f"There's no {day_of_week} between those dates."}
    if granularity == "hour" and hours.normalize().nunique() > MAX_HOURLY_DAYS:
        return {
            "error": f'"hour" covers at most {MAX_HOURLY_DAYS} days. Use "hour_of_day" '
            'or "day" for longer periods.'
        }

    X = build_calendar_features(hours)[bundle["features"]]
    hourly = pd.DataFrame(index=hours)
    hourly["trips"] = bundle["trips_model"].predict(X)
    hourly["fare_total"] = hourly["trips"] * bundle["fare_model"].predict(X)
    ranges = bundle["ranges"]

    if granularity == "hour_of_day":
        days = hours.normalize().nunique()
        by_hour = hourly.groupby(hours.hour).sum()
        r = ranges["day"]  # each hour's figure is an average over days
        periods = [
            {
                "hour_of_day": int(hour),
                "avg_trips": _with_range(row["trips"] / days, r["trips"], 0),
                "avg_fare": _with_range(row["fare_total"] / row["trips"], r["avg_fare"], 2),
            }
            for hour, row in by_hour.iterrows()
        ]
    else:
        totals = aggregate(hourly, granularity, "fare_total", "trips")
        hours_covered = pd.Series(1, index=hours).groupby(
            period_start(hours, granularity)
        ).size()
        r = ranges[granularity]
        time_format = "%Y-%m-%d %H:%M" if granularity == "hour" else "%Y-%m-%d"
        periods = []
        for start_of_period, row in totals.iterrows():
            full_hours = {
                "hour": 1,
                "day": 24,
                "week": 7 * 24,
                "month": start_of_period.days_in_month * 24,
            }[granularity]
            period = {
                "period_start": start_of_period.strftime(time_format),
                "trips": _with_range(row["trips"], r["trips"], 0),
                "avg_fare": _with_range(row["avg_fare"], r["avg_fare"], 2),
            }
            # Edges of the date range, or a day_of_week filter, leave gaps.
            if hours_covered[start_of_period] < full_hours:
                period["partly_covered"] = True
            periods.append(period)

    # The whole period's total uses the range for the nearest granularity.
    span_days = hours.normalize().nunique()
    total_range = ranges["month" if span_days >= 28 else "week" if span_days >= 7 else "day"]
    trips_total = hourly["trips"].sum()
    total_trips = _with_range(trips_total, total_range["trips"], 0)
    total_fare = _with_range(
        hourly["fare_total"].sum() / trips_total, total_range["avg_fare"], 2
    )

    saved = _save_row(
        FORECASTS_TABLE,
        {
            "forecast_id": str(uuid.uuid4()),
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "session_id": tool_context.session.id,
            "user_id": tool_context.user_id,
            "requested_start": start_date.strip(),
            "requested_end": end_date.strip(),
            "forecast_from": f"{hours[0]:%Y-%m-%d}",
            "forecast_to": f"{hours[-1]:%Y-%m-%d}",
            "granularity": granularity,
            "day_of_week": day_of_week.strip() or None,
            "periods": [_forecast_period_row(p) for p in periods],
            "total_trips": total_trips[0],
            "total_trips_low": total_trips[1],
            "total_trips_high": total_trips[2],
            "total_avg_fare": total_fare[0],
            "total_avg_fare_low": total_fare[1],
            "total_avg_fare_high": total_fare[2],
            "warnings": warnings,
            "model_trained_at": bundle["trained_at"],
            "model_data_end": data_end.isoformat(),
        },
        "forecast",
    )

    return {
        "forecast_from": f"{hours[0]:%Y-%m-%d}",
        "forecast_to": f"{hours[-1]:%Y-%m-%d}",
        "latest_actual_data": f"{data_end:%Y-%m-%d}",
        "values_are": "[forecast, low, high]",
        "periods": periods,
        "total": {"trips": total_trips, "avg_fare": total_fare},
        "warnings": warnings,
        "saved_to_history": saved,
    }


@functools.cache
def _bigquery_client() -> bigquery.Client:
    return bigquery.Client(
        project=PROJECT_ID, credentials=credentials, location=BIGQUERY_LOCATION
    )


def _save_row(table: str, row: dict, what: str) -> bool:
    """Add one row to a history table. Never raises: a failed save shouldn't
    stop the user getting their answer."""
    try:
        errors = _bigquery_client().insert_rows_json(table, [row])
    except Exception as exc:
        log.warning("Couldn't save %s to %s: %s", what, table, exc)
        return False
    if errors:
        log.error("BigQuery rejected %s for %s: %s", what, table, errors)
        return False
    return True


def predict_fare(
    trip_distance_miles: float,
    pickup_datetime: str,
    duration_minutes: float,
    passenger_count: int,
    tool_context: ToolContext,
) -> dict:
    """Estimate the metered fare of a NYC yellow taxi trip with a trained model.

    The fare excludes tips, tolls and surcharges.

    Args:
        trip_distance_miles: Trip distance in miles.
        pickup_datetime: Pickup date and time in New York local time, as
            "YYYY-MM-DD HH:MM", for example "2019-03-15 18:05".
        duration_minutes: Time from pickup to dropoff, in minutes.
        passenger_count: Number of passengers, from 1 to 6.

    Returns:
        The estimated fare in dollars, the model's typical error, and any
        warnings about inputs outside what the model was trained on.
    """
    if not FARE_MODEL_PATH.exists():
        return {"error": "The fare model hasn't been trained or deployed yet."}
    try:
        pickup = datetime.fromisoformat(pickup_datetime.strip())
    except ValueError:
        return {"error": 'pickup_datetime must look like "2019-03-15 18:05".'}
    if trip_distance_miles <= 0 or duration_minutes <= 0:
        return {"error": "Distance and duration must both be greater than zero."}

    bundle = _load_fare_model()
    trip = pd.DataFrame(
        {
            "pickup_datetime": [pickup],
            "dropoff_datetime": [pickup + timedelta(minutes=duration_minutes)],
            "trip_distance": [trip_distance_miles],
            "passenger_count": [passenger_count],
        }
    )
    fare = float(bundle["model"].predict(build_features(trip)[bundle["features"]])[0])

    # The model only knows trips like the ones it was cleaned down to.
    warnings = []
    first_year, last_year = bundle["year_range"]
    if not first_year <= pickup.year <= last_year:
        warnings.append(
            f"The model was trained on {first_year}-{last_year} trips, so this is "
            f"priced as if it were {min(max(pickup.year, first_year), last_year)}."
        )
    if trip_distance_miles > 100:
        warnings.append("Trips over 100 miles were not in the training data.")
    if not 1 <= duration_minutes <= 180:
        warnings.append("Trips under 1 minute or over 3 hours were not in the training data.")
    if not 1 <= passenger_count <= 6:
        warnings.append("Only trips with 1-6 passengers were in the training data.")

    estimated_fare = round(fare, 2)
    saved = _save_row(
        PREDICTIONS_TABLE,
        {
            "prediction_id": str(uuid.uuid4()),
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "session_id": tool_context.session.id,
            "user_id": tool_context.user_id,
            "trip_distance": trip_distance_miles,
            "pickup_datetime": pickup.isoformat(timespec="seconds"),
            "duration_min": duration_minutes,
            "passenger_count": passenger_count,
            "estimated_fare": estimated_fare,
            "warnings": warnings,
            "model_trained_at": bundle["trained_at"],
        },
        "fare prediction",
    )

    return {
        "estimated_fare_usd": estimated_fare,
        "typical_error_usd": round(bundle["test_scores"]["mae"], 2),
        "includes": "metered fare only, no tips, tolls or surcharges",
        "warnings": warnings,
        "saved_to_history": saved,
    }

credentials, _ = google.auth.default()

bigquery_tools = BigQueryToolset(
    credentials_config=BigQueryCredentialsConfig(credentials=credentials),
    bigquery_tool_config=BigQueryToolConfig(
        write_mode=WriteMode.BLOCKED,  # SELECT only
        maximum_bytes_billed=MAX_BYTES_PER_QUERY,
        compute_project_id=PROJECT_ID,
        default_project_id=PROJECT_ID,
        default_dataset_id=DATASET_ID,
        location=BIGQUERY_LOCATION,
    ),
    tool_filter=["get_table_info", "list_table_ids", "execute_sql"],
)

root_agent = Agent(
    name="taxi_data_explorer",
    model=MODEL,
    instruction=instruction_with_today,
    tools=[bigquery_tools, predict_fare, forecast_trips],
)


async def ask(runner: InMemoryRunner, session_id: str, question: str) -> None:
    """Send one question to the agent, printing any SQL it runs and its answer."""
    message = types.Content(role="user", parts=[types.Part(text=question)])
    async for event in runner.run_async(
        user_id="local", session_id=session_id, new_message=message
    ):
        for call in event.get_function_calls():
            if call.name == "execute_sql":
                print(f"\n[SQL]\n{call.args.get('query')}\n")
        if event.is_final_response() and event.content and event.content.parts:
            print("".join(part.text or "" for part in event.content.parts))


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent, app_name="taxi_data_explorer")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="local"
    )

    if len(sys.argv) > 1:
        await ask(runner, session.id, " ".join(sys.argv[1:]))
        return

    print("Ask about the taxi data. Press Enter on an empty line to quit.")
    while question := (await asyncio.to_thread(input, "\n> ")).strip():
        await ask(runner, session.id, question)


if __name__ == "__main__":
    asyncio.run(main())
