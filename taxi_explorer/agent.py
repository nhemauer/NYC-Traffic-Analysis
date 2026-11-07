"""Ask questions in plain English about the NYC taxi data in BigQuery.

The agent can also estimate a trip's fare with the random forest in
fare_model.joblib. Build that file first with `python model/train_fare_rf.py`.

Usage:
    python taxi_explorer/agent.py                       # interactive chat
    python taxi_explorer/agent.py "Busiest year?"       # ask one question and exit
    adk web                                            # browser UI (run from the project root)

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements-agent.txt
"""

import asyncio
import functools
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import google.auth
import joblib
import pandas as pd
from google.adk.agents import Agent
from google.adk.integrations.bigquery import (
    BigQueryCredentialsConfig,
    BigQueryToolset,
)
from google.adk.integrations.bigquery.config import BigQueryToolConfig, WriteMode
from google.adk.runners import InMemoryRunner
from google.genai import types

# A relative import works when ADK loads this as a package; running the file
# directly needs the plain import instead.
try:
    from .features import build_features
except ImportError:
    from features import build_features

PROJECT_ID = "traffic-513918"
DATASET_ID = "taxi"
TABLE_ID = "yellow_temporal_sample"
BIGQUERY_LOCATION = "US"
MODEL = "gemini-3.5-flash"
MAX_BYTES_PER_QUERY = 1_000_000_000  # queries that would scan more than 1 GB fail
FARE_MODEL_PATH = Path(__file__).resolve().parent / "fare_model.joblib"

# Call Gemini through Vertex AI, using the same Google login as BigQuery.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
# Assigned, not defaulted: `adk deploy` sets this to the deploy region, but the
# model is called through the global endpoint.
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

INSTRUCTION = f"""
You answer questions about NYC yellow taxi trips by querying BigQuery.

Use the table `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`. Check its columns with
get_table_info before writing SQL, then answer from the results of execute_sql.

The table is a sample holding about 1 in every 1,000 trips from the public TLC
data. Averages and percentages can be reported as they are. Counts and totals
describe the sample, so say so, and multiply by 1,000 if the user asks for an
estimate of the real figure.

Say plainly when the data can't answer a question. Never guess at a number.

To estimate what a particular trip would cost, use predict_fare rather than
SQL. It needs the distance in miles, the pickup date and time, the trip
duration in minutes, and the number of passengers. Ask for any of these the
user hasn't given, except passengers, which you can assume is 1 (say so). The
estimate is the metered fare only, without tips, tolls or surcharges, so say
that, and pass on any warnings the tool returns.
"""


@functools.cache
def _load_fare_model() -> dict:
    """Load the model on first use, so the agent starts quickly."""
    return joblib.load(FARE_MODEL_PATH)


def predict_fare(
    trip_distance_miles: float,
    pickup_datetime: str,
    duration_minutes: float,
    passenger_count: int,
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

    return {
        "estimated_fare_usd": round(fare, 2),
        "typical_error_usd": round(bundle["test_scores"]["mae"], 2),
        "includes": "metered fare only, no tips, tolls or surcharges",
        "warnings": warnings,
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
    instruction=INSTRUCTION,
    tools=[bigquery_tools, predict_fare],
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
