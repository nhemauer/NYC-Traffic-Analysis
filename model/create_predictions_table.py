"""Create the BigQuery tables where the agent saves its fare estimates and forecasts.

Creates, if they don't exist yet:
  * taxi.fare_predictions: one row per predict_fare estimate
  * taxi.forecast_requests: one row per forecast_trips forecast
and lets the agent's service account add rows to those two tables. The agent
can still only read the rest of the dataset. Safe to run again: existing rows
are left alone.

Both tables record which model answered (model_trained_at), so estimates can
be compared across retrains, and forecasts can be checked against what
actually happened once the data arrives.

Usage:
    python model/create_predictions_table.py

Auth: run `gcloud auth application-default login` once beforehand. You need
permission to create tables and change table access in the project.
Requires: pip install -r requirements.txt
"""

from google.cloud import bigquery

PROJECT_ID = "traffic-513918"
DATASET_REF = f"{PROJECT_ID}.taxi"
LOCATION = "US"
AGENT_SERVICE_ACCOUNT = "taxi-agent-runner@traffic-513918.iam.gserviceaccount.com"
WRITER_ROLE = "roles/bigquery.dataEditor"

Field = bigquery.SchemaField

# Who asked, shared by both tables.
REQUEST_FIELDS = [
    Field("environment", "STRING", description="Cloud Run service name, or 'local'"),
    Field("session_id", "STRING", description="Agent conversation"),
    Field("user_id", "STRING"),
]

FARE_PREDICTIONS_SCHEMA = [
    Field("prediction_id", "STRING", mode="REQUIRED"),
    Field("predicted_at", "TIMESTAMP", mode="REQUIRED"),
    *REQUEST_FIELDS,
    Field("trip_distance", "FLOAT", description="Miles"),
    Field("pickup_datetime", "DATETIME", description="New York local time"),
    Field("duration_min", "FLOAT"),
    Field("passenger_count", "INTEGER"),
    Field("estimated_fare", "FLOAT", description="Metered fare in dollars"),
    Field("warnings", "STRING", mode="REPEATED"),
    Field("model_trained_at", "TIMESTAMP", description="Which model made the estimate"),
]

FORECAST_PERIOD_FIELDS = [
    Field(
        "period_start",
        "DATETIME",
        description="First hour of the hour, day, week or month (New York local). "
        "Empty for hour_of_day forecasts",
    ),
    Field(
        "hour_of_day",
        "INTEGER",
        description="0-23, only for hour_of_day forecasts, where trips is the "
        "average per hour across the forecast's days",
    ),
    Field("trips", "FLOAT"),
    Field("trips_low", "FLOAT"),
    Field("trips_high", "FLOAT"),
    Field("avg_fare", "FLOAT", description="Metered fare in dollars"),
    Field("avg_fare_low", "FLOAT"),
    Field("avg_fare_high", "FLOAT"),
    Field(
        "partly_covered",
        "BOOLEAN",
        description="The forecast's dates or weekday filter cover only part of this period",
    ),
]

FORECAST_REQUESTS_SCHEMA = [
    Field("forecast_id", "STRING", mode="REQUIRED"),
    Field("requested_at", "TIMESTAMP", mode="REQUIRED"),
    *REQUEST_FIELDS,
    Field("requested_start", "STRING", description="Start date as the agent asked"),
    Field("requested_end", "STRING", description="End date as the agent asked"),
    Field(
        "forecast_from",
        "DATE",
        description="First day forecast. Later than requested_start if that had actual data",
    ),
    Field("forecast_to", "DATE"),
    Field("granularity", "STRING", description="hour, day, week, month or hour_of_day"),
    Field("day_of_week", "STRING", description="Weekday filter, if any"),
    Field("periods", "RECORD", mode="REPEATED", fields=FORECAST_PERIOD_FIELDS),
    Field("total_trips", "FLOAT", description="All trips from forecast_from to forecast_to"),
    Field("total_trips_low", "FLOAT"),
    Field("total_trips_high", "FLOAT"),
    Field("total_avg_fare", "FLOAT"),
    Field("total_avg_fare_low", "FLOAT"),
    Field("total_avg_fare_high", "FLOAT"),
    Field("warnings", "STRING", mode="REPEATED"),
    Field("model_trained_at", "TIMESTAMP", description="Which models made the forecast"),
    Field("model_data_end", "DATETIME", description="Last hour of data the models learned from"),
]

TABLES = [
    (
        "fare_predictions",
        FARE_PREDICTIONS_SCHEMA,
        "predicted_at",
        "Fare estimates made by the taxi_explorer agent's predict_fare tool.",
    ),
    (
        "forecast_requests",
        FORECAST_REQUESTS_SCHEMA,
        "requested_at",
        "Trip and fare forecasts made by the taxi_explorer agent's forecast_trips tool.",
    ),
]


def main() -> None:
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    member = f"serviceAccount:{AGENT_SERVICE_ACCOUNT}"

    for name, schema, partition_field, description in TABLES:
        table = bigquery.Table(f"{DATASET_REF}.{name}", schema=schema)
        table.description = description
        table.time_partitioning = bigquery.TimePartitioning(field=partition_field)
        table = client.create_table(table, exists_ok=True)
        print(f"Table ready: {table.full_table_id}")

        policy = client.get_iam_policy(table)
        if any(b["role"] == WRITER_ROLE and member in b["members"] for b in policy.bindings):
            print(f"  {AGENT_SERVICE_ACCOUNT} can already write to it")
            continue
        policy.bindings.append({"role": WRITER_ROLE, "members": {member}})
        client.set_iam_policy(table, policy)
        print(f"  Gave {AGENT_SERVICE_ACCOUNT} write access to this table only")


if __name__ == "__main__":
    main()
