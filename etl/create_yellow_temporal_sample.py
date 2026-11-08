"""Build a ~0.1% sample of NYC yellow taxi trips in BigQuery.

Creates the `taxi` dataset (if needed) and builds
`taxi.yellow_temporal_sample` from the public TLC yellow trip tables.

The table only needs rebuilding when one of these is true:
  * it doesn't exist yet
  * the query below has changed since it was built
  * Google has updated the public source tables since it was built
Checking this only reads table metadata, so it's free.

Usage:
    python etl/create_yellow_temporal_sample.py              # always rebuild
    python etl/create_yellow_temporal_sample.py --check      # report whether a rebuild is needed
    python etl/create_yellow_temporal_sample.py --if-needed  # rebuild only if needed (used by Cloud Build)
    python etl/create_yellow_temporal_sample.py --dry-run    # estimate bytes a rebuild would scan
    python etl/create_yellow_temporal_sample.py --stamp      # mark an existing table as built from this query

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements.txt
"""

import argparse
import hashlib
from datetime import datetime

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

PROJECT_ID = "traffic-513918"
DATASET_ID = "taxi"
TABLE_ID = "yellow_temporal_sample"
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
LOCATION = "US"
SAMPLE_RATE = 1000  # keep 1 in every 1,000 trips

SOURCE_DATASET = "bigquery-public-data.new_york_taxi_trips"
SOURCE_PREFIX = "tlc_yellow_trips_"

SAMPLE_SQL = f"""
CREATE OR REPLACE TABLE `{TABLE_REF}` AS
SELECT
  CAST(pickup_datetime AS TIMESTAMP)  AS pickup_datetime,
  CAST(dropoff_datetime AS TIMESTAMP) AS dropoff_datetime,
  CAST(passenger_count AS INT64)      AS passenger_count,
  CAST(trip_distance AS FLOAT64)      AS trip_distance,
  CAST(fare_amount AS FLOAT64)        AS fare_amount,
  CAST(tip_amount AS FLOAT64)         AS tip_amount,
  CAST(total_amount AS FLOAT64)       AS total_amount,
  CAST(payment_type AS STRING)        AS payment_type,
  _TABLE_SUFFIX                       AS source_year
FROM `{SOURCE_DATASET}.{SOURCE_PREFIX}*` t
WHERE MOD(ABS(FARM_FINGERPRINT(CONCAT(
        CAST(pickup_datetime AS STRING), CAST(total_amount AS STRING)))), {SAMPLE_RATE}) = 0
  AND EXTRACT(YEAR FROM pickup_datetime) = CAST(_TABLE_SUFFIX AS INT64)
"""

# The table is labelled with this fingerprint of the query, so a later run can
# tell whether the table was built from the current version of the query.
HASH_LABEL = "sql_hash"


def sql_fingerprint() -> str:
    """Short hash of the query, ignoring whitespace-only edits."""
    normalized = " ".join(SAMPLE_SQL.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def latest_source_update(client: bigquery.Client) -> datetime:
    """Most recent modification time across the public yellow-trip tables."""
    times = [
        client.get_table(t.reference).modified
        for t in client.list_tables(SOURCE_DATASET)
        if t.table_id.startswith(SOURCE_PREFIX)
    ]
    return max(times)


def needs_rebuild(client: bigquery.Client) -> tuple[bool, str]:
    """Return (rebuild needed?, reason)."""
    try:
        table = client.get_table(TABLE_REF)
    except NotFound:
        return True, "the table doesn't exist yet"

    built_from = (table.labels or {}).get(HASH_LABEL)
    if built_from is None:
        return True, (
            "the table has no record of which query built it "
            "(if it was built from this exact query, run with --stamp)"
        )
    if built_from != sql_fingerprint():
        return True, "the query has changed since the table was built"

    source_updated = latest_source_update(client)
    if source_updated > table.modified:
        return True, (
            f"the public source data was updated {source_updated:%Y-%m-%d %H:%M} UTC, "
            f"after the table was built ({table.modified:%Y-%m-%d %H:%M} UTC)"
        )

    return False, (
        f"the table is up to date (built {table.modified:%Y-%m-%d %H:%M} UTC, "
        f"source last updated {source_updated:%Y-%m-%d %H:%M} UTC)"
    )


def create_dataset(client: bigquery.Client) -> None:
    """Equivalent of CREATE SCHEMA IF NOT EXISTS ... OPTIONS (location = 'US').

    Looks first, because the Cloud Build account may only edit the dataset,
    not create datasets, and BigQuery then refuses even an exists_ok create.
    """
    try:
        client.get_dataset(f"{PROJECT_ID}.{DATASET_ID}")
    except NotFound:
        dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
        dataset.location = LOCATION
        client.create_dataset(dataset)
    print(f"Dataset {PROJECT_ID}.{DATASET_ID} is ready.")


def stamp(client: bigquery.Client) -> None:
    """Label the table with the current query's fingerprint."""
    table = client.get_table(TABLE_REF)
    table.labels = {**(table.labels or {}), HASH_LABEL: sql_fingerprint()}
    client.update_table(table, ["labels"])
    print(f"Labelled {TABLE_REF} with {HASH_LABEL}={sql_fingerprint()}.")


def build_sample(client: bigquery.Client, dry_run: bool = False) -> None:
    job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=False)
    job = client.query(SAMPLE_SQL, job_config=job_config, location=LOCATION)

    if dry_run:
        gb = job.total_bytes_processed / 1e9
        print(f"Dry run: this query would scan about {gb:,.2f} GB.")
        return

    print(f"Started job {job.job_id}; waiting for it to finish...")
    job.result()  # raises if the query fails
    stamp(client)

    table = client.get_table(TABLE_REF)
    print(
        f"Created {table.full_table_id} with {table.num_rows:,} rows "
        f"(scanned {job.total_bytes_processed / 1e9:,.2f} GB)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="report whether a rebuild is needed; change nothing")
    mode.add_argument("--if-needed", action="store_true",
                      help="rebuild only if the table is missing or out of date")
    mode.add_argument("--dry-run", action="store_true",
                      help="estimate bytes a rebuild would scan; change nothing")
    mode.add_argument("--stamp", action="store_true",
                      help="mark the existing table as built from the current query")
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)

    if args.dry_run:
        build_sample(client, dry_run=True)
        return
    if args.stamp:
        stamp(client)
        return
    if args.check or args.if_needed:
        rebuild, reason = needs_rebuild(client)
        print(f"{'Rebuild needed' if rebuild else 'No rebuild needed'}: {reason}.")
        if args.check or not rebuild:
            return

    create_dataset(client)
    build_sample(client)


if __name__ == "__main__":
    main()
