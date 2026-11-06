"""Build a ~0.1% sample of NYC yellow taxi trips in BigQuery.

Creates the `taxi` dataset (if needed) and (re)creates
`taxi.yellow_temporal_sample` from the public TLC yellow trip tables.

Usage:
    python create_yellow_sample.py            # run it
    python create_yellow_sample.py --dry-run  # only estimate bytes scanned

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install google-cloud-bigquery
"""

import argparse

from google.cloud import bigquery

PROJECT_ID = "traffic-513918"
DATASET_ID = "taxi"
TABLE_ID = "yellow_temporal_sample"
LOCATION = "US"
SAMPLE_RATE = 1000  # keep 1 in every 1,000 trips

SAMPLE_SQL = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` AS
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
FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_*` t
WHERE MOD(ABS(FARM_FINGERPRINT(CONCAT(
        CAST(pickup_datetime AS STRING), CAST(total_amount AS STRING)))), {SAMPLE_RATE}) = 0
  AND EXTRACT(YEAR FROM pickup_datetime) = CAST(_TABLE_SUFFIX AS INT64)
"""


def create_dataset(client: bigquery.Client) -> None:
    """Equivalent of CREATE SCHEMA IF NOT EXISTS ... OPTIONS (location = 'US')."""
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset {PROJECT_ID}.{DATASET_ID} is ready.")


def build_sample(client: bigquery.Client, dry_run: bool = False) -> None:
    job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=False)
    job = client.query(SAMPLE_SQL, job_config=job_config, location=LOCATION)

    if dry_run:
        gb = job.total_bytes_processed / 1e9
        print(f"Dry run: this query would scan about {gb:,.2f} GB.")
        return

    print(f"Started job {job.job_id}; waiting for it to finish...")
    job.result()  # raises if the query fails

    table = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
    print(
        f"Created {table.full_table_id} with {table.num_rows:,} rows "
        f"(scanned {job.total_bytes_processed / 1e9:,.2f} GB)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="estimate bytes scanned without creating anything",
    )
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    if not args.dry_run:
        create_dataset(client)
    build_sample(client, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
