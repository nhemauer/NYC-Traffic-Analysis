"""Load the NYC TLC's monthly yellow taxi files into BigQuery.

The public BigQuery taxi tables stop at November 2022. The TLC still publishes
a file per month (about two months behind), so this script carries the data
forward from December 2022. For each month it:

  1. downloads the TLC file and loads it into a temporary BigQuery table
  2. adds 1 in every 1,000 trips to taxi.yellow_tlc_sample, using the same rule
     as taxi.yellow_temporal_sample
  3. adds hourly totals, counted from every trip, to taxi.hourly_trips
  4. records the month in taxi.tlc_months_loaded

Steps 2-4 run in one transaction, so a month is either fully loaded or not at
all. Loading a month again replaces it.

It also keeps:
  * taxi.yellow_trips_sample: a view joining yellow_temporal_sample
    (2011 - Nov 2022) and yellow_tlc_sample (Dec 2022 on). Query this one.
  * hourly totals for 2011 - Nov 2022 in taxi.hourly_trips, built from the full
    public tables when run with --public.

The two sample tables are kept apart so that rebuilding yellow_temporal_sample
with create_yellow_temporal_sample.py can't wipe the TLC months.

Usage:
    python etl/load_tlc_months.py                  # load every new month the TLC has published
    python etl/load_tlc_months.py --dry-run        # list the months that would be loaded
    python etl/load_tlc_months.py --month 2026-09  # load (or reload) one month
    python etl/load_tlc_months.py --public         # (re)build 2011 - Nov 2022 hourly totals
                                                   # (scans about 38 GB, roughly $0.23)

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements.txt
"""

import argparse
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from google.cloud import bigquery

# Shares the sampling rate and source with the script that builds the
# 2011 - Nov 2022 sample, so both halves of the sample match.
from create_yellow_temporal_sample import (
    LOCATION,
    PROJECT_ID,
    SAMPLE_RATE,
    SOURCE_DATASET,
    SOURCE_PREFIX,
)
from create_yellow_temporal_sample import TABLE_REF as PUBLIC_SAMPLE_REF

DATASET_REF = f"{PROJECT_ID}.taxi"
TLC_SAMPLE_REF = f"{DATASET_REF}.yellow_tlc_sample"
SAMPLE_VIEW_REF = f"{DATASET_REF}.yellow_trips_sample"
HOURLY_REF = f"{DATASET_REF}.hourly_trips"
LEDGER_REF = f"{DATASET_REF}.tlc_months_loaded"

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
# The public tables end on 30 Nov 2022, so the TLC files take over from here.
FIRST_TLC_MONTH = date(2022, 12, 1)
# Fares outside (0, MAX_FARE] are left out of the hourly fare totals, matching
# the fare model's cleaning. Every trip still counts towards the trip totals.
MAX_FARE = 500
USER_AGENT = "nyc-traffic-analysis-loader"

HOURLY_SCHEMA = [
    bigquery.SchemaField(
        "pickup_hour", "DATETIME", mode="REQUIRED", description="New York local time"
    ),
    bigquery.SchemaField("trips", "INTEGER", description="Every trip picked up this hour"),
    bigquery.SchemaField(
        "fare_trips", "INTEGER", description=f"Trips with a fare in ($0, ${MAX_FARE}]"
    ),
    bigquery.SchemaField(
        "fare_sum", "FLOAT", description="Sum of those fares. Average = fare_sum / fare_trips"
    ),
    bigquery.SchemaField("source", "STRING", description="'bq_public' or 'tlc'"),
    bigquery.SchemaField("month", "DATE", description="First day of the pickup month"),
]
LEDGER_SCHEMA = [
    bigquery.SchemaField("month", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("loaded_at", "TIMESTAMP"),
    bigquery.SchemaField("trips", "INTEGER"),
    bigquery.SchemaField("sampled_trips", "INTEGER"),
    bigquery.SchemaField("source_url", "STRING"),
]

# A trip counts in the month and hour its pickup clock time falls in. The TLC
# times are New York local; casting through TIMESTAMP keeps the clock time
# whether BigQuery loaded the column as DATETIME or TIMESTAMP.
HOURLY_SELECT = f"""
    SELECT
      DATETIME_TRUNC(pickup, HOUR) AS pickup_hour,
      COUNT(*) AS trips,
      COUNTIF(fare > 0 AND fare <= {MAX_FARE}) AS fare_trips,
      SUM(IF(fare > 0 AND fare <= {MAX_FARE}, fare, 0)) AS fare_sum,
      @source AS source,
      DATE_TRUNC(DATE(pickup), MONTH) AS month
    FROM trips
    GROUP BY pickup_hour, month
"""

PUBLIC_HOURLY_SQL = f"""
BEGIN TRANSACTION;
DELETE FROM `{HOURLY_REF}` WHERE source = @source;
INSERT INTO `{HOURLY_REF}` (pickup_hour, trips, fare_trips, fare_sum, source, month)
WITH trips AS (
  SELECT
    CAST(CAST(pickup_datetime AS TIMESTAMP) AS DATETIME) AS pickup,
    CAST(fare_amount AS FLOAT64) AS fare
  FROM `{SOURCE_DATASET}.{SOURCE_PREFIX}*`
  WHERE EXTRACT(YEAR FROM pickup_datetime) = CAST(_TABLE_SUFFIX AS INT64)
    AND CAST(CAST(pickup_datetime AS TIMESTAMP) AS DATETIME) < DATETIME(@first_tlc_month)
)
{HOURLY_SELECT};
COMMIT TRANSACTION;
"""

MONTH_SQL = f"""
BEGIN TRANSACTION;

DELETE FROM `{HOURLY_REF}` WHERE source = @source AND month = @month;
DELETE FROM `{TLC_SAMPLE_REF}` WHERE DATE_TRUNC(DATE(pickup_datetime), MONTH) = @month;
DELETE FROM `{LEDGER_REF}` WHERE month = @month;

INSERT INTO `{HOURLY_REF}` (pickup_hour, trips, fare_trips, fare_sum, source, month)
WITH trips AS (
  SELECT
    CAST(CAST(tpep_pickup_datetime AS TIMESTAMP) AS DATETIME) AS pickup,
    CAST(fare_amount AS FLOAT64) AS fare
  FROM `{{staging}}`
  WHERE CAST(CAST(tpep_pickup_datetime AS TIMESTAMP) AS DATETIME)
        >= DATETIME(@month)
    AND CAST(CAST(tpep_pickup_datetime AS TIMESTAMP) AS DATETIME)
        < DATETIME(DATE_ADD(@month, INTERVAL 1 MONTH))
)
{HOURLY_SELECT};

-- Same columns and the same 1-in-{SAMPLE_RATE} rule as yellow_temporal_sample.
INSERT INTO `{TLC_SAMPLE_REF}`
SELECT
  CAST(tpep_pickup_datetime AS TIMESTAMP)   AS pickup_datetime,
  CAST(tpep_dropoff_datetime AS TIMESTAMP)  AS dropoff_datetime,
  SAFE_CAST(passenger_count AS INT64)       AS passenger_count,
  CAST(trip_distance AS FLOAT64)            AS trip_distance,
  CAST(fare_amount AS FLOAT64)              AS fare_amount,
  CAST(tip_amount AS FLOAT64)               AS tip_amount,
  CAST(total_amount AS FLOAT64)             AS total_amount,
  CAST(payment_type AS STRING)              AS payment_type,
  FORMAT_DATE('%Y', @month)                 AS source_year
FROM `{{staging}}`
WHERE MOD(ABS(FARM_FINGERPRINT(CONCAT(
        CAST(CAST(tpep_pickup_datetime AS TIMESTAMP) AS STRING),
        CAST(total_amount AS STRING)))), {SAMPLE_RATE}) = 0
  AND CAST(CAST(tpep_pickup_datetime AS TIMESTAMP) AS DATETIME)
      >= DATETIME(@month)
  AND CAST(CAST(tpep_pickup_datetime AS TIMESTAMP) AS DATETIME)
      < DATETIME(DATE_ADD(@month, INTERVAL 1 MONTH));

INSERT INTO `{LEDGER_REF}` (month, loaded_at, trips, sampled_trips, source_url)
SELECT
  @month,
  CURRENT_TIMESTAMP(),
  (SELECT SUM(trips) FROM `{HOURLY_REF}` WHERE source = @source AND month = @month),
  (SELECT COUNT(*) FROM `{TLC_SAMPLE_REF}`
   WHERE DATE_TRUNC(DATE(pickup_datetime), MONTH) = @month),
  @url;

COMMIT TRANSACTION;
"""


def ensure_tables(client: bigquery.Client) -> None:
    """Create the tables and view this script writes to, if they're missing."""
    hourly = bigquery.Table(HOURLY_REF, schema=HOURLY_SCHEMA)
    hourly.description = (
        "Trips and fares per pickup hour, counted from every trip (not a sample)."
    )
    hourly.clustering_fields = ["month"]
    client.create_table(hourly, exists_ok=True)

    ledger = bigquery.Table(LEDGER_REF, schema=LEDGER_SCHEMA)
    ledger.description = "TLC monthly files loaded by etl/load_tlc_months.py."
    client.create_table(ledger, exists_ok=True)

    # Same columns, in the same order, as the 2011 - Nov 2022 sample.
    tlc_sample = bigquery.Table(
        TLC_SAMPLE_REF, schema=client.get_table(PUBLIC_SAMPLE_REF).schema
    )
    tlc_sample.description = (
        f"1 in {SAMPLE_RATE} yellow taxi trips from the TLC monthly files, Dec 2022 on."
    )
    client.create_table(tlc_sample, exists_ok=True)

    client.query_and_wait(f"""
        CREATE OR REPLACE VIEW `{SAMPLE_VIEW_REF}`
        OPTIONS (description = '1 in {SAMPLE_RATE} yellow taxi trips, 2011 to the latest TLC month.')
        AS
        SELECT * FROM `{PUBLIC_SAMPLE_REF}`
        UNION ALL
        SELECT * FROM `{TLC_SAMPLE_REF}`
    """)


def month_url(month: date) -> str:
    return TLC_URL.format(month=month.strftime("%Y-%m"))


def is_published(month: date) -> bool:
    request = urllib.request.Request(
        month_url(month), method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True
    except urllib.error.HTTPError:
        return False  # the TLC returns 403 for months not published yet


def next_month(month: date) -> date:
    return (month.replace(day=1) + timedelta(days=32)).replace(day=1)


def published_months() -> list[date]:
    """Every TLC month from FIRST_TLC_MONTH up to the newest one published."""
    months = []
    month = FIRST_TLC_MONTH
    this_month = date.today().replace(day=1)
    while month <= this_month and is_published(month):
        months.append(month)
        month = next_month(month)
    return months


def loaded_months(client: bigquery.Client) -> set[date]:
    rows = client.query_and_wait(f"SELECT month FROM `{LEDGER_REF}`")
    return {row.month for row in rows}


def run(client: bigquery.Client, sql: str, **params) -> None:
    types = {date: "DATE", str: "STRING"}
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(name, types[type(value)], value)
            for name, value in params.items()
        ]
    )
    client.query_and_wait(sql, job_config=config)


def build_public_hourly(client: bigquery.Client) -> None:
    print(f"Building hourly totals for 2011 - {FIRST_TLC_MONTH:%b %Y} (exclusive)...")
    started = time.perf_counter()
    run(
        client,
        PUBLIC_HOURLY_SQL,
        source="bq_public",
        first_tlc_month=FIRST_TLC_MONTH,
    )
    print(f"Done in {time.perf_counter() - started:.0f}s")


def load_month(client: bigquery.Client, month: date) -> None:
    started = time.perf_counter()
    url = month_url(month)
    staging_ref = f"{DATASET_REF}._staging_tlc_{month:%Y%m}"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trips.parquet"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response, open(path, "wb") as f:
            shutil.copyfileobj(response, f)

        load_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with open(path, "rb") as f:
            client.load_table_from_file(f, staging_ref, job_config=load_config).result()

    try:
        # In case this run dies before the cleanup below, BigQuery deletes it.
        staging = client.get_table(staging_ref)
        staging.expires = datetime.now(timezone.utc) + timedelta(days=1)
        client.update_table(staging, ["expires"])

        run(
            client,
            MONTH_SQL.replace("{staging}", staging_ref),
            source="tlc",
            month=month,
            url=url,
        )
    finally:
        client.delete_table(staging_ref, not_found_ok=True)

    row = next(iter(client.query_and_wait(
        f"SELECT trips, sampled_trips FROM `{LEDGER_REF}` WHERE month = '{month}'"
    )))
    print(
        f"  {month:%Y-%m}: {row.trips:,} trips, {row.sampled_trips:,} sampled "
        f"({time.perf_counter() - started:.0f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--month", help="load (or reload) this month, as YYYY-MM")
    parser.add_argument(
        "--public",
        action="store_true",
        help="(re)build hourly totals for 2011 - Nov 2022 from the public tables",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the months that would be loaded"
    )
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    if not args.dry_run:
        ensure_tables(client)

    if args.public:
        if args.dry_run:
            print("Would rebuild hourly totals for 2011 - Nov 2022")
        else:
            build_public_hourly(client)
        return

    if args.month:
        month = datetime.strptime(args.month, "%Y-%m").date()
        if month < FIRST_TLC_MONTH:
            parser.error(f"TLC months start at {FIRST_TLC_MONTH:%Y-%m}; earlier ones come from --public")
        pending = [month]
    else:
        print("Checking which months the TLC has published...")
        published = published_months()
        try:
            already = loaded_months(client)
        except Exception:
            already = set()  # dry run before the ledger table exists
        pending = [m for m in published if m not in already]
        if published:
            print(
                f"Published: {published[0]:%Y-%m} to {published[-1]:%Y-%m}; "
                f"{len(published) - len(pending)} already loaded"
            )

    if not pending:
        print("Nothing to load.")
        return
    print(f"{'Would load' if args.dry_run else 'Loading'} {len(pending)} month(s): "
          f"{pending[0]:%Y-%m} to {pending[-1]:%Y-%m}")
    if args.dry_run:
        return
    for month in pending:
        load_month(client, month)


if __name__ == "__main__":
    main()
