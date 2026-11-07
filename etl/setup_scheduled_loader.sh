#!/usr/bin/env bash
# Sets up weekly loading of new NYC TLC monthly files into BigQuery.
#
# Creates:
#   * tlc-loader: a service account that can run BigQuery jobs and write to
#     the taxi dataset, and nothing else
#   * tlc-loader: a Cloud Run job that runs etl/load_tlc_months.py
#   * tlc-loader-weekly: a Cloud Scheduler job that starts it every Monday
#
# Safe to run again: it skips what already exists, and redeploys the Cloud Run
# job with the current code. Run it again after changing load_tlc_months.py.
#
# Usage (from the project root):
#   bash etl/setup_scheduled_loader.sh

set -euo pipefail

PROJECT_ID="traffic-513918"
REGION="us-central1"
DATASET="taxi"
JOB_NAME="tlc-loader"
SCHEDULER_NAME="tlc-loader-weekly"
SERVICE_ACCOUNT_NAME="tlc-loader"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
# Mondays at 6am New York time. The TLC posts each month on no fixed day, so a
# weekly check picks up a new file within a week; weeks with nothing new cost
# nothing.
SCHEDULE="0 6 * * 1"
TIME_ZONE="America/New_York"

ETL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Turning on the Cloud Scheduler API"
gcloud services enable cloudscheduler.googleapis.com --project "$PROJECT_ID"

echo "== Service account $SERVICE_ACCOUNT"
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --project "$PROJECT_ID" \
    --display-name "Loads TLC monthly taxi files into BigQuery"
fi

echo "== Letting it run BigQuery jobs"
# A new service account can take a minute to become visible to IAM, so retry.
for attempt in $(seq 1 12); do
  if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
      --member "serviceAccount:$SERVICE_ACCOUNT" \
      --role roles/bigquery.jobUser \
      --condition None >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 12 ]; then
    echo "Couldn't grant roles/bigquery.jobUser after 2 minutes. Rerun this script." >&2
    exit 1
  fi
  echo "   Waiting for the new service account to be ready..."
  sleep 10
done

echo "== Letting it write to the $DATASET dataset only"
# bq can only grant access on tables, so the dataset grant goes through the
# BigQuery Python client from the project's virtual environment.
"$ETL_DIR/../.venv/bin/python" - "$PROJECT_ID" "$DATASET" "$SERVICE_ACCOUNT" <<'PY'
import sys
from google.cloud import bigquery

project, dataset_id, account = sys.argv[1:]
client = bigquery.Client(project=project)
dataset = client.get_dataset(f"{project}.{dataset_id}")
entry = bigquery.AccessEntry("WRITER", "userByEmail", account)
if entry in dataset.access_entries:
    print(f"   {account} can already write to {dataset_id}")
else:
    dataset.access_entries = [*dataset.access_entries, entry]
    client.update_dataset(dataset, ["access_entries"])
    print(f"   Gave {account} write access to {dataset_id}")
PY

echo "== Building and deploying the Cloud Run job (takes a few minutes)"
gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source "$ETL_DIR" \
  --service-account "$SERVICE_ACCOUNT" \
  --memory 1Gi \
  --task-timeout 3600 \
  --max-retries 1

echo "== Letting the scheduler start the job"
gcloud run jobs add-iam-policy-binding "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member "serviceAccount:$SERVICE_ACCOUNT" \
  --role roles/run.invoker >/dev/null

echo "== Weekly schedule"
RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run"
if gcloud scheduler jobs describe "$SCHEDULER_NAME" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  ACTION=update
else
  ACTION=create
fi
gcloud scheduler jobs "$ACTION" http "$SCHEDULER_NAME" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "$SCHEDULE" \
  --time-zone "$TIME_ZONE" \
  --uri "$RUN_URI" \
  --http-method POST \
  --oauth-service-account-email "$SERVICE_ACCOUNT"

echo
echo "Done. The loader runs every Monday at 6am New York time."
echo "Run it now:   gcloud run jobs execute $JOB_NAME --region $REGION --wait"
echo "Past runs:    gcloud run jobs executions list --job $JOB_NAME --region $REGION"
