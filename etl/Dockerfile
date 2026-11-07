# Image for the tlc-loader Cloud Run job, which runs load_tlc_months.py on a
# schedule. Built from this folder by etl/setup_scheduled_loader.sh.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# load_tlc_months.py imports settings from create_yellow_temporal_sample.py.
COPY create_yellow_temporal_sample.py load_tlc_months.py ./

ENTRYPOINT ["python", "load_tlc_months.py"]
