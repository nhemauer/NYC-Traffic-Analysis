# The website's chat endpoint (chat_server.py), deployed to Cloud Run as
# taxi-chat. See chat_server.py for the deploy command.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt requirements-agent.txt ./
COPY etl/requirements.txt etl/
COPY taxi_explorer/requirements.txt taxi_explorer/
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements-agent.txt

COPY taxi_explorer/ taxi_explorer/
COPY chat_server.py .

ENV PYTHONUNBUFFERED=1
CMD exec uvicorn chat_server:app --host 0.0.0.0 --port ${PORT:-8080}
