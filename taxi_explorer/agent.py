"""Ask questions in plain English about the NYC taxi data in BigQuery.

Usage:
    python taxi_explorer/agent.py                       # interactive chat
    python taxi_explorer/agent.py "Busiest year?"       # ask one question and exit
    adk web                                            # browser UI (run from the project root)

Auth: run `gcloud auth application-default login` once beforehand.
Requires: pip install -r requirements-agent.txt
"""

import asyncio
import os
import sys

import google.auth
from google.adk.agents import Agent
from google.adk.integrations.bigquery import (
    BigQueryCredentialsConfig,
    BigQueryToolset,
)
from google.adk.integrations.bigquery.config import BigQueryToolConfig, WriteMode
from google.adk.runners import InMemoryRunner
from google.genai import types

PROJECT_ID = "traffic-513918"
DATASET_ID = "taxi"
TABLE_ID = "yellow_temporal_sample"
BIGQUERY_LOCATION = "US"
MODEL = "gemini-3.5-flash"
MAX_BYTES_PER_QUERY = 1_000_000_000  # queries that would scan more than 1 GB fail

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
"""

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
    tools=[bigquery_tools],
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
