"""A public chat endpoint for the taxi agent, called by the personal website.

ADK's own server (`adk web`, `adk api_server`) exposes every session and user
endpoint, so it stays private. This serves only POST /chat/stream (which the
page uses: progress and the answer arrive as they happen) and POST /chat (the
whole answer at once), with limits so a public page can't run up the Gemini
and BigQuery bill.

Sessions and rate limits live in memory, so the Cloud Run service must run a
single instance (--max-instances 1), kept running (--min-instances 1) so the
daily limits don't reset when it would otherwise go idle. A restart still
forgets open conversations and the day's counts; the page starts a new one.

Usage:
    uvicorn chat_server:app --port 8080     # local test, from the project root

Deploy (from the project root; the Dockerfile builds it):
    gcloud run deploy taxi-chat --source . --region us-central1 \
        --project traffic-513918 --allow-unauthenticated \
        --service-account taxi-agent-runner@traffic-513918.iam.gserviceaccount.com \
        --max-instances 1 --min-instances 1 --memory 1Gi --timeout 300

Settings (environment variables):
    ALLOWED_ORIGINS    comma-separated origins allowed to call /chat
    DAILY_LIMIT        questions per day for everyone together (default 25)
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from taxi_explorer.agent import NEW_YORK, root_agent

APP_NAME = "taxi_data_explorer"
USER_ID = "website"  # saved with each fare estimate and forecast
MAX_QUESTION_CHARS = 1000
MAX_TURNS_PER_SESSION = 20
SESSION_IDLE_SECONDS = 60 * 60
PER_MINUTE_LIMIT = 6  # questions per visitor (IP address)
PER_DAY_LIMIT = 5
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "25"))
ANSWER_TIMEOUT_SECONDS = 120
# Gemini calls per question. Simple answers take 2-3; open questions such as
# "how did COVID change trips?" explore and can take over 10. ADK's default of
# 500 would let one question run up the bill.
MAX_LLM_CALLS_PER_QUESTION = int(os.environ.get("MAX_LLM_CALLS", "20"))
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://nickhemauer.com,https://www.nickhemauer.com,"
    "https://nhemauer-website.web.app,https://nhemauer-website.firebaseapp.com",
).split(",")

log = logging.getLogger("chat_server")
logging.basicConfig(level=logging.INFO)

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
    max_age=3600,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sql: list[str]  # queries the agent ran, shown on the page
    tools: list[str]  # other tools it used, such as predict_fare


# session id -> [last used (monotonic seconds), turns so far]
sessions: dict[str, list] = {}
recent_by_visitor: dict[str, deque] = defaultdict(deque)
daily = {"date": None, "count": 0, "by_visitor": defaultdict(int)}


def _visitor(request: Request) -> str:
    # Cloud Run's front end appends the caller's address last.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _check_limits(visitor: str) -> None:
    """Raise a 429 if this visitor, or everyone together, has asked too much."""
    today = datetime.now(NEW_YORK).date()
    if daily["date"] != today:
        daily.update(date=today, count=0, by_visitor=defaultdict(int))
    if daily["count"] >= DAILY_LIMIT:
        raise HTTPException(429, "The agent has answered all it can for today. Try again tomorrow.")
    if daily["by_visitor"][visitor] >= PER_DAY_LIMIT:
        raise HTTPException(429, "You've reached today's question limit. Try again tomorrow.")

    now = time.monotonic()
    recent = recent_by_visitor[visitor]
    while recent and now - recent[0] > 60:
        recent.popleft()
    if len(recent) >= PER_MINUTE_LIMIT:
        raise HTTPException(429, "That's a lot of questions at once. Wait a minute and try again.")

    recent.append(now)
    daily["count"] += 1
    daily["by_visitor"][visitor] += 1


async def _forget_idle_sessions() -> None:
    now = time.monotonic()
    for session_id, (last_used, _) in list(sessions.items()):
        if now - last_used > SESSION_IDLE_SECONDS:
            del sessions[session_id]
            await session_service.delete_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
    for visitor, recent in list(recent_by_visitor.items()):
        if not recent or now - recent[-1] > 60:
            del recent_by_visitor[visitor]


# What the page shows while each tool runs.
STEP_LABELS = {
    "list_table_ids": "Listing the tables",
    "get_table_info": "Reading a table's columns",
    "execute_sql": "Running a BigQuery query",
    "predict_fare": "Estimating the fare with the random forest",
    "forecast_trips": "Forecasting with the gradient-boosted models",
}
LIMIT_MESSAGE = (
    "That question needed more steps than this demo allows. Try asking about "
    "one part of it at a time."
)
ERROR_MESSAGE = "The agent hit an error. Try again, or rephrase the question."
EMPTY_MESSAGE = "I couldn't come up with an answer to that. Try rephrasing it."


async def _events(session_id: str, question: str):
    """Run the agent on one question, yielding what it does as it happens.

    Yields dicts: {"type": "step", "text"} when a tool starts, "sql" with
    each query, "delta" with pieces of the answer as Gemini writes it, and
    finally "done" with the whole answer. Raises if the agent fails.
    """
    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer, sql, tools = "", [], []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(
            max_llm_calls=MAX_LLM_CALLS_PER_QUESTION,
            streaming_mode=StreamingMode.SSE,
        ),
    ):
        parts = event.content.parts if event.content and event.content.parts else []
        text = "".join(p.text or "" for p in parts if not p.thought)
        if event.partial:
            # Streaming sends tool calls twice; act on the complete copy only.
            if text:
                yield {"type": "delta", "text": text}
            continue
        for call in event.get_function_calls():
            yield {"type": "step", "text": STEP_LABELS.get(call.name, f"Using {call.name}")}
            if call.name == "execute_sql":
                query = call.args.get("query", "")
                sql.append(query)
                yield {"type": "sql", "query": query}
            elif call.name not in tools:
                tools.append(call.name)
        if event.is_final_response() and text:
            answer = text
    yield {
        "type": "done",
        "session_id": session_id,
        "answer": answer or EMPTY_MESSAGE,
        "sql": sql,
        "tools": tools,
    }


def _hit_call_limit(exc: BaseException) -> bool:
    """Whether the error, or one it was raised from, is ADK's call limit.

    ADK wraps it in its own errors, so look along the chain by name.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        if type(exc).__name__ == "LlmCallsLimitExceededError":
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _failure(exc: Exception, question: str) -> tuple[int, str]:
    """The status code and visitor-facing message for an agent failure."""
    if isinstance(exc, TimeoutError):
        return 504, "The agent took too long. Try a simpler question."
    if _hit_call_limit(exc):
        log.warning("Question hit the %d-call limit: %r", MAX_LLM_CALLS_PER_QUESTION, question)
        return 422, LIMIT_MESSAGE
    log.error("Agent failed on a question", exc_info=exc)
    return 502, ERROR_MESSAGE


async def _start(body: ChatRequest, request: Request) -> tuple[str, str]:
    """Check the question and the limits, and return (session id, question)."""
    question = body.message.strip()
    if not question:
        raise HTTPException(400, "Ask a question first.")
    await _forget_idle_sessions()

    session_id = body.session_id
    if session_id in sessions and sessions[session_id][1] >= MAX_TURNS_PER_SESSION:
        raise HTTPException(429, "This conversation is at its limit. Start a new one.")
    # Limits come before a session is made, so refused requests can't pile up
    # sessions in memory.
    _check_limits(_visitor(request))
    if session_id not in sessions:  # new visitor, or the server restarted
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        session_id = session.id
        sessions[session_id] = [time.monotonic(), 0]
    sessions[session_id][0] = time.monotonic()
    sessions[session_id][1] += 1
    return session_id, question


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Answer one question, sending progress as it happens.

    The body is newline-delimited JSON: one object per line, as described in
    _events, or {"type": "error", "detail"} if the agent fails partway.
    Limits are checked first, so a refused question still gets a plain 429.
    """
    session_id, question = await _start(body, request)

    async def lines():
        try:
            async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
                async for item in _events(session_id, question):
                    yield json.dumps(item) + "\n"
        except Exception as exc:
            _, detail = _failure(exc, question)
            yield json.dumps({"type": "error", "detail": detail}) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Answer one question in a single response (the page before streaming)."""
    session_id, question = await _start(body, request)

    async def collect() -> dict:
        async for item in _events(session_id, question):
            if item["type"] == "done":
                return item
        raise RuntimeError("The agent stopped without finishing.")

    try:
        done = await asyncio.wait_for(collect(), ANSWER_TIMEOUT_SECONDS)
    except Exception as exc:
        raise HTTPException(*_failure(exc, question))
    return ChatResponse(**{k: v for k, v in done.items() if k != "type"})
