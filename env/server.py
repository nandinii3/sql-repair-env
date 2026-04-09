"""
env/server.py — FastAPI server for the SQL Query Repair RL environment.

Exposes an OpenEnv-compatible HTTP API so any agent (local script, HF Space,
notebook, etc.) can interact with the environment over plain JSON.

Endpoints
---------
POST /reset   — start a new episode (optional task_id; random if omitted)
POST /step    — submit a fixed SQL query, receive reward
GET  /state   — current episode state (task_id, attempt, last_score)
GET  /health  — liveness probe
GET  /tasks   — list all tasks with metadata

Run
---
    python env/server.py
    # or
    uvicorn env.server:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import random
import re
import sqlite3
import sys
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path wiring — allow running as  `python env/server.py`  or via uvicorn from
# the project root with  `uvicorn env.server:app`.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.tasks import TASKS, TASK_INDEX, TaskGrader, seed_database  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_IDS: list[str] = [t["task_id"] for t in TASKS]
MAX_ATTEMPTS: int = 3

# Dangerous SQL pattern — identical to the one in environment.py so behaviour
# is consistent whether the agent talks to the old env or this new server.
_DANGEROUS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Module-level server state  (single-session; one episode at a time)
# ---------------------------------------------------------------------------

current_task: dict[str, Any] | None = None
current_db: sqlite3.Connection | None = None
attempt: int = 0
last_score: float = 0.0
last_feedback: str = ""

_grader = TaskGrader()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Body for POST /reset.  All fields are optional — send {} to randomise."""

    task_id: str = Field(
        default="",
        description="Task ID to start. Empty string (default) picks a random task.",
    )


class SQLRepairAction(BaseModel):
    """The agent's proposed SQL fix, submitted to POST /step."""

    fixed_query: str = Field(..., min_length=1, description="Agent's proposed SQL query.")


class StepResult(BaseModel):
    """Unified response for /reset and /step."""

    # Episode identity
    task_id: str
    difficulty: str
    task_description: str
    schema_ddl: str
    broken_query: str
    expected_row_count: int

    # Step state
    attempt: int
    feedback: str

    # Reward (populated after /step; zeros/empty on /reset)
    score: float
    done: bool


class StateResponse(BaseModel):
    task_id: str | None
    attempt: int
    last_score: float


class TaskMeta(BaseModel):
    task_id: str
    difficulty: str
    task_description: str
    broken_query: str
    expected_row_count: int


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Query Repair — RL Environment",
    description=(
        "OpenEnv-compatible REST API. Agents POST to /reset to start an episode, "
        "then POST fixed SQL queries to /step to receive reward signals."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_db(task: dict[str, Any]) -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection and seed it for *task*."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    seed_database(conn, task)
    return conn


def _build_step_result(
    *,
    task: dict[str, Any],
    current_attempt: int,
    feedback: str,
    score: float,
    done: bool,
) -> StepResult:
    return StepResult(
        task_id=task["task_id"],
        difficulty=task["difficulty"],
        task_description=task["task_description"],
        schema_ddl=task["schema_ddl"],
        broken_query=task["broken_query"],
        expected_row_count=task["expected_row_count"],
        attempt=current_attempt,
        feedback=feedback,
        score=score,
        done=done,
    )


def _execute_query(
    conn: sqlite3.Connection, query: str
) -> tuple[list[sqlite3.Row] | None, str | None]:
    """
    Run *query* and return (rows, error).  Never raises.

    Returns (None, error_str) on any exception so callers can pass the error
    directly to the grader.
    """
    try:
        cursor = conn.execute(query)
        return cursor.fetchall(), None
    except sqlite3.Error as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=StepResult, tags=["Environment"])
async def reset_endpoint(request: ResetRequest = ResetRequest()) -> StepResult:  # noqa: B008
    """
    Start a fresh episode.

    Send ``{}`` to pick a random task, or ``{"task_id": "syntax_fix"}`` to
    choose a specific one.  Returns the initial :class:`StepResult` with
    ``score=0.0``, ``done=False``, and ``feedback=""``.
    """
    global current_task, current_db, attempt, last_score, last_feedback

    # Resolve task ---------------------------------------------------------
    task_id = request.task_id.strip() if request.task_id else ""
    if not task_id:
        task_id = random.choice(TASK_IDS)

    if task_id not in TASK_INDEX:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Valid IDs: {TASK_IDS}",
        )

    # Tear down any existing DB connection ---------------------------------
    if current_db is not None:
        try:
            current_db.close()
        except Exception:
            pass

    # Initialise fresh state -----------------------------------------------
    current_task = TASK_INDEX[task_id]
    current_db = _open_db(current_task)
    attempt = 0
    last_score = 0.0
    last_feedback = ""

    return _build_step_result(
        task=current_task,
        current_attempt=attempt,
        feedback="",
        score=0.0,
        done=False,
    )


@app.post("/step", response_model=StepResult, tags=["Environment"])
async def step_endpoint(action: SQLRepairAction) -> StepResult:
    """
    Submit a fixed SQL query and advance the episode.

    **Reward scoring:**

    | Outcome                                      | Score |
    |----------------------------------------------|-------|
    | Dangerous SQL (DROP / DELETE / UPDATE / …)   | 0.0   |
    | Parse or runtime error                        | 0.1   |
    | Query runs but wrong row count                | 0.4   |
    | Correct rows, but correlated sub-query (T3)   | 0.7   |
    | Correct — solved on attempt 3                 | 0.7   |
    | Correct — solved on attempt 2                 | 0.85  |
    | Correct — solved on attempt 1                 | 1.0   |

    The episode ends (``done=True``) when the query is correct **or** the
    agent has used all three attempts.
    """
    global attempt, last_score, last_feedback

    if current_task is None or current_db is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    if attempt >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode already finished (attempt {attempt}/{MAX_ATTEMPTS}). "
                "Call POST /reset to start a new episode."
            ),
        )

    attempt += 1
    query = action.fixed_query.strip()

    # ── Safety gate ────────────────────────────────────────────────────────
    if _DANGEROUS.search(query):
        feedback = (
            "Blocked: query contains a forbidden keyword "
            "(DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, "
            "REPLACE, ATTACH, DETACH, or PRAGMA)."
        )
        final_score = 0.05  # strictly > 0 as required by validator
        done = attempt >= MAX_ATTEMPTS
        last_score = final_score
        last_feedback = feedback
        return _build_step_result(
            task=current_task,
            current_attempt=attempt,
            feedback=feedback,
            score=final_score,
            done=done,
        )

    # ── Execute ────────────────────────────────────────────────────────────
    rows, error = _execute_query(current_db, query)

    # ── Grade ──────────────────────────────────────────────────────────────
    base_score, feedback = _grader.grade(
        task=current_task,
        result_rows=rows,
        query_used=query,
        error=error,
    )

    # ── Apply attempt-based scaling for correct answers ────────────────────
    # Scores must be strictly between 0 and 1 (validator rejects 0.0 and 1.0)
    if base_score == 1.0:
        if attempt == 1:
            final_score = 0.95
        elif attempt == 2:
            final_score = 0.85
        else:
            final_score = 0.7
    else:
        final_score = base_score

    done = (base_score == 1.0) or (attempt >= MAX_ATTEMPTS)
    last_score = final_score
    last_feedback = feedback

    return _build_step_result(
        task=current_task,
        current_attempt=attempt,
        feedback=feedback,
        score=final_score,
        done=done,
    )


@app.get("/state", response_model=StateResponse, tags=["Environment"])
async def state_endpoint() -> StateResponse:
    """Return a lightweight snapshot of the current episode state."""
    return StateResponse(
        task_id=current_task["task_id"] if current_task else None,
        attempt=attempt,
        last_score=last_score,
    )


@app.get("/health", tags=["Meta"])
async def health() -> dict[str, str]:
    """Liveness probe — always returns 200 OK while the server is running."""
    return {"status": "ok"}


@app.get("/tasks", response_model=list[TaskMeta], tags=["Discovery"])
async def list_tasks() -> list[TaskMeta]:
    """List all available tasks with their metadata (no seed data)."""
    return [
        TaskMeta(
            task_id=t["task_id"],
            difficulty=t["difficulty"],
            task_description=t["task_description"],
            broken_query=t["broken_query"],
            expected_row_count=t["expected_row_count"],
        )
        for t in TASKS
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "env.server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )