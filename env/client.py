"""
env/client.py — Async HTTP client for the SQL Query Repair RL server.

Wraps the flat StepResult returned by env/server.py into the structured
  result.observation / result.reward / result.done / result.info
shape expected by inference.py.

Public API
----------
    env = await SQLRepairEnv.from_docker_image(image_name)   # or None → localhost
    result = await env.reset(task_id="syntax_fix")           # or "" for random
    result = await env.step(SQLRepairAction(fixed_query=sql))
    await env.close()

Both reset() and step() return an EpisodeResult.  The .observation field is
a StepResult (all server fields available as attributes).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared action model (used by both client and inference.py)
# ---------------------------------------------------------------------------


class SQLRepairAction(BaseModel):
    """Agent action — the proposed SQL fix."""

    fixed_query: str


# ---------------------------------------------------------------------------
# Thin wrapper around the server's StepResult JSON
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """
    All observation fields surfaced directly as attributes so inference.py
    can do  obs.schema_ddl, obs.broken_query, obs.feedback, etc.
    """

    task_id: str
    difficulty: str
    task_description: str
    schema_ddl: str
    broken_query: str
    expected_row_count: int
    attempt: int
    feedback: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        return cls(
            task_id=d["task_id"],
            difficulty=d["difficulty"],
            task_description=d["task_description"],
            schema_ddl=d["schema_ddl"],
            broken_query=d["broken_query"],
            expected_row_count=d["expected_row_count"],
            attempt=d["attempt"],
            feedback=d.get("feedback", ""),
        )


@dataclass
class EpisodeResult:
    """
    Unified return type for reset() and step().

    Attributes
    ----------
    observation : Observation
        Current environment observation.
    reward : float
        Score assigned by the grader for this step (0.0 on reset).
    done : bool
        True when the episode has ended.
    info : dict[str, Any]
        Auxiliary info.  Contains "error" key (str | None) extracted from
        the server's feedback when the grader detected a SQL error.
    """

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "EpisodeResult":
        obs = Observation.from_dict(data)
        reward = float(data.get("score", 0.0))
        done = bool(data.get("done", False))

        # Surface SQL errors in info["error"] so callers can log them cleanly.
        feedback: str = data.get("feedback", "")
        error: str | None = None
        if feedback.startswith("SQL error:"):
            error = feedback[len("SQL error:"):].strip()
        elif feedback.startswith("Blocked:"):
            error = feedback

        return cls(
            observation=obs,
            reward=reward,
            done=done,
            info={"error": error, "feedback": feedback},
        )


# ---------------------------------------------------------------------------
# Async HTTP client
# ---------------------------------------------------------------------------


class SQLRepairEnv:
    """
    Async HTTP client that speaks to env/server.py.

    Use the ``from_docker_image`` class-method to construct an instance;
    it resolves the server base URL from the image name or falls back to
    localhost:7860 when IMAGE_NAME is None (local development).
    """

    DEFAULT_URL = "http://localhost:7860"
    HEALTH_RETRIES = 20
    HEALTH_INTERVAL = 1.5  # seconds between health-check attempts

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(cls, image_name: str | None) -> "SQLRepairEnv":
        """
        Resolve the server URL and wait until it is healthy.

        When *image_name* is None (local dev), connects to localhost:7860.
        When *image_name* is provided the caller is expected to have already
        started the container; this method just waits for /health to respond.

        Args:
            image_name: Docker image name (unused for URL resolution here;
                        set IMAGE_URL env var to override the base URL).

        Returns:
            A ready-to-use :class:`SQLRepairEnv` instance.

        Raises:
            RuntimeError: If the server does not become healthy in time.
        """
        import os

        base_url = os.getenv("ENV_BASE_URL") or cls.DEFAULT_URL
        instance = cls(base_url)
        await instance._wait_for_health()
        return instance

    async def _wait_for_health(self) -> None:
        """Poll /health until the server responds 200 OK."""
        last_exc: Exception | None = None
        for attempt in range(self.HEALTH_RETRIES):
            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    return
            except Exception as exc:
                last_exc = exc
            await asyncio.sleep(self.HEALTH_INTERVAL)
        raise RuntimeError(
            f"Server at {self._base_url} did not become healthy after "
            f"{self.HEALTH_RETRIES} attempts. Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    async def reset(self, task_id: str = "") -> EpisodeResult:
        """
        Start a fresh episode.

        Args:
            task_id: Task to run. Empty string picks a random task.

        Returns:
            :class:`EpisodeResult` with reward=0.0, done=False.
        """
        payload = {"task_id": task_id}
        resp = await self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return EpisodeResult.from_response(resp.json())

    async def step(self, action: SQLRepairAction) -> EpisodeResult:
        """
        Submit a fixed SQL query and advance the episode.

        Args:
            action: :class:`SQLRepairAction` with the agent's SQL fix.

        Returns:
            :class:`EpisodeResult` with the grader's reward and updated obs.
        """
        payload = {"fixed_query": action.fixed_query}
        resp = await self._client.post("/step", json=payload)
        resp.raise_for_status()
        return EpisodeResult.from_response(resp.json())

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()