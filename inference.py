"""
inference.py — LLM agent baseline for the SQL Query Repair RL environment.

Log format
----------
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage
-----
    export HF_TOKEN=hf_...
    python inference.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import OpenAI

from env import SQLRepairEnv, SQLRepairAction
from env.client import Observation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY: str | None = (
    os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
)
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME: str | None = os.getenv("IMAGE_NAME")

BENCHMARK: str = "sql-repair-env"
MAX_STEPS: int = 3
SUCCESS_THRESHOLD: float = 0.6
TASK_IDS: list[str] = ["syntax_fix", "logic_fix", "optimization_fix"]

SYSTEM_PROMPT: str = (
    "You are an expert SQL developer. You will be given a broken SQL query and must fix it.\n"
    "Return ONLY the corrected SQL query — no explanation, no markdown, no backticks. "
    "Just the raw SQL."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={r_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(obs: Observation, step: int, rewards: list[float]) -> str:
    lines: list[str] = [
        f"## Task: {obs.task_description}",
        "",
        "### Database Schema",
        "```sql",
        obs.schema_ddl,
        "```",
        "",
        "### Broken Query (fix this)",
        "```sql",
        obs.broken_query,
        "```",
    ]

    if step > 1 and obs.feedback:
        lines += [
            "",
            "### Feedback from your last attempt",
            obs.feedback,
        ]

    if step > 1 and rewards:
        prev_scores = ", ".join(f"{r:.2f}" for r in rewards)
        lines += [
            "",
            f"### Reward history so far: [{prev_scores}]",
            f"Attempt {step} of {MAX_STEPS} — try a different approach.",
        ]

    lines += [
        "",
        "Return ONLY the corrected SQL query. No explanation, no markdown fences.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(client: OpenAI, prompt: str, fallback_query: str = "") -> str:
    """Call LLM; fall back to fallback_query on ANY exception so episode continues."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        raw: str = response.choices[0].message.content or ""
        return _strip_markdown(raw).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] LLM call failed: {exc} — using fallback query", flush=True)
        return fallback_query


def _strip_markdown(text: str) -> str:
    stripped = text.strip()
    for fence_open in ("```sql\n", "```sql", "```\n", "```"):
        if stripped.startswith(fence_open):
            stripped = stripped[len(fence_open):]
            break
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_task(env: SQLRepairEnv, client: OpenAI, task_id: str) -> float:
    rewards: list[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs: Observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            prompt = build_prompt(obs, step, rewards)
            fixed_sql = call_llm(client, prompt, fallback_query=obs.broken_query)

            result = await env.step(SQLRepairAction(fixed_query=fixed_sql))

            reward: float = result.reward
            done: bool = result.done
            error: str | None = result.info.get("error")
            obs = result.observation

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=fixed_sql.replace("\n", " ").strip(),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score = max(rewards) if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:  # noqa: BLE001
        # Guarantees [END] is always emitted even on unexpected mid-episode errors
        print(f"[WARN] run_task({task_id}) crashed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all tasks sequentially and print a summary table."""

    # ── KEY FIX: OpenAI() raises OpenAIError if api_key is None or "".
    # Use a placeholder string so the constructor always succeeds.
    # Actual API errors surface in call_llm's try/except → fallback SQL.
    effective_key = API_KEY or "no-key-set"
    if not API_KEY:
        print(
            "[WARN] No API key found in HF_TOKEN / API_KEY / OPENAI_API_KEY. "
            "LLM calls will use fallback (broken query). Set HF_TOKEN to enable LLM.",
            flush=True,
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=effective_key)
    env = await SQLRepairEnv.from_docker_image(IMAGE_NAME)

    scores: dict[str, float] = {}
    try:
        for task_id in TASK_IDS:
            scores[task_id] = await run_task(env, client, task_id)
    finally:
        await env.close()

    print("\n=== BASELINE RESULTS ===")
    for tid, s in scores.items():
        print(f"{tid}: {s:.2f}")
    if scores:
        print(f"AVERAGE: {sum(scores.values())/len(scores):.2f}")


if __name__ == "__main__":
    asyncio.run(main())