---
title: Sql Repair Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


# SQL Query Repair — RL Environment

An [OpenEnv](https://github.com/openenv)-compliant reinforcement-learning environment
where an AI agent repairs broken SQL queries against a live in-memory SQLite database.

## Motivation

SQL bugs cost engineering teams hours every week. Analysts write queries with subtle
logical errors wrong JOIN types, incorrect filter values, broken window functions
that silently return wrong results. This environment trains and evaluates agents on
exactly these real-world repair tasks, with deterministic graders and partial-progress
reward signals that make learning tractable.

Unlike toy environments, every task here maps to a class of bug that appears regularly
in production data pipelines. The hard task (`optimization_fix`) requires fixing two
independent bugs simultaneously matching the compound errors that frustrate even
experienced engineers.

---

## Environment Description

The agent interacts with a FastAPI server that manages an in-memory SQLite database.
On each episode the agent receives a broken SQL query, the full database schema, and
a description of the expected output. The agent submits fixed queries (up to 3
attempts) and receives a reward signal and feedback after each attempt.

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `fixed_query` | `string` | The agent's proposed corrected SQL query |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `string` | Identifier of the current task |
| `difficulty` | `string` | `easy`, `medium`, or `hard` |
| `task_description` | `string` | Natural-language description of the bug(s) to fix |
| `schema_ddl` | `string` | `CREATE TABLE` statements for the database |
| `broken_query` | `string` | The SQL query the agent must repair |
| `expected_row_count` | `integer` | How many rows the correct query must return |
| `attempt` | `integer` | Current attempt number (0 on reset, 1–3 during episode) |
| `feedback` | `string` | Result of the last attempt (empty on reset) |

---

## Task Descriptions

| Task ID | Difficulty | Description | Success Criteria |
|---------|------------|-------------|-----------------|
| `syntax_fix` | Easy | Fix a missing comma between column names in a SELECT statement | Returns exactly 4 Engineering employees |
| `logic_fix` | Medium | Fix INNER→LEFT JOIN and wrong status filter value (`'complete'` → `'completed'`) | Returns all 5 customers including those with zero completed order totals |
| `optimization_fix` | Hard | Fix two independent bugs in a CTE ranking query: INNER→LEFT JOIN + COALESCE, and RANK()→DENSE_RANK() | Returns exactly 8 rows (top-2 per department with correct tie-breaking) |

---

## Reward Function

Rewards provide a signal at every step — not just at episode end:

| Outcome | Score |
|---------|-------|
| Query contains forbidden keyword (DROP, DELETE, UPDATE…) | `0.0` |
| Query fails to parse or raises a SQL runtime error | `0.1` |
| Query runs but returns wrong number of rows | `0.4` |
| Correct answer on attempt 3 | `0.7` |
| Correct answer on attempt 2 | `0.85` |
| Correct answer on attempt 1 | `1.0` |

The hard task is designed so that fixing only one of the two bugs yields a different
wrong row count (5, 6, or 7 instead of 8), ensuring the agent cannot partially succeed
— it must identify and fix both bugs to earn any correctness reward.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": ""}` (empty = random) |
| `POST` | `/step` | Submit a fix. Body: `{"fixed_query": "SELECT ..."}` |
| `GET` | `/state` | Current episode state (task_id, attempt, last_score) |
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/tasks` | List all tasks with metadata |

---

## Setup & Usage

### Local (Python)

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server (Terminal 1)
uvicorn env.server:app --host 0.0.0.0 --port 7860 --reload

# 4. Run baseline inference (Terminal 2)
export HF_TOKEN=hf_yourTokenHere
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker

```bash
# Build
docker build -t sql-repair-env .

# Run
docker run -p 7860:7860 sql-repair-env

# Run inference against the container
export HF_TOKEN=hf_yourTokenHere
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate openenv.yaml
```

---

## Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API:

| Task | Difficulty | Score | Notes |
|------|------------|-------|-------|
| `syntax_fix` | Easy | 1.00 | Solved on attempt 1 |
| `logic_fix` | Medium | 1.00 | Solved on attempt 1 |
| `optimization_fix` | Hard | 0.40 | Fixes LEFT JOIN but misses DENSE_RANK() |
| **Average** | | **0.80** | |

The hard task correctly challenges frontier models — the model consistently fixes one
bug (LEFT JOIN + COALESCE) but fails to identify the RANK() → DENSE_RANK() issue,
demonstrating meaningful difficulty progression in the environment.

---

## Project Structure

```
sql-repair-env/
├── env/
│   ├── __init__.py        # Package exports
│   ├── client.py          # Async HTTP client (SQLRepairEnv SDK)
│   ├── environment.py     # Core environment logic
│   ├── models.py          # Pydantic models
│   ├── server.py          # FastAPI server
│   └── tasks.py           # Task catalogue + grader
├── inference.py           # Baseline LLM agent script
├── openenv.yaml           # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Requirements

```
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
httpx>=0.27.0
openai>=1.0.0
python-dotenv>=1.0.0
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace API token for inference |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `IMAGE_NAME` | No | `None` (uses localhost) | Docker image name |