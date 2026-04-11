<<<<<<< HEAD
---
title: Sql Repair Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


# SQL Query Repair (RL Environment)

An OpenEnv compliant reinforcement-learning environment
where an AI agent repairs broken SQL queries against a live in memory SQLite database.
<img width="1849" height="938" alt="image" src="https://github.com/user-attachments/assets/30b0e905-0fb8-47e6-9ec3-721697699485" />

<img width="1841" height="955" alt="image" src="https://github.com/user-attachments/assets/31430f3e-340d-46c2-9623-e36d7afee8fb" />


---

## Motivation

SQL bugs cost engineering teams hours every week. Analysts write queries with subtle
logical errors wrong JOIN types, incorrect filter values, broken window functions
that silently return wrong results. This environment trains and evaluates agents on
exactly these real world repair tasks, with deterministic graders and partial progress
reward signals that make learning tractable.

Every task maps to a class of bug that appears regularly in production data pipelines.
The hard task (`optimization_fix`) requires fixing two independent bugs simultaneously,
matching the compound errors that frustrate even experienced engineers.

---

## Live Demo

**HuggingFace Space:** https://huggingface.co/spaces/yora3/sql-repair-env

```bash
# Health check
curl https://yora3-sql-repair-env.hf.space/health
# → {"status":"ok"}

# Start a random episode
curl -X POST https://yora3-sql-repair-env.hf.space/reset \
     -H "Content-Type: application/json" -d "{}"

# Submit a fix
curl -X POST https://yora3-sql-repair-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"fixed_query": "SELECT id, name, department FROM employees WHERE department = '\''Engineering'\''"}'
```

---

## Baseline Results

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Score | Notes |
|------|------------|-------|-------|
| `syntax_fix` | Easy | **0.95** | Solved on attempt 1  perfect |
| `logic_fix` | Medium | **0.95** | Solved on attempt 1  perfect |
| `optimization_fix` | Hard | **0.40** | Fixes LEFT JOIN but misses RANK→DENSE_RANK |
| **Average** | | **0.77** | |

The hard task correctly challenges frontier models, Qwen consistently repairs one
of the two bugs (LEFT JOIN + COALESCE) but fails to identify the RANK() → DENSE_RANK()
tie-breaking issue, demonstrating genuine difficulty progression across tasks.

---

## Environment Description

The agent interacts with a FastAPI server backed by an in-memory SQLite database.
On each episode reset the agent receives the broken query, full schema DDL, and a
natural-language description of what needs fixing. The agent submits fixed queries
(up to 3 attempts) and receives a shaped reward and textual feedback after each step.

---

## Tasks

### Task 1 — `syntax_fix` (Easy)

**Schema:** `employees(id, name, department, salary)` 10 rows  
**Bug:** Missing comma between column names in the SELECT list → parse error  
**Broken:** `SELECT id name department FROM employees WHERE department = 'Engineering'`  
**Fixed:** `SELECT id, name, department FROM employees WHERE department = 'Engineering'`  
**Expected rows:** 4

---

### Task 2 — `logic_fix` (Medium)

**Schema:** `customers(id, name, region)` + `orders(id, customer_id, amount, status)`  
**Bugs (2):**
1. `INNER JOIN` silently drops customers with no completed orders
2. Status filter uses `'complete'` instead of `'completed'`

**Broken:**
```sql
SELECT c.name, SUM(o.amount) as total
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
WHERE o.status = 'complete'
GROUP BY c.name
```
**Expected rows:** 5 (all customers, including those with zero totals)

---

### Task 3 — `optimization_fix` (Hard)

**Schema:** `employees(id, name, department, salary)` + `bonuses(id, employee_id, amount)`  
**Goal:** Top-2 employees by total compensation per department  
**Bugs (2, must fix both simultaneously):**
1. `INNER JOIN bonuses` drops employees with no bonus rows (Frank, Iris)
2. `RANK()` skips rank-2 on ties Carol (Engineering) gets rank 3 and is excluded

**Proof that fixing only one bug is insufficient:**

| Fix applied | Rows returned | Score |
|-------------|--------------|-------|
| Neither (broken) | 5 | 0.40 |
| DENSE_RANK only | 6 | 0.40 |
| LEFT JOIN only | 7 | 0.40 |
| **Both (correct)** | **8** | **0.95** |

**Broken:**
```sql
WITH comp AS (
    SELECT e.id, e.name, e.department,
           e.salary + SUM(b.amount) AS total_comp
    FROM employees e
    INNER JOIN bonuses b ON e.id = b.employee_id
    GROUP BY e.id, e.name, e.department, e.salary
),
ranked AS (
    SELECT name, department, total_comp,
           RANK() OVER (PARTITION BY department ORDER BY total_comp DESC) AS dept_rank
    FROM comp
)
SELECT name, department, total_comp, dept_rank
FROM ranked WHERE dept_rank <= 2
ORDER BY department, dept_rank, name;
```

**Fixed:** Replace `INNER JOIN` → `LEFT JOIN` + `COALESCE(SUM(b.amount), 0)`, and `RANK()` → `DENSE_RANK()`

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
| `feedback` | `string` | Result of the last attempt empty on reset |

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `fixed_query` | `string` | The agent's proposed corrected SQL query |

---

## Reward Function

All scores are **strictly between 0 and 1** shaped to provide a learning signal at every step:

| Outcome | Score |
|---------|-------|
| Forbidden keyword (DROP, DELETE, UPDATE…) | `0.05` |
| SQL parse or runtime error | `0.10` |
| Query runs but wrong row count | `0.40` |
| Correct on attempt 3 | `0.70` |
| Correct on attempt 2 | `0.85` |
| Correct on attempt 1 | `0.95` |

**Safety gate:** Queries containing `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`,
`TRUNCATE`, `CREATE`, `REPLACE`, `ATTACH`, `DETACH`, or `PRAGMA` are blocked
and score `0.05` without touching the database.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": ""}` (empty = random task) |
| `POST` | `/step` | Submit a fix. Body: `{"fixed_query": "SELECT ..."}` |
| `GET` | `/state` | Current episode state `task_id`, `attempt`, `last_score` |
| `GET` | `/health` | Liveness probe returns `{"status": "ok"}` |
| `GET` | `/tasks` | List all tasks with metadata |

---

## Setup & Usage

### Local (Python)

```bash
# 1. Clone and create virtual environment
git clone https://github.com/nandinii3/sql-repair-env
cd sql-repair-env
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
# source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your token
echo HF_TOKEN=hf_yourTokenHere > .env

# 4. Start the server (Terminal 1)
uvicorn env.server:app --host 0.0.0.0 --port 7860 --reload

# 5. Run baseline inference (Terminal 2)
python inference.py
```

### Docker

```bash
# Build
docker build -t sql-repair-env .

# Run
docker run -p 7860:7860 sql-repair-env

# Run inference against the container
python inference.py
```

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate openenv.yaml
```

---

## Project Structure

```
sql-repair-env/
├── env/
│   ├── __init__.py        # Package exports
│   ├── client.py          # Async HTTP client (SQLRepairEnv SDK)
│   ├── environment.py     # Core OpenEnv environment logic
│   ├── models.py          # Pydantic v2 models
│   ├── server.py          # FastAPI server (port 7860)
│   └── tasks.py           # Task catalogue + TaskGrader
├── server/
│   └── app.py             # Multi-mode deployment entry point
├── inference.py           # Baseline LLM agent (OpenEnv log format)
├── openenv.yaml           # OpenEnv spec metadata
├── Dockerfile             # HF Spaces / Docker deployment
├── pyproject.toml         # Package config + entry points
├── requirements.txt       # Runtime dependencies
├── .env                   # Local secrets (not committed)
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace API token for LLM inference |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API base URL |
| `IMAGE_NAME` | No | `None` (uses localhost:7860) | Docker image name |

Store these in a `.env` file at the repo root loaded automatically by `python-dotenv`.

---

## Dependencies

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
httpx>=0.27.0
openai>=1.0.0
python-dotenv>=1.0.0
```

---

## Links

- 🤗 **HF Space:** https://huggingface.co/spaces/yora3/sql-repair-env
- 💻 **GitHub:** https://github.com/nandinii3/sql-repair-env
- 📋 **Resources:** https://github.com/raun/openenv-course/tree/main
