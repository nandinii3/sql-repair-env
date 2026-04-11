
# SQL Query Repair ( RL Environment )

An OpenEnv compliant reinforcement learning environment
where an AI agent repairs broken SQL queries against a live in memory SQLite database.
<img width="1849" height="938" alt="image" src="https://github.com/user-attachments/assets/30b0e905-0fb8-47e6-9ec3-721697699485" />

<img width="1841" height="955" alt="image" src="https://github.com/user-attachments/assets/31430f3e-340d-46c2-9623-e36d7afee8fb" />

---

## Motivation

SQL bugs cost engineering teams hours every week. Analysts write queries with subtle
logical errors wrong JOIN types, incorrect filter values, broken window functions
that silently return wrong results. This environment trains and evaluates agents on
exactly these real-world repair tasks, with deterministic graders and partial-progress
reward signals that make learning tractable.

Every task maps to a class of bug that appears regularly in production data pipelines.
The hard task (`optimization_fix`) requires fixing three independent bugs simultaneously,
matching the compound errors that frustrate even experienced engineers.

---

## Live Demo

**HuggingFace Space:** https://huggingface.co/spaces/yora3/sql-repair-env

```bash
# Health check
curl https://yora3-sql-repair-env.hf.space/health

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

Evaluated using `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Score | Notes |
|------|------------|-------|-------|
| `syntax_fix` | Easy | **0.95** | Solved on attempt 1 |
| `logic_fix` | Medium | **0.95** | Solved on attempt 1 |
| `optimization_fix` | Hard | **0.40** | Fixes window bug but misses MAX→MIN and ordering bugs |
| **Average** | | **0.77** | |

The hard task correctly challenges frontier models Llama consistently repairs one
of the three bugs but fails to identify all issues simultaneously, demonstrating
genuine difficulty progression across tasks.

---

## Environment Description

The agent interacts with a FastAPI server backed by an in memory SQLite database.
On each episode reset the agent receives the broken query, full schema DDL, and a
natural language description of what needs fixing. The agent submits fixed queries
(up to 3 attempts) and receives a shaped reward and textual feedback after each step.

---

## Tasks

### Task 1  `syntax_fix` (Easy)

**Schema:** `employees(id, name, department, salary)` 10 rows  
**Bug:** Missing comma between column names in the SELECT list → parse error  
**Broken:** `SELECT id name department FROM employees WHERE department = 'Engineering'`  
**Fixed:** `SELECT id, name, department FROM employees WHERE department = 'Engineering'`  
**Expected rows:** 4

---

### Task 2  `logic_fix` (Medium)

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

### Task 3  `optimization_fix` (Hard)

**Schema:** `users(id, name, channel, signup_ts)` + `events(id, user_id, event_type, event_ts)`

**Goal:** Find users who completed a 3-step onboarding funnel in the correct sequence
(step1 → step2 → step3) within 24 hours of signup, using their **first** completion
timestamp for each step.

**Bugs (3, must fix all simultaneously):**
1. **Time window** — uses `172800s` (48h) instead of `86400s` (24h)
2. **Ordering** — missing `ts1 < ts2 AND ts2 < ts3` check, so out-of-order completions qualify
3. **Aggregation** — uses `MAX()` instead of `MIN()` for step timestamps, so retried steps use the latest attempt instead of the first

**Proof that fixing any 1 or 2 bugs is insufficient:**

| Fix applied | Rows returned | Correct? |
|-------------|--------------|----------|
| None (broken) | 6 | ✗ |
| Window only (48h→24h) | 5 | ✗ |
| Ordering only | 4 | ✗ |
| MAX→MIN only | 5 | ✗ |
| Window + Ordering | 4 | ✗ |
| Window + MAX→MIN | 4 | ✗ |
| **All three fixed** | **3** | **✓** |

**Broken:**
```sql
WITH steps AS (
    SELECT u.id, u.name, u.channel,
           MAX(CASE WHEN e.event_type = 'step1_complete' THEN e.event_ts END) AS ts1,
           MAX(CASE WHEN e.event_type = 'step2_complete' THEN e.event_ts END) AS ts2,
           MAX(CASE WHEN e.event_type = 'step3_complete' THEN e.event_ts END) AS ts3,
           u.signup_ts
    FROM users u
    LEFT JOIN events e ON u.id = e.user_id
    GROUP BY u.id, u.name, u.channel, u.signup_ts
)
SELECT name, channel FROM steps
WHERE ts1 IS NOT NULL AND ts2 IS NOT NULL AND ts3 IS NOT NULL
  AND ts3 - signup_ts <= 172800
ORDER BY name;
```

**Fixed:** Replace `MAX()` → `MIN()` for all three steps, add `ts1 < ts2 AND ts2 < ts3`, change `172800` → `86400`  
**Expected rows:** 3 (Alice, Eve, Hiro)

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

| Outcome | Score |
|---------|-------|
| Forbidden keyword (DROP, DELETE, UPDATE…) | `0.05` |
| SQL parse or runtime error | `0.10` |
| Query runs but wrong row count | `0.40` |
| Correct on attempt 3 | `0.70` |
| Correct on attempt 2 | `0.85` |
| Correct on attempt 1 | `0.95` |

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
git clone https://github.com/nandinii3/sql-repair-env
cd sql-repair-env
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

# Terminal 1 — start server
uvicorn env.server:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2 — run baseline
export HF_TOKEN=hf_yourTokenHere
python inference.py
```

### Docker

```bash
docker build -t sql-repair-env .
docker run -p 7860:7860 sql-repair-env
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
├── uv.lock                # Locked dependencies for openenv validate
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace API token for LLM inference |
| `MODEL_NAME` | No | `meta-llama/Llama-3.3-70B-Instruct` | Model identifier |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API base URL |
| `IMAGE_NAME` | No | `None` (localhost:7860) | Docker image name |

---

## Links

- 🤗 **HF Space:** https://huggingface.co/spaces/yora3/sql-repair-env
- 💻 **GitHub:** https://github.com/nandinii3/sql-repair-env
- 📋 **Resources:** https://github.com/raun/openenv-course/tree/main
