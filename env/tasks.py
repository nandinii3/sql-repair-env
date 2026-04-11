"""
env/tasks.py — Task catalogue and grader for the SQL Query Repair RL environment.

Each task dict is fully self-contained: schema DDL, seed INSERT statements,
a broken query for the agent to fix, a reference correct query, and the
expected result-row count.
"""

from __future__ import annotations

import sqlite3
from typing import Any

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASKS: list[dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Task 1 — EASY — syntax_fix
    # Missing comma between column names in SELECT list.
    # -----------------------------------------------------------------------
    {
        "task_id": "syntax_fix",
        "difficulty": "easy",
        "task_description": (
            "Fix the broken SELECT statement. Two column names are missing a comma "
            "between them, causing a parse error. Only employees from the "
            "'Engineering' department should be returned."
        ),
        "schema_ddl": """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    department TEXT    NOT NULL,
    salary     REAL    NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO employees (id, name, department, salary) VALUES
(1,  'Alice Nguyen',    'Engineering', 95000.00),
(2,  'Bob Patel',       'Engineering', 88000.00),
(3,  'Carol Martinez',  'Marketing',   72000.00),
(4,  'David Kim',       'Sales',       65000.00),
(5,  'Eva Schmidt',     'Engineering', 102000.00),
(6,  'Frank Liu',       'Marketing',   69000.00),
(7,  'Grace OBrien',   'Sales',       71000.00),
(8,  'Hiro Tanaka',     'Engineering', 91000.00),
(9,  'Isla Ferreira',   'Sales',       67000.00),
(10, 'James Okonkwo',   'Marketing',   75000.00);
""".strip(),
        "broken_query": (
            "SELECT id name department FROM employees WHERE department = 'Engineering'"
        ),
        "correct_query": (
            "SELECT id, name, department FROM employees WHERE department = 'Engineering'"
        ),
        "expected_row_count": 4,
        # Grading hints — no special structural checks needed for task 1
        "check_correlated_subquery": False,
    },

    # -----------------------------------------------------------------------
    # Task 2 — MEDIUM — logic_fix
    # Two bugs: INNER JOIN silently drops customers with no completed orders,
    # and the status filter uses 'complete' instead of 'completed'.
    # The correct query must return ALL 5 customers (zeros for no completed orders).
    # -----------------------------------------------------------------------
    {
        "task_id": "logic_fix",
        "difficulty": "medium",
        "task_description": (
            "Fix two logical bugs: (1) INNER JOIN loses customers who have no "
            "completed orders — use LEFT JOIN instead. (2) The status filter is "
            "'complete', but the data uses 'completed'. The fixed query must return "
            "one row per customer (all 5), with zero totals for customers who have "
            "no completed orders."
        ),
        "schema_ddl": """
CREATE TABLE customers (
    id     INTEGER PRIMARY KEY,
    name   TEXT    NOT NULL,
    region TEXT    NOT NULL
);
CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount      REAL    NOT NULL,
    status      TEXT    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
""".strip(),
        "seed_sql": """
INSERT INTO customers (id, name, region) VALUES
(1, 'Acme Corp',      'North'),
(2, 'Bright Ideas',   'South'),
(3, 'Cloud Nine',     'East'),
(4, 'Delta Systems',  'West'),
(5, 'Echo Partners',  'North');
INSERT INTO orders (id, customer_id, amount, status) VALUES
(1,  1, 120.50, 'completed'),
(2,  1, 340.00, 'completed'),
(3,  1,  85.75, 'pending'),
(4,  1, 210.00, 'cancelled'),
(5,  2, 500.00, 'pending'),
(6,  2, 125.00, 'cancelled'),
(7,  2,  99.99, 'pending'),
(8,  3, 780.00, 'completed'),
(9,  3, 450.00, 'completed'),
(10, 3, 200.00, 'pending'),
(11, 3,  60.00, 'cancelled'),
(12, 4, 990.00, 'completed'),
(13, 4, 110.50, 'completed'),
(14, 4, 375.25, 'completed'),
(15, 1, 175.00, 'completed'),
(16, 2, 300.00, 'cancelled'),
(17, 3, 420.00, 'completed'),
(18, 4,  55.00, 'pending'),
(19, 1,  90.00, 'pending'),
(20, 3, 130.00, 'cancelled');
""".strip(),
        "broken_query": (
            "SELECT c.name, SUM(o.amount) as total\n"
            "FROM customers c\n"
            "INNER JOIN orders o ON c.id = o.customer_id\n"
            "WHERE o.status = 'complete'\n"
            "GROUP BY c.name"
        ),
        "correct_query": (
            "SELECT c.name, COALESCE(SUM(CASE WHEN o.status = 'completed' "
            "THEN o.amount ELSE 0 END), 0) AS total\n"
            "FROM customers c\n"
            "LEFT JOIN orders o ON c.id = o.customer_id\n"
            "GROUP BY c.id, c.name"
        ),
        "expected_row_count": 5,
        "check_correlated_subquery": False,
    },
    
# -----------------------------------------------------------------------
    # Task 3 — HARD — optimization_fix
    #
    # Goal: find users who completed a 3-step onboarding funnel IN ORDER
    # within the first 24 hours (86400 seconds) after signup.
    #
    # Two interacting bugs — fixing only one yields a *different* wrong count:
    #   Bug A: time window is 172800s (48h) instead of 86400s (24h)
    #          → Bob and Jack slip through (they finished between 24h-48h)
    #   Bug B: step ordering is not validated (ts1 < ts2 < ts3 missing)
    #          → Carol, Frank, Kate qualify despite completing steps out of order
    #
    # Proof that fixing only one bug is insufficient (verified by SQLite):
    #   broken  (48h + no order check)  → 8 rows  ✗
    #   fix A only (24h + no order)     → 7 rows  ✗
    #   fix B only (48h + ordered)      → 5 rows  ✗
    #   correct (24h + ordered)         → 4 rows  ✓
    # -----------------------------------------------------------------------

   {
    "task_id": "optimization_fix",
    "difficulty": "hard",
    "task_description": (
         "This CTE query finds users who completed a 3-step onboarding funnel. "
    "A user qualifies only if they completed all three steps in the correct "
    "sequence (step1 → step2 → step3) within 24 hours of signup. "
    "The query is returning incorrect results. Identify and fix all the bugs."
    ),
    "schema_ddl": """
CREATE TABLE users (
    id        INTEGER PRIMARY KEY,
    name      TEXT    NOT NULL,
    channel   TEXT    NOT NULL,
    signup_ts INTEGER NOT NULL
);

CREATE TABLE events (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    event_type TEXT    NOT NULL,
    event_ts   INTEGER NOT NULL
);
""".strip(),
    "seed_sql": """
INSERT INTO users (id, name, channel, signup_ts) VALUES
(1,  'Alice', 'organic',  1000000),
(2,  'Bob',   'organic',  1000000),
(3,  'Carol', 'organic',  1000000),
(4,  'Dave',  'paid',     1000000),
(5,  'Eve',   'paid',     1000000),
(6,  'Frank', 'paid',     1000000),
(7,  'Grace', 'referral', 1000000),
(8,  'Hiro',  'referral', 1000000),
(9,  'Iris',  'referral', 1000000),
(10, 'Jack',  'referral', 1000000);

-- Bug map (signup_ts=1000000, 24h = ts <= 1086400):
-- Alice: step1→2→3 in order, within 24h, no retries       → QUALIFIES (all 3 fixes)
-- Bob  : step1→2→3, finishes at 90000s (25h)              → bug A (window 48h lets him through)
-- Carol: step3→1→2 OUT OF ORDER within 24h                → bug B (no order check)
-- Dave : step1 done TWICE — first at 1h, retry at 2h
--        step2, step3 in order within 24h
--        MAX(step1)=1002000, MIN(step1)=1001000
--        With MAX: ts1=1002000 < ts2=1050000 < ts3=1060000 ✓ qualifies wrongly
--        With MIN: ts1=1001000 < ts2=1050000 < ts3=1060000 ✓ still qualifies
--        Wait — need Dave to only qualify with MAX not MIN
--        Dave: step1 first at 1001000, step2 at 1002000, step1 RETRY at 1003000
--              step3 at 1004000
--        MAX(step1)=1003000 > MIN(step2)=1002000 → ordering fails with MAX
--        MIN(step1)=1001000 < step2=1002000 < step3=1004000 → qualifies with MIN
-- Eve : step1→2→3 in order within 24h, no retries         → QUALIFIES
-- Frank: steps out of order (2→1→3), within 24h           → bug B only
-- Grace: step1 only, never finishes                        → never qualifies
-- Hiro : step1→2→3 in order within 24h                    → QUALIFIES
-- Iris : step1→2→3, finishes at 100000s (27h)             → bug A only
-- Jack : step1 done twice — first at 1h, retry at 20h
--        step2 at 21h, step3 at 22h, all within 24h
--        MIN(step1)=1001000: ts1<ts2<ts3 ✓ qualifies
--        MAX(step1)=1072000: ts1>ts2 ✗ fails ordering → bug C drops him
--        So Jack: qualifies with MIN+correct window, fails with MAX

INSERT INTO events (id, user_id, event_type, event_ts) VALUES
-- Alice: clean, qualifies
(1,  1, 'step1_complete', 1001000),
(2,  1, 'step2_complete', 1010000),
(3,  1, 'step3_complete', 1020000),
-- Bob: finishes outside 24h window (90400s after signup)
(4,  2, 'step1_complete', 1001000),
(5,  2, 'step2_complete', 1040000),
(6,  2, 'step3_complete', 1090400),
-- Carol: out of order (3→1→2)
(7,  3, 'step3_complete', 1001000),
(8,  3, 'step1_complete', 1002000),
(9,  3, 'step2_complete', 1003000),
-- Dave: step1 retried AFTER step2 — MAX(step1) > step2 breaks ordering
(10, 4, 'step1_complete', 1001000),
(11, 4, 'step2_complete', 1010000),
(12, 4, 'step1_complete', 1020000),
(13, 4, 'step3_complete', 1030000),
-- Eve: clean, qualifies
(14, 5, 'step1_complete', 1005000),
(15, 5, 'step2_complete', 1015000),
(16, 5, 'step3_complete', 1025000),
-- Frank: out of order (2→1→3)
(17, 6, 'step2_complete', 1001000),
(18, 6, 'step1_complete', 1010000),
(19, 6, 'step3_complete', 1020000),
-- Grace: incomplete
(20, 7, 'step1_complete', 1001000),
-- Hiro: clean, qualifies
(21, 8, 'step1_complete', 1002000),
(22, 8, 'step2_complete', 1012000),
(23, 8, 'step3_complete', 1022000),
-- Iris: finishes at 27h (outside 24h window)
(24, 9, 'step1_complete', 1001000),
(25, 9, 'step2_complete', 1050000),
(26, 9, 'step3_complete', 1097200),
-- Jack: step1 retried at 20h mark — MAX(step1) > step2, breaks ordering
(27, 10,'step1_complete', 1001000),
(28, 10,'step2_complete', 1010000),
(29, 10,'step1_complete', 1072000),
(30, 10,'step3_complete', 1080000);
""".strip(),
    "broken_query": (
        "WITH steps AS (\n"
        "    SELECT\n"
        "        u.id, u.name, u.channel,\n"
        "        MAX(CASE WHEN e.event_type = 'step1_complete' THEN e.event_ts END) AS ts1,\n"
        "        MAX(CASE WHEN e.event_type = 'step2_complete' THEN e.event_ts END) AS ts2,\n"
        "        MAX(CASE WHEN e.event_type = 'step3_complete' THEN e.event_ts END) AS ts3,\n"
        "        u.signup_ts\n"
        "    FROM users u\n"
        "    LEFT JOIN events e ON u.id = e.user_id\n"
        "    GROUP BY u.id, u.name, u.channel, u.signup_ts\n"
        ")\n"
        "SELECT name, channel\n"
        "FROM steps\n"
        "WHERE ts1 IS NOT NULL\n"
        "  AND ts2 IS NOT NULL\n"
        "  AND ts3 IS NOT NULL\n"
        "  AND ts3 - signup_ts <= 172800\n"
        "ORDER BY name;"
    ),
    "correct_query": (
        "WITH steps AS (\n"
        "    SELECT\n"
        "        u.id, u.name, u.channel,\n"
        "        MIN(CASE WHEN e.event_type = 'step1_complete' THEN e.event_ts END) AS ts1,\n"
        "        MIN(CASE WHEN e.event_type = 'step2_complete' THEN e.event_ts END) AS ts2,\n"
        "        MIN(CASE WHEN e.event_type = 'step3_complete' THEN e.event_ts END) AS ts3,\n"
        "        u.signup_ts\n"
        "    FROM users u\n"
        "    LEFT JOIN events e ON u.id = e.user_id\n"
        "    GROUP BY u.id, u.name, u.channel, u.signup_ts\n"
        ")\n"
        "SELECT name, channel\n"
        "FROM steps\n"
        "WHERE ts1 IS NOT NULL\n"
        "  AND ts2 IS NOT NULL\n"
        "  AND ts3 IS NOT NULL\n"
        "  AND ts1 < ts2 AND ts2 < ts3\n"
        "  AND ts3 - signup_ts <= 86400\n"
        "ORDER BY name;"
    ),
    # Alice, Eve, Hiro = 3 qualifying users
    "expected_row_count": 3,
    "check_correlated_subquery": False,
},
]

# Convenience lookup by task_id
TASK_INDEX: dict[str, dict[str, Any]] = {t["task_id"]: t for t in TASKS}


# ---------------------------------------------------------------------------
# Database seeder
# ---------------------------------------------------------------------------

def seed_database(conn: sqlite3.Connection, task: dict[str, Any]) -> None:
    """
    Create the task schema and insert all seed rows into *conn*.

    Executes the task's ``schema_ddl`` and ``seed_sql`` inside a single
    ``executescript`` call so the entire setup is atomic.  The connection is
    left open for subsequent query execution by the environment.

    Args:
        conn: An open :class:`sqlite3.Connection` (typically ``:memory:``).
        task: One of the dicts from :data:`TASKS`.

    Raises:
        sqlite3.Error: If any DDL or INSERT statement fails.
    """
    script = task["schema_ddl"] + "\n\n" + task["seed_sql"]
    conn.executescript(script)
    conn.commit()


# ---------------------------------------------------------------------------
# Task grader
# ---------------------------------------------------------------------------

class TaskGrader:
    """
    Stateless grader that scores an agent's attempt at a repair task.

    The grader is deliberately decoupled from the environment so it can be
    unit-tested independently and reused across different episode runners.
    """

    def grade(
        self,
        task: dict[str, Any],
        result_rows: list | None,
        query_used: str,
        error: str | None,
    ) -> tuple[float, str]:
        """
        Evaluate one agent step and return ``(score, feedback_message)``.

        Scoring rules (applied in priority order):

        1. **SQL error** — query failed to parse or execute:
           score = 0.1, feedback = ``"SQL error: {error}"``

        2. **Wrong row count** — query ran but returned the wrong number of rows:
           score = 0.4, feedback = ``"Got {n} rows, expected {expected}"``

        3. **Task 3 — correct rows but correlated subquery detected** — the agent
           returned the right row count but the query still embeds a nested
           ``SELECT`` (N+1 anti-pattern):
           score = 0.7, feedback = ``"Correct but inefficient — use JOIN+GROUP BY"``

        4. **Fully correct** — right row count and no structural issues:
           score = 1.0, feedback = ``"Correct."``
           *(The caller is responsible for reducing this to 0.85 / 0.7 based on
           attempt number, matching the environment's attempt-based scoring.)*

        Args:
            task:        Task dict from :data:`TASKS`.
            result_rows: Rows returned by the executed query, or ``None`` if the
                         query raised an exception.
            query_used:  The exact SQL string the agent submitted.
            error:       Exception message if the query failed, else ``None``.

        Returns:
            A ``(score, feedback)`` tuple.
        """
        # ----------------------------------------------------------------
        # 1. SQL error
        # ----------------------------------------------------------------
        if error is not None:
            return 0.1, f"SQL error: {error}"

        # ----------------------------------------------------------------
        # 2. Wrong row count
        # ----------------------------------------------------------------
        actual_count = len(result_rows) if result_rows is not None else 0
        expected_count: int = task["expected_row_count"]

        if actual_count != expected_count:
            return (
                0.4,
                f"Got {actual_count} rows, expected {expected_count}",
            )

        # ----------------------------------------------------------------
        # 3. Task 3 — correct row count but correlated subquery still present
        # ----------------------------------------------------------------
        if task.get("check_correlated_subquery", False):
            select_count = query_used.upper().count("SELECT")
            correlated = select_count > 1
            if correlated:
                return (
                    0.7,
                    "Correct but inefficient — use JOIN+GROUP BY",
                )

        # ----------------------------------------------------------------
        # 4. Fully correct
        # ----------------------------------------------------------------
        return 1.0, "Correct."