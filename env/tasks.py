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
-- Acme Corp (customer 1): mix of all statuses
(1,  1, 120.50, 'completed'),
(2,  1, 340.00, 'completed'),
(3,  1,  85.75, 'pending'),
(4,  1, 210.00, 'cancelled'),
-- Bright Ideas (customer 2): only pending/cancelled — NO completed
(5,  2, 500.00, 'pending'),
(6,  2, 125.00, 'cancelled'),
(7,  2,  99.99, 'pending'),
-- Cloud Nine (customer 3): mix
(8,  3, 780.00, 'completed'),
(9,  3, 450.00, 'completed'),
(10, 3, 200.00, 'pending'),
(11, 3,  60.00, 'cancelled'),
-- Delta Systems (customer 4): only completed
(12, 4, 990.00, 'completed'),
(13, 4, 110.50, 'completed'),
(14, 4, 375.25, 'completed'),
-- Echo Partners (customer 5): no orders at all
-- (intentionally empty to test LEFT JOIN behaviour)
-- Extra orders for realism
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
        # Accepted correct form — agents may write equivalent SQL.
        # The grader only checks row count, not exact SQL text.
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
    # Goal: return the top-2 employees by total compensation (salary + bonuses)
    # within each department.  Employees with NO bonus rows must still appear.
    #
    # Two interacting bugs — fixing only one yields a *different* wrong count:
    #   Bug A: RANK() instead of DENSE_RANK()
    #          → Alice & Bob both score 100k in Engineering, so the next person
    #            (Carol, 85k) jumps to RANK 3 and is silently excluded.
    #   Bug B: INNER JOIN bonuses instead of LEFT JOIN + COALESCE
    #          → Frank (Sales) and Iris (HR) have no bonus row; INNER JOIN
    #            drops them from the aggregation pool entirely.
    #
    # Proof that fixing only one bug is insufficient (verified by SQLite):
    #   broken (RANK + INNER JOIN)      → 5 rows   ✗
    #   fix A only (DENSE_RANK + INNER) → 6 rows   ✗
    #   fix B only (RANK + LEFT JOIN)   → 7 rows   ✗
    #   correct (DENSE_RANK + LEFT JOIN)→ 8 rows   ✓
    # -----------------------------------------------------------------------
    {
        "task_id": "optimization_fix",
        "difficulty": "hard",
        "task_description": (
            "This CTE-based query should return the top 2 employees by total "
            "compensation (salary plus any bonus payments) within each department. "
            "Employees who received no bonuses must still be included in the ranking. "
            "The query is returning fewer rows than expected due to two independent "
            "bugs in the join and window-function logic. Identify and fix both."
        ),
        "schema_ddl": """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    department TEXT    NOT NULL,
    salary     REAL    NOT NULL
);

CREATE TABLE bonuses (
    id          INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    amount      REAL    NOT NULL
);
""".strip(),
        "seed_sql": """
-- Engineering: Alice & Bob both earn 100k total → tie at rank-1
-- Carol (85k) must reach dept_rank 2 via DENSE_RANK (RANK gives her rank 3)
INSERT INTO employees (id, name, department, salary) VALUES
(1, 'Alice', 'Engineering', 80000),
(2, 'Bob',   'Engineering', 85000),
(3, 'Carol', 'Engineering', 75000),
(4, 'Dave',  'Engineering', 65000),
-- Sales: Frank has NO bonus row → INNER JOIN silently drops him
(5, 'Eve',   'Sales', 65000),
(6, 'Frank', 'Sales', 60000),
(7, 'Grace', 'Sales', 55000),
-- HR: Iris has NO bonus row → INNER JOIN silently drops her
(8, 'Henry', 'HR', 50000),
(9, 'Iris',  'HR', 55000);

-- Alice  80k + 20k = 100k
-- Bob    85k + 15k = 100k  (tie with Alice)
-- Carol  75k + 10k =  85k
-- Dave   65k +  5k =  70k
-- Eve    65k + 20k =  85k
-- Frank  60k +  0  =  60k  (NO bonus row — tests LEFT JOIN)
-- Grace  55k +  5k =  60k  (ties with Frank when Frank is included)
-- Henry  50k +  8k =  58k
-- Iris   55k +  0  =  55k  (NO bonus row — tests LEFT JOIN)
INSERT INTO bonuses (id, employee_id, amount) VALUES
(1, 1, 20000),
(2, 2, 15000),
(3, 3, 10000),
(4, 4,  5000),
(5, 5, 20000),
(6, 7,  5000),
(7, 8,  8000);
""".strip(),
        "broken_query": (
            "WITH comp AS (\n"
            "    SELECT e.id, e.name, e.department,\n"
            "           e.salary + SUM(b.amount) AS total_comp\n"
            "    FROM employees e\n"
            "    INNER JOIN bonuses b ON e.id = b.employee_id\n"
            "    GROUP BY e.id, e.name, e.department, e.salary\n"
            "),\n"
            "ranked AS (\n"
            "    SELECT name, department, total_comp,\n"
            "           RANK() OVER (PARTITION BY department\n"
            "                        ORDER BY total_comp DESC) AS dept_rank\n"
            "    FROM comp\n"
            ")\n"
            "SELECT name, department, total_comp, dept_rank\n"
            "FROM ranked\n"
            "WHERE dept_rank <= 2\n"
            "ORDER BY department, dept_rank, name;"
        ),
        "correct_query": (
            "WITH comp AS (\n"
            "    SELECT e.id, e.name, e.department,\n"
            "           e.salary + COALESCE(SUM(b.amount), 0) AS total_comp\n"
            "    FROM employees e\n"
            "    LEFT JOIN bonuses b ON e.id = b.employee_id\n"
            "    GROUP BY e.id, e.name, e.department, e.salary\n"
            "),\n"
            "ranked AS (\n"
            "    SELECT name, department, total_comp,\n"
            "           DENSE_RANK() OVER (PARTITION BY department\n"
            "                              ORDER BY total_comp DESC) AS dept_rank\n"
            "    FROM comp\n"
            ")\n"
            "SELECT name, department, total_comp, dept_rank\n"
            "FROM ranked\n"
            "WHERE dept_rank <= 2\n"
            "ORDER BY department, dept_rank, name;"
        ),
        # expected: 3 (Engineering) + 3 (Sales) + 2 (HR) = 8
        "expected_row_count": 8,
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