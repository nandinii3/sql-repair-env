"""
SQL Query Repair & Optimization RL Environment.

OpenEnv-compliant environment that challenges an agent to fix broken SQL queries
against an in-memory SQLite database.
"""

import re
import sqlite3
from typing import Any

from .models import Action, Observation, Reward

# ---------------------------------------------------------------------------
# Built-in task catalogue
# ---------------------------------------------------------------------------

_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "task_001",
        "task_description": (
            "Fix the broken SELECT query so it returns all employees whose salary "
            "exceeds 50000. The query has a typo in the column name."
        ),
        "schema_ddl": (
            "CREATE TABLE employees ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  department TEXT NOT NULL,"
            "  salary REAL NOT NULL"
            ");"
        ),
        "seed_data": (
            "INSERT INTO employees (name, department, salary) VALUES "
            "('Alice', 'Engineering', 90000),"
            "('Bob', 'Marketing', 45000),"
            "('Carol', 'Engineering', 75000),"
            "('Dave', 'HR', 30000);"
        ),
        "broken_query": "SELECT * FROM employees WHERE salry > 50000;",
        "expected_row_count": 2,
    },
    {
        "task_id": "task_002",
        "task_description": (
            "Fix the query so it returns the total number of orders per customer. "
            "The GROUP BY clause is missing."
        ),
        "schema_ddl": (
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
            "CREATE TABLE orders ("
            "  id INTEGER PRIMARY KEY,"
            "  customer_id INTEGER NOT NULL,"
            "  amount REAL NOT NULL,"
            "  FOREIGN KEY (customer_id) REFERENCES customers(id)"
            ");"
        ),
        "seed_data": (
            "INSERT INTO customers (name) VALUES ('Alice'), ('Bob'), ('Carol');"
            "INSERT INTO orders (customer_id, amount) VALUES "
            "(1, 100), (1, 200), (2, 150), (3, 300), (3, 50);"
        ),
        "broken_query": (
            "SELECT customers.name, COUNT(orders.id) AS order_count "
            "FROM customers JOIN orders ON customers.id = orders.customer_id;"
        ),
        "expected_row_count": 3,
    },
    {
        "task_id": "task_003",
        "task_description": (
            "Fix the query to return products with stock below 10 units. "
            "The comparison operator is wrong (> instead of <)."
        ),
        "schema_ddl": (
            "CREATE TABLE products ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  price REAL NOT NULL,"
            "  stock INTEGER NOT NULL"
            ");"
        ),
        "seed_data": (
            "INSERT INTO products (name, price, stock) VALUES "
            "('Widget A', 9.99, 5),"
            "('Widget B', 19.99, 100),"
            "('Widget C', 4.99, 3),"
            "('Widget D', 29.99, 200);"
        ),
        "broken_query": "SELECT * FROM products WHERE stock > 10;",
        "expected_row_count": 2,
    },
    {
        "task_id": "task_004",
        "task_description": (
            "Fix the query to return students who scored above 80 in both Math "
            "and Science. The AND operator is incorrectly written as OR."
        ),
        "schema_ddl": (
            "CREATE TABLE students ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  math_score INTEGER NOT NULL,"
            "  science_score INTEGER NOT NULL"
            ");"
        ),
        "seed_data": (
            "INSERT INTO students (name, math_score, science_score) VALUES "
            "('Alice', 95, 88),"
            "('Bob', 70, 92),"
            "('Carol', 85, 76),"
            "('Dave', 90, 91);"
        ),
        "broken_query": (
            "SELECT * FROM students WHERE math_score > 80 OR science_score > 80;"
        ),
        "expected_row_count": 2,
    },
    {
        "task_id": "task_005",
        "task_description": (
            "Fix the query to return the names of departments that have more than "
            "2 employees. The HAVING clause threshold is wrong (>= 2 should be > 2)."
        ),
        "schema_ddl": (
            "CREATE TABLE departments ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL"
            ");"
            "CREATE TABLE staff ("
            "  id INTEGER PRIMARY KEY,"
            "  dept_id INTEGER NOT NULL,"
            "  name TEXT NOT NULL"
            ");"
        ),
        "seed_data": (
            "INSERT INTO departments (name) VALUES ('Engineering'), ('HR'), ('Sales');"
            "INSERT INTO staff (dept_id, name) VALUES "
            "(1, 'Alice'), (1, 'Bob'), (1, 'Carol'),"
            "(2, 'Dave'),"
            "(3, 'Eve'), (3, 'Frank');"
        ),
        "broken_query": (
            "SELECT departments.name, COUNT(staff.id) AS headcount "
            "FROM departments JOIN staff ON departments.id = staff.dept_id "
            "GROUP BY departments.name "
            "HAVING headcount >= 2;"
        ),
        "expected_row_count": 1,
    },
]

# ---------------------------------------------------------------------------
# Dangerous SQL patterns — block at environment level
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def _is_dangerous(query: str) -> bool:
    """Return True if the query contains any mutating / DDL keywords."""
    return bool(_DANGEROUS_PATTERN.search(query))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SQLRepairEnv:
    """
    OpenEnv-compliant environment for SQL Query Repair & Optimization.

    Episode flow:
        obs = env.reset(task_id)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
    """

    MAX_ATTEMPTS: int = 3

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {t["task_id"]: t for t in _TASKS}
        self.current_task: dict[str, Any] | None = None
        self.attempt: int = 0
        self._last_score: float = 0.0
        self._db: sqlite3.Connection | None = None
        self._last_feedback: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Start a fresh episode for the given task.

        Creates a new in-memory SQLite database, runs the schema DDL and seed
        data, and returns the first Observation with attempt=1 and empty feedback.

        Args:
            task_id: One of the task IDs from the built-in catalogue.

        Returns:
            Observation for the first step.

        Raises:
            ValueError: If task_id is not recognised.
        """
        if task_id not in self._tasks:
            available = ", ".join(sorted(self._tasks.keys()))
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available tasks: {available}"
            )

        self.current_task = self._tasks[task_id]
        self.attempt = 0
        self._last_score = 0.0
        self._last_feedback = ""

        # Open a brand-new in-memory database
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row

        # Execute schema + seed data
        try:
            self._db.executescript(self.current_task["schema_ddl"])
            self._db.executescript(self.current_task["seed_data"])
            self._db.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to initialise task database: {exc}") from exc

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Execute the agent's proposed fix and return (obs, reward, done, info).

        Args:
            action: Action containing the agent's fixed SQL query.

        Returns:
            obs:    Updated Observation (same task, incremented attempt, new feedback).
            reward: Reward with score, reason and done flag.
            done:   True if the episode has ended (correct query or max attempts reached).
            info:   Auxiliary diagnostic dict.
        """
        if self.current_task is None or self._db is None:
            raise RuntimeError("Environment has not been reset. Call reset(task_id) first.")

        self.attempt += 1
        query = action.fixed_query.strip()

        # ----------------------------------------------------------------
        # Safety gate: block mutating / DDL statements
        # ----------------------------------------------------------------
        if _is_dangerous(query):
            reward = Reward(
                score=0.0,
                reason=(
                    "Blocked: the query contains a forbidden SQL keyword "
                    "(DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, "
                    "REPLACE, ATTACH, DETACH, or PRAGMA)."
                ),
                done=self.attempt >= self.MAX_ATTEMPTS,
            )
            self._last_score = reward.score
            self._last_feedback = reward.reason
            obs = self._build_observation()
            return obs, reward, reward.done, self._info()

        # ----------------------------------------------------------------
        # Try executing the query
        # ----------------------------------------------------------------
        try:
            cursor = self._db.execute(query)
            rows = cursor.fetchall()
            row_count = len(rows)
        except sqlite3.Error as exc:
            error_msg = str(exc)
            feedback = f"SQL error: {error_msg}"
            reward = Reward(
                score=0.1,
                reason=feedback,
                done=self.attempt >= self.MAX_ATTEMPTS,
            )
            self._last_score = reward.score
            self._last_feedback = feedback
            obs = self._build_observation()
            return obs, reward, reward.done, self._info()

        # ----------------------------------------------------------------
        # Evaluate correctness
        # ----------------------------------------------------------------
        expected = self.current_task["expected_row_count"]
        correct = row_count == expected

        if not correct:
            feedback = f"Got {row_count} rows, expected {expected}"
            reward = Reward(
                score=0.4,
                reason=feedback,
                done=self.attempt >= self.MAX_ATTEMPTS,
            )
        else:
            # Correct answer — score depends on which attempt
            if self.attempt == 1:
                score, reason = 1.0, "Correct on first attempt — perfect score!"
            elif self.attempt == 2:
                score, reason = 0.85, "Correct on second attempt."
            else:
                score, reason = 0.7, "Correct, but required all three attempts."

            feedback = reason
            reward = Reward(score=score, reason=reason, done=True)

        self._last_score = reward.score
        self._last_feedback = feedback
        obs = self._build_observation()
        return obs, reward, reward.done, self._info()

    def state(self) -> dict[str, Any]:
        """Return a lightweight summary of the current environment state."""
        return self._info()

    def available_tasks(self) -> list[str]:
        """Return the list of all built-in task IDs."""
        return sorted(self._tasks.keys())

    def task_catalogue(self) -> list[dict[str, Any]]:
        """Return task metadata (without seed data) for all built-in tasks."""
        return [
            {
                "task_id": t["task_id"],
                "task_description": t["task_description"],
                "broken_query": t["broken_query"],
                "expected_row_count": t["expected_row_count"],
            }
            for t in _TASKS
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        # current_task is guaranteed non-None here: reset() sets it before
        # calling this method, and step() guards on it at the top.
        assert self.current_task is not None, "_build_observation called before reset()"
        task = self.current_task
        return Observation(
            task_id=task["task_id"],
            task_description=task["task_description"],
            schema_ddl=task["schema_ddl"],
            broken_query=task["broken_query"],
            expected_row_count=task["expected_row_count"],
            attempt=max(self.attempt, 1),  # always 1-indexed
            feedback=self._last_feedback,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "task_id": self.current_task["task_id"] if self.current_task else None,
            "attempt": self.attempt,
            "last_score": self._last_score,
        }