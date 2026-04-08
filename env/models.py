"""
Pydantic v2 models for the SQL Query Repair & Optimization RL Environment.
"""

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """State returned to the agent at each step."""

    task_id: str = Field(..., description="Unique identifier for the current task")
    task_description: str = Field(..., description="Human-readable description of the repair goal")
    schema_ddl: str = Field(..., description="CREATE TABLE statements for the in-memory DB")
    broken_query: str = Field(..., description="The SQL query the agent must fix")
    expected_row_count: int = Field(..., description="Number of rows the correct query should return")
    attempt: int = Field(..., ge=1, le=3, description="Current attempt number (1-indexed, max 3)")
    feedback: str = Field(
        default="",
        description="Result feedback from the last attempt; empty string on first step",
    )


class Action(BaseModel):
    """The agent's proposed SQL fix."""

    fixed_query: str = Field(..., min_length=1, description="The agent's proposed SQL fix")


class Reward(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(..., ge=0.0, le=1.0, description="Reward score between 0.0 and 1.0")
    reason: str = Field(..., description="Human-readable explanation of the score")
    done: bool = Field(..., description="Whether the episode has ended")