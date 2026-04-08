"""
server/app.py — Multi-mode deployment entry point for OpenEnv validate.

Required contract:
  - Exposes `app`  (the FastAPI application)
  - Exposes `main()` function (callable)
  - Runs uvicorn when executed as __main__
"""

import uvicorn
from env.server import app  # re-export the FastAPI app

__all__ = ["app", "main"]


def main() -> None:
    """Start the SQL Repair RL environment server on port 7860."""
    uvicorn.run(
        "env.server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()