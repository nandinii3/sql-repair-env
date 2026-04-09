"""
server/app.py — Multi-mode deployment entry point for OpenEnv validate.

Required contract (checked by openenv validate):
  - Exposes `app`       (the FastAPI ASGI application)
  - Exposes `main()`    (callable entry point function)
  - Has `if __name__ == '__main__': main()`

The sys.path fix ensures `env` package is importable whether this file is
run as `python server/app.py` (CWD = repo root) or from inside /app/server/.
"""

import os
import sys

# Ensure the repo root is on sys.path so `env` package is always importable
# regardless of which directory this file is executed from.
_HERE = os.path.dirname(os.path.abspath(__file__))        # .../server/
_ROOT = os.path.dirname(_HERE)                             # .../  (repo root)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from env.server import app  # re-export so `server.app:app` also works

__all__ = ["app", "main"]


def main() -> None:
    """Start the SQL Repair RL environment server on port 7860."""
    uvicorn.run(
        "env.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()