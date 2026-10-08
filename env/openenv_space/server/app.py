"""OpenEnv entry point for the hub manifest (`app: server.app:app`): the same FastAPI app `env/server.py` builds.

Runs inside the image from tests/behavioral/env/Dockerfile, where tests/behavioral is on PYTHONPATH and Medplum is
reachable; see openenv.yaml next to this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BEHAVIORAL = Path(__file__).resolve().parents[3]  # tests/behavioral
if str(_BEHAVIORAL) not in sys.path:
    sys.path.insert(0, str(_BEHAVIORAL))

from env.server import build_app  # noqa: E402

app = build_app()


def main() -> None:
    """`server` script entry (openenv.yaml / [project.scripts]): serve on ENV_HOST:ENV_PORT, localhost by default."""
    import os

    import uvicorn

    uvicorn.run(app, host=os.environ.get("ENV_HOST", "127.0.0.1"), port=int(os.environ.get("ENV_PORT", "8011")))


if __name__ == "__main__":
    main()
