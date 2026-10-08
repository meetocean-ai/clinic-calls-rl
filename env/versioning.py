"""Version stamps on every episode, so a number is never read without knowing what produced it.

    dataset_version  tasks-v<schema>+gen-<sha of tasks.py + world.py>   what the tasks and worlds were
    method_version   verify-<sha of verify.py>[+judge-<model>]           how they were scored

Same idea as the assistant repo's `envs/scheduling/versioning.py`; the hashes are of THIS repo's files.
A changed generator or verifier changes the stamp, and the scorecard shows the versions per row, so two
runs are only compared when both stamps agree.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent


def _sha(*names: str) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:10]


# Hashed ONCE, when the package is imported: a stamp describes the code that is running, so a file edited while a sweep
# is in flight does not relabel the episodes it did not score.
_GEN_SHA = _sha("tasks.py", "world.py")
_VERIFY_SHA = _sha("verify.py")


def dataset_version() -> str:
    from .tasks import EhrTask

    return f"tasks-v{EhrTask.schema_version}+gen-{_GEN_SHA}"


def method_version(judge_model: Optional[str] = None, audio: bool = False) -> str:
    """`+audio` marks episodes whose caller lines reached the policy as telephone audio (env/audio): a text-mode and an
    audio-mode run of the same tasks are never read as one series."""
    v = f"verify-{_VERIFY_SHA}"
    if judge_model:
        v += f"+judge-{judge_model}"
    if audio:
        v += "+audio"
    return v


def stamp(extra: Dict[str, Any], judge_model: Optional[str] = None) -> Dict[str, Any]:
    extra["dataset_version"] = dataset_version()
    extra["method_version"] = method_version(judge_model)
    return extra


__all__ = ["dataset_version", "method_version", "stamp"]
