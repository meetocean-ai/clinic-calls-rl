"""`ehr_scheduling` — the verifiers environment id for the Ocean EHR scheduling environment.

A shim: the implementation is `tests/behavioral/env/verifiers_env.py` (one copy, next to the environment it wraps).
This module makes it importable by name for `vf-eval` and prime-rl (`env.taskset.id = "ehr_scheduling"`), whether
installed with `pip install -e env/vf_package` or run with `tests/behavioral` on PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BEHAVIORAL = Path(__file__).resolve().parents[2]  # tests/behavioral
if str(_BEHAVIORAL) not in sys.path:
    sys.path.insert(0, str(_BEHAVIORAL))

from env.verifiers_env import FAMILIES, HELDOUT_SEEDS, EhrVerifiersEnv, load_environment, parse_action  # noqa: E402

__all__ = ["EhrVerifiersEnv", "FAMILIES", "HELDOUT_SEEDS", "load_environment", "parse_action"]
