"""Client side of the OpenEnv server (roadmap P1-03/P1-09): the pydantic schemas and the reference client, with no
dependency on the agent code or the FHIR adapters — a trainer process imports this module alone.

    from env.client import EhrAction, EhrEnvClient
    async with EhrEnvClient(base_url="ws://127.0.0.1:8011") as env:
        first = await env.reset(seed=5, family="booking", tier=3)
        result = await env.step(EhrAction(kind="say", text="Could I have your date of birth, please?"))

`env/server.py` re-exports these next to the server itself. Needs `openenv-core[core]`; without it only the schemas
are defined (as plain pydantic models) and `EhrEnvClient` is absent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Action as _Action
    from openenv.core.env_server import Observation as _Observation
    from openenv.core.env_server import State as _State

    OPENENV_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where openenv-core is absent
    OPENENV_AVAILABLE = False
    from pydantic import BaseModel

    class _Action(BaseModel):  # type: ignore[no-redef]
        metadata: Dict[str, Any] = {}

    class _Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = {}

    class _State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0


class EhrAction(_Action):
    """`say` (text) · `tool` (tool_name + arguments) · `end_call` · `transfer` — the same four kinds as in-process."""

    kind: str = "say"
    text: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = {}


class EhrObservation(_Observation):
    patient_text: str = ""
    tool_result: Any = None
    turn: int = 0
    task_id: str = ""
    tools: List[str] = []
    patient_info: Dict[str, Any] = {}
    stopped_reason: str = ""
    diffs: List[str] = []
    error: str = ""
    # audio mode: the caller's line as telephone audio {wav_b64, sample_rate, seconds, spoken, voice, tts}
    audio: Optional[Dict[str, Any]] = None


class EhrState(_State):
    task_id: str = ""
    seed: int = 0
    family: str = ""
    tier: int = 0
    split: str = ""
    chaos: List[str] = []
    turn: int = 0
    tool_calls: int = 0
    verified: bool = False
    done: bool = False
    stopped_reason: str = ""
    reward: Optional[int] = None
    diffs: List[str] = []
    dataset_version: str = ""
    method_version: str = ""


def flatten_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The server may serialize `{"observation": {...}, "reward": r, "done": d}`; read either shape as one dict."""
    flat = dict(payload.get("observation") or payload) if isinstance(payload.get("observation"), dict) else dict(payload)
    flat.setdefault("done", payload.get("done", False))
    flat.setdefault("reward", payload.get("reward"))
    return flat


if OPENENV_AVAILABLE:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    class EhrEnvClient(EnvClient):
        """The OpenEnv reference client bound to this environment's schemas — what a trainer on another box uses."""

        def _step_payload(self, action: EhrAction) -> Dict[str, Any]:
            return action.model_dump()

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
            flat = flatten_observation(payload)
            obs = EhrObservation(**{k: v for k, v in flat.items() if k in EhrObservation.model_fields})
            return StepResult(observation=obs, reward=obs.reward, done=obs.done)

        def _parse_state(self, payload: Dict[str, Any]) -> EhrState:
            return EhrState(**{k: v for k, v in payload.items() if k in EhrState.model_fields})

    __all__ = ["EhrAction", "EhrEnvClient", "EhrObservation", "EhrState", "OPENENV_AVAILABLE", "flatten_observation"]
else:
    __all__ = ["EhrAction", "EhrObservation", "EhrState", "OPENENV_AVAILABLE", "flatten_observation"]
