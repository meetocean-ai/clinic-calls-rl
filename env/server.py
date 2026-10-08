"""OpenEnv server for the EHR scheduling environment (roadmap P1-03).

`EhrSchedulingEnv` behind the OpenEnv `reset / step / state` contract (`openenv-core`), one environment instance
per WebSocket session, so a trainer or any OpenEnv client drives episodes over HTTP/WS instead of in-process:

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.server            # 127.0.0.1:8011
    # any OpenEnv client:  reset(seed=5, family="booking", tier=3)  →  step({"kind": "tool", "tool_name": …})

Binds to localhost by default (ENV_HOST / ENV_PORT); it has no auth of its own — the trainer and the environment
share a box or a private network, and the local Medplum it writes to is reachable only there. Needs
`pip install -e tests/behavioral[env-server]` (openenv-core[core]); without it the module imports, the server does not.

The schemas and the reference client live in `env/client.py` (no agent dependencies — a trainer imports that alone)
and are re-exported here. `GET /schema` serves them. The reward is the environment's own `verify()` — the server
adds nothing.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from .client import OPENENV_AVAILABLE, EhrAction, EhrObservation, EhrState
from .env import TOOLS, EhrSchedulingEnv, SchedulingAction
from .tasks import FAMILIES, TIERS

if OPENENV_AVAILABLE:
    from openenv.core.env_server import Environment as _Environment

    from .client import EhrEnvClient as EhrEnvClient  # re-exported for callers that import the server module
else:  # pragma: no cover - exercised only where openenv-core is absent
    from abc import ABC

    class _Environment(ABC):  # type: ignore[no-redef]
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self, *a: Any, **k: Any) -> None:
            pass


ENV_NAME = "ehr_scheduling"


def _observation(obs: Dict[str, Any], *, task_id: str = "") -> EhrObservation:
    reward = obs.get("reward")
    return EhrObservation(
        done=bool(obs.get("done", False)), reward=reward,
        patient_text=obs.get("patient_text") or "", tool_result=obs.get("tool_result"), turn=int(obs.get("turn") or 0),
        task_id=obs.get("task_id") or task_id, tools=list(obs.get("tools") or []), patient_info=dict(obs.get("patient_info") or {}),
        stopped_reason=obs.get("stopped_reason") or "", diffs=list(obs.get("diffs") or []), error=obs.get("error") or "",
        audio=obs.get("audio"),
    )


class EhrOpenEnv(_Environment):
    """One instance per session. The Medplum adapter is opened on the first reset and closed with the session; every
    episode is its own Organization compartment, so sessions never see each other's data."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, medplum=None, *, scripted_caller: Optional[bool] = None, max_turns: int = 16, **kwargs: Any) -> None:
        if OPENENV_AVAILABLE:
            super().__init__(**kwargs)
        self._medplum = medplum
        self._owns_medplum = medplum is None
        # Default to the deterministic caller when no LLM key is configured, the LLM simulator otherwise.
        self.scripted_caller = scripted_caller if scripted_caller is not None else os.environ.get("EHR_ENV_SCRIPTED_CALLER", "0") == "1"
        # EHR_ENV_AUDIO=1 serves every caller line as telephone audio too (env/audio); EHR_ENV_TTS=tone|kokoro picks the voice backend
        audio = os.environ.get("EHR_ENV_AUDIO", "0") == "1"
        self.env = EhrSchedulingEnv(None, scripted_caller=self.scripted_caller, max_turns=max_turns, policy_id="openenv",
                                    audio=audio, tts_prefer=os.environ.get("EHR_ENV_TTS") or None)
        self._episode_id: Optional[str] = None
        self._steps = 0

    async def _ensure_medplum(self) -> None:
        if self._medplum is None:
            from adapters.medplum import MedplumAdapter

            self._medplum = await MedplumAdapter.from_env()
        self.env.medplum = self._medplum

    # ---- OpenEnv contract ------------------------------------------------------------

    async def reset_async(self, seed: Optional[int] = None, episode_id: Optional[str] = None, *, family: str = "booking",
                          tier: int = 1, chaos: Any = None, **kwargs: Any) -> EhrObservation:
        """`family`, `tier` and `chaos` (list or comma-separated string) ride on the reset request next to `seed`."""
        tier = int(tier or 1)
        if isinstance(chaos, str):
            chaos = [c for c in chaos.split(",") if c]
        chaos = list(chaos) if chaos else None
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        await self._ensure_medplum()
        obs = await self.env.reset(seed=int(seed if seed is not None else 1), family=family, tier=tier, chaos=chaos)
        self._episode_id = episode_id or f"{self.env.task.id}-{os.getpid()}-{id(self)}"
        self._steps = 0
        return _observation(obs)

    async def step_async(self, action: EhrAction, timeout_s: Optional[float] = None, **kwargs: Any) -> EhrObservation:
        if self.env.world is None:
            raise RuntimeError("call reset first")
        act = SchedulingAction(kind=action.kind, text=action.text, tool_name=action.tool_name, arguments=dict(action.arguments or {}))
        coro = self.env.step_async(act)
        obs = await (asyncio.wait_for(coro, timeout_s) if timeout_s else coro)
        self._steps += 1
        return _observation(obs, task_id=self.env.task.id)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> EhrObservation:
        return _run(self.reset_async(seed=seed, episode_id=episode_id, **kwargs))

    def step(self, action: EhrAction, timeout_s: Optional[float] = None, **kwargs: Any) -> EhrObservation:
        return _run(self.step_async(action, timeout_s=timeout_s, **kwargs))

    @property
    def state(self) -> EhrState:
        env = self.env
        if env.task is None:
            return EhrState(episode_id=self._episode_id, step_count=self._steps)
        from .versioning import dataset_version, method_version

        return EhrState(
            episode_id=self._episode_id, step_count=self._steps, task_id=env.task.id, seed=env.task.seed, family=env.task.family,
            tier=env.task.tier, split=env.task.split, chaos=list(env.task.chaos), turn=env.turn, tool_calls=env.tool_calls,
            verified=env.verified, done=env.done, stopped_reason=env.stopped_reason,
            reward=env.result.reward if env.result else None, diffs=list(env.result.diffs) if env.result else [],
            dataset_version=dataset_version(), method_version=method_version(audio=env.audio),
        )

    async def aclose(self) -> None:
        await self.env.close()
        if self._owns_medplum and self._medplum is not None:
            await self._medplum.close()
            self._medplum = None

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            loop.create_task(self.aclose())


def _run(coro):
    """Sync wrappers for callers without an event loop (the OpenEnv HTTP endpoints use the async methods)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("use the async methods inside a running event loop")


def build_app(max_concurrent_envs: Optional[int] = None):
    if not OPENENV_AVAILABLE:
        raise RuntimeError("openenv-core is not installed: pip install -e tests/behavioral[env-server]")
    from openenv.core.env_server import create_app

    return create_app(EhrOpenEnv, EhrAction, EhrObservation, env_name=ENV_NAME,
                      max_concurrent_envs=max_concurrent_envs or int(os.environ.get("MAX_CONCURRENT_ENVS", "16")))


def main() -> None:
    import uvicorn

    try:
        from dotenv import load_dotenv
        from pathlib import Path

        here = Path(__file__).resolve().parents[1]
        load_dotenv(here / ".env", override=False)
        load_dotenv(here.parents[1] / ".env", override=False)
    except ImportError:
        pass
    uvicorn.run(build_app(), host=os.environ.get("ENV_HOST", "127.0.0.1"), port=int(os.environ.get("ENV_PORT", "8011")))


if __name__ == "__main__":
    main()


__all__ = ["ENV_NAME", "EhrAction", "EhrObservation", "EhrOpenEnv", "EhrState", "OPENENV_AVAILABLE", "TOOLS", "build_app"]
if OPENENV_AVAILABLE:
    __all__.append("EhrEnvClient")
