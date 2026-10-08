"""The cascaded voice policy (roadmap P2-04): a text policy that hears the caller through a local ASR instead of
reading the caller's text. The env already emits every caller line as telephone audio (audio mode, P2-03); here the
policy's ears are a local Whisper, so what the agent works from is the transcript — misheard dates, respelled names
and all — while the env keeps scoring the FHIR end state and the safety gate keeps reading what the caller really said.

    policy = CascadedVoicePolicy(OpenAICompatiblePolicy(...))      # id = "<inner id>+voice"
    env = EhrSchedulingEnv(medplum, audio=True)                     # the policy refuses a text-only env
    await run_episode(env, policy, ...)

The **voice gap** is pass@1 of the inner policy in text mode minus pass@1 of the same policy wrapped like this, on the
same tasks (`env.voice_gap`). Two policy shapes are wrapped:

- a policy that drives the env step by step (`OpenAICompatiblePolicy`, oracle, random): it gets a view of the env whose
  observations carry the transcript as `patient_text`; the clean line stays in `patient_text_clean`;
- the production agents (`ProductionPolicy`): the caller the agents talk to is the simulator wrapped in `VoicedCaller`,
  which speaks each line down the task's phone line and hands the agents the transcript.

Every heard line is logged in the manifest (`asr`: said / heard / similarity) so a lost episode can be traced to the ear.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from .roundtrip import load_asr, similarity, wav_samples_from_clip


class Transcriber:
    """One local ASR (mlx-whisper on Apple silicon, faster-whisper elsewhere) over the env's audio clips."""

    def __init__(self, asr=None) -> None:
        self.asr = asr or load_asr()
        self.name = getattr(self.asr, "name", type(self.asr).__name__)

    def hear(self, clip: Dict[str, Any]) -> str:
        samples, sr = wav_samples_from_clip(clip)
        return self.transcribe_samples(samples, sr)

    def transcribe_samples(self, samples, sample_rate: int) -> str:
        return self.asr.transcribe(samples, sample_rate).strip()


def asr_row(said: str, clip: Dict[str, Any], heard: str, turn: int) -> Dict[str, Any]:
    """What the manifest keeps per caller line: the clean text, the disfluent text that was voiced, the transcript and
    how close the transcript came to what was voiced."""
    return {"turn": turn, "said": said, "spoken": clip.get("spoken", said), "heard": heard,
            "similarity": round(similarity(clip.get("spoken", said), heard), 3), "seconds": clip.get("seconds")}


class VoicedCaller:
    """A simulator (first_utterance / reply_to) whose lines reach the agent as an ASR transcript of the task's phone
    audio. `said` holds the clean lines (what the caller really said — the env's safety gate and verify_patient read
    those), `rows` the per-line ASR log."""

    def __init__(self, simulator, channel, transcriber: Transcriber) -> None:
        self.simulator, self.channel, self.transcriber = simulator, channel, transcriber
        self.said: List[str] = []
        self.heard: List[str] = []
        self.rows: List[Dict[str, Any]] = []

    @property
    def tenant_id(self):  # run_conversation reads it off the simulator
        return getattr(self.simulator, "tenant_id", None)

    @property
    def profile(self):
        return getattr(self.simulator, "profile", None)

    def _hear(self, text: str) -> str:
        if not text:
            return text
        clip = self.channel.render(text)
        heard = self.transcriber.hear(clip)
        self.said.append(text)
        self.heard.append(heard)
        self.rows.append(asr_row(text, clip, heard, turn=len(self.rows)))
        return heard

    async def first_utterance(self) -> str:
        return self._hear(await self.simulator.first_utterance())

    async def reply_to(self, agent_message: str) -> str:
        return self._hear(await self.simulator.reply_to(agent_message))


class HeardEnv:
    """The env as the wrapped policy sees it: identical, except that `patient_text` in every observation is the ASR
    transcript of the clip the env attached, and the clean text moves to `patient_text_clean`. Tool calls, scoring
    and the trajectory stay the env's own; the heard lines are logged into `world.extra["asr"]`."""

    def __init__(self, env, transcriber: Transcriber) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_transcriber", transcriber)
        object.__setattr__(self, "_initial", None)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._env, name, value)

    def _heard(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        clip = obs.get("audio")
        if not clip or not obs.get("patient_text"):
            return obs
        out = dict(obs)
        heard = self._transcriber.hear(clip)
        out["patient_text_clean"], out["patient_text"] = obs["patient_text"], heard
        obs["patient_text_heard"] = heard  # the env's own record (trajectory / initial observation) keeps both texts
        self._env.world.extra.setdefault("asr", []).append(asr_row(obs["patient_text"], clip, heard, turn=obs.get("turn", 0)))
        return out

    @property
    def initial_observation(self) -> Dict[str, Any]:
        if self._initial is None or self._initial.get("task_id") != self._env.initial_observation.get("task_id"):
            object.__setattr__(self, "_initial", self._heard(self._env.initial_observation))
        return self._initial

    async def step_async(self, action):
        return self._heard(await self._env.step_async(action))


class CascadedVoicePolicy:
    """ASR in front of any text policy. The policy id gets `+voice`, so a text and a voice sweep of the same model are
    two columns, never one series; the env must be in audio mode or there is nothing to hear."""

    def __init__(self, inner, *, transcriber: Optional[Transcriber] = None) -> None:
        self.inner = inner
        self.transcriber = transcriber or Transcriber()
        self.id = f"{inner.id}+voice"

    async def run(self, env):
        if getattr(env, "channel", None) is None:
            raise RuntimeError(f"{self.id}: the env is not in audio mode (EhrSchedulingEnv(audio=True)); nothing to transcribe")
        env.world.extra["asr_model"] = self.transcriber.name
        if hasattr(self.inner, "run_with_caller"):  # the production agents: wrap the caller they talk to
            return await self.inner.run_with_caller(env, lambda sim: VoicedCaller(sim, env.channel, self.transcriber), log_to=env.world.extra)
        return await self.inner.run(HeardEnv(env, self.transcriber))


def asr_summary(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mean / min similarity over an episode's heard lines, for the manifest and the scorecard."""
    if not rows:
        return None
    sims = [r["similarity"] for r in rows]
    return {"lines": len(rows), "mean_similarity": round(sum(sims) / len(sims), 3), "min_similarity": round(min(sims), 3),
            "below_floor": sum(1 for s in sims if s < 0.85)}


def decode_clip(clip: Dict[str, Any]) -> bytes:
    return base64.b64decode(clip["wav_b64"])


__all__ = ["CascadedVoicePolicy", "HeardEnv", "Transcriber", "VoicedCaller", "asr_row", "asr_summary", "decode_clip"]
