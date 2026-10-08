"""Gemini Live as a closed reference policy (roadmap P2-07). Per-minute audio pricing: held-out subset, k=1, under the
roadmap's ledger cap, never without a ledger line.

google-genai Live API: `client.aio.live.connect(model=…, config={system_instruction, tools, response_modalities: ["AUDIO"]})`
gives a session; per caller line `session.send_realtime_input(audio=types.Blob(data=pcm16, mime_type="audio/pcm;rate=16000"))`
(+ `audio_stream_end=True` to close the turn), then `async for msg in session.receive()`: `msg.server_content.model_turn.parts[]`
carry `inline_data` (the agent's PCM audio at 24 kHz) and `text`; `msg.tool_call.function_calls[]` are answered with
`session.send_tool_response(function_responses=[types.FunctionResponse(id, name, response)])`; `msg.server_content.turn_complete`
ends the turn. The adapter talks to any object with those three methods; tests inject a scripted one.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..env import SchedulingAction
from .openai_realtime import INSTRUCTIONS

IN_RATE, OUT_RATE = 16000, 24000


def pcm16(samples: np.ndarray, sr: int, rate: int = IN_RATE) -> bytes:
    from ..audio.degrade import resample

    x = resample(np.asarray(samples, dtype=np.float32), sr, rate) if sr != rate else np.asarray(samples, dtype=np.float32)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def clip_from_pcm16(chunks: List[bytes], rate: int = OUT_RATE) -> Optional[Dict[str, Any]]:
    from ..audio.channel import wav_bytes

    if not chunks:
        return None
    x = np.frombuffer(b"".join(chunks), dtype="<i2").astype(np.float32) / 32767.0
    return {"wav_b64": base64.b64encode(wav_bytes(x, rate)).decode("ascii"), "sample_rate": rate, "seconds": round(len(x) / rate, 3)}


def gemini_tools() -> List[Dict[str, Any]]:
    from ..env import TOOL_SCHEMAS

    return [{"function_declarations": [{"name": s["name"], "description": s["description"], "parameters": s["parameters"]} for s in TOOL_SCHEMAS]}]


async def connect_live(model: str = "gemini-2.5-flash-native-audio-preview", api_key: Optional[str] = None):
    """The real session (an async context manager); the policy's `session_factory` returns an entered one."""
    from google import genai  # type: ignore[import-not-found]

    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return client.aio.live.connect(model=model, config={"system_instruction": INSTRUCTIONS, "tools": gemini_tools(), "response_modalities": ["AUDIO"]})


class GeminiLivePolicy:
    def __init__(self, session_factory, *, model: str = "gemini-2.5-flash-native-audio-preview", id: Optional[str] = None, max_turns: int = 40) -> None:
        self.session_factory, self.model, self.max_turns = session_factory, model, max_turns
        self.id = id or f"closed-gemini-{model.replace('/', '-')}"

    @staticmethod
    def _parts(msg) -> List[Any]:
        sc = getattr(msg, "server_content", None)
        turn = getattr(sc, "model_turn", None) if sc is not None else None
        return list(getattr(turn, "parts", None) or [])

    async def _turn(self, session, env) -> Optional[SchedulingAction]:
        audio: List[bytes] = []
        words: List[str] = []
        t0, first_audio_at = time.perf_counter(), None
        async for msg in session.receive():
            for p in self._parts(msg):
                data = getattr(getattr(p, "inline_data", None), "data", None)
                if data:
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter() - t0
                    audio.append(data)
                if getattr(p, "text", None):
                    words.append(p.text)
            tc = getattr(msg, "tool_call", None)
            if tc is not None and getattr(tc, "function_calls", None):
                responses = []
                for fc in tc.function_calls:
                    args = fc.args if isinstance(fc.args, dict) else {}
                    res = await env.step_async(SchedulingAction(kind="tool", tool_name=fc.name, arguments=args))
                    if res["done"]:
                        return None
                    responses.append({"id": fc.id, "name": fc.name, "response": {"result": json.loads(json.dumps(res.get("tool_result"), default=str))}})
                await session.send_tool_response(function_responses=responses)
            sc = getattr(msg, "server_content", None)
            if sc is not None and getattr(sc, "turn_complete", False):
                break
        if not audio and not words:
            return None
        action = SchedulingAction(kind="say", text="".join(words).strip(), onset_s=None if first_audio_at is None else round(first_audio_at, 3))
        clip = clip_from_pcm16(audio)
        if clip:
            action.audio_b64, action.sample_rate = clip["wav_b64"], clip["sample_rate"]
        return action

    async def run(self, env):
        from ..audio.channel import wav_to_samples

        session = await self.session_factory()
        try:
            obs = env.initial_observation
            await session.send_client_content(turns=[{"role": "user", "parts": [{"text": f"[call connected] Caller record: {json.dumps(obs.get('patient_info', {}), default=str)}"}]}], turn_complete=False)
            last = obs
            for _ in range(self.max_turns):
                clip = obs.get("audio")
                if not clip:
                    raise RuntimeError("GeminiLivePolicy needs the env in audio mode (--audio-env)")
                samples, sr = wav_to_samples(base64.b64decode(clip["wav_b64"]))
                await session.send_realtime_input(audio={"data": pcm16(samples, sr), "mime_type": f"audio/pcm;rate={IN_RATE}"})
                await session.send_realtime_input(audio_stream_end=True)
                action = await self._turn(session, env)
                if env.done:
                    return {"done": True, "reward": env.result.reward if env.result else None}
                if action is None:
                    continue
                last = await env.step_async(action)
                if last["done"]:
                    return last
                obs = last
            return last
        finally:
            close = getattr(session, "close", None) or getattr(session, "aclose", None)
            if close:
                await close()


__all__ = ["GeminiLivePolicy", "IN_RATE", "OUT_RATE", "clip_from_pcm16", "connect_live", "gemini_tools", "pcm16"]
