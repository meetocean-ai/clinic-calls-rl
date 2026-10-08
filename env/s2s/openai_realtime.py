"""OpenAI Realtime (gpt-realtime) as a closed reference policy (roadmap P2-07). Per-minute audio pricing, so it runs
only on the held-out subset, k=1, under the cap written in the roadmap's ledger — and never without a ledger line.

GA wire protocol (post 2026-05-12): one WebSocket at `wss://api.openai.com/v1/realtime?model=<model>` with a bearer
header; the client sends `session.update` (instructions, tools, audio formats), then per caller line
`input_audio_buffer.append` (base64 PCM16) + `input_audio_buffer.commit` + `response.create`, and reads events until
`response.done`: `response.output_audio.delta` (the agent's speech; legacy `response.audio.delta` accepted),
`response.output_audio_transcript.delta` / `response.output_text.delta` (its words), `response.function_call_arguments.done`
(a tool call: run it, send `conversation.item.create` with the `function_call_output`, then `response.create` again).

The adapter talks to a `session` object with `send(event: dict)` and `recv() -> dict` (the `websockets` connection
wrapped by `connect_realtime`); tests inject a scripted one.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..env import SchedulingAction

INSTRUCTIONS = ("You are the front-desk assistant of a chiropractic clinic on a phone call. Keep each turn to one or two short "
                "sentences. Verify the caller (verify_patient with the date of birth they said) before reading or changing anything; "
                "register a caller with no record with create_patient after they gave their name and date of birth. Never describe a "
                "visit on file to an unverified caller. Offer a time and get a yes before booking. For emergency symptoms tell the caller "
                "to hang up and call 911 and transfer with an urgent reason. When the caller says goodbye, call end_call.")
PCM_RATE = 24000  # the Realtime API's PCM16 rate in both directions


def pcm16_b64(samples: np.ndarray, sr: int) -> str:
    from ..audio.degrade import resample

    x = resample(np.asarray(samples, dtype=np.float32), sr, PCM_RATE) if sr != PCM_RATE else np.asarray(samples, dtype=np.float32)
    return base64.b64encode((np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()).decode("ascii")


def wav_b64_from_pcm16(chunks: List[str]) -> Optional[Dict[str, Any]]:
    from ..audio.channel import wav_bytes

    if not chunks:
        return None
    raw = b"".join(base64.b64decode(c) for c in chunks)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    return {"wav_b64": base64.b64encode(wav_bytes(x, PCM_RATE)).decode("ascii"), "sample_rate": PCM_RATE, "seconds": round(len(x) / PCM_RATE, 3)}


class RealtimeSession:
    """`websockets` connection as send(dict) / recv() → dict."""

    def __init__(self, ws) -> None:
        self.ws = ws

    async def send(self, event: Dict[str, Any]) -> None:
        await self.ws.send(json.dumps(event))

    async def recv(self) -> Dict[str, Any]:
        return json.loads(await self.ws.recv())

    async def close(self) -> None:
        await self.ws.close()


async def connect_realtime(model: str, api_key: Optional[str] = None, url: str = "wss://api.openai.com/v1/realtime") -> RealtimeSession:
    import websockets  # type: ignore[import-not-found]

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    ws = await websockets.connect(f"{url}?model={model}", additional_headers={"Authorization": f"Bearer {key}"}, max_size=None)
    return RealtimeSession(ws)


class OpenAIRealtimePolicy:
    def __init__(self, session_factory, *, model: str = "gpt-realtime", id: Optional[str] = None, voice: str = "marin", max_turns: int = 40) -> None:
        self.session_factory, self.model, self.voice, self.max_turns = session_factory, model, voice, max_turns
        self.id = id or f"closed-openai-{model}"

    def _tools(self) -> List[Dict[str, Any]]:
        from ..env import TOOL_SCHEMAS

        return [{"type": "function", "name": s["name"], "description": s["description"], "parameters": s["parameters"]} for s in TOOL_SCHEMAS]

    async def _respond(self, session, env, caller_seconds: float):
        """One `response.create` → events until `response.done`; tool calls are run and answered inside."""
        await session.send({"type": "response.create"})
        audio_chunks: List[str] = []
        words: List[str] = []
        t0, first_audio_at = time.perf_counter(), None
        expect_more = False  # a tool result was sent with a new response.create: the model's answer is still to come
        while True:
            e = await session.recv()
            kind = e.get("type", "")
            if kind in ("response.output_audio.delta", "response.audio.delta"):
                if first_audio_at is None:
                    first_audio_at = time.perf_counter() - t0
                audio_chunks.append(e.get("delta", ""))
            elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta", "response.audio_transcript.delta", "response.text.delta"):
                words.append(e.get("delta", ""))
            elif kind == "response.function_call_arguments.done":
                try:
                    args = json.loads(e.get("arguments") or "{}")
                except ValueError:
                    args = {}
                res = await env.step_async(SchedulingAction(kind="tool", tool_name=e.get("name", ""), arguments=args if isinstance(args, dict) else {}))
                if res["done"]:
                    return res, None
                await session.send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": e.get("call_id"),
                                                                                "output": json.dumps(res.get("tool_result"), default=str)[:4000]}})
                await session.send({"type": "response.create"})
                expect_more = True
            elif kind == "response.done":
                if any(w for w in words) or audio_chunks:
                    clip = wav_b64_from_pcm16(audio_chunks)
                    action = SchedulingAction(kind="say", text="".join(words).strip(), onset_s=None if first_audio_at is None else round(first_audio_at, 3))
                    if clip:
                        action.audio_b64, action.sample_rate = clip["wav_b64"], clip["sample_rate"]
                    return None, action
                if expect_more:  # this `done` closed the tool-calling response; the spoken answer follows
                    expect_more = False
                    continue
                return None, None  # a response that only called tools and asked for nothing more
            elif kind == "error":
                raise RuntimeError(f"realtime error: {e.get('error')}")

    async def run(self, env):
        from ..audio.channel import wav_to_samples

        session = await self.session_factory()
        try:
            await session.send({"type": "session.update", "session": {
                "type": "realtime", "instructions": INSTRUCTIONS, "tools": self._tools(), "tool_choice": "auto",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": PCM_RATE}, "turn_detection": None},
                          "output": {"format": {"type": "audio/pcm", "rate": PCM_RATE}, "voice": self.voice}}}})
            obs = env.initial_observation
            await session.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": f"[call connected] Caller record: {json.dumps(obs.get('patient_info', {}), default=str)}"}]}})
            last = obs
            for _ in range(self.max_turns):
                clip = obs.get("audio")
                if not clip:
                    raise RuntimeError("OpenAIRealtimePolicy needs the env in audio mode (--audio-env)")
                samples, sr = wav_to_samples(base64.b64decode(clip["wav_b64"]))
                await session.send({"type": "input_audio_buffer.append", "audio": pcm16_b64(samples, sr)})
                await session.send({"type": "input_audio_buffer.commit"})
                done, action = await self._respond(session, env, float(clip.get("seconds") or 0.0))
                if done is not None:
                    return done
                if action is None:
                    continue  # tools only, nothing said: hand the model the caller's next line
                last = await env.step_async(action)
                if last["done"]:
                    return last
                obs = last
            return last
        finally:
            await session.close()


__all__ = ["INSTRUCTIONS", "OpenAIRealtimePolicy", "PCM_RATE", "RealtimeSession", "connect_realtime", "pcm16_b64", "wav_b64_from_pcm16"]
