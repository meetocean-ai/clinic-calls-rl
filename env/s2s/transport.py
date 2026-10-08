"""The event protocol the full-duplex adapters speak (roadmap P2-06).

A duplex model is driven one caller line at a time: the adapter sends the caller's clip, then reads events until the
model yields the turn. Events are plain dicts with `t` = seconds since the START of the caller's clip, so the agent's
onset relative to the end of the caller's line is `t_first_audio − caller_seconds` (negative = cut the caller off):

    {"type": "audio", "t": 1.9, "wav_b64": "...", "sample_rate": 24000}   a chunk of the agent's speech
    {"type": "text", "t": 1.9, "text": "Sure — "}                           the agent's words as it speaks them
    {"type": "toolcall", "t": 2.4, "name": "verify_patient", "arguments": {...}, "id": "c1"}
    {"type": "onhold", "t": 2.4, "text": "One moment while I check that."}   VoiceChat's filler while a tool runs
    {"type": "delegate", "t": 2.4}                                          PersonaPlex: hand the turn to the text LLM
    {"type": "end_turn", "t": 3.8}                                           the model yielded the floor
    {"type": "hangup", "t": 3.8}                                             the model ended the call

The adapters depend only on this shape; `WebSocketJsonTransport` is one binding (JSON frames over a WebSocket, the
shape a NIM / Moshi server speaks once its message names are mapped), used nowhere until the model is served.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np


class DuplexTransport(Protocol):
    """Send the caller's clip (plus the system prompt on the first turn), get the model's events for that turn."""

    name: str

    async def open(self, system_prompt: str) -> None: ...

    async def turn(self, wav_b64: str, sample_rate: int) -> List[Dict[str, Any]]: ...

    async def tool_result(self, call_id: str, result: Any) -> List[Dict[str, Any]]:
        """Feed a tool result back; returns the events the model produces in response (its spoken answer)."""
        ...

    async def prefill(self, text: str) -> List[Dict[str, Any]]:
        """PersonaPlex delegation: put text into the model's own text stream as if it had said it, get the speech back."""
        ...

    async def close(self) -> None: ...


def concat_audio(events: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """All `audio` events of a turn as one clip: {wav_b64, sample_rate, seconds, onset_t} (onset_t = `t` of the first chunk)."""
    from ..audio.channel import wav_bytes, wav_to_samples

    chunks, sr, onset = [], None, None
    for e in events:
        if e.get("type") != "audio" or not e.get("wav_b64"):
            continue
        samples, rate = wav_to_samples(base64.b64decode(e["wav_b64"]))
        if sr is None:
            sr, onset = rate, e.get("t")
        elif rate != sr:
            from ..audio.degrade import resample

            samples = resample(samples, rate, sr)
        chunks.append(np.asarray(samples, dtype=np.float32))
    if not chunks:
        return None
    x = np.concatenate(chunks)
    return {"wav_b64": base64.b64encode(wav_bytes(x, sr)).decode("ascii"), "sample_rate": sr, "seconds": round(len(x) / sr, 3), "onset_t": onset}


def text_of(events: Iterable[Dict[str, Any]]) -> str:
    return "".join(e.get("text", "") for e in events if e.get("type") == "text").strip()


def onset_s(events: Iterable[Dict[str, Any]], caller_seconds: float) -> Optional[float]:
    """When the agent began relative to the END of the caller's line (negative = barge-in); None if it never spoke."""
    for e in events:
        if e.get("type") == "audio" and e.get("t") is not None:
            return round(float(e["t"]) - caller_seconds, 3)
    return None


class WebSocketJsonTransport:
    """JSON frames over one WebSocket (needs `websockets`). Message names are a mapping (`frames`) because every server
    spells them differently; the default is the event protocol above sent as-is. Unused until P2-11 binds it to a server."""

    name = "websocket-json"

    def __init__(self, url: str, *, frames: Optional[Dict[str, str]] = None) -> None:
        self.url, self.frames, self._ws = url, {"open": "open", "turn": "turn", "tool_result": "tool_result", "prefill": "prefill", **(frames or {})}, None

    async def _connect(self):
        if self._ws is None:
            import websockets  # type: ignore[import-not-found]

            self._ws = await websockets.connect(self.url, max_size=None)
        return self._ws

    async def _roundtrip(self, frame: Dict[str, Any]) -> List[Dict[str, Any]]:
        ws = await self._connect()
        await ws.send(json.dumps(frame))
        events: List[Dict[str, Any]] = []
        while True:
            e = json.loads(await ws.recv())
            events.append(e)
            if e.get("type") in ("end_turn", "hangup"):
                return events

    async def open(self, system_prompt: str) -> None:
        ws = await self._connect()
        await ws.send(json.dumps({"type": self.frames["open"], "system_prompt": system_prompt}))

    async def turn(self, wav_b64: str, sample_rate: int) -> List[Dict[str, Any]]:
        return await self._roundtrip({"type": self.frames["turn"], "wav_b64": wav_b64, "sample_rate": sample_rate})

    async def tool_result(self, call_id: str, result: Any) -> List[Dict[str, Any]]:
        return await self._roundtrip({"type": self.frames["tool_result"], "id": call_id, "result": result})

    async def prefill(self, text: str) -> List[Dict[str, Any]]:
        return await self._roundtrip({"type": self.frames["prefill"], "text": text})

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


__all__ = ["DuplexTransport", "WebSocketJsonTransport", "concat_audio", "onset_s", "text_of"]
