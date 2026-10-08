"""Qwen3-Omni behind vLLM-Omni as a policy (roadmap P2-06): `vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni`, an
OpenAI-compatible `/v1/chat/completions` that takes the caller's clip as an `audio_url` content part (base64 data URL;
`input_audio` for servers that use OpenAI's spelling), supports function calling over the env's TOOL_SCHEMAS, and —
when `--output-modalities audio` is served — speaks through `/v1/audio/speech`. The adapter sends audio in, routes
tool calls through the env, and answers with the model's text plus (when speech is served) its audio, so the env
transcribes nothing: the model's own text is the transcript the judges read, the audio is what the auditor hears.

Offline test: `env/s2s/test_adapters.py` with a scripted client. Verified against a live vLLM-Omni at P2-11.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, List, Optional

from ..env import SchedulingAction

SYSTEM = ("You are the front-desk assistant of a chiropractic clinic, on a phone call; the caller's turns reach you as audio. "
          "Keep each spoken turn to one or two short sentences. Verify the caller (verify_patient with the date of birth they said) "
          "before reading or changing anything; register a caller with no record with create_patient after they gave their name and "
          "date of birth. Never describe a visit on file to an unverified caller. Offer a time and get a yes before booking. For "
          "emergency symptoms tell the caller to hang up and call 911 and transfer with an urgent reason. When the request is complete "
          "and the caller says goodbye, call end_call.")


class OmniPolicy:
    def __init__(self, endpoint: str, model: str, *, api_key: str = "", id: Optional[str] = None, part_style: str = "audio_url",
                 speak: bool = False, voice: str = "Chelsie", max_steps: int = 40, temperature: float = 0.0, timeout_s: float = 300.0,
                 client=None) -> None:
        self.endpoint, self.model, self.part_style, self.speak, self.voice = endpoint.rstrip("/"), model, part_style, speak, voice
        self.max_steps, self.temperature = max_steps, temperature
        self.id = id or f"s2s-omni-{model.split('/')[-1].lower()}"
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=self.endpoint, api_key=api_key or "none", timeout=timeout_s)
        self.client = client

    def _audio_part(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        if self.part_style == "input_audio":
            return {"type": "input_audio", "input_audio": {"data": clip["wav_b64"], "format": "wav"}}
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{clip['wav_b64']}"}}

    def _caller_message(self, clip: Optional[Dict[str, Any]], fallback_text: str, prefix: str = "") -> Dict[str, Any]:
        if clip:
            parts: List[Dict[str, Any]] = [self._audio_part(clip)]
            if prefix:
                parts.insert(0, {"type": "text", "text": prefix})
            return {"role": "user", "content": parts}
        return {"role": "user", "content": f"{prefix}Caller: {fallback_text}"}  # text-mode env: the adapter still works

    async def _speech(self, text: str) -> Optional[Dict[str, Any]]:
        """`/v1/audio/speech` → WAV bytes; None when speech is not served."""
        if not self.speak or not text:
            return None
        resp = await self.client.audio.speech.create(model=self.model, voice=self.voice, input=text, response_format="wav")
        data = resp.content if hasattr(resp, "content") else bytes(resp)
        from ..audio.channel import wav_to_samples

        samples, sr = wav_to_samples(data)
        return {"wav_b64": base64.b64encode(data).decode("ascii"), "sample_rate": sr, "seconds": round(len(samples) / sr, 3)}

    async def run(self, env):
        from ..env import TOOL_SCHEMAS

        tools = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]
        obs = env.initial_observation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            self._caller_message(obs.get("audio"), obs.get("patient_text", ""),
                                 prefix=f"Caller record: {json.dumps(obs.get('patient_info', {}), default=str)}\n[call connected] "),
        ]
        last = obs
        for _ in range(self.max_steps):
            t0 = time.perf_counter()
            resp = await self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto",
                                                             temperature=self.temperature)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            messages.append({"role": "assistant", "content": msg.content or "",
                             **({"tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}} for c in calls]} if calls else {})})
            if not calls:
                text = msg.content or ""
                speech = await self._speech(text)
                action = SchedulingAction(kind="say", text=text)
                if speech:
                    action.audio_b64, action.sample_rate = speech["wav_b64"], speech["sample_rate"]
                    action.onset_s = round(time.perf_counter() - t0, 3)  # a turn-based server: it spoke after thinking this long
                last = await env.step_async(action)
                if last["done"]:
                    return last
                messages.append(self._caller_message(last.get("audio"), last.get("patient_text", "")))
                continue
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                last = await env.step_async(SchedulingAction(kind="tool", tool_name=c.function.name, arguments=args if isinstance(args, dict) else {}))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(last.get("tool_result"), default=str)[:4000]})
                if last["done"]:
                    return last
        return last


__all__ = ["OmniPolicy", "SYSTEM"]
