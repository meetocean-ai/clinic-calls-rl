"""PersonaPlex (NVIDIA, Moshi architecture) as a policy (roadmap P2-06): a 7B full-duplex speech-to-speech model with
persona / voice conditioning and **no tool calling of its own**. The adapter adds the frontend–backend delegation the
roadmap describes: the duplex model is the voice on the line; when it emits the delegation token (a `delegate` event,
or `<delegate>` in its text), a text LLM with the env's TOOL_SCHEMAS takes the transcript so far, calls the env tools,
and its one-line answer is prefilled back into the duplex model's text stream, which speaks it.

    events = transport.turn(caller clip)
    if delegate in events: text = backend(transcript) [runs env tools]; events += transport.prefill(text)
    env.step_async(say(audio, text, onset))

Offline test: scripted transport + scripted backend. Real bindings (moshi server, vLLM for the backend) at P2-11.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..env import SchedulingAction
from .transport import DuplexTransport, concat_audio, onset_s, text_of

DELEGATE_TOKEN = "<delegate>"
PERSONA = ("You are the front-desk receptionist of a chiropractic clinic, warm and brief. You cannot look anything up yourself: "
           f"whenever the caller's request needs the schedule or their record (verifying a date of birth, finding or booking a time, "
           f"cancelling, registering, transferring), say {DELEGATE_TOKEN} and wait for the words to say.")
BACKEND_SYSTEM = ("You are the back office of a clinic receptionist on a live call. You get the transcript so far; use the tools to do "
                  "what the caller needs (verify the date of birth they said before reading or changing anything; never describe a "
                  "visit on file to an unverified caller; offer a time and get a yes before booking; emergencies → 911 and an urgent "
                  "transfer; end_call once the caller says goodbye). Reply with ONE short sentence for the receptionist to say next.")


class PersonaPlexPolicy:
    def __init__(self, transport: DuplexTransport, backend_client, backend_model: str, *, id: str = "s2s-personaplex-7b",
                 max_turns: int = 40, max_backend_steps: int = 8) -> None:
        self.transport, self.backend, self.backend_model = transport, backend_client, backend_model
        self.id, self.max_turns, self.max_backend_steps = id, max_turns, max_backend_steps

    @staticmethod
    def _delegated(events: List[Dict[str, Any]]) -> bool:
        return any(e.get("type") == "delegate" for e in events) or DELEGATE_TOKEN in text_of(events)

    async def _backend(self, env, transcript: List[Dict[str, str]]) -> str:
        """The text LLM runs the env tools and returns the line to speak."""
        from ..env import TOOL_SCHEMAS

        tools = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": BACKEND_SYSTEM},
                                          {"role": "user", "content": "Transcript so far:\n" + "\n".join(f"{t['role']}: {t['text']}" for t in transcript)
                                           + f"\n\nCaller record: {json.dumps(env.initial_observation.get('patient_info', {}), default=str)}"}]
        for _ in range(self.max_backend_steps):
            resp = await self.backend.chat.completions.create(model=self.backend_model, messages=messages, tools=tools, tool_choice="auto", temperature=0.0)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            if not calls:
                return (msg.content or "").strip()
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}} for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                res = await env.step_async(SchedulingAction(kind="tool", tool_name=c.function.name, arguments=args if isinstance(args, dict) else {}))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res.get("tool_result"), default=str)[:4000]})
                if res["done"]:
                    return ""
        return ""

    async def run(self, env):
        await self.transport.open(PERSONA)
        obs = env.initial_observation
        transcript: List[Dict[str, str]] = [{"role": "caller", "text": obs.get("patient_text", "")}]
        last = obs
        try:
            for _ in range(self.max_turns):
                clip = obs.get("audio")
                if not clip:
                    raise RuntimeError("PersonaPlexPolicy needs the env in audio mode (EhrSchedulingEnv(audio=True))")
                events = await self.transport.turn(clip["wav_b64"], clip["sample_rate"])
                if self._delegated(events):
                    line = await self._backend(env, transcript)
                    if env.done:
                        return {"done": True, "reward": env.result.reward if env.result else None}
                    events = [e for e in events if e.get("type") != "delegate"] + await self.transport.prefill(line)
                text = text_of(events).replace(DELEGATE_TOKEN, "").strip()
                audio = concat_audio(events)
                if audio is None and not text:
                    return last
                action = SchedulingAction(kind="say", text=text, onset_s=onset_s(events, float(clip.get("seconds") or 0.0)))
                if audio:
                    action.audio_b64, action.sample_rate = audio["wav_b64"], audio["sample_rate"]
                last = await env.step_async(action)
                transcript.append({"role": "agent", "text": text})
                if last["done"]:
                    return last
                transcript.append({"role": "caller", "text": last.get("patient_text", "")})
                obs = last
                if any(e.get("type") == "hangup" for e in events):
                    return last
            return last
        finally:
            await self.transport.close()


__all__ = ["BACKEND_SYSTEM", "DELEGATE_TOKEN", "PERSONA", "PersonaPlexPolicy"]
