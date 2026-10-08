"""NemotronLabs VoiceChat as a policy (roadmap P2-06): an 11B full-duplex speech-to-speech model (NIM container) that
keeps talking while it calls tools — tool calls arrive on a separate `<TOOLCALL>` channel and operator-defined
"on-hold" lines fill the gap while the tool runs. Over the duplex event protocol (`transport.py`) one caller line is:

    events = transport.turn(caller clip)          audio / text / onhold / toolcall … end_turn | hangup
    for each toolcall: result = env tool → events += transport.tool_result(id, result)
    env.step_async(say(audio = all audio chunks, text = all text, onset = first chunk − caller seconds))

`end_call` / `warm_transfer_to_human` are tool calls like any other; `hangup` without them ends the episode as
`policy_returned`. Offline test with a scripted transport; the real NIM binding is P2-11.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..env import SchedulingAction
from .transport import DuplexTransport, concat_audio, onset_s, text_of

SYSTEM = ("Front-desk assistant of a chiropractic clinic on a phone call. Verify the caller's date of birth with verify_patient before "
          "reading or changing anything; never describe a visit on file to an unverified caller; offer a time and get a yes before "
          "booking; emergencies → tell the caller to call 911 and warm_transfer_to_human with an urgent reason; end_call when the "
          "caller says goodbye. On-hold line while a tool runs: 'One moment while I check that.'")


class VoiceChatPolicy:
    def __init__(self, transport: DuplexTransport, *, id: str = "s2s-voicechat-11b", max_turns: int = 40) -> None:
        self.transport, self.id, self.max_turns = transport, id, max_turns

    @staticmethod
    def _clip(obs: Dict[str, Any]) -> Dict[str, Any]:
        clip = obs.get("audio")
        if not clip:
            raise RuntimeError("VoiceChatPolicy needs the env in audio mode (EhrSchedulingEnv(audio=True))")
        return clip

    async def _speak(self, env, events: List[Dict[str, Any]], caller_seconds: float):
        """Everything the model said this turn (its on-hold lines included) as one `say` action."""
        clip = concat_audio(events)
        # the on-hold lines are spoken too: the transcript keeps them in order with the model's own words
        text = " ".join(e.get("text", "") for e in events if e.get("type") in ("text", "onhold")).strip()
        if clip is None and not text:
            return None
        action = SchedulingAction(kind="say", text=text, onset_s=onset_s(events, caller_seconds))
        if clip:
            action.audio_b64, action.sample_rate = clip["wav_b64"], clip["sample_rate"]
        return await env.step_async(action)

    async def run(self, env):
        await self.transport.open(SYSTEM)
        obs = env.initial_observation
        last = obs
        try:
            for _ in range(self.max_turns):
                clip = self._clip(obs)
                events = await self.transport.turn(clip["wav_b64"], clip["sample_rate"])
                spoken: List[Dict[str, Any]] = []
                hangup = False
                while events:
                    pending = [e for e in events if e.get("type") == "toolcall"]
                    spoken += [e for e in events if e.get("type") in ("audio", "text", "onhold")]
                    hangup = hangup or any(e.get("type") == "hangup" for e in events)
                    events = []
                    for call in pending:  # the model keeps talking (on-hold) while these run; results go straight back
                        args = call.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                args = {}
                        last = await env.step_async(SchedulingAction(kind="tool", tool_name=call.get("name", ""), arguments=args))
                        if last["done"]:
                            return last
                        events += await self.transport.tool_result(call.get("id", ""), last.get("tool_result"))
                said = await self._speak(env, spoken, float(clip.get("seconds") or 0.0))
                if said is not None:
                    last = said
                    if last["done"]:
                        return last
                    obs = last
                if hangup:
                    return last
            return last
        finally:
            await self.transport.close()


__all__ = ["SYSTEM", "VoiceChatPolicy"]
