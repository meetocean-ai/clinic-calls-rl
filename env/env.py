"""EhrSchedulingEnv (E5): reset / step / state over the local Medplum, with the production tool
names so a policy prompt transfers between this environment and the assistant repo's.

    env = EhrSchedulingEnv(medplum)
    obs = await env.reset(seed=5, family="booking")          # caller's first line
    obs = await env.step_async(SchedulingAction(kind="tool", tool_name="verify_patient", arguments={"dob": "..."}))
    obs = await env.step_async(SchedulingAction(kind="tool", tool_name="look_up_availability"))
    obs = await env.step_async(SchedulingAction(kind="tool", tool_name="confirm_appointment", arguments={"slot_index": 0}))
    obs = await env.step_async(SchedulingAction(kind="end_call"))  # obs["reward"] from verify()

Tools call the same `agents.tools_fhir` functions the production specialists call, tenant-scoped
to the episode's Organization. The production agents themselves run through `drive_inbound`
(policies.ProductionPolicy) and are scored by the same `verify()`.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from runtime.end_state import appointments_for

from .tasks import EhrTask, generate
from .verify import VerifyResult, verify
from .world import World, clinic_today, seed_world

TOOLS = ("verify_patient", "create_patient", "look_up_availability", "confirm_appointment", "get_appointments", "cancel_appointment",
         "reschedule_appointment", "enroll_in_waitlist", "warm_transfer_to_human", "end_call")
GATED_TOOLS = frozenset({"look_up_availability", "confirm_appointment", "get_appointments", "cancel_appointment",
                         "reschedule_appointment", "enroll_in_waitlist"})
# OpenAI-style function schemas for the same tools — what an OpenAI-compatible policy (policies.OpenAICompatiblePolicy)
# or a served trainer sees. One source of truth next to TOOLS; the descriptions carry the rules the gate enforces.
TOOL_SCHEMAS = [
    {"name": "verify_patient", "description": "Verify the caller against the record by the date of birth THEY SAID on this call "
                                              "(echoing the record verifies nobody). Required before any other tool.",
     "parameters": {"type": "object", "properties": {"dob": {"type": "string", "description": "YYYY-MM-DD as the caller said it"}},
                    "required": ["dob"]}},
    {"name": "create_patient", "description": "Register a caller who has no record, after they gave their full name and date of birth. "
                                              "Verifies them as the new patient.",
     "parameters": {"type": "object", "properties": {"first_name": {"type": "string"}, "last_name": {"type": "string"},
                                                     "dob": {"type": "string", "description": "YYYY-MM-DD"}},
                    "required": ["first_name", "last_name", "dob"]}},
    {"name": "look_up_availability", "description": "Open slots in the next two weeks (or from a given date) for one provider or any. "
                                                    "Returns slots with slot_index, start, provider, provider_id.",
     "parameters": {"type": "object", "properties": {"provider_id": {"type": "string", "description": "Practitioner id; omit for any provider"},
                                                     "date": {"type": "string", "description": "YYYY-MM-DD to start the search"}}}},
    {"name": "confirm_appointment", "description": "Book an offered slot after the caller agreed to it.",
     "parameters": {"type": "object", "properties": {"slot_index": {"type": "integer"},
                                                     "additional_visit": {"type": "boolean", "description": "true when the caller wants this on top of a visit they already have with the provider"}},
                    "required": ["slot_index"]}},
    {"name": "get_appointments", "description": "The caller's upcoming appointments: appointment_id, start, provider, provider_id, status.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "cancel_appointment", "description": "Cancel one of the caller's appointments.",
     "parameters": {"type": "object", "properties": {"appointment_id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["appointment_id"]}},
    {"name": "reschedule_appointment", "description": "Move one of the caller's appointments to an offered slot (after look_up_availability).",
     "parameters": {"type": "object", "properties": {"appointment_id": {"type": "string"}, "slot_index": {"type": "integer"}},
                    "required": ["appointment_id", "slot_index"]}},
    {"name": "enroll_in_waitlist", "description": "Put the caller on the waitlist for a provider who has nothing open.",
     "parameters": {"type": "object", "properties": {"provider_id": {"type": "string"}, "notes": {"type": "string"}}}},
    {"name": "warm_transfer_to_human", "description": "Hand the call to a staff member; ends the call. Say 'urgent' in the reason for a medical emergency.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}}},
    {"name": "end_call", "description": "End the call once the caller's request is complete and they said goodbye.",
     "parameters": {"type": "object", "properties": {}}},
]
assert [t["name"] for t in TOOL_SCHEMAS] == list(TOOLS)
MAX_DOB_ATTEMPTS = 3
SEARCH_WINDOW_DAYS = 14

_DOB_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y", "%m-%d-%Y", "%b %d, %Y", "%b %d %Y", "%d %B %Y")
# Speech-only caller behaviour (roadmap P2-02b), audio mode only
LATENCY_HELLO_S = 1.5  # the silence a caller tolerates before "hello?" (the same budget the audio judges use)
LATENCY_HELLO_LINES = ("Hello? Are you still there?", "Hello?? I said —", "Are you there? I'm going to hang up. I said —")
INTERRUPT_ON_TURN = 2  # the `interruptions` trait: the caller talks over the agent's second line
TALK_OVER_ONSET_S = -0.6  # … starting this long before the agent finished

# Spoken forms too ("April 11th, 1964" is what an ASR writes down): the ordinal suffix is dropped before parsing.
_DATE_IN_TEXT = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}|[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}|\d{1,2} [A-Za-z]{3,9} \d{4})\b")
_ORDINAL_SUFFIX = re.compile(r"(\d{1,2})(?:st|nd|rd|th)\b")


def _normalize_dob(raw: str) -> str:
    raw = _ORDINAL_SUFFIX.sub(r"\1", str(raw or "").strip().replace(".", ""))
    for fmt in _DOB_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def _dates_spoken(utterances: List[str]) -> set:
    found = set()
    for text in utterances:
        for m in _DATE_IN_TEXT.finditer(text or ""):
            found.add(_normalize_dob(m.group(1)))
    return found


@dataclass
class SchedulingAction:
    kind: str = "say"  # say | tool | end_call | transfer
    text: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    # A speech policy answers with audio (roadmap P2-05): 16-bit WAV (or anything libsndfile reads) as base64. The env
    # transcribes it for the judges and the safety gate when `text` is empty, keeps the clip in the episode record
    # (Opus) and measures turn-taking. `onset_s` = when the agent began relative to the END of the caller's line
    # (negative = cut the caller off); None for a turn-based policy, whose latency is the wall time the env measures.
    audio_b64: str = ""
    sample_rate: int = 0
    onset_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"kind": self.kind, "text": self.text, "tool_name": self.tool_name, "arguments": self.arguments}
        if self.audio_b64:
            d["has_audio"] = True  # the bytes live next to the record, not in it
            if self.onset_s is not None:
                d["onset_s"] = self.onset_s
        return d


class ScriptedCaller:
    """Deterministic caller for LLM-free runs: says yes to everything, follows the caller-side knobs."""

    def __init__(self, profile, task: EhrTask):
        self.profile, self.task = profile, task
        self._history: List[Dict[str, str]] = []
        self._dob_asks = 0
        self._switched = False

    async def first_utterance(self) -> str:
        return self.profile.opening_line

    async def reply_to(self, agent_message: str) -> str:
        self._history.append({"role": "user", "content": agent_message})
        low = agent_message.lower()
        if "birth" in low or "dob" in low:
            self._dob_asks += 1
            dob = self.profile.dob
            if "dob_corrected" in self.task.chaos and self.task.misstated_dob and self._dob_asks == 1:
                dob = self.task.misstated_dob
            text = f"My name is {self.profile.name}, and my date of birth is {dob}." if "name" in low else f"It's {dob}."
        elif "name" in low and self.task.family == "new_patient_intake":
            text = f"My name is {self.profile.name}."
        elif "mind_change" in self.task.chaos and not self._switched:
            self._switched = True
            appt = (self.profile.appointments_on_file or [{}])[0]
            text = f"Actually, I changed my mind. I need to cancel my appointment on {appt.get('date')} at {appt.get('time')} instead."
        elif len(self._history) < 8:
            text = "Yes, that works."
        else:
            text = "Thank you, goodbye."
        self._history.append({"role": "assistant", "content": text})
        return text


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 1)


def latency_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-turn wall-clock of the production policy's turns (LLM + tools), with the percentiles the
    article-style SLAs quote — P99, not the median. Text mode: no audio, so this is the model + tool
    budget, not time-to-first-audio."""
    turns = [float(s["action"]["turn_ms"]) for s in steps if (s.get("action") or {}).get("turn_ms")]
    ttft = [float(s["action"]["ttft_ms"]) for s in steps if (s.get("action") or {}).get("ttft_ms")]
    return {"turns": [round(t, 1) for t in turns], "p50": _pct(turns, 0.5), "p95": _pct(turns, 0.95), "p99": _pct(turns, 0.99),
            "max": _pct(turns, 1.0), "ttft_p50": _pct(ttft, 0.5), "ttft_p95": _pct(ttft, 0.95)}


def _asr_block(env) -> Optional[Dict[str, Any]]:
    """The manifest's `asr` block: what a cascaded voice policy's ear heard per caller line (env/audio/cascade.py);
    None for text policies. Reads the live world, or the extras the env kept after closing it."""
    world = getattr(env, "world", None)
    extra = world.extra if world is not None else (getattr(env, "last_world_extra", None) or {})
    rows = extra.get("asr") or []
    if not rows and not extra.get("asr_model"):
        return None
    from .audio.cascade import asr_summary

    return {"model": extra.get("asr_model"), "summary": asr_summary(rows), "lines": rows}


class EhrSchedulingEnv:
    def __init__(self, medplum, *, scripted_caller: bool = False, max_turns: int = 16, policy_id: str = "unknown",
                 audio: bool = False, tts_prefer: Optional[str] = None, ears=None, auditor=None):
        self.medplum = medplum
        self.scripted_caller = scripted_caller
        self.max_turns = max_turns
        self.policy_id = policy_id
        # Audio mode (roadmap P2-03): every caller line also arrives as telephone audio (env/audio/channel.py); the
        # text stays in the observation, the mode goes into method_version, the voice/line specs into the manifest.
        self.audio = audio
        self.tts_prefer = tts_prefer
        self.channel = None
        # Agent audio (roadmap P2-05): `ears` transcribes a speech policy's clips (env/audio/cascade.Transcriber by
        # default, built on first use); the clips themselves are kept for the record and the turn-taking metrics.
        self.ears = ears
        self.auditor = auditor  # P2-05b: an AudioAuditor scoring the agent's clips, logged in the manifest, never rewarded
        self.agent_clips: List[Dict[str, Any]] = []
        self._obs_returned_at: Optional[float] = None
        self.task: Optional[EhrTask] = None
        self.world: Optional[World] = None
        self.simulator = None
        self.verified = False
        self.acting_patient_id: Optional[str] = None  # shared_phone: which record the DOB picked
        self.dob_attempts = 0
        self.turn = 0
        self.tool_calls = 0
        self.done = False
        self.stopped_reason = ""
        self.result: Optional[VerifyResult] = None
        self.offered: List[Dict[str, Any]] = []
        self.caller_lines: List[str] = []
        self.steps: List[Dict[str, Any]] = []
        self.initial_observation: Dict[str, Any] = {}
        self.seconds: float = 0.0
        self.verify_error: str = ""

    # ---- lifecycle -------------------------------------------------------

    async def reset(self, *, seed: int, family: str = "booking", tier: int = 1, chaos: Optional[List[str]] = None,
                    task: Optional[EhrTask] = None) -> Dict[str, Any]:
        await self.close()
        self.task = task or generate(family, seed, tier=tier, today=clinic_today(), chaos=chaos)
        self.world = await seed_world(self.task, self.medplum)
        profile = self.world.patient.profile
        if self.scripted_caller:
            self.simulator = ScriptedCaller(profile, self.task)
        else:
            from runtime.patient_simulator import PatientSimulator

            self.simulator = PatientSimulator(profile=profile, tenant_id=self.world.tenant.organization_id)
        self.verified, self.dob_attempts, self.turn, self.tool_calls = False, 0, 0, 0
        self.acting_patient_id = None
        self.done, self.stopped_reason, self.result, self.offered = False, "", None, []
        self.steps = []
        self.agent_clips = []
        opening = await self.simulator.first_utterance()
        self.caller_lines = [opening]
        self.initial_observation = {
            "patient_text": opening, "task_id": self.task.id, "tools": list(TOOLS),
            "patient_info": self._patient_info(),
        }
        if self.audio:
            from .audio.channel import VoiceChannel

            self.channel = VoiceChannel.for_task(self.task.seed, self.task.tier, prefer=self.tts_prefer)
            self.initial_observation["audio"] = self.channel.render(opening)
            self._attach_pauses(self.initial_observation["audio"])
        self._obs_returned_at = time.perf_counter()
        return dict(self.initial_observation)

    @staticmethod
    def _attach_pauses(clip: Dict[str, Any]) -> None:
        """The pause map of a caller clip travels in the record, so turn-taking can be read without the audio."""
        from .audio.turntaking import clip_pauses_from_b64

        try:
            _, clip["pauses"] = clip_pauses_from_b64(clip["wav_b64"])
        except Exception:  # noqa: BLE001 — a clip that will not decode has no pause map
            clip["pauses"] = []

    def _caller_speech(self, text: str, agent_latency_s: Optional[float]) -> Tuple[str, Dict[str, Any]]:
        """What only a caller on the phone does (roadmap P2-02b), scripted and timestamped: after a long silence they say
        "hello?" and repeat themselves (and lose patience — cooperation drops one notch per time); with the `hesitant_dob`
        trait a date is voiced in pieces with pauses inside it; with the `interruptions` trait they talk over the agent
        once. Every event is recorded in `world.extra["caller_events"]` for the turn-taking metrics."""
        events = self.world.extra.setdefault("caller_events", [])
        speech: Dict[str, Any] = {}
        if agent_latency_s is not None and agent_latency_s > LATENCY_HELLO_S and text:
            n = sum(1 for e in events if e["kind"] == "latency_hello")
            text = (LATENCY_HELLO_LINES[min(n, len(LATENCY_HELLO_LINES) - 1)] + " " + text).strip()
            self.world.extra["cooperation"] = max(1, int(self.world.extra.get("cooperation", self.task.cooperation)) - 1)
            events.append({"turn": self.turn, "kind": "latency_hello", "agent_latency_s": round(agent_latency_s, 3),
                           "cooperation": self.world.extra["cooperation"]})
        if "hesitant_dob" in self.task.realism and _DATE_IN_TEXT.search(text) and not any(e["kind"] == "hesitant_dob" for e in events):
            speech["hesitate_dob"] = True
            events.append({"turn": self.turn, "kind": "hesitant_dob"})
        if "interruptions" in self.task.realism and self.turn == INTERRUPT_ON_TURN and not any(e["kind"] == "talk_over" for e in events):
            speech["onset_s"] = TALK_OVER_ONSET_S
            text = f"Actually, wait — {text}"
            events.append({"turn": self.turn, "kind": "talk_over", "caller_onset_s": TALK_OVER_ONSET_S})
        return text, speech

    def _hear_agent(self, action: SchedulingAction) -> None:
        """A speech policy's clip: transcribe it when no text came with it, keep it for the record, measure its onset."""
        import base64

        from .audio.turntaking import decode_audio

        raw = base64.b64decode(action.audio_b64)
        samples, sr = decode_audio(raw)
        if not action.text:
            if self.ears is None:
                from .audio.cascade import Transcriber

                self.ears = Transcriber()
            action.text = self.ears.transcribe_samples(samples, sr) if hasattr(self.ears, "transcribe_samples") else self.ears.hear(
                {"wav_b64": action.audio_b64, "spoken": ""})
        wall = None if self._obs_returned_at is None else round(time.perf_counter() - self._obs_returned_at, 3)
        clip = {"turn": self.turn, "seconds": round(len(samples) / sr, 3), "sample_rate": sr, "onset_s": action.onset_s,
                "wall_latency_s": wall, "heard": action.text, "samples": samples}
        self.agent_clips.append(clip)

    def _patient_info(self) -> Dict[str, Any]:
        """What the agent is handed at call start. With `shared_phone` it is the lookup's multi-match shape:
        every record on the phone, no patient_id, verified False — the DOB picks the record."""
        p = self.task.patient
        if self.task.family == "new_patient_intake":
            if not self.task.duplicate:  # unknown number: the lookup found nobody
                return {"unknown_caller": True, "phone_number": p["phone"], "patient_id": None, "verified": False}
            d = self.task.duplicate  # the one record on the number is somebody else — a first-match lookup hands it over
            return {"first_name": d["given"], "last_name": d["family"], "phone_number": p["phone"], "dob": d["dob"],
                    "patient_id": self.world.duplicate_patient_id, "verified": False}
        if not self.task.duplicate:
            return {"first_name": p["given"], "last_name": p["family"], "phone_number": p["phone"], "dob": p["dob"],
                    "patient_id": self.world.patient_id, "verified": False}
        d = self.task.duplicate
        return {
            "multiple_patients": True, "phone_number": p["phone"], "verified": False,
            "patients": [
                {"first_name": d["given"], "last_name": d["family"], "dob": d["dob"], "patient_id": self.world.duplicate_patient_id},
                {"first_name": p["given"], "last_name": p["family"], "dob": p["dob"], "patient_id": self.world.patient_id},
            ],
        }

    @property
    def patient_id(self) -> str:
        """The record the tools act on: the one the DOB picked on a shared phone, else the caller's."""
        return self.acting_patient_id or self.world.patient_id

    async def close(self) -> None:
        if self.world is not None:
            self.last_world_extra = dict(self.world.extra)  # post-mortems after the tenant is gone (agent_turns, asr log)
            await self.world.close()
            self.world = None

    # ---- step ------------------------------------------------------------

    async def step_async(self, action: SchedulingAction) -> Dict[str, Any]:
        if self.world is None:
            return {"error": "call reset() first", "done": False}
        if self.done:
            return {"done": True, "reward": self.result.reward if self.result else None, "error": "episode is over"}
        obs: Dict[str, Any] = {"patient_text": "", "tool_result": None, "turn": self.turn, "done": False, "reward": None,
                               "stopped_reason": "", "error": ""}
        try:
            if action.kind == "say":
                self.turn += 1
                if action.audio_b64:
                    self._hear_agent(action)
                # What the agent said, whether the caller was verified at that moment, and what the caller had said
                # so far — the safety gate (verify.safety_violations) reads this, nothing else in the reward does.
                self.world.extra.setdefault("agent_turns", []).append(
                    {"text": action.text or "", "verified": self.verified, "caller_so_far": list(self.caller_lines)})
                wall = None if self._obs_returned_at is None else time.perf_counter() - self._obs_returned_at
                text = await self.simulator.reply_to(action.text or "")
                if self.channel is not None:  # speech-only caller behaviour (roadmap P2-02b) decides the line and its audio
                    text, speech = self._caller_speech(text, wall)
                self.caller_lines.append(text)
                obs["patient_text"], obs["turn"] = text, self.turn
                if self.channel is not None and text:
                    obs["audio"] = self.channel.render(text, **speech)
                    self._attach_pauses(obs["audio"])
                    this_turn = [e for e in self.world.extra.get("caller_events") or [] if e["turn"] == self.turn]
                    if this_turn:
                        obs["audio"]["caller_events"] = this_turn
                if self.turn >= self.max_turns:
                    await self.finish("max_turns")
            elif action.kind in ("tool", "end_call", "transfer"):
                name = action.tool_name if action.kind == "tool" else {"end_call": "end_call", "transfer": "warm_transfer_to_human"}[action.kind]
                obs["tool_result"] = await self.call_tool(name, dict(action.arguments))
                self.tool_calls += 1
            else:
                obs["error"] = f"unknown action kind {action.kind!r}"
        except Exception as e:  # noqa: BLE001
            obs["error"] = f"{type(e).__name__}: {e}"
        if self.done:
            obs.update(done=True, reward=self.result.reward if self.result else None, stopped_reason=self.stopped_reason,
                       diffs=list(self.result.diffs) if self.result else [])
        record = action.to_dict()
        if action.kind == "say" and action.audio_b64 and self.agent_clips:
            c = self.agent_clips[-1]
            record["agent_audio"] = {k: c[k] for k in ("turn", "seconds", "sample_rate", "onset_s", "wall_latency_s", "heard")}
        self.trajectory_record(record, obs)
        self._obs_returned_at = time.perf_counter()
        return obs

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in TOOLS:
            return {"error": f"unknown tool {name!r}", "tools": list(TOOLS)}
        if name in GATED_TOOLS and not self.verified:
            return {"error": "The caller is not verified yet. Ask for their date of birth and call verify_patient first."}
        return await getattr(self, f"_tool_{name}")(**args)

    # ---- tools (the production tools_fhir functions, tenant-scoped) ---------

    async def _tool_verify_patient(self, dob: str = "", **_):
        self.dob_attempts += 1
        self.world.extra["verify_attempts"] = self.dob_attempts  # verify_fail_transfer: a transfer counts only after real tries
        given, on_file = _normalize_dob(dob), _normalize_dob(self.task.patient["dob"])
        if given not in _dates_spoken(self.caller_lines):
            return {"verified": False, "message": "The caller has not given that date of birth on this call."}
        if given and given == on_file and self.world.patient.patient_id:
            self.verified, self.acting_patient_id = True, self.world.patient_id
            return {"verified": True, "message": f"Verified {self.task.patient['given']}. Proceed with their request."}
        if given and self.task.duplicate and given == _normalize_dob(self.task.duplicate["dob"]):
            # The other person on the phone: a match, and every later write lands on their chart.
            self.verified, self.acting_patient_id = True, self.world.duplicate_patient_id
            return {"verified": True, "message": f"Verified {self.task.duplicate['given']}. Proceed with their request."}
        if not self.world.patient.patient_id and not self.world.extra.get("created_patient_id"):
            return {"verified": False, "message": "No patient record on this number matches that date of birth. If the caller "
                                                  "says they are new to the clinic, take their full name and date of birth and "
                                                  "call create_patient."}
        remaining = MAX_DOB_ATTEMPTS - self.dob_attempts
        if remaining <= 0:
            return {"verified": False, "message": "The date of birth does not match after several attempts. Offer to transfer the caller (warm_transfer_to_human)."}
        return {"verified": False, "message": f"That date of birth does not match our records. Ask the caller to repeat it ({remaining} attempt(s) left)."}

    async def _tool_create_patient(self, first_name: str = "", last_name: str = "", dob: str = "", **_):
        """Register a caller who has no record (the product's `register_patient_in_org`, as Reception's
        `register_new_patient` calls it). Name and DOB must have been said on the call; the phone is the caller ID."""
        from agents.tools_patient import register_patient_in_org

        if self.verified:
            return {"created": False, "reason": "already_verified", "message": "The caller is already verified against a record."}
        given = _normalize_dob(dob)
        if given not in _dates_spoken(self.caller_lines):
            return {"created": False, "reason": "dob_not_spoken", "message": "The caller has not given that date of birth on this call."}
        said = " ".join(self.caller_lines).lower()
        if not first_name.strip() or not last_name.strip() or first_name.strip().lower() not in said or last_name.strip().lower() not in said:
            return {"created": False, "reason": "name_not_spoken", "message": "Ask the caller for their full name before registering them."}
        res = await register_patient_in_org(first_name, last_name, given, phone=self.task.patient["phone"],
                                            organization_id=self.world.tenant.organization_id)
        if res.get("created"):
            self.verified, self.acting_patient_id = True, res["patient_id"]
            self.world.extra["created_patient_id"] = res["patient_id"]
            res["message"] = f"Registered {first_name.strip()} {last_name.strip()} and verified. Proceed with their request."
        return res

    async def _tool_look_up_availability(self, provider_id: Optional[str] = None, date: Optional[str] = None, **_):
        from agents.tools_fhir import search_available_slots

        start = date or (clinic_today() + timedelta(days=1)).isoformat()
        end = (datetime.fromisoformat(start).date() + timedelta(days=SEARCH_WINDOW_DAYS)).isoformat()
        slots = await search_available_slots(practitioner_id=provider_id or None, date_str=start, date_end_str=end,
                                             organization_id=self.world.tenant.organization_id, max_results=5)
        self.offered = list(slots)
        return {"found": bool(slots), "slots": [
            {"slot_index": i, "start": s.get("start"), "end": s.get("end"), "provider": s.get("practitioner_name"),
             "provider_id": s.get("practitioner_id")} for i, s in enumerate(slots)]}

    async def _slot_now_taken(self, slot: Dict[str, Any]) -> bool:
        """The chaos knobs `slot_taken` / `slot_taken_twice`: the offered slot goes to someone else between
        search and book — once, or on the first two writes."""
        limit = 2 if "slot_taken_twice" in self.task.chaos else (1 if "slot_taken" in self.task.chaos else 0)
        taken = int(self.world.extra.get("slots_taken", 0))
        if taken >= limit:
            return False
        self.world.extra["slots_taken"] = taken + 1
        from zoneinfo import ZoneInfo

        from .world import CLINIC_TZ, book_bystander

        tz = ZoneInfo(CLINIC_TZ)  # slot times are clinic-local and naive; FHIR wants an offset
        start = datetime.fromisoformat(slot["start"]).replace(tzinfo=tz)
        end = datetime.fromisoformat(slot["end"]).replace(tzinfo=tz)
        # Seeding may already have made bystander 1 (exact_time_unavailable); mid-episode ones number past it.
        n = int(self.world.extra.get("bystanders", 1 if self.world.bystander_patient_id else 0)) + 1
        self.world.extra["bystanders"] = n
        self.world.bystander_patient_id = await book_bystander(self.medplum, self.world.tenant, practitioner_id=slot["practitioner_id"],
                                                               start=start, end=end, n=n, practitioner_display=slot.get("practitioner_name") or "")
        # The bystander and their booking are the environment's doing, not the policy's mutations.
        self.world.baseline.n_patients += 1
        self.world.baseline.n_appointments += 1
        # The environment's own writes during the call, for a grader working from the before-snapshot (env/dataset)
        writes = self.world.extra.setdefault("env_writes", {"Appointment": 0, "Patient": 0})
        writes["Appointment"] += 1
        writes["Patient"] += 1
        return True

    async def _tool_confirm_appointment(self, slot_index: int = 0, additional_visit: bool = False, **_):
        from agents.tools_fhir import book_appointment

        if not (0 <= int(slot_index) < len(self.offered)):
            return {"booked": False, "message": "Search for availability first (look_up_availability)."}
        slot = self.offered[int(slot_index)]
        if await self._slot_now_taken(slot):
            return {"booked": False, "status": 409, "message": "That time was just taken. Search again and offer another."}
        if not additional_visit:
            existing = await appointments_for(self.medplum, self.world.tenant, self.patient_id, status="booked")
            same = [a for a in existing if any((p.get("actor") or {}).get("reference") == f"Practitioner/{slot['practitioner_id']}" for p in a.get("participant") or [])]
            if same:  # B-106: a second visit with the same provider is a reschedule unless the caller asked for both
                return {"booked": False, "message": "The patient already has an upcoming visit with this provider; use reschedule_appointment, or pass additional_visit=true if they want both.", "appointment_id": same[0]["id"]}
        appt = await book_appointment(patient_id=self.patient_id, practitioner_id=slot["practitioner_id"],
                                      start_time=slot["start"], end_time=slot["end"],
                                      patient_display=self.world.patient.profile.name, practitioner_display=slot.get("practitioner_name"),
                                      organization_id=self.world.tenant.organization_id)
        if not appt or not appt.get("id"):
            return {"booked": False, "message": "Booking failed."}
        return {"booked": True, "appointment_id": appt["id"], "start": appt.get("start")}

    async def _tool_enroll_in_waitlist(self, provider_id: Optional[str] = None, notes: str = "", **_):
        """No open slot for the provider the caller wants: a waitlist Task, not a booking elsewhere."""
        from agents.tools_fhir import add_to_waitlist

        task = await add_to_waitlist(patient_id=self.patient_id, preferred_practitioner_id=provider_id or self.world.provider_id,
                                     notes=notes, patient_display=self.world.patient.profile.name,
                                     organization_id=self.world.tenant.organization_id)
        return {"waitlisted": bool(task and task.get("id")), "task_id": (task or {}).get("id")}

    async def _tool_get_appointments(self, **_):
        from agents.tools_fhir import get_patient_appointments

        rows = await get_patient_appointments(self.patient_id, upcoming_only=True, organization_id=self.world.tenant.organization_id)
        return {"appointments": [{"appointment_id": r.get("id"), "start": r.get("start_iso") or r.get("start"), "provider": r.get("provider"),
                                  "provider_id": r.get("practitioner_id"), "status": r.get("status")} for r in rows]}

    async def _tool_cancel_appointment(self, appointment_id: str = "", reason: str = "", **_):
        from agents.tools_fhir import cancel_appointment

        res = await cancel_appointment(appointment_id, reason=reason or "patient request", organization_id=self.world.tenant.organization_id)
        return {"cancelled": bool(res and res.get("status") == "cancelled"), "appointment_id": appointment_id}

    async def _tool_reschedule_appointment(self, appointment_id: str = "", slot_index: int = 0, **_):
        from agents.tools_fhir import reschedule_appointment

        if not (0 <= int(slot_index) < len(self.offered)):
            return {"rescheduled": False, "message": "Search for availability first (look_up_availability)."}
        slot = self.offered[int(slot_index)]
        if await self._slot_now_taken(slot):
            return {"rescheduled": False, "status": 409, "message": "That time was just taken. Search again and offer another."}
        res = await reschedule_appointment(appointment_id, new_start=slot["start"], new_end=slot["end"],
                                           new_practitioner_id=slot.get("practitioner_id"),
                                           organization_id=self.world.tenant.organization_id)
        return {"rescheduled": bool(res and res.get("id")), "appointment_id": appointment_id}

    async def _tool_warm_transfer_to_human(self, reason: str = "", **_):
        from agents.tools_fhir import create_task

        await create_task(action_code="warm-transfer", action_display="Warm transfer to staff", patient_id=self.patient_id,
                          patient_display=self.world.patient.profile.name, description=reason or "caller transferred",
                          organization_id=self.world.tenant.organization_id)
        await self.finish("transfer")
        return {"transferred": True}

    async def _tool_end_call(self, **_):
        await self.finish("end_call")
        return {"ended": True}

    # ---- scoring and trajectories ---------------------------------------------

    async def finish(self, reason: str) -> VerifyResult:
        if self.done:
            return self.result
        self.done, self.stopped_reason = True, reason
        try:
            self.result = await verify(self.world)
        except Exception as e:  # noqa: BLE001 — the verdict could not be read: the episode is an error, never a silent None
            self.verify_error = f"verify failed: {type(e).__name__}: {e}"
            self.stopped_reason = "verify_error"
            raise
        return self.result

    def trajectory_record(self, action: Dict[str, Any], observation: Dict[str, Any]) -> None:
        self.steps.append({"i": len(self.steps), "action": action, "observation": observation})

    def manifest(self, error: str = "") -> Dict[str, Any]:
        """Scored, versioned, logged: the reward and diffs, the dataset / method stamps that produced them,
        and every agent turn's judge verdicts (`turn_scores`, never part of the reward)."""
        from .turn_scores import score_turns
        from .versioning import dataset_version, method_version

        task = self.task.to_dict()
        return {
            "schema_version": 1, "task_id": self.task.id,
            "task_hash": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
            "seed": self.task.seed, "family": self.task.family, "tier": self.task.tier, "split": self.task.split,
            "chaos": list(self.task.chaos), "provenance": list(self.task.provenance),
            "simulator": {"class": type(self.simulator).__name__}, "policy_id": self.policy_id,
            "environment": "ehr", "steps": len(self.steps), "stopped_reason": self.stopped_reason,
            "reward": self.result.reward if self.result else None, "diffs": list(self.result.diffs) if self.result else [],
            "turns": self.turn, "tool_calls": self.tool_calls, "error": error or getattr(self, "verify_error", ""),
            "steps_hash": hashlib.sha256(json.dumps(self.steps, sort_keys=True, default=str).encode()).hexdigest(),
            "dataset_version": dataset_version(), "method_version": method_version(audio=bool(getattr(self, "audio", False))),
            "turn_scores": score_turns(self.steps, self.initial_observation),
            "seconds": self.seconds, "latency_ms": latency_summary(self.steps),
            "safety_flags": [d for d in (self.result.diffs if self.result else []) if d.startswith("safety_gate:")],
            "voice": self.channel.spec() if getattr(self, "channel", None) is not None else None,
            # Cascaded voice policies (env/audio/cascade.py) log what their ASR heard per caller line; text policies: None.
            "asr": _asr_block(self),
            # Speech policies (roadmap P2-05): what the two audio streams say about turn-taking; None without agent audio.
            "turn_taking": _turn_taking_block(self),
            # Audio judges (P2-05b), logged not rewarded: deterministic latency / interruption / line facts + the auditor's rubric.
            "audio_scores": _audio_scores_block(self),
            # The caller's speech-only behaviour this episode (P2-02b): "hello?" after silences, hesitant date, talk-over.
            "caller_events": _caller_events_block(self),
        }

    def write_trajectory(self, root: Path | str, run_id: str, trial: int = 0, error: str = "") -> Dict[str, Path]:
        # the same (family, seed) is a different task at every tier, so a run that covers two tiers keeps them apart
        folder = Path(root) / run_id / f"{self.task.id}-t{self.task.tier}"
        folder.mkdir(parents=True, exist_ok=True)
        steps = folder / f"{trial}.jsonl"
        lines = [json.dumps({"i": -1, "observation": self.initial_observation}, sort_keys=True, default=str)]
        lines += [json.dumps(s, sort_keys=True, default=str) for s in self.steps]
        steps.write_text("\n".join(lines) + "\n")
        manifest = folder / f"{trial}.manifest.json"
        manifest.write_text(json.dumps(self.manifest(error), sort_keys=True, indent=2) + "\n")
        out = {"steps": steps, "manifest": manifest}
        if self.agent_clips:  # the agent's own voice, Opus, next to the record (pruned by prune_agent_audio)
            from .audio.turntaking import encode_opus

            for c in self.agent_clips:
                p = folder / f"{trial}.agent-{c['turn']:02d}.opus"
                p.write_bytes(encode_opus(c["samples"], c["sample_rate"]))
                out[f"agent-{c['turn']:02d}"] = p
        return out


def _turn_taking_block(env) -> Optional[Dict[str, Any]]:
    steps = getattr(env, "steps", None) or []
    if not any((s.get("action") or {}).get("agent_audio") for s in steps):
        return None
    from .audio.turntaking import turn_taking

    block = turn_taking(steps) or {}
    block["ears"] = getattr(getattr(env, "ears", None), "name", None)
    return block


def _caller_events_block(env) -> Optional[Dict[str, Any]]:
    world = getattr(env, "world", None)
    extra = world.extra if world is not None else (getattr(env, "last_world_extra", None) or {})
    events = extra.get("caller_events")
    if not events:
        return None
    return {"events": list(events), "latency_hellos": sum(1 for e in events if e["kind"] == "latency_hello"),
            "hesitant_dob": any(e["kind"] == "hesitant_dob" for e in events), "talk_overs": sum(1 for e in events if e["kind"] == "talk_over"),
            "cooperation_end": extra.get("cooperation")}


def _audio_scores_block(env) -> Optional[Dict[str, Any]]:
    tt = _turn_taking_block(env)
    if tt is None:
        return None
    from .audio.audio_judges import audio_scores

    voice = env.channel.spec() if getattr(env, "channel", None) is not None else None
    return audio_scores(tt, voice, getattr(env, "agent_clips", None) or [], auditor=getattr(env, "auditor", None))


def prune_agent_audio(root: Path | str, *, keep_runs: int = 5) -> List[Path]:
    """Delete the agent audio (`*.agent-NN.opus`) of every run under `root` except the newest `keep_runs` run folders
    (named like 20261008-001153) — the assistant's voice-report rule: raw audio for the recent runs, records forever."""
    root = Path(root)
    if not root.exists():
        return []
    runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    removed: List[Path] = []
    for run in runs[:-keep_runs] if keep_runs > 0 else runs:
        for p in run.rglob("*.agent-*.opus"):
            p.unlink()
            removed.append(p)
    return removed


__all__ = ["EhrSchedulingEnv", "GATED_TOOLS", "MAX_DOB_ATTEMPTS", "SchedulingAction", "ScriptedCaller", "TOOLS", "TOOL_SCHEMAS",
           "_dates_spoken", "prune_agent_audio"]
