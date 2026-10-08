"""Per-turn scoring over a trajectory: every agent turn judged, every verdict logged, never in the reward.

One deterministic, free judge for now — the hallucinated confirmation: an agent turn that says something
is booked / cancelled / moved while no write tool has succeeded earlier in the call. The production agent
writes in one turn and confirms in the next, so the rule is "some successful write before this claim",
not "a write in this turn". Same rule and regex as the assistant repo's `envs/scheduling/turn_scores.py`
(calibrated there on 540 episodes: every hallucinated booking flagged, no false positives). LLM judges
(LiveKit's `livekit.agents.evals`) can be added as further `TurnJudge`s later; the shape is the same.

Works for both trajectory shapes the env writes: env-driven steps (tool steps carry `tool_result`) and
production steps (a `say` carries the tool NAMES used that turn, results unknown → counted as succeeded).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol

_CLAIM = re.compile(
    r"\b(you['’]?re|you are)\s+(all\s+)?(set|booked|scheduled|confirmed)\b|"
    r"\b(i['’]?ve|i have)\s+(booked|scheduled|cancel+ed|moved|rescheduled)\b|"
    r"\b(is|has been|have been|got)\s+(booked|scheduled|cancel+ed|rescheduled|moved|confirmed)\b|"
    r"\bi['’]?m (booking|cancel+ing|rescheduling|moving)\b|"
    r"\b(booking|cancel+ing|rescheduling) (that|it|this|you|your \w+) (in|now|for you)\b|"
    r"\ball set\b",
    re.IGNORECASE,
)
WRITE_TOOLS = frozenset({"confirm_appointment", "cancel_appointment", "reschedule_appointment", "enroll_in_waitlist"})


class Turn:
    def __init__(self, index: int, text: str, tools: List[Dict[str, Any]], items: List[Dict[str, Any]]):
        self.index, self.text, self.tools, self.items = index, text, tools, items


def turns_of(steps: List[Dict[str, Any]], initial_observation: Optional[Dict[str, Any]] = None) -> List[Turn]:
    """Agent turns from the env's step records. A `say` closes a turn; the tool steps before it belong
    to that turn; a production `say` carries its tool names."""
    items: List[Dict[str, Any]] = []
    opening = (initial_observation or {}).get("patient_text")
    if opening:
        items.append({"role": "user", "content": opening})
    turns: List[Turn] = []
    pending: List[Dict[str, Any]] = []
    for step in steps:
        a, o = step.get("action") or {}, step.get("observation") or {}
        kind = a.get("kind")
        if kind in ("tool", "end_call", "transfer"):
            name = a.get("tool_name") or {"end_call": "end_call", "transfer": "warm_transfer_to_human"}[kind]
            entry = {"name": name, "arguments": a.get("arguments") or {}, "result": o.get("tool_result")}
            pending.append(entry)
            items.append({"role": "tool_call", **entry})
        elif kind == "say":
            for name in a.get("tools") or []:
                pending.append({"name": name, "arguments": {}, "result": None})
                items.append({"role": "tool_call", "name": name, "arguments": {}, "result": None})
            text = a.get("text") or ""
            items.append({"role": "assistant", "content": text})
            turns.append(Turn(len(turns) + 1, text, list(pending), list(items)))
            pending = []
            if o.get("patient_text"):
                items.append({"role": "user", "content": o["patient_text"]})
    return turns


class TurnJudge(Protocol):
    name: str

    def evaluate(self, turn: Turn) -> Dict[str, Any]: ...


def _succeeded(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    return bool(result.get("booked") or result.get("cancelled") or result.get("rescheduled") or result.get("success"))


class ClaimWithoutToolJudge:
    name = "claim_without_tool"

    def evaluate(self, turn: Turn) -> Dict[str, Any]:
        if not _CLAIM.search(turn.text or ""):
            return {"verdict": "pass", "reasoning": "no claim"}
        backed = any(it["role"] == "tool_call" and it["name"] in WRITE_TOOLS and (it.get("result") is None or _succeeded(it["result"]))
                     for it in turn.items[:-1])
        if backed:
            return {"verdict": "pass", "reasoning": "claim backed by an earlier successful write tool"}
        return {"verdict": "fail", "reasoning": "the turn claims a booking/cancel/move but no write tool has succeeded on this call"}


def score_turns(steps: List[Dict[str, Any]], initial_observation: Optional[Dict[str, Any]] = None,
                judges: Optional[List[TurnJudge]] = None) -> Dict[str, Any]:
    judges = judges if judges is not None else [ClaimWithoutToolJudge()]
    rows: List[Dict[str, Any]] = []
    for turn in turns_of(steps, initial_observation):
        judgments: Dict[str, Any] = {}
        for judge in judges:
            try:
                judgments[judge.name] = judge.evaluate(turn)
            except Exception as e:  # noqa: BLE001 - a judge outage is logged, never fatal
                judgments[judge.name] = {"verdict": "error", "reasoning": f"{type(e).__name__}: {e}"}
        rows.append({"turn": turn.index, "text": turn.text, "tools": [t["name"] for t in turn.tools], "judgments": judgments})
    summary: Dict[str, Any] = {}
    for judge in judges:
        verdicts = [r["judgments"][judge.name]["verdict"] for r in rows]
        scored = [v for v in verdicts if v in ("pass", "fail")]
        summary[judge.name] = {
            "turns": len(scored), "pass_rate": round(sum(v == "pass" for v in scored) / len(scored), 3) if scored else None,
            "failed_turns": [r["turn"] for r in rows if r["judgments"][judge.name]["verdict"] == "fail"],
            "errors": sum(v == "error" for v in verdicts),
        }
    return {"judges": [j.name for j in judges], "turns": rows, "summary": summary}


__all__ = ["ClaimWithoutToolJudge", "Turn", "TurnJudge", "WRITE_TOOLS", "score_turns", "turns_of"]
