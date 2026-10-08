"""Audio-side judges, logged not rewarded (roadmap P2-05b). Two layers over a speech policy's episode:

- a **deterministic** layer read off the recording (`env/audio/turntaking.py`): response latency against the product's
  budget, interruptions and talk-over, overlap, and the frames the phone line dropped (known from the degradation spec);
- a **multimodal auditor** — any OpenAI-compatible chat model that takes audio (Qwen3-Omni behind vLLM-Omni locally, or
  a hosted audio model on the capped held-out subset) — scoring the agent's own clips for tone / empathy,
  mispronunciation and artifacts on a 1–5 rubric, with a sentence of reasoning per clip.

Both land in the manifest as `audio_scores` next to `turn_scores`; the leaderboard's latency column prefers the
measured response latency when a policy spoke. Nothing here touches the reward.

    scores = audio_scores(steps, turn_taking_block, voice_spec, auditor=None)
"""
from __future__ import annotations

import base64
import json
import re
import statistics
from typing import Any, Dict, List, Optional, Protocol

LATENCY_BUDGET_S = 1.5  # the product's turn budget (what the caller tolerates before "hello?"); P2-02b uses the same number
RUBRIC = ("tone", "empathy", "mispronunciation", "artifacts")

AUDITOR_PROMPT = (
    "You are auditing one spoken turn of a clinic front-desk agent on a phone call. Rate the agent's AUDIO on a 1–5 scale "
    "for each of: tone (1 = cold or curt, 5 = warm and professional), empathy (1 = ignores the caller's state, 5 = acknowledges "
    "it naturally), mispronunciation (1 = names, dates or times garbled, 5 = every word clear), artifacts (1 = clicks, cut-offs, "
    "robotic glitches or wrong speed, 5 = clean speech). The transcript of the turn was: {transcript!r}. Reply with JSON only: "
    '{{"tone": n, "empathy": n, "mispronunciation": n, "artifacts": n, "reasoning": "one sentence"}}'
)


class AudioAuditor(Protocol):
    name: str

    def audit(self, wav_b64: str, sample_rate: int, transcript: str) -> Dict[str, Any]: ...


class OpenAIAudioAuditor:
    """An OpenAI-compatible chat model with audio input (vLLM-Omni's `audio_url` content part, or OpenAI's `input_audio`).
    One request per agent clip; the reply must be the JSON of the rubric. Failures are logged as `error`, never raised."""

    def __init__(self, endpoint: str, model: str, *, api_key: str = "", part_style: str = "audio_url", timeout_s: float = 120.0,
                 client=None) -> None:
        self.endpoint, self.model, self.part_style = endpoint.rstrip("/"), model, part_style
        self.name = f"audio-auditor-{model.replace('/', '-').replace(':', '-')}"
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=self.endpoint, api_key=api_key or "none", timeout=timeout_s)
        self.client = client

    def _audio_part(self, wav_b64: str) -> Dict[str, Any]:
        if self.part_style == "input_audio":
            return {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}}

    def audit(self, wav_b64: str, sample_rate: int, transcript: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": [self._audio_part(wav_b64), {"type": "text", "text": AUDITOR_PROMPT.format(transcript=transcript)}]}]
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.0)
            text = resp.choices[0].message.content or ""
            return parse_rubric(text)
        except Exception as e:  # noqa: BLE001 — a judge that fails is a logged error, not a failed episode
            return {"error": f"{type(e).__name__}: {e}"}


def parse_rubric(text: str) -> Dict[str, Any]:
    """The JSON object in a model reply (fenced or bare); scores clamped to 1–5; anything else is an error."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"error": f"no JSON in reply: {text[:120]!r}"}
    try:
        obj = json.loads(m.group(0))
    except ValueError as e:
        return {"error": f"bad JSON: {e}"}
    out: Dict[str, Any] = {}
    for k in RUBRIC:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            out[k] = int(min(5, max(1, round(v))))
    if len(out) != len(RUBRIC):
        return {"error": f"rubric incomplete: {sorted(obj)}"}
    out["reasoning"] = str(obj.get("reasoning") or "")[:300]
    return out


def deterministic_scores(turn_taking: Optional[Dict[str, Any]], voice_spec: Optional[Dict[str, Any]], *,
                         budget_s: float = LATENCY_BUDGET_S) -> Optional[Dict[str, Any]]:
    """What the recording says without a model: latency within the budget per turn, no interruptions, no talk-over,
    and the line's dropped frames (the degradation spec's frame-drop rate — the caller's audio, which the agent had to
    hear through)."""
    if not turn_taking:
        return None
    turns = turn_taking.get("turns") or []
    lat = [t["response_latency_s"] for t in turns if t.get("response_latency_s") is not None]
    within = [x <= budget_s for x in lat]
    deg = (voice_spec or {}).get("degradation") or {}
    return {
        "turns": len(turns),
        "latency_within_budget": {"budget_s": budget_s, "rate": round(sum(within) / len(within), 3) if within else None,
                                  "p50_s": turn_taking.get("response_latency_s", {}).get("p50"), "p95_s": turn_taking.get("response_latency_s", {}).get("p95")},
        "no_interruptions": {"rate": round(1 - turn_taking.get("interruptions", 0) / len(turns), 3) if turns else None,
                             "interruptions": turn_taking.get("interruptions", 0), "takeovers_in_pause": turn_taking.get("takeovers_in_pause", 0)},
        "talk_over": {"overlap_s": turn_taking.get("overlap_s", 0.0), "backchannels": turn_taking.get("backchannels", 0)},
        "line": {"frame_drop_rate": deg.get("frame_drop_p"), "snr_db": deg.get("snr_db")},  # DegradationSpec.to_dict keys
    }


def auditor_scores(agent_clips: List[Dict[str, Any]], auditor: Optional[AudioAuditor]) -> Optional[Dict[str, Any]]:
    """One rubric per agent clip, plus means; None without an auditor or clips."""
    if auditor is None or not agent_clips:
        return None
    from .channel import wav_bytes

    rows = []
    for c in agent_clips:
        wav_b64 = base64.b64encode(wav_bytes(c["samples"], c["sample_rate"])).decode("ascii")
        rows.append({"turn": c["turn"], **auditor.audit(wav_b64, c["sample_rate"], c.get("heard") or "")})
    means = {k: round(statistics.fmean(r[k] for r in rows if k in r), 2) for k in RUBRIC if any(k in r for r in rows)}
    return {"auditor": auditor.name, "turns": rows, "means": means, "errors": sum(1 for r in rows if "error" in r)}


def audio_scores(turn_taking: Optional[Dict[str, Any]], voice_spec: Optional[Dict[str, Any]], agent_clips: List[Dict[str, Any]],
                 auditor: Optional[AudioAuditor] = None) -> Optional[Dict[str, Any]]:
    det = deterministic_scores(turn_taking, voice_spec)
    if det is None:
        return None
    return {"deterministic": det, "auditor": auditor_scores(agent_clips, auditor), "rewarded": False}


__all__ = ["AUDITOR_PROMPT", "AudioAuditor", "LATENCY_BUDGET_S", "OpenAIAudioAuditor", "RUBRIC", "audio_scores", "auditor_scores",
           "deterministic_scores", "parse_rubric"]
