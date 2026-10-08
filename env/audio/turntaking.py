"""Turn-taking measured from the two audio streams (roadmap P2-05, the FDB-v3 way): when a policy answers with audio,
the environment lines the caller's clip and the agent's clip up on one timeline and reads off response latency,
barge-ins (the agent started before the caller finished), whether a barge-in fell into a pause of the caller's line
(a *takeover*) or over their speech (an *interruption*), backchannels (short overlapping clips), overlap and talk
time. Deterministic signal work, no judge.

    summary = turn_taking(steps)        # steps = EhrSchedulingEnv.steps (audio mode, agent clips attached)

An agent clip arrives as `SchedulingAction(kind="say", audio_b64=…, sample_rate=…, onset_s=…)`: `onset_s` is when
the agent started speaking relative to the END of the caller's last line (negative = cut the caller off). A
turn-based policy that cannot say gets `None`, and its response latency is the wall time the env measured between
handing it the caller's line and receiving the reply.
"""
from __future__ import annotations

import base64
import io
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PAUSE_DBFS = -40.0  # frames quieter than this are silence
PAUSE_MIN_S = 0.3  # a silence this long inside the caller's line is a pause the agent could take over in
FRAME_S = 0.02
BACKCHANNEL_MAX_S = 1.0  # an overlapping agent clip this short is a backchannel, not a turn


def encode_opus(samples: np.ndarray, sample_rate: int) -> bytes:
    """Ogg/Opus bytes (libsndfile ≥ 1.1); the format the episode record keeps agent audio in."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0), sample_rate, format="OGG", subtype="OPUS")
    return buf.getvalue()


def decode_audio(data: bytes) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(data), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def pauses(samples: np.ndarray, sample_rate: int, *, min_s: float = PAUSE_MIN_S, dbfs: float = PAUSE_DBFS) -> List[Tuple[float, float]]:
    """(start_s, end_s) of silences inside the clip (leading / trailing silence excluded): where a listener could
    believe the speaker was done."""
    frame = max(1, int(sample_rate * FRAME_S))
    n = len(samples) // frame
    if n == 0:
        return []
    x = np.asarray(samples[: n * frame], dtype=np.float32).reshape(n, frame)
    rms = np.sqrt(np.mean(x * x, axis=1)) + 1e-9
    quiet = 20 * np.log10(rms) < dbfs
    out: List[Tuple[float, float]] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if start > 0 and (i - start) * FRAME_S >= min_s:  # leading silence is not a pause
                out.append((round(start * FRAME_S, 3), round(i * FRAME_S, 3)))
            start = None
    return out


def classify_onset(onset_s: Optional[float], caller_seconds: float, caller_pauses: List[Tuple[float, float]], agent_seconds: float) -> str:
    """How the agent took the floor: `after` (waited for the caller to finish), `takeover` (started inside a pause of
    the caller's line), `interruption` (started over the caller's speech), `backchannel` (a short overlapping clip)."""
    if onset_s is None or onset_s >= 0:
        return "after"
    at = caller_seconds + onset_s  # the instant on the caller's clip where the agent began
    if agent_seconds <= BACKCHANNEL_MAX_S:
        return "backchannel"
    if any(s <= at <= e for s, e in caller_pauses):
        return "takeover"
    return "interruption"


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))], 3)


def turn_taking(steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Over an episode's steps: every `say` with agent audio is paired with the caller line it answered (the previous
    caller clip in the record). None when no agent audio was recorded."""
    rows: List[Dict[str, Any]] = []
    caller_talk = 0.0
    caller_barge_ins = 0  # the caller starting before the agent finished (P2-02b talk-over), from the clip's own onset
    last_caller: Optional[Dict[str, Any]] = None
    for s in steps:
        a, o = s.get("action") or {}, s.get("observation") or {}
        if (o.get("audio") or {}).get("caller_onset_s") is not None and o["audio"]["caller_onset_s"] < 0:
            caller_barge_ins += 1
        agent = a.get("agent_audio")
        if a.get("kind") == "say" and agent:
            caller_seconds = float((last_caller or {}).get("seconds") or 0.0)
            caller_pauses = (last_caller or {}).get("pauses") or []
            onset = agent.get("onset_s")
            latency = onset if onset is not None else agent.get("wall_latency_s")
            kind = classify_onset(onset, caller_seconds, caller_pauses, float(agent.get("seconds") or 0.0))
            overlap = min(-onset, caller_seconds, float(agent.get("seconds") or 0.0)) if onset is not None and onset < 0 else 0.0
            rows.append({"turn": o.get("turn"), "response_latency_s": None if latency is None else round(float(latency), 3),
                         "onset_s": onset, "kind": kind, "overlap_s": round(overlap, 3), "agent_seconds": agent.get("seconds"),
                         "caller_seconds": caller_seconds, "caller_pauses": len(caller_pauses)})
        clip = o.get("audio")
        if clip and clip.get("seconds"):
            caller_talk += float(clip["seconds"])
            last_caller = clip
    if not rows:
        return None
    lat = [r["response_latency_s"] for r in rows if r["response_latency_s"] is not None]
    kinds = [r["kind"] for r in rows]
    agent_talk = sum(float(r["agent_seconds"] or 0.0) for r in rows)
    return {
        "agent_turns_with_audio": len(rows),
        "response_latency_s": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "max": _pct(lat, 1.0),
                               "mean": round(statistics.fmean(lat), 3) if lat else None},
        "barge_ins": sum(1 for k in kinds if k in ("takeover", "interruption", "backchannel")),
        "takeovers_in_pause": kinds.count("takeover"), "interruptions": kinds.count("interruption"),
        "backchannels": kinds.count("backchannel"), "waited": kinds.count("after"), "caller_barge_ins": caller_barge_ins,
        "overlap_s": round(sum(r["overlap_s"] for r in rows), 3),
        "agent_talk_s": round(agent_talk, 3), "caller_talk_s": round(caller_talk, 3),
        "talk_ratio": round(agent_talk / caller_talk, 3) if caller_talk else None,
        "turns": rows,
    }


def clip_pauses_from_b64(wav_b64: str) -> Tuple[float, List[Tuple[float, float]]]:
    """(seconds, pauses) of a caller clip in the observation, so the record can carry the pause map without the audio."""
    samples, sr = decode_audio(base64.b64decode(wav_b64))
    return round(len(samples) / sr, 3), pauses(samples, sr)


__all__ = ["BACKCHANNEL_MAX_S", "PAUSE_MIN_S", "classify_onset", "clip_pauses_from_b64", "decode_audio", "encode_opus", "pauses", "turn_taking"]
