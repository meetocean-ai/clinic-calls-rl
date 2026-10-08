"""ASR round trip (roadmap P2-01): no caller line may be unintelligible by construction. Each synthesized (and,
later, degraded) line is transcribed by a local Whisper and compared with the text it came from; a line ships only
when the normalized similarity is ≥ 0.85.

    report = roundtrip_check(lines, tts=load_tts(), asr=load_asr(), degrade_spec=None)
    report["pass_rate"], report["failures"]

Normalization makes the comparison about words, not formatting: lower case, punctuation dropped, digits kept
(dates and times are what the environment cares about), whitespace collapsed.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

SIMILARITY_FLOOR = 0.85


def normalize(text: str) -> str:
    text = text.lower().replace("—", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio over normalized text — 1.0 identical, 0.0 nothing in common."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def levenshtein(a: str, b: str) -> int:
    a, b = normalize(a), normalize(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class FasterWhisperASR:
    """faster-whisper (CTranslate2) on the CPU; `base.en` is enough for a round-trip check and downloads once."""

    name = "faster-whisper"

    def __init__(self, model_size: str = "base.en", compute_type: str = "int8") -> None:
        self._size, self._compute, self._model = model_size, compute_type, None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._size, device="cpu", compute_type=self._compute)
        return self._model

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        from env.audio.degrade import resample, to_mono_float

        x = resample(to_mono_float(samples), sample_rate, 16000)
        segments, _info = self._load().transcribe(x, beam_size=1, language="en", vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


class MlxWhisperASR:
    """mlx-whisper (Apple-silicon native) — the ASR on the Mac, where faster-whisper's tokenizers pin does not install
    on the agent's Python. `base.en` weights download once from the Hub."""

    name = "mlx-whisper"

    def __init__(self, repo: str = "mlx-community/whisper-base.en-mlx") -> None:
        self._repo = repo

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        import mlx_whisper

        from env.audio.degrade import resample, to_mono_float

        x = resample(to_mono_float(samples), sample_rate, 16000)
        out = mlx_whisper.transcribe(x, path_or_hf_repo=self._repo, language="en", fp16=False)
        return str(out.get("text", "")).strip()


def wav_samples_from_clip(clip: Dict[str, Any]):
    """Decode a `VoiceChannel.render` clip back to (samples, sample_rate)."""
    import base64

    from env.audio.channel import wav_to_samples

    return wav_to_samples(base64.b64decode(clip["wav_b64"]))


def asr_available() -> bool:
    for mod in ("mlx_whisper", "faster_whisper"):
        try:
            __import__(mod)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def load_asr():
    """Whichever local Whisper is installed: mlx-whisper on Apple silicon, faster-whisper elsewhere."""
    try:
        import mlx_whisper  # noqa: F401

        return MlxWhisperASR()
    except Exception:  # noqa: BLE001
        return FasterWhisperASR()


_ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b")
_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\.?(?!\w)", re.IGNORECASE)


def canon(text: str) -> str:
    """Normalization for comparing facts across the TTS → ASR trip: ISO dates spoken out, ordinals dropped
    ("15th" → "15"), "a.m." / "a m" → "am", ":00" dropped, then the word-level normalization."""
    from env.audio.disfluencies import verbalize

    t = verbalize(text)
    t = _ORDINAL.sub(r"\1", t)
    t = _AMPM.sub(lambda m: f"{int(m.group(1))}{':' + m.group(2) if m.group(2) and m.group(2) != '00' else ''} {m.group(3).lower()}m", t)
    t = re.sub(r"\b(\d{1,2}):00\b", r"\1", t)
    return normalize(t)


def _name_match(surname: str, heard_token: str) -> bool:
    """A surname counts as heard when the transcript has a token that sounds like it: Whisper spells unfamiliar names
    phonetically (Facchinato → Fackinato, Fakinato), which is the right doctor said right. SequenceMatcher ≥ 0.75 on
    the letters keeps Campos / Campbell (0.57) apart while accepting those respellings (≥ 0.8)."""
    if not heard_token or heard_token[0] != surname[0]:
        return False
    return levenshtein(surname, heard_token) <= 1 or SequenceMatcher(None, surname, heard_token).ratio() >= 0.75


def facts_heard(line: str, heard: str) -> bool:
    """Every protected token of the clean line (dates, times, DOBs, phone digits, `Dr. Name`) appears in the
    transcript after canonicalization — the test that matters for a scheduling call: fillers may be garbled, the
    facts may not. A name is accepted within one edit (Eddins / Eddens); a date without its year is accepted too."""
    from env.audio.disfluencies import _PROTECTED

    heard_c = canon(heard)
    heard_tokens = heard_c.split()
    for m in _PROTECTED.finditer(line):
        token = canon(m.group(0))
        if token in heard_c:
            continue
        if m.group(0).startswith("Dr.") and any(_name_match(token.split()[-1], h) for h in heard_tokens):
            continue
        parts = token.split()
        if len(parts) == 3 and parts[2].isdigit() and " ".join(parts[:2]) in heard_c:  # "april 11 1964" heard without the year
            continue
        return False
    return True


def roundtrip_check(lines: Iterable[Tuple[str, str]], *, tts, asr, degrade_spec=None, floor: float = SIMILARITY_FLOOR,
                    spoken_lines: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """`lines` = (text, voice) pairs; `spoken_lines` = what was actually voiced when disfluencies were injected (defaults
    to the text). Synthesize → (degrade) → transcribe → compare the transcript with the spoken line, and check that
    the clean line's facts were heard. Returns the per-line table."""
    from env.audio.degrade import degrade

    rows: List[Dict[str, Any]] = []
    spoken_list = list(spoken_lines) if spoken_lines is not None else None
    for i, (text, voice) in enumerate(lines):
        spoken = spoken_list[i] if spoken_list is not None else text
        samples, sr = tts.synthesize(spoken, voice=voice)
        if degrade_spec is not None:
            samples, sr = degrade(samples, sr, degrade_spec)
        heard = asr.transcribe(samples, sr)
        score = similarity(spoken, heard)
        facts = facts_heard(text, heard)
        rows.append({"text": text, "spoken": spoken, "voice": voice, "heard": heard, "similarity": round(score, 3), "facts": facts,
                     "seconds": round(len(samples) / sr, 2), "ok": score >= floor and facts})
    n = len(rows)
    return {"n": n, "pass_rate": round(sum(r["ok"] for r in rows) / n, 3) if n else None, "floor": floor,
            "tts": getattr(tts, "name", "?"), "asr": getattr(asr, "name", "?"),
            "degradation": degrade_spec.to_dict() if degrade_spec is not None else None,
            "failures": [r for r in rows if not r["ok"]], "rows": rows}


__all__ = ["SIMILARITY_FLOOR", "FasterWhisperASR", "MlxWhisperASR", "asr_available", "canon", "facts_heard", "levenshtein",
           "load_asr", "normalize", "roundtrip_check", "similarity", "wav_samples_from_clip"]
