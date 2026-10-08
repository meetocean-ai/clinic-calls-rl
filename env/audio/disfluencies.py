"""Text-side disfluency injection before TTS (roadmap P2-02, the FDB-v3 disfluency classes): fillers, false starts
and self-corrections added to a caller line, seeded, without touching the facts the environment verifies — names,
dates, times and numbers stay exactly as written so the spoken-date guard and the DOB check still work.

    line, spec = inject("I need to move my appointment on Tuesday, January 15 at 11 AM.", seed=5, tier=3)
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

FILLERS = ("um", "uh", "er", "hmm", "let me think", "you know")
FALSE_STARTS = ("I— ", "So I— ", "It's— ", "We— ")
CORRECTION_PATTERNS = ("— sorry, I mean ", "— no, wait, ")

# Tokens we never alter or split: dates in any spoken or ISO form, clock times, phone-like digit runs, titles + names.
_PROTECTED = re.compile(
    r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\+?\d[\d\s().-]{6,}\d"
    r"|\bDr\.\s+\w+\b)"
)


@dataclass(frozen=True)
class DisfluencySpec:
    fillers: int = 0  # how many fillers to insert
    false_start: bool = False
    self_correction: bool = False
    seed: int = 0

    @classmethod
    def from_seed(cls, seed: int, tier: int = 1) -> "DisfluencySpec":
        rng = random.Random(f"ehr-env-disfluency:{seed}:{tier}")
        if tier <= 1:
            return cls(seed=seed)
        if tier == 2:
            return cls(fillers=rng.choice((0, 1)), seed=seed)
        if tier == 3:
            return cls(fillers=rng.choice((1, 2)), false_start=rng.random() < 0.3, self_correction=rng.random() < 0.3, seed=seed)
        return cls(fillers=rng.choice((2, 3)), false_start=rng.random() < 0.5, self_correction=rng.random() < 0.5, seed=seed)

    def to_dict(self) -> Dict:
        return asdict(self)


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def verbalize(text: str) -> str:
    """What a person says versus what the script stores: ISO dates become spoken dates ("1964-04-11" → "April 11, 1964")
    before TTS. The text-mode line keeps the ISO form (the environment's spoken-date guard reads it); the audio says it
    the way a caller would, which is also what an ASR can transcribe back."""

    def _spoken(m: re.Match) -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12 or not 1 <= d <= 31:
            return m.group(0)
        return f"{_MONTHS[mo - 1]} {d}, {y}"

    return _ISO_DATE.sub(_spoken, text)


def _segments(text: str) -> List[Tuple[str, bool]]:
    """Split into (chunk, protected) so insertions only land inside unprotected prose."""
    out, pos = [], 0
    for m in _PROTECTED.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _insert_fillers(segments: List[Tuple[str, bool]], n: int, rng: random.Random) -> List[Tuple[str, bool]]:
    """Put `n` fillers at word boundaries inside unprotected chunks (never at a chunk's edges)."""
    points: List[Tuple[int, int]] = []
    for i, (chunk, protected) in enumerate(segments):
        if protected:
            continue
        points += [(i, m.start()) for m in re.finditer(r"(?<=\S)\s+(?=\S)", chunk)]
    segments = list(segments)
    for _ in range(min(n, len(points))):
        i, at = points.pop(rng.randrange(len(points)))
        chunk = segments[i][0]
        insert = ", " + rng.choice(FILLERS) + ","
        segments[i] = (chunk[:at] + insert + chunk[at:], False)
        points = [(j, p + len(insert) if (j == i and p > at) else p) for j, p in points]
    return segments


def _self_correct(text: str, rng: random.Random) -> str:
    """Say an unimportant word twice with a correction marker: the first unprotected word of 6+ letters, cut after at
    least three letters so the fragment is pronounceable ("appoint— sorry, I mean appointment"); a two-letter cut
    ("th—") came back from the ASR as initials and sank the round trip (2026-10-07)."""
    segs = _segments(text)
    for i, (chunk, protected) in enumerate(segs):
        if protected:
            continue
        m = re.search(r"\b([A-Za-z]{6,})\b", chunk)
        if m and m.group(1).lower() not in ("sorry", "mean", "wait"):
            word = m.group(1)
            segs[i] = (chunk[:m.start()] + word[: max(3, len(word) * 2 // 3)] + rng.choice(CORRECTION_PATTERNS) + word + chunk[m.end():], False)
            return "".join(c for c, _ in segs)
    return text


def inject(text: str, seed: int, tier: int = 1, spec: Optional[DisfluencySpec] = None) -> Tuple[str, DisfluencySpec]:
    """Return the disfluent line and the spec that produced it. Protected tokens are byte-identical in the output."""
    spec = spec or DisfluencySpec.from_seed(seed, tier)
    rng = random.Random(f"ehr-env-disfluency:{spec.seed}:{spec.fillers}:{spec.false_start}:{spec.self_correction}")
    out = "".join(c for c, _ in _insert_fillers(_segments(text), spec.fillers, rng))
    if spec.false_start:
        start = rng.choice(FALSE_STARTS)
        out = start + (out[0].lower() + out[1:] if out and out[0].isupper() and not out.startswith("I ") else out)
    if spec.self_correction:
        out = _self_correct(out, rng)
    return out, spec


__all__ = ["CORRECTION_PATTERNS", "FALSE_STARTS", "FILLERS", "DisfluencySpec", "inject", "verbalize"]
