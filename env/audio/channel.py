"""The caller's lines as telephone audio (roadmap P2-03): what `EhrSchedulingEnv` attaches to an observation when it
runs in audio mode. One channel per episode — the seed picks the voice, the tier picks the disfluencies and the phone
line — so the same episode always sounds the same.

    channel = VoiceChannel.for_task(seed=5, tier=3)             # Kokoro when installed, else the tone stand-in
    clip = channel.render("Hi, I'd like to book a follow-up visit.")
    clip["wav_b64"], clip["sample_rate"], clip["seconds"], clip["spoken"]

The text stays in the observation next to the audio, so text-mode policies keep working; `method_version` carries
the mode, the manifest carries the specs.
"""
from __future__ import annotations

import base64
import io
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .degrade import DegradationSpec, degrade
from .disfluencies import DisfluencySpec, inject, verbalize
from .tts import TTS, load_tts, voice_for

HESITATION_PAUSE_S = (0.6, 1.0)  # a pause this long inside a date is where an eager agent barges in
_SPOKEN_DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}),? (\d{4})\b")


def _date_pieces(spoken: str) -> Optional[List[str]]:
    """'It's April 11, 1964.' → ['It's April', '11', '1964.'] — the month, the day and the year as separate breaths."""
    m = _SPOKEN_DATE.search(spoken)
    if not m:
        return None
    before, after = spoken[: m.start()].rstrip(), spoken[m.end():].lstrip()
    pieces = [f"{before} {m.group(1)}".strip(), m.group(2), f"{m.group(3)}{(' ' + after) if after and after[0].isalnum() else after}".strip()]
    return [p for p in pieces if p]


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(samples, -1.0, 1.0).astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_to_samples(data: bytes):
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(data), dtype="float32")
    return samples, sr


@dataclass
class VoiceChannel:
    seed: int
    tier: int
    voice: str
    degradation: DegradationSpec
    disfluency: DisfluencySpec
    tts: TTS

    @classmethod
    def for_task(cls, seed: int, tier: int, *, tts: Optional[TTS] = None, prefer: Optional[str] = None) -> "VoiceChannel":
        return cls(seed=seed, tier=tier, voice=voice_for(seed), degradation=DegradationSpec.from_seed(seed, tier),
                   disfluency=DisfluencySpec.from_seed(seed, tier), tts=tts or load_tts(prefer))

    def render(self, text: str, *, hesitate_dob: bool = False, onset_s: Optional[float] = None) -> Dict[str, Any]:
        """The caller's line as it reaches the agent: disfluent, spoken, down the phone line. `spoken` is the text
        actually voiced (with the disfluencies); the clean text travels separately in the observation.

        Speech-only behaviour (roadmap P2-02b): `hesitate_dob` voices a date in pieces with pauses inside it (the
        barge-in trap — an agent that takes a pause for the end of the line cuts the caller off mid-date);
        `onset_s` < 0 marks the caller starting before the agent finished (talk-over). Both leave timestamped `events`."""
        spoken, _ = inject(verbalize(text), self.seed, self.tier, spec=self.disfluency)
        events: List[Dict[str, Any]] = []
        pieces = _date_pieces(spoken) if hesitate_dob else None
        if pieces:
            rng = random.Random(f"hesitate:{self.seed}:{spoken}")
            chunks, sr, t = [], None, 0.0
            for i, piece in enumerate(pieces):
                s, sr = self.tts.synthesize(piece, voice=self.voice)
                events.append({"t": round(t, 3), "kind": "speech", "text": piece})
                chunks.append(s)
                t += len(s) / sr
                if i < len(pieces) - 1:
                    pause = round(rng.uniform(*HESITATION_PAUSE_S), 2)
                    events.append({"t": round(t, 3), "kind": "pause", "seconds": pause})
                    chunks.append(np.zeros(int(pause * sr), dtype=np.float32))
                    t += pause
            samples = np.concatenate(chunks)
            spoken = " … ".join(pieces)
        else:
            samples, sr = self.tts.synthesize(spoken, voice=self.voice)
            events.append({"t": 0.0, "kind": "speech", "text": spoken})
        samples, sr = degrade(samples, sr, self.degradation)
        clip = {"wav_b64": base64.b64encode(wav_bytes(samples, sr)).decode("ascii"), "sample_rate": sr,
                "seconds": round(len(samples) / sr, 3), "spoken": spoken, "voice": self.voice, "tts": self.tts.name, "events": events}
        if onset_s is not None:
            clip["caller_onset_s"] = onset_s  # relative to the END of the agent's line; negative = the caller talked over it
        return clip

    def spec(self) -> Dict[str, Any]:
        return {"voice": self.voice, "tts": self.tts.name, "degradation": self.degradation.to_dict(),
                "disfluency": self.disfluency.to_dict()}


__all__ = ["HESITATION_PAUSE_S", "VoiceChannel", "wav_bytes", "wav_to_samples"]
