"""Telephony degradation pipeline (roadmap P2-02, the τ-voice recipe): what a clinic's phone line does to a caller's
voice, as a seeded chain of numpy/scipy stages so every episode's degradation is reproducible from its spec.

    spec = DegradationSpec.from_seed(seed=5, tier=3)      # the knobs a task draws
    y, sr = degrade(samples, sample_rate, spec)           # float32 mono at 8 kHz

Stages, in the order a call goes through them: resample to 8 kHz (narrowband), band-limit to 300–3400 Hz, additive
noise at a target SNR, G.711 μ-law round trip (8-bit companding), frame drops (20 ms frames, packet loss), gain
variation (a slow level wobble). Each stage is a function you can test on its own; `degrade` composes them.
The spec is recorded in the episode manifest (P2-03) next to `method_version`.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

TELEPHONY_SR = 8000
BAND_LOW_HZ, BAND_HIGH_HZ = 300.0, 3400.0
FRAME_MS = 20


@dataclass(frozen=True)
class DegradationSpec:
    """The knobs of one episode's phone line. `tier` 1–2 = a clean line; 3 = a typical mobile call; 4 = a bad one."""

    sample_rate: int = TELEPHONY_SR
    bandlimit: bool = True
    snr_db: Optional[float] = None  # None = no added noise
    mu_law: bool = True
    frame_drop_p: float = 0.0  # probability a 20 ms frame is lost
    gain_wobble_db: float = 0.0  # peak-to-peak slow gain variation
    seed: int = 0

    @classmethod
    def from_seed(cls, seed: int, tier: int = 1) -> "DegradationSpec":
        rng = random.Random(f"ehr-env-audio:{seed}:{tier}")
        if tier <= 2:
            return cls(snr_db=None, frame_drop_p=0.0, gain_wobble_db=0.0, seed=seed)
        if tier == 3:
            return cls(snr_db=rng.choice((25.0, 20.0, 15.0)), frame_drop_p=rng.choice((0.0, 0.01, 0.02)),
                       gain_wobble_db=rng.choice((0.0, 3.0)), seed=seed)
        return cls(snr_db=rng.choice((15.0, 10.0, 5.0)), frame_drop_p=rng.choice((0.02, 0.05, 0.1)),
                   gain_wobble_db=rng.choice((3.0, 6.0)), seed=seed)

    def to_dict(self) -> Dict:
        return asdict(self)


# ── stages ───────────────────────────────────────────────────────────────


def to_mono_float(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return np.clip(x.astype(np.float32), -1.0, 1.0)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from scipy.signal import resample_poly

    from math import gcd

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def bandlimit(x: np.ndarray, sr: int, low_hz: float = BAND_LOW_HZ, high_hz: float = BAND_HIGH_HZ) -> np.ndarray:
    """4th-order Butterworth band-pass, zero-phase — the narrowband telephone channel."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    high = min(high_hz, nyq * 0.98)
    sos = butter(4, [low_hz / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White noise scaled so the signal-to-noise ratio over the whole clip is `snr_db`."""
    p_signal = float(np.mean(x.astype(np.float64) ** 2)) or 1e-12
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape) * np.sqrt(p_noise)
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def mu_law_roundtrip(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 μ-law companding to 8 bits and back — the codec of a POTS/SIP call."""
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0  # 256 levels
    return (np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)).astype(np.float32)


def drop_frames(x: np.ndarray, sr: int, p: float, rng: np.random.Generator, frame_ms: int = FRAME_MS) -> Tuple[np.ndarray, int]:
    """Zero out 20 ms frames with probability `p` (lost packets, played as silence). Returns (signal, frames dropped)."""
    if p <= 0:
        return x.astype(np.float32), 0
    n = int(sr * frame_ms / 1000)
    y = x.astype(np.float32).copy()
    dropped = 0
    for start in range(0, len(y), n):
        if rng.random() < p:
            y[start:start + n] = 0.0
            dropped += 1
    return y, dropped


def gain_wobble(x: np.ndarray, sr: int, peak_to_peak_db: float, rng: np.random.Generator) -> np.ndarray:
    """A slow (0.2–0.6 Hz) level variation of the given peak-to-peak size in dB."""
    if peak_to_peak_db <= 0:
        return x.astype(np.float32)
    f = rng.uniform(0.2, 0.6)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(len(x)) / sr
    db = (peak_to_peak_db / 2.0) * np.sin(2 * np.pi * f * t + phase)
    return np.clip(x * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32)


# ── the chain ────────────────────────────────────────────────────────────


def degrade(samples: np.ndarray, sample_rate: int, spec: DegradationSpec) -> Tuple[np.ndarray, int]:
    """Apply the spec to a clip. Deterministic for a given (clip, spec)."""
    rng = np.random.default_rng(spec.seed)
    x = to_mono_float(samples)
    x = resample(x, sample_rate, spec.sample_rate)
    if spec.bandlimit:
        x = bandlimit(x, spec.sample_rate)
    if spec.snr_db is not None:
        x = add_noise(x, spec.snr_db, rng)
    if spec.mu_law:
        x = mu_law_roundtrip(x)
    x, _ = drop_frames(x, spec.sample_rate, spec.frame_drop_p, rng)
    x = gain_wobble(x, spec.sample_rate, spec.gain_wobble_db, rng)
    return x.astype(np.float32), spec.sample_rate


def band_energy_ratio(x: np.ndarray, sr: int, above_hz: float) -> float:
    """Share of the clip's energy above `above_hz` — what the band-limit test measures."""
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = float(spectrum.sum()) or 1e-12
    return float(spectrum[freqs > above_hz].sum()) / total


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    return 10.0 * np.log10((np.mean(clean.astype(np.float64) ** 2) or 1e-12) / (np.mean(noise ** 2) or 1e-12))


__all__ = ["BAND_HIGH_HZ", "BAND_LOW_HZ", "DegradationSpec", "FRAME_MS", "TELEPHONY_SR", "add_noise", "band_energy_ratio",
           "bandlimit", "degrade", "drop_frames", "gain_wobble", "measured_snr_db", "mu_law_roundtrip", "resample",
           "to_mono_float"]
