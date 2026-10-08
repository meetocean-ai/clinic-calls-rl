"""Signal-property tests for the telephony degradation pipeline and the disfluency injector (roadmap P2-02). No audio
files, no server: synthetic tones and noise, measured.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio -v
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from env.audio.degrade import (BAND_HIGH_HZ, TELEPHONY_SR, DegradationSpec, add_noise, band_energy_ratio, bandlimit, degrade,
                               drop_frames, gain_wobble, measured_snr_db, mu_law_roundtrip, resample)
from env.audio.disfluencies import _PROTECTED, DisfluencySpec, inject

pytestmark = [pytest.mark.covers("rl-env-voice")]


def _tone(freq: float, sr: int = 16000, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speechlike(sr: int = 16000, seconds: float = 3.0, seed: int = 1) -> np.ndarray:
    """A few harmonics between 150 Hz and 3 kHz plus a 6 kHz component the phone line must remove."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    x = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f, a in ((160, 0.3), (480, 0.2), (1200, 0.15), (2600, 0.1), (6000, 0.2)))
    return (x / np.max(np.abs(x)) * 0.6).astype(np.float32)


def test_spec_is_seeded_and_tiered():
    assert DegradationSpec.from_seed(5, 3) == DegradationSpec.from_seed(5, 3)
    assert DegradationSpec.from_seed(5, 1).snr_db is None and DegradationSpec.from_seed(5, 1).frame_drop_p == 0.0
    bad = [DegradationSpec.from_seed(s, 4) for s in range(1, 21)]
    assert all(b.snr_db <= 15.0 and b.frame_drop_p >= 0.02 for b in bad)
    assert len({(b.snr_db, b.frame_drop_p, b.gain_wobble_db) for b in bad}) > 3, "tier 4 draws vary"
    assert DegradationSpec.from_seed(5, 3).to_dict()["sample_rate"] == TELEPHONY_SR


def test_resample_to_8k_keeps_the_tone():
    y = resample(_tone(440), 16000, 8000)
    assert len(y) == 8000 * 2
    spectrum = np.abs(np.fft.rfft(y))
    assert abs(np.fft.rfftfreq(len(y), 1 / 8000)[np.argmax(spectrum)] - 440) < 5


def test_bandlimit_removes_energy_outside_300_3400():
    x = _speechlike()
    y = bandlimit(resample(x, 16000, 8000), 8000)
    assert band_energy_ratio(y, 8000, BAND_HIGH_HZ + 200) < 0.02
    low = _tone(100, sr=8000)
    assert float(np.sqrt(np.mean(bandlimit(low, 8000) ** 2))) < 0.1 * float(np.sqrt(np.mean(low ** 2)))
    mid = _tone(1000, sr=8000)
    assert float(np.sqrt(np.mean(bandlimit(mid, 8000) ** 2))) > 0.8 * float(np.sqrt(np.mean(mid ** 2)))


@pytest.mark.parametrize("snr", (25.0, 10.0, 5.0))
def test_noise_lands_on_the_requested_snr(snr):
    x = _speechlike(sr=8000)
    y = add_noise(x, snr, np.random.default_rng(3))
    assert abs(measured_snr_db(x, y) - snr) < 1.0


def test_mu_law_roundtrip_is_small_and_level_dependent():
    x = _speechlike(sr=8000)
    y = mu_law_roundtrip(x)
    err = float(np.max(np.abs(y - x)))
    assert 0 < err < 0.02, err  # 8-bit companding: quiet parts keep more precision than loud ones
    quiet, loud = np.array([0.001], dtype=np.float32), np.array([0.9], dtype=np.float32)
    assert abs(mu_law_roundtrip(quiet)[0] - quiet[0]) < abs(mu_law_roundtrip(loud)[0] - loud[0])


def test_frame_drops_match_the_probability_and_zero_whole_frames():
    x = np.ones(8000 * 10, dtype=np.float32) * 0.5
    y, dropped = drop_frames(x, 8000, 0.1, np.random.default_rng(7))
    frames = len(x) // 160
    assert abs(dropped / frames - 0.1) < 0.03
    zero_frames = sum(1 for s in range(0, len(y), 160) if np.all(y[s:s + 160] == 0))
    assert zero_frames == dropped
    assert drop_frames(x, 8000, 0.0, np.random.default_rng(1))[1] == 0


def test_gain_wobble_stays_within_the_peak_to_peak_range():
    x = np.ones(8000 * 5, dtype=np.float32) * 0.1
    y = gain_wobble(x, 8000, 6.0, np.random.default_rng(2))
    ratio_db = 20 * np.log10(y / x)
    assert ratio_db.max() - ratio_db.min() > 5.0 and abs(ratio_db).max() <= 3.01


def test_degrade_is_deterministic_and_outputs_telephony_audio():
    x = _speechlike()
    spec = DegradationSpec.from_seed(5, 4)
    y1, sr1 = degrade(x, 16000, spec)
    y2, sr2 = degrade(x, 16000, spec)
    assert sr1 == sr2 == TELEPHONY_SR and np.array_equal(y1, y2) and y1.dtype == np.float32
    assert band_energy_ratio(y1, sr1, 3800) < 0.05
    assert np.max(np.abs(y1)) <= 1.0
    other, _ = degrade(x, 16000, DegradationSpec.from_seed(6, 4))
    assert not np.array_equal(y1, other)


# ── disfluencies ─────────────────────────────────────────────────────────

LINES = (
    "Hi, I need to move my appointment on Tuesday, January 15 at 11 AM to another day.",
    "My name is Skyler Northcott, and my date of birth is 1964-04-11.",
    "Hi, I'd like to book a follow-up visit with Dr. Eddins on Monday, January 14 at 10 AM, please.",
    "I have heavy pressure in my chest that's spreading into my jaw, and I'm sweating.",
)


def test_disfluency_spec_is_seeded_and_tiered():
    assert DisfluencySpec.from_seed(5, 3) == DisfluencySpec.from_seed(5, 3)
    assert DisfluencySpec.from_seed(5, 1) == DisfluencySpec(seed=5)
    assert all(DisfluencySpec.from_seed(s, 4).fillers >= 2 for s in range(1, 21))


@pytest.mark.parametrize("line", LINES)
@pytest.mark.parametrize("tier", (2, 3, 4))
def test_injection_never_touches_dates_names_or_numbers(line, tier):
    for seed in range(1, 16):
        out, spec = inject(line, seed, tier)
        assert inject(line, seed, tier)[0] == out
        for m in _PROTECTED.finditer(line):
            assert m.group(0) in out, (m.group(0), out)
        if spec.fillers:
            assert any(f in out for f in ("um", "uh", "er", "hmm", "let me think", "you know")), out
        if spec.false_start:
            assert out.startswith(("I— ", "So I— ", "It's— ", "We— ")), out
        if spec.self_correction:
            assert "— sorry, I mean " in out or "— no, wait, " in out, out
        # the plain words of the original are all still there, in order (a false start lowercases the first one)
        plain = [w.lower() for w in re.findall(r"[A-Za-z']+", line)]
        pos, hay = 0, out.lower()
        for w in plain:
            i = hay.find(w, pos)
            assert i >= 0, (w, out)
            pos = i + len(w)


def test_tier1_is_the_clean_line():
    for line in LINES:
        assert inject(line, 5, 1)[0] == line
