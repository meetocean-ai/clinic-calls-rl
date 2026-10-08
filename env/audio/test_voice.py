"""The caller's voice and the ASR round trip (roadmap P2-01). The roster and the comparison need no model; the
synthesis and transcription tests run when Kokoro / faster-whisper are installed (`tests/behavioral[voice]`) and
their weights are cached, and skip otherwise.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio/test_voice.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from env.audio.degrade import DegradationSpec, degrade
from env.audio.roundtrip import asr_available, levenshtein, normalize, roundtrip_check, similarity
from env.audio.tts import KOKORO_VOICES, ToneTTS, Voice, kokoro_available, load_tts, voice_for

pytestmark = [pytest.mark.covers("rl-env-voice")]
needs_models = pytest.mark.skipif(not (kokoro_available() and asr_available()) or os.environ.get("EHR_SKIP_VOICE_MODELS") == "1",
                                  reason="Kokoro + faster-whisper not installed (pip install -e tests/behavioral[voice])")

LINES = (
    "Hi, I'd like to book a follow-up visit with Dr. Eddins, please.",
    "Hi, I need to move my appointment on Tuesday, January 15 at 11 AM to another day.",
    "My name is Skyler Northcott, and my date of birth is 1964-04-11.",
    "Hi, I'm a new patient and I'd like to book a first visit with Dr. Campos.",
    "I have heavy pressure in my chest that's spreading into my jaw, and I'm sweating.",
)


def test_roster_has_twelve_plus_distinct_voices_across_gender_and_accent():
    voices = [Voice.parse(v) for v in KOKORO_VOICES]
    assert len(KOKORO_VOICES) >= 12 and len(set(KOKORO_VOICES)) == len(KOKORO_VOICES)
    assert {v.gender for v in voices} == {"f", "m"} and {v.accent for v in voices} == {"american", "british"}


def test_voice_for_seed_is_deterministic_and_spread():
    assert voice_for(5) == voice_for(5)
    assert len({voice_for(s) for s in range(1, 101)}) >= 10


def test_normalize_and_similarity_care_about_words_not_formatting():
    assert normalize("Hi, I'd like to book — at 11 AM!") == "hi i'd like to book at 11 am"
    assert similarity("Tuesday, January 15 at 11 AM", "tuesday january 15 at 11 am") == 1.0
    assert similarity("my date of birth is 1964-04-11", "my date of birth is 1964 04 11") == 1.0
    assert similarity("book with Dr. Eddins", "book with Dr. Edwins") > 0.85
    assert similarity("book with Dr. Eddins", "cancel everything tomorrow") < 0.5
    assert levenshtein("Dr. Eddins", "Dr. Eddinson") == 2


def test_verbalize_and_canon_make_dates_and_times_comparable_across_the_trip():
    from env.audio.disfluencies import verbalize
    from env.audio.roundtrip import canon, facts_heard

    assert verbalize("my date of birth is 1964-04-11.") == "my date of birth is April 11, 1964."
    assert verbalize("on 2030-13-40") == "on 2030-13-40"  # not a date, left alone
    assert canon("Tuesday, January 15th at 11:00 A.M.") == canon("tuesday january 15 at 11 am")
    assert facts_heard("I need to cancel my appointment on Tuesday, January 15 at 11 AM.", "I need to cancel my appointment on Tuesday January 15th at 11 a.m.")
    assert facts_heard("My date of birth is 1964-04-11.", "my date of birth is April 11th 1964")
    assert facts_heard("My date of birth is 1964-04-11.", "my date of birth is april 11")  # year dropped is still the date
    assert facts_heard("book with Dr. Eddins", "book with doctor Eddens")  # one edit on a name
    assert not facts_heard("book with Dr. Campos", "book with Dr. Campbell")  # a different name is a different doctor
    assert not facts_heard("on Tuesday, January 15 at 11 AM", "on Tuesday January 16 at 11 am")


def test_tone_backend_tracks_text_length_and_survives_the_phone_line():
    tts = ToneTTS()
    short, sr = tts.synthesize("Hi.", voice="af_heart")
    long_, _ = tts.synthesize(LINES[1], voice="am_adam")
    assert sr == 16000 and len(long_) > len(short) and np.max(np.abs(long_)) <= 0.31
    y, sr8 = degrade(long_, sr, DegradationSpec.from_seed(5, 4))
    assert sr8 == 8000 and len(y) == len(long_) // 2 and np.max(np.abs(y)) <= 1.0
    assert load_tts(prefer="tone").name == "tone"


@needs_models
def test_kokoro_speaks_and_whisper_hears_the_clean_line():
    from env.audio.roundtrip import load_asr

    report = roundtrip_check([(LINES[0], "af_heart"), (LINES[2], "am_adam"), (LINES[1], "bf_emma")], tts=load_tts(prefer="kokoro"), asr=load_asr())
    assert report["tts"] == "kokoro" and report["pass_rate"] == 1.0, report["failures"]
    assert all(1.0 < r["seconds"] < 12.0 for r in report["rows"]), report["rows"]


@needs_models
def test_round_trip_survives_a_tier3_phone_line():
    from env.audio.roundtrip import load_asr

    report = roundtrip_check([(LINES[0], "af_bella"), (LINES[3], "bm_george")], tts=load_tts(prefer="kokoro"), asr=load_asr(),
                             degrade_spec=DegradationSpec.from_seed(5, 3))
    assert report["pass_rate"] == 1.0, report["failures"]
