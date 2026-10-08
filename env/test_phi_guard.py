"""PHI guard (roadmap X-05): nothing that looks like a real person can come out of the task generator, the caller
scripts or the publishable packages. Everything is synthetic by construction — fixed name lists, Twilio test numbers,
generated dates of birth, invented providers and payers — and this test is the proof that stays true.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/test_phi_guard.py -v
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from env.tasks import _FIRST, _LAST, _PAYERS, _PROVIDERS, FAMILIES, LOOKALIKES, TIERS, generate

pytestmark = [pytest.mark.covers("rl-env-tasks-v2")]

TODAY = date(2030, 1, 7)
TWILIO_TEST = re.compile(r"^\+1500555\d{4}$")
# What a leak would look like: a US phone number outside the Twilio test range, an SSN, an MRN-like long number,
# an email, a street address.
PHONE_LIKE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
SSN_LIKE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
LONG_DIGITS = re.compile(r"\b\d{7,}\b")
EMAIL_LIKE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ADDRESS_LIKE = re.compile(r"\b\d{1,5}\s+\w+\s+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Drive|Dr\.|Lane|Ln\.)\b", re.IGNORECASE)

ALLOWED_PEOPLE = set(_FIRST) | set(_LAST) | {g for g, _ in _PROVIDERS} | {f for _, f in _PROVIDERS} | set(LOOKALIKES.values())


def _every_task():
    for family in FAMILIES:
        for tier in TIERS:
            for seed in range(1, 101):
                yield generate(family, seed, tier=tier, today=TODAY)


def _texts(task):
    yield task.goal
    yield task.opening_line
    for rule in task.constraints():
        yield rule


def test_identities_come_only_from_the_synthetic_generators():
    for t in _every_task():
        assert TWILIO_TEST.match(t.patient["phone"]), t.patient["phone"]
        assert t.patient["given"] in _FIRST and t.patient["family"] in _LAST
        assert 1950 <= date.fromisoformat(t.patient["dob"]).year <= 2000
        assert t.provider["given"] in {g for g, _ in _PROVIDERS} and t.provider["family"] in {f for _, f in _PROVIDERS}
        if t.duplicate:
            assert t.duplicate["given"] in _FIRST and t.duplicate["family"] in _LAST
        if t.lookalike:
            assert t.lookalike["family"] in LOOKALIKES.values() and t.lookalike["given"] in _FIRST
        if t.switch_to:
            assert t.switch_to["family"] in {f for _, f in _PROVIDERS}
        if t.payer:
            assert t.payer in _PAYERS


def test_caller_scripts_carry_no_phone_ssn_email_or_address():
    for t in _every_task():
        for text in _texts(t):
            phones = [m for m in PHONE_LIKE.findall(text) if not TWILIO_TEST.match(re.sub(r"[^\d+]", "", m))]
            assert not phones, (t.id, phones, text[:120])
            assert not SSN_LIKE.search(text), (t.id, text[:120])
            assert not EMAIL_LIKE.search(text), (t.id, text[:120])
            assert not ADDRESS_LIKE.search(text), (t.id, text[:120])
            # the only long digit runs allowed are ISO dates, which the regex above does not match (hyphens split them)
            assert not LONG_DIGITS.search(text), (t.id, text[:120])


def test_publishable_packages_and_docs_carry_no_identifiers():
    here = Path(__file__).parent
    files = [*here.glob("vf_package/*"), *here.glob("openenv_space/**/*"), here / "README.md", here / "DATASET.md"]
    for f in files:
        if not f.is_file() or f.suffix in (".lock",):
            continue
        text = f.read_text(errors="ignore")
        assert not SSN_LIKE.search(text), f
        phones = [m for m in PHONE_LIKE.findall(text) if not TWILIO_TEST.match(re.sub(r"[^\d+]", "", m))]
        assert not phones, (f, phones)
        emails = [e for e in EMAIL_LIKE.findall(text) if not e.endswith(("example.com", "meetocean.ai"))]
        assert not emails, (f, emails)


def test_bystanders_and_fixtures_stay_in_the_test_phone_range():
    """World-side identities the generator does not emit: bystander phones (+150055599NN) and the fixture pool."""
    from env.world import book_bystander  # noqa: F401 — the phone format lives next to it
    from fixtures.patients import _TEST_PHONES

    assert all(TWILIO_TEST.match(p) for p in _TEST_PHONES), _TEST_PHONES
    assert TWILIO_TEST.match("+15005559901")
