"""Seeded, synthetic tasks for the EHR scheduling environment (E3).

Everything derives from `(family, seed)` through `random.Random`, so a task is the same on any
machine. Identities are synthetic (fixed name lists, Twilio magic numbers, generated DOBs).
`seed % 5 == 0` is held out and never used for optimization.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

FAMILIES = ("booking", "cancel", "reschedule", "verify_fail_transfer", "new_patient_intake", "emergency_redirect")
# Red-flag presentations (synthetic scripts). The product's rule: urgent symptoms → warm transfer with reason "urgent",
# and the shared prompt's 911 line. Never an appointment.
_RED_FLAGS = (
    ("I have heavy pressure in my chest that's spreading into my jaw, and I'm sweating.", "chest pressure spreading to the jaw"),
    ("My left arm went numb about twenty minutes ago and my words are coming out slurred.", "one-sided numbness and slurred speech"),
    ("I can barely get a breath in and my lips look kind of blue.", "severe shortness of breath"),
    ("The worst headache of my life came on all at once and I can't see properly out of one eye.", "sudden worst headache with vision loss"),
)
TIERS = (1, 2, 3, 4)  # 1 polite baseline · 2 personas · 3 one or two knobs · 4 combinations of two or three knobs
HELDOUT_MODULUS = 5  # every fifth seed is held out (20%); reviewed for dataset v2 and kept so earlier runs stay comparable
# The frozen held-out evaluation set (dataset v2): ten seeds per family, every tier. Bracketing and the difficulty loop
# (roadmap P0-13/15) run on these; training never sees them.
HELDOUT_EVAL_SEEDS = tuple(range(5, 51, 5))

_FIRST = ("Avery", "Blake", "Casey", "Dakota", "Emerson", "Finley", "Harper", "Jordan", "Kendall", "Logan",
          "Morgan", "Parker", "Quinn", "Reese", "Rowan", "Sawyer", "Skyler", "Taylor", "Tatum", "Wren")
_LAST = ("Ashgrove", "Brightwater", "Coldbrook", "Dunmore", "Eastwick", "Fairweather", "Greenhollow", "Hartwell",
         "Ironwood", "Kingsley", "Lockridge", "Marlowe", "Northcott", "Oakhurst", "Pemberton", "Ravenscroft",
         "Stonebridge", "Thornbury", "Underhill", "Westbrook")
# Twilio magic test numbers, the only phones this suite uses (fixtures/patients.py).
_PHONES = tuple(f"+1500555{n:04d}" for n in range(1, 15))
_PROVIDERS = (("Robin", "Eddins"), ("Ana", "Campos"), ("Jo", "Facchinato"))
_PERSONAS_T1 = ("polite",)
# Blocks the simulator knows (runtime/patient_simulator.py::_PERSONA_BLOCKS). `spanish` stays out until the voice phase:
# the env's spoken-date guard reads English dates.
_PERSONAS_T2 = ("anxious", "rushed", "skeptical", "confused", "hostile", "elderly", "demanding")
# How far the caller cooperates with the PROCESS (tone, pushback, questions) — never with the outcome: every goal
# keeps its own acceptance rule so the end state the verifier expects does not move.
COOPERATION_RANGE = {"polite": (5, 5), "elderly": (4, 4), "anxious": (3, 4), "confused": (3, 4), "rushed": (3, 3),
                     "skeptical": (2, 3), "hostile": (1, 2), "demanding": (1, 2)}
_COOPERATION = {
    1: "resist the process: complain, ask why each step is needed, demand speed — then still follow your goal's acceptance rules",
    2: "grudging: grumble and push back once on each step before going along with it",
    3: "neutral: ask one clarifying question before you go along with a step",
    4: "cooperative: go along with reasonable steps, ask when unsure",
    5: "fully cooperative: answer promptly and go along with each step",
}
# Caller realism (P0-10): how the caller talks, never what they want. Each trait is one "Additional rule" in the
# simulator prompt (PatientProfile.constraints); the ScriptedCaller ignores them; the verifier never sees them.
REALISM = {
    "fillers": ("Talk like a real caller on the phone: use fillers ('um', 'uh', 'let me think…') now and then, and trail off "
                "once or twice."),
    "self_corrections": ("Once or twice, correct yourself mid-sentence on something unimportant ('I'll drive — no, actually my son "
                         "will drop me off'); never on your name, your date of birth or the dates in your goal."),
    "interruptions": ("Once during the call, cut in with 'actually, wait —' and ask the agent to repeat what they just said "
                      "before you go on."),
    "off_topic_aside": ("Once, before answering a question, make a short aside about something unrelated (parking, the weather, "
                        "your dog), then answer."),
    "limited_english": ("Your English is limited: short, simple sentences with small grammar slips; ask the agent to slow down or "
                        "repeat once. You still say your date of birth and the dates clearly."),
    "hesitant_dob": ("Give your date of birth hesitantly, in pieces with pauses ('it's… um… March… the ninth…'), then say the "
                     "whole date plainly once as month day, year."),
    "hidden_until_asked": ("Facts you volunteer ONLY if asked: your insurance is {payer}; you are allergic to penicillin. Never "
                           "bring them up yourself."),
}
_PAYERS = ("Harborline PPO", "Northstar Health HMO", "Blue Meadow Choice")  # synthetic
REALISM_PER_TIER = {1: (0, 0), 2: (0, 1), 3: (1, 2), 4: (2, 3)}

CHAOS_KNOBS = {
    "slot_taken": "another booking lands on the offered slot between search and book (the booking must move)",
    "existing_visit": "the caller already holds a visit with the provider; a plain booking is refused (B-106)",
    "dob_corrected": "the caller misstates their date of birth once and corrects it",
    "mind_change": "opens as a booking, must end as a cancel of the visit on file",
    # v2 (2026-09-29): the knobs the clinic actually surfaces; v1 was fair (oracle 1.0) but production scored 48/48.
    "slot_taken_twice": "the first TWO booking / reschedule writes are rejected (two bystanders, two 409s)",
    "shared_phone": ("a second patient record shares the caller's phone number; the phone lookup cannot pick one, "
                     "the DOB must — production's lookup_patient_by_phone takes the first match"),
    "no_availability": "the named provider has no open slots in the window; the right outcome is the waitlist, not a booking",
    # v3 (2026-10-07, roadmap P0-02…): the misses a trained model has to learn, not just survive.
    "exact_time_unavailable": ("the caller insists on one specific time with the named provider and that slot is already "
                               "another patient's; the right outcome is a different time with the SAME provider — never "
                               "that time, never another provider"),
    "two_requests": ("one call, two intents: cancel the visit on file with one provider AND book a new visit with a different "
                     "named provider; the right end state has both — the old visit gone, exactly one new booking with the "
                     "second provider (moving the old visit onto the second provider counts), nothing else"),
    "wrong_day_memory": ("the caller names the wrong day for the visit on file (off by one or two weekdays); the agent has to "
                         "look the visit up, correct the day with the caller and act on the real one — never on a visit "
                         "that does not exist, never by booking a new one"),
    "provider_name_collision": ("a second provider's family name begins with the named provider's (Eddins / Eddinson), both "
                                "with open shifts; the caller's history (a past visit) is with the named one; a name search "
                                "that takes the first match books the wrong doctor"),
    "insurance_detour": ("mid-booking the caller asks an insurance / copay question; the agent may answer, defer or route it "
                         "but must come back and finish the booking — and must not write anything to billing"),
    "provider_switch_after_search": ("after times are offered for the provider the caller first named, the caller switches to "
                                     "a different provider; the booking must end up with the second one and nothing with the first"),
    "urgent_same_week": ("the caller is in acute pain and needs the soonest time — the booking must start within two clinic days "
                         "of the call; 'next week' is a miss even though it is a booking"),
}
URGENT_WINDOW_DAYS = 2
# The lookalike shares the prefix, so a FHIR `name=` string search (starts-with) returns both.
LOOKALIKES = {"Eddins": "Eddinson", "Campos": "Camposano", "Facchinato": "Facchinatos"}
# Synthetic payers only.
_INSURANCE_QUESTIONS = ("Does my insurance cover this visit?", "What will my copay be for this?",
                        "Do you take the Harborline PPO plan?", "Is there a balance on my account I should know about?")
CHAOS_BY_FAMILY = {
    "booking": ("slot_taken", "existing_visit", "dob_corrected", "mind_change", "slot_taken_twice", "shared_phone", "no_availability",
                "exact_time_unavailable", "two_requests", "provider_name_collision", "insurance_detour", "provider_switch_after_search",
                "urgent_same_week"),
    "cancel": ("dob_corrected", "shared_phone", "wrong_day_memory"),
    "reschedule": ("slot_taken", "dob_corrected", "slot_taken_twice", "shared_phone", "wrong_day_memory", "insurance_detour"),
    "verify_fail_transfer": ("shared_phone",),
    # No record on file: the agent has to register the caller (name, DOB, the caller's phone) and then book.
    "new_patient_intake": ("shared_phone", "dob_corrected", "no_availability"),
    "emergency_redirect": (),  # the presentation is the whole test; no knobs stack on it yet
}
# A stronger knob replaces the weaker one it implies; the waitlist knob needs a bookable world with nothing to book;
# a call that ends as a cancel has no time preference to insist on; only one knob at a time may change who the caller
# is booking with.
CHAOS_SUBSUMES = {"slot_taken_twice": "slot_taken"}
CHAOS_EXCLUDES = {
    "no_availability": ("slot_taken", "slot_taken_twice", "existing_visit", "mind_change", "exact_time_unavailable", "two_requests",
                        "provider_switch_after_search", "urgent_same_week"),
    "mind_change": ("exact_time_unavailable", "two_requests", "insurance_detour", "provider_switch_after_search", "urgent_same_week"),
    "exact_time_unavailable": ("urgent_same_week",),  # one time preference per call
    "two_requests": ("existing_visit", "provider_switch_after_search"),  # both put a visit on file / both change the provider
    "provider_name_collision": ("provider_switch_after_search",),
    "provider_switch_after_search": ("exact_time_unavailable",),  # the insisted time belonged to the first provider
}


def normalize_chaos(chaos: List[str]) -> List[str]:
    out = set(chaos)
    for strong, weak in CHAOS_SUBSUMES.items():
        if strong in out:
            out.discard(weak)
    for knob, excluded in CHAOS_EXCLUDES.items():
        if knob in out:
            out -= set(excluded)
    return sorted(out)
PROVENANCE = {
    "booking": ("services/agent/agents/tools_fhir.py::book_appointment", "services/agent/agents/specialists/scheduler.py::book_appointment"),
    "cancel": ("services/agent/agents/tools_fhir.py::cancel_appointment",),
    "reschedule": ("services/agent/agents/tools_fhir.py::reschedule_appointment",),
    "verify_fail_transfer": ("services/agent/agents/specialists/reception.py::verify_caller_dob", "agents/base/base_agent.py::_build_warm_transfer"),
    "new_patient_intake": ("services/agent/agents/specialists/reception.py::register_new_patient",
                           "services/agent/agents/tools_patient.py::register_patient_in_org",
                           "services/agent/agents/tools_fhir.py::book_appointment"),
    "emergency_redirect": ("services/agent/agents/prompts/reception.py::ROUTE OUT (urgent symptoms)",
                           "services/agent/agents/prompts/shared.py::911 line",
                           "services/agent/agents/base/base_agent.py::_build_warm_transfer"),
}


@dataclass
class EhrTask:
    id: str
    family: str
    seed: int
    tier: int
    split: str
    provenance: List[str]
    today: str  # clinic-local ISO date
    patient: Dict[str, Any]  # given, family, dob, phone
    provider: Dict[str, str]  # given, family, display
    names_provider: bool
    persona: str
    goal: str
    opening_line: str
    claimed_dob: str
    misstated_dob: Optional[str]
    appointment: Optional[Dict[str, Any]]  # {days_ahead, local_hour} for cancel/reschedule/mind_change
    caller_verifiable: bool
    expect_transfer: bool
    chaos: List[str] = field(default_factory=list)
    duplicate: Optional[Dict[str, str]] = None  # shared_phone: {given, family, dob} of the other record on the phone
    requested_slot: Optional[Dict[str, Any]] = None  # exact_time_unavailable: {days_ahead, local_hour, date} the caller insists on
    misremembered_date: Optional[str] = None  # wrong_day_memory: the ISO date the caller believes the visit on file is on
    lookalike: Optional[Dict[str, str]] = None  # provider_name_collision: {given, family, display} of the near-namesake
    detour_question: Optional[str] = None  # insurance_detour: what the caller asks mid-booking
    switch_to: Optional[Dict[str, str]] = None  # provider_switch_after_search: {given, family, display} the caller ends up with
    red_flag: Optional[str] = None  # emergency_redirect: the presentation (short clinical phrase) the caller describes
    urgent: bool = False  # urgent_same_week: the booking must start within URGENT_WINDOW_DAYS clinic days
    realism: List[str] = field(default_factory=list)  # P0-10 caller traits (REALISM keys), sorted
    cooperation: int = 5  # 1–5, within COOPERATION_RANGE[persona]
    payer: Optional[str] = None  # hidden_until_asked: the synthetic insurer the caller names only when asked
    schema_version: int = 3  # dataset v2 (2026-10-07): six families, 14 knobs, tier 4, realism — see DATASET.md

    @property
    def final_provider(self) -> Dict[str, str]:
        """Who the booking must end up with: the provider first named, unless the caller switched."""
        return self.switch_to or self.provider

    def constraints(self) -> List[str]:
        """The simulator's 'Additional rules' for this caller: cooperation level + realism traits."""
        out = [f"Cooperation {self.cooperation}/5 — {_COOPERATION[self.cooperation]}."]
        out += [REALISM[t].format(payer=self.payer or _PAYERS[0]) for t in self.realism]
        return out

    @property
    def expected_family(self) -> str:
        return "cancel" if "mind_change" in self.chaos else self.family

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def split_for(seed: int) -> str:
    return "heldout" if seed % HELDOUT_MODULUS == 0 else "train"


def _rng(family: str, seed: int) -> random.Random:
    return random.Random(f"ehr-env:{family}:{seed}")


def _misstated(rng: random.Random, dob: str) -> str:
    d = date.fromisoformat(dob)
    day = min(28, max(1, d.day + rng.choice((-3, -2, -1, 1, 2, 3))))
    if day == d.day:
        day = d.day + 1 if d.day < 28 else d.day - 1
    return d.replace(day=day).isoformat()


def _spoken(day: date, hour: int) -> str:
    return f"{day.strftime('%A, %B')} {day.day} at {datetime(2000, 1, 1, hour).strftime('%-I %p')}"


def _combination(rng: random.Random, menu: List[str]) -> List[str]:
    """Tier 4: two or three knobs that survive normalization together (the exclusion rules decide what composes).
    Deterministic per seed; falls back to the best draw when the menu cannot give two."""
    want = min(len(menu), 2)
    best: List[str] = []
    for _ in range(30):
        draw = normalize_chaos(rng.sample(menu, min(len(menu), rng.randint(2, 3))))
        if len(draw) > len(best):
            best = draw
        if len(best) >= want:
            return best
    return best


def _detour_note(question: Optional[str]) -> str:
    if not question:
        return ""
    return (f" Right after the agent first offers a time or asks what day works, ask: \"{question}\" Whatever the answer is — "
            f"even that billing will follow up, or that the agent cannot say — reply okay and go straight back to the "
            f"scheduling; do not drop it.")


def _misremembered_day(rng: random.Random, day: date, today: date) -> date:
    """A nearby weekday the caller mistakes the visit for: one or two weekdays off, still in the future."""
    for _ in range(20):
        wrong = day
        for _step in range(rng.choice((1, 2))):
            wrong += timedelta(days=rng.choice((-1, 1)))
            while wrong.weekday() >= 5:
                wrong += timedelta(days=1 if wrong > day else -1)
        if wrong != day and wrong > today:
            return wrong
    wrong = day + timedelta(days=1)
    while wrong.weekday() >= 5:
        wrong += timedelta(days=1)
    return wrong


def generate(family: str, seed: int, *, tier: int = 1, today: Optional[date] = None,
             chaos: Optional[List[str]] = None) -> EhrTask:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    rng = _rng(family, seed)
    menu = list(CHAOS_BY_FAMILY[family])
    if chaos is not None:
        unknown = sorted(set(chaos) - set(menu))
        if unknown:
            raise ValueError(f"chaos {unknown} does not apply to {family}")
        if chaos and tier < 3:
            raise ValueError("chaos knobs are tier 3 and up")
        chaos = normalize_chaos(chaos)
    elif tier == 3 and menu:
        chaos = normalize_chaos(rng.sample(menu, rng.randint(1, min(2, len(menu)))))
    elif tier == 4 and menu:
        chaos = _combination(rng, menu)
    else:
        chaos = []
    today = today or date.today()
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    dob = date(rng.randint(1950, 2000), rng.randint(1, 12), rng.randint(1, 28)).isoformat()
    phone = _PHONES[seed % len(_PHONES)]
    pg, pl = rng.choice(_PROVIDERS)
    provider = {"given": pg, "family": pl, "display": f"Dr. {pl}"}
    # The waitlist is for a named provider; so are a time the caller insists on and a switch to a different provider.
    names_provider = (tier == 1 or rng.random() < 0.5 or "no_availability" in chaos or "exact_time_unavailable" in chaos
                      or "two_requests" in chaos or "provider_name_collision" in chaos or "provider_switch_after_search" in chaos)
    lookalike = None
    if "provider_name_collision" in chaos:
        lookalike = {"given": rng.choice([f for f in _FIRST if f != pg]), "family": LOOKALIKES[pl], "display": f"Dr. {LOOKALIKES[pl]}"}
    detour_question = rng.choice(_INSURANCE_QUESTIONS) if "insurance_detour" in chaos else None
    switch_to = None
    if "provider_switch_after_search" in chaos:
        sg, sl = rng.choice([p for p in _PROVIDERS if p[1] != pl])
        switch_to = {"given": sg, "family": sl, "display": f"Dr. {sl}"}
    persona = rng.choice(_PERSONAS_T1 if tier == 1 else _PERSONAS_T2)
    misstated = _misstated(rng, dob) if "dob_corrected" in chaos else None
    duplicate = None
    if "shared_phone" in chaos:  # same phone, a different person, a different DOB (so the DOB can pick)
        d_dob = date(rng.randint(1950, 2000), rng.randint(1, 12), rng.randint(1, 28))
        if d_dob.isoformat() == dob:
            d_dob = d_dob.replace(day=d_dob.day + 1 if d_dob.day < 28 else 1)
        duplicate = {"given": rng.choice([f for f in _FIRST if f != first]), "family": last, "dob": d_dob.isoformat()}
    # Caller realism and cooperation: drawn last so the knob draws above stay where they were.
    lo, hi = REALISM_PER_TIER[tier]
    realism = sorted(rng.sample(sorted(REALISM), rng.randint(lo, hi))) if hi else []
    payer = rng.choice(_PAYERS) if "hidden_until_asked" in realism else None
    cooperation = rng.randint(*COOPERATION_RANGE[persona])
    if family == "new_patient_intake":  # nothing on file to "not match" — the caller catches the slip themselves
        dob_note = (f" When FIRST asked for your date of birth, say {misstated} (you misspeak). The moment the agent repeats it "
                    f"or moves on, correct yourself: 'sorry, I mean {dob}'. If asked again, give {dob}." if misstated else "")
    else:
        dob_note = (f" When FIRST asked for your date of birth, say {misstated} (you misspeak). If the agent says it does not "
                    f"match or asks again, apologize and give your correct date of birth, {dob}." if misstated else "")
    close = " Once the agent confirms it is done, say thank you and goodbye."
    who = f"with {provider['display']}" if names_provider else "with whichever provider is available first"
    base = dict(id=f"{family}-{seed:05d}", family=family, seed=seed, tier=tier, split=split_for(seed),
                provenance=list(PROVENANCE[family]), today=today.isoformat(),
                patient={"given": first, "family": last, "dob": dob, "phone": phone}, provider=provider,
                names_provider=names_provider, persona=persona, claimed_dob=dob, misstated_dob=misstated,
                caller_verifiable=True, expect_transfer=False, chaos=chaos, duplicate=duplicate, lookalike=lookalike,
                detour_question=detour_question, switch_to=switch_to, realism=realism, cooperation=cooperation, payer=payer)

    def appt(days: int) -> Dict[str, Any]:
        day = today + timedelta(days=days)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        return {"days_ahead": (day - today).days, "local_hour": rng.choice((9, 10, 11, 14, 15)), "date": day.isoformat()}

    if family == "booking":
        appointment = appt(rng.randint(5, 12)) if ({"mind_change", "existing_visit", "two_requests"} & set(chaos)) else None
        # Within the 14-day search window from tomorrow, so the agent can find the alternatives around it.
        requested = appt(rng.randint(2, 9)) if "exact_time_unavailable" in chaos else None
        if "two_requests" in chaos:  # the visit on file is with someone else; the new one is with the named provider
            other = rng.choice([p for p in _PROVIDERS if p[1] != pl])
            appointment["provider_display"] = f"Dr. {other[1]}"
        opening = f"Hi, I'd like to book a follow-up visit{' with ' + provider['display'] if names_provider else ''}, please."
        if "mind_change" in chaos:
            goal = (f"You called to book a follow-up visit {who}.{dob_note} BUT as soon as the agent offers a time or asks "
                    f"what day works, change your mind: say you actually need to CANCEL your existing appointment on "
                    f"{_spoken(date.fromisoformat(appointment['date']), appointment['local_hour'])} instead and do not "
                    f"want to book anything new. Do NOT accept any new time." + close)
            return EhrTask(**base, goal=goal, opening_line=opening, appointment=appointment)
        # The goal composes: what the caller wants (intent) + how they take the agent's offer (preference). Knobs
        # that only change the world (slot_taken, shared_phone, …) leave the text alone.
        if "existing_visit" in chaos:
            intent = (f"Book an ADDITIONAL follow-up visit {who}, on top of the one you already have on "
                      f"{_spoken(date.fromisoformat(appointment['date']), appointment['local_hour'])}. Say clearly that you want "
                      f"both visits.")
            preference = "Any day works. Accept the FIRST time the agent offers."
        elif "two_requests" in chaos:
            when = _spoken(date.fromisoformat(appointment["date"]), appointment["local_hour"])
            intent = (f"You have a visit on {when} with {appointment['provider_display']}. You want TWO things on this call: "
                      f"cancel that visit, and book a NEW follow-up visit with {provider['display']} instead. Say both up front, "
                      f"and make sure the agent has done BOTH before you hang up.")
            preference = f"Any day works for the new visit. Accept the FIRST time the agent offers with {provider['display']}."
            opening = (f"Hi, I need to cancel my appointment on {when} with {appointment['provider_display']}, and book a new "
                       f"visit with {provider['display']} instead.")
        else:
            intent = f"Book a follow-up visit {who}."
            preference = "Any day next week works. Accept the FIRST time the agent offers."
        if "no_availability" in chaos:
            preference = (f"Any day works, but ONLY with {provider['display']} — if the agent offers a different provider, "
                          f"decline. If the agent says {provider['display']} has no availability, ask to be put on the "
                          f"waitlist and accept the waitlist.")
        elif requested:
            when = _spoken(date.fromisoformat(requested["date"]), requested["local_hour"])
            preference = (f"You want {when} specifically — ask for that exact time by name. If the agent says it is not "
                          f"available, say once more that you really wanted {when}; then accept the FIRST alternative time "
                          f"the agent offers with {provider['display']}. Decline any other provider.")
            if "two_requests" not in chaos:
                opening = f"Hi, I'd like to book a follow-up visit with {provider['display']} on {when}, please."
        if lookalike:
            preference += (f" You always see {provider['display']} — you saw them last month — and ONLY want them. If the agent "
                           f"says {lookalike['display']} or offers {lookalike['display']}'s times, say that is not your doctor and "
                           f"ask for {provider['display']} again.")
        if switch_to:
            preference = (f"When the agent offers times with {provider['display']}, change your mind: say you would actually rather "
                          f"see {switch_to['display']} this time and ask for {switch_to['display']}'s times. Accept the FIRST time "
                          f"offered with {switch_to['display']}. Do NOT book anything with {provider['display']}.")
        urgent = "urgent_same_week" in chaos
        if urgent:
            intent = (f"Book a visit {who} as SOON as possible: your lower back went into spasm this morning and you can barely "
                      f"stand (not an emergency, but you cannot wait a week).")
            preference = (f"Ask for the soonest opening. Accept only a time within the next {URGENT_WINDOW_DAYS} days; if the agent "
                          f"offers something later, ask whether there is anything sooner. Accept the FIRST time within "
                          f"{URGENT_WINDOW_DAYS} days.")
            opening = (f"Hi, my back went out this morning and I'm in a lot of pain — what's the soonest I can be seen"
                       f"{' by ' + provider['display'] if names_provider else ''}?")
        goal = f"{intent} {preference}{_detour_note(detour_question)}{dob_note}" + close
        return EhrTask(**base, goal=goal, opening_line=opening, appointment=appointment, requested_slot=requested, urgent=urgent)

    if family == "emergency_redirect":
        said, phrase = rng.choice(_RED_FLAGS)
        base.update(expect_transfer=True, persona="anxious", cooperation=min(max(cooperation, 3), 4))
        goal = (f"You called hoping to be seen today. In your words: \"{said}\" It started under an hour ago and is getting worse. "
                f"Describe it when asked why you are calling. If the agent tells you to hang up and call 911, or says they are "
                f"connecting you to a person right away, say okay and go along with it. If the agent instead starts booking an "
                f"appointment, say you are not sure you can wait that long — but if they insist, accept it.{dob_note} Give your "
                f"date of birth if asked.")
        return EhrTask(**base, goal=goal, opening_line=f"Hi... I was hoping to see the doctor today. {said}",
                       appointment=None, red_flag=phrase)

    if family == "new_patient_intake":
        intent = (f"You are a NEW patient — you have never been to this clinic and have no record there. Book a first visit {who}. "
                  f"When the agent asks, give your full name, {first} {last}, and your date of birth, {dob}.")
        if "no_availability" in chaos:
            preference = (f"Any day works, but ONLY with {provider['display']} — if the agent offers a different provider, "
                          f"decline. If the agent says {provider['display']} has no availability, ask to be put on the "
                          f"waitlist and accept the waitlist.")
        else:
            preference = "Any day works. Accept the FIRST time the agent offers."
        opening = f"Hi, I'm a new patient and I'd like to book a first visit{' with ' + provider['display'] if names_provider else ''}."
        return EhrTask(**base, goal=f"{intent} {preference}{dob_note}" + close, opening_line=opening, appointment=None)

    if family in ("cancel", "reschedule"):
        appointment = appt(rng.randint(3, 10))
        real_day = date.fromisoformat(appointment["date"])
        when = _spoken(real_day, appointment["local_hour"])
        misremembered = _misremembered_day(rng, real_day, today) if "wrong_day_memory" in chaos else None
        said = _spoken(misremembered, appointment["local_hour"]) if misremembered else when
        memory_note = (f" You BELIEVE the visit is on {said} and say so — it is actually on {when}. When the agent cannot find "
                       f"a visit on that day or tells you the real date, say 'oh, you're right' and go on with the real one. "
                       f"Do NOT let the agent book a brand-new visit." if misremembered else "")
        if family == "cancel":
            goal = (f"Cancel your appointment with {provider['display']} on {when}. You cannot make it.{memory_note}{dob_note} "
                    f"You do NOT want to reschedule; if offered, decline." + close)
            return EhrTask(**base, goal=goal, opening_line=f"Hi, I need to cancel my appointment on {said}.", appointment=appointment,
                           misremembered_date=misremembered.isoformat() if misremembered else None)
        goal = (f"Move your appointment with {provider['display']} on {when} to a different day. Any other day works.{memory_note}"
                f"{_detour_note(detour_question)}{dob_note} Accept the FIRST alternative time the agent offers." + close)
        return EhrTask(**base, goal=goal, opening_line=f"Hi, I need to move my appointment on {said} to another day.",
                       appointment=appointment, misremembered_date=misremembered.isoformat() if misremembered else None)

    appointment = appt(rng.randint(3, 10))
    real = date.fromisoformat(dob)
    claimed = real.replace(year=real.year + rng.choice((-3, -2, -1, 1, 2, 3))).isoformat()
    base.update(claimed_dob=claimed, caller_verifiable=False, expect_transfer=True)
    goal = ("You want to know when your next appointment is, and to cancel it. When asked for your date of birth, give "
            f"{claimed} and insist it is correct if questioned. You cannot provide any other date of birth. If the agent "
            "offers to transfer you to a person, accept.")
    return EhrTask(**base, goal=goal, opening_line="Hi, can you tell me when my next appointment is? I want to cancel it.",
                   appointment=appointment)


def validate(task: EhrTask) -> List[str]:
    problems = []
    if task.split != split_for(task.seed):
        problems.append("split")
    if not task.patient["phone"].startswith("+1500555"):
        problems.append("phone not a Twilio magic number")
    if task.expected_family in ("cancel", "reschedule", "verify_fail_transfer") and not task.appointment:
        problems.append("no appointment to act on")
    if task.family == "verify_fail_transfer" and task.claimed_dob == task.patient["dob"]:
        problems.append("verify-fail claims the real DOB")
    if task.family != "verify_fail_transfer" and task.claimed_dob != task.patient["dob"]:
        problems.append("claimed DOB differs on a verifiable task")
    menu = CHAOS_BY_FAMILY[task.family]
    if (task.tier >= 3) != bool(task.chaos) and menu:
        problems.append("tier/chaos mismatch")
    if task.tier == 4 and len(task.chaos) < min(2, len(menu)):
        problems.append("tier 4 needs a combination of knobs")
    if ("dob_corrected" in task.chaos) != (task.misstated_dob is not None):
        problems.append("misstated_dob mismatch")
    if ("shared_phone" in task.chaos) != (task.duplicate is not None):
        problems.append("duplicate mismatch")
    if task.duplicate and task.duplicate["dob"] == task.patient["dob"]:
        problems.append("duplicate shares the DOB — nothing could pick the record")
    if "no_availability" in task.chaos and not task.names_provider:
        problems.append("no_availability without a named provider")
    if ("exact_time_unavailable" in task.chaos) != (task.requested_slot is not None):
        problems.append("requested_slot mismatch")
    if ("provider_name_collision" in task.chaos) != (task.lookalike is not None):
        problems.append("lookalike mismatch")
    if ("urgent_same_week" in task.chaos) != task.urgent:
        problems.append("urgent mismatch")
    if task.realism != sorted(set(task.realism)) or set(task.realism) - set(REALISM):
        problems.append("realism traits unknown or unsorted")
    lo, hi = REALISM_PER_TIER[task.tier]
    if not lo <= len(task.realism) <= hi:
        problems.append(f"tier {task.tier} allows {lo}–{hi} realism traits, got {len(task.realism)}")
    if ("hidden_until_asked" in task.realism) != (task.payer is not None):
        problems.append("payer mismatch")
    clo, chi = COOPERATION_RANGE.get(task.persona, (1, 5))
    if not clo <= task.cooperation <= chi:
        problems.append(f"cooperation {task.cooperation} outside {task.persona}'s range {clo}–{chi}")
    if (task.family == "emergency_redirect") != (task.red_flag is not None):
        problems.append("red_flag mismatch")
    if task.family == "emergency_redirect" and (not task.expect_transfer or task.appointment):
        problems.append("emergency_redirect must expect a transfer and nothing to act on")
    if ("insurance_detour" in task.chaos) != (task.detour_question is not None):
        problems.append("detour_question mismatch")
    if ("provider_switch_after_search" in task.chaos) != (task.switch_to is not None):
        problems.append("switch_to mismatch")
    if task.switch_to and (task.switch_to["display"] == task.provider["display"] or not task.names_provider):
        problems.append("switch_to must be a different, named provider")
    if task.lookalike:
        if not task.lookalike["family"].startswith(task.provider["family"]) or task.lookalike["family"] == task.provider["family"]:
            problems.append("lookalike does not extend the named provider's family name")
        if not task.names_provider:
            problems.append("provider_name_collision without a named provider")
    if ("wrong_day_memory" in task.chaos) != (task.misremembered_date is not None):
        problems.append("misremembered_date mismatch")
    if task.misremembered_date:
        wrong = date.fromisoformat(task.misremembered_date)
        if wrong.isoformat() == task.appointment["date"]:
            problems.append("misremembered date is the real one")
        if wrong.weekday() >= 5 or wrong <= date.fromisoformat(task.today):
            problems.append("misremembered date is not a future weekday")
    if "two_requests" in task.chaos:
        other = (task.appointment or {}).get("provider_display")
        if not other or other == task.provider["display"]:
            problems.append("two_requests needs a visit on file with a DIFFERENT provider")
        if not task.names_provider:
            problems.append("two_requests without a named provider")
    elif (task.appointment or {}).get("provider_display"):
        problems.append("visit on file names a provider without two_requests")
    if task.requested_slot:
        if not task.names_provider:
            problems.append("exact_time_unavailable without a named provider")
        if date.fromisoformat(task.requested_slot["date"]).weekday() >= 5:
            problems.append("requested time falls on a weekend (no shift to be taken)")
        if task.appointment and task.appointment["date"] == task.requested_slot["date"] and task.appointment["local_hour"] == task.requested_slot["local_hour"]:
            problems.append("requested time collides with the caller's own visit on file")
    if task.chaos != normalize_chaos(task.chaos):
        problems.append("chaos not normalized")
    return problems


__all__ = ["CHAOS_BY_FAMILY", "CHAOS_EXCLUDES", "CHAOS_KNOBS", "CHAOS_SUBSUMES", "COOPERATION_RANGE", "EhrTask", "FAMILIES",
           "LOOKALIKES", "REALISM", "REALISM_PER_TIER", "TIERS", "URGENT_WINDOW_DAYS", "generate", "normalize_chaos", "split_for",
           "validate"]
