"""Seed the world for one episode on the local Medplum (E2/E3): a fresh Organization, the providers
with working-hours Schedules, the patient the task describes, and the appointment on file when the
family needs one. Everything goes through the tenant-scoped adapter and the same fixtures the
behavioral scenarios use, so the environment and the scenarios agree on what a clinic looks like.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional

from fixtures.patients import PatientFixtureResult, _create_patient, appointment_start_on_day, clinic_local_now
from fixtures.practitioners import chiropractor
from runtime.end_state import appointments_for, transfer_tasks
from runtime.patient_simulator import PatientProfile
from runtime.tenant_lifecycle import provision_test_tenant, teardown_test_tenant

from .tasks import EhrTask

CLINIC_TZ = "America/Los_Angeles"  # provision_test_tenant's clinic timezone


@dataclass
class Baseline:
    appointments: Dict[str, Dict[str, Any]]  # id -> {status, start}
    n_appointments: int
    n_patients: int
    n_tasks: int
    n_transfers: int
    n_coverage: int = 0


@dataclass
class World:
    tenant: Any
    medplum: Any
    task: EhrTask
    patient: PatientFixtureResult
    practitioners: Dict[str, Dict[str, str]]  # display -> {id, display}
    target_appointment_id: Optional[str]
    baseline: Baseline
    bystander_patient_id: Optional[str] = None
    duplicate_patient_id: Optional[str] = None  # shared_phone: the other record on the caller's number
    requested_start: Optional[str] = None  # exact_time_unavailable: the instant the caller insists on (ISO, clinic offset)
    requested_bystander_id: Optional[str] = None  # … and whose visit it is (mid-episode slot_taken bystanders are separate)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def patient_id(self) -> Optional[str]:
        """The caller's record — seeded, or (new_patient_intake) the one the policy registered, once known."""
        return self.patient.patient_id or self.extra.get("created_patient_id")

    async def resolve_patient_id(self) -> Optional[str]:
        """new_patient_intake: find the record the policy created for the caller — their name and DOB on the caller's
        number, not the other person who may share it. Remembers it in `extra` so every check reads the same record."""
        if self.patient_id:
            return self.patient_id
        p = self.task.patient
        rows = await self.medplum.search("Patient", tenant_id=self.tenant.organization_id,
                                         params={"phone": p["phone"], "_count": "20"})
        mine = [r for r in rows if r.get("id") != self.duplicate_patient_id and _patient_is(r, p)]
        if len(mine) == 1:
            self.extra["created_patient_id"] = mine[0]["id"]
            return mine[0]["id"]
        return None

    @property
    def provider_id(self) -> Optional[str]:
        """The provider the booking must end up with (after a mid-call switch, the second one)."""
        return self.practitioners[self.task.final_provider["display"]]["id"] if self.task.names_provider else None

    async def counts(self) -> Dict[str, int]:
        out = {}
        for rt in ("Appointment", "Patient", "Task", "Coverage"):
            rows = await self.medplum.search(rt, tenant_id=self.tenant.organization_id, params={"_count": "200"})
            out[rt] = len(rows)
        return out

    async def close(self) -> None:
        try:
            await teardown_test_tenant(self.tenant)
        except Exception:  # noqa: BLE001 - teardown is best effort; orphans are swept
            pass


def _patient_is(resource: Dict[str, Any], p: Dict[str, Any]) -> bool:
    """A Patient resource carries the task's name and DOB (case-insensitive)."""
    for n in resource.get("name") or []:
        given = " ".join(n.get("given") or []).strip().lower()
        if given == p["given"].lower() and (n.get("family") or "").strip().lower() == p["family"].lower():
            return resource.get("birthDate") == p["dob"]
    return False


def _participants(patient_id: str, patient_display: str, practitioner_id: str, practitioner_display: str) -> list:
    """Shaped like the product's own bookings: the actor's `display` is what `get_my_appointments` reads back as
    the provider / patient name, so a seeded visit must carry it too."""
    return [{"actor": {"reference": f"Patient/{patient_id}", "display": patient_display}, "status": "accepted"},
            {"actor": {"reference": f"Practitioner/{practitioner_id}", "display": practitioner_display}, "status": "accepted"}]


async def book_bystander(medplum, tenant, *, practitioner_id: str, start, end, n: int = 1, practitioner_display: str = "") -> str:
    """Another patient's booking at a given slot — the environment's doing, never the policy's. The phone is
    outside the task range (+1500555 0001–0014) so a bystander never shares the caller's number by accident.
    Returns the bystander's patient id."""
    by = await medplum.create_patient(tenant_id=tenant.organization_id, given=["Bystander"], family=f"Booked{n}",
                                      birth_date="1980-01-01", phone=f"+150055599{n:02d}", email=f"bystander{n}@example.com")
    await medplum.create("Appointment", {
        "resourceType": "Appointment", "status": "booked", "description": "Follow-up",
        "start": start.isoformat(), "end": end.isoformat(),
        "participant": _participants(by["id"], f"Bystander Booked{n}", practitioner_id, practitioner_display),
    }, tenant_id=tenant.organization_id)
    return by["id"]


async def seed_world(task: EhrTask, medplum) -> World:
    tenant = await provision_test_tenant(name_suffix=f"env-{task.id}")
    practitioners: Dict[str, Dict[str, str]] = {}

    def _bookable(display: str) -> bool:
        # no_availability: the named provider exists but has no Schedule → the slot search finds nothing for them
        return not ("no_availability" in task.chaos and display == task.provider["display"])

    for given, family in (("Robin", "Eddins"), ("Ana", "Campos"), ("Jo", "Facchinato")):
        res = await chiropractor(tenant=tenant, medplum=medplum, given=[given], family=family, with_schedule=_bookable(f"Dr. {family}"))
        practitioners[res.display_name] = {"id": res.practitioner_id, "display": res.display_name}
    if task.provider["display"] not in practitioners:
        res = await chiropractor(tenant=tenant, medplum=medplum, given=[task.provider["given"]], family=task.provider["family"],
                                 with_schedule=_bookable(task.provider["display"]))
        practitioners[res.display_name] = {"id": res.practitioner_id, "display": res.display_name}
    if task.lookalike:  # provider_name_collision: the near-namesake works here too, with open shifts
        res = await chiropractor(tenant=tenant, medplum=medplum, given=[task.lookalike["given"]], family=task.lookalike["family"])
        practitioners[res.display_name] = {"id": res.practitioner_id, "display": res.display_name}

    p = task.patient
    duplicate_id = None
    if task.duplicate:  # shared_phone: the other record is created FIRST, so a first-match lookup meets it first
        d = task.duplicate
        dup = await _create_patient(
            tenant=tenant, medplum=medplum, seed_name=f"env-{task.id}-dup",
            given_override=[d["given"]], family_override=d["family"], dob_override=d["dob"], phone_override=p["phone"],
            persona="polite", goal="", last_visit_days_ago=60,
        )
        duplicate_id = dup.patient_id
    if task.family == "new_patient_intake":  # no record: the caller exists only as the simulator's profile
        patient = PatientFixtureResult(patient_id=None, profile=PatientProfile(
            name=f"{p['given']} {p['family']}", dob=p["dob"], phone=p["phone"], persona=task.persona, goal=task.goal,
            constraints=task.constraints(), has_prior_visits=False, clinic_name=tenant.organization_name,
            today=clinic_local_now(tenant.timezone).strftime("%A, %B %-d, %Y")))
    else:
        patient = await _create_patient(
            tenant=tenant, medplum=medplum, seed_name=f"env-{task.id}",
            given_override=[p["given"]], family_override=p["family"], dob_override=p["dob"], phone_override=p["phone"],
            persona=task.persona, goal=task.goal, constraints=task.constraints(), last_visit_days_ago=30,
        )
    patient.profile.opening_line = task.opening_line
    patient.profile.dob = task.claimed_dob  # what the caller SAYS; the record holds task.patient["dob"]

    target_id = None
    if task.appointment:
        start = appointment_start_on_day(tenant.timezone, days_ahead=task.appointment["days_ahead"],
                                         local_hour=task.appointment["local_hour"])
        end = start + timedelta(minutes=30)
        # two_requests: the visit on file is with someone other than the provider the caller wants next
        prov = practitioners[task.appointment.get("provider_display") or task.provider["display"]]
        appt = await medplum.create("Appointment", {
            "resourceType": "Appointment", "status": "booked", "description": "Follow-up",
            "start": start.isoformat(), "end": end.isoformat(),
            "participant": _participants(patient.patient_id, f"{p['given']} {p['family']}", prov["id"], prov["display"]),
        }, tenant_id=tenant.organization_id)
        target_id = appt["id"]
        patient.profile.appointments_on_file = [{
            "id": target_id, "date": start.strftime("%A, %B %-d, %Y"), "time": start.strftime("%-I:%M %p"),
            "start": start.isoformat(), "provider": prov["display"], "type": "Follow-up",
        }]
        patient.profile.has_prior_visits = True

    if task.lookalike:  # the caller's history: a past visit with the named provider, none with the namesake
        prov = practitioners[task.provider["display"]]
        past = appointment_start_on_day(tenant.timezone, days_ahead=-30, local_hour=10)
        await medplum.create("Appointment", {
            "resourceType": "Appointment", "status": "fulfilled", "description": "Follow-up",
            "start": past.isoformat(), "end": (past + timedelta(minutes=30)).isoformat(),
            "participant": _participants(patient.patient_id, f"{p['given']} {p['family']}", prov["id"], prov["display"]),
        }, tenant_id=tenant.organization_id)

    bystander_id, requested_start = None, None
    if task.requested_slot:  # exact_time_unavailable: the time the caller wants is already someone else's
        start = appointment_start_on_day(tenant.timezone, days_ahead=task.requested_slot["days_ahead"],
                                         local_hour=task.requested_slot["local_hour"])
        prov = practitioners[task.provider["display"]]
        bystander_id = await book_bystander(medplum, tenant, practitioner_id=prov["id"], start=start,
                                            end=start + timedelta(minutes=30), practitioner_display=prov["display"])
        requested_start = start.isoformat()

    appts = await appointments_for(medplum, tenant, patient.patient_id) if patient.patient_id else []
    transfers = await transfer_tasks(medplum, tenant)
    world = World(tenant=tenant, medplum=medplum, task=task, patient=patient, practitioners=practitioners,
                  target_appointment_id=target_id, duplicate_patient_id=duplicate_id,
                  bystander_patient_id=bystander_id, requested_start=requested_start, requested_bystander_id=bystander_id,
                  baseline=Baseline(appointments={a["id"]: {"status": a.get("status"), "start": a.get("start")} for a in appts},
                                    n_appointments=0, n_patients=0, n_tasks=0, n_transfers=len(transfers)))
    counts = await world.counts()
    world.baseline.n_appointments, world.baseline.n_patients, world.baseline.n_tasks, world.baseline.n_coverage = (
        counts["Appointment"], counts["Patient"], counts["Task"], counts["Coverage"])
    return world


def clinic_today():
    return clinic_local_now(CLINIC_TZ).date()


__all__ = ["Baseline", "CLINIC_TZ", "World", "book_bystander", "clinic_today", "seed_world"]
