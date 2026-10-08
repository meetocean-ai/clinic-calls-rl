"""Snapshots of an episode's tenant for the dataset package (roadmap P2-08): the compartment as a FHIR Bundle plus the
world facts the verifier needs (ids the policy never sees, the baseline counts, the clinic day). Taken before the
call for `snapshots/<task_id>.json`, after the call for the reference run's end state.

    snap = await snapshot_world(env.world)            # {"task", "world", "bundle"}
    end = await export_bundle(env.medplum, env.world.tenant)
    verify_json(snap, end, agent_turns, extra) == env.result   (env/dataset/test_verifier.py proves it)
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..tasks import URGENT_WINDOW_DAYS
from ..world import CLINIC_TZ, World

# The product's tenant export list (services/agent/hatchet/workflows/tenant_export.py::_EXPORT_RESOURCE_TYPES); copied so
# the dataset code does not import the Hatchet workflows. env/dataset/test_verifier.py asserts the two lists agree.
COMPARTMENT_TYPES = (
    "Patient", "Practitioner", "PractitionerRole", "Organization", "Appointment", "Encounter", "EpisodeOfCare",
    "ClinicalImpression", "Observation", "Condition", "Procedure", "CarePlan", "Coverage", "Claim", "ClaimResponse",
    "Communication", "CommunicationRequest", "Consent", "DocumentReference", "Task", "AuditEvent",
)

# Tenant snapshots are data, not a log: AuditEvents grow with every read and would make two identical end states differ.
SNAPSHOT_TYPES = tuple(t for t in COMPARTMENT_TYPES if t != "AuditEvent")

# Fields that differ between two otherwise identical exports (server bookkeeping), dropped from the bundle.
PAGE = 1000  # Medplum's maximum _count
_VOLATILE = ("meta",)


def _clean(resource: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in resource.items() if k not in _VOLATILE}


async def export_bundle(medplum, tenant, *, types=SNAPSHOT_TYPES) -> Dict[str, Any]:
    """Every resource of the given types in the tenant's compartment, as a FHIR `collection` Bundle."""
    entries: List[Dict[str, Any]] = []
    for rt in types:
        rows = await medplum.search(rt, tenant_id=tenant.organization_id, params={"_count": str(PAGE), "_sort": "_id"})
        if len(rows) >= PAGE:  # an episode tenant holds a few dozen resources; a full page means the export is truncated
            raise RuntimeError(f"export of {rt} hit the page size ({PAGE}) — page the export before trusting this snapshot")
        entries += [{"resource": _clean(r)} for r in rows]
    return {"resourceType": "Bundle", "type": "collection", "total": len(entries), "entry": entries,
            "compartment": f"Organization/{tenant.organization_id}", "types": list(types)}


def world_meta(world: World) -> Dict[str, Any]:
    """What `verifier.Episode` reads from `snapshot["world"]`."""
    b = world.baseline
    return {
        "organization_id": world.tenant.organization_id,
        "patient_id": world.patient.patient_id,  # None for new_patient_intake: the caller has no record yet
        "duplicate_patient_id": world.duplicate_patient_id,
        "bystander_patient_id": world.bystander_patient_id,
        "requested_bystander_id": world.requested_bystander_id,
        "requested_start": world.requested_start,
        "target_appointment_id": world.target_appointment_id,
        "practitioners": {k: dict(v) for k, v in world.practitioners.items()},
        "baseline": {"appointments": {k: dict(v) for k, v in b.appointments.items()}, "n_appointments": b.n_appointments,
                     "n_patients": b.n_patients, "n_tasks": b.n_tasks, "n_transfers": b.n_transfers, "n_coverage": b.n_coverage},
        "clinic_tz": CLINIC_TZ,
        "today": world.task.today,
        "urgent_window_days": URGENT_WINDOW_DAYS,
    }


async def snapshot_world(world: World) -> Dict[str, Any]:
    return {"schema_version": 1, "task": world.task.to_dict(), "world": world_meta(world),
            "bundle": await export_bundle(world.medplum, world.tenant)}


def runtime_extra(world: World) -> Dict[str, Any]:
    """The runtime-only facts the verifier takes as `extra`."""
    return {"verify_attempts": int(world.extra.get("verify_attempts", 0)),
            "env_writes": dict(world.extra.get("env_writes") or {}),
            **({"created_patient_id": world.extra["created_patient_id"]} if world.extra.get("created_patient_id") else {})}


__all__ = ["COMPARTMENT_TYPES", "SNAPSHOT_TYPES", "export_bundle", "runtime_extra", "snapshot_world", "world_meta"]
