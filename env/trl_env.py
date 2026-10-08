"""The EHR scheduling environment in TRL's `environment_factory` shape (roadmap P1-09): a class whose public methods
are the tools and whose `reset(**kwargs)` opens an episode on the OpenEnv server. `GRPOTrainer(environment_factory=
EhrToolEnv, reward_funcs=[episode_reward])` then drives the multi-turn loop itself — generation, tool-call parsing,
execution, feeding results back — and reads the environment's verify() reward when the episode is over.

Imports only the OpenEnv client side (`env.server` client classes), so it runs in the training venv
(`tests/behavioral/.venv-verifiers`), not the agent's. The server runs separately (`python -m env.server`).

    ENV_URL = ws://127.0.0.1:8011   (EHR_ENV_URL)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

ENV_URL = os.environ.get("EHR_ENV_URL", "ws://127.0.0.1:8011")


def _client():
    from env.client import EhrEnvClient

    return EhrEnvClient(base_url=ENV_URL, message_timeout_s=300.0).sync()


class EhrToolEnv:
    """One instance per generation. Dataset columns `family`, `seed`, `tier` (and optional `chaos`) select the task."""

    def __init__(self) -> None:
        self.client = None
        self.reward: float = 0.0
        self.done: bool = False
        self.diffs: List[str] = []
        self.task_id: str = ""
        self.steps: int = 0

    # ---- episode -------------------------------------------------------------

    def reset(self, **kwargs: Any) -> Optional[str]:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
        self.client = _client()
        self.client.__enter__()
        self.reward, self.done, self.diffs, self.steps = 0.0, False, [], 0
        first = self.client.reset(seed=int(kwargs.get("seed", 1)), family=kwargs.get("family", "booking"),
                                  tier=int(kwargs.get("tier", 1)), chaos=kwargs.get("chaos") or None)
        obs = first.observation
        self.task_id = obs.task_id
        import json

        return (f"Caller record: {json.dumps(obs.patient_info, default=str)}\n\n[call connected] Caller: {obs.patient_text}\n\n"
                f"Speak to the caller with say(text). Verify them before reading or changing anything. End with end_call().")

    def _step(self, kind: str, tool_name: str = "", **arguments: Any) -> Dict[str, Any]:
        from env.client import EhrAction

        if self.done:
            raise ValueError("The call is over.")
        result = self.client.step(EhrAction(kind=kind, text=arguments.pop("text", ""), tool_name=tool_name, arguments=arguments))
        self.steps += 1
        obs = result.observation
        if obs.done:
            self.done = True
            self.reward = float(obs.reward or 0.0)
            self.diffs = list(obs.diffs)
        return obs.model_dump()

    def _finish(self) -> float:
        """Score whatever state the episode reached if the model stopped without ending the call."""
        if not self.done and self.client is not None:
            try:
                self._step("end_call")
            except Exception:  # noqa: BLE001
                pass
        return self.reward

    # ---- tools (the trainer reads the docstrings for the schemas) -------------

    def say(self, text: str) -> str:
        """
        Say something to the caller and hear their reply.

        Args:
            text: What you say to the caller, one or two short sentences.

        Returns:
            The caller's reply.
        """
        return f"Caller: {self._step('say', text=text)['patient_text']}"

    def verify_patient(self, dob: str) -> str:
        """
        Verify the caller against their record by the date of birth they said on this call.

        Args:
            dob: The date of birth the caller said, as YYYY-MM-DD.

        Returns:
            Whether the caller is verified, and what to do next.
        """
        return str(self._step("tool", "verify_patient", dob=dob)["tool_result"])

    def create_patient(self, first_name: str, last_name: str, dob: str) -> str:
        """
        Register a caller who has no record, after they gave their full name and date of birth.

        Args:
            first_name: The caller's first name as they said it.
            last_name: The caller's last name as they said it.
            dob: The caller's date of birth, YYYY-MM-DD.

        Returns:
            Whether the record was created.
        """
        return str(self._step("tool", "create_patient", first_name=first_name, last_name=last_name, dob=dob)["tool_result"])

    def look_up_availability(self, provider_id: str = "", date: str = "") -> str:
        """
        Find open appointment slots.

        Args:
            provider_id: The provider's id to search for; empty for any provider.
            date: YYYY-MM-DD to start the search from; empty for tomorrow.

        Returns:
            The open slots with their slot_index, start time, provider and provider_id.
        """
        return str(self._step("tool", "look_up_availability", provider_id=provider_id or None, date=date or None)["tool_result"])

    def confirm_appointment(self, slot_index: int, additional_visit: bool = False) -> str:
        """
        Book an offered slot after the caller agreed to it.

        Args:
            slot_index: The slot_index from look_up_availability.
            additional_visit: True when the caller wants this on top of a visit they already have with the provider.

        Returns:
            Whether the appointment was booked.
        """
        return str(self._step("tool", "confirm_appointment", slot_index=int(slot_index), additional_visit=bool(additional_visit))["tool_result"])

    def get_appointments(self) -> str:
        """
        List the caller's upcoming appointments.

        Returns:
            Each appointment's appointment_id, start, provider and provider_id.
        """
        return str(self._step("tool", "get_appointments")["tool_result"])

    def cancel_appointment(self, appointment_id: str, reason: str = "patient request") -> str:
        """
        Cancel one of the caller's appointments.

        Args:
            appointment_id: The appointment_id from get_appointments.
            reason: Why the caller is cancelling.

        Returns:
            Whether it was cancelled.
        """
        return str(self._step("tool", "cancel_appointment", appointment_id=appointment_id, reason=reason)["tool_result"])

    def reschedule_appointment(self, appointment_id: str, slot_index: int) -> str:
        """
        Move one of the caller's appointments to an offered slot (search first with look_up_availability).

        Args:
            appointment_id: The appointment_id from get_appointments.
            slot_index: The slot_index from look_up_availability.

        Returns:
            Whether it was moved.
        """
        return str(self._step("tool", "reschedule_appointment", appointment_id=appointment_id, slot_index=int(slot_index))["tool_result"])

    def enroll_in_waitlist(self, provider_id: str = "", notes: str = "") -> str:
        """
        Put the caller on the waitlist for a provider who has nothing open.

        Args:
            provider_id: The provider's id.
            notes: Anything the caller asked to note.

        Returns:
            Whether the caller was waitlisted.
        """
        return str(self._step("tool", "enroll_in_waitlist", provider_id=provider_id or None, notes=notes)["tool_result"])

    def warm_transfer_to_human(self, reason: str = "") -> str:
        """
        Hand the call to a staff member; this ends the call. Say 'urgent' in the reason for a medical emergency.

        Args:
            reason: Why the caller is being transferred.

        Returns:
            Confirmation that the call was transferred.
        """
        return str(self._step("tool", "warm_transfer_to_human", reason=reason)["tool_result"])

    def end_call(self) -> str:
        """
        End the call once the caller's request is complete and they said goodbye.

        Returns:
            Confirmation that the call ended.
        """
        return str(self._step("end_call")["tool_result"])


def episode_reward(environments: List[EhrToolEnv], **kwargs: Any) -> List[float]:
    """The environment's verify() reward per episode — binary, from the FHIR end state (plus the safety gate)."""
    return [float(env._finish()) for env in environments]


__all__ = ["ENV_URL", "EhrToolEnv", "episode_reward"]
