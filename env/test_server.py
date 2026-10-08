"""The OpenEnv server over the EHR environment (env/server.py, roadmap P1-03): schemas and routes without a server,
a full episode over the real `/ws` session protocol against the local Medplum (no LLM — scripted caller).

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/test_server.py -v
"""
from __future__ import annotations

import os

import pytest

from env.server import OPENENV_AVAILABLE, EhrAction, EhrObservation, EhrOpenEnv, EhrState, _observation

pytestmark = [pytest.mark.covers("rl-env-server")]
needs_openenv = pytest.mark.skipif(not OPENENV_AVAILABLE, reason="pip install -e tests/behavioral[env-server]")
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def test_schemas_mirror_the_in_process_contract():
    a = EhrAction(kind="tool", tool_name="verify_patient", arguments={"dob": "1984-03-09"})
    assert a.model_dump()["arguments"] == {"dob": "1984-03-09"} and EhrAction().kind == "say"
    obs = _observation({"patient_text": "Hi.", "task_id": "booking-00005", "tools": ["verify_patient"], "patient_info": {"verified": False},
                        "done": False, "reward": None, "tool_result": None, "turn": 0})
    assert isinstance(obs, EhrObservation) and obs.task_id == "booking-00005" and obs.done is False and obs.reward is None
    final = _observation({"done": True, "reward": 0, "diffs": ["booking_matches: no new booked appointment for the patient"],
                          "stopped_reason": "end_call", "patient_text": "", "turn": 3}, task_id="booking-00005")
    assert final.done and final.reward == 0 and final.diffs[0].startswith("booking_matches") and final.task_id == "booking-00005"
    assert EhrState().task_id == "" and EhrState(task_id="x", reward=1).reward == 1


@needs_openenv
def test_app_exposes_the_openenv_routes_and_schema():
    from env.server import build_app

    app = build_app(max_concurrent_envs=2)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert {"/reset", "/step", "/state", "/ws", "/schema", "/health", "/metadata"} <= paths
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        schema = client.get("/schema").json()
        text = str(schema)
        assert "patient_text" in text and "tool_name" in text and "kind" in text


@needs_openenv
@local_only
def test_full_episode_over_the_ws_session_protocol():
    """reset → say → verify → search → book → end_call, through the real server, scored by the environment."""
    from starlette.testclient import TestClient

    from env.server import build_app

    os.environ["EHR_ENV_SCRIPTED_CALLER"] = "1"
    try:
        with TestClient(build_app(max_concurrent_envs=2)) as client, client.websocket_connect("/ws") as ws:
            def send(msg):
                ws.send_json(msg)
                reply = ws.receive_json()
                assert reply["type"] != "error", reply
                d = reply["data"]
                # the server may serialize {"observation": {...}, "reward": r, "done": d}; read either shape as one dict
                if isinstance(d.get("observation"), dict):
                    flat = dict(d["observation"])
                    flat.setdefault("done", d.get("done"))
                    flat.setdefault("reward", d.get("reward"))
                    reply = {**reply, "data": flat}
                return reply

            first = send({"type": "reset", "data": {"seed": 5, "family": "booking", "tier": 1}})
            obs = first["data"]
            assert obs["task_id"] == "booking-00005" and "verify_patient" in obs["tools"] and obs["patient_text"]
            dob = obs["patient_info"]["dob"]
            blocked = send({"type": "step", "data": {"kind": "tool", "tool_name": "look_up_availability"}})["data"]
            assert "not verified" in blocked["tool_result"]["error"]
            heard = send({"type": "step", "data": {"kind": "say", "text": "Could I have your date of birth, please?"}})["data"]
            assert dob in heard["patient_text"]
            ok = send({"type": "step", "data": {"kind": "tool", "tool_name": "verify_patient", "arguments": {"dob": dob}}})["data"]
            assert ok["tool_result"]["verified"] is True
            state = send({"type": "state"})["data"]
            assert state["verified"] is True and state["family"] == "booking" and state["dataset_version"].startswith("tasks-v3+gen-")
            found = send({"type": "step", "data": {"kind": "tool", "tool_name": "look_up_availability", "arguments": {}}})["data"]
            assert found["tool_result"]["found"]
            booked = send({"type": "step", "data": {"kind": "tool", "tool_name": "confirm_appointment", "arguments": {"slot_index": 0}}})["data"]
            assert booked["tool_result"]["booked"]
            final = send({"type": "step", "data": {"kind": "end_call"}})["data"]
            assert final["done"] is True and final["reward"] == 1, final
            ws.send_json({"type": "close"})
    finally:
        os.environ.pop("EHR_ENV_SCRIPTED_CALLER", None)


@needs_openenv
@local_only
async def test_reference_client_against_a_live_server():
    """The OpenEnv reference client (EnvClient) talking to `python -m env.server` over a real socket — the spec check."""
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path

    import httpx

    from env.server import EhrEnvClient

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    here = Path(__file__).resolve().parents[1]
    env = {**os.environ, "ENV_PORT": str(port), "ENV_HOST": "127.0.0.1", "EHR_ENV_SCRIPTED_CALLER": "1",
           "PYTHONPATH": f"{here.parents[1] / 'services' / 'agent'}:{here}"}
    proc = subprocess.Popen([sys.executable, "-m", "env.server"], cwd=here, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise AssertionError(f"server never came up: {proc.stderr.read().decode()[-2000:]}")
        async with EhrEnvClient(base_url=f"ws://127.0.0.1:{port}") as client:
            first = await client.reset(seed=5, family="verify_fail_transfer", tier=1)
            assert first.observation.task_id == "verify_fail_transfer-00005" and not first.done
            wrong = first.observation.patient_info["dob"]  # the record's DOB — the caller will claim another
            for _ in range(2):
                heard = await client.step(EhrAction(kind="say", text="Could I have your date of birth, please?"))
                said = heard.observation.patient_text
                assert wrong not in said and said
                r = await client.step(EhrAction(kind="tool", tool_name="verify_patient", arguments={"dob": said.split("It's ")[-1].rstrip(".")}))
                assert r.observation.tool_result["verified"] is False
            state = await client.state()
            assert state.family == "verify_fail_transfer" and state.verified is False and state.step_count == 4
            final = await client.step(EhrAction(kind="transfer"))
            assert final.done and final.reward == 1, final.observation.diffs
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@local_only
async def test_in_process_environment_object_runs_an_episode_and_reports_state():
    env = EhrOpenEnv(scripted_caller=True)
    try:
        obs = await env.reset_async(seed=10, family="cancel", tier=1)
        assert obs.task_id == "cancel-00010" and env.state.family == "cancel" and env.state.step_count == 0
        heard = await env.step_async(EhrAction(kind="say", text="Your date of birth, please?"))
        dob = env.env.task.patient["dob"]
        assert dob in heard.patient_text
        assert (await env.step_async(EhrAction(kind="tool", tool_name="verify_patient", arguments={"dob": dob}))).tool_result["verified"]
        mine = (await env.step_async(EhrAction(kind="tool", tool_name="get_appointments"))).tool_result["appointments"]
        await env.step_async(EhrAction(kind="tool", tool_name="cancel_appointment", arguments={"appointment_id": mine[0]["appointment_id"]}))
        final = await env.step_async(EhrAction(kind="end_call"))
        assert final.done and final.reward == 1 and env.state.reward == 1 and env.state.step_count == 5
        with pytest.raises(ValueError):
            await env.reset_async(seed=1, family="nope")
    finally:
        await env.aclose()
