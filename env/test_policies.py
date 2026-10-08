"""Policies that need no LLM to test: the OpenAI-compatible policy loop against a fake chat client (roadmap P1-07) and
the tool schemas it advertises. The real thing runs with `--policy openai-compatible --endpoint … --model …`.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/test_policies.py -v
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from env.env import TOOL_SCHEMAS, TOOLS, EhrSchedulingEnv
from env.policies import OpenAICompatiblePolicy, make_policy, run_episode

pytestmark = [pytest.mark.covers("rl-scheduling-env")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def test_tool_schemas_cover_every_tool_with_valid_json_schema():
    assert [s["name"] for s in TOOL_SCHEMAS] == list(TOOLS)
    for s in TOOL_SCHEMAS:
        assert s["description"] and s["parameters"]["type"] == "object"
        for req in s["parameters"].get("required", []):
            assert req in s["parameters"]["properties"], (s["name"], req)
    json.dumps(TOOL_SCHEMAS)  # serializable as-is for an OpenAI `tools=` payload


def test_make_policy_needs_endpoint_and_model_for_openai_compatible():
    with pytest.raises(ValueError):
        make_policy("openai-compatible")
    p = make_policy("openai-compatible", endpoint="http://localhost:11434/v1", model="qwen3:8b")
    assert p.id == "openai-compatible-qwen3-8b" and p.endpoint == "http://localhost:11434/v1"


def test_policy_id_relabels_a_candidate_for_shadow_runs():
    """P4-02: a shadow run of a checkpoint is `candidate-<name>` (and `+voice`), never mistaken for production."""
    from env.audio.cascade import Transcriber

    class _Ear(Transcriber):
        def __init__(self):
            self.name = "ear"

    assert make_policy("oracle", policy_id="candidate-lora-s400").id == "candidate-lora-s400"
    assert make_policy("oracle", voice=True, policy_id="candidate-lora-s400", transcriber=_Ear()).id == "candidate-lora-s400+voice"
    assert make_policy("oracle").id == "oracle"


class _FakeChat:
    """Scripted assistant turns: each item is either text (spoken) or a list of (tool_name, arguments) calls."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, str):
            msg = SimpleNamespace(content=item, tool_calls=None)
        else:
            msg = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(item)])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _scripted_policy(script):
    p = OpenAICompatiblePolicy("http://fake/v1", "fake-model")
    p.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeChat(script)))
    return p


@local_only
async def test_openai_compatible_policy_books_through_the_env_tools(medplum):
    """A model that follows the rules: ask DOB → verify with what the caller said → search → offer → book → end."""
    script = [
        "Hi! Could I have your date of birth, please?",
        None,  # filled after we know the DOB the scripted caller will say
        [("look_up_availability", {})],
        "I have an opening tomorrow morning. Does that work?",
        [("confirm_appointment", {"slot_index": 0})],
        "You're all set. Anything else?",
        [("end_call", {})],
    ]
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking")
    script[1] = [("verify_patient", {"dob": env.task.patient["dob"]})]
    policy = _scripted_policy(script)
    await env.close()
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), policy, seed=5, family="booking")
    assert r.reward == 1, (r.diffs, r.error)
    assert r.policy_id == "openai-compatible-fake-model" and r.tool_calls == 4 and r.turns == 3
    first = policy.client.chat.completions.calls[0]
    assert first["tools"][0]["function"]["name"] == "verify_patient" and "Caller record" in first["messages"][1]["content"]
    assert len(policy.client.chat.completions.calls) == 7  # one model turn per script item
    # `messages` is the live transcript the policy keeps appending to: all four tool results (verify, search, book,
    # end_call) came back as tool messages, every caller line as a user message
    roles = [m["role"] for m in policy.client.chat.completions.calls[-1]["messages"]]
    assert roles.count("tool") == 4 and roles.count("user") >= 3 and roles[-1] == "tool"


@local_only
async def test_openai_compatible_policy_echoing_the_record_dob_is_refused_and_scored_zero(medplum):
    """The anti-cheat the served model meets: verify_patient with the DOB from the record, never said by the caller."""
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking")
    dob = env.task.patient["dob"]
    await env.close()
    policy = _scripted_policy([[("verify_patient", {"dob": dob})], [("look_up_availability", {})], [("end_call", {})]])
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), policy, seed=5, family="booking")
    assert r.reward == 0 and any(d.startswith("booking_matches") for d in r.diffs)
    tool_msgs = [m for m in policy.client.chat.completions.calls[-1]["messages"] if m["role"] == "tool"]
    assert "has not given that date of birth" in tool_msgs[0]["content"] and "not verified" in tool_msgs[1]["content"]


@pytest.fixture
async def medplum():
    from adapters.medplum import MedplumAdapter

    m = await MedplumAdapter.from_env()
    try:
        yield m
    finally:
        await m.close()
