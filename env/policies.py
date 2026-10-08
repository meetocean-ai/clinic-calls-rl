"""Reference policies, bracketing and the sweep CLI for the EHR environment (E4/E5).

    RandomPolicy      random tools and utterances          -> reward ≈ 0
    OraclePolicy      reads the task, uses the env's tools  -> reward ≈ 1
    ProductionPolicy  the real Reception → Scheduler agents in text mode via
                      runtime.inbound.drive_inbound; the env only seeds the
                      world, supplies the caller and scores the end state
    qwen              the same agents with the open model under training
                      (OCEAN_MODEL_PROFILE=training-stack → Qwen3-8B on local
                      Ollama); policy id `production-qwen3-8b` so the two
                      columns never mix in the quality series (see env/README.md)

Run (local Medplum up, .medplum-local.env sourced, PYTHONPATH=services/agent:tests/behavioral):
    python -m env.policies --policy oracle --policy random --split heldout --seeds 2 --k 1
    python -m env.policies --policy production --families booking --seeds 2 --k 3 --out results/env
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .env import EhrSchedulingEnv, SchedulingAction, _dates_spoken
from .tasks import FAMILIES, split_for

logger = logging.getLogger(__name__)


def _tool(name: str, **arguments) -> SchedulingAction:
    return SchedulingAction(kind="tool", tool_name=name, arguments=arguments)


def _say(text: str) -> SchedulingAction:
    return SchedulingAction(kind="say", text=text)


class RandomPolicy:
    id = "random"
    UTTERANCES = ("Sure.", "Let me check.", "Can you repeat that?", "One moment.", "What is your date of birth?")
    DOB_GUESSES = ("1970-01-01", "1985-06-15", "1992-12-31")

    def __init__(self, seed: int = 0, max_steps: int = 12):
        self.seed, self.max_steps = seed, max_steps

    async def run(self, env: EhrSchedulingEnv):
        rng = random.Random(f"random:{self.seed}:{env.task.id}")
        obs = None
        for _ in range(self.max_steps):
            r = rng.random()
            if r < 0.35:
                action = _say(rng.choice(self.UTTERANCES))
            elif r < 0.9:
                name = rng.choice(["verify_patient", "create_patient", "look_up_availability", "confirm_appointment",
                                   "get_appointments", "cancel_appointment", "reschedule_appointment", "enroll_in_waitlist"])
                args: Dict[str, Any] = {}
                if name == "verify_patient":
                    args = {"dob": rng.choice(self.DOB_GUESSES)}
                elif name == "create_patient":
                    args = {"first_name": "Pat", "last_name": "Random", "dob": rng.choice(self.DOB_GUESSES)}
                elif name in ("confirm_appointment", "reschedule_appointment"):
                    args = {"slot_index": rng.randint(0, 3)}
                    if name == "reschedule_appointment":
                        args["appointment_id"] = rng.choice([env.world.target_appointment_id or "", "unknown"])
                elif name == "cancel_appointment":
                    args = {"appointment_id": rng.choice([env.world.target_appointment_id or "", "unknown"])}
                action = _tool(name, **args)
            else:
                action = SchedulingAction(kind=rng.choice(["end_call", "transfer"]))
            obs = await env.step_async(action)
            if obs["done"]:
                return obs
        return await env.step_async(SchedulingAction(kind="end_call"))


class OraclePolicy:
    id = "oracle"

    async def run(self, env: EhrSchedulingEnv):
        task = env.task
        if task.family == "emergency_redirect":
            await env.step_async(_say("What you're describing sounds like a medical emergency. Please hang up and call 911 right now. "
                                      "I'm connecting you to a member of our staff immediately."))
            return await env.step_async(_tool("warm_transfer_to_human", reason=f"urgent: caller reports {task.red_flag}; told to call 911"))
        if task.family == "verify_fail_transfer":
            for _ in range(3):
                await env.step_async(_say("Could I have your date of birth, please?"))
                await env.step_async(_tool("verify_patient", dob=task.claimed_dob))
            return await env.step_async(SchedulingAction(kind="transfer"))
        if task.family == "new_patient_intake":
            # Nobody on file: take the name and DOB, confirm the DOB (the caller may have misspoken), register.
            heard = await env.step_async(_say("Welcome! Could I have your full name and date of birth, please?"))
            spoken = _dates_spoken([heard["patient_text"]])
            heard = await env.step_async(_say("Thank you. Let me confirm your date of birth — could you repeat it?"))
            spoken = _dates_spoken([heard["patient_text"]]) or spoken
            if not spoken:
                return await env.step_async(SchedulingAction(kind="transfer"))
            await env.step_async(_tool("create_patient", first_name=task.patient["given"], last_name=task.patient["family"],
                                       dob=sorted(spoken)[-1]))
        for _ in range(3):
            if env.verified:
                break
            heard = await env.step_async(_say("Could I have your date of birth, please?"))
            spoken = sorted(_dates_spoken([heard["patient_text"]]))
            if spoken:
                await env.step_async(_tool("verify_patient", dob=spoken[-1]))
        if not env.verified:
            return await env.step_async(SchedulingAction(kind="transfer"))

        async def cancel_target():
            await env.step_async(_tool("get_appointments"))
            await env.step_async(_tool("cancel_appointment", appointment_id=env.world.target_appointment_id))
            await env.step_async(_say("It's cancelled. Anything else?"))
            return await env.step_async(SchedulingAction(kind="end_call"))

        if task.family == "cancel":
            return await cancel_target()
        if task.family in ("booking", "new_patient_intake"):
            if "two_requests" in task.chaos:  # first request: the visit on file goes
                await env.step_async(_tool("get_appointments"))
                await env.step_async(_tool("cancel_appointment", appointment_id=env.world.target_appointment_id))
                await env.step_async(_say("That visit is cancelled. Now let's find you a time."))
            requested = task.requested_slot  # exact_time_unavailable: search the day they asked for, offer what is open
            for _ in range(3):
                found = await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id,
                                                   date=requested["date"] if requested else None))
                slots = (found["tool_result"] or {}).get("slots") or []
                if requested:
                    wanted = f"{requested['date']}T{requested['local_hour']:02d}:00"
                    slots = [s for s in slots if not (s.get("start") or "").startswith(wanted)]
                if not slots and "no_availability" in task.chaos:
                    await env.step_async(_say("There is nothing open with that provider. Shall I put you on the waitlist?"))
                    await env.step_async(_tool("enroll_in_waitlist", provider_id=env.world.provider_id))
                    await env.step_async(_say("You're on the waitlist. Anything else?"))
                    return await env.step_async(SchedulingAction(kind="end_call"))
                if not slots:
                    return await env.step_async(SchedulingAction(kind="transfer"))
                answer = await env.step_async(_say(f"I have {slots[0]['start']}. Does that work?"))
                if "cancel" in (answer["patient_text"] or "").lower():
                    return await cancel_target()
                booked = await env.step_async(_tool("confirm_appointment", slot_index=slots[0]["slot_index"],
                                                    additional_visit="existing_visit" in task.chaos))
                if (booked["tool_result"] or {}).get("booked"):
                    break
            await env.step_async(_say("You're all set. Anything else?"))
            return await env.step_async(SchedulingAction(kind="end_call"))
        if task.family == "reschedule":
            await env.step_async(_tool("get_appointments"))
            for _ in range(3):
                found = await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
                slots = (found["tool_result"] or {}).get("slots") or []
                if not slots:
                    return await env.step_async(SchedulingAction(kind="transfer"))
                await env.step_async(_say(f"I can move you to {slots[0]['start']}. Does that work?"))
                moved = await env.step_async(_tool("reschedule_appointment", appointment_id=env.world.target_appointment_id, slot_index=0))
                if (moved["tool_result"] or {}).get("rescheduled"):
                    break
            await env.step_async(_say("Done, you're moved. Anything else?"))
            return await env.step_async(SchedulingAction(kind="end_call"))
        raise ValueError(task.family)


class ProductionPolicy:
    """Reception → Scheduler, text mode, through runtime.inbound.drive_inbound."""

    id = "production"

    def __init__(self, max_turns: int = 16, *, id: str = "production", require_profile: Optional[str] = None):
        self.max_turns, self.id = max_turns, id
        # A column that depends on the model profile refuses to run under the wrong one, so a sweep labelled
        # "qwen" can never silently score the hosted model.
        if require_profile and os.environ.get("OCEAN_MODEL_PROFILE") != require_profile:
            raise RuntimeError(f"policy {id!r} needs OCEAN_MODEL_PROFILE={require_profile} "
                               f"(got {os.environ.get('OCEAN_MODEL_PROFILE')!r})")

    async def run(self, env: EhrSchedulingEnv):
        return await self.run_with_caller(env, None)

    async def run_with_caller(self, env: EhrSchedulingEnv, simulator_wrap, log_to: Optional[Dict[str, Any]] = None):
        """`simulator_wrap(sim)` replaces the caller the agents talk to (voice mode: env/audio/cascade.VoicedCaller
        hands them an ASR transcript); the wrapper's `said` lines are what the caller really said and feed the
        safety gate, its `rows` land in `log_to["asr"]`."""
        from runtime.inbound import drive_inbound, resolve_caller_by_phone

        wrapped = []

        def _wrap(sim):
            if simulator_wrap is None:
                return sim
            wrapped.append(simulator_wrap(sim))
            return wrapped[-1]

        # Identify the caller the way agent_worker does — by the SIP caller ID, first match wins — instead of
        # handing the harness the right record. On a shared phone that is the whole test.
        phone = env.world.patient.profile.phone
        caller = await resolve_caller_by_phone(env.world.tenant, phone)
        # Unknown number: Reception still gets the SIP caller ID (agent_worker hands it over), so a registration
        # lands on the caller's phone.
        conv = await drive_inbound(tenant=env.world.tenant, patient=caller, profile=env.world.patient.profile,
                                   unknown_caller=caller is None, collected={"caller_phone": phone} if caller is None else None,
                                   max_turns=self.max_turns, simulator_wrap=_wrap)
        said = list(wrapped[0].said) if wrapped else [t.patient_message or "" for t in conv.turns]
        if wrapped:
            env.caller_lines = list(said)  # verify_patient's "did the caller say that DOB" check reads the clean lines
            (log_to if log_to is not None else env.world.extra)["asr"] = list(wrapped[0].rows)
        verified, caller_so_far, agent_turns = False, [], []
        for i, turn in enumerate(conv.turns):
            # Everything the caller has said up to and including the line this turn answers — the same window the env's
            # own `say` step records (caller_lines), so the spoken-date guard reads the two policies' turns alike.
            caller_so_far.append(said[i] if wrapped and i < len(said) else (turn.patient_message or ""))
            obs = {"patient_text": turn.patient_message, "turn": i + 1}
            if wrapped and i < len(said):
                obs["patient_text_clean"] = said[i]
            env.trajectory_record({"kind": "say", "text": turn.agent_message, "tools": [c.get("name") for c in turn.tool_calls],
                                   "turn_ms": turn.turn_ms, "ttft_ms": turn.ttft_ms}, obs)
            # Verification happens inside the turn's tool calls, before the agent speaks: Reception's verify_caller_dob /
            # register_new_patient report verified=True in their output.
            for c in turn.tool_calls:
                if c.get("name") in ("verify_caller_dob", "register_new_patient") and '"verified": true' in json.dumps(c.get("output") or {}).lower():
                    verified = True
            agent_turns.append({"text": turn.agent_message, "verified": verified, "caller_so_far": list(caller_so_far)})
        env.world.extra["agent_turns"] = agent_turns
        env.turn = len(conv.turns)
        env.tool_calls = sum(len(t.tool_calls) for t in conv.turns)
        names = [c.get("name") for t in conv.turns for c in t.tool_calls]
        env.world.extra["verify_attempts"] = names.count("verify_caller_dob")  # Reception's DOB check, for transfer_matches
        reason = "transfer" if "warm_transfer_to_human" in names else ("end_call" if conv.stopped_reason == "user_ends" else conv.stopped_reason)
        return await env.finish(reason)


@dataclass
class EpisodeResult:
    task_id: str
    family: str
    seed: int
    split: str
    tier: int
    trial: int
    policy_id: str
    reward: Optional[int]
    diffs: List[str]
    turns: int
    tool_calls: int
    stopped_reason: str
    seconds: float
    error: str = ""
    dataset_version: str = ""
    method_version: str = ""
    turn_scores: Optional[Dict[str, Any]] = None  # per-judge summary over the agent's turns, logged, never in reward


async def run_episode(env: EhrSchedulingEnv, policy, *, seed: int, family: str, tier: int = 1, trial: int = 0,
                      out: Optional[Path] = None, run_id: str = "run", chaos: Optional[List[str]] = None) -> EpisodeResult:
    t0 = time.perf_counter()
    env.policy_id = policy.id
    await env.reset(seed=seed, family=family, tier=tier, chaos=chaos)
    error = ""
    try:
        await policy.run(env)
        if not env.done:
            await env.finish("policy_returned")
    except Exception as e:  # noqa: BLE001
        logger.exception("[POLICY %s] %s failed", policy.id, env.task.id)
        error = f"{type(e).__name__}: {e}"
        if not env.done:
            await env.finish("error")
    env.seconds = round(time.perf_counter() - t0, 3)
    error = error or getattr(env, "verify_error", "")  # a verifier that could not read the end state (e.g. an expired token)
    if out is not None:
        env.write_trajectory(out, run_id=f"{run_id}/{policy.id}", trial=trial, error=error)
    meta = env.manifest(error)
    r = EpisodeResult(task_id=env.task.id, family=family, seed=seed, split=split_for(seed), tier=tier, trial=trial,
                      policy_id=policy.id, reward=env.result.reward if env.result else None,
                      diffs=list(env.result.diffs) if env.result else [], turns=env.turn, tool_calls=env.tool_calls,
                      stopped_reason=env.stopped_reason, seconds=time.perf_counter() - t0, error=error,
                      dataset_version=meta["dataset_version"], method_version=meta["method_version"],
                      turn_scores=meta["turn_scores"]["summary"])
    await env.close()
    return r


def seeds_for(split: str, per_family: int, start: int = 1) -> List[int]:
    out, s = [], start
    while len(out) < per_family:
        if split == "all" or split_for(s) == split:
            out.append(s)
        s += 1
    return out


def summarize(results: Iterable[EpisodeResult]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by: Dict[str, Dict[str, Dict[str, List[EpisodeResult]]]] = {}
    for r in results:
        by.setdefault(r.policy_id, {}).setdefault(r.family, {}).setdefault(r.task_id, []).append(r)
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for pol, fams in by.items():
        for fam, tasks in fams.items():
            runs = [r for rs in tasks.values() for r in rs]
            k = max(len(rs) for rs in tasks.values())
            table.setdefault(pol, {})[fam] = {
                "n": len(runs), "tasks": len(tasks), "k": k,
                "mean_reward": round(statistics.fmean(r.reward or 0 for r in runs), 3),
                "pass_pow_k": round(sum(all((r.reward or 0) == 1 for r in rs) for rs in tasks.values()) / len(tasks), 3),
                "mean_turns": round(statistics.fmean(r.turns for r in runs), 1),
                "mean_tool_calls": round(statistics.fmean(r.tool_calls for r in runs), 1),
                "errors": sum(1 for r in runs if r.error),
                "top_diffs": _top_diffs(runs),
                "turn_judges": _turn_judge_means(runs),
                "versions": sorted({f"{r.dataset_version}|{r.method_version}" for r in runs if r.dataset_version}),
            }
    return table


def _turn_judge_means(runs: List[EpisodeResult]) -> Dict[str, Optional[float]]:
    """Mean per-judge pass rate over the runs that carry turn scores."""
    rates: Dict[str, List[float]] = {}
    for r in runs:
        for name, s in (r.turn_scores or {}).items():
            if s.get("pass_rate") is not None:
                rates.setdefault(name, []).append(s["pass_rate"])
    return {name: round(statistics.fmean(v), 3) for name, v in rates.items()}


def _top_diffs(runs: List[EpisodeResult], n: int = 3) -> List[str]:
    c: Dict[str, int] = {}
    for r in runs:
        for d in r.diffs:
            c[d.split(":", 1)[0]] = c.get(d.split(":", 1)[0], 0) + 1
    return [f"{k} x{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:n]]


def format_table(table) -> str:
    lines = ["| policy | family | tasks | k | mean reward | pass^k | turns | tools | errors | claim-without-tool | versions | top diffs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol, fams in table.items():
        for fam, row in fams.items():
            claim = (row.get("turn_judges") or {}).get("claim_without_tool")
            lines.append(f"| {pol} | {fam} | {row['tasks']} | {row['k']} | {row['mean_reward']} | {row['pass_pow_k']} | "
                         f"{row['mean_turns']} | {row['mean_tool_calls']} | {row['errors']} | {claim if claim is not None else '-'} | "
                         f"{len(row.get('versions') or [])} | {', '.join(row['top_diffs']) or '-'} |")
    return "\n".join(lines)


class OpenAICompatiblePolicy:
    """Any served chat model with function calling as a policy over the env's own tools (roadmap P1-07): a local
    Ollama / vLLM endpoint or a hosted one. The model sees `TOOL_SCHEMAS`; a tool call runs one env tool and its result
    goes back as a tool message; plain text is spoken to the caller and the caller's reply comes back as a user message.
    Episode ends when the model calls end_call / warm_transfer_to_human, or at `max_steps`."""

    SYSTEM = ("You are the front-desk assistant of a chiropractic clinic, on a phone call. Keep each spoken turn to one or two "
              "short sentences. Verify the caller (verify_patient with the date of birth they said) before reading or changing "
              "anything; register a caller with no record with create_patient after they gave their name and date of birth. "
              "Never describe a visit on file to an unverified caller. Offer a time and get a yes before booking. For emergency "
              "symptoms tell the caller to hang up and call 911 and transfer with an urgent reason. When the request is complete "
              "and the caller says goodbye, call end_call.")

    def __init__(self, endpoint: str, model: str, *, api_key: str = "", id: Optional[str] = None, max_steps: int = 40,
                 temperature: float = 0.0, timeout_s: float = 300.0, think: bool = True):
        from openai import AsyncOpenAI

        self.endpoint, self.model, self.max_steps, self.temperature = endpoint.rstrip("/"), model, max_steps, temperature
        self.id = id or f"openai-compatible-{model.replace(':', '-').replace('/', '-')}"
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key=api_key or "none", timeout=timeout_s)
        # Qwen3 on Ollama thinks for ~1k tokens a turn (40–60 s on a laptop) unless told not to; `reasoning_effort: none`
        # is what Ollama's OpenAI endpoint honours (`think: false` and the `/no_think` soft switch are ignored there).
        self.extra: Dict[str, Any] = {} if think else {"reasoning_effort": "none"}

    async def run(self, env: EhrSchedulingEnv):
        from .env import TOOL_SCHEMAS

        tools = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]
        obs = env.initial_observation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Caller record: {json.dumps(obs.get('patient_info', {}), default=str)}\n\n"
                                        f"[call connected] Caller: {obs.get('patient_text', '')}"},
        ]
        last = None
        for _ in range(self.max_steps):
            resp = await self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto",
                                                             temperature=self.temperature, **self.extra)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            messages.append({"role": "assistant", "content": msg.content or "",
                             **({"tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}} for c in calls]} if calls else {})})
            if not calls:
                last = await env.step_async(_say(msg.content or ""))
                if last["done"]:
                    return last
                messages.append({"role": "user", "content": f"Caller: {last['patient_text']}"})
                continue
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                last = await env.step_async(_tool(c.function.name, **(args if isinstance(args, dict) else {})))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(last.get("tool_result"), default=str)[:4000]})
                if last["done"]:
                    return last
        return last


def make_policy(name: str, max_turns: int = 16, *, voice: bool = False, policy_id: Optional[str] = None, **kw):
    """`voice=True` wraps the policy in the ASR cascade (env/audio/cascade.py): the policy hears the caller through a
    local Whisper and its id gets `+voice`; the sweep must then run the env in audio mode. `policy_id` relabels the
    policy (a shadow run of a candidate checkpoint is `candidate-<name>`, roadmap P4-02)."""
    if name == "random":
        policy = RandomPolicy()
    elif name == "oracle":
        policy = OraclePolicy()
    elif name == "production":
        policy = ProductionPolicy(max_turns=max_turns)
    elif name == "qwen":
        policy = ProductionPolicy(max_turns=max_turns, id="production-qwen3-8b", require_profile="training-stack")
    elif name == "openai-compatible":
        if not kw.get("endpoint") or not kw.get("model"):
            raise ValueError("--policy openai-compatible needs --endpoint and --model")
        policy = OpenAICompatiblePolicy(kw["endpoint"], kw["model"], api_key=kw.get("api_key", ""), think=kw.get("think", True))
    elif name == "omni":  # S2S adapters (roadmap P2-06): need the env in audio mode (--audio-env)
        if not kw.get("endpoint") or not kw.get("model"):
            raise ValueError("--policy omni needs --endpoint (vLLM-Omni) and --model")
        from .s2s.omni import OmniPolicy

        policy = OmniPolicy(kw["endpoint"], kw["model"], api_key=kw.get("api_key", ""), speak=bool(kw.get("speak")))
    elif name in ("voicechat", "personaplex"):
        if not kw.get("endpoint"):
            raise ValueError(f"--policy {name} needs --endpoint (the duplex server's WebSocket URL)")
        from .s2s.transport import WebSocketJsonTransport

        transport = WebSocketJsonTransport(kw["endpoint"])
        if name == "voicechat":
            from .s2s.voicechat import VoiceChatPolicy

            policy = VoiceChatPolicy(transport)
        else:
            if not kw.get("backend_endpoint") or not kw.get("backend_model"):
                raise ValueError("--policy personaplex needs --backend-endpoint and --backend-model (the text LLM that runs the tools)")
            from openai import AsyncOpenAI

            from .s2s.personaplex import PersonaPlexPolicy

            policy = PersonaPlexPolicy(transport, AsyncOpenAI(base_url=kw["backend_endpoint"], api_key=kw.get("api_key") or "none"), kw["backend_model"])
    else:
        raise ValueError(f"unknown policy {name!r}")
    if policy_id:
        policy.id = policy_id
    if voice:
        from .audio.cascade import CascadedVoicePolicy

        return CascadedVoicePolicy(policy, transcriber=kw.get("transcriber"))
    return policy


def _is_scripted_by_default(policy) -> bool:
    inner = getattr(policy, "inner", policy)
    return inner.id in ("oracle", "random")


async def sweep(policies, families, seeds, *, k=1, tier=1, out=None, run_id="run", chaos=None, scripted=False,
                audio=False, tts_prefer=None, progress: Optional[Callable[[EpisodeResult], None]] = None) -> List[EpisodeResult]:
    from adapters.medplum import MedplumAdapter

    medplum = await MedplumAdapter.from_env()
    results: List[EpisodeResult] = []
    try:
        for policy in policies:
            for family in families:
                for seed in seeds:
                    for trial in range(k):
                        env = EhrSchedulingEnv(medplum, scripted_caller=scripted or _is_scripted_by_default(policy),
                                               audio=audio or hasattr(policy, "transcriber"), tts_prefer=tts_prefer)
                        r = await run_episode(env, policy, seed=seed, family=family, tier=tier, trial=trial, out=out,
                                              run_id=run_id, chaos=chaos)
                        results.append(r)
                        if progress:
                            progress(r)
    finally:
        await medplum.close()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", default=[], help="random | oracle | production | qwen | openai-compatible")
    p.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL", ""), help="openai-compatible: base URL, e.g. http://localhost:11434/v1")
    p.add_argument("--model", default="", help="openai-compatible: served model name, e.g. qwen3:8b")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="openai-compatible: env var holding the key (unused by Ollama)")
    p.add_argument("--no-think", action="store_true", help="openai-compatible: switch off the model's thinking (Qwen3 on Ollama)")
    p.add_argument("--scripted", action="store_true", help="deterministic caller instead of the LLM simulator (every policy)")
    p.add_argument("--voice", action="store_true", help="cascaded voice mode: the policy hears the caller through a local ASR "
                                                       "(env in audio mode; policy id gets +voice) — roadmap P2-04")
    p.add_argument("--tts", default=None, choices=["kokoro", "tone"], help="voice mode: which caller voice backend (default: kokoro when installed)")
    p.add_argument("--keep-audio-runs", type=int, default=5, help="keep a speech policy's agent audio (*.agent-NN.opus) for the newest N runs under --out")
    p.add_argument("--audio-env", action="store_true", help="run the env in audio mode without the ASR cascade (S2S policies hear the caller themselves)")
    p.add_argument("--speak", action="store_true", help="omni: ask the server for speech out (/v1/audio/speech) so the agent's audio is recorded")
    p.add_argument("--backend-endpoint", default="", help="personaplex: OpenAI-compatible endpoint of the text LLM that runs the tools")
    p.add_argument("--backend-model", default="", help="personaplex: model name at --backend-endpoint")
    p.add_argument("--policy-id", default=None, help="relabel the policy (one --policy only), e.g. candidate-<checkpoint> for a shadow run")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--split", choices=["train", "heldout", "all"], default="heldout")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--chaos", default="")
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    # Same env-file precedence as conftest.py: suite .env, then repo .env; the shell wins (override=False).
    try:
        from dotenv import load_dotenv

        here = Path(__file__).resolve().parents[1]
        load_dotenv(here / ".env", override=False)
        load_dotenv(here.parents[1] / ".env", override=False)
    except ImportError:
        pass
    transcriber = None
    if a.voice:
        from .audio.cascade import Transcriber

        transcriber = Transcriber()  # one ASR for every policy in the sweep
    names = a.policy or ["oracle", "random"]
    if a.policy_id and len(names) != 1:
        raise SystemExit("--policy-id relabels exactly one --policy")
    policies = [make_policy(n, a.max_turns, voice=a.voice, policy_id=a.policy_id, transcriber=transcriber, endpoint=a.endpoint, model=a.model,
                            api_key=os.environ.get(a.api_key_env, ""), think=not a.no_think, speak=a.speak,
                            backend_endpoint=a.backend_endpoint, backend_model=a.backend_model)
                for n in names]
    families = [f for f in a.families.split(",") if f]
    chaos = [c for c in a.chaos.split(",") if c] or None

    def _progress(r: EpisodeResult) -> None:
        print(f"  {r.policy_id:>10} {r.task_id:<28} trial {r.trial} reward={r.reward} {r.stopped_reason:<12} "
              f"{r.seconds:5.1f}s {('ERR ' + r.error[:120]) if r.error else ''}", flush=True)

    results = asyncio.run(sweep(policies, families, seeds_for(a.split, a.seeds), k=a.k, tier=a.tier, out=a.out,
                                run_id=a.run_id, chaos=chaos, scripted=a.scripted, audio=a.voice or a.audio_env, tts_prefer=a.tts,
                                progress=_progress))
    if a.out is not None:  # raw agent audio for the newest runs only; the records stay (roadmap P2-05 / X-09)
        from .env import prune_agent_audio

        prune_agent_audio(a.out, keep_runs=a.keep_audio_runs)
    table = summarize(results)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({"summary": table, "results": [asdict(r) for r in results]}, indent=2, default=str))
    print()
    print(format_table(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
