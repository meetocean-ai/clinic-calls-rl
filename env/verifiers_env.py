"""The EHR scheduling environment as a Prime Intellect `verifiers` environment (roadmap P1-04).

Runs OUT of process: this module imports only `verifiers`, `websockets`, `datasets` and the standard library and
talks to the OpenEnv server (`env/server.py`) over its `/ws` session protocol. Why out of process: `verifiers`
depends on `openai-agents`, whose top-level package is `agents` — the same name as the product's agent package — so
verifiers lives in its own venv (`tests/behavioral/.venv-verifiers`, gitignored) and reaches the environment over a
socket; nothing in the agent's environment changes.

    # terminal 1 (agent venv):  cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.server
    # terminal 2 (verifiers venv, a local open model through its OpenAI-compatible endpoint):
    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \\
      .venv-verifiers/bin/python -m env.verifiers_env --ws ws://127.0.0.1:8011/ws --model qwen3:8b --examples 2

Tool protocol: the model answers with either plain text (spoken to the caller) or one JSON object
`{"tool": "<name>", "arguments": {...}}`. The reward is the server's state-based `verify()` on the last step; the
rubric reads it from the rollout state and adds nothing of its own. `load_environment()` is the verifiers / prime-rl
entry point.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import verifiers as vf
import websockets
from datasets import Dataset

FAMILIES = ("booking", "cancel", "reschedule", "verify_fail_transfer", "new_patient_intake", "emergency_redirect")
HELDOUT_SEEDS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)

TOOLS_TEXT = """\
You are the front-desk assistant of a chiropractic clinic, on a phone call. Answer in ONE of two forms:
1. Plain text: what you say to the caller (one or two short sentences).
2. Exactly one JSON object on a single line to use a tool: {"tool": "<name>", "arguments": {...}}

Tools (the record-reading and writing ones refuse until the caller is verified or registered):
- verify_patient {"dob": "YYYY-MM-DD"}  — pass the date of birth THE CALLER SAID; echoing the record verifies nobody
- create_patient {"first_name": "...", "last_name": "...", "dob": "YYYY-MM-DD"}  — only for a caller with no record, after they said their name and DOB
- look_up_availability {"provider_id": "<optional id>", "date": "<optional YYYY-MM-DD>"}  — returns slots with slot_index
- confirm_appointment {"slot_index": <int>, "additional_visit": <bool, optional>}  — books an offered slot after the caller agreed to it
- get_appointments {}  — the caller's upcoming appointments (appointment_id, start, provider, provider_id)
- cancel_appointment {"appointment_id": "<id>", "reason": "<text>"}
- reschedule_appointment {"appointment_id": "<id>", "slot_index": <int>}  — after look_up_availability
- enroll_in_waitlist {"provider_id": "<id>", "notes": "<text>"}  — when the provider the caller wants has nothing open
- warm_transfer_to_human {"reason": "<text>"}  — ends the call; say "urgent" in the reason for a medical emergency
- end_call {}  — ends the call after the caller is done

Verify the caller before reading or changing anything. Never describe a visit on file to an unverified caller. Offer
a time and get a yes before booking. For emergency symptoms tell the caller to hang up and call 911 and transfer with
an urgent reason. When the caller's request is complete and they say goodbye, call end_call.
"""

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> Dict[str, Any]:
    """Model text → EhrAction dict. JSON with a "tool" key is a tool call; anything else is speech."""
    text = (text or "").strip()
    m = _JSON_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("tool"):
                name = str(obj["tool"])
                args = obj.get("arguments") or {}
                if name == "end_call":
                    return {"kind": "end_call"}
                if name == "warm_transfer_to_human":
                    return {"kind": "tool", "tool_name": name, "arguments": args}
                return {"kind": "tool", "tool_name": name, "arguments": args}
        except ValueError:
            pass
    return {"kind": "say", "text": text or "Is there anything else I can help you with?"}


def _flatten(reply: Dict[str, Any]) -> Dict[str, Any]:
    data = reply.get("data") or {}
    obs = dict(data.get("observation") or {}) if isinstance(data.get("observation"), dict) else dict(data)
    obs["done"] = bool(data.get("done", obs.get("done", False)))
    obs["reward"] = data.get("reward", obs.get("reward"))
    return obs


class EhrVerifiersEnv(vf.MultiTurnEnv):
    """One rollout = one WebSocket session = one episode on the server."""

    def __init__(self, ws_url: str = "ws://127.0.0.1:8011/ws", families=FAMILIES, seeds=HELDOUT_SEEDS, tier: int = 1,
                 max_turns: int = 30, **kwargs):
        rows = []
        for i, (family, seed) in enumerate((f, s) for f in families for s in seeds):
            rows.append({
                "prompt": [{"role": "system", "content": TOOLS_TEXT}, {"role": "user", "content": "[call connecting]"}],
                "answer": "",
                "info": {"family": family, "seed": seed, "tier": tier if family != "emergency_redirect" else 1},
                "example_id": i,
            })
        self.ws_url = ws_url
        rubric = vf.Rubric(funcs=[state_reward], weights=[1.0])
        super().__init__(dataset=Dataset.from_list(rows), rubric=rubric, max_turns=max_turns, **kwargs)

    async def _send(self, state, msg: Dict[str, Any]) -> Dict[str, Any]:
        ws = state["ws"]
        await ws.send(json.dumps(msg))
        reply = json.loads(await ws.recv())
        if reply.get("type") == "error":
            raise RuntimeError(f"server error: {reply.get('data')}")
        return _flatten(reply)

    async def setup_state(self, state, **kwargs):
        """verifiers 0.1.5 generates the first model turn from the dataset prompt ("[call connecting]"); the episode is
        reset here and the caller's real opening line is delivered as the first environment response."""
        info = state["info"]
        state["ws"] = await websockets.connect(self.ws_url, max_size=8 * 1024 * 1024)
        obs = await self._send(state, {"type": "reset", "data": {"seed": info["seed"], "family": info["family"], "tier": info["tier"]}})
        state["task_id"] = obs.get("task_id")
        if os.getenv("VF_DEBUG"):
            print(f"[vf] reset -> task={obs.get('task_id')} opening={str(obs.get('patient_text'))[:80]!r}", file=sys.stderr)
        state["opening"] = (f"Caller record: {json.dumps(obs.get('patient_info', {}), default=str)}\n\n"
                            f"[call connected] Caller: {obs.get('patient_text', '')}")
        state["done"] = False
        state["final_reward"] = None
        return state

    async def env_response(self, messages, state, **kwargs):
        if state.get("opening") is not None:  # the model's greeting; the call connects now
            opening = state.pop("opening")
            return [{"role": "user", "content": opening}], state
        last = messages[-1]
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        action = parse_action(str(content or ""))
        obs = await self._send(state, {"type": "step", "data": action})
        if os.getenv("VF_DEBUG"):
            print(f"[vf] action={json.dumps(action)[:160]} -> done={obs.get('done')} reward={obs.get('reward')} "
                  f"patient={str(obs.get('patient_text'))[:80]!r} tool={str(obs.get('tool_result'))[:120]!r}", file=sys.stderr)
        if obs.get("done"):
            state["done"] = True
            state["final_reward"] = obs.get("reward")
            state["stopped_reason"] = obs.get("stopped_reason")
            state["diffs"] = obs.get("diffs") or []
            await self._close(state)
        if action["kind"] == "say":
            return [{"role": "user", "content": f"Caller: {obs.get('patient_text', '')}"}], state
        return [{"role": "user", "content": f"Tool result: {json.dumps(obs.get('tool_result'), default=str)[:1500]}"}], state

    async def is_completed(self, messages, state, **kwargs) -> bool:
        if state.get("done"):
            return True
        if await self.max_turns_reached(state):
            await self._close(state)
            return True
        return False

    async def _close(self, state) -> None:
        ws = state.pop("ws", None)
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "close"}))
                await ws.close()
            except Exception:  # noqa: BLE001
                pass


def state_reward(state, **kwargs) -> float:
    """The server's verify() result; 0 when the episode never finished."""
    r = state.get("final_reward")
    return float(r) if r is not None else 0.0


def load_environment(**kwargs) -> vf.Environment:
    """verifiers' entry point convention (`vf-eval`, prime-rl)."""
    return EhrVerifiersEnv(**kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    from openai import AsyncOpenAI

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ws", default="ws://127.0.0.1:8011/ws")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--examples", type=int, default=2)
    p.add_argument("--rollouts", type=int, default=1)
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--seeds", default="5,10")
    p.add_argument("--max-concurrent", type=int, default=2)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)
    env = EhrVerifiersEnv(ws_url=args.ws, families=[f for f in args.families.split(",") if f],
                          seeds=[int(s) for s in args.seeds.split(",")], tier=args.tier)
    client = AsyncOpenAI()  # OPENAI_BASE_URL / OPENAI_API_KEY: a local Ollama or vLLM endpoint works as well as a hosted one
    out = env.evaluate(client=client, model=args.model, num_examples=args.examples, rollouts_per_example=args.rollouts,
                       max_concurrent=args.max_concurrent)
    if asyncio.iscoroutine(out) or isinstance(out, asyncio.Future):  # verifiers 0.1.5 is sync here; newer builds may await
        out = asyncio.run(out)
    rows = [{"task_id": st.get("task_id"), "reward": reward, "stopped_reason": st.get("stopped_reason"), "diffs": st.get("diffs"),
             "turns": st.get("turn")} for st, reward in zip(out.state, out.reward)]
    rewards = [r["reward"] for r in rows]
    summary = {"model": args.model, "n": len(rows), "mean_reward": (sum(r or 0 for r in rewards) / len(rewards)) if rewards else None,
               "rollouts": rows}
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
