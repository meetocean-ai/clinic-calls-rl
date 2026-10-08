"""GRPO pipeline smoke on the Mac (roadmap P1-09): a few optimizer steps of a small Qwen3 through the whole loop —
TRL GRPOTrainer → tool calls → OpenEnv server → EhrSchedulingEnv → Medplum → verify() → reward → LoRA update.
No quality target; it proves every wire before a GPU is rented.

    # terminal 1 (agent venv): cd tests/behavioral && EHR_ENV_SCRIPTED_CALLER=1 PYTHONPATH=../../services/agent:. python -m env.server
    # terminal 2 (training venv):
    cd tests/behavioral && .venv-verifiers/bin/python -m env.train_smoke --steps 3 --generations 4 --model Qwen/Qwen3-0.6B

Writes the adapter and a run summary under results/env/train-smoke/<run-id>/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_BEHAVIORAL = Path(__file__).resolve().parents[1]
if str(_BEHAVIORAL) not in sys.path:
    sys.path.insert(0, str(_BEHAVIORAL))

from env.trl_env import EhrToolEnv, episode_reward  # noqa: E402

TRAIN_SEEDS = tuple(s for s in range(1, 41) if s % 5)  # train split: seeds not divisible by 5
SYSTEM = ("You are the front-desk assistant of a chiropractic clinic, on a phone call. Use the tools: speak to the caller with "
          "say, verify them with verify_patient using the date of birth they said, search with look_up_availability, book "
          "with confirm_appointment after the caller agrees to a time, and finish with end_call. Never describe a visit on "
          "file to an unverified caller.")


def build_dataset(n: int, families, tier: int):
    from datasets import Dataset

    rows = []
    for i in range(n):
        family = families[i % len(families)]
        rows.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "[call connecting]"}],
                     "family": family, "seed": TRAIN_SEEDS[i % len(TRAIN_SEEDS)], "tier": tier})
    return Dataset.from_list(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--families", default="booking")
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--max-completion", type=int, default=1536)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--out", type=Path, default=_BEHAVIORAL / "results" / "env" / "train-smoke")
    a = p.parse_args(argv)

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = a.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    families = [f for f in a.families.split(",") if f]
    dataset = build_dataset(a.steps * a.generations * 2, families, a.tier)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = GRPOConfig(
        output_dir=str(out), max_steps=a.steps, num_generations=a.generations, per_device_train_batch_size=a.generations,
        gradient_accumulation_steps=1, learning_rate=a.lr, max_completion_length=a.max_completion,
        temperature=1.0, chat_template_kwargs={"enable_thinking": False}, log_completions=True, logging_steps=1,
        save_strategy="no", report_to=[], bf16=False, fp16=False, use_vllm=False, use_cpu=(device == "cpu"),
    )
    t0 = time.time()
    # Load on the CPU in fp32 ourselves: transformers 5's threaded weight loader aborts when the trainer loads straight
    # onto MPS ("Fatal Python error: Aborted" in core_model_loading._materialize_copy, 2026-10-07). The trainer moves
    # the model to the device afterwards.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=config, train_dataset=dataset, reward_funcs=[episode_reward],
                          environment_factory=EhrToolEnv,
                          peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    history = [h for h in trainer.state.log_history if "reward" in h or "loss" in h]
    summary = {"run_id": run_id, "model": a.model, "device": device, "steps": a.steps, "generations": a.generations,
               "families": families, "tier": a.tier, "seconds": round(time.time() - t0, 1), "log_history": history,
               "env_url": os.environ.get("EHR_ENV_URL", "ws://127.0.0.1:8011")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))
    for h in history:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k in ("step", "loss", "reward", "reward_std", "kl", "completions/mean_length")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
