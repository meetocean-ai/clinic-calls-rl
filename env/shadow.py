"""Shadow runner (roadmap P4-02): on every release, run a candidate model through the environment next to production —
text and voice, the held-out set — and land the runs in the quality series as `rl-env-candidate-<name>…` suites, where
the leaderboard and the graduation gate (P4-03) read them.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.shadow --name qwen3-8b-lora-r16-s400 \
        --endpoint http://vllm:8000/v1 --model qwen3-8b-lora-r16-s400 [--no-think] [--seeds 10 --k 3 --tier 3] \
        [--out results/env/shadow] [--skip-voice] [--no-record]

Each sweep is the ordinary `env.policies` sweep with the policy relabelled `candidate-<name>` (and `candidate-<name>+voice`),
so a candidate is never mistaken for production. After the sweeps: `tests/quality/build.py record --env <run>/<policy>` for
both, the leaderboard regenerates, and the graduation verdict for the candidate is printed (eligible only if ≥ production
on held-out pass@1 with non-overlapping intervals, zero wrong writes, zero claimed-without-tool).
"""
from __future__ import annotations

import argparse
import subprocess  # nosec B404 — fixed argv: this repo's own scripts
import sys
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
BEHAVIORAL = HERE.parents[1]


def sweep_args(a, *, voice: bool) -> List[str]:
    args = ["--policy", "openai-compatible", "--endpoint", a.endpoint, "--model", a.model, "--policy-id", f"candidate-{a.name}",
            "--split", "heldout", "--seeds", str(a.seeds), "--k", str(a.k), "--tier", str(a.tier), "--scripted",
            "--out", str(a.out), "--run-id", a.run_id]
    if a.no_think:
        args.append("--no-think")
    if voice:
        args.append("--voice")
    return args


def run_sweeps(a) -> List[Path]:
    """The text sweep, then the voice sweep; returns the policy folders to record."""
    from . import policies

    folders = []
    for voice in ([False] if a.skip_voice else [False, True]):
        print(f"== shadow {'voice' if voice else 'text'} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        rc = policies.main(sweep_args(a, voice=voice))
        if rc != 0:
            raise SystemExit(f"sweep failed ({rc})")
        folders.append(Path(a.out) / a.run_id / (f"candidate-{a.name}" + ("+voice" if voice else "")))
    return folders


def record(folders: List[Path]) -> str:
    cmd = [sys.executable, str(REPO_ROOT / "tests" / "quality" / "build.py"), "record"]
    for f in folders:
        cmd += ["--env", str(f)]
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout  # nosec B603
    subprocess.run([sys.executable, str(REPO_ROOT / "tests" / "quality" / "build.py"), "leaderboard"], cwd=REPO_ROOT, check=True, capture_output=True)  # nosec B603
    return out.strip()


def verdict(name: str) -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location("quality_graduation", REPO_ROOT / "tests" / "quality" / "graduation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render_for(f"candidate-{name}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="candidate checkpoint name → policy id candidate-<name>")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--tier", type=int, default=3)
    p.add_argument("--out", type=Path, default=BEHAVIORAL / "results" / "env" / "shadow")
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--skip-voice", action="store_true")
    p.add_argument("--no-record", action="store_true", help="run the sweeps, do not touch the quality series")
    a = p.parse_args(argv)
    folders = run_sweeps(a)
    if a.no_record:
        print("\n".join(str(f) for f in folders))
        return 0
    print(record(folders))
    print(verdict(a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "record", "run_sweeps", "sweep_args", "verdict"]
