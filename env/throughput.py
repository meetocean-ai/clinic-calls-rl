"""Reset latency and episode throughput of the EHR environment against the local Medplum (roadmap P1-01 / P1-02).

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.throughput --resets 20 --concurrency 1,4,8 --episodes 16

Reports `reset` p50/p95 (provision tenant + seed world + first caller line) and oracle episodes per hour at each
concurrency level, and checks that concurrent episodes never cross-talk (every oracle episode still scores 1 — each
episode is its own Organization compartment). Numbers go into the roadmap's evidence column; nothing is written to
the repo by this script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List

from .env import EhrSchedulingEnv
from .policies import OraclePolicy, run_episode
from .tasks import FAMILIES, HELDOUT_EVAL_SEEDS


def _pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))], 3)


async def measure_resets(medplum, n: int, tier: int = 3) -> Dict[str, Any]:
    """reset = provision the tenant + seed the world + the caller's first line; close = the real tenant-delete workflow."""
    resets: List[float] = []
    closes: List[float] = []
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        seed = HELDOUT_EVAL_SEEDS[i % len(HELDOUT_EVAL_SEEDS)]
        t0 = time.perf_counter()
        await env.reset(seed=seed, family=family, tier=tier if family != "emergency_redirect" else 1)
        t1 = time.perf_counter()
        await env.close()
        resets.append(t1 - t0)
        closes.append(time.perf_counter() - t1)
    return {"n": n, "reset": {"p50_s": _pct(resets, 0.5), "p95_s": _pct(resets, 0.95), "max_s": _pct(resets, 1.0)},
            "teardown": {"p50_s": _pct(closes, 0.5), "p95_s": _pct(closes, 0.95), "max_s": _pct(closes, 1.0)}}


async def measure_throughput(medplum, concurrency: int, episodes: int, tier: int = 3) -> Dict[str, Any]:
    """`episodes` oracle episodes spread over `concurrency` workers; wall-clock → episodes/hour."""
    jobs = [(FAMILIES[i % len(FAMILIES)], HELDOUT_EVAL_SEEDS[(i // len(FAMILIES)) % len(HELDOUT_EVAL_SEEDS)]) for i in range(episodes)]
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    rewards: List[int] = []
    errors: List[str] = []

    async def worker() -> None:
        while True:
            try:
                family, seed = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            env = EhrSchedulingEnv(medplum, scripted_caller=True)
            r = await run_episode(env, OraclePolicy(), seed=seed, family=family, tier=tier if family != "emergency_redirect" else 1)
            rewards.append(r.reward or 0)
            if r.error or r.reward != 1:
                errors.append(f"{r.task_id}: {r.error or r.diffs}")

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "episodes": episodes, "wall_s": round(wall, 1),
            "episodes_per_hour": round(episodes / wall * 3600), "oracle_pass": sum(rewards), "errors": errors}


async def main_async(a) -> Dict[str, Any]:
    from adapters.medplum import MedplumAdapter

    medplum = await MedplumAdapter.from_env()
    out: Dict[str, Any] = {}
    try:
        out["reset"] = await measure_resets(medplum, a.resets)
        out["throughput"] = [await measure_throughput(medplum, c, a.episodes) for c in a.concurrency]
    finally:
        await medplum.close()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resets", type=int, default=20)
    p.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 4, 8])
    p.add_argument("--episodes", type=int, default=16)
    a = p.parse_args(argv)
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        here = Path(__file__).resolve().parents[1]
        load_dotenv(here / ".env", override=False)
        load_dotenv(here.parents[1] / ".env", override=False)
    except ImportError:
        pass
    out = asyncio.run(main_async(a))
    print(json.dumps(out, indent=2))
    r = out["reset"]
    print(f"\nreset: p50 {r['reset']['p50_s']} s · p95 {r['reset']['p95_s']} s · max {r['reset']['max_s']} s over {r['n']}; "
          f"teardown: p50 {r['teardown']['p50_s']} s · p95 {r['teardown']['p95_s']} s")
    for t in out["throughput"]:
        print(f"concurrency {t['concurrency']}: {t['episodes_per_hour']} oracle episodes/hour "
              f"({t['episodes']} in {t['wall_s']} s), oracle {t['oracle_pass']}/{t['episodes']}, errors {len(t['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["measure_resets", "measure_throughput"]
