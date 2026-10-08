"""
Scorecard for the EHR environment: pass rates per policy, family and tier with Wilson 95% intervals, escalation
rate, per-turn latency percentiles, the dataset / method versions of every row, and a diff column against a
baseline. Reads a run directory of manifests (`results/env/<run>/<policy>/<task>/<trial>.manifest.json`) or the
`--json-out` summary. Same columns as the assistant repo's, so the two tables read alike.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.scorecard results/env/bracket-t1 results/env/bracket-t3
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def wilson(successes: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a proportion (the interval Artificial Analysis publishes next to every rate)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 1)


def latency_summary(steps: Any) -> Dict[str, Any]:
    """Per-turn wall-clock of the production policy's turns (LLM + tools) with P50/P95/P99 — the
    percentile an SLA is judged at is P99, not the median. Text mode: this is the model + tool budget
    per turn, not time-to-first-audio. Accepts StepRecords or step dicts."""
    turns: List[float] = []
    for s in steps:
        a = s.get("action") if isinstance(s, dict) else getattr(s, "action", None)
        v = (a or {}).get("turn_ms")
        if v:
            turns.append(float(v))
    return {"turns": [round(t, 1) for t in turns], "p50": _pct(turns, 0.5), "p95": _pct(turns, 0.95), "p99": _pct(turns, 0.99), "max": _pct(turns, 1.0)}


def load_run_dir(run_dir: Path) -> Dict[str, Any]:
    """`{"results": [...]}` from every `<policy>/<task>/<trial>.manifest.json` under a run directory."""
    results: List[Dict[str, Any]] = []
    # a run (`<run>/<policy>/<task>/<trial>`) or one policy of a run (`<run>/<policy>`): manifests at either depth
    for path in sorted(Path(run_dir).rglob("*.manifest.json")):
        m = json.loads(path.read_text())
        if "turns" not in m:  # older manifests: count from the step log
            m.update(_counts_from_steps(path.with_name(path.name.replace(".manifest.json", ".jsonl"))))
        results.append({
            "policy_id": m.get("policy_id") or path.parent.parent.name, "family": m["family"], "tier": int(m.get("tier", 1)),
            "task_id": m.get("task_id") or path.parent.name, "trial": int(path.name.split(".")[0]),
            "reward": m.get("reward"), "turns": int(m.get("turns") or 0), "tool_calls": int(m.get("tool_calls") or 0),
            "seconds": float(m.get("seconds") or 0.0), "error": m.get("error") or "", "judge": m.get("judge"),
            "tau": m.get("tau"), "dataset_version": m.get("dataset_version", ""), "method_version": m.get("method_version", ""),
            "stopped_reason": m.get("stopped_reason", ""), "latency_ms": m.get("latency_ms") or {},
            "chaos": list(m.get("chaos") or []), "diffs": list(m.get("diffs") or []),
            "asr": (m.get("asr") or {}).get("summary"),  # cascaded voice policies: what the ear did this episode
        })
    return {"results": results, "source": str(run_dir)}


def _counts_from_steps(steps_path: Path) -> Dict[str, int]:
    """turns = agent `say` steps; tool_calls = tool / end_call / transfer steps plus the tool names a production
    `say` step carries."""
    turns = tools = 0
    if steps_path.exists():
        for line in steps_path.read_text().splitlines():
            row = json.loads(line)
            a = row.get("action") or {}
            if a.get("kind") == "say":
                turns += 1
                tools += len(a.get("tools") or [])
            elif a.get("kind") in ("tool", "end_call", "transfer"):
                tools += 1
    return {"turns": turns, "tool_calls": tools}


def load(path: Path) -> Dict[str, Any]:
    return load_run_dir(path) if Path(path).is_dir() else json.loads(Path(path).read_text())


def _rows(results: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    """{(policy, family, tier): row} with mean reward (+ Wilson CI), pass^k (+ CI), pass@k, turns, tools, errors,
    seconds, judge means, τ and the dataset / method versions the runs carry."""
    grouped: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = {}
    for r in results:
        key = (r["policy_id"], r["family"], int(r.get("tier", 1)))
        grouped.setdefault(key, {}).setdefault(r["task_id"], []).append(r)
    rows: Dict[tuple, Dict[str, Any]] = {}
    for key, tasks in grouped.items():
        runs = [r for rs in tasks.values() for r in rs]
        passes = sum(1 for r in runs if (r.get("reward") or 0) == 1)
        all_pass = sum(all((r.get("reward") or 0) == 1 for r in rs) for rs in tasks.values())
        rows[key] = {
            "tasks": len(tasks),
            "k": max(len(rs) for rs in tasks.values()),
            "n": len(runs),
            "mean_reward": round(passes / len(runs), 3),
            "mean_reward_ci": wilson(passes, len(runs)),
            "pass_pow_k": round(all_pass / len(tasks), 3),
            "pass_pow_k_ci": wilson(all_pass, len(tasks)),
            "pass_at_k": round(sum(any((r.get("reward") or 0) == 1 for r in rs) for rs in tasks.values()) / len(tasks), 3),
            "turns": round(statistics.fmean(r.get("turns", 0) for r in runs), 1),
            "tool_calls": round(statistics.fmean(r.get("tool_calls", 0) for r in runs), 1),
            "seconds": round(statistics.fmean(r.get("seconds", 0.0) for r in runs), 1),
            "errors": sum(1 for r in runs if r.get("error")),
            "judge": _judge_means(runs),
            "tau": round(statistics.fmean(int((r.get("tau") or {}).get("reward", 0)) for r in runs), 3) if any(r.get("tau") for r in runs) else None,
            # Resolution vs containment: the reward is resolution; this is how often the agent handed the call to a human.
            "transfer_rate": round(sum(1 for r in runs if r.get("stopped_reason") == "transfer") / len(runs), 3),
            "latency": _pooled_latency(runs),
            "dataset_versions": sorted({r.get("dataset_version") or "" for r in runs} - {""}),
            "method_versions": sorted({r.get("method_version") or "" for r in runs} - {""}),
        }
    return rows


def _pooled_latency(runs: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    turns = [t for r in runs for t in ((r.get("latency_ms") or {}).get("turns") or [])]
    if not turns:
        return None
    return {"p50": _pct(turns, 0.5), "p95": _pct(turns, 0.95), "p99": _pct(turns, 0.99), "n": len(turns)}


def _judge_means(runs: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Task 20: mean judge item over judged runs; None when the run had no judge. Never part of reward."""
    judged = [r["judge"] for r in runs if r.get("judge") and not r["judge"].get("error")]
    if not judged:
        return None
    items = ("consent_before_booking", "verified_before_disclosure")
    return {item: round(statistics.fmean(int(v.get(item, 0)) for v in judged), 3) for item in items}


def _judge_cell(row: Dict[str, Any]) -> str:
    j = row.get("judge")
    return f"{j['consent_before_booking']} / {j['verified_before_disclosure']}" if j else "-"


def _latency_cell(row: Dict[str, Any]) -> str:
    lat = row.get("latency")
    return f"{lat['p50']:.0f} / {lat['p99']:.0f}" if lat else "-"


def _ci_cell(ci: Optional[Tuple[float, float]]) -> str:
    return f"[{ci[0]}, {ci[1]}]" if ci else "-"


def _versions_cell(values: List[str]) -> str:
    return ", ".join(values) if values else "-"


def scorecard(current: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> str:
    cur = _rows(current["results"])
    base = _rows(baseline["results"]) if baseline else {}
    head = ("| policy | family | tier | tasks | k | mean reward | Δ | 95% CI | pass^k | Δ | 95% CI | pass@k | escalated | turns | tools | "
            "turn p50/p99 ms | s/episode | errors | judge consent / verified-first | τ | dataset | method |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for key in sorted(cur):
        c = cur[key]
        b = base.get(key)

        def delta(name: str) -> str:
            if not b:
                return "new"
            d = c[name] - b[name]
            return f"{d:+.3f}" if d else "="

        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {c['tasks']} | {c['k']} | {c['mean_reward']} | {delta('mean_reward')} | "
            f"{_ci_cell(c['mean_reward_ci'])} | {c['pass_pow_k']} | {delta('pass_pow_k')} | {_ci_cell(c['pass_pow_k_ci'])} | "
            f"{c['pass_at_k']} | {c['transfer_rate']} | {c['turns']} | {c['tool_calls']} | {_latency_cell(c)} | {c['seconds']} | {c['errors']} | {_judge_cell(c)} | "
            f"{c.get('tau') if c.get('tau') is not None else '-'} | {_versions_cell(c['dataset_versions'])} | "
            f"{_versions_cell(c['method_versions'])} |"
        )
    gone = sorted(set(base) - set(cur))
    if gone:
        lines.append("")
        lines.append("Not in this run: " + ", ".join("/".join(map(str, k)) for k in gone))
    return "\n".join(lines)


def knob_rows(results: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    """{(policy, knob): row} — pass rate (+ Wilson CI) over every episode whose task carries the knob, whatever else
    it was combined with (tier 4 stacks two or three), plus a `(policy, "— none —")` row for knob-free episodes and
    the check that failed most often. This is where a trained model's gain shows per difficulty, not per family."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        for knob in (r.get("chaos") or ["— none —"]):
            grouped.setdefault((r["policy_id"], knob), []).append(r)
    rows: Dict[tuple, Dict[str, Any]] = {}
    for key, runs in grouped.items():
        passes = sum(1 for r in runs if (r.get("reward") or 0) == 1)
        failed_checks: Counter = Counter(d.split(":", 1)[0] for r in runs for d in (r.get("diffs") or []))
        rows[key] = {
            "n": len(runs), "tasks": len({r["task_id"] for r in runs}),
            "pass_rate": round(passes / len(runs), 3), "pass_rate_ci": wilson(passes, len(runs)),
            "alone": sum(1 for r in runs if len(r.get("chaos") or []) <= 1),
            "top_diff": (failed_checks.most_common(1) or [("-", 0)])[0][0],
        }
    return rows


def knob_table(current: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> str:
    cur = knob_rows(current["results"])
    base = knob_rows(baseline["results"]) if baseline else {}
    head = "| policy | knob | episodes | tasks | alone | pass rate | Δ | 95% CI | most failed check |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for key in sorted(cur, key=lambda k: (k[0], k[1] == "— none —", k[1])):
        c, b = cur[key], base.get(key)
        delta = "new" if not b else (f"{c['pass_rate'] - b['pass_rate']:+.3f}" if c["pass_rate"] != b["pass_rate"] else "=")
        lines.append(f"| {key[0]} | {key[1]} | {c['n']} | {c['tasks']} | {c['alone']} | {c['pass_rate']} | {delta} | "
                     f"{_ci_cell(c['pass_rate_ci'])} | {c['top_diff']} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("current", type=Path, nargs="+", help="JSON from `policies --json`, or run directories of manifests")
    parser.add_argument("--baseline", type=Path, default=None, help="JSON or run directory of the last accepted run")
    parser.add_argument("--knobs", action="store_true", help="also print the per-knob breakdown")
    args = parser.parse_args(argv)
    current = {"results": [r for p in args.current for r in load(p)["results"]]}
    baseline = load(args.baseline) if args.baseline else None
    print(scorecard(current, baseline))
    if args.knobs:
        print()
        print(knob_table(current, baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["knob_rows", "knob_table", "latency_summary", "load", "load_run_dir", "scorecard", "wilson"]
