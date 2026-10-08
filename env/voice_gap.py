"""The voice gap (roadmap P2-04): the same policy on the same held-out tasks, once reading the caller's text and once
hearing it through the ASR cascade (`--voice`). gap = pass@1(text) − pass@1(voice), per family and overall, with
Wilson intervals on both rates, the paired task counts (lost / gained in voice) and what the ear did (mean transcript
similarity, lines below the 0.85 floor).

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.voice_gap \
        --text results/env/p204/<run-id>/openai-compatible-qwen3-8b --voice results/env/p204/<run-id>/openai-compatible-qwen3-8b+voice

Both arguments are run directories of manifests (or `--json-out` files). Pairs are matched on (family, task_id); a
task present on one side only is reported, not silently dropped.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scorecard import load, wilson


def _by_task(results: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        out.setdefault((r["family"], r["task_id"]), []).append(r)
    return out


def _rate(runs: List[Dict[str, Any]]) -> float:
    return sum(1 for r in runs if (r.get("reward") or 0) == 1) / len(runs) if runs else 0.0


def voice_gap_rows(text: List[Dict[str, Any]], voice: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{family | "all": row}. A task counts as *lost* when every text trial passed and the voice mean is lower, *gained*
    the other way round; the gap is over all trials (pass@1), so k=3 on each side is 3 vs 3 per task."""
    t_by, v_by = _by_task(text), _by_task(voice)
    families = sorted({k[0] for k in t_by} | {k[0] for k in v_by})
    rows: Dict[str, Dict[str, Any]] = {}
    for fam in families + ["all"]:
        keys = [k for k in set(t_by) | set(v_by) if fam == "all" or k[0] == fam]
        paired = [k for k in keys if k in t_by and k in v_by]
        t_runs = [r for k in paired for r in t_by[k]]
        v_runs = [r for k in paired for r in v_by[k]]
        t_pass, v_pass = sum(1 for r in t_runs if (r.get("reward") or 0) == 1), sum(1 for r in v_runs if (r.get("reward") or 0) == 1)
        asr = [r["asr"] for r in v_runs if r.get("asr")]
        rows[fam] = {
            "tasks": len(paired), "unpaired": sorted(f"{k[1]}:{'text' if k in t_by else 'voice'}-only" for k in keys if k not in paired),
            "k": max([len(t_by[k]) for k in paired] + [len(v_by[k]) for k in paired] + [0]),
            "text_pass": round(t_pass / len(t_runs), 3) if t_runs else None, "text_ci": wilson(t_pass, len(t_runs)),
            "voice_pass": round(v_pass / len(v_runs), 3) if v_runs else None, "voice_ci": wilson(v_pass, len(v_runs)),
            "gap": round(t_pass / len(t_runs) - v_pass / len(v_runs), 3) if t_runs and v_runs else None,
            "lost": sum(1 for k in paired if _rate(t_by[k]) > _rate(v_by[k])),
            "gained": sum(1 for k in paired if _rate(v_by[k]) > _rate(t_by[k])),
            "text_errors": sum(1 for r in t_runs if r.get("error")), "voice_errors": sum(1 for r in v_runs if r.get("error")),
            "asr_mean_similarity": round(statistics.fmean(a["mean_similarity"] for a in asr), 3) if asr else None,
            "asr_lines": sum(a["lines"] for a in asr), "asr_below_floor": sum(a["below_floor"] for a in asr),
            "voice_top_diffs": _top_diffs(v_runs),
        }
    return rows


def _top_diffs(runs: List[Dict[str, Any]], n: int = 3) -> List[str]:
    c: Dict[str, int] = {}
    for r in runs:
        for d in r.get("diffs") or []:
            k = d.split(":", 1)[0]
            c[k] = c.get(k, 0) + 1
    return [f"{k} x{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:n]]


def _ci(ci) -> str:
    return f"{ci[0]:.2f}–{ci[1]:.2f}" if ci else "-"


def voice_gap_table(rows: Dict[str, Dict[str, Any]], *, text_id: str = "text", voice_id: str = "voice") -> str:
    head = (f"| family | tasks | k | {text_id} pass@1 | 95% CI | {voice_id} pass@1 | 95% CI | voice gap | lost / gained | "
            "ASR similarity | lines < 0.85 | voice top diffs |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for fam, r in rows.items():
        lines.append(f"| {'**all**' if fam == 'all' else fam} | {r['tasks']} | {r['k']} | {r['text_pass']} | {_ci(r['text_ci'])} | "
                     f"{r['voice_pass']} | {_ci(r['voice_ci'])} | {('%+.3f' % r['gap']) if r['gap'] is not None else '-'} | "
                     f"{r['lost']} / {r['gained']} | {r['asr_mean_similarity'] if r['asr_mean_similarity'] is not None else '-'} | "
                     f"{r['asr_below_floor']}/{r['asr_lines']} | {', '.join(r['voice_top_diffs']) or '-'} |")
    unpaired = sorted({u for r in rows.values() for u in r["unpaired"]})
    if unpaired:
        lines += ["", "Unpaired tasks (not in the gap): " + ", ".join(unpaired)]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=Path, required=True, help="run directory (or --json-out) of the text-mode sweep")
    p.add_argument("--voice", type=Path, required=True, help="run directory (or --json-out) of the --voice sweep")
    p.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args(argv)
    text, voice = load(a.text)["results"], load(a.voice)["results"]
    rows = voice_gap_rows(text, voice)
    ids = ({r["policy_id"] for r in text} or {"text"}), ({r["policy_id"] for r in voice} or {"voice"})
    print(voice_gap_table(rows, text_id=sorted(ids[0])[0], voice_id=sorted(ids[1])[0]))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({"text": str(a.text), "voice": str(a.voice), "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["voice_gap_rows", "voice_gap_table"]
