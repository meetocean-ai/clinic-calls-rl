"""
A human-readable report for an EHR-environment run (or several): what the pipeline is, how every policy scored with
intervals, and every miss with its transcript and judge flags. One self-contained HTML file, no
runtime, no network — opens from disk and publishes as-is.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.report results/env/bracket-t1 results/env/bracket-t3 \
        --title "EHR Scheduling Bracket" --out results/env/bracket.html
    python -m env.report results/env/my-run --baseline results/env/last-accepted --out run.html

Reads the same run directories the scorecard reads (`<policy>/<task>/<trial>.manifest.json` + `.jsonl`).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scorecard import _rows, knob_rows, load_run_dir

# ── data ─────────────────────────────────────────────────────────────────


def _transcript(steps_path: Path) -> List[Dict[str, Any]]:
    """Agent turns with the tools used and the caller's reply, from the step log."""
    turns: List[Dict[str, Any]] = []
    pending: List[str] = []
    opening = ""
    if not steps_path.exists():
        return turns
    for line in steps_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("i") == -1:
            opening = (row.get("observation") or {}).get("patient_text") or ""
            continue
        a, o = row.get("action") or {}, row.get("observation") or {}
        kind = a.get("kind")
        if kind in ("tool", "end_call", "transfer"):
            name = a.get("tool_name") or {"end_call": "end_call", "transfer": "warm_transfer_to_human"}[kind]
            res = o.get("tool_result")
            ok = None
            if isinstance(res, dict):
                ok = bool(res.get("booked") or res.get("cancelled") or res.get("rescheduled") or res.get("verified")
                          or res.get("transferred") or res.get("ended") or res.get("found") or res.get("appointments") is not None)
                if res.get("error") or res.get("status") == 409:
                    ok = False
            pending.append(name + ("" if ok is None else (" ✓" if ok else " ✗")))
        elif kind == "say":
            pending = list(a.get("tools") or []) + pending
            turns.append({"agent": a.get("text") or "", "tools": pending, "caller": o.get("patient_text") or ""})
            pending = []
    if opening:
        turns.insert(0, {"agent": "", "tools": [], "caller": opening, "opening": True})
    if pending:
        turns.append({"agent": "", "tools": pending, "caller": ""})
    return turns


def collect(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("*/*/*.manifest.json")):
            m = json.loads(path.read_text())
            summary = ((m.get("turn_scores") or {}).get("summary") or {})
            flags = {name: s.get("failed_turns") or [] for name, s in summary.items() if s.get("failed_turns")}
            episodes.append({
                "run": Path(run_dir).name, "path": str(path), "policy_id": m.get("policy_id") or path.parent.parent.name,
                "task_id": m.get("task_id") or path.parent.name, "trial": int(path.name.split(".")[0]),
                "family": m.get("family"), "tier": int(m.get("tier", 1)), "seed": m.get("seed"), "chaos": m.get("chaos") or [],
                "reward": m.get("reward"), "diffs": m.get("diffs") or [], "stopped_reason": m.get("stopped_reason") or "",
                "error": m.get("error") or "", "tau": (m.get("tau") or {}), "flags": flags,
                "judge": m.get("judge") or {}, "dataset_version": m.get("dataset_version", ""),
                "claim_rate": (summary.get("claim_without_tool") or {}).get("pass_rate"),
                "method_version": m.get("method_version", ""),
                "transcript": _transcript(path.with_name(path.name.replace(".manifest.json", ".jsonl"))),
            })
    return episodes


# ── html ─────────────────────────────────────────────────────────────────

_E = html.escape

_CSS = """
:root{--bg:#F5F6F3;--paper:#FFFFFF;--ink:#1B2430;--muted:#5D6873;--line:#D9DDD8;--accent:#0F766E;--accent-ink:#0B5C56;
--good:#2F7D4F;--good-bg:#E4F2E9;--warn:#A8681A;--warn-bg:#F7ECD8;--bad:#B23A48;--bad-bg:#F8E3E6;--chip:#EEF0EC;--code:#EDF0EE}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151A1F;--paper:#1D242B;--ink:#E7EAEE;--muted:#9AA5B1;
--line:#2E3740;--accent:#3FB8AD;--accent-ink:#7ED9D0;--good:#6CC48F;--good-bg:#1E3527;--warn:#E2A85A;--warn-bg:#3A2E1A;
--bad:#F08594;--bad-bg:#40232A;--chip:#28313A;--code:#242C34}}
:root[data-theme="dark"]{--bg:#151A1F;--paper:#1D242B;--ink:#E7EAEE;--muted:#9AA5B1;--line:#2E3740;--accent:#3FB8AD;
--accent-ink:#7ED9D0;--good:#6CC48F;--good-bg:#1E3527;--warn:#E2A85A;--warn-bg:#3A2E1A;--bad:#F08594;--bad-bg:#40232A;--chip:#28313A;--code:#242C34}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;font-size:15px;line-height:1.5;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 64px;padding-inline:20px}
h1,h2,h3{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;text-wrap:balance;margin:0}
h1{font-size:2rem;line-height:1.15}h2{font-size:1.35rem;margin-block:40px 12px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:1.05rem;margin-block:20px 8px}
p{max-width:68ch}.muted{color:var(--muted)}.eyebrow{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-ink);font-weight:600}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em}
code{background:var(--code);padding:1px 5px;border-radius:4px}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.stat{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.stat .n{font-family:"IBM Plex Serif",Georgia,serif;font-size:1.7rem;line-height:1.1;font-variant-numeric:tabular-nums}
.stat .l{font-size:.78rem;color:var(--muted);margin-top:2px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}
th{font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--paper);position:sticky;top:0}
tr:last-child td{border-bottom:0}td.num{text-align:right}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;font-weight:600;font-variant-numeric:tabular-nums}
.good{background:var(--good-bg);color:var(--good)}.warn{background:var(--warn-bg);color:var(--warn)}.bad{background:var(--bad-bg);color:var(--bad)}
.chip{display:inline-block;background:var(--chip);border-radius:4px;padding:0 6px;font-size:.78rem;margin-right:4px;font-family:"IBM Plex Mono",monospace}
.bar{height:6px;background:var(--chip);border-radius:3px;overflow:hidden;min-width:80px}.bar i{display:block;height:100%;background:var(--accent)}
details{background:var(--paper);border:1px solid var(--line);border-radius:6px;margin-top:10px}
summary{cursor:pointer;padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.body{padding:4px 14px 14px;border-top:1px solid var(--line)}
.diff{color:var(--bad);font-family:"IBM Plex Mono",monospace;font-size:.82rem}
.turn{display:grid;grid-template-columns:5.5rem 1fr;gap:6px 12px;padding:6px 0;border-bottom:1px dashed var(--line);font-size:.9rem}
.turn:last-child{border-bottom:0}.who{font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding-top:3px}
.turn .tools{grid-column:2;font-size:.78rem;color:var(--accent-ink);font-family:"IBM Plex Mono",monospace}
.flag{border-left:3px solid var(--bad);padding-left:8px}
.diagram{overflow-x:auto;background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:12px}
.diagram svg{display:block;min-width:900px;width:100%;height:auto;font-family:"IBM Plex Sans",system-ui,sans-serif}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px 24px;margin-top:12px;font-size:.9rem}
.legend b{color:var(--accent-ink)}
@media (max-width:600px){.turn{grid-template-columns:1fr}.turn .tools{grid-column:1}}
@media (prefers-reduced-motion:no-preference){details[open] .body{animation:fade .18s ease-out}@keyframes fade{from{opacity:.4}to{opacity:1}}}
"""

_FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono&display=swap">'


def _pipeline_svg() -> str:
    """The run, as boxes: seed → task → world + caller → policy → episode → verify → reward, and the
    logging rail underneath that every episode feeds."""
    boxes = [
        ("seed", "Seed + family + tier", "chaos knobs drawn deterministically", 0),
        ("task", "Task", "patient, DOB, offered slots, expected end state", 1),
        ("world", "World on local Medplum", "a fresh Organization per episode: practitioners, schedules, the patient, the visit on file", 2),
        ("policy", "Policy", "Reception → Scheduler via drive_inbound · oracle · random", 3),
        ("episode", "Episode", "tools_fhir writes, tenant-scoped; simulated caller replies", 4),
        ("verify", "verify()", "FHIR end state → reward 0/1 + named diffs; tenant torn down", 5),
    ]
    rail = [
        ("traj", "Trajectory + manifest", "steps, reward, diffs, dataset/method version, τ, per-turn judge verdicts"),
        ("runs", "Run directory", "results/env/<run>/<policy>/<task>/<trial>"),
        ("score", "Scorecard + report", "pass^k, pass@k, Wilson 95% CI, misses with transcripts"),
        ("fix", "Triage → fix branch", "a miss becomes a unit-tested code fix, then a re-run"),
    ]
    W, H, gap, top = 170, 72, 22, 30
    parts = [f'<svg viewBox="0 0 {6 * (W + gap) + 10} 300" role="img" aria-label="How a run works">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>']
    for _, title, sub, i in boxes:
        x = 10 + i * (W + gap)
        parts.append(f'<rect x="{x}" y="{top}" width="{W}" height="{H}" rx="6" fill="var(--paper)" stroke="var(--accent)" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 10}" y="{top + 24}" font-size="14" font-weight="600" fill="var(--ink)">{_E(title)}</text>')
        parts.append(_wrap_text(sub, x + 10, top + 42, W - 20, 11, "var(--muted)"))
        if i < 5:
            parts.append(f'<line x1="{x + W}" y1="{top + H / 2}" x2="{x + W + gap - 2}" y2="{top + H / 2}" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>')
    # reward + logging rail
    ry = 170
    parts.append(f'<path d="M{10 + 5 * (W + gap) + W / 2},{top + H} L{10 + 5 * (W + gap) + W / 2},{ry - 8}" stroke="var(--muted)" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>')
    rw = 250
    for j, (_, title, sub) in enumerate(rail):
        x = 10 + 6 * (W + gap) - (4 - j) * (rw + gap) - 2
        parts.append(f'<rect x="{x}" y="{ry}" width="{rw}" height="{H + 10}" rx="6" fill="var(--paper)" stroke="var(--line)" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 10}" y="{ry + 24}" font-size="14" font-weight="600" fill="var(--ink)">{_E(title)}</text>')
        parts.append(_wrap_text(sub, x + 10, ry + 42, rw - 20, 11, "var(--muted)"))
        if j < 3:
            parts.append(f'<line x1="{x + rw}" y1="{ry + (H + 10) / 2}" x2="{x + rw + gap - 2}" y2="{ry + (H + 10) / 2}" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>')
    # loop back from fix to seed
    x_fix = 10 + 6 * (W + gap) - (rw + gap) - 2 + rw / 2
    parts.append(f'<path d="M{x_fix},{ry + H + 10} L{x_fix},{ry + H + 34} L{10 + W / 2},{ry + H + 34} L{10 + W / 2},{ry + H + 34}" stroke="var(--accent)" stroke-width="1.5" fill="none" stroke-dasharray="5 4"/>')
    parts.append(f'<path d="M{10 + W / 2},{ry + H + 34} L{10 + W / 2},{top + H + 8}" stroke="var(--accent)" stroke-width="1.5" fill="none" stroke-dasharray="5 4" marker-end="url(#arr)"/>')
    parts.append(f'<text x="{10 + W / 2 + 8}" y="{ry + H + 30}" font-size="11" fill="var(--accent-ink)">re-run the same seeds after the fix — the reward is the regression test</text>')
    parts.append("</svg>")
    return "".join(parts)


def _wrap_text(text: str, x: float, y: float, width: float, size: int, fill: str) -> str:
    words, lines, cur = text.split(), [], ""
    per_line = int(width / (size * 0.55))
    for w in words:
        if len(cur) + len(w) + 1 > per_line and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "".join(f'<text x="{x}" y="{y + i * (size + 3)}" font-size="{size}" fill="{fill}">{_E(l)}</text>' for i, l in enumerate(lines[:3]))


def _pill(v: Optional[float], good=0.999, warn=0.66) -> str:
    if v is None:
        return '<span class="muted">-</span>'
    cls = "good" if v >= good else ("warn" if v >= warn else "bad")
    return f'<span class="pill {cls}">{v:.3f}</span>'


def _leaderboard(run_dirs: List[Path], episodes: List[Dict[str, Any]], baseline_dirs: Optional[List[Path]]) -> str:
    rows = _rows([r for d in run_dirs for r in load_run_dir(d)["results"]])
    base = _rows([r for d in baseline_dirs for r in load_run_dir(d)["results"]]) if baseline_dirs else {}
    out = ['<div class="scroll"><table><thead><tr><th>policy</th><th>family</th><th>tier</th><th>tasks × k</th>'
           '<th>mean reward</th><th>95% CI</th><th>Δ vs baseline</th><th>pass^k</th><th>pass@k</th><th>escalated</th><th>turns</th><th>tools</th>'
           '<th>turn p50 / p99 ms</th><th>τ</th><th>claim judge</th><th>stop reasons</th></tr></thead><tbody>']
    by_key: Dict[tuple, List[Dict[str, Any]]] = {}
    for e in episodes:
        by_key.setdefault((e["policy_id"], e["family"], e["tier"]), []).append(e)
    for key in sorted(rows):
        r = rows[key]
        eps = by_key[key]
        b = base.get(key)
        delta = "new" if not b else (f"{r['mean_reward'] - b['mean_reward']:+.3f}" if r["mean_reward"] != b["mean_reward"] else "=")
        claim_rates = [e["claim_rate"] for e in eps if e.get("claim_rate") is not None]
        claim = f"{statistics.fmean(claim_rates):.3f}" if claim_rates else "-"
        stops = ", ".join(f"{k} {v}" for k, v in Counter(e["stopped_reason"] for e in eps).most_common(3))
        ci = r["mean_reward_ci"]
        out.append(f"<tr><td>{_E(key[0])}</td><td>{_E(key[1])}</td><td>{key[2]}</td><td class='num'>{r['tasks']} × {r['k']}</td>"
                   f"<td>{_pill(r['mean_reward'])}</td><td class='mono'>[{ci[0]:.2f}, {ci[1]:.2f}]</td><td class='mono'>{delta}</td>"
                   f"<td>{_pill(r['pass_pow_k'])}</td><td class='num'>{r['pass_at_k']}</td><td class='num'>{r['transfer_rate']}</td><td class='num'>{r['turns']}</td><td class='num'>{r['tool_calls']}</td>"
                   f"<td class='num mono'>{(str(int(r['latency']['p50'])) + ' / ' + str(int(r['latency']['p99']))) if r.get('latency') else '-'}</td>"
                   f"<td class='num'>{r['tau'] if r['tau'] is not None else '-'}</td><td class='num'>{claim}</td><td class='muted'>{_E(stops)}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _knob_board(run_dirs: List[Path], baseline_dirs: Optional[List[Path]]) -> str:
    """Pass rate per policy × knob over every episode that carries the knob (alone or stacked at tier 4), with the
    check that failed most often — the difficulty a policy actually loses on, independent of family."""
    rows = knob_rows([r for d in run_dirs for r in load_run_dir(d)["results"]])
    base = knob_rows([r for d in baseline_dirs for r in load_run_dir(d)["results"]]) if baseline_dirs else {}
    out = ['<div class="scroll"><table><thead><tr><th>policy</th><th>knob</th><th>episodes</th><th>tasks</th><th>alone</th>'
           '<th>pass rate</th><th>95% CI</th><th>Δ vs baseline</th><th>most failed check</th></tr></thead><tbody>']
    for key in sorted(rows, key=lambda k: (k[0], k[1] == "— none —", k[1])):
        r, b = rows[key], base.get(key)
        delta = "new" if not b else (f"{r['pass_rate'] - b['pass_rate']:+.3f}" if r["pass_rate"] != b["pass_rate"] else "=")
        ci = r["pass_rate_ci"]
        out.append(f"<tr><td>{_E(key[0])}</td><td><span class='chip'>{_E(key[1])}</span></td><td class='num'>{r['n']}</td><td class='num'>{r['tasks']}</td>"
                   f"<td class='num'>{r['alone']}</td><td>{_pill(r['pass_rate'])}</td><td class='mono'>[{ci[0]:.2f}, {ci[1]:.2f}]</td>"
                   f"<td class='mono'>{delta}</td><td class='mono'>{_E(r['top_diff'])}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _episode_details(e: Dict[str, Any], open_: bool = False) -> str:
    reward = e["reward"]
    pill = '<span class="pill good">reward 1</span>' if reward == 1 else '<span class="pill bad">reward 0</span>'
    chaos = "".join(f'<span class="chip">{_E(c)}</span>' for c in e["chaos"]) or '<span class="muted">no knobs</span>'
    flags = "".join(f'<span class="pill warn">{_E(n)}: turn {", ".join(map(str, t))}</span>' for n, t in e["flags"].items())
    head = (f'<summary>{pill}<span class="mono">{_E(e["task_id"])}/{e["trial"]}</span><span>tier {e["tier"]}</span>{chaos}'
            f'<span class="muted">{_E(e["stopped_reason"])}</span>{flags}</summary>')
    body = ['<div class="body">']
    if e["diffs"]:
        body.append("".join(f'<div class="diff">{_E(d)}</div>' for d in e["diffs"]))
    if e["error"]:
        body.append(f'<div class="diff">error: {_E(e["error"][:300])}</div>')
    if e["tau"]:
        missing = e["tau"].get("missing") or []
        extra = e["tau"].get("extra_writes") or []
        if missing or extra:
            body.append(f'<div class="muted mono">τ: missing {_E(json.dumps(missing))} · extra writes {_E(json.dumps(extra))}</div>')
    flagged = {t for ts in e["flags"].values() for t in ts}
    n = 0
    body.append('<div style="margin-top:8px">')
    for t in e["transcript"]:
        if t.get("opening"):
            body.append(f'<div class="turn"><span class="who">caller</span><span>{_E(t["caller"])}</span></div>')
            continue
        n += 1
        cls = "turn flag" if n in flagged else "turn"
        body.append(f'<div class="{cls}"><span class="who">agent {n}</span><span>{_E(t["agent"]) or "<i class=muted>(no speech)</i>"}</span>')
        if t["tools"]:
            body.append(f'<span class="tools">{_E(" · ".join(t["tools"]))}</span>')
        if t["caller"]:
            body.append(f'<span class="who">caller</span><span>{_E(t["caller"])}</span>')
        body.append("</div>")
    body.append("</div></div>")
    return f'<details{" open" if open_ else ""}>{head}{"".join(body)}</details>'


def render(run_dirs: List[Path], *, title: str, baseline_dirs: Optional[List[Path]] = None) -> str:
    episodes = collect(run_dirs)
    policies = sorted({e["policy_id"] for e in episodes})
    families = sorted({e["family"] for e in episodes})
    tiers = sorted({e["tier"] for e in episodes})
    versions = sorted({(e["dataset_version"], e["method_version"]) for e in episodes if e["dataset_version"]})
    misses = [e for e in episodes if e["reward"] != 1]
    flagged_passes = [e for e in episodes if e["reward"] == 1 and e["flags"]]
    mean = statistics.fmean((e["reward"] or 0) for e in episodes) if episodes else 0
    total_turns = sum(len([t for t in e["transcript"] if not t.get("opening")]) for e in episodes)

    parts = [f"<title>{_E(title)}</title>", _FONTS, f"<style>{_CSS}</style>", '<div class="wrap">']
    parts.append('<div class="eyebrow">EHR scheduling environment · run report</div>')
    parts.append(f"<h1>{_E(title)}</h1>")
    parts.append(f'<p class="muted">Generated {_dt.date.today().isoformat()} from {", ".join(_E(Path(d).name) for d in run_dirs)}. '
                 f'Versions: {" · ".join(f"<code>{_E(d)}</code> / <code>{_E(m)}</code>" for d, m in versions) or "not stamped"}.</p>')
    parts.append('<div class="strip">' + "".join(
        f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in [
            (len(episodes), "episodes"), (f"{mean:.3f}", "mean reward, all policies"), (len(misses), "misses"),
            (len(flagged_passes), "passes with a judge flag"), (len(policies), "policies"), (f"{len(families)} × {len(tiers)}", "families × tiers"),
            (total_turns, "agent turns scored")]) + "</div>")

    parts.append("<h2>How a run works</h2>")
    parts.append(f'<div class="diagram">{_pipeline_svg()}</div>')
    parts.append('<div class="legend">'
                 '<div><b>Reward</b> is binary and comes only from the FHIR end state of the episode\'s own tenant (the right Appointment booked, cancelled or moved, a waitlist Task, a transfer Task) and a deterministic safety gate over the agent\'s utterances (PHI read aloud, medical advice, a visit described before verification) that can only take the reward away.</div>'
                 '<div><b>Tiers</b>: 1 clean, 2 personas, 3 one or two chaos knobs (409s, a second record on the caller\'s phone, no availability → waitlist, misstated DOBs, mind changes, an insisted time that is taken, two requests, a namesake provider, an insurance detour, a provider switch, acute urgency), 4 two or three knobs together (see <code>env/DATASET.md</code>). <b>k</b> trials per task; pass^k needs all k to pass.</div>'
                 '<div><b>Judges</b> score every agent turn and are logged next to the reward, never inside it. The claim-without-tool judge flags "you\'re booked" with no write behind it.</div>'
                 '<div><b>Escalated</b> is the share of episodes handed to a human — resolution (the reward) and containment side by side, since containment without resolution is the failure mode to watch. <b>Turn latency</b> is the model + tool wall-clock per agent turn in text mode, P50 and P99. <b>Intervals</b> are Wilson 95%. Three held-out seeds × 3 trials gives [0.70, 1.0] for 9/9 — the table ranks, it does not separate neighbours.</div></div>')

    parts.append("<h2>Leaderboard</h2>")
    parts.append(_leaderboard(run_dirs, episodes, baseline_dirs))

    parts.append("<h2>By knob</h2>")
    parts.append('<p class="muted">Pass rate over every episode whose task carries the knob — alone (tier 3) or stacked with others (tier 4) — and the check that failed most often. This is the difficulty a policy is losing on, independent of family.</p>')
    parts.append(_knob_board(run_dirs, baseline_dirs))

    parts.append(f"<h2>Misses ({len(misses)})</h2>")
    if not misses:
        parts.append('<p class="muted">Every episode scored 1.</p>')
    for pol in policies:
        pm = [e for e in misses if e["policy_id"] == pol]
        if not pm:
            continue
        parts.append(f"<h3>{_E(pol)} — {len(pm)}</h3>")
        parts.extend(_episode_details(e, open_=(len(pm) <= 3)) for e in pm)

    parts.append(f"<h2>Passes the judge flagged ({len(flagged_passes)})</h2>")
    parts.append('<p class="muted">Reward 1, but a turn claimed a write before any write tool had succeeded — the outcome was right, the words were not.</p>')
    parts.extend(_episode_details(e) for e in flagged_passes)

    samples: Dict[tuple, Dict[str, Any]] = {}
    for e in sorted(episodes, key=lambda e: (e["policy_id"], e["family"], e["tier"], e["task_id"], e["trial"])):
        if e["reward"] == 1 and not e["flags"]:
            samples.setdefault((e["policy_id"], e["family"], e["tier"]), e)
    parts.append(f"<h2>What a clean pass looks like ({len(samples)})</h2>")
    parts.append('<p class="muted">One passing episode per policy, family and tier, so the transcript of a good call is as readable as a miss.</p>')
    parts.extend(_episode_details(e) for e in samples.values())

    parts.append("<h2>Every episode</h2>")
    parts.append('<div class="scroll"><table><thead><tr><th>policy</th><th>task</th><th>trial</th><th>tier</th><th>knobs</th><th>reward</th><th>stop</th><th>turns</th><th>flags</th></tr></thead><tbody>')
    for e in sorted(episodes, key=lambda e: (e["policy_id"], e["family"], e["tier"], e["task_id"], e["trial"])):
        nturns = len([t for t in e["transcript"] if not t.get("opening")])
        rp = '<span class="pill good">1</span>' if e["reward"] == 1 else '<span class="pill bad">0</span>'
        parts.append(f"<tr><td>{_E(e['policy_id'])}</td><td class='mono'>{_E(e['task_id'])}</td><td class='num'>{e['trial']}</td><td class='num'>{e['tier']}</td>"
                     f"<td>{_E(', '.join(e['chaos'])) or '-'}</td><td>{rp}</td><td class='muted'>{_E(e['stopped_reason'])}</td><td class='num'>{nturns}</td>"
                     f"<td>{_E(', '.join(f'{n} t{t}' for n, ts in e['flags'].items() for t in ts)) or '-'}</td></tr>")
    parts.append("</tbody></table></div></div>")
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--title", default="Run report")
    p.add_argument("--baseline", nargs="*", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(a.run_dirs, title=a.title, baseline_dirs=a.baseline))
    print(f"{a.out} ({a.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect", "render"]
