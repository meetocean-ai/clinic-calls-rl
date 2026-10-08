"""The round-trip check over generated caller lines (roadmap P2-01): for each task in a sample, inject the tier's
disfluencies, speak the opening line with the seed's voice, push it through the tier's phone line, transcribe it
locally and score the similarity. Writes a JSON report and prints pass rates per tier and voice.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.audio.roundtrip_report --seeds 10 --tiers 1,3,4 \
        --out results/env/voice-roundtrip.json

A line below the floor (0.85) is a line the environment must not ship as audio; the report names it, the voice it
used and what the ASR heard, so the fix is either the line, the voice or the degradation knob.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from env.audio.channel import VoiceChannel
from env.audio.tts import load_tts
from env.audio.roundtrip import SIMILARITY_FLOOR, facts_heard, load_asr, similarity, wav_samples_from_clip
from env.tasks import FAMILIES, HELDOUT_EVAL_SEEDS, generate

# The lines a caller actually says: the opening line, plus the DOB answer every verification asks for.
def _lines(task):
    yield "opening", task.opening_line
    yield "dob", f"It's {task.claimed_dob}."


def run(seeds, tiers, families, *, tts, asr, floor=SIMILARITY_FLOOR):
    rows = []
    for tier in tiers:
        for family in families:
            for seed in seeds:
                task = generate(family, seed, tier=tier if family != "emergency_redirect" else 1)
                channel = VoiceChannel.for_task(seed, tier, tts=tts)
                for kind, line in _lines(task):
                    t0 = time.perf_counter()
                    clip = channel.render(line)  # verbalize → disfluencies → voice → phone line
                    samples, sr = wav_samples_from_clip(clip)
                    heard = asr.transcribe(samples, sr)
                    intelligible = similarity(clip["spoken"], heard)  # was what was said understood?
                    facts = facts_heard(line, heard)  # did the facts of the clean line survive?
                    rows.append({"task_id": task.id, "kind": kind, "tier": tier, "family": family, "voice": channel.voice, "line": line,
                                 "spoken": clip["spoken"], "heard": heard, "similarity": round(intelligible, 3), "facts": facts,
                                 "ok": intelligible >= floor and facts, "degradation": channel.degradation.to_dict(),
                                 "disfluency": channel.disfluency.to_dict(), "audio_s": clip["seconds"],
                                 "wall_s": round(time.perf_counter() - t0, 2)})
    return rows


def summarize(rows, floor=SIMILARITY_FLOOR):
    by_tier, by_voice, by_kind = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r["ok"])
        by_voice[r["voice"]].append(r["ok"])
        by_kind[r["kind"]].append(r["ok"])
    return {
        "n": len(rows), "floor": floor, "pass_rate": round(sum(r["ok"] for r in rows) / len(rows), 3) if rows else None,
        "intelligibility_rate": round(sum(r["similarity"] >= floor for r in rows) / len(rows), 3) if rows else None,
        "facts_rate": round(sum(r["facts"] for r in rows) / len(rows), 3) if rows else None,
        "mean_similarity": round(statistics.fmean(r["similarity"] for r in rows), 3) if rows else None,
        "by_tier": {str(t): round(sum(v) / len(v), 3) for t, v in sorted(by_tier.items())},
        "by_kind": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_kind.items())},
        "by_voice": {v: round(sum(x) / len(x), 3) for v, x in sorted(by_voice.items())},
        "failures": [{k: r[k] for k in ("task_id", "kind", "tier", "voice", "line", "spoken", "heard", "similarity", "facts")} for r in rows if not r["ok"]],
        "seconds_per_line": round(statistics.fmean(r["wall_s"] for r in rows), 2) if rows else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5, help="held-out seeds per family (from HELDOUT_EVAL_SEEDS)")
    p.add_argument("--tiers", default="1,3,4")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    tiers = [int(t) for t in a.tiers.split(",") if t]
    families = [f for f in a.families.split(",") if f]
    tts, asr = load_tts(), load_asr()
    rows = run(HELDOUT_EVAL_SEEDS[: a.seeds], tiers, families, tts=tts, asr=asr)
    summary = {"tts": tts.name, "asr": asr.name, **summarize(rows)}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run", "summarize"]
