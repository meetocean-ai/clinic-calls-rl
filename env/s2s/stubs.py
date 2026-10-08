"""Helpers for the scripted servers in the S2S tests: a stubbed model must book with the provider the caller named,
exactly as a real model would read it off the availability result — `look_up_availability` returns every provider's
open slots in the server's order, so a hard-coded `slot_index: 0` books whoever the server lists first (a flaky test
that fails on the named-provider check). Scripts write `PICK` where the slot index goes; the stub fills it from the
last availability result it saw."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

PICK = "<pick-slot>"
_DR = re.compile(r"\bDr\.?\s+([A-Z][a-z]+)")


def named_surname(text: str) -> Optional[str]:
    m = _DR.search(text or "")
    return m.group(1) if m else None


def pick_slot(result: Any, surname: Optional[str]) -> int:
    """The first slot whose provider display carries the surname; the first slot when no provider was named."""
    slots = (result or {}).get("slots") or [] if isinstance(result, dict) else []
    if surname:
        for s in slots:
            if surname.lower() in str(s.get("provider") or "").lower():
                return int(s.get("slot_index", 0))
    return int(slots[0].get("slot_index", 0)) if slots else 0


def substitute(obj: Any, slot: int) -> Any:
    """Replace PICK anywhere inside dicts / lists / JSON strings with the chosen slot index."""
    if isinstance(obj, dict):
        return {k: substitute(v, slot) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, slot) for v in obj]
    if isinstance(obj, tuple):
        return tuple(substitute(v, slot) for v in obj)
    if isinstance(obj, str):
        if obj == PICK:
            return slot
        if PICK in obj:
            return obj.replace(f'"{PICK}"', str(slot)).replace(PICK, str(slot))
    return obj


def slots_in(result: Any) -> bool:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return False
    return isinstance(result, dict) and "slots" in result


def pick_from_messages(messages: List[Dict[str, Any]], surname: Optional[str]) -> int:
    """For an OpenAI-shaped fake: the last tool message carrying an availability result."""
    for m in reversed(messages):
        if m.get("role") == "tool" and slots_in(m.get("content")):
            return pick_slot(json.loads(m["content"]), surname)
    return 0


__all__ = ["PICK", "named_surname", "pick_from_messages", "pick_slot", "slots_in", "substitute"]
