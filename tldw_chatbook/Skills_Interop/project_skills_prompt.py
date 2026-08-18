# tldw_chatbook/Skills_Interop/project_skills_prompt.py
"""Prompt ledger + gating for .SKILLS/ import offers (spec 2026-08-17 §5.3)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_LEDGER_FILENAME = "project_prompts.json"


def _ledger_key(directory: str | Path) -> str:
    """Normalize a directory to its resolved absolute form (spec §5.3)."""
    return str(Path(directory).expanduser().resolve())


class ProjectSkillsPromptLedger:
    """Atomic-replace JSON ledger under <user_data_dir>/skills/."""

    def __init__(self, user_data_dir: str | Path) -> None:
        self._path = Path(user_data_dir) / "skills" / _LEDGER_FILENAME

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "entries": {}}
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return data

    def decision_for(self, directory: Path) -> tuple[str, str] | None:
        entry = self._load()["entries"].get(_ledger_key(directory))
        if not isinstance(entry, dict):
            return None
        return str(entry.get("decision", "")), str(entry.get("fingerprint", ""))

    def record(self, directory: Path, decision: str, fingerprint: str) -> None:
        data = self._load()
        data["entries"][_ledger_key(directory)] = {
            "decision": decision,
            "fingerprint": fingerprint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Writer-unique temp name (pid + thread id) so two concurrent app
        # instances/threads never race on the same temp file -- with a
        # fixed name, one writer's `replace` can consume the other's temp
        # file out from under it, raising FileNotFoundError. This ledger is
        # advisory only (a lost record just causes one extra prompt later),
        # so a write failure here must never crash the caller; best-effort
        # write-and-replace, swallow OSError, clean up the temp file if it's
        # still there. The two pre-existing sites with this same fixed-temp-
        # name flaw (local_skills_service.py, skill_trust_store.py) are
        # tracked separately -- not fixed here.
        temp = self._path.with_name(
            f"{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp.replace(self._path)
        except OSError:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            return


def should_offer_project_skills_prompt(
    enabled: bool,
    entry: tuple[str, str] | None,
    fingerprint: str,
) -> bool:
    """Offer iff enabled AND (never seen OR changed-and-not-never) (spec §5.3)."""
    if not enabled:
        return False
    if entry is None:
        return True
    decision, recorded_fingerprint = entry
    if decision == "never":
        return False
    return recorded_fingerprint != fingerprint
