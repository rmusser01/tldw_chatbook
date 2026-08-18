# tldw_chatbook/Skills_Interop/project_skills_prompt.py
"""Prompt ledger + gating for .SKILLS/ import offers (spec 2026-08-17 §5.3)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    ProjectSkillsDiscovery,
    discover_project_skills,
    find_project_dir_with_skills,
)

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


def startup_discovery_for(
    start: Path, *, enabled: bool, ledger_dir: Path
) -> ProjectSkillsDiscovery | None:
    """Pure gate deciding whether to offer a project's .SKILLS/ import at startup.

    Combines Task 1's ancestor-walk discovery with the ledger gating above
    into the one seam the app-startup worker calls, so ``app.py`` itself
    stays a thin caller. Returns ``None`` (never offer) when: the kill
    switch (``enabled``) is off; no project directory with a recognizable
    ``.SKILLS``/``.skills`` folder is found walking up from ``start``; that
    folder has no usable entries; or the ledger (built fresh from
    ``ledger_dir``, keeping the ledger path defined in this one module)
    says this exact fingerprint has already been shown and dismissed with
    "never", or seen unchanged after "declined"/"imported".
    """
    if not enabled:
        return None
    project_dir = find_project_dir_with_skills(start)
    if project_dir is None:
        return None
    discovery = discover_project_skills(project_dir)
    if discovery is None or not discovery.entries:
        return None
    ledger = ProjectSkillsPromptLedger(ledger_dir)
    if not should_offer_project_skills_prompt(
        True, ledger.decision_for(discovery.root), discovery.fingerprint
    ):
        return None
    return discovery
