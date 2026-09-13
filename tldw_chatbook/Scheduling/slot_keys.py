"""Deterministic idempotency keys for automation runs.

Byte-identical to tldw_server's ``recurring_question_jobs.py`` recipe so
a definition's slot identity survives handoff in either direction
(spec-2026-08-31-schedules-handoff-parity.md §7.2).
"""

from __future__ import annotations

import hashlib
import json


def canonical_hash(payload: dict) -> str:
    """Return a stable SHA-256 hex digest for a JSON-compatible payload."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_scheduled_run_idempotency_key(
    *, definition_id: str, definition_version: int, schedule_slot: str
) -> str:
    """Return the deterministic key for one scheduled slot (server parity)."""
    return "scheduled-task-rq:" + canonical_hash(
        {
            "definition_id": definition_id,
            "definition_version": definition_version,
            "schedule_slot": schedule_slot,
        }
    )


def build_manual_run_idempotency_payload(*, definition_id: str) -> dict[str, str]:
    """Return the idempotency payload for a manual run (server parity)."""
    return {
        "action": "create_manual_run",
        "definition_id": definition_id,
        "trigger_reason": "manual",
    }
