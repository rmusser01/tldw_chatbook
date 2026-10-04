"""Disable execution-bearing rule state only in qualified imported candidates."""

import json
import sqlite3
from pathlib import Path
from threading import Event
from typing import Any, cast

from .models import canonical_json


def prepare_imported_rules(cursor: sqlite3.Cursor) -> None:
    """Retain rule history while removing imported activation and pending checks.

    Caller must first validate/migrate a disposable selected archive payload.
    Exact rollback snapshots and preserved local owners never call this route.
    Historical receipts remain facts; they do not recreate live registries.
    """
    cursor.execute(
        "UPDATE console_response_rule_bindings SET state='disabled',binding_revision=binding_revision+1 WHERE state='enabled'"
    )
    for row in cursor.execute(
        "SELECT assessment_id,assessment_json FROM console_response_rule_assessments WHERE state='pending'"
    ).fetchall():
        document = json.loads(row[1])
        document["state"] = "cancelled"
        cursor.execute(
            "UPDATE console_response_rule_assessments SET state='cancelled',assessment_json=? WHERE assessment_id=?",
            (canonical_json(document), row[0]),
        )


def prepare_imported_candidate(candidate: Path, cancel: Event) -> None:
    """Revalidate the selected disposable payload before narrowly disabling it."""
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation
    from tldw_chatbook.DB.recovery_core import core_adapters

    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    issues = validate_candidate(owner, candidate, cancel, migrate=False)
    if issues:
        raise ValueError(issues[0])
    with open_recovery_validation(
        owner.owner_id, candidate, writable=True, with_restrictions=True, cancel=cancel
    ) as opened:
        connection, restrictions = cast(tuple[sqlite3.Connection, Any], opened)
        restrictions.preparing_response_rules = True
        try:
            connection.execute("BEGIN IMMEDIATE")
            prepare_imported_rules(connection.cursor())
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            restrictions.preparing_response_rules = False
