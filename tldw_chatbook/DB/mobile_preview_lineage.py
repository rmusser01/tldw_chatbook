"""Fixed incompatible-preview rejection before native selected-store writers.

These reads only reject known unmerged lineages. They do not authenticate a
schema or grant a migration, recovery, execution, or file capability (ADR-244).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path
from tldw_chatbook.Utils.platform_files import os


class IncompatiblePreviewError(ValueError):
    """The selected store requires its matching retained preview build."""


_GUIDANCE = (
    "incompatible_unmerged_preview: Retain this database, its captures, and any "
    "SQLite coordination sidecars. "
    "Use its matching build on an isolated retained copy, or retain it for a "
    "separately reviewed conversion. Select a compatible native profile."
)


def _existing_preview_target(path: str | Path):
    """Distinguish native missing targets from failed or unsafe inspections."""
    if isinstance(path, str) and path == ":memory:":
        return None
    selected = lexical_path(path)
    try:
        identity = os.lstat(selected)
    except FileNotFoundError:
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                os.lstat(Path(str(selected) + suffix))
            except FileNotFoundError:
                continue
            raise ValueError("preview_preflight_orphaned_sidecar")
        return None
    return selected, identity


def _reject_preview_connection(connection, family, deadline):
    """Inspect only fixed schema names/stamps in one bounded live read snapshot."""
    if family not in {"chat", "workspaces", "runs"}:
        raise ValueError("preview_preflight_family_invalid")
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("preview_preflight_deadline")
        connection.execute("BEGIN")
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name IN "
                "('mobile_consolidation_batches','mobile_import_authority_holds',"
                "'console_command_receipts','schema_version','db_schema_version')"
            )
        }
        mobile = bool(
            names & {"mobile_consolidation_batches", "mobile_import_authority_holds"}
        )
        commands = family == "runs" and "console_command_receipts" in names
        if not mobile and not commands:
            return
        if family == "chat" and "db_schema_version" in names:
            rows = connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema' LIMIT 2"
            ).fetchall()
            version = rows[0][0] if len(rows) == 1 else None
        elif family != "chat" and "schema_version" in names:
            version = connection.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]
        else:
            version = None
        minimum = {"chat": 86, "workspaces": 11, "runs": 27 if mobile else 26}[family]
        if type(version) is not int or version < minimum:
            raise IncompatiblePreviewError(_GUIDANCE)
    except sqlite3.OperationalError as error:
        if time.monotonic() >= deadline:
            raise TimeoutError("preview_preflight_deadline") from error
        raise
    finally:
        connection.set_progress_handler(None, 0)
        # The module owner physically closes its admitted connection, releasing
        # this read transaction and its native storage lease without checkpoint.


def reject_incompatible_console_stores(
    chat_path: str | Path,
    workspace_path: str | Path,
    run_path: str | Path,
) -> None:
    """Preflight all actual selected stores before any coordinated owner upgrade."""
    from .AgentRuns_DB import reject_incompatible_preview as runs
    from .ChaChaNotes_DB import reject_incompatible_preview as chat
    from .Workspace_DB import reject_incompatible_preview as workspaces

    chat(chat_path)
    workspaces(workspace_path)
    runs(run_path)
