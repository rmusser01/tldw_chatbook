"""WorkspaceDB v7 + assistant_defaults roundtrip + effective resolution."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.assistant_defaults import (
    WorkspaceEffectiveAssistantDefault,
    resolve_effective_assistant_default,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


def build_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    )


def test_v2_database_migrates_preserving_rows(tmp_path):
    legacy = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (2);
        CREATE TABLE workspace_records (
            workspace_id TEXT PRIMARY KEY, name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '', authority TEXT NOT NULL,
            sync_status TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO workspace_records VALUES
            ('w-1', 'Research', '', 'local_only', 'not_configured', 0, 0, 't1', 't1');
        """
    )
    conn.commit()
    conn.close()
    db = WorkspaceDB(legacy, client_id="client-1")
    assert db.get_schema_version() == 7
    with db.connection() as read_conn:
        cols = {
            row[1] for row in read_conn.execute("PRAGMA table_info(workspace_records)")
        }
        assert "assistant_defaults" in cols
        row = read_conn.execute(
            "SELECT name FROM workspace_records WHERE workspace_id = 'w-1'"
        ).fetchone()
        assert row[0] == "Research"
    assert db.is_agent_backfill_complete() is False
    db.mark_agent_backfill_complete()
    assert WorkspaceDB(legacy, client_id="client-1").is_agent_backfill_complete() is True


def test_defaults_roundtrip_and_validation(tmp_path):
    registry = build_registry(tmp_path)
    record = registry.create_workspace(workspace_id="w-9", name="Lit Review")
    assert record.assistant_defaults is None
    defaults = WorkspaceAssistantDefaults(
        assistant_id="local-persona-abc", tool_policy_profile_id="ws-w-9"
    )
    updated = registry.set_assistant_defaults("w-9", defaults)
    assert updated.assistant_defaults == defaults
    assert registry.get_workspace("w-9").assistant_defaults == defaults
    cleared = registry.clear_assistant_defaults("w-9")
    assert cleared.assistant_defaults is None


def test_read_write_requires_confirmation(tmp_path):
    registry = build_registry(tmp_path)
    registry.create_workspace(workspace_id="w-2", name="W2")
    defaults = WorkspaceAssistantDefaults(
        assistant_id="p1", persona_memory_mode="read_write"
    )
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.set_assistant_defaults("w-2", defaults)
    registry.set_assistant_defaults("w-2", defaults, confirm_read_write=True)


def test_malformed_stored_json_degrades_to_none(tmp_path):
    registry = build_registry(tmp_path)
    registry.create_workspace(workspace_id="w-3", name="W3")
    with registry.db.transaction() as conn:
        conn.execute(
            "UPDATE workspace_records SET assistant_defaults = ? WHERE workspace_id = 'w-3'",
            ("{not json",),
        )
    assert registry.get_workspace("w-3").assistant_defaults is None


def test_effective_resolution_reason_codes():
    none = resolve_effective_assistant_default(None, lambda _id: {})
    assert (none.status, none.degraded_reason) == ("none", None)
    deleted = resolve_effective_assistant_default(
        WorkspaceAssistantDefaults(assistant_id="gone"), lambda _id: None
    )
    assert (deleted.status, deleted.degraded_reason) == ("unavailable", "persona_deleted")
    ok = resolve_effective_assistant_default(
        WorkspaceAssistantDefaults(assistant_id="p"),
        lambda _id: {"id": "p", "name": "Lit Agent"},
    )
    assert (ok.status, ok.label, ok.source) == ("available", "Lit Agent", "workspace")


def test_effective_resolution_degrades_non_mapping_lookup_results():
    """A malformed (non-mapping) persona lookup degrades, never raises."""
    for bad in (["not", "a", "mapping"], "just-a-string"):
        result = resolve_effective_assistant_default(
            WorkspaceAssistantDefaults(assistant_id="p"), lambda _id, _b=bad: _b
        )
        assert (result.status, result.degraded_reason) == (
            "unavailable",
            "persona_unavailable",
        )
