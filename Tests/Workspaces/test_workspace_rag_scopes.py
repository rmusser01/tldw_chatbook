"""Tests for task-12: workspace-level RAG retrieval scope storage.

Covers the ``workspace_rag_scopes`` table (co-located with the workspace
registry in ``WorkspaceDB`` -- NOT ``ChaChaNotes_DB``, since a workspace has
no row in that database) and ``LocalWorkspaceRegistryService.get_workspace_scope``/
``set_workspace_scope``. Mirrors ``Tests/Chat/test_rag_scope_storage.py``'s
conventions (real file-backed DB, tmp_path) and
``Tests/Workspaces/test_workspace_registry_service.py``'s fixture pattern
(``build_test_registry``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem, SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.registry_service import WorkspaceNotFound

pytestmark = pytest.mark.unit


def build_test_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="task12-test")
    )


@pytest.fixture()
def registry(tmp_path):
    return build_test_registry(tmp_path)


@pytest.fixture()
def workspace_id(registry):
    record = registry.create_workspace(workspace_id="ws-a", name="Sales reports")
    return record.workspace_id


class TestTableCreatedIdempotently:
    def test_repeat_init_on_same_file_does_not_raise(self, tmp_path):
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="c1")

        # Second init against the same file must be a no-op, not an error
        # (CREATE TABLE IF NOT EXISTS, mirroring every other workspace table).
        second = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="c1")

        with second.connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='workspace_rag_scopes'"
            ).fetchone()
        assert row is not None


class TestRoundTrip:
    def test_set_then_get_round_trips(self, registry, workspace_id):
        scope = RagScope(
            items=(ScopeItem(SOURCE_TYPE_MEDIA, "42"), ScopeItem(SOURCE_TYPE_NOTE, "n1")),
            updated_at="2026-07-21T00:00:00+00:00",
        )

        registry.set_workspace_scope(workspace_id, scope)
        result = registry.get_workspace_scope(workspace_id)

        assert result == scope

    def test_set_persists_across_reload(self, tmp_path):
        registry = build_test_registry(tmp_path)
        record = registry.create_workspace(workspace_id="ws-reload", name="Reload target")
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_NOTE, "n9"),), updated_at="t1")

        registry.set_workspace_scope(record.workspace_id, scope)

        reloaded = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="task12-test")
        )
        assert reloaded.get_workspace_scope(record.workspace_id) == scope

    def test_set_updates_an_existing_scope(self, registry, workspace_id):
        first = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        second = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "2"),), updated_at="t2")

        registry.set_workspace_scope(workspace_id, first)
        registry.set_workspace_scope(workspace_id, second)

        assert registry.get_workspace_scope(workspace_id) == second


class TestSetNoneDeletes:
    def test_set_none_deletes_the_row(self, registry, workspace_id):
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        registry.set_workspace_scope(workspace_id, scope)
        assert registry.get_workspace_scope(workspace_id) is not None

        registry.set_workspace_scope(workspace_id, None)

        assert registry.get_workspace_scope(workspace_id) is None

    def test_set_none_on_a_never_set_workspace_is_a_noop(self, registry, workspace_id):
        registry.set_workspace_scope(workspace_id, None)  # must not raise

        assert registry.get_workspace_scope(workspace_id) is None


class TestZeroItemNormalization:
    def test_set_zero_item_scope_normalizes_to_delete(self, registry, workspace_id):
        """Mirrors the conversation-level contract (Task 5 review): 'save with
        zero selected' means 'clear scope', not 'scoped with nothing in it'."""
        empty_scope = RagScope(items=(), updated_at="t1")

        registry.set_workspace_scope(workspace_id, empty_scope)

        assert registry.get_workspace_scope(workspace_id) is None


class TestMissingOrNeverSet:
    def test_get_on_never_set_workspace_is_none(self, registry, workspace_id):
        assert registry.get_workspace_scope(workspace_id) is None

    def test_get_on_unknown_workspace_id_is_none(self, registry):
        assert registry.get_workspace_scope("does-not-exist") is None


class TestSetOnUnknownWorkspaceRaises:
    def test_set_scope_for_nonexistent_workspace_raises(self, registry):
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")

        with pytest.raises(WorkspaceNotFound):
            registry.set_workspace_scope("does-not-exist", scope)


class TestCorruptPayloadGuard:
    def _write_raw_payload(self, registry, workspace_id, payload: str) -> None:
        with registry.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO workspace_rag_scopes (workspace_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (workspace_id, payload, "t1"),
            )

    def test_malformed_json_payload_reads_as_none(self, registry, workspace_id):
        self._write_raw_payload(registry, workspace_id, "{not valid json")

        assert registry.get_workspace_scope(workspace_id) is None

    def test_non_dict_json_payload_reads_as_none(self, registry, workspace_id):
        self._write_raw_payload(registry, workspace_id, '["not", "a", "dict"]')

        assert registry.get_workspace_scope(workspace_id) is None

    def test_wrong_version_payload_reads_as_none(self, registry, workspace_id):
        self._write_raw_payload(
            registry,
            workspace_id,
            json.dumps({"version": 99, "items": [], "updated_at": "t1"}),
        )

        assert registry.get_workspace_scope(workspace_id) is None

    def test_missing_items_key_reads_as_none(self, registry, workspace_id):
        self._write_raw_payload(
            registry, workspace_id, json.dumps({"version": 1, "updated_at": "t1"})
        )

        assert registry.get_workspace_scope(workspace_id) is None


class TestWorkspaceDeletionCascades:
    def test_deleting_the_workspace_row_drops_the_scope_row(self, registry, workspace_id):
        """The registry has no hard-delete API yet (only archive), but the
        storage layer must not orphan scope rows once one is added -- proven
        directly against the FK cascade the same way workspace_memberships
        and workspace_runtime_bindings already rely on."""
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        registry.set_workspace_scope(workspace_id, scope)
        assert registry.get_workspace_scope(workspace_id) is not None

        with registry.db.transaction() as conn:
            conn.execute("DELETE FROM workspace_records WHERE workspace_id = ?", (workspace_id,))

        with registry.db.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM workspace_rag_scopes WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
        assert row is None
        assert registry.get_workspace_scope(workspace_id) is None
