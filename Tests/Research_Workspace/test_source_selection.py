from __future__ import annotations

import json

from tldw_chatbook.Chat.rag_scope import (
    RagScope,
    ScopeItem,
    parse_scope,
    resolve_effective_scope,
    serialize_scope,
)
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def _registry(tmp_path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="research-selection")
    )


def test_research_explicit_empty_scope_round_trips_and_fails_closed_after_restart(
    tmp_path,
) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    scope = RagScope(items=(), updated_at="t1", empty_is_scoped=True)

    registry.set_workspace_scope("workspace-1", scope)
    reloaded = _registry(tmp_path)
    stored = reloaded.get_workspace_scope("workspace-1")

    assert stored == scope
    effective = resolve_effective_scope(None, stored, lambda _kind, ids: ids)
    assert effective.state == "empty"
    assert effective.allowlist == {}


def test_console_ordinary_empty_scope_still_clears_to_unscoped(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Console")

    registry.set_workspace_scope("workspace-1", RagScope(items=(), updated_at="t1"))

    assert registry.get_workspace_scope("workspace-1") is None


def test_explicit_empty_payload_is_versioned_and_v1_remains_compatible() -> None:
    encoded = serialize_scope(RagScope(items=(), updated_at="t1", empty_is_scoped=True))
    old = {
        "version": 1,
        "updated_at": "t0",
        "items": [{"source_type": "media", "source_id": "7"}],
    }

    assert encoded == {
        "version": 2,
        "updated_at": "t1",
        "items": [],
        "empty_is_scoped": True,
    }
    assert parse_scope(old) == RagScope(
        items=(ScopeItem("media", "7"),),
        updated_at="t0",
    )


def test_corrupt_explicit_scope_payload_fails_closed(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    with registry.db.transaction() as connection:
        connection.execute(
            """
            INSERT INTO workspace_rag_scopes (workspace_id, payload, updated_at)
            VALUES (?, ?, ?)
            """,
            (
                "workspace-1",
                json.dumps(
                    {
                        "version": 2,
                        "updated_at": "t1",
                        "items": "corrupt",
                        "empty_is_scoped": True,
                    }
                ),
                "t1",
            ),
        )

    stored = registry.get_workspace_scope("workspace-1")

    assert stored == RagScope(items=(), updated_at="t1", empty_is_scoped=True)
    assert (
        resolve_effective_scope(None, stored, lambda _kind, ids: ids).state == "empty"
    )


def test_truncated_workspace_scope_fails_closed_after_restart(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    with registry.db.transaction() as connection:
        connection.execute(
            """
            INSERT INTO workspace_rag_scopes (workspace_id, payload, updated_at)
            VALUES (?, ?, ?)
            """,
            ("workspace-1", '{"version":2,"items":[', "t-corrupt"),
        )

    stored = _registry(tmp_path).get_workspace_scope("workspace-1")

    assert stored == RagScope(
        items=(), updated_at="t-corrupt", empty_is_scoped=True
    )
    assert (
        resolve_effective_scope(None, stored, lambda _kind, ids: ids).state == "empty"
    )


def test_unlink_last_research_source_preserves_explicit_empty_scope(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    registry.link_membership(
        "workspace-1", item_type="media", item_id="7", role="source"
    )
    registry.set_workspace_scope(
        "workspace-1",
        RagScope(
            items=(ScopeItem("media", "7"),),
            updated_at="t1",
            empty_is_scoped=True,
        ),
    )

    assert registry.unlink_membership(
        "workspace-1", item_type="media", item_id="7", role="source"
    )

    assert registry.get_workspace_scope("workspace-1") == RagScope(
        items=(),
        updated_at=registry.get_workspace_scope("workspace-1").updated_at,
        empty_is_scoped=True,
    )
