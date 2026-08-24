from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

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


def test_reconcile_selected_source_preserves_absent_implicit_selection(tmp_path) -> None:
    """Dropping the missing row would deselect older implicitly selected sources."""

    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    for media_id in ("1", "2"):
        registry.link_membership(
            "workspace-1", item_type="media", item_id=media_id, role="source"
        )

    result = registry.reconcile_research_source_selection(
        "workspace-1", media_id="2", desired_selected=True
    )

    assert result is None
    assert registry.get_workspace_scope("workspace-1") is None


def test_reconcile_unselected_source_materializes_other_implicit_sources(
    tmp_path,
) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    registry.link_membership(
        "workspace-1", item_type="media", item_id="1", role="source"
    )
    registry.link_membership(
        "workspace-1", item_type="note", item_id="note-1", role="source"
    )
    registry.link_membership(
        "workspace-1", item_type="media", item_id="2", role="source"
    )

    narrowed = registry.reconcile_research_source_selection(
        "workspace-1", media_id="2", desired_selected=False
    )
    restored = registry.reconcile_research_source_selection(
        "workspace-1", media_id="2", desired_selected=True
    )

    assert narrowed is not None
    assert narrowed.items == (
        ScopeItem("media", "1"),
        ScopeItem("note", "note-1"),
    )
    assert restored is not None
    assert restored.items == (
        ScopeItem("media", "1"),
        ScopeItem("note", "note-1"),
        ScopeItem("media", "2"),
    )
    assert restored.empty_is_scoped is True


def test_reconcile_explicit_scope_changes_only_target_media(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    for media_id in ("1", "2"):
        registry.link_membership(
            "workspace-1", item_type="media", item_id=media_id, role="source"
        )
    registry.set_workspace_scope(
        "workspace-1",
        RagScope(
            items=(ScopeItem("note", "note-1"), ScopeItem("media", "1")),
            updated_at="before",
            empty_is_scoped=True,
        ),
    )

    added = registry.reconcile_research_source_selection(
        "workspace-1", media_id="2", desired_selected=True
    )
    removed = registry.reconcile_research_source_selection(
        "workspace-1", media_id="1", desired_selected=False
    )

    assert added is not None
    assert added.items == (
        ScopeItem("note", "note-1"),
        ScopeItem("media", "1"),
        ScopeItem("media", "2"),
    )
    assert removed is not None
    assert removed.items == (
        ScopeItem("note", "note-1"),
        ScopeItem("media", "2"),
    )


def test_reconcile_malformed_scope_fails_closed_before_target_change(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    for media_id in ("1", "2"):
        registry.link_membership(
            "workspace-1", item_type="media", item_id=media_id, role="source"
        )
    with registry.db.transaction() as connection:
        connection.execute(
            """
            INSERT INTO workspace_rag_scopes (workspace_id, payload, updated_at)
            VALUES (?, ?, ?)
            """,
            ("workspace-1", '{"version":2,"items":[', "corrupt"),
        )

    unselected = registry.reconcile_research_source_selection(
        "workspace-1", media_id="1", desired_selected=False
    )
    with registry.db.transaction() as connection:
        connection.execute(
            """
            UPDATE workspace_rag_scopes
            SET payload = ?, updated_at = ?
            WHERE workspace_id = ?
            """,
            ('{"version":2,"items":[', "corrupt-again", "workspace-1"),
        )
    selected = registry.reconcile_research_source_selection(
        "workspace-1", media_id="2", desired_selected=True
    )

    assert unselected is not None
    assert unselected.items == ()
    assert unselected.empty_is_scoped is True
    assert selected is not None
    assert selected.items == (ScopeItem("media", "2"),)
    assert selected.empty_is_scoped is True


def test_concurrent_reconcile_serializes_read_modify_write_without_loss(tmp_path) -> None:
    path = tmp_path / "workspaces.sqlite"
    setup = LocalWorkspaceRegistryService(WorkspaceDB(path, client_id="setup"))
    setup.create_workspace(workspace_id="workspace-1", name="Research")
    for media_id in ("0", "1", "2"):
        setup.link_membership(
            "workspace-1", item_type="media", item_id=media_id, role="source"
        )
    setup.set_workspace_scope(
        "workspace-1",
        RagScope(
            items=(ScopeItem("media", "0"),),
            updated_at="before",
            empty_is_scoped=True,
        ),
    )
    start = Barrier(3)

    def reconcile(media_id: str) -> None:
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(path, client_id=f"worker-{media_id}")
        )
        start.wait()
        registry.reconcile_research_source_selection(
            "workspace-1", media_id=media_id, desired_selected=True
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reconcile, media_id) for media_id in ("1", "2")]
        start.wait()
        for future in futures:
            future.result()

    stored = setup.get_workspace_scope("workspace-1")
    assert stored is not None
    assert set(stored.items) == {
        ScopeItem("media", "0"),
        ScopeItem("media", "1"),
        ScopeItem("media", "2"),
    }
