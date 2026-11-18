"""Exact-target recovery effects against real SQLite stores."""

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.archive_recovery.operations import (
    allowed_restore_scopes,
    execute_restore,
    read_recovery_target,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


@pytest.fixture
def recovery_stores(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.sqlite", client_id="recovery-effects")
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite")
    registry = LocalWorkspaceRegistryService(workspace_db)
    registry.create_workspace(workspace_id="project", name="Project")
    registry.archive_workspace("project")
    cid = db.add_conversation(
        {"title": "Original", "scope_type": "workspace", "workspace_id": "project"}
    )
    db.set_conversations_archived([cid], archived=True, expected_versions={cid: 1})
    app = SimpleNamespace(
        local_chat_conversation_service=ChatConversationService(db),
        workspace_registry_service=registry,
    )
    try:
        yield app, cid, db
    finally:
        # Off-loop recovery reads/writes own thread-local handles too.
        try:
            with db.quiesce_connections(timeout_seconds=2.0):
                pass
        finally:
            workspace_db.close()


@pytest.mark.parametrize(
    "chat,workspace,expected",
    [
        (True, False, ("chat_only",)),
        (True, True, ("chat_only", "chat_and_workspace")),
        (False, True, ("workspace_only",)),
        (False, False, ()),
    ],
)
def test_only_meaningful_recovery_choices(chat, workspace, expected):
    assert allowed_restore_scopes(chat, workspace) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["chat", "workspace", "name", "owner", "deleted"])
async def test_stale_preflight_never_begins_workspace_write(recovery_stores, drift):
    app, cid, db = recovery_stores
    target = await read_recovery_target(app, cid)
    if drift == "workspace":
        app.workspace_registry_service.unarchive_workspace("project")
        app.workspace_registry_service.archive_workspace("project")
    elif drift == "name":
        with app.workspace_registry_service.db.transaction() as conn:
            conn.execute(
                "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
                ("Renamed", "project"),
            )
    else:
        with db.transaction() as conn:
            if drift == "chat":
                conn.execute(
                    "UPDATE conversations SET version = version + 1 WHERE id = ?",
                    (cid,),
                )
            elif drift == "owner":
                conn.execute(
                    "UPDATE conversations SET workspace_id = NULL, scope_type = 'global', version = version + 1 WHERE id = ?",
                    (cid,),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET deleted = 1 WHERE id = ?", (cid,)
                )
    result = await execute_restore(
        app,
        target,
        "chat_and_workspace",
        replacement_name=None,
        cancel_event=threading.Event(),
    )
    assert result.status in ("stale", "not_found")
    assert not result.workspace_changed and not result.chat_changed
    assert app.workspace_registry_service.get_workspace("project").archived


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,chat_changed,workspace_changed,available",
    [
        ("chat_only", True, False, False),
        ("chat_and_workspace", True, True, True),
        ("workspace_only", False, True, True),
    ],
)
async def test_exact_scopes_and_availability(
    recovery_stores, scope, chat_changed, workspace_changed, available
):
    app, cid, db = recovery_stores
    if scope == "workspace_only":
        db.set_conversations_archived([cid], archived=False, expected_versions={cid: 2})
    target = await read_recovery_target(app, cid)
    result = await execute_restore(
        app, target, scope, replacement_name=None, cancel_event=threading.Event()
    )
    assert result.status == "restored"
    assert (result.chat_changed, result.workspace_changed, result.chat_available) == (
        chat_changed,
        workspace_changed,
        available,
    )
    assert result.next_action == ("open" if available else "restore_workspace")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drift",
    ["owner", "deleted", "chat_archived", "observation_failure", "chat_failure"],
)
async def test_post_workspace_write_observes_truth(recovery_stores, monkeypatch, drift):
    app, cid, db = recovery_stores
    if drift != "chat_failure":
        db.set_conversations_archived([cid], archived=False, expected_versions={cid: 2})
    target = await read_recovery_target(app, cid)
    registry = app.workspace_registry_service
    original = registry.unarchive_workspace

    def restore(*args, **kwargs):
        result = original(*args, **kwargs)
        if drift == "chat_archived":
            changed = db.set_conversations_archived(
                [cid],
                archived=True,
                expected_versions={cid: target.conversation_version},
            )
            assert cid in changed["changed"]
        elif drift in ("owner", "deleted"):
            with db.transaction() as conn:
                if drift == "owner":
                    conn.execute(
                        "UPDATE conversations SET workspace_id = NULL, scope_type = 'global', version = version + 1 WHERE id = ?",
                        (cid,),
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET deleted = 1 WHERE id = ?", (cid,)
                    )
        else:

            def fail(*_args, **_kwargs):
                raise RuntimeError("private failure details")

            monkeypatch.setattr(
                app.local_chat_conversation_service,
                "get_conversation_metadata"
                if drift == "observation_failure"
                else "set_conversations_archived",
                fail,
            )
        return result

    monkeypatch.setattr(registry, "unarchive_workspace", restore)
    result = await execute_restore(
        app,
        target,
        "chat_and_workspace" if drift == "chat_failure" else "workspace_only",
        replacement_name=None,
        cancel_event=threading.Event(),
    )
    assert result.status == "partial"
    assert result.workspace_changed and not result.chat_changed
    assert result.chat_available is (None if drift == "observation_failure" else False)
    assert "private" not in result.message
    if drift == "chat_archived":
        assert db.get_conversation_by_id(cid)["archived"]
        assert not registry.get_workspace("project").archived
        assert result.next_action == "refresh"
        assert "selected chat is archived" in result.message
        assert "workspace is active" in result.message
        assert "confirm restoration of the chat" in result.message
        assert "workspace remains archived" not in result.message


@pytest.mark.asyncio
async def test_cancel_before_any_write_and_name_collision(recovery_stores):
    app, cid, _ = recovery_stores
    target = await read_recovery_target(app, cid)
    cancel = threading.Event()
    cancel.set()
    outcome = await execute_restore(
        app, target, "chat_and_workspace", replacement_name=None, cancel_event=cancel
    )
    assert outcome.status == "cancelled" and not outcome.workspace_changed
    app.workspace_registry_service.create_workspace(
        workspace_id="other", name="Project"
    )
    outcome = await execute_restore(
        app,
        target,
        "chat_and_workspace",
        replacement_name=None,
        cancel_event=threading.Event(),
    )
    assert outcome.status == "name_conflict" and not outcome.workspace_changed
    assert outcome.next_action == "restore_as"


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["name", "revision"])
async def test_workspace_transaction_rejects_drift_after_preflight(
    recovery_stores, monkeypatch, drift
):
    app, cid, _ = recovery_stores
    registry = app.workspace_registry_service
    target = await read_recovery_target(app, cid)
    original = registry.unarchive_workspace

    def changed(*args, **kwargs):
        if drift == "name":
            with registry.db.transaction() as conn:
                conn.execute(
                    "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
                    ("Changed", "project"),
                )
        else:
            original("project")
            registry.archive_workspace("project")
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "unarchive_workspace", changed)
    result = await execute_restore(
        app,
        target,
        "chat_and_workspace",
        replacement_name="Reviewed",
        cancel_event=threading.Event(),
    )
    assert result.status == "stale"
    assert not result.workspace_changed and not result.chat_changed
    assert registry.get_workspace("project").archived


@pytest.mark.asyncio
async def test_owning_memory_database_uses_storage_call_exception():
    db = CharactersRAGDB(":memory:", client_id="memory-recovery")
    try:
        cid = db.add_conversation({"title": "Memory"})
        db.set_conversations_archived([cid], archived=True, expected_versions={cid: 1})
        app = SimpleNamespace(
            local_chat_conversation_service=ChatConversationService(db)
        )
        target = await read_recovery_target(app, cid)
        result = await execute_restore(
            app,
            target,
            "chat_only",
            replacement_name=None,
            cancel_event=threading.Event(),
        )
        assert result.status == "restored" and result.chat_available
        assert not db.get_conversation_by_id(cid)["archived"]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_explicit_recovery_fixture_closes_real_worker_connections(
    tmp_path, monkeypatch
):
    """Teardown closes handles opened by the actual off-loop target read."""
    import sqlite3

    fixture = recovery_stores.__wrapped__(tmp_path)
    app, cid, db = next(fixture)
    original = db.get_connection
    handles = {id(original()): original()}

    def record_connection(*args, **kwargs):
        connection = original(*args, **kwargs)
        handles[id(connection)] = connection
        return connection

    monkeypatch.setattr(db, "get_connection", record_connection)
    try:
        await read_recovery_target(app, cid)
        assert len(handles) >= 2, "actual worker must own a distinct SQLite handle"
        fixture.close()
        for connection in handles.values():
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
    finally:
        fixture.close()
        with db.quiesce_connections(timeout_seconds=2.0):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "archived_at",
    [
        "2026-09-20T12:00:00+00:00",
        "2026-09-20T12:00:00Z",
        "2026-09-20T13:00:00+01:00",
        "invalid legacy time",
        None,
    ],
)
async def test_searched_workspace_target_uses_same_time_normalization_for_restore(
    recovery_stores, archived_at
):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveSearchRequest
    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService

    app, cid, _ = recovery_stores
    with app.workspace_registry_service.db.transaction() as conn:
        conn.execute(
            "UPDATE workspace_records SET archived_at = ? WHERE workspace_id = ?",
            (archived_at, "project"),
        )
    page = ArchiveSearchService(
        app.local_chat_conversation_service, app.workspace_registry_service
    ).search(ArchiveSearchRequest())
    target = page.items[0].target
    current = await read_recovery_target(app, cid)
    assert current == target
    outcome = await execute_restore(
        app,
        target,
        "chat_and_workspace",
        replacement_name=None,
        cancel_event=threading.Event(),
    )
    assert outcome.status == "restored"
    assert outcome.workspace_changed and outcome.chat_changed
