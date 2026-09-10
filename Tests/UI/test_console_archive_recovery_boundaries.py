"""Recovery failure and navigation boundaries retain exact conversation ownership."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.UI.Console_Modules import archive
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleConversationResumeIntent,
    HandoffChannel,
    PendingHandoffStore,
)


def recovery_app():
    row = {
        "id": "chat-a",
        "version": 4,
        "archived": True,
        "workspace_id": "workspace-a",
    }
    workspace = SimpleNamespace(
        workspace_id="workspace-a", name="Project", archived=True
    )
    registry = SimpleNamespace(
        get_workspace=Mock(return_value=workspace),
        list_workspaces=Mock(return_value=[]),
        unarchive_workspace=Mock(
            side_effect=lambda *a, **kw: setattr(workspace, "archived", False)
        ),
    )
    service = SimpleNamespace(
        get_conversation_metadata=Mock(side_effect=lambda cid: dict(row))
    )
    app = SimpleNamespace(
        local_chat_conversation_service=service,
        workspace_registry_service=registry,
        pending_handoffs=PendingHandoffStore(),
        notify=Mock(),
        post_message=Mock(),
        push_screen=Mock(),
        run_worker=Mock(),
    )
    return app, row, workspace


async def confirm_recovery(app):
    app.push_screen.call_args.kwargs["callback"](True)
    await app.run_worker.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["version", "workspace", "missing"])
async def test_confirmation_revalidates_before_restoring_workspace(monkeypatch, change):
    app, row, workspace = recovery_app()
    restore = AsyncMock(return_value={"changed": {"chat-a": 5}, "failures": {}})
    monkeypatch.setattr(archive, "change_conversation_archive", restore)
    await archive.request_conversation_resume(app, "chat-a")
    if change == "version":
        row["version"] = 5
    elif change == "workspace":
        row["workspace_id"] = "workspace-other"
    else:
        app.local_chat_conversation_service.get_conversation_metadata.side_effect = (
            lambda cid: None
        )
    await confirm_recovery(app)
    assert workspace.archived
    app.workspace_registry_service.unarchive_workspace.assert_not_called()
    restore.assert_not_called()
    assert (
        app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME) is None
    )
    assert "try" in app.notify.call_args.args[0].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_failure", [False, True])
async def test_workspace_restore_partial_failure_is_explicit_and_retryable(
    monkeypatch, storage_failure
):
    app, _row, workspace = recovery_app()
    restore = AsyncMock(
        side_effect=RuntimeError("write failed") if storage_failure else None,
        return_value={"changed": {}, "failures": {"chat-a": "stale_version"}},
    )
    monkeypatch.setattr(archive, "change_conversation_archive", restore)
    await archive.request_conversation_resume(app, "chat-a")
    await confirm_recovery(app)
    assert not workspace.archived
    assert "Workspace restored" in app.notify.call_args.args[0]
    assert "Resume" in app.notify.call_args.args[0]
    assert (
        app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME) is None
    )
    restore.side_effect = None
    restore.return_value = {"changed": {"chat-a": 5}, "failures": {}}
    await archive.request_conversation_resume(app, "chat-a")
    await confirm_recovery(app)
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
    assert claim.value.conversation_id == "chat-a"
    app.workspace_registry_service.unarchive_workspace.assert_called_once()


@pytest.mark.asyncio
async def test_existing_resume_releases_claim_after_navigation():
    handoffs = PendingHandoffStore()
    handoffs.stage(
        HandoffChannel.CONSOLE_CONVERSATION_RESUME,
        ConsoleConversationResumeIntent("chat-a"),
    )
    screen = SimpleNamespace()
    app = SimpleNamespace(screen=screen)
    screen.app = app
    screen.app_instance = SimpleNamespace(
        pending_handoffs=handoffs,
        notify=Mock(),
        local_chat_conversation_service=SimpleNamespace(
            get_conversation_metadata=lambda cid: {"id": cid, "archived": False}
        ),
    )
    screen._ensure_console_chat_store = lambda: SimpleNamespace(
        sessions=lambda: [
            SimpleNamespace(id="session-a", persisted_conversation_id="chat-a")
        ]
    )

    async def activate(*args, **kwargs):
        await asyncio.sleep(0)
        app.screen = object()

    screen._session = SimpleNamespace(_activate_native_console_session=activate)
    await archive.consume_conversation_resume(screen)
    claim = handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
    assert claim is not None and claim.value.conversation_id == "chat-a"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before", "scope", "ui"])
async def test_existing_activation_stops_after_losing_authority(boundary):
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    allowed = boundary != "before"
    store = SimpleNamespace(active_session_id="old")
    controller = SimpleNamespace(
        store=store,
        switch_session=Mock(
            side_effect=lambda sid: setattr(store, "active_session_id", sid)
        ),
    )

    async def cross_boundary(name, *args, **kwargs):
        nonlocal allowed
        await asyncio.sleep(0)
        if boundary == name:
            allowed = False

    fake = SimpleNamespace(
        _ensure_console_chat_controller=lambda: controller,
        _hide_console_activity_notice=Mock(),
        _capture_console_draft_switch_snapshot=Mock(),
        _note_console_follow_intent=Mock(),
        _set_active_workspace_for_session=Mock(),
        _active_native_console_session=lambda: SimpleNamespace(id="session-a"),
        _refresh_console_effective_scope_and_sync=AsyncMock(
            side_effect=lambda *a, **kw: cross_boundary("scope")
        ),
        _sync_native_console_chat_ui=AsyncMock(
            side_effect=lambda: cross_boundary("ui")
        ),
        _focus_console_composer_if_needed=Mock(),
    )

    # AsyncMock side effects must await the actual boundary coroutine.
    async def scope(*args, **kwargs):
        await cross_boundary("scope")

    async def ui():
        await cross_boundary("ui")

    fake._refresh_console_effective_scope_and_sync.side_effect = scope
    fake._sync_native_console_chat_ui.side_effect = ui
    await ConsoleSessionController._activate_native_console_session(
        fake, "session-a", activate_if=lambda: allowed
    )
    fake._focus_console_composer_if_needed.assert_not_called()
    if boundary in {"before", "scope"}:
        fake._sync_native_console_chat_ui.assert_not_called()
    if boundary == "before":
        controller.switch_session.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_guard_prevents_stale_projection_paint():
    from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController

    allowed = True

    async def resolve(session):
        nonlocal allowed
        await asyncio.sleep(0)
        allowed = False

    fake = SimpleNamespace(
        _resolve_console_effective_scope_state=resolve,
        _is_mounted=lambda: True,
        _sync_retrieval_scope_row=Mock(),
        _sync_control_bar=Mock(),
    )
    await ConsoleRetrievalController._refresh_console_effective_scope_and_sync(
        fake, object(), refresh_if=lambda: allowed
    )
    fake._sync_retrieval_scope_row.assert_not_called()
    fake._sync_control_bar.assert_not_called()


@pytest.mark.asyncio
async def test_superseded_existing_resume_drains_latest_request():
    handoffs = PendingHandoffStore()
    channel = HandoffChannel.CONSOLE_CONVERSATION_RESUME
    handoffs.stage(channel, ConsoleConversationResumeIntent("chat-a"))
    screen = SimpleNamespace()
    screen.app = SimpleNamespace(screen=screen)
    screen.app_instance = SimpleNamespace(
        pending_handoffs=handoffs,
        notify=Mock(),
        local_chat_conversation_service=SimpleNamespace(
            get_conversation_metadata=lambda cid: {"id": cid, "archived": False}
        ),
    )
    screen._ensure_console_chat_store = lambda: SimpleNamespace(
        sessions=lambda: [
            SimpleNamespace(id="session-a", persisted_conversation_id="chat-a"),
            SimpleNamespace(id="session-b", persisted_conversation_id="chat-b"),
        ]
    )
    focused = []

    async def activate(session_id, *, activate_if):
        if session_id == "session-a":
            handoffs.stage(channel, ConsoleConversationResumeIntent("chat-b"))
        await asyncio.sleep(0)
        if activate_if():
            focused.append(session_id)

    screen._session = SimpleNamespace(_activate_native_console_session=activate)
    await archive.consume_conversation_resume(screen)
    assert focused == ["session-b"]
    assert handoffs.claim(channel) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", ["metadata", "workspace", "recovery", "resume", "archive", "undo"]
)
async def test_recovery_error_logs_bind_only_identity_context(monkeypatch, phase):
    from loguru import logger

    records = []
    sink = logger.add(lambda message: records.append(message.record), level="ERROR")
    app, row, _workspace = recovery_app()
    row.update(title="PRIVATE TITLE", content="PRIVATE MESSAGE")
    try:
        if phase in {"metadata", "workspace", "recovery"}:
            if phase == "metadata":
                app.local_chat_conversation_service.get_conversation_metadata.side_effect = RuntimeError(
                    "read failed"
                )
            elif phase == "workspace":
                app.workspace_registry_service.get_workspace.side_effect = RuntimeError(
                    "workspace read failed"
                )
            else:
                app.workspace_registry_service.unarchive_workspace.side_effect = (
                    RuntimeError("restore failed")
                )
            await archive.request_conversation_resume(app, "chat-a")
            if phase == "recovery":
                await confirm_recovery(app)
        elif phase == "resume":
            row["archived"] = False
            _workspace.archived = False
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_CONVERSATION_RESUME,
                ConsoleConversationResumeIntent("chat-a"),
            )
            screen = SimpleNamespace(app_instance=app)
            screen.app = SimpleNamespace(screen=screen)
            screen._ensure_console_chat_store = lambda: SimpleNamespace(
                sessions=lambda: [
                    SimpleNamespace(
                        id="session-a",
                        persisted_conversation_id="chat-a",
                        workspace_id="workspace-a",
                    )
                ]
            )
            screen._session = SimpleNamespace(
                _activate_native_console_session=AsyncMock(
                    side_effect=RuntimeError("activate failed")
                )
            )
            await archive.consume_conversation_resume(screen)
        else:
            callbacks = []

            async def push_screen(modal, *, callback):
                callbacks.append(callback)

            screen = SimpleNamespace(
                app_instance=app,
                app=SimpleNamespace(push_screen=push_screen),
                _current_console_conversation_id=lambda: "chat-a",
                _workspace=SimpleNamespace(
                    _invalidate_console_persisted_rows_cache=Mock()
                ),
                _sync_native_console_chat_ui=AsyncMock(),
            )
            if phase == "archive":
                app.local_chat_conversation_service.get_conversation_metadata.side_effect = RuntimeError(
                    "read failed"
                )
            else:
                change = AsyncMock(
                    side_effect=[
                        {"changed": {"chat-a": 5}, "failures": {}},
                        RuntimeError("undo failed"),
                    ]
                )
                monkeypatch.setattr(archive, "change_conversation_archive", change)
            await archive.archive_current_conversation(screen)
            if phase == "undo":
                await callbacks[0]("undo")
        assert len(records) == 1
        assert records[0]["extra"]["conversation_id"] == "chat-a"
        if phase in {"workspace", "recovery", "resume", "undo"}:
            assert records[0]["extra"]["workspace_id"] == "workspace-a"
        assert set(records[0]["extra"]) <= {"conversation_id", "workspace_id"}
        assert records[0]["exception"] is not None
        assert "PRIVATE" not in records[0]["message"]
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("archived", [True, False])
async def test_existing_resume_rechecks_durable_archive_before_activation(
    monkeypatch, archived
):
    app, row, workspace = recovery_app()
    row["archived"] = archived
    workspace.archived = False
    screen = SimpleNamespace(app_instance=app)
    screen.app = SimpleNamespace(screen=screen)
    screen._ensure_console_chat_store = lambda: SimpleNamespace(
        sessions=lambda: [
            SimpleNamespace(id="open", persisted_conversation_id="chat-a")
        ]
    )
    activate = AsyncMock()
    screen._session = SimpleNamespace(_activate_native_console_session=activate)
    request = AsyncMock()
    monkeypatch.setattr(archive, "request_conversation_resume", request)
    channel = HandoffChannel.CONSOLE_CONVERSATION_RESUME
    app.pending_handoffs.stage(channel, ConsoleConversationResumeIntent("chat-a"))
    await archive.consume_conversation_resume(screen)
    if archived:
        activate.assert_not_called()
        request.assert_awaited_once_with(app, "chat-a")
        assert app.pending_handoffs.claim(channel) is not None
    else:
        activate.assert_awaited_once()
        assert app.pending_handoffs.claim(channel) is None


@pytest.mark.asyncio
async def test_initially_active_resume_revalidates_before_staging():
    app, row, workspace = recovery_app()
    workspace.archived = False
    row["archived"] = False
    app.local_chat_conversation_service.get_conversation_metadata.side_effect = [
        dict(row), {**row, "archived": True, "version": 5},
    ]
    await archive.request_conversation_resume(app, "chat-a")
    app.post_message.assert_not_called()
    assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME) is None
    assert "changed" in app.notify.call_args.args[0]
