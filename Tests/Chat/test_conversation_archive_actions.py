from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_archive_refuses_live_work_without_calling_storage():
    from tldw_chatbook.Chat.conversation_archive_actions import (
        conversation_archive_refusal,
    )

    session = SimpleNamespace(
        id="s", persisted_conversation_id="c", draft="", pending_attachments=[]
    )
    store = SimpleNamespace(
        sessions=lambda: [session], messages_for_session=lambda _: []
    )
    controller = SimpleNamespace(
        lifecycle_impact=lambda **_: SimpleNamespace(has_loss_risk=True)
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=controller)
    )
    assert "running or queued" in conversation_archive_refusal(app, "c")


@pytest.mark.asyncio
async def test_archive_only_mutates_safe_targets_and_keeps_exact_versions():
    from tldw_chatbook.Chat.conversation_archive_actions import (
        change_conversation_archive,
    )

    session = SimpleNamespace(
        id="s",
        persisted_conversation_id="busy",
        draft="keep me",
        pending_attachments=[],
    )
    store = SimpleNamespace(
        sessions=lambda: [session], messages_for_session=lambda _: []
    )
    change = Mock(return_value={"changed": {"idle": 3}, "failures": {}})
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=None),
        local_chat_conversation_service=SimpleNamespace(
            set_conversations_archived=change
        ),
    )
    result = await change_conversation_archive(
        app, ["idle", "busy"], archived=True, expected_versions={"idle": 2, "busy": 5}
    )
    assert result["changed"] == {"idle": 3}
    assert "draft" in result["failures"]["busy"]
    change.assert_called_once_with(
        ["idle"], archived=True, expected_versions={"idle": 2}
    )
    assert not app._conversation_archive_inflight


def test_typed_resume_channel_preserves_identity_and_rejects_source_payload():
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        ConsoleConversationResumeIntent,
        HandoffChannel,
        PendingHandoffStore,
    )

    store = PendingHandoffStore()
    store.stage(
        HandoffChannel.CONSOLE_CONVERSATION_RESUME,
        ConsoleConversationResumeIntent("saved-original"),
    )
    claim = store.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
    assert claim.value.conversation_id == "saved-original"
    with pytest.raises((ValueError, TypeError)):
        store.stage(HandoffChannel.CONSOLE_CONVERSATION_RESUME, {"source_id": "other"})


@pytest.mark.asyncio
async def test_resume_reuses_open_session_and_acknowledges_after_activation():
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Console_Modules.archive import consume_conversation_resume
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        ConsoleConversationResumeIntent,
        HandoffChannel,
        PendingHandoffStore,
    )

    handoffs = PendingHandoffStore()
    handoffs.stage(
        HandoffChannel.CONSOLE_CONVERSATION_RESUME,
        ConsoleConversationResumeIntent("original"),
    )
    session = SimpleNamespace(
        id="already-open", persisted_conversation_id="original", draft="preserved"
    )
    activate = AsyncMock()
    hydrate = AsyncMock()
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            pending_handoffs=handoffs,
            local_chat_conversation_service=SimpleNamespace(
                get_conversation_metadata=lambda cid: {"id": cid, "archived": False}
            ),
        ),
        _ensure_console_chat_store=lambda: SimpleNamespace(sessions=lambda: [session]),
        _session=SimpleNamespace(_activate_native_console_session=activate),
        _workspace=SimpleNamespace(_resume_console_workspace_conversation=hydrate),
    )
    screen.app = SimpleNamespace(screen=screen)
    await consume_conversation_resume(screen)
    activate.assert_awaited_once()
    assert activate.await_args.args == ("already-open",)
    assert callable(activate.await_args.kwargs["activate_if"])
    hydrate.assert_not_called()
    assert session.draft == "preserved"
    assert handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME) is None


@pytest.mark.asyncio
async def test_real_archive_send_guard_restore_resume_keeps_original_id(tmp_path):
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.conversation_archive_actions import (
        change_conversation_archive,
        conversation_send_refusal,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.archive import request_conversation_resume
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )

    db = CharactersRAGDB(tmp_path / "resume.db", client_id="resume-test")
    service = ChatConversationService(db)
    cid = service.create_conversation(title="Original")
    db.add_message(
        {"conversation_id": cid, "sender": "user", "content": "Retained history"}
    )
    dialogs = []
    workers = []
    app = SimpleNamespace(
        local_chat_conversation_service=service,
        pending_handoffs=PendingHandoffStore(),
        notify=Mock(),
        post_message=Mock(),
        push_screen=lambda dialog, **kw: dialogs.append((dialog, kw)),
        run_worker=lambda coroutine, **kw: workers.append(coroutine),
    )
    try:
        version = service.get_conversation_metadata(cid)["version"]
        await change_conversation_archive(
            app, [cid], archived=True, expected_versions={cid: version}
        )
        assert "archived" in await conversation_send_refusal(app, cid)
        await request_conversation_resume(app, cid)
        assert len(dialogs) == 1
        assert (
            app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
            is None
        )
        dialogs[0][1]["callback"](True)
        await workers.pop()
        assert await conversation_send_refusal(app, cid) is None
        assert (
            app.pending_handoffs.claim(
                HandoffChannel.CONSOLE_CONVERSATION_RESUME
            ).value.conversation_id
            == cid
        )
        assert (
            service.get_library_conversation_messages(cid)["messages"][0]["text"]
            == "Retained history"
        )
        app.post_message.assert_called_once()
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_archive_refuses_unsaved_transcript_and_clears_reservation_on_failure():
    from tldw_chatbook.Chat.conversation_archive_actions import (
        change_conversation_archive,
        conversation_archive_refusal,
    )

    session = SimpleNamespace(
        id="s", persisted_conversation_id="c", draft="", pending_attachments=[]
    )
    message = SimpleNamespace(id="unsaved", persisted_message_id=None)
    store = SimpleNamespace(
        sessions=lambda: [session], messages_for_session=lambda _: [message]
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=None),
        local_chat_conversation_service=SimpleNamespace(
            set_conversations_archived=Mock(side_effect=RuntimeError("disk"))
        ),
    )
    assert "saving" in conversation_archive_refusal(app, "c")
    with pytest.raises(RuntimeError):
        await change_conversation_archive(
            app, ["other"], archived=True, expected_versions={"other": 2}
        )
    assert not app._conversation_archive_inflight


@pytest.mark.asyncio
async def test_newer_resume_is_drained_when_staged_during_previous_load():
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Console_Modules.archive import consume_conversation_resume
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        ConsoleConversationResumeIntent,
        HandoffChannel,
        PendingHandoffStore,
    )

    handoffs = PendingHandoffStore()
    handoffs.stage(
        HandoffChannel.CONSOLE_CONVERSATION_RESUME,
        ConsoleConversationResumeIntent("first"),
    )
    loaded = []

    async def hydrate(cid, **kwargs):
        loaded.append(cid)
        if cid == "first":
            handoffs.stage(
                HandoffChannel.CONSOLE_CONVERSATION_RESUME,
                ConsoleConversationResumeIntent("latest"),
            )
            assert kwargs["resume_if"]() is False
            return None
        assert kwargs["preserve_persisted_scope"] is True
        return True

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            pending_handoffs=handoffs,
            notify=Mock(),
            local_chat_conversation_service=SimpleNamespace(
                get_conversation_metadata=lambda cid: {"id": cid, "archived": False}
            ),
        ),
        _ensure_console_chat_store=lambda: SimpleNamespace(sessions=list),
        _session=SimpleNamespace(_activate_native_console_session=AsyncMock()),
        _workspace=SimpleNamespace(_resume_console_workspace_conversation=hydrate),
    )
    screen.app = SimpleNamespace(screen=screen)
    await consume_conversation_resume(screen)
    assert loaded == ["first", "latest"]
    assert not handoffs.has_pending(HandoffChannel.CONSOLE_CONVERSATION_RESUME)


def test_archive_refuses_send_before_controller_has_started_a_run():
    from tldw_chatbook.Chat.conversation_archive_actions import (
        conversation_archive_refusal,
    )

    session = SimpleNamespace(
        id="s", persisted_conversation_id="c", draft="", pending_attachments=[]
    )
    app = SimpleNamespace(
        _conversation_send_inflight={"c": 1},
        console_runtime=SimpleNamespace(
            chat_store=SimpleNamespace(sessions=lambda: [session]), chat_controller=None
        ),
    )
    assert "send" in conversation_archive_refusal(app, "c")


@pytest.mark.asyncio
async def test_send_rechecks_archive_reservation_after_async_storage_read(monkeypatch):
    from tldw_chatbook.Chat import conversation_archive_actions as actions

    app = SimpleNamespace(
        local_chat_conversation_service=object(), _conversation_archive_inflight=set()
    )

    async def read(*args, **kwargs):
        app._conversation_archive_inflight.add("c")
        return {"c": False}

    monkeypatch.setattr(actions, "storage_call", read)
    assert "Wait" in await actions.conversation_send_refusal(app, "c")


@pytest.mark.asyncio
async def test_cancelled_archive_holds_reservation_until_thread_write_settles():
    import asyncio
    import threading

    from tldw_chatbook.Chat.conversation_archive_actions import (
        change_conversation_archive,
        conversation_send_refusal,
    )

    started, finish = threading.Event(), threading.Event()
    states = {"c": False}

    def mutate(ids, **kwargs):
        started.set()
        assert finish.wait(5)
        states["c"] = True
        return {"changed": {"c": 2}, "failures": {}}

    service = SimpleNamespace(
        set_conversations_archived=mutate,
        get_conversation_archive_states=lambda ids: dict(states),
    )
    app = SimpleNamespace(local_chat_conversation_service=service)
    operation = asyncio.create_task(
        change_conversation_archive(
            app, ["c"], archived=True, expected_versions={"c": 1}
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert "Wait" in await conversation_send_refusal(app, "c")
    finally:
        finish.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if not app._conversation_archive_inflight:
                break
    assert not app._conversation_archive_inflight
    assert app._conversation_archive_states == {"c": True}


@pytest.mark.asyncio
async def test_send_reconciles_external_restore_and_preserves_unrelated_cache():
    from tldw_chatbook.Chat.conversation_archive_actions import (
        conversation_send_refusal,
    )

    app = SimpleNamespace(
        local_chat_conversation_service=SimpleNamespace(
            get_conversation_archive_states=lambda ids: {"c": False}
        ),
        _conversation_archive_states={"c": True, "other": True},
    )
    assert await conversation_send_refusal(app, "c") is None
    assert app._conversation_archive_states == {"c": False, "other": True}


@pytest.mark.asyncio
async def test_send_does_not_publish_read_older_than_completed_archive(monkeypatch):
    from tldw_chatbook.Chat import conversation_archive_actions as actions

    app = SimpleNamespace(
        local_chat_conversation_service=SimpleNamespace(
            set_conversations_archived=lambda *a, **k: {
                "changed": {"c": 2},
                "failures": {},
            }
        ),
        _conversation_archive_states={"c": False},
    )
    original_call = actions.storage_call

    async def interleaved_read(service, method, *args, **kwargs):
        if method == "get_conversation_archive_states":
            await actions.change_conversation_archive(
                app, ["c"], archived=True, expected_versions={"c": 1}
            )
            return {"c": False}
        return await original_call(service, method, *args, **kwargs)

    monkeypatch.setattr(actions, "storage_call", interleaved_read)
    assert await actions.conversation_send_refusal(app, "c") is not None
    assert app._conversation_archive_states == {"c": True}


@pytest.mark.parametrize("hidden_state", ["unsaved", "pending", "saved"])
def test_archive_checks_hidden_branch_persistence_for_conversation_and_workspace(
    hidden_state,
):
    from tldw_chatbook.Chat.conversation_archive_actions import (
        conversation_archive_refusal,
        workspace_archive_refusal,
    )

    session = SimpleNamespace(
        id="session",
        persisted_conversation_id="conversation",
        workspace_id="workspace",
        draft="",
        pending_attachments=[],
    )
    active = SimpleNamespace(id="active", persisted_message_id="saved-active")
    hidden = SimpleNamespace(
        id="hidden",
        persisted_message_id=None if hidden_state == "unsaved" else "saved-hidden",
    )
    store = SimpleNamespace(
        sessions=lambda: [session],
        messages_for_session=lambda _: [active],
        all_messages_for_session=lambda _: [active, hidden],
        _pending_persistence_message_ids={"hidden"}
        if hidden_state == "pending"
        else set(),
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=None)
    )

    for refusal in (
        conversation_archive_refusal(app, "conversation"),
        workspace_archive_refusal(app, "workspace"),
    ):
        if hidden_state == "saved":
            assert refusal is None
        else:
            assert refusal is not None and "saving" in refusal


@pytest.mark.parametrize("surface", ["stack", "retained"])
@pytest.mark.parametrize("guard_kind", ["conversation", "workspace"])
def test_archive_captures_covered_console_draft_for_its_visible_owner(
    surface, guard_kind
):
    from tldw_chatbook.Chat.conversation_archive_actions import (
        conversation_archive_refusal,
        workspace_archive_refusal,
    )

    owner = SimpleNamespace(
        id="owner", persisted_conversation_id="c", workspace_id="w", draft=""
    )
    active = SimpleNamespace(
        id="active", persisted_conversation_id="other", workspace_id="other", draft=""
    )
    store = SimpleNamespace(
        sessions=lambda: [owner, active],
        active_session_id="active",
        messages_for_session=lambda _: [],
        set_session_draft=lambda sid, text: setattr(
            owner if sid == "owner" else active, "draft", text
        ),
    )
    text = [""]
    console = SimpleNamespace(
        is_mounted=True,
        _console_visible_draft_session_id="owner",
        _console_composer_or_none=lambda: SimpleNamespace(draft_text=lambda: text[0]),
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=None),
        screen_stack=[console, object()] if surface == "stack" else [],
        _reusable_screen_instances={"chat": (object(), console)}
        if surface == "retained"
        else {},
    )
    check = (
        (lambda: conversation_archive_refusal(app, "c"))
        if guard_kind == "conversation"
        else (lambda: workspace_archive_refusal(app, "w"))
    )
    assert check() is None
    text[0] = "Typed while the Console is covered"
    assert "draft" in (check() or "")
    assert owner.draft == text[0]
    assert active.draft == ""


@pytest.mark.parametrize("owner_is_active", [True, False])
def test_empty_archive_composer_clears_stale_draft_only_for_settled_owner(
    owner_is_active,
):
    from tldw_chatbook.Chat.conversation_archive_actions import (
        workspace_archive_refusal,
    )

    owner = SimpleNamespace(
        id="owner",
        persisted_conversation_id="c",
        workspace_id="w",
        draft="Previously typed",
    )
    store = SimpleNamespace(
        active_session_id="owner" if owner_is_active else "new-active",
        sessions=lambda: [owner],
        messages_for_session=lambda _: [],
        set_session_draft=lambda _, text: setattr(owner, "draft", text),
    )
    console = SimpleNamespace(
        is_mounted=True,
        _console_visible_draft_session_id="owner",
        _console_composer_or_none=lambda: SimpleNamespace(draft_text=lambda: ""),
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=None),
        screen_stack=[console],
    )
    refusal = workspace_archive_refusal(app, "w")
    if owner_is_active:
        assert refusal is None
        assert owner.draft == ""
    else:
        assert "draft" in refusal
        assert owner.draft == "Previously typed"
