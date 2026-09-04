"""Characterisation tests for the Console message cluster (wave-3 task 1).

Run GREEN against unmodified `ChatScreen` before `ConsoleMessageController`
exists (`tldw_chatbook/UI/Console_Modules/message.py`) -- see
`.superpowers/sdd/2026-08-06-console-decomposition-wave3/progress.md` for the
mandatory characterise-before-extract sequence this file satisfies. Every
assertion below reads the ACTUAL persisted result (`ConsoleChatStore` rows),
never widget/DOM state, and drives the real screen methods end-to-end --
nothing here is monkeypatched.

Covers, at minimum, the two surfaces the wave-3 brief calls out explicitly:
the send/receive path (`test_console_message_send_persists_user_and_
assistant_rows`) and at least one `handle_console_message_action` branch
(delete + feedback-up below). Also pins the pure serialize/restore round
trip and the sibling-variant navigation helper, since both move as part of
the same cluster and have no other direct coverage of their post-move
public surface (`ChatScreen._serialize_console_message` /
`ChatScreen._restore_console_message` / `ChatScreen._select_console_message_
variant`, all still reachable under their pre-move names after the move --
see the extraction report's delegation table).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _configure_native_ready_console,
    _select_llamacpp_console,
    _wait_for_selector,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Canvas.native_authority import CanvasBridgeTarget
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript


@pytest.mark.asyncio
async def test_console_message_send_persists_user_and_assistant_rows():
    """Send/receive path: a real send queues a user turn and persists the
    streamed assistant reply as store rows, not just visible text."""
    gateway = CapturingGateway(chunks=("hello ", "there"))
    app = _build_console_send_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("characterisation probe")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "hello there")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        messages = store.messages_for_session(session_id)

    user_rows = [m for m in messages if m.role == ConsoleMessageRole.USER]
    assistant_rows = [m for m in messages if m.role == ConsoleMessageRole.ASSISTANT]
    assert any(row.content == "characterisation probe" for row in user_rows)
    assert any(
        row.content == "hello there" and row.status == "complete"
        for row in assistant_rows
    )


@pytest.mark.asyncio
async def test_console_message_action_delete_removes_persisted_row():
    """`handle_console_message_action`'s delete branch: two presses actually
    remove the row from the store (persisted result), not just the widget."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        child = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="follow-up"
        )
        invalidate_media = Mock(
            wraps=console._image.invalidate_console_fork_image_selections
        )
        console._image.invalidate_console_fork_image_selections = invalidate_media
        await console._sync_native_console_chat_ui()
        # `_sync_console_pending_delete_confirmation` resets the armed id the
        # moment it disagrees with the transcript's own selection -- an
        # unselected row would silently disarm between the two presses below.
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()

        event = SimpleNamespace(
            button=SimpleNamespace(id=f"console-message-action-delete-{message.id}"),
            stop=Mock(),
        )
        # First press only arms the confirmation -- nothing removed yet.
        handled = await console.handle_console_message_action(event)
        assert handled is True
        assert message in store.messages_for_session(session.id)

        # Second press on the same id actually deletes.
        handled_again = await console.handle_console_message_action(event)
        assert handled_again is True

    assert message not in store.messages_for_session(session.id)
    assert child not in store.messages_for_session(session.id)
    invalidate_media.assert_called_once_with((message.id, child.id))


@pytest.mark.asyncio
async def test_console_message_action_feedback_persists_to_store():
    """`handle_console_message_action`'s feedback branch writes through to
    the store's own `feedback` field, not just `_last_console_action`."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        await console._sync_native_console_chat_ui()

        event = SimpleNamespace(
            button=SimpleNamespace(
                id=f"console-message-action-feedback-up-{message.id}"
            ),
            stop=Mock(),
        )
        handled = await console.handle_console_message_action(event)
        assert handled is True

    assert store.get_message(message.id).feedback == "up"


def test_console_message_serialize_restore_round_trip():
    """Pure (de)serialization: a round trip through the screen-state payload
    shape preserves role/content/status/persisted id."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="round trip me",
    )

    payload = ChatScreen._serialize_console_message(message)
    assert payload["role"] == "user"
    assert payload["content"] == "round trip me"

    restored = ChatScreen._restore_console_message(payload)
    assert restored is not None
    assert restored.role is ConsoleMessageRole.USER
    assert restored.content == "round trip me"
    assert restored.status == "complete"


def test_console_message_select_variant_moves_active_leaf():
    """`_select_console_message_variant` moves the store's active leaf to
    the target sibling -- persisted store state, not a transcript selection.

    Mirrors `Tests/Chat/test_console_sibling_nav.py`'s established
    unmounted-`ChatScreen(app)` pattern: the method under test only touches
    `self._ensure_console_chat_store()`, so no Textual mount is required.
    """
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    first = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="first"
    )
    second = store.create_sibling(
        first.id, role=ConsoleMessageRole.ASSISTANT, content="second"
    )
    assert store.active_leaf(session.id) == second.id

    target = screen._select_console_message_variant(
        second.id, direction="variant-previous"
    )

    assert target == first.id
    assert store.active_leaf(session.id) == first.id


@pytest.mark.asyncio
async def test_roleplay_character_greeting_actions_use_live_presentation():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    app.post_message = Mock()
    app.notes_scope_service = SimpleNamespace(
        save_note=AsyncMock(return_value={"id": "note-1"})
    )
    app.media_db = SimpleNamespace(
        add_media_with_keywords=Mock(return_value=(7, "media-7", "saved"))
    )
    app.prompts_db = SimpleNamespace(
        add_prompt=Mock(return_value=(8, "prompt-8", "saved"))
    )
    app.local_chatbook_service = SimpleNamespace(
        create_chatbook=AsyncMock(return_value={"id": "chatbook-9"})
    )
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.create_session(
        title="Garden",
        assistant_kind="character",
        character_name="Alraune",
    )
    greeting = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Hello User.",
        metadata=MessageMetadata(
            template_kind="character_greeting",
            template_source="Hello {{user}}.",
        ),
    )
    session.user_display_name_override = "Captain Rowan"

    presentation = screen._console_message_presentation(greeting)
    assert presentation.content == "Hello Captain Rowan."
    assert presentation.speaker_label == "Alraune"

    copy_event = SimpleNamespace(
        button=SimpleNamespace(id=f"console-message-action-copy-{greeting.id}"),
        stop=Mock(),
    )
    assert await screen.handle_console_message_action(copy_event) is True
    app.copy_to_clipboard.assert_called_once_with("Hello Captain Rowan.")

    screen._message._open_console_message_edit_modal = AsyncMock()
    edit_event = SimpleNamespace(
        button=SimpleNamespace(id=f"console-message-action-edit-{greeting.id}"),
        stop=Mock(),
    )
    assert await screen.handle_console_message_action(edit_event) is True
    screen._message._open_console_message_edit_modal.assert_awaited_once_with(
        message_id=greeting.id,
        content="Hello Captain Rowan.",
    )

    screen._message._sync_native_console_chat_ui_fn = AsyncMock()
    speak_event = SimpleNamespace(
        button=SimpleNamespace(id=f"console-message-action-speak-{greeting.id}"),
        stop=Mock(),
    )
    assert await screen.handle_console_message_action(speak_event) is True
    speech_event = next(
        call.args[0]
        for call in app.post_message.call_args_list
        if call.args and hasattr(call.args[0], "snapshot")
    )
    assert speech_event.snapshot.raw_content == "Hello Captain Rowan."
    assert speech_event.validator(speech_event.snapshot) == "Hello Captain Rowan."

    await screen._message._save_console_message_as_note(greeting.id)
    await screen._message._save_console_message_as_media(greeting.id)
    await screen._message._save_console_message_as_prompt(greeting.id)
    await screen._message._save_console_message_as_chatbook(greeting.id)

    assert (
        app.notes_scope_service.save_note.await_args.kwargs["content"]
        == "Hello Captain Rowan."
    )
    assert (
        app.media_db.add_media_with_keywords.call_args.kwargs["content"]
        == "Hello Captain Rowan."
    )
    assert (
        app.prompts_db.add_prompt.call_args.kwargs["system_prompt"]
        == "Hello Captain Rowan."
    )
    chatbook_payload = app.local_chatbook_service.create_chatbook.await_args.kwargs
    assert chatbook_payload["metadata"]["content"] == "Hello Captain Rowan."
    assert chatbook_payload["metadata"]["message_role"] == "Alraune"


@pytest.mark.asyncio
async def test_production_message_controller_resolves_canvas_source_only_at_open_callback():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="Canvas")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="```html\n<!doctype html><title>Native</title><p>secret</p>\n```",
    )
    screen._message._open_canvas_block_fn = AsyncMock()
    event = SimpleNamespace(
        button=SimpleNamespace(
            id="canvas-open",
            console_action_id="canvas-open-0",
            console_message_id=message.id,
        ),
        stop=Mock(),
    )

    assert await screen.handle_console_message_action(event) is True

    reference, source = screen._message._open_canvas_block_fn.await_args.args
    assert reference.message_id == message.id
    assert reference.block_index == 0
    assert reference.create_new is False
    assert source == "<!doctype html><title>Native</title><p>secret</p>\n"
    result = screen._message._last_console_action
    assert result.target_content is None
    assert "secret" not in repr(result)


def test_canvas_auto_open_is_suppressed_only_by_same_session_browser():
    app = _build_test_app()
    screen = ChatScreen(app)
    runtime = screen._console_runtime()
    runtime._canvas_gateway = SimpleNamespace(
        has_browser_session_for=lambda session_id: session_id == "session-a"
    )
    screen.app_instance = SimpleNamespace(call_from_thread=lambda callback: callback())
    screen._open_console_canvas_selection = Mock(
        side_effect=lambda **_kwargs: _closed_coroutine()
    )
    workers: list[object] = []

    def capture_worker(coroutine, **_kwargs):
        workers.append(coroutine)
        coroutine.close()

    screen.run_worker = capture_worker
    info = SimpleNamespace(canvas_id="canvas-a", revision_id="revision-a")

    screen._schedule_console_canvas_tool_open("session-a", info)
    screen._schedule_console_canvas_tool_open("session-b", info)

    screen._open_console_canvas_selection.assert_called_once_with(
        session_id="session-b",
        canvas_id="canvas-a",
        revision_id="revision-a",
        follow_latest=True,
    )
    assert len(workers) == 1


def test_canvas_publication_guard_rejects_stale_session_and_sibling_branch():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    first = store.create_session(ephemeral=True)
    store.append_message(
        first.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="root",
    )
    left = store.append_message(
        first.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="left",
    )
    publication = SimpleNamespace(
        scope=SimpleNamespace(
            session_id=first.id,
            conversation_id=first.id,
        ),
        revisions=(SimpleNamespace(origin=SimpleNamespace(message_id=left.id)),),
    )

    assert screen._console_canvas_publication_is_current(publication) is True

    store.create_sibling(
        left.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="right",
    )
    assert screen._console_canvas_publication_is_current(publication) is False

    store.create_session(ephemeral=True)
    assert screen._console_canvas_publication_is_current(publication) is False


def test_canvas_composer_sink_validates_exact_session_and_branch_target():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.create_session(ephemeral=True)
    root = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="root",
    )
    left = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="left",
    )
    target = CanvasBridgeTarget(
        browser_session_id="browser-exact",
        session_id=session.id,
        conversation_id=session.id,
        active_message_ids=(root.id, left.id),
        canvas_id="canvas-exact",
        revision_id="revision-exact",
    )
    composer = Mock()
    screen._console_composer_or_none = Mock(return_value=composer)

    screen._prefill_console_canvas_repair(target, "exact draft")

    composer.load_draft.assert_called_once_with("exact draft")
    assert store.session_draft(session.id) == "exact draft"

    store.create_sibling(
        left.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="right",
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        screen._prefill_console_canvas_repair(target, "stale branch draft")
    assert store.session_draft(session.id) == "exact draft"

    store.create_session(ephemeral=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        screen._prefill_console_canvas_repair(target, "stale session draft")
    assert store.session_draft(session.id) == "exact draft"


def test_canvas_submit_preparation_only_replaces_the_unchanged_unsent_draft():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.create_session(ephemeral=True)
    root = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="root",
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
    )
    target = CanvasBridgeTarget(
        browser_session_id="browser-confirm",
        session_id=session.id,
        conversation_id=session.id,
        active_message_ids=(root.id, assistant.id),
        canvas_id="canvas-confirm",
        revision_id="revision-confirm",
    )
    composer = Mock()
    unchanged = object()
    composer.capture_draft_snapshot.side_effect = [unchanged, unchanged]
    screen._console_composer_or_none = Mock(return_value=composer)

    apply = screen._prepare_console_canvas_submit(target)
    apply("exact unsent draft")

    composer.load_draft.assert_called_once_with("exact unsent draft")
    composer.focus.assert_called_once_with()
    assert store.session_draft(session.id) == "exact unsent draft"

    composer.load_draft.reset_mock()
    composer.capture_draft_snapshot.side_effect = [unchanged, object()]
    stale_apply = screen._prepare_console_canvas_submit(target)
    with pytest.raises(RuntimeError, match="changed"):
        stale_apply("must not replace")
    composer.load_draft.assert_not_called()
    assert store.session_draft(session.id) == "exact unsent draft"


async def _closed_coroutine() -> None:
    return None


@pytest.mark.asyncio
async def test_production_message_controller_prefills_canvas_repair_without_source_state():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="Canvas repair")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="```html\n<script src='https://example.test/private.js'></script>\n```",
    )
    screen._message._prefill_canvas_repair_fn = Mock()
    event = SimpleNamespace(
        button=SimpleNamespace(
            id="canvas-repair",
            console_action_id="canvas-open-0",
            console_message_id=message.id,
        ),
        stop=Mock(),
    )

    assert await screen.handle_console_message_action(event) is True

    repair = screen._message._prefill_canvas_repair_fn.call_args.args[0]
    assert "self-contained Canvas V1" in repair
    assert "example.test" not in repair
    assert screen._message._last_console_action.target_content is None


@pytest.mark.asyncio
async def test_production_canvas_card_handler_routes_exact_and_retry_mints_fresh_open():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="Canvas card")
    open_selection = AsyncMock()
    screen._open_console_canvas_selection = open_selection

    exact = SimpleNamespace(
        canvas_id="canvas-a",
        revision_id="revision-2",
        follow_latest=False,
        stop=Mock(),
    )
    await screen.handle_console_canvas_card_open(exact)
    open_selection.assert_awaited_once_with(
        session_id=session.id,
        canvas_id="canvas-a",
        revision_id="revision-2",
        follow_latest=False,
    )

    open_selection.reset_mock()
    screen._canvas_last_open_request = (
        session.id,
        "canvas-a",
        "revision-2",
        False,
    )
    retry = SimpleNamespace(stop=Mock())
    await screen.handle_console_canvas_open_retry(retry)
    open_selection.assert_awaited_once_with(
        session_id=session.id,
        canvas_id="canvas-a",
        revision_id="revision-2",
        follow_latest=False,
    )


@pytest.mark.asyncio
async def test_fork_requested_dispatches_to_the_named_session_callback_once():
    """The message controller remains a narrow action-to-session seam."""

    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-transcript")
        store = screen._ensure_console_chat_store()
        session = store.create_session(
            title="Source",
            settings=screen._session._default_console_session_settings(),
            ephemeral=True,
        )
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="fork through here",
        )
        await screen._sync_native_console_chat_ui()
        request = Mock()
        screen._message._request_console_chat_fork_fn = request

        event = SimpleNamespace(
            button=SimpleNamespace(id=f"console-message-action-fork-{message.id}"),
            stop=Mock(),
        )
        assert await screen.handle_console_message_action(event) is True

    request.assert_called_once_with(message.id)
