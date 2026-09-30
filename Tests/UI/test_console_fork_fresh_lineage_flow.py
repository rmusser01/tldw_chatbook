"""TASK-33621.10: Fork opens its dialog on freshly sent turns in the real Console.

Drives the mounted ``ChatScreen`` (``ConsoleHarness``) with its real store,
controller and ``ChatPersistenceService`` -- only the provider gateway is a
double. Each send takes the durable-commit path every saved chat uses, and no
test reloads the conversation, so the live tree is exactly what a user who just
sent a message is looking at.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    ConsoleHarness,
    _build_console_send_test_app,
    _select_llamacpp_console,
    _visible_text,
    _wait_for_selector,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_fork_chat_modal import ConsoleForkChatModal
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

_REPLY = "hello there friend"


def _host(*, fail_first_call: bool = False) -> ConsoleHarness:
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    calls = 0

    class _Gateway(CapturingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            nonlocal calls
            calls += 1
            if fail_first_call and calls == 1:
                raise RuntimeError("provider dropped the request")
            async for chunk in super().stream_chat(resolution, messages, **kwargs):
                yield chunk

    app.console_provider_gateway_factory = lambda: _Gateway(chunks=(_REPLY,))
    return ConsoleHarness(app)


def _active(console) -> list[ConsoleChatMessage]:
    store = console._ensure_console_chat_store()
    return [
        store.get_message(native_id)
        for native_id in store.active_path_message_ids(store.active_session_id)
    ]


def _last(console, role: ConsoleMessageRole) -> ConsoleChatMessage:
    return [message for message in _active(console) if message.role is role][-1]


async def _submit(console, pilot, draft: str) -> None:
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft(draft)
    console.query_one("#console-send-message", Button).press()
    await pilot.pause()


async def _send(console, pilot, draft: str) -> None:
    replies_before = sum(
        message.role is ConsoleMessageRole.ASSISTANT for message in _active(console)
    )
    await _submit(console, pilot, draft)
    for _ in range(200):
        replies = [
            message
            for message in _active(console)
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        if (
            len(replies) > replies_before
            and replies[-1].status == "complete"
            and replies[-1].persisted_message_id is not None
            and console._ensure_console_chat_store().dispatch_recovery_for_session(
                console._ensure_console_chat_store().active_session_id
            )
            is None
        ):
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Reply never settled. Visible: {_visible_text(console)!r}")


async def _select(console, pilot, message: ConsoleChatMessage) -> ConsoleTranscript:
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.select_message(message.id)
    await console._sync_native_console_chat_ui()
    await _wait_for_selector(
        console, pilot, f"#console-message-action-fork-{message.id}"
    )
    await pilot.pause()
    return transcript


def _guide(console) -> str:
    return " ".join(_visible_text(console).split())


async def _press_f_opens_fork_dialog(host, console, pilot, message) -> None:
    transcript = await _select(console, pilot, message)
    fork = console.query_one(f"#console-message-action-fork-{message.id}", Button)
    assert not fork.disabled, fork.tooltip
    guide = _guide(console)
    assert "f Fork" in guide
    assert "Fork unavailable" not in guide
    notices: list[str] = []
    original_notify = console.app_instance.notify

    def capture(message_text, *args, **kwargs):
        notices.append(str(message_text))
        return original_notify(message_text, *args, **kwargs)

    console.app_instance.notify = capture
    try:
        transcript.focus()
        await pilot.press("f")
        for _ in range(80):
            if isinstance(host.screen_stack[-1], ConsoleForkChatModal):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(f"Fork dialog did not open; notices={notices!r}")
    finally:
        console.app_instance.notify = original_notify
    await pilot.press("escape")
    for _ in range(80):
        if not isinstance(host.screen_stack[-1], ConsoleForkChatModal):
            break
        await pilot.pause(0.05)
    else:
        raise AssertionError("Fork dialog did not close on Escape")


@pytest.mark.asyncio
async def test_fresh_saved_pair_opens_the_fork_dialog_from_both_rows():
    host = _host()

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        await _send(console, pilot, "Reply with exactly: hello there friend")
        store = console._ensure_console_chat_store()
        session = next(
            item for item in store.sessions() if item.id == store.active_session_id
        )
        assert session.ephemeral is False
        assert session.persisted_conversation_id is not None

        await _press_f_opens_fork_dialog(
            host, console, pilot, _last(console, ConsoleMessageRole.ASSISTANT)
        )
        await _press_f_opens_fork_dialog(
            host, console, pilot, _last(console, ConsoleMessageRole.USER)
        )


@pytest.mark.asyncio
async def test_help_then_send_then_fork_opens_the_fork_dialog():
    host = _host()

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        await _submit(console, pilot, "/help")
        for _ in range(80):
            if any(
                message.role is ConsoleMessageRole.SYSTEM
                for message in _active(console)
            ):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError("/help printed nothing")
        await _send(console, pilot, "Reply with exactly: hello there friend")
        # The /help output is still on the active path above the new turn.
        assert _active(console)[0].role is ConsoleMessageRole.SYSTEM

        await _press_f_opens_fork_dialog(
            host, console, pilot, _last(console, ConsoleMessageRole.ASSISTANT)
        )
        await _press_f_opens_fork_dialog(
            host, console, pilot, _last(console, ConsoleMessageRole.USER)
        )


@pytest.mark.asyncio
async def test_refused_fork_is_not_advertised_and_names_the_blocking_row():
    host = _host(fail_first_call=True)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()

        # The provider drops the first request: a saved, failed, EMPTY reply
        # stays in the chat's history. It cannot be copied, so it genuinely
        # blocks forking any later message.
        await _submit(console, pilot, "first question")
        for _ in range(200):
            active = _active(console)
            replies = [m for m in active if m.role is ConsoleMessageRole.ASSISTANT]
            # Settled: the failed reply plus the failure notice after it.
            if (
                replies
                and replies[-1].status == "failed"
                and active[-1].role is ConsoleMessageRole.SYSTEM
            ):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(
                f"No failed reply. Visible: {_visible_text(console)!r}"
            )
        # Let the failure's own post-run UI sync land before selecting.
        await pilot.pause(1.0)
        assert replies[-1].content == ""
        assert replies[-1].persisted_message_id is not None
        session_id = store.active_session_id
        # The later turn is saved through the store's ordinary persist path.
        # This `:memory:` harness blocks a SECOND provider send into an
        # existing conversation ("Trace capture blocked"; the log shows
        # `Console RAG capture unavailable; reason=capture_provider_failure`),
        # and the refusal under test depends only on the saved tree. The
        # store-level twin in Tests/Chat/test_console_fork_fresh_lineage.py
        # builds the same shape with two real sends.
        store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content="second question",
            persist=True,
        )
        later = store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="second answer",
            persist=True,
        )
        await console._sync_native_console_chat_ui()

        transcript = await _select(console, pilot, later)
        fork = console.query_one(f"#console-message-action-fork-{later.id}", Button)
        guide = _guide(console)

        assert fork.disabled
        assert "f Fork" not in guide
        assert "Fork unavailable" in guide
        assert "The failed Assistant reply above this message" in guide
        assert 'Fork from the User message "first question" instead.' in guide
        assert str(fork.tooltip) == store.fork_eligibility(later.id).reason

        notices: list[str] = []
        # A disabled action button's key repeats its reason via the
        # transcript's own ``notify`` (``_press_selected_action_button``).
        transcript.notify = lambda message_text, **_kwargs: notices.append(
            str(message_text)
        )
        transcript.focus()
        await pilot.press("f")
        await pilot.pause()
        assert not isinstance(host.screen_stack[-1], ConsoleForkChatModal)
        assert notices == [store.fork_eligibility(later.id).reason]
