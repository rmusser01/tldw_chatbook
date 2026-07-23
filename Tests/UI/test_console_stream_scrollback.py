"""TASK-336: wheel scrollback must work while a reply is streaming.

The task-298 anchor keeps the transcript pinned to the tail during streams;
its contract says a user scroll releases the follow (never yanked). Live
evidence showed mouse-wheel scrollback inert mid-stream while keyboard
PageUp worked (UX review finding j4-wheel-scroll-locked-during-stream).
"""

import pytest
from textual import events

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_console_regenerate_feedback import GatedGateway
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript


def _wheel_up(transcript: ConsoleTranscript) -> None:
    """Deliver a mouse wheel-up the way the pointer path would."""
    transcript.post_message(
        events.MouseScrollUp(
            transcript, 4, 4, 0, -1, 0, False, False, False
        )
    )


@pytest.mark.asyncio
async def test_wheel_scrollback_detaches_follow_during_stream():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)

        # Overflow the transcript so there is real scrollback.
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        for n in range(30):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER if n % 2 else ConsoleMessageRole.ASSISTANT,
                content=f"history row {n} " + "x" * 40,
            )
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("stream please")
        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        assert transcript.max_scroll_y > 0, "transcript must overflow for this test"

        console.query_one("#console-send-message").press()
        # First chunk arrives; the gateway then holds the stream open.
        for _ in range(40):
            await pilot.pause(0.05)
            if any(
                m.content.startswith("first-chunk")
                for m in store.messages_for_session(store.active_session_id)
                if m.role is ConsoleMessageRole.ASSISTANT
            ):
                break
        # Anchored: pinned to the bottom while streaming.
        assert transcript.scroll_y == pytest.approx(transcript.max_scroll_y, abs=1)

        _wheel_up(transcript)
        await pilot.pause()
        await pilot.pause()
        scrolled_to = transcript.scroll_y
        assert scrolled_to < transcript.max_scroll_y, (
            "wheel-up during the stream must move the viewport"
        )

        # More content streams in — the viewport must NOT be yanked back.
        gateway.release.set()
        for _ in range(10):
            await pilot.pause(0.05)
        assert transcript.scroll_y < transcript.max_scroll_y, (
            "viewport was yanked back to the tail after wheel-up"
        )


@pytest.mark.asyncio
async def test_send_after_scrollback_still_jumps_to_tail():
    """The task-298 contract boundary: a send is a fresh follow intent —
    it must yank to the tail even from deep scrollback (only a scroll
    AFTER the send outranks it)."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        for n in range(30):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER if n % 2 else ConsoleMessageRole.ASSISTANT,
                content=f"history row {n} " + "x" * 40,
            )
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        _wheel_up(transcript)
        _wheel_up(transcript)
        await pilot.pause()
        assert transcript.scroll_y < transcript.max_scroll_y

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("send from scrollback")
        console.query_one("#console-send-message").press()
        gateway.release.set()
        for _ in range(20):
            await pilot.pause(0.05)
            if transcript.scroll_y == pytest.approx(transcript.max_scroll_y, abs=1):
                break

        assert transcript.scroll_y == pytest.approx(transcript.max_scroll_y, abs=1), (
            "a send from scrollback must re-anchor to the tail"
        )


@pytest.mark.asyncio
async def test_wheel_accepted_while_layout_transiently_collapsed():
    """TASK-336 live mechanism: during heavy row churn the arrangement can
    read max_scroll_y == 0 at the moment the wheel arrives; the gesture must
    still be accepted so release_anchor registers the reader's intent."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="short"
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        # Content fits: the base gate would read False here — the same
        # False the churn-collapsed layout produces mid-stream.
        assert transcript.max_scroll_y == 0
        assert transcript.allow_vertical_scroll is True

        transcript.anchor()
        assert transcript._anchor_released is False
        _wheel_up(transcript)
        await pilot.pause()
        assert transcript._anchor_released is True, (
            "wheel gesture must register (release the anchor) even on a "
            "collapsed layout"
        )
