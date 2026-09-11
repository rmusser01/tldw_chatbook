"""Screen wiring for editing displayable thinking block text (TASK-32312).

The thinking disclosure row's Edit action must open the block-scoped
``ConsoleEditThinkingModal`` (not the message edit modal, and not the dead
store lookup that treated the display-only activity id as a tree node), and
saving must route through ``store.update_message_thinking_block`` so the
answer content and the generation envelope stay one owner group.
"""

from types import SimpleNamespace

import pytest
from textual.widgets import TextArea
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditThinkingModal,
)


async def _wait_until(predicate, pilot, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError("Condition never became true.")


def _reasoning_envelope(text: str) -> ThinkingEnvelope:
    return ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                block_id="reasoning-1",
                round_ordinal=0,
                provider="llama_cpp",
                model="test-model",
                protocol="chat_completions",
                source_format="think_tag",
                status="complete",
                text=text,
            ),
        )
    )


@pytest.mark.asyncio
async def test_thinking_row_edit_opens_modal_and_saves_block_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="the answer"
        )
        store.replace_message_thinking(
            assistant.id, _reasoning_envelope("original reasoning")
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        activity_id = next(iter(transcript._thinking_activity_refs))
        transcript.select_message(activity_id)
        await console._sync_native_console_chat_ui()

        # Thinking disclosures carry no action buttons (copy-style keyboard
        # seam): the `e` shortcut routes through the transcript action and
        # bubbles ConsoleThinkingEditRequested to the screen.
        transcript.action_invoke_selected_action("edit")
        await _wait_until(
            lambda: any(
                isinstance(s, ConsoleEditThinkingModal)
                for s in console.app.screen_stack
            ),
            pilot,
        )
        modal_screen = next(
            s
            for s in reversed(console.app.screen_stack)
            if isinstance(s, ConsoleEditThinkingModal)
        )
        await pilot.pause()
        editor = modal_screen.query_one("#console-edit-thinking-body", TextArea)
        assert editor.text == "original reasoning"

        editor.text = "cleaned reasoning"
        await pilot.pause()
        await pilot.click("#console-edit-thinking-save")
        await pilot.pause()
        await _wait_until(
            lambda: (
                store.get_message(assistant.id).thinking is not None
                and store.get_message(assistant.id).thinking.blocks[0].text
                == "cleaned reasoning"
            ),
            pilot,
            attempts=150,
        )

        edited = store.get_message(assistant.id)
        assert edited.content == "the answer"
        assert edited.thinking is not None
        assert edited.thinking.blocks[0].block_id == "reasoning-1"


@pytest.mark.asyncio
async def test_thinking_only_edit_changes_transcript_fingerprint():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="the answer"
        )
        store.replace_message_thinking(
            assistant.id, _reasoning_envelope("original reasoning")
        )

        before = store.get_message(assistant.id)
        fingerprint_before = console._native_console_transcript_fingerprint([before])

        store.update_message_thinking_block(
            assistant.id, "reasoning-1", "cleaned reasoning"
        )
        after = store.get_message(assistant.id)

        assert (
            console._native_console_transcript_fingerprint([after])
            != fingerprint_before
        )


@pytest.mark.asyncio
async def test_saved_thinking_edit_repaints_mounted_disclosure_row():
    """End-to-end repaint check (Qodo #3): after saving a block-text edit,
    the mounted disclosure row's rendered text must show the edited text."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="the answer"
        )
        store.replace_message_thinking(
            assistant.id, _reasoning_envelope("original reasoning")
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        activity_id = next(iter(transcript._thinking_activity_refs))
        assert transcript.thinking_detail_text(activity_id) == "original reasoning"

        # The production path: modal -> save -> controller schedules the
        # same console-sync worker the screen uses after an edit.
        await console._message._open_console_thinking_edit_modal(
            owner_message_id=assistant.id,
            block_id="reasoning-1",
            text="original reasoning",
        )
        await _wait_until(
            lambda: any(
                isinstance(s, ConsoleEditThinkingModal)
                for s in console.app.screen_stack
            ),
            pilot,
        )
        modal = next(
            s
            for s in reversed(console.app.screen_stack)
            if isinstance(s, ConsoleEditThinkingModal)
        )
        from textual.widgets import TextArea

        editor = modal.query_one("#console-edit-thinking-body", TextArea)
        editor.text = "cleaned reasoning"
        await pilot.pause()
        await pilot.click("#console-edit-thinking-save")
        await pilot.pause()

        await _wait_until(
            lambda: transcript.thinking_detail_text(activity_id)
            == "cleaned reasoning",
            pilot,
        )
        # The MOUNTED row's visible text (not just the display model):
        from tldw_chatbook.Widgets.Console.console_assistant_turn import (
            ConsoleActivityDisclosure,
        )

        disclosure = transcript.query_one(
            f"#console-activity-disclosure-{activity_id}",
            ConsoleActivityDisclosure,
        )
        disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        await _wait_until(
            lambda: any(
                "cleaned reasoning" in (w.renderable if isinstance(w.renderable, str) else str(getattr(w.renderable, "plain", "")))
                for w in disclosure.detail_stack.children
            ),
            pilot,
        )
