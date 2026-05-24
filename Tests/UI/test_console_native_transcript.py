import pytest
from textual.app import App, ComposeResult

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleSaveDestination
from tldw_chatbook.Widgets.Console.console_save_as_modal import ConsoleSaveAsModal
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class TranscriptHarness(App):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello", id="m1"),
                ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m2"),
            ]
        )
        yield transcript


class SaveAsModalHarness(App):
    def on_mount(self) -> None:
        self.push_screen(
            ConsoleSaveAsModal(destinations=save_as_modal_destinations())
        )


def save_as_modal_destinations() -> list[ConsoleSaveDestination]:
    return [
        ConsoleSaveDestination(label="Chatbook", available=True, reason=""),
        ConsoleSaveDestination(
            label="Note",
            available=False,
            reason="WIP: save as Note is not wired yet.",
        ),
    ]


def test_console_transcript_renderable_uses_full_width_rules():
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
            ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="world"),
        ]
    )

    plain = transcript.to_plain_text(width=40)

    assert "─" * 40 in plain
    assert "User" in plain
    assert "Assistant" in plain
    assert "hello" in plain
    assert "world" in plain


def test_console_transcript_widget_rules_are_long_enough_to_clip_full_width():
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
        ]
    )

    first_rule = transcript._message_widgets()[0]
    renderable = getattr(first_rule, "renderable", "")

    assert len(str(renderable)) >= 160


def test_console_transcript_selected_message_shows_action_row():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1")
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    plain = transcript.to_plain_text(width=80)

    assert "Copy | Edit | Save as..." in plain
    assert "--->" in plain


def test_console_transcript_variant_navigation_changes_displayed_content():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    assert "first" in transcript.to_plain_text(width=80)

    transcript.select_next_variant("m1")

    rendered = transcript.to_plain_text(width=80)
    assert "second" in rendered
    assert "first" not in rendered
    assert "<" in rendered
    assert ">" in rendered


@pytest.mark.asyncio
async def test_console_transcript_keyboard_selects_messages_and_enter_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        text = _visible_text(app)

    assert "Copy | Edit | Save as..." in text


@pytest.mark.asyncio
async def test_console_transcript_click_selects_message_and_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        text = _visible_text(app)

    assert "Copy | Edit | Save as..." in text


@pytest.mark.asyncio
async def test_console_selected_message_has_visible_terminal_frame():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-m2")
        selected = app.query_one("#console-message-m2")

    assert "console-transcript-message-selected" in selected.classes
    assert selected.styles.border.top[0] == "solid"
    assert selected.styles.border.right[0] == "solid"
    assert selected.styles.border.bottom[0] == "solid"
    assert selected.styles.border.left[0] == "solid"


@pytest.mark.asyncio
async def test_console_transcript_action_buttons_have_stable_ids():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-copy-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-save-as-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-regenerate-m2")
        text = _visible_text(app)

    assert "Copy" in text
    assert "Save as..." in text
    assert "♻" in text


@pytest.mark.asyncio
async def test_console_transcript_action_tooltips_explain_power_user_symbols():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-continue-m2")
        continue_action = app.query_one("#console-message-action-continue-m2")
        save_action = app.query_one("#console-message-action-save-as-m2")

    assert "extend" in str(continue_action.tooltip).lower()
    assert "destination" in str(save_action.tooltip).lower()


@pytest.mark.asyncio
async def test_console_transcript_escape_collapses_selected_action_row():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        assert "Save as..." in _visible_text(app)

        await pilot.press("escape")

        assert "Save as..." not in _visible_text(app)


@pytest.mark.asyncio
async def test_save_as_modal_lists_available_and_wip_destinations():
    app = SaveAsModalHarness()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)
        await _wait_for_selector(app.screen, pilot, "#console-save-as-modal")
        text = _visible_text(app.screen)

    assert "Chatbook" in text
    assert "Note" in text
    assert "WIP: save as Note is not wired yet." in text


@pytest.mark.asyncio
async def test_console_mounts_native_transcript_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        assert console.query_one("#console-native-transcript", ConsoleTranscript)


@pytest.mark.asyncio
async def test_console_tab_reaches_major_console_screen_regions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-tray")
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-run-inspector")

        seen_focus_ids = set()
        for _ in range(24):
            focused = getattr(console.app, "focused", None)
            if focused is not None and getattr(focused, "id", None):
                seen_focus_ids.add(focused.id)
            await pilot.press("tab")

    assert "console-native-transcript" in seen_focus_ids
    assert "console-native-composer" in seen_focus_ids
