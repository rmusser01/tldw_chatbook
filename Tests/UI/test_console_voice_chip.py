"""Console composer voice-status chip tests."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Chat.console_voice_input import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_PREPARING,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


class ComposerApp(App):
    def compose(self):
        yield ConsoleComposerBar(id="console-native-composer")


def _visible(widget) -> bool:
    """True only when the widget and every ancestor are displayed.

    `renderable` having text proves nothing: #console-composer-status carries
    `console-hidden-control` (display: none) and would happily hold a string
    no user can see.
    """
    node = widget
    while node is not None:
        if not getattr(node, "display", True):
            return False
        node = node.parent
    return True


@pytest.mark.asyncio
async def test_idle_collapses_a_chip_that_was_showing():
    """Show it first: asserting width==0 on a never-shown chip proves nothing.

    The chip starts at width 0 from `compose()`, so a bare idle assertion
    would pass even if `set_voice_status` were a no-op.
    """
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_LISTENING, partial="hello", elapsed_seconds=1)
        chip = composer.query_one("#console-voice-status", Static)
        assert chip.styles.width.value > 0

        composer.set_voice_status(STATE_IDLE)

        assert chip.styles.width.value == 0
        assert str(chip.renderable) == ""


@pytest.mark.asyncio
async def test_chip_is_actually_visible_while_listening():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="and compare them to", elapsed_seconds=7
        )
        chip = composer.query_one("#console-voice-status", Static)
        assert _visible(chip)
        assert chip.styles.width.value > 0
        assert "0:07" in str(chip.renderable)


@pytest.mark.asyncio
async def test_whisper_bracket_tokens_render_literally():
    """[BLANK_AUDIO] is routine Whisper output and is not Rich markup."""
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_LISTENING, partial="[BLANK_AUDIO] [Music]")
        chip = composer.query_one("#console-voice-status", Static)
        assert "[BLANK_AUDIO]" in str(chip.renderable)


@pytest.mark.asyncio
async def test_narrow_terminal_drops_the_partial_not_the_draft():
    app = ComposerApp()
    async with app.run_test(size=(30, 12)):
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="a very long partial transcript", elapsed_seconds=3
        )
        chip = composer.query_one("#console-voice-status", Static)
        assert "very long partial" not in str(chip.renderable)
        assert "●" in str(chip.renderable)


@pytest.mark.asyncio
async def test_preparing_and_error_states_render_their_message():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_PREPARING, message="Loading model…")
        chip = composer.query_one("#console-voice-status", Static)
        assert "Loading model…" in str(chip.renderable)

        composer.set_voice_status(STATE_ERROR, message="No microphone access.")
        assert "No microphone access." in str(chip.renderable)


@pytest.mark.asyncio
async def test_there_is_exactly_one_microphone_button():
    """The composer ships #console-dictation; this feature must not add a second."""
    from textual.widgets import Button

    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        mic_like = [
            button
            for button in composer.query(Button)
            if "mic" in str(button.label).lower() or "dictat" in (button.id or "")
        ]
        assert len(mic_like) == 1
        assert mic_like[0].id == "console-dictation"
        assert not composer.query("#console-voice-toggle")


@pytest.mark.asyncio
async def test_chip_mirrors_the_shipping_dictation_states():
    """The chip must track the button's real four-state lifecycle.

    ``ConsoleComposerBar.sync_dictation_state`` is the existing driver behind
    ``#console-dictation`` (see ``Tests/UI/test_console_dictation.py``); this
    asserts the voice-status chip mirrors it end to end.
    """
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        chip = composer.query_one("#console-voice-status", Static)

        composer.sync_dictation_state("starting")
        assert _visible(chip)
        assert "Preparing" in str(chip.renderable)

        composer.sync_dictation_state("recording")
        assert "●" in str(chip.renderable)

        composer.sync_dictation_state("transcribing")
        assert "Transcribing" in str(chip.renderable)

        composer.sync_dictation_state("idle")
        assert chip.styles.width.value == 0
