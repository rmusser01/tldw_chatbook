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


def _painted(widget) -> str:
    """Return the text the widget actually paints on its first (only) row.

    Not `str(widget.renderable)`: that is the raw value handed to `update()`,
    *before* Textual's markup parser has had a go at it. Asserting on it let
    `test_whisper_bracket_tokens_render_literally` pass while the terminal
    showed "● 0:03    hi" -- the uppercase Whisper tokens were parsed as markup
    tags and dropped. Mirrors the helper in `test_console_dictation_firstrun.py`.

    The painted line is also truncated to the chip's width (42 cells at most),
    so anything asserted through here has to be short enough to fit.
    """
    return widget.render_line(0).text.rstrip()


@pytest.mark.asyncio
async def test_idle_collapses_a_chip_that_was_showing():
    """Show it first: asserting width==0 on a never-shown chip proves nothing.

    The chip starts at width 0 from `compose()`, so a bare idle assertion
    would pass even if `set_voice_status` were a no-op.
    """
    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_LISTENING, partial="hello", elapsed_seconds=1)
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        assert chip.styles.width.value > 0
        assert _painted(chip) != ""

        composer.set_voice_status(STATE_IDLE)
        await pilot.pause()

        assert chip.styles.width.value == 0
        assert _painted(chip) == ""


@pytest.mark.asyncio
async def test_chip_is_actually_visible_while_listening():
    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="and compare them to", elapsed_seconds=7
        )
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        assert _visible(chip)
        assert chip.styles.width.value > 0
        assert "0:07" in _painted(chip)
        assert "and compare them to" in _painted(chip)


@pytest.mark.asyncio
async def test_whisper_bracket_tokens_render_literally():
    """[BLANK_AUDIO] is routine Whisper output and is not Rich markup.

    Asserted on the painted line, not `renderable`: `rich.markup.escape` only
    escapes tags opening with `[a-z#/@]`, so these uppercase tokens went through
    it unchanged and were then stripped by Textual's markup parser at paint
    time. The old `renderable` assertion could never have caught that -- it read
    the string *before* parsing, and passed with no fix in place at all.
    """
    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="[BLANK_AUDIO] [Music] hi", elapsed_seconds=3
        )
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        painted = _painted(chip)
        assert "[BLANK_AUDIO]" in painted
        assert "[Music]" in painted
        assert painted.endswith("hi")


@pytest.mark.asyncio
async def test_a_bracketed_path_in_a_partial_does_not_blank_the_chip():
    """The other half: `[/tmp/x]` is a `MarkupError`, not a swallowed tag.

    Escaping used to be what stopped this crashing the paint; whatever replaced
    it has to cover this case too, or the fix for the tokens above reintroduces
    a chip that raises instead of rendering.
    """
    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="[/tmp/x] ok", elapsed_seconds=3
        )
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        painted = _painted(chip)
        assert "[/tmp/x]" in painted
        assert painted.endswith("ok")


@pytest.mark.asyncio
async def test_narrow_terminal_drops_the_partial_not_the_draft():
    app = ComposerApp()
    async with app.run_test(size=(30, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="a very long partial transcript", elapsed_seconds=3
        )
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        assert "very long partial" not in _painted(chip)
        assert "●" in _painted(chip)


@pytest.mark.asyncio
async def test_preparing_and_error_states_render_their_message():
    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_PREPARING, message="Loading model…")
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        assert "Loading model…" in _painted(chip)

        composer.set_voice_status(STATE_ERROR, message="No microphone access.")
        await pilot.pause()
        assert "No microphone access." in _painted(chip)


@pytest.mark.asyncio
async def test_the_mic_tooltips_claim_nothing_the_backend_no_longer_does():
    """The button's own copy outlived the backend it described.

    It promised "Record one English utterance with local Parakeet v2." and
    "Transcribing locally with Parakeet v2 INT8…". Dictation is now streaming
    (many accumulated segments, not one utterance), runs in whatever
    `transcription.default_language` says, and picks its provider through
    `console_voice_input.resolve()` -- faster-whisper on the machine this was
    verified on. Every one of those three claims was false.
    """
    from textual.widgets import Button

    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        mic = composer.query_one("#console-dictation", Button)

        # A static string cannot know any of these, so it must not name them.
        forbidden = ("parakeet", "whisper", "english", "onnx", "int8", "utterance")
        for state in ("idle", "starting", "recording", "transcribing"):
            composer.sync_dictation_state(state)
            await pilot.pause()
            tooltip = str(mic.tooltip)
            lowered = tooltip.lower()
            assert tooltip, f"{state} has no tooltip"
            assert len(tooltip) <= 90, f"{state} tooltip is too long to hover: {tooltip}"
            for word in forbidden:
                assert word not in lowered, f"{state} tooltip still claims {word!r}"

        # The one word the shipped contract test pins (test_console_dictation.py).
        composer.sync_dictation_state("recording")
        await pilot.pause()
        assert "Stop" in str(mic.tooltip)

        composer.sync_dictation_state("idle")
        await pilot.pause()
        assert str(mic.tooltip) == ConsoleComposerBar.DICTATION_IDLE_TOOLTIP


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
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        chip = composer.query_one("#console-voice-status", Static)

        composer.sync_dictation_state("starting")
        await pilot.pause()
        assert _visible(chip)
        assert "Preparing" in _painted(chip)

        composer.sync_dictation_state("recording")
        await pilot.pause()
        assert "●" in _painted(chip)

        composer.sync_dictation_state("transcribing")
        await pilot.pause()
        assert "Transcribing" in _painted(chip)

        composer.sync_dictation_state("idle")
        await pilot.pause()
        assert chip.styles.width.value == 0
        assert _painted(chip) == ""
