"""Console composer voice-status chip tests."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Static

from Tests.UI.consolidated_css import APP_STYLESHEETS
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


class ProductionCssComposerApp(ComposerApp):
    """Mount the composer with the generated production stylesheet."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


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
        assert "◉" in _painted(chip)


@pytest.mark.asyncio
async def test_the_transcribing_label_truncates_from_the_right_not_the_left():
    """Review L3: at a composer width that gives the label a room of 9..14
    cells, a right-truncating `[-room:]` slice (correct for `partial`, wrong
    for a fixed constant) painted "scribing…" -- the label's own trailing
    ellipsis surviving while its meaningful prefix was cut. The fix keeps
    the START of the label and puts the ellipsis at the cut, so "Transcr"
    (the readable, identifying part) must survive, never "scribing".

    Terminal width 44 was probed to land the composer's own `room` at 11
    cells -- squarely inside the buggy 9..14 window (composer widths
    ~42-47 columns per the review).
    """
    app = ComposerApp()
    async with app.run_test(size=(44, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, elapsed_seconds=7, segment_transcribing=True
        )
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        painted = _painted(chip)
        assert "◉" in painted
        assert "Transcr" in painted, f"expected the label's start to survive, got {painted!r}"
        assert "scribing" not in painted, (
            f"the label was still truncated from the left, got {painted!r}"
        )


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
async def test_busy_dictation_copy_uses_the_existing_chip_and_clears_at_idle():
    app = ProductionCssComposerApp()
    async with app.run_test(size=(80, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        message = "Local transcription busy — dictation will run next."
        presentation = [
            composer.query_one(selector)
            for selector in (
                "#console-composer-collapse",
                "#console-composer-menu",
                "#console-command-visible-text",
                "#console-send-disabled-reason",
            )
        ]
        actions = composer.query_one("#console-composer-actions")
        mic = composer.query_one("#console-dictation")
        idle_action_width = actions.region.width
        idle_mic_width = mic.region.width
        assert all(_visible(widget) for widget in presentation)

        composer.sync_dictation_state("starting")
        composer.set_voice_preparing_message(message)
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)

        assert message in _painted(chip)
        assert _visible(chip)
        assert all(not _visible(widget) for widget in presentation)
        assert idle_action_width == 25
        assert actions.region.width == idle_action_width
        assert mic.region.width == idle_mic_width
        assert actions.region.right <= composer.region.right <= app.size.width
        assert mic.region.right <= composer.region.right
        assert _visible(mic)
        assert await pilot.click(mic) is True

        # An ordinary control-bar refresh must not erase the busy status.
        composer.sync_dictation_state("starting")
        await pilot.pause()
        assert message in _painted(chip)

        composer.sync_dictation_state("idle")
        await pilot.pause()
        assert chip.styles.width.value == 0
        assert _painted(chip) == ""
        assert all(_visible(widget) for widget in presentation)


@pytest.mark.asyncio
async def test_busy_presentation_preserves_draft_and_caret_then_restores_recording():
    app = ProductionCssComposerApp()
    async with app.run_test(size=(80, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.load_draft("keep this draft")
        for _ in range(5):
            composer.move_cursor_left()
        before_text = composer.draft_text()
        before_caret = composer.cursor_index
        controls = [
            composer.query_one(selector)
            for selector in (
                "#console-composer-collapse",
                "#console-composer-menu",
                "#console-command-visible-text",
            )
        ]
        reason = composer.query_one("#console-send-disabled-reason")

        composer.sync_dictation_state("starting")
        composer.set_voice_preparing_message(
            "Local transcription busy — dictation will run next."
        )
        await pilot.pause()

        assert all(not _visible(widget) for widget in (*controls, reason))
        assert composer.draft_text() == before_text
        assert composer.cursor_index == before_caret

        composer.sync_dictation_state("recording")
        await pilot.pause()

        assert all(_visible(widget) for widget in controls)
        assert not _visible(reason)
        assert composer.draft_text() == before_text
        assert composer.cursor_index == before_caret


@pytest.mark.asyncio
async def test_busy_presentation_suppresses_and_restores_staged_attachment():
    app = ProductionCssComposerApp()
    async with app.run_test(size=(80, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        indicator = composer.query_one("#console-attachment-indicator", Static)
        clear_button = composer.query_one("#console-clear-attachment")
        actions = composer.query_one("#console-composer-actions")

        composer.set_pending_attachment_label("2 files", count=2, total=5)
        await pilot.pause()
        assert "2 files" in _painted(indicator)
        assert _visible(indicator)
        assert _visible(clear_button)
        assert str(clear_button.tooltip) == "Remove all 2 pending attachments."
        assert actions.region.width == 29

        composer.set_voice_status(
            STATE_PREPARING,
            message="Local transcription busy — dictation will run next.",
        )
        # A production control-bar refresh may reapply this setter while the
        # deferred capture still owns the full-width preparing presentation.
        composer.set_pending_attachment_label("2 files", count=2, total=5)
        await pilot.pause()

        assert not _visible(indicator)
        assert not _visible(clear_button)
        assert actions.region.width == 25
        assert "2 files" in str(indicator.renderable)
        assert str(clear_button.tooltip) == "Remove all 2 pending attachments."

        expanded = composer.query_one("#console-composer-expanded")
        collapsed = composer.query_one("#console-composer-collapsed")
        composer.set_collapsed(True)
        composer.set_pending_attachment_label("2 files", count=2, total=5)
        assert not expanded.display
        assert collapsed.display
        composer.set_collapsed(False)

        composer.set_voice_status(STATE_LISTENING, elapsed_seconds=0)
        await pilot.pause()

        assert "2 files" in _painted(indicator)
        assert _visible(indicator)
        assert _visible(clear_button)
        assert str(clear_button.tooltip) == "Remove all 2 pending attachments."
        assert actions.region.width == 29

        composer.set_pending_attachment_label(None)
        await pilot.pause()
        assert not _visible(indicator)
        assert not _visible(clear_button)
        assert actions.region.width == 25


@pytest.mark.asyncio
async def test_production_css_busy_status_stays_meaningful_at_narrow_width():
    app = ProductionCssComposerApp()
    async with app.run_test(size=(48, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)

        composer.sync_dictation_state("starting")
        composer.set_voice_preparing_message(
            "Local transcription busy — dictation will run next."
        )
        await pilot.pause()

        chip = composer.query_one("#console-voice-status", Static)
        painted = _painted(chip)
        assert painted.startswith("Local trans")
        assert painted.endswith("…")
        assert _visible(chip)
        assert chip.region.x + chip.region.width <= composer.region.right


@pytest.mark.asyncio
async def test_production_css_ordinary_voice_states_restore_normal_padding():
    app = ProductionCssComposerApp()
    async with app.run_test(size=(80, 12)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        chip = composer.query_one("#console-voice-status", Static)

        composer.set_voice_status(
            STATE_PREPARING,
            message="Local transcription busy — dictation will run next.",
        )
        composer.set_voice_status(STATE_PREPARING, message="Loading model…")
        await pilot.pause()
        assert "Loading model…" in _painted(chip)
        assert chip.styles.padding.left == 1
        assert chip.styles.padding.right == 1
        assert chip.styles.margin.left == 0
        assert chip.styles.margin.right == 0

        composer.set_voice_status(STATE_ERROR, message="No microphone access.")
        await pilot.pause()
        assert "No microphone access." in _painted(chip)
        assert chip.styles.padding.left == 1
        assert chip.styles.padding.right == 1
        assert chip.styles.margin.left == 0
        assert chip.styles.margin.right == 0

        composer.set_voice_status(
            STATE_LISTENING,
            partial="ordinary listening partial",
            elapsed_seconds=7,
        )
        await pilot.pause()
        assert "0:07" in _painted(chip)
        assert "ordinary listening partial" in _painted(chip)
        assert chip.styles.padding.left == 1
        assert chip.styles.padding.right == 1
        assert chip.styles.margin.left == 0
        assert chip.styles.margin.right == 0


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
        assert "◉" in _painted(chip)

        composer.sync_dictation_state("transcribing")
        await pilot.pause()
        assert "Transcribing" in _painted(chip)

        composer.sync_dictation_state("idle")
        await pilot.pause()
        assert chip.styles.width.value == 0
        assert _painted(chip) == ""
