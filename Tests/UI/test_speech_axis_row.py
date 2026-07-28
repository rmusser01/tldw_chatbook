"""The always-visible comparison axes, and whether an override is visible."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Speech.speech_axis_row import (
    AXIS_LABELS,
    SpeechAxisRow,
    axis_chip_id,
)
from tldw_chatbook.UI.Speech.speech_playground_model import AXIS_CONTROLS


class _Harness(App[None]):
    def __init__(self, values, defaults):
        super().__init__()
        self._values, self._defaults = values, defaults

    def compose(self) -> ComposeResult:
        yield SpeechAxisRow(values=self._values, defaults=self._defaults)


@pytest.mark.asyncio
async def test_an_overridden_axis_is_marked_and_a_matching_one_is_not():
    """A session override must be visible, not implied.

    The Playground owns session-scoped values that never write back. If an
    override looks identical to a saved default, the user cannot tell what
    they have changed -- and the whole point is comparing deliberate
    variations.
    """
    app = _Harness(
        values={"tts-voice-select": "Nova", "tts-format-select": "mp3"},
        defaults={"tts-voice-select": "Server default", "tts-format-select": "mp3"},
    )
    async with app.run_test(size=(120, 10)) as pilot:
        await pilot.pause()
        row = app.query_one(SpeechAxisRow)
        voice = app.query_one(f"#{axis_chip_id('tts-voice-select')}", Static)
        fmt = app.query_one(f"#{axis_chip_id('tts-format-select')}", Static)

        assert row.is_override("tts-voice-select") is True
        assert row.is_override("tts-format-select") is False
        assert voice.has_class("speech-chip-override")
        assert not fmt.has_class("speech-chip-override")


@pytest.mark.asyncio
async def test_the_override_is_not_signalled_by_colour_alone():
    """Colour is not available to every reader, and this app is keyboard-first.

    The marker must survive being read as plain text.
    """
    app = _Harness(
        values={"tts-voice-select": "Nova"},
        defaults={"tts-voice-select": "Server default"},
    )
    async with app.run_test(size=(120, 10)) as pilot:
        await pilot.pause()
        rendered = app.query_one(
            f"#{axis_chip_id('tts-voice-select')}", Static
        ).render_line(0).text
        assert "Nova" in rendered
        assert "*" in rendered, "override carried by colour only"


@pytest.mark.asyncio
async def test_every_axis_gets_a_chip_even_with_no_value():
    """A missing value must render as unset, not vanish.

    An axis that disappears when unset would change the row's shape as the
    user configures things, which is exactly the instability the redesign
    is removing.
    """
    app = _Harness(values={}, defaults={})
    async with app.run_test(size=(160, 10)) as pilot:
        await pilot.pause()
        for axis in AXIS_CONTROLS:
            chip = app.query_one(f"#{axis_chip_id(axis)}", Static)
            assert AXIS_LABELS[axis] in str(chip.renderable)


@pytest.mark.asyncio
async def test_an_axis_with_no_saved_default_is_not_an_override():
    """Unset is not overridden. Marking it would cry wolf on first run."""
    app = _Harness(values={"tts-voice-select": "Nova"}, defaults={})
    async with app.run_test(size=(120, 10)) as pilot:
        await pilot.pause()
        row = app.query_one(SpeechAxisRow)
        assert row.is_override("tts-voice-select") is False
