"""Original voice bodies with current CSS and explicit geometry prerequisites."""

from contextlib import asynccontextmanager

import pytest
from textual.widgets import Static

from Tests.UI import test_console_voice_chip as original

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class VisibleChipApp(original.ProductionCssComposerApp):
    """Give ordinary chip states room beside current composer furniture."""

    @asynccontextmanager
    async def run_test(self, *, size=(120, 24), **kwargs):
        async with super().run_test(size=size, **kwargs) as pilot:
            composer = self.query_one(original.ConsoleComposerBar)
            assert composer._voice_chip_width_cap() >= len("No microphone access.") + 2
            yield pilot


@pytest.mark.parametrize(
    "case",
    [
        "test_idle_collapses_a_chip_that_was_showing",
        "test_preparing_and_error_states_render_their_message",
    ],
)
async def test_original_voice_chip_body(monkeypatch, case):
    monkeypatch.setattr(original, "ComposerApp", VisibleChipApp)
    await getattr(original, case)()


@pytest.mark.parametrize("state", [original.STATE_LISTENING, original.STATE_ERROR])
async def test_original_narrow_composer_keeps_draft_space_and_hides_chip(state):
    app = original.ProductionCssComposerApp()
    async with app.run_test(size=(80, 24)) as pilot:
        composer = app.query_one(original.ConsoleComposerBar)
        composer.load_draft("draft remains visible")
        await pilot.pause()
        draft = composer.query_one("#console-command-visible-text", Static)
        before = (draft.region.width, draft.render_line(0).text)
        assert before[0] > 0 and "draft remains" in before[1]
        assert composer._voice_chip_width_cap() == 0
        composer.set_voice_status(
            state, partial="hello", message="No microphone access."
        )
        await pilot.pause()
        chip = composer.query_one("#console-voice-status", Static)
        assert chip.styles.width.value == 0
        assert not chip.display and original._painted(chip) == ""
        assert (draft.region.width, draft.render_line(0).text) == before
