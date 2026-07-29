"""Committing the winner of a comparison.

The Playground exists to identify which option works best. Comparison
without a way to keep the result is half a tool -- the user finds the voice
they want and then has to reproduce their choice by hand in Settings.

Session-scoped overrides stay session-scoped: this is the ONE explicit path
by which the Playground writes a persisted default, and it happens because
the user asked for it. The spec's direction rule is unchanged -- Settings
never reads Playground state.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from Tests.UI.test_stts_playground_audio_cpp import (
    FakeTTSService,
    _resolved,
    _wait_until,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from textual.widgets import Select


@pytest.fixture
def faked_service(monkeypatch):
    """A loaded catalog, so there is something to commit."""
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane, "_check_higgs_installation", lambda self: None
    )
    return service


class _CapturingHost(App[None]):
    def __init__(self):
        super().__init__()
        self.saved: list[STTSSettingsSaveEvent] = []
        self.notices: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(id="speech-playground-pane")

    def post_message(self, message):
        if isinstance(message, STTSSettingsSaveEvent):
            self.saved.append(message)
            return True
        return super().post_message(message)

    def notify(self, message, *, severity="information", **kwargs):
        self.notices.append((message, severity))


@pytest.mark.asyncio
async def test_the_action_exists_and_is_reachable():
    """A comparison tool with no way to commit its result is unfinished."""
    app = _CapturingHost()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        button = app.query_one("#tts-save-default-btn")
        assert button.allow_focus()
        pane = app.query_one("#speech-playground-pane")
        assert pane.region.contains_region(button.region)


@pytest.mark.asyncio
async def test_saving_the_default_posts_the_axes_as_preferences(faked_service):
    """What it commits is what the axes show, not what Settings holds."""
    app = _CapturingHost()
    async with app.run_test(size=(200, 60)) as pilot:
        select = app.query_one("#tts-provider-select", Select)
        await _wait_until(
            pilot, lambda: any(isinstance(v, str) for _l, v in select._options)
        )
        app.query_one("#tts-save-default-btn").press()
        await pilot.pause()

    assert app.saved, "Save as default posted nothing"
    preferences = app.saved[-1].preferences
    assert preferences is not None
    assert preferences.provider_id
    assert preferences.response_format


@pytest.mark.asyncio
async def test_it_refuses_rather_than_writing_a_half_chosen_default():
    """Before a catalog loads there is no provider to commit. Writing one
    anyway would persist a sentinel as if it were a choice."""
    app = _CapturingHost()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        app.query_one("#tts-save-default-btn").press()
        await pilot.pause()

    assert not app.saved, "wrote a default with no provider chosen"
    assert app.notices, "refused silently"
