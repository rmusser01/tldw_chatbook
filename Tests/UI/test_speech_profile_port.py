"""dev's profile library must survive the Playground rebuild.

While this rebuild was in flight, dev shipped a TTS profile library and
built it into `TTSPlaygroundWidget` -- the class the rebuild retires. Merging
the retirement without porting would have deleted the UI half of a feature
someone shipped over eight commits, and git would not have said a word: the
conflict resolves cleanly if you simply keep your own side.

So these assert the port, not the rebuild: the behaviour is dev's methods
verbatim, inherited rather than reimplemented, and the controls they query
are mounted under the ids they expect.
"""

from __future__ import annotations

import pathlib

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_profile_mixin import SpeechProfileMixin

#: The methods dev's profile feature added to the playground.
PORTED = (
    "_clear_profile_voice_validation",
    "_dismiss_profile_name_modal",
    "_end_profile_preset",
    "_prime_profile_preset_controls",
    "_profile_preview_blocked_presentation",
    "_project_profile_preset_controls",
    "_save_current_result_as_profile",
    "_sync_profile_preview_status",
    "_sync_save_profile_action",
)

#: The controls those methods query by id.
PROFILE_CONTROLS = ("tts-profile-preview-status", "audio-save-profile-btn")

_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = _BUNDLE

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(id="speech-playground-pane")

    def notify(self, *args, **kwargs):
        pass


@pytest.mark.unit
@pytest.mark.parametrize("name", PORTED)
def test_every_profile_method_survived(name):
    """A missing one is a piece of dev's feature deleted by the rebuild."""
    assert callable(getattr(SpeechPlaygroundPane, name, None)), (
        f"{name} was lost when TTSPlaygroundWidget was retired"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", PORTED)
def test_the_pane_inherits_rather_than_redefining(name):
    """Ported, not reimplemented. A pane-local copy would drift from dev's
    version the moment either changed."""
    assert getattr(SpeechPlaygroundPane, name) is getattr(SpeechProfileMixin, name)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", PROFILE_CONTROLS)
async def test_the_profile_controls_are_mounted(control):
    """The ported methods query these by id; absent, they raise NoMatches
    at the moment the user saves or previews a profile."""
    app = _Harness()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app.query(f"#{control}"), f"{control} is not mounted"


@pytest.mark.asyncio
async def test_the_profile_state_exists_before_any_action():
    """dev's methods read these without guards."""
    app = _Harness()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        for attribute in (
            "_profile_preset",
            "_profile_effective_availability",
            "_profile_preview_loading",
            "_profile_configuration_revision",
            "_profile_save_suppressed",
            "_active_profile_name_modal",
            "_profile_voice_validation_token",
        ):
            assert hasattr(pane, attribute), attribute
