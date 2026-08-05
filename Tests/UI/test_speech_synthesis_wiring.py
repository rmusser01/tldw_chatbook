"""Generate must reach the shared synthesis path.

The button existing and being above the fold is not the same as it working:
before `SpeechActionStrip`, it rendered perfectly and its handler could never
match its id. These tests press it and assert something downstream happened.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_synthesis_mixin import SpeechSynthesisMixin


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(provider="audio_cpp")


@pytest.mark.unit
def test_synthesis_comes_from_the_shared_mixin():
    """The 322-line generate path is inherited, not copied into the pane.

    It lived in the legacy widget until that was retired; keeping it in the
    mixin is what let the pane adopt it whole rather than reimplementing it,
    and is what the remaining Speech surfaces will inherit in turn."""
    assert issubclass(SpeechPlaygroundPane, SpeechSynthesisMixin)
    assert (
        SpeechPlaygroundPane._generate_tts is SpeechSynthesisMixin._generate_tts
    ), "the pane redefined generate instead of inheriting it"


@pytest.mark.asyncio
async def test_pressing_generate_invokes_the_synthesis_path(monkeypatch):
    """The wiring this whole phase turns on.

    Asserts the press reaches `_generate_tts` -- not that synthesis
    succeeds, which needs a configured provider.
    """
    called: list[bool] = []
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_generate_tts",
        lambda self: called.append(True),
        raising=True,
    )

    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Enable it deliberately. With no catalog loaded the button is
        # CORRECTLY disabled -- generation cannot resolve a provider -- and
        # a disabled button swallows the press, so the test would pass or
        # fail on the button's state rather than on the wiring it is about.
        button = app.query_one("#tts-generate-btn")
        button.disabled = False
        button.press()
        await pilot.pause()

    assert called == [True], "Generate did not reach the synthesis path"


@pytest.mark.asyncio
async def test_the_synthesis_state_exists_before_any_press():
    """`_generate_tts` reads these four with no guard; a host that forgets
    `init_synthesis_state` fails with AttributeError mid-generation."""
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        for attribute in (
            "reference_audio_path",
            "higgs_reference_audio_path",
            "_provider_ids",
            "_generation_operation_id",
        ):
            assert hasattr(pane, attribute), attribute
