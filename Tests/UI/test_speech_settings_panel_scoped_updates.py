"""Speech & TTS settings dropdowns rebuild ONE card, not the panel (task-15475).

The input-latency audit found every provider/policy dropdown in
``SpeechTTSSettingsPanel`` calling ``await self.recompose()`` -- ~200 widgets
destroyed and rebuilt to repaint the handful of rows that actually changed,
with focus dropped on the floor every time (measured: ``app.focused`` is
``None`` after the change, even though the user had just operated a Select).

Identity, not counting, is the evidence here: a widget that survives a change
is the SAME Python object. These tests pin which cards survive which change,
plus the rendered outcome each change is supposed to produce (so a "scoped"
update that silently stopped updating something would fail too).
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Select

from Tests.UI.test_settings_speech_tts_panel import _PanelHarness

pytestmark = pytest.mark.asyncio

_BANNER = "#settings-speech-scope-banner"
_DEFAULTS = "#settings-speech-global-defaults"
_SETUP = "#settings-speech-provider-setup"
_REALTIME = "#settings-speech-realtime"
_INSPECTOR = "#settings-speech-inspector"
_ACTIONS = "#settings-speech-actions"


def _identities(panel, selectors) -> dict[str, int]:
    return {selector: id(panel.query_one(selector)) for selector in selectors}


async def _settle(pilot) -> None:
    for _ in range(8):
        await pilot.pause()


async def test_default_provider_change_rebuilds_only_defaults_and_inspector():
    """AC#2: the Global defaults card (and the inspector that reads it) are
    the only cards a Default TTS Provider change may replace."""
    host = _PanelHarness(configure_provider="openai")
    async with host.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        panel = host.query_one("#panel")
        untouched = _identities(panel, (_BANNER, _SETUP, _REALTIME, _ACTIONS))
        defaults_before = id(panel.query_one(_DEFAULTS))

        selector = panel.query_one("#settings-speech-default-provider", Select)
        selector.focus()
        await pilot.pause()
        selector.value = "audio_cpp"
        await _settle(pilot)

        assert _identities(panel, untouched) == untouched, (
            "A Default TTS Provider change replaced cards it does not feed."
        )
        # The defaults card itself is NOT replaced -- only its children are --
        # so the outcome, not the card's identity, is what proves it repainted.
        assert id(panel.query_one(_DEFAULTS)) == defaults_before
        assert isinstance(
            panel.query_one("#settings-speech-model-value"), Select
        ), "audio.cpp exposes Model value as a Select over its observed models."
        assert panel.query_one("#settings-speech-speed", Input).disabled is True
        constraints = panel.query_one("#settings-speech-default-constraints")
        assert "audio.cpp requires WAV" in str(constraints.renderable)


async def test_default_provider_change_keeps_focus_on_the_control_used():
    """The whole-panel recompose dropped focus entirely; a scoped swap must
    put the user back on the Select they just operated."""
    host = _PanelHarness(configure_provider="openai")
    async with host.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        panel = host.query_one("#panel")
        selector = panel.query_one("#settings-speech-default-provider", Select)
        selector.focus()
        await pilot.pause()
        assert getattr(host.focused, "id", None) == "settings-speech-default-provider"

        selector.value = "audio_cpp"
        await _settle(pilot)

        assert getattr(host.focused, "id", None) == "settings-speech-default-provider"


async def test_model_policy_change_rebuilds_only_defaults_and_inspector():
    """The Model policy dropdown swaps the Model value control's enablement."""
    host = _PanelHarness(configure_provider="openai")
    async with host.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        panel = host.query_one("#panel")
        untouched = _identities(panel, (_BANNER, _SETUP, _REALTIME, _ACTIONS))

        policy = panel.query_one("#settings-speech-model-policy", Select)
        policy.focus()
        await pilot.pause()
        policy.value = "first_available"
        await _settle(pilot)

        assert _identities(panel, untouched) == untouched
        assert panel.query_one("#settings-speech-model-value", Input).disabled is True
        assert getattr(host.focused, "id", None) == "settings-speech-model-policy"


async def test_configure_provider_change_rebuilds_only_setup_and_inspector():
    """Configure Provider owns the Provider setup card; it must not disturb
    the Global defaults card (it deliberately does not change the default)."""
    host = _PanelHarness(configure_provider="openai")
    async with host.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        panel = host.query_one("#panel")
        untouched = _identities(panel, (_BANNER, _DEFAULTS, _REALTIME, _ACTIONS))
        default_provider_before = id(
            panel.query_one("#settings-speech-default-provider")
        )

        selector = panel.query_one("#settings-speech-configure-provider", Select)
        selector.focus()
        await pilot.pause()
        selector.value = "elevenlabs"
        await _settle(pilot)

        assert _identities(panel, untouched) == untouched
        assert (
            id(panel.query_one("#settings-speech-default-provider"))
            == default_provider_before
        ), "Configure Provider must not rebuild the Global defaults controls."
        assert panel.query("#settings-speech-provider-elevenlabs"), (
            "The Provider setup card must show the newly configured provider."
        )
        summary = panel.query_one("#settings-speech-inspector-summary")
        assert "ElevenLabs" in str(summary.renderable)
