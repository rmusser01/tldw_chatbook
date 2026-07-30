"""One provider's settings: one row each, and a header that says its state."""

from __future__ import annotations

import pathlib

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible

from tldw_chatbook.UI.Speech.speech_settings_group import (
    SETTING_LABELS,
    SpeechSettingsGroup,
)
from tldw_chatbook.UI.Speech.speech_settings_model import (
    SETTINGS_PROVIDER_ORDER,
    settings_for_provider,
)


#: The row-cost assertion measures a rule that lives in the app-tier bundle,
#: and a bare `App` never loads it -- so without this the harness cannot see
#: the fix in either direction and the test proves nothing.
_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = _BUNDLE

    def __init__(self, provider, values=None, collapsed=True):
        super().__init__()
        self._provider = provider
        self._values = values or {}
        self._collapsed = collapsed

    def compose(self) -> ComposeResult:
        yield SpeechSettingsGroup(
            provider=self._provider,
            values=self._values,
            collapsed=self._collapsed,
        )


@pytest.mark.asyncio
async def test_a_collapsed_group_states_its_provider_state():
    """Eight identical closed boxes tell the user nothing. The header has to
    answer "is this one set up?" without being opened."""
    app = _Harness("elevenlabs", {"elevenlabs-api-key-input": "sk-live"})
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        title = app.query_one(Collapsible).title
        assert "ElevenLabs" in title
        assert "configured" in title.lower()


@pytest.mark.asyncio
async def test_an_incomplete_provider_says_what_is_missing():
    """"Incomplete" alone sends the user hunting through the group."""
    app = _Harness("openai", {"openai-org-id-input": "org-1"})
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        title = app.query_one(Collapsible).title
        assert "incomplete" in title.lower()
        assert "API key" in title


@pytest.mark.asyncio
async def test_the_group_collapses_for_real():
    """Assert what renders, not the flag.

    Subclassing Collapsible and overriding compose() replaces the contents
    container it toggles, so the group renders fully open while still
    reporting `collapsed is True`. A flag-only assertion passed exactly that
    bug through in phase 1.
    """
    app = _Harness("openai")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        group = app.query_one(Collapsible)
        assert group.collapsed is True
        assert not app.query_one("#openai-api-key-input").region.height, (
            "collapsed group is still rendering its settings"
        )


@pytest.mark.asyncio
async def test_each_setting_costs_one_row_not_four():
    """The defect this phase exists to fix, asserted as a number.

    Measured on the shipped screen: every control cost 4 rows -- a blank, the
    input's top border, the label+value row, the bottom border -- which is
    why Save sat at y=102.
    """
    app = _Harness("audio_cpp", collapsed=False)
    async with app.run_test(size=(180, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        rows = sorted(
            w.region.y for w in app.query(".speech-setting-row") if w.region.height
        )
        assert len(rows) > 2, "no setting rows rendered"
        gaps = [b - a for a, b in zip(rows, rows[1:]) if b > a]
        assert max(gaps) == 1, f"settings still cost up to {max(gaps)} rows each"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", SETTINGS_PROVIDER_ORDER)
async def test_every_setting_gets_a_control_and_a_label(provider):
    """A bare id is not a control, and a missing one is a lost capability."""
    app = _Harness(provider, collapsed=False)
    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        for setting in settings_for_provider(provider):
            assert app.query(f"#{setting}"), f"{provider}: {setting} not rendered"
            assert setting in SETTING_LABELS, f"{setting} has no label"
