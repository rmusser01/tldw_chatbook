"""The axes must offer real choices, not render as empty selects.

This is the difference between a screen that looks finished and one that
works. The rebuilt pane can mount all 57 controls, put Generate above the
fold and wire the press to synthesis, and still be useless if nothing ever
fills the provider, model and voice lists.

The fake service and catalog come from the legacy playground's own tests, so
both panes are driven by identical inputs -- the point of sharing
`SpeechCatalogMixin` is that they behave the same.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from Tests.UI.test_stts_playground_audio_cpp import (
    FakeTTSService,
    _resolved,
    _wait_until,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(provider="audio_cpp")


def _option_values(select: Select[Any]) -> tuple[Any, ...]:
    return tuple(value for _label, value in select._options)


@pytest.fixture
def faked_service(monkeypatch):
    """Point the pane's service hook at the legacy tests' fake."""
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


@pytest.mark.asyncio
async def test_the_provider_axis_offers_real_providers(faked_service):
    """An empty provider list means nothing downstream can resolve.

    Before `on_mount` kicked off the catalog, this select held only
    `Select.NULL` -- a control with nothing to choose.
    """
    app = _Harness()
    async with app.run_test(size=(160, 60)) as pilot:
        select = app.query_one("#tts-provider-select", Select)
        await _wait_until(
            pilot,
            lambda: any(isinstance(v, str) for v in _option_values(select)),
        )
        values = _option_values(select)

    assert "audio_cpp" in values, f"provider axis never populated: {values}"
