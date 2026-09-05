"""Library test interactions must use the current mounted control owner."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button

from Tests.UI.test_library_shell import _wait_for_selector


@pytest.mark.asyncio
async def test_selector_wait_rechecks_owner_after_pilot_settlement(monkeypatch):
    class SelectorApp(App):
        def compose(self) -> ComposeResult:
            yield Vertical(Button("Original", id="action"), id="owner")

    app = SelectorApp()
    async with app.run_test() as pilot:
        original = app.query_one("#action", Button)
        replacement = Button("Replacement", id="action")
        pause = pilot.pause
        replaced = False

        async def replace_during_pause(*args, **kwargs):
            nonlocal replaced
            if not replaced:
                replaced = True
                await original.remove()
                await app.query_one("#owner", Vertical).mount(replacement)
            await pause(*args, **kwargs)

        monkeypatch.setattr(pilot, "pause", replace_during_pause)
        result = await _wait_for_selector(app.screen, pilot, "#action")

        assert result is replacement
        assert result.is_attached
        assert not original.is_attached
