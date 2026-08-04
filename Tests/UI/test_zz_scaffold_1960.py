"""TEMPORARY instrumentation scaffold for task-1960. DELETE BEFORE COMMIT."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
from textual.widget import Widget
from textual.widgets import Button, Input, Select
from textual.widgets._select import SelectCurrent

from Tests.UI.full_app_destination_context import (
    StaticWatchlistsScopeService,
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
    wait_for_selector as _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

EVENTS: list[str] = []


def _log(msg: str) -> None:
    EVENTS.append(f"{time.monotonic():.6f} {msg}")


@pytest.fixture
def instrument(monkeypatch):
    EVENTS.clear()

    real_screen_recompose = WatchlistsCollectionsScreen.recompose

    async def patched_screen_recompose(self):
        _log(">>> SCREEN recompose ENTER")
        try:
            return await real_screen_recompose(self)
        finally:
            _log("<<< SCREEN recompose EXIT")

    monkeypatch.setattr(
        WatchlistsCollectionsScreen, "recompose", patched_screen_recompose
    )

    real_pane_recompose = SourcesPane.recompose

    async def patched_pane_recompose(self):
        _log("  >>> PANE recompose ENTER")
        try:
            return await real_pane_recompose(self)
        finally:
            _log("  <<< PANE recompose EXIT")

    monkeypatch.setattr(SourcesPane, "recompose", patched_pane_recompose)

    real_mount = Widget.mount

    def patched_mount(self, *widgets, before=None, after=None):
        if self._closing or self._pruning:
            _log(
                f"      !! MOUNT-SUPPRESSED {type(self).__name__}(id={self.id}) "
                f"closing={self._closing} pruning={self._pruning} "
                f"lost={[type(w).__name__ for w in widgets]}"
            )
        return real_mount(self, *widgets, before=before, after=after)

    monkeypatch.setattr(Widget, "mount", patched_mount)

    real_update = SelectCurrent.update

    def patched_update(self, label):
        if not self.query("#label"):
            _log(
                f"      *** FATAL SelectCurrent#label missing "
                f"is_mounted={self.is_mounted} pruning={self._pruning} "
                f"parent_select={getattr(self.parent, 'id', None)}"
            )
        return real_update(self, label)

    monkeypatch.setattr(SelectCurrent, "update", patched_update)

    yield EVENTS


def _watchlists_host():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    return _visual_destination_harness(app, "watchlists_collections")


async def _open_sources_create_form(pilot, host):
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await _wait_for_selector(screen, pilot, "#watchlists-sources-pane", timeout=5.0)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    screen.query_one("#sources-new-button", Button).press()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        focused = screen.focused
        selects = list(pane.query(Select))
        if (
            pane.query("#sources-create-form")
            and focused is not None
            and focused.id == "sources-create-name"
            and selects
            and all(bool(select.query("#label")) for select in selects)
        ):
            break
        await pilot.pause(0.02)
    assert pane.query("#sources-create-form"), "the create form never opened"
    await pilot.pause()
    return screen, pane


@pytest.mark.parametrize("size", [(160, 42), (235, 52)])
@pytest.mark.asyncio
async def test_scaffold_instrumented_create(instrument, size):
    host = _watchlists_host()
    try:
        async with host.run_test(size=size) as pilot:
            screen, pane = await _open_sources_create_form(pilot, host)
            created = AsyncMock(return_value={"id": 1, "name": "Morning"})
            screen._controller.create_source = created

            await pilot.press(*"Morning")
            await pilot.press("tab")
            await pilot.pause(0.05)
            await pilot.press(*"https://example.com/feed")
            await pilot.pause(0.05)
            assert pane.query_one("#sources-create-name", Input).value == "Morning"
            _log("=== SUBMIT PRESSED ===")
            pane.query_one("#sources-create-submit", Button).press()
            for _ in range(200):
                if created.await_count == 1 and not pane.query("#sources-create-form"):
                    break
                await pilot.pause(0.01)
            _log("=== DONE ===")
            assert created.await_count == 1
            assert not pane.query("#sources-create-form")
    finally:
        print("\n\n========== EVENT LOG ==========")
        for line in EVENTS:
            print(line)
        print("========== END EVENT LOG ==========\n")
