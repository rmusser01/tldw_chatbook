"""Helpers for exercising production destination screens in the full app.

This module deliberately contains no ``App`` subclass. Tests using it run the
real :class:`TldwCli` built by ``_build_test_app`` and mount the production
Watchlists screen with the production stylesheet and application lifecycle.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import patch

from textual.widgets import Button, Static

from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)


class StaticWatchlistsScopeService:
    """Small deterministic backend double for Watchlists list operations."""

    def __init__(self, watch_items):
        self.watch_items = tuple(watch_items)
        self.calls = []

    async def list_watch_items(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.watch_items)


def _settings_without_splash(section, key=None, default=None):
    """Keep the full app deterministic without replacing its composition."""
    if section == "splash_screen" and key == "enabled":
        return False
    return default


class _ScreenWorkerView:
    """Expose only workers owned by the mounted destination screen."""

    def __init__(self, app, screen) -> None:
        self._app = app
        self._screen = screen

    def _owned_workers(self):
        return [
            worker
            for worker in self._app.workers
            if self._screen in worker.node.ancestors_with_self
        ]

    def __iter__(self):
        return iter(self._owned_workers())

    async def wait_for_complete(self) -> None:
        unfinished = [
            worker for worker in self._owned_workers() if not worker.is_finished
        ]
        if unfinished:
            await self._app.workers.wait_for_complete(unfinished)


class FullAppDestinationContext:
    """Mount one production destination inside the production application."""

    def __init__(self, app, destination: str) -> None:
        if destination != "watchlists_collections":
            raise ValueError(f"Unsupported production destination: {destination}")
        self.app = app
        self.context_screen = WatchlistsCollectionsScreen(app)

    @property
    def screen_stack(self):
        return self.app.screen_stack

    @property
    def workers(self):
        return _ScreenWorkerView(self.app, self.context_screen)

    async def push_screen(self, *args, **kwargs):
        return await self.app.push_screen(*args, **kwargs)

    async def pop_screen(self, *args, **kwargs):
        return await self.app.pop_screen(*args, **kwargs)

    @asynccontextmanager
    async def run_test(self, **kwargs):
        with patch(
            "tldw_chatbook.app.get_cli_setting",
            side_effect=_settings_without_splash,
        ):
            async with self.app.run_test(**kwargs) as pilot:
                await self.app.push_screen(self.context_screen)
                await pilot.pause()
                yield pilot


def active_destination_screen(host: FullAppDestinationContext):
    """Return the production destination mounted above the app's base screen."""
    return host.context_screen


def full_app_destination_context(app, route: str) -> FullAppDestinationContext:
    """Build a full-application context for a production destination."""
    return FullAppDestinationContext(app, route)


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _visible_text(screen) -> str:
    return " ".join(
        [
            *(
                _static_text(widget)
                for widget in screen.query(Static)
                if widget.display and hasattr(widget, "renderable")
            ),
            *(
                str(button.label)
                for button in screen.query(Button)
                if button.display and button.label is not None
            ),
        ]
    )


async def wait_for_selector(
    screen, pilot, selector: str, *, timeout: float = 2.0
) -> None:
    """Wait for a selector on the mounted production destination."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if screen.query(selector):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for {selector}. Visible text: {_visible_text(screen)}"
    )
