"""Regression coverage for legacy MediaViewerPanel content-search timing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.await_complete import AwaitComplete
from textual.widgets import Input, Markdown

from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


CONTENT = "budget planning mentions budget twice, then a third budget follows."
NEW_CONTENT = "replacement budget transcript with a fresh matching word."


class _TestMediaViewerPanel(MediaViewerPanel):
    def populate_providers(self) -> None:
        pass


class _MediaViewerApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.panel = _TestMediaViewerPanel(Mock())

    def compose(self) -> ComposeResult:
        yield self.panel


def _record_content_updates(
    monkeypatch: pytest.MonkeyPatch, panel: MediaViewerPanel
) -> tuple[Markdown, list[tuple[str, AwaitComplete]]]:
    content_display = panel.query_one("#content-display", Markdown)
    updates: list[tuple[str, AwaitComplete]] = []
    original_update = content_display.update

    def recording_update(markdown: str) -> AwaitComplete:
        completion = original_update(markdown)
        updates.append((markdown, completion))
        return completion

    monkeypatch.setattr(content_display, "update", recording_update)
    return content_display, updates


async def _await_updates(updates: list[tuple[str, AwaitComplete]]) -> None:
    for _, completion in updates:
        await completion


async def _mounted_panel(
    pilot, monkeypatch: pytest.MonkeyPatch, *, record_id: str | None = "media-1"
) -> tuple[MediaViewerPanel, Input, list[tuple[str, AwaitComplete]]]:
    panel = pilot.app.panel
    panel.load_media({"id": record_id, "title": "Budget", "content": CONTENT})
    await pilot.pause()
    _, updates = _record_content_updates(monkeypatch, panel)
    return panel, panel.query_one("#content-search-input", Input), updates


@pytest.mark.asyncio
async def test_content_search_burst_renders_only_final_query_after_debounce(monkeypatch):
    app = _MediaViewerApp()
    async with app.run_test() as pilot:
        _, search_input, updates = await _mounted_panel(pilot, monkeypatch)
        search_input.value = "b"
        await pilot.pause(0.04)
        search_input.value = "bu"
        await pilot.pause(0.04)
        search_input.value = "budget"
        await pilot.pause(0.10)
        assert updates == []
        await pilot.pause(0.20)
        await _await_updates(updates)
        assert len(updates) == 1
        assert updates[0][0]
        assert "▶ budget ◀" in updates[0][0]


@pytest.mark.asyncio
async def test_clearing_search_renders_unhighlighted_content_without_stale_callback(monkeypatch):
    app = _MediaViewerApp()
    async with app.run_test() as pilot:
        _, search_input, updates = await _mounted_panel(pilot, monkeypatch)
        search_input.value = "budget"
        await pilot.pause(0.05)
        search_input.value = ""
        await pilot.pause()
        await pilot.pause(0.30)
        await _await_updates(updates)
        assert len(updates) == 1
        assert "▶" not in updates[0][0]
        assert CONTENT in updates[0][0]


@pytest.mark.asyncio
async def test_loading_replacement_prevents_pending_query_from_rendering_on_new_content(monkeypatch):
    app = _MediaViewerApp()
    async with app.run_test() as pilot:
        panel, search_input, updates = await _mounted_panel(pilot, monkeypatch)
        search_input.value = "budget"
        await pilot.pause(0.05)
        panel.load_media({"id": "media-2", "title": "New", "content": NEW_CONTENT})
        await pilot.pause(0.30)
        await _await_updates(updates)
        assert updates
        assert all("▶ budget ◀" not in markdown for markdown, _ in updates)
        assert updates[-1][0] == NEW_CONTENT


@pytest.mark.asyncio
async def test_clear_display_remains_authoritative_after_pending_query(monkeypatch):
    app = _MediaViewerApp()
    async with app.run_test() as pilot:
        panel, search_input, updates = await _mounted_panel(pilot, monkeypatch)
        search_input.value = "budget"
        await pilot.pause(0.05)
        panel.clear_display()
        await pilot.pause(0.30)
        await _await_updates(updates)
        assert updates
        assert updates[-1][0] == "*No item selected*"
        assert all("▶ budget ◀" not in markdown for markdown, _ in updates)


@pytest.mark.asyncio
async def test_unmounted_panel_ignores_pending_search_callback(monkeypatch):
    app = _MediaViewerApp()
    async with app.run_test() as pilot:
        panel, search_input, updates = await _mounted_panel(pilot, monkeypatch)
        search_input.value = "budget"
        await pilot.pause(0.05)
        await panel.remove()
        await pilot.pause(0.30)
        assert updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("record_id", ["same-id", None])
async def test_generation_invalidates_pending_query_for_same_or_missing_record_id(
    monkeypatch, record_id
):
    app = _MediaViewerApp()
    async with app.run_test() as pilot:
        panel, _, updates = await _mounted_panel(
            pilot, monkeypatch, record_id=record_id
        )
        panel.handle_content_search(SimpleNamespace(value="budget"))
        await pilot.pause(0.05)
        panel.load_media({"id": record_id, "title": "Second", "content": NEW_CONTENT})
        await pilot.pause(0.30)
        await _await_updates(updates)
        assert updates
        assert updates[-1][0] == NEW_CONTENT
        assert all("▶ budget ◀" not in markdown for markdown, _ in updates)
