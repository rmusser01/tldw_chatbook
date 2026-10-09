"""B19: config search reuses one index per pane and reads live widget values.

Covers three efficiency defects in the settings-window config search:

1. Three keystrokes must execute ``_build_index`` exactly once (the engine
   is cached per pane and ``search()`` no longer re-walks the DOM).
2. Widget values are read live at search time: a value typed into a setting
   after the index was built is still found (no stale snapshots).
3. Switching the active settings tab invalidates the cached engine, so the
   next search rebuilds exactly once for the new pane.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, ListView, TextArea

# The autouse Tests/UI catalog-refresh fixture lazily imports
# `tldw_chatbook.app`; under the per-test config redirect that first import
# fails the raw-source admission (RecoveryRequired). Importing it here, at
# collection time while the bootstrap config is still bound, is the repo's
# established standalone-run pattern (see Tests/UI/conftest.py's notes).
import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow
from tldw_chatbook.UI.Widgets.config_search_widget import UIElementSearchEngine

# Reuse the established full-app harness for the settings window (same
# pattern as Tests/UI/test_tools_settings_window.py).
from Tests.UI.test_tools_settings_window import (
    _build_full_tools_app,
    _mounted_tools_window,
)

SEARCH_DEBOUNCE_SECONDS = 0.3

# The full-app boot in `_build_full_tools_app()` runs load_settings(); the
# per-test config redirect fails that closed with
# RecoveryRequired("raw_source_selection_changed") when the config module is
# still bound to the session's bootstrap source. The bootstrap_profile marker
# (Tests/conftest.py, TASK-32873) keeps the collection-time profile, which is
# what the mounted window's search-only assertions need (no config writes).
pytestmark = pytest.mark.bootstrap_profile


def _spy_build_index(monkeypatch) -> list[int]:
    """Count UIElementSearchEngine._build_index executions."""
    build_calls: list[int] = []
    real_build = UIElementSearchEngine._build_index

    def spy_build(self):
        build_calls.append(1)
        real_build(self)

    monkeypatch.setattr(UIElementSearchEngine, "_build_index", spy_build)
    return build_calls


async def _search_via_input(pilot, window, query: str) -> None:
    """Type a query into the config search input and settle the debounce."""
    search_input = window.query_one("#config-search-input", Input)
    search_input.focus()
    await pilot.pause()
    await pilot.press(*query)
    await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.2)


@pytest.mark.asyncio
async def test_three_keystrokes_build_index_exactly_once(monkeypatch):
    app = _build_full_tools_app()
    build_calls = _spy_build_index(monkeypatch)

    async with _mounted_tools_window(app) as (window, pilot):
        assert isinstance(window, ToolsSettingsWindow)

        await _search_via_input(pilot, window, "abc")

        assert len(build_calls) == 1, (
            f"expected exactly one index build across 3 keystrokes, "
            f"got {len(build_calls)}"
        )

        # A further search on the same pane reuses the cache: still one build.
        await _search_via_input(pilot, window, "abcd")
        assert len(build_calls) == 1


@pytest.mark.asyncio
async def test_search_reads_live_widget_values(monkeypatch):
    """Values are read live at search time from the cached index.

    Uses a synthetic pane: the legacy ToolsSettingsWindow compose flattens
    its bare ``yield``s out of the tab panes (pre-existing; the TASK-1346
    deprecated surface), so its panes index zero form elements and cannot
    exercise value matching. The engine-level contract under test -- cached
    index, live value reads, invalidate-rebuilds-once -- is identical.
    """
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll
    from textual.widgets import Label

    class _PaneHost(App[None]):
        def compose(self) -> ComposeResult:
            with VerticalScroll(id="pane-a"):
                yield Label("API Key")
                yield Input(value="short", id="api-key-input")

    app = _PaneHost()
    build_calls = _spy_build_index(monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()

        pane = app.query_one("#pane-a", VerticalScroll)
        engine = UIElementSearchEngine(pane)
        # Built once at construction, not per search.
        assert len(build_calls) == 1
        assert [e["widget_id"] for e in engine.elements_index] == ["api-key-input"]

        # Mutate the value AFTER the index was built, then search for the new
        # value: it must match via a live read (a stale snapshot would miss).
        api_input = app.query_one("#api-key-input", Input)
        api_input.value = "sk-zzuniqval-123"
        await pilot.pause()

        results = engine.search("zzuniqval")
        assert [r["widget"] for r in results] == [api_input]
        assert results[0]["current_value"] == "sk-zzuniqval-123"

        # The search reused the cached index (no rebuild).
        assert len(build_calls) == 1

        # invalidate() forces exactly one rebuild on the next search, and
        # the live read still reflects the current value.
        engine.invalidate()
        results = engine.search("zzuniqval")
        assert len(build_calls) == 2
        assert [r["widget"] for r in results] == [api_input]


@pytest.mark.asyncio
async def test_pane_switch_invalidates_and_rebuilds_once(monkeypatch):
    from textual.widgets import TabbedContent

    app = _build_full_tools_app()
    build_calls = _spy_build_index(monkeypatch)

    async with _mounted_tools_window(app) as (window, pilot):
        # First search builds the index once for the default (raw TOML) pane.
        await _search_via_input(pilot, window, "toml")
        assert len(build_calls) == 1

        # Switch to the General pane.
        tabs = window.query_one("#config-tabs", TabbedContent)
        tabs.active = "tab-general-config"
        await pilot.pause()

        # Next search rebuilds exactly once for the new pane...
        await _search_via_input(pilot, window, "general")
        assert len(build_calls) == 2, (
            f"pane switch must rebuild exactly once, builds={len(build_calls)}"
        )

        # ...and subsequent searches on the same pane are served from cache.
        await _search_via_input(pilot, window, "general2")
        assert len(build_calls) == 2
