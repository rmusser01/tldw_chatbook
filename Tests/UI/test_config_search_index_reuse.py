"""B19: the config search engine reuses one index per pane and reads live values.

The engine was written for the legacy settings window's search box. That window is gone
(TASK-33244), so its two mounted tests (one index build across keystrokes, a rebuild on a
pane switch) went with it; the engine-level contract they relied on is pinned here on a
synthetic pane: the index is built once, values are read live at search time, and
``invalidate()`` forces exactly one rebuild.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input

# The autouse Tests/UI catalog-refresh fixture lazily imports
# `tldw_chatbook.app`; under the per-test config redirect that first import
# fails the raw-source admission (RecoveryRequired). Importing it here, at
# collection time while the bootstrap config is still bound, is the repo's
# established standalone-run pattern (see Tests/UI/conftest.py's notes).
import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.UI.Widgets.config_search_widget import UIElementSearchEngine


def _spy_build_index(monkeypatch) -> list[int]:
    """Count UIElementSearchEngine._build_index executions."""
    build_calls: list[int] = []
    real_build = UIElementSearchEngine._build_index

    def spy_build(self):
        build_calls.append(1)
        real_build(self)

    monkeypatch.setattr(UIElementSearchEngine, "_build_index", spy_build)
    return build_calls


@pytest.mark.asyncio
async def test_search_reads_live_widget_values(monkeypatch):
    """Values are read live at search time from the cached index.

    Uses a synthetic pane holding one labelled input.
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
