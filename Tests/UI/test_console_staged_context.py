"""Console staged context tray tests (Inspector-rail Context section)."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleStagedContextState,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
)


def _five_reference_launch() -> ConsoleLiveWorkLaunch:
    """A real 5-reference bundle, staged the way Library RAG actually stages one.

    Each reference explodes into 3-4 provenance rows (Evidence source /
    Evidence authority / Evidence status / optional Snippet) inside
    ``ConsoleStagedContextState.from_live_work``, so 5 references yield
    17-22 display rows -- the exact gap that let the tray render
    "Sources 18" (D1a): the OLD code rendered ``len(rows)`` instead of the
    true staged-source count.
    """
    bundle = EvidenceBundle(
        bundle_id="bundle-5",
        query="What changed?",
        references=tuple(
            EvidenceReference(
                evidence_id=f"S{index}",
                source_id=f"media-{index}",
                source_type="media",
                title=f"Source {index}",
                snippet=f"Body {index}",
                authority_label="local",
                status="available",
                source_owner="local",
            )
            for index in range(1, 6)
        ),
    )
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={"query": "What changed?", "evidence_bundle": bundle.to_payload()},
        status="staged",
    )


@pytest.mark.asyncio
async def test_staged_context_renders_source_count() -> None:
    """The tray header shows the number of staged sources."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(
                    heading="Context",
                    summary="",
                    rows=(ConsoleDisplayRow("Source", "readme.md", status="ready"),),
                )
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        count = tray.query_one("#console-staged-context-count", Static)
        assert str(count.renderable) == "1"


@pytest.mark.asyncio
async def test_staged_context_omits_attach_button() -> None:
    """The redesign removes the Attach button from the empty tray."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(heading="Context", summary="", rows=())
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        assert not list(tray.query("#console-staged-context-attach"))


@pytest.mark.asyncio
async def test_staged_context_empty_shows_guidance() -> None:
    """An empty tray prompts the user to stage sources from the Library."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(heading="Context", summary="", rows=())
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        empty = tray.query_one("#console-staged-context-empty", Static)
        assert "Stage sources from Library" in str(empty.renderable)


@pytest.mark.asyncio
async def test_staged_context_row_renders_name_and_normalized_status() -> None:
    """Each source renders its value and a normalized status line."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(
                    heading="Context",
                    summary="",
                    rows=(
                        ConsoleDisplayRow("Source", "readme.md", status="available"),
                        ConsoleDisplayRow("Source", "missing.txt", status="missing"),
                    ),
                )
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        name = tray.query_one("#console-staged-source-name-0", Static)
        status = tray.query_one("#console-staged-source-status-0", Static)
        assert str(name.renderable) == "readme.md"
        assert str(status.renderable) == "ready"
        assert status.has_class("ready")

        blocked_status = tray.query_one("#console-staged-source-status-1", Static)
        assert str(blocked_status.renderable) == "blocked"
        assert blocked_status.has_class("blocked")


@pytest.mark.asyncio
async def test_staged_context_tray_counts_sources_not_display_rows() -> None:
    """D1a: a 5-reference bundle must render '5', not the exploded row count.

    Built via the REAL ``from_live_work`` (not a hand-built 1-row state --
    that shape is exactly what let "Sources 18" ship, since a 1-reference
    launch's row count and source count coincide).
    """
    launch = _five_reference_launch()
    state = ConsoleStagedContextState.from_live_work(launch)
    # Sanity: this bundle really does explode past the source count --
    # otherwise this test could not distinguish the fix from the bug.
    assert len(state.rows) > 5
    assert state.source_count == 5

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(state)

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        count = tray.query_one("#console-staged-context-count", Static)
        assert str(count.renderable) == "5"


@pytest.mark.asyncio
async def test_staged_context_tray_counts_zero_when_genuinely_empty() -> None:
    """The genuinely-empty state (nothing staged at all) still renders '0'."""
    state = ConsoleStagedContextState.empty()
    assert state.source_count == 0

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(state)

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        count = tray.query_one("#console-staged-context-count", Static)
        assert str(count.renderable) == "0"
