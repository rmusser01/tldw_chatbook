"""Console staged context tray tests (Inspector-rail Context section)."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

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
    """A real five-reference bundle staged the way Library RAG stages one."""
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
    """Five references produce five compact, activatable primary rows."""
    launch = _five_reference_launch()
    state = ConsoleStagedContextState.from_live_work(launch)
    assert len(state.source_rows) == 5
    assert state.source_count == 5

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(state)

    app = TestApp()
    async with app.run_test() as pilot:
        tray = app.query_one(ConsoleStagedContextTray)
        count = tray.query_one("#console-staged-context-count", Static)
        assert str(count.renderable) == "5"
        primary_rows = list(tray.query(".console-staged-source-primary"))
        assert len(primary_rows) == 5
        first = tray.query_one("#console-staged-source-primary-0", Button)
        assert "Ready · Source 1 · media" == first.label.plain
        detail = tray.query_one("#console-staged-source-detail-0")
        assert detail.display is False

        await pilot.click("#console-staged-source-primary-0")

        assert detail.display is True
        detail_text = " ".join(
            str(item.renderable) for item in detail.query(Static)
        )
        assert "Body 1" in detail_text
        assert "Authority: local" in detail_text
        assert "Freshness: Current" in detail_text
        assert "Open in Library" in str(
            detail.query_one("#console-staged-source-action-0", Button).label
        )


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


@pytest.mark.asyncio
async def test_staged_context_tray_survives_a_markup_hostile_title() -> None:
    """PR-T1 I1: a launch title containing Rich markup must not crash compose.

    The summary line interpolates the launch title, and it was the only
    Static in this widget rendering untrusted text with markup ENABLED
    (every sibling row already passes ``markup=False``). A title carrying
    a stray closing tag -- ``[/]`` -- raised ``MarkupError`` from compose,
    and ``[bold]`` silently swallowed the text into a style.

    That was survivable while a staged launch died on navigation. It is
    not survivable now that D3 persists it: the restore succeeds on every
    later visit, so the crash lands in compose INSIDE ``switch_screen``
    (where ``app.py`` reports it as a navigation failure) and Console
    becomes permanently unopenable -- a sticky lockout from one badly
    named note.

    Asserted: compose completes, and the markup renders LITERALLY rather
    than being interpreted or dropped.
    """
    hostile_title = "Rotation [/] runbook [bold]v2"
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title=hostile_title,
        payload={"query": "rotation"},
        status="staged",
    )
    state = ConsoleStagedContextState.from_live_work(launch)
    assert hostile_title in state.summary

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(state)

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        summary = tray.query_one("#console-staged-context-summary", Static)
        rendered = str(summary.renderable)
        assert hostile_title in rendered
        # Not interpreted away: the literal tags are still in the output.
        assert "[/]" in rendered
        assert "[bold]" in rendered


@pytest.mark.asyncio
async def test_staged_context_summary_normalizes_launch_text() -> None:
    """PR-T1 I1 (defence in depth): the summary goes through the module's
    shared display-text normalizer instead of raw f-string interpolation,
    so it no longer stands out as this module's one unescaped exit.

    This does NOT fix the markup crash above (HTML escaping leaves ``[/]``
    untouched -- that is fixed at the sink with ``markup=False``); it is
    the same treatment every other value in
    ``ConsoleStagedContextState`` already receives.
    """
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Q&A <notes>",
        payload={},
        status="staged",
    )
    state = ConsoleStagedContextState.from_live_work(launch)
    assert "&amp;" in state.summary
    assert "&lt;notes&gt;" in state.summary
    assert "<notes>" not in state.summary
