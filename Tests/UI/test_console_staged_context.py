"""Console staged context tray tests (Inspector-rail Context section)."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

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
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
)
from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
    ConsoleStagedSourceOpenRequested,
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


def _single_reference_launch(source_type: str) -> ConsoleLiveWorkLaunch:
    bundle = EvidenceBundle(
        bundle_id=f"bundle-{source_type}",
        query="Open it",
        references=(
            EvidenceReference(
                evidence_id="S1",
                source_id="source-1",
                source_type=source_type,
                title="Openable source",
                snippet="Body",
                authority_label="local",
                status="available",
                source_owner="local",
            ),
        ),
    )
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={"query": "Open it", "evidence_bundle": bundle.to_payload()},
        status="staged",
    )


@pytest.mark.parametrize(
    ("raw_source_type", "canonical_source_type"),
    (
        ("note", "notes"),
        ("notes", "notes"),
        ("media_chunk", "media"),
        ("conversation", "conversations"),
        ("prompt", "prompt"),
        ("prompts", "prompt"),
    ),
)
def test_staged_context_canonicalizes_openable_library_source_types(
    raw_source_type: str,
    canonical_source_type: str,
) -> None:
    state = ConsoleStagedContextState.from_live_work(
        _single_reference_launch(raw_source_type)
    )

    row = state.source_rows[0]
    assert row.source_type == canonical_source_type
    assert row.action_label == "Open in Library"


@pytest.mark.asyncio
async def test_staged_context_posts_canonical_source_navigation() -> None:
    state = ConsoleStagedContextState.from_live_work(_single_reference_launch("note"))

    class TestApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.open_request: tuple[str, str] | None = None

        def compose(self):
            yield ConsoleStagedContextTray(state)

        def on_console_staged_source_open_requested(
            self, event: ConsoleStagedSourceOpenRequested
        ) -> None:
            self.open_request = (event.source_type, event.source_id)

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.click("#console-staged-source-primary-0")
        await pilot.click("#console-staged-source-action-0")
        await pilot.pause()

        assert app.open_request == ("notes", "source-1")


def test_inspector_rail_routes_staged_source_request_to_library() -> None:
    rail = Mock()
    event = Mock(source_type="prompt", source_id="prompt-1")

    ConsoleInspectorRail.open_staged_source(rail, event)

    event.stop.assert_called_once_with()
    message = rail.app.post_message.call_args.args[0]
    assert isinstance(message, NavigateToScreen)
    assert message.screen_name == "library"
    assert message.screen_context == {
        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: "prompt",
        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: "prompt-1",
    }


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


@pytest.mark.asyncio
async def test_staged_context_recovery_renders_markup_hostile_text_literally() -> None:
    hostile_recovery = "Retry [/] then inspect [bold]details"
    state = ConsoleStagedContextState(
        heading="Context",
        summary="",
        rows=(),
        recovery=hostile_recovery,
    )

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(state)

    app = TestApp()
    async with app.run_test():
        recovery = app.query_one("#console-staged-context-recovery", Static)
        assert recovery._render_markup is False
        assert str(recovery.renderable) == hostile_recovery
