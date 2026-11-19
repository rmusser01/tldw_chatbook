"""Home triage rail + focus canvas contracts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Screens import home_screen as home_screen_module
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.app_factory import _build_test_app

TRIAGE_TEST_SIZE = (160, 44)


@pytest.fixture(autouse=True)
def _stub_home_rail_preference_disk_writes(monkeypatch):
    """Keep this file's rail-preference assertions off the shared disk config.

    ``HomeScreen._save_home_rail_preferences`` writes ``[home.rail_state]``
    through the real ``save_setting_to_cli_config``; under the shared
    bootstrap profile that persists into every later test's loaded config
    (observed flipping section defaults for tests that run after
    ``test_home_details_toggle_persists``). The toggle test asserts the
    in-memory ``app_config`` mutation, which this stub does not touch --
    only the disk write is suppressed (mirrors the home_screen file's own
    autouse CLI-fallback stub rationale).
    """
    monkeypatch.setattr(
        home_screen_module,
        "save_setting_to_cli_config",
        lambda *args, **kwargs: True,
    )


def _triage_input(**overrides) -> HomeDashboardInput:
    defaults = dict(
        model_ready=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="wf:approve-1",
                title="Approval: publish chatbook",
                source="Workflows",
                status="pending_approval",
                detail_route="workflows",
                console_available=True,
                updated_at=(
                    datetime.now(timezone.utc) - timedelta(minutes=3)
                ).isoformat(),
            ),
            HomeActiveWorkItem(
                item_id="watch:run-1",
                title="Watchlist sweep",
                source="Watchlists",
                status="running",
                detail_route="subscriptions",
                updated_at="2026-07-04T12:00:00+00:00",
            ),
        ),
        recent_work_items=(
            HomeActiveWorkItem(
                item_id="recent:1",
                title="Done: nightly digest",
                source="Watchlists",
                status="completed",
                detail_route="subscriptions",
                updated_at="2026-07-04T02:00:00+00:00",
            ),
        ),
    )
    defaults.update(overrides)
    return HomeDashboardInput(**defaults)


def _visible_text(screen) -> str:
    chunks = []
    for widget in screen.query("Static"):
        renderable = getattr(widget, "renderable", "")
        chunks.append(getattr(renderable, "plain", str(renderable)))
    for widget in screen.query("Button"):
        label = getattr(widget, "label", "")
        chunks.append(str(label) if label is not None else "")
    return " ".join(chunks)


async def _wait_for_selector(screen, pilot, selector: str):
    for _ in range(40):
        matches = list(screen.query(selector))
        if matches:
            return matches[0]
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} never mounted")


@pytest.mark.asyncio
async def test_home_rail_renders_sections_with_counts_and_selection():
    app = _build_test_app()
    app._home_dashboard_test_input = _triage_input()
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail")
        text = _visible_text(home)
        assert "Needs Attention (1)" in text
        assert "Running (1)" in text
        assert "Approval: publish chatbook" in text
        assert "3m" in text  # age label from the now-relative fixture timestamp
        assert (
            home.query_one("#home-rail-section-body-details").styles.display == "none"
        )


@pytest.mark.asyncio
async def test_home_row_click_switches_canvas():
    app = _build_test_app()
    app._home_dashboard_test_input = _triage_input()
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail")
        title = home.query_one("#home-canvas-title")
        assert "Approval: publish chatbook" in str(
            getattr(title.renderable, "plain", title.renderable)
        )
        running_button = next(
            btn
            for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == "watch:run-1"
        )
        running_button.press()
        await pilot.pause(0.1)
        title = home.query_one("#home-canvas-title")
        assert "Watchlist sweep" in str(
            getattr(title.renderable, "plain", title.renderable)
        )


@pytest.mark.asyncio
async def test_home_canvas_action_dispatches_control():
    app = _build_test_app()
    app._home_dashboard_test_input = _triage_input()
    calls = []
    app.approve_active_home_item = lambda **kwargs: calls.append(("approve", kwargs))
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-canvas-actions")
        approve = home.query_one("#home-approve")
        approve.press()
        await pilot.pause(0.1)
        assert calls and calls[0][0] == "approve"


@pytest.mark.asyncio
async def test_home_empty_input_shows_next_action_canvas_and_empty_copy():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(model_ready=True)
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail")
        text = _visible_text(home)
        assert "No approvals or failures pending." in text
        assert "Nothing running right now." in text
        assert home.query_one("#home-primary-action")


@pytest.mark.asyncio
async def test_home_details_toggle_persists():
    app = _build_test_app()
    app._home_dashboard_test_input = _triage_input()
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail-section-header-details")
        toggle = home.query_one("#console-rail-section-toggle-home-details")
        toggle.press()
        await pilot.pause(0.15)
        assert (
            home.query_one("#home-rail-section-body-details").styles.display != "none"
        )
    sections = app.app_config.get("home", {}).get("rail_state", {}).get("sections", {})
    assert sections.get("details_open") is True


@pytest.mark.asyncio
async def test_home_rail_rows_truncate_titles_and_keep_source_line():
    app = _build_test_app()
    long_title = "Approval: publish the quarterly research chatbook digest"
    app._home_dashboard_test_input = _triage_input(
        active_work_items=(
            HomeActiveWorkItem(
                item_id="wf:approve-long",
                title=long_title,
                source="Workflows",
                status="pending_approval",
                detail_route="workflows",
                console_available=True,
                updated_at=(
                    datetime.now(timezone.utc) - timedelta(minutes=3)
                ).isoformat(),
            ),
        ),
    )
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail")
        row = next(
            btn
            for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == "wf:approve-long"
        )
        title_line, source_line = str(row.label).split("\n")
        assert title_line.endswith("...")
        assert long_title not in title_line
        assert len(title_line) <= 26  # marker + glyph + 20-char visible title
        assert "Workflows" in source_line
        assert "3m" in source_line  # age moves to the source line, never clipped
        assert row.tooltip and long_title in str(row.tooltip)


@pytest.mark.asyncio
async def test_home_canvas_actions_render_in_horizontal_toolbar():
    from textual.containers import Horizontal

    app = _build_test_app()
    app._home_dashboard_test_input = _triage_input()
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-canvas-actions")
        toolbar = home.query_one("#home-canvas-actions .ds-toolbar")
        assert isinstance(toolbar, Horizontal)
        approve = home.query_one("#home-approve")
        assert toolbar in approve.ancestors
        assert not approve.has_class("ds-toolbar")


def test_generated_stylesheet_includes_home_triage_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "features" / "_home.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        "#home-triage-grid",
        ".home-rail-row",
        ".home-rail-row-selected",
        ".home-rail-empty-copy",
        "#home-next-action-callout",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    for stale in ("#home-dashboard-grid", ".home-pane-divider", "#home-followup-row"):
        assert stale not in component_css, stale
        assert stale not in generated_css, stale


# --- TASK-2762: rail/canvas scroll when content exceeds the pane ----------


def _full_attention_input(count: int = 12) -> HomeDashboardInput:
    """A needs-attention list past the fold: each row is 2 cells + 1 margin,
    so 12 rows need ~36 rows of rail -- far beyond a 24-row terminal."""
    return _triage_input(
        active_work_items=tuple(
            HomeActiveWorkItem(
                item_id=f"local:watchlist_run:{index}",
                title=f"Failed run {index}",
                source="W+C",
                status="failed",
                detail_route="subscriptions",
                updated_at="2026-07-04T12:00:00+00:00",
            )
            for index in range(count)
        ),
    )


@pytest.mark.asyncio
async def test_home_rail_scrolls_to_rows_past_the_fold():
    """TASK-2762 AC1+AC2: with a full attention list on a 24-row terminal,
    the rail must actually scroll (overflow-y auto) and every row stays
    reachable -- scrolling to the end brings the LAST row on screen."""
    app = _build_test_app()
    app._home_dashboard_test_input = _full_attention_input()
    host = HomeHarness(app)

    async with host.run_test(size=(120, 24)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail")
        rail = home.query_one("#home-rail")

        assert str(rail.styles.overflow_y) == "auto", (
            "rail must be scrollable when content exceeds the viewport "
            "(Textual Vertical default is hidden)"
        )
        assert rail.max_scroll_y > 0, "12 attention rows must overflow a 24-row pane"
        assert rail.show_vertical_scrollbar, (
            "the overflow must be visible: a scrollbar (or cue), not a silent clip"
        )

        last_row = next(
            button
            for button in home.query("Button.home-rail-row")
            if str(getattr(button, "row_id", "")) == "local:watchlist_run:11"
        )
        assert last_row.region is None or not _row_visible_in(rail, last_row), (
            "precondition: the last row starts past the fold"
        )

        rail.scroll_to_widget(last_row, animate=False)
        # Regions update on the next refresh passes -- wait for the last
        # row to land inside the rail's box rather than racing one pause.
        for _ in range(40):
            if _row_visible_in(rail, last_row):
                break
            await pilot.pause(0.05)
        assert _row_visible_in(rail, last_row), (
            "the last rail row must be reachable by scrolling"
        )


def _row_visible_in(rail, row) -> bool:
    """True when the row's screen region intersects the rail's box."""
    rail_box = rail.content_region
    row_region = row.region
    if row_region is None:
        return False
    return (
        row_region.y < rail_box.y + rail_box.height
        and row_region.y + row_region.height > rail_box.y
    )


@pytest.mark.asyncio
async def test_home_canvas_scrolls_when_content_exceeds_the_pane():
    """TASK-2762 AC1: the canvas can overflow too (failed item with a
    verbose reason + a full control toolbar on a 24-row terminal) -- it
    must scroll rather than clip."""
    from dataclasses import replace as dataclass_replace

    app = _build_test_app()
    app._home_dashboard_test_input = dataclass_replace(
        _full_attention_input(count=1),
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:watchlist_run:0",
                title="Failed run with a very long verbose reason attached",
                source="W+C",
                status="failed",
                detail_route="subscriptions",
                status_detail="Error: " + ("retryable parse failure " * 8),
                updated_at="2026-07-04T12:00:00+00:00",
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=(120, 24)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-canvas")
        canvas = home.query_one("#home-canvas")
        assert str(canvas.styles.overflow_y) == "auto"
