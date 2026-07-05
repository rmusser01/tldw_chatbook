"""Home triage rail + focus canvas contracts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_screen_navigation import _build_test_app

TRIAGE_TEST_SIZE = (160, 44)


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
                updated_at=(datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat(),
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
        assert home.query_one("#home-rail-section-body-details").styles.display == "none"


@pytest.mark.asyncio
async def test_home_row_click_switches_canvas():
    app = _build_test_app()
    app._home_dashboard_test_input = _triage_input()
    host = HomeHarness(app)
    async with host.run_test(size=TRIAGE_TEST_SIZE) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-rail")
        title = home.query_one("#home-canvas-title")
        assert "Approval: publish chatbook" in str(getattr(title.renderable, "plain", title.renderable))
        running_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == "watch:run-1"
        )
        running_button.press()
        await pilot.pause(0.1)
        title = home.query_one("#home-canvas-title")
        assert "Watchlist sweep" in str(getattr(title.renderable, "plain", title.renderable))


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
        assert home.query_one("#home-rail-section-body-details").styles.display != "none"
    sections = app.app_config.get("home", {}).get("rail_state", {}).get("sections", {})
    assert sections.get("details_open") is True


def test_generated_stylesheet_includes_home_triage_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
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
