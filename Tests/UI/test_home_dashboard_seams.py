"""Tests for task-282: Home dashboard seam threading/caching + targeted
rail/canvas patches.

Covers:
  * LocalNotificationHomeActiveWorkAdapter's short-TTL active-work cache
    (build_dashboard_input hits the cache instead of re-querying every
    call; a triage-action invalidation hook forces a recompute).
  * refresh_active_work_cache_async running the seam queries off the
    calling thread for a file-backed store, and staying inline for a
    per-connection ``:memory:`` store (the same sqlite thread-affinity
    hazard documented elsewhere in this codebase for ChaChaNotes).
  * HomeRail/HomeCanvas.sync_state patching targeted widgets instead of
    recomposing for selection-only / lines-only changes, while still
    recomposing for structural changes (rows added/removed, a different
    item's toolbar).
  * HomeScreen wiring the cache-warm worker on mount.
"""

import threading

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Home.active_work_adapter import LocalNotificationHomeActiveWorkAdapter
from tldw_chatbook.Home.dashboard_state import (
    HomeAction,
    HomeCanvasState,
    HomeDashboardInput,
    HomeRailRow,
    HomeRailSectionState,
    HomeTriageState,
)
from tldw_chatbook.Home.home_rail_state import HomeRailPreferences
from tldw_chatbook.Widgets.Home.home_canvas import HomeCanvas
from tldw_chatbook.Widgets.Home.home_rail import HomeRail

from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_screen_navigation import _build_test_app


# ---------------------------------------------------------------------------
# Adapter-level: short-TTL cache + off-loop threading
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, is_memory_db: bool):
        self.is_memory_db = is_memory_db


class _CountingNotificationService:
    """Records call count + calling thread, mirroring ClientNotificationsService."""

    def __init__(self, *, is_memory_db: bool = False):
        self.store = _FakeStore(is_memory_db)
        self.calls = 0
        self.thread_idents: list[int] = []

    def list_queue(self, *, limit=100, include_dismissed=False, category=None):
        self.calls += 1
        self.thread_idents.append(threading.get_ident())
        return []


def _adapter(notification_service):
    return LocalNotificationHomeActiveWorkAdapter(notification_service=notification_service)


def test_active_work_cache_hits_within_ttl():
    service = _CountingNotificationService()
    adapter = _adapter(service)

    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)
    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)

    assert service.calls == 1  # second call served from the warm cache


def test_active_work_cache_recomputes_after_ttl_expires():
    service = _CountingNotificationService()
    adapter = _adapter(service)

    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)
    assert service.calls == 1

    # Simulate TTL elapsed without a real sleep.
    adapter._active_work_cache_at -= 10.0
    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)
    assert service.calls == 2


def test_invalidate_active_work_cache_forces_recompute():
    service = _CountingNotificationService()
    adapter = _adapter(service)

    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)
    adapter.invalidate_active_work_cache()
    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)

    assert service.calls == 2


@pytest.mark.asyncio
async def test_refresh_active_work_cache_runs_off_loop_for_file_backed_store():
    service = _CountingNotificationService(is_memory_db=False)
    adapter = _adapter(service)
    caller_thread = threading.get_ident()

    await adapter.refresh_active_work_cache_async()

    assert service.calls == 1
    assert service.thread_idents[0] != caller_thread


@pytest.mark.asyncio
async def test_refresh_active_work_cache_stays_inline_for_memory_backed_store():
    """A per-connection ``:memory:`` store must not be queried off-thread.

    ClientNotificationsDB caches a single sqlite connection for
    ``:memory:`` paths rather than opening thread-local ones; sqlite
    defaults to check_same_thread=True, so a background thread would
    raise (not silently see an empty DB) if this guard were missing.
    """
    service = _CountingNotificationService(is_memory_db=True)
    adapter = _adapter(service)
    caller_thread = threading.get_ident()

    await adapter.refresh_active_work_cache_async()

    assert service.calls == 1
    assert service.thread_idents[0] == caller_thread


@pytest.mark.asyncio
async def test_refresh_active_work_cache_warms_cache_for_sync_reads():
    service = _CountingNotificationService()
    adapter = _adapter(service)

    await adapter.refresh_active_work_cache_async()
    assert service.calls == 1

    # A subsequent synchronous build_dashboard_input (the compose-path
    # call, which cannot await) should hit the warmed cache.
    adapter.build_dashboard_input(providers_models={}, has_recent_work=False)
    assert service.calls == 1


# ---------------------------------------------------------------------------
# HomeRail / HomeCanvas: targeted patch vs. recompose
# ---------------------------------------------------------------------------


def _row(row_id, section_id, *, title="Item", glyph="●", age="", source="Src"):
    return HomeRailRow(
        row_id=row_id,
        section_id=section_id,
        glyph=glyph,
        title=title,
        age_label=age,
        source=source,
    )


def _triage(rows_by_section, *, selected_row_id="", details_lines=("Line",)):
    sections = tuple(
        HomeRailSectionState(section_id, section_id.title(), len(rows), tuple(rows), "empty")
        for section_id, rows in rows_by_section.items()
    )
    canvas = HomeCanvasState(
        title="Canvas",
        lines=("line",),
        actions=(),
        next_action=HomeAction(action_id="a", label="A", target_route="chat", reason="r"),
        next_action_is_canvas=False,
    )
    return HomeTriageState(
        header_line="Header",
        sections=sections,
        details_lines=details_lines,
        canvas=canvas,
        selected_row_id=selected_row_id,
    )


class _RailHarness(App):
    def __init__(self, triage: HomeTriageState, preferences: HomeRailPreferences):
        super().__init__()
        self._triage = triage
        self._preferences = preferences

    def compose(self) -> ComposeResult:
        yield HomeRail(self._triage, self._preferences, id="home-rail")


def _spy_recompose(widget):
    """Wrap widget.refresh so recompose=True calls are recorded, real calls still run."""
    calls: list[dict] = []
    original_refresh = widget.refresh

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original_refresh(*args, **kwargs)

    widget.refresh = spy
    return calls


@pytest.mark.asyncio
async def test_home_rail_selection_only_change_patches_without_recompose():
    rows = {
        "attention": [_row("row-a", "attention", title="Alpha")],
        "running": [_row("row-b", "running", title="Beta")],
        "recent": [],
    }
    prefs = HomeRailPreferences()
    triage_a = _triage(rows, selected_row_id="row-a")
    app = _RailHarness(triage_a, prefs)

    async with app.run_test() as pilot:
        rail = app.query_one("#home-rail", HomeRail)
        calls = _spy_recompose(rail)

        triage_b = _triage(rows, selected_row_id="row-b")
        rail.sync_state(triage_b, prefs)
        await pilot.pause()

        assert not any(call.get("recompose") for call in calls)
        buttons = {btn.row_id: btn for btn in rail.query("Button.home-rail-row")}
        assert buttons["row-b"].has_class("home-rail-row-selected")
        assert not buttons["row-a"].has_class("home-rail-row-selected")
        assert "▸" in str(buttons["row-b"].label)
        assert "▸" not in str(buttons["row-a"].label)


@pytest.mark.asyncio
async def test_home_rail_details_only_change_patches_without_recompose():
    rows = {"attention": [_row("row-a", "attention")], "running": [], "recent": []}
    prefs = HomeRailPreferences()
    triage_a = _triage(rows, selected_row_id="row-a", details_lines=("Status: ok",))
    app = _RailHarness(triage_a, prefs)

    async with app.run_test() as pilot:
        rail = app.query_one("#home-rail", HomeRail)
        calls = _spy_recompose(rail)

        triage_b = _triage(rows, selected_row_id="row-a", details_lines=("Status: busy",))
        rail.sync_state(triage_b, prefs)
        await pilot.pause()

        assert not any(call.get("recompose") for call in calls)
        details = rail.query_one("#home-details-body", Static)
        assert "Status: busy" in str(details.renderable)


@pytest.mark.asyncio
async def test_home_rail_structural_change_still_recomposes():
    prefs = HomeRailPreferences()
    rows_a = {"attention": [_row("row-a", "attention")], "running": [], "recent": []}
    rows_b = {
        "attention": [_row("row-a", "attention"), _row("row-c", "attention", title="Gamma")],
        "running": [],
        "recent": [],
    }
    triage_a = _triage(rows_a, selected_row_id="row-a")
    app = _RailHarness(triage_a, prefs)

    async with app.run_test() as pilot:
        rail = app.query_one("#home-rail", HomeRail)
        calls = _spy_recompose(rail)

        triage_b = _triage(rows_b, selected_row_id="row-a")
        rail.sync_state(triage_b, prefs)
        await pilot.pause()

        assert any(call.get("recompose") for call in calls)
        row_ids = {btn.row_id for btn in rail.query("Button.home-rail-row")}
        assert row_ids == {"row-a", "row-c"}


class _CanvasHarness(App):
    def __init__(self, canvas: HomeCanvasState):
        super().__init__()
        self._canvas = canvas

    def compose(self) -> ComposeResult:
        yield HomeCanvas(
            self._canvas,
            action_button_factory=lambda label, control_id, primary=False: Button(
                label, id=control_id
            ),
            id="home-canvas",
        )


def _canvas_state(lines, *, title="Canvas", primary_control_id=""):
    return HomeCanvasState(
        title=title,
        lines=lines,
        actions=(),
        next_action=HomeAction(action_id="a", label="A", target_route="chat", reason="r"),
        next_action_is_canvas=False,
        primary_control_id=primary_control_id,
    )


@pytest.mark.asyncio
async def test_home_canvas_lines_only_change_patches_without_recompose():
    canvas_a = _canvas_state(("line one",))
    app = _CanvasHarness(canvas_a)

    async with app.run_test() as pilot:
        canvas = app.query_one("#home-canvas", HomeCanvas)
        calls = _spy_recompose(canvas)

        canvas_b = _canvas_state(("line two", "line three"))
        canvas.sync_state(canvas_b)
        await pilot.pause()

        assert not any(call.get("recompose") for call in calls)
        lines_widget = canvas.query_one("#home-canvas-lines", Static)
        assert "line two" in str(lines_widget.renderable)
        assert "line one" not in str(lines_widget.renderable)


@pytest.mark.asyncio
async def test_home_canvas_identical_state_is_a_no_op():
    canvas_a = _canvas_state(("line",))
    app = _CanvasHarness(canvas_a)

    async with app.run_test() as pilot:
        canvas = app.query_one("#home-canvas", HomeCanvas)
        calls = _spy_recompose(canvas)

        canvas.sync_state(_canvas_state(("line",)))
        await pilot.pause()

        assert calls == []


@pytest.mark.asyncio
async def test_home_canvas_title_change_still_recomposes():
    canvas_a = _canvas_state(("line",), title="Item A")
    app = _CanvasHarness(canvas_a)

    async with app.run_test() as pilot:
        canvas = app.query_one("#home-canvas", HomeCanvas)
        calls = _spy_recompose(canvas)

        canvas.sync_state(_canvas_state(("line",), title="Item B"))
        await pilot.pause()

        assert any(call.get("recompose") for call in calls)
        title_widget = canvas.query_one("#home-canvas-title", Static)
        assert "Item B" in str(title_widget.renderable)


# ---------------------------------------------------------------------------
# HomeScreen wiring: cache-warm worker fires on mount
# ---------------------------------------------------------------------------


class _AsyncRefreshAdapter:
    """Minimal adapter double implementing only the async refresh hook."""

    def __init__(self):
        self.refresh_calls = 0

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        return HomeDashboardInput(model_ready=True, has_recent_work=has_recent_work)

    def handle_control(self, action, *, target_id=None, target_route=None):
        raise NotImplementedError

    async def refresh_active_work_cache_async(self) -> None:
        self.refresh_calls += 1


@pytest.mark.asyncio
async def test_home_screen_warms_active_work_cache_on_mount():
    app = _build_test_app()
    adapter = _AsyncRefreshAdapter()
    app.home_active_work_adapter = adapter
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.2)
        assert _active_home_screen(host) is not None

    assert adapter.refresh_calls >= 1
