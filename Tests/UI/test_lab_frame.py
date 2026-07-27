"""The shared Lab frame: regions, status row, lazy body, and collapse."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.css.query import QueryError
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
)
from tldw_chatbook.UI.Screens import lab_frame
from tldw_chatbook.UI.Screens.lab_frame import LabScreen, LabStatusChip
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchHeaderState
from Tests.UI.test_screen_navigation import _build_test_app


def _disable_splash_race(monkeypatch) -> None:
    """Neutralise the same splash-screen race documented in
    ``test_lab_frame_mode_keys.py`` and ``test_llm_screen_lab_adoption.py``.

    ``SplashScreen.on_mount`` starts a REAL 1.5s wall-clock ``set_timer``
    whose callback mounts the app's actual default-tab screen regardless of
    what a test has since pushed. The press/pause sequences below (collapse
    and expand, in particular) were observed to occasionally run long enough
    -- when sharing a test session with many other files -- to let that
    timer fire mid-test and steal ``app.screen`` out from under a
    ``pilot.click``. Forcing ``splash_screen.enabled`` False skips the
    splash branch entirely, exactly as the other two files do.

    Deliberately NOT an autouse fixture: applying it file-wide changes
    ``TldwCli.compose()``'s startup timing enough that
    ``test_the_body_is_absent_at_first_paint_and_present_after_deferral``
    began failing every run -- disabling splash removed enough intervening
    work that ``call_after_refresh``'s deferred body mount completed before
    that test's very first, no-pause assertion. Only the tests that actually
    press buttons call this.

    Args:
        monkeypatch: The caller's ``monkeypatch`` fixture; reverts the patch
            automatically at the end of that test.
    """

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


class _ProbeBody(Static):
    """Stands in for a mode's expensive legacy window."""


class _ProbeLabScreen(LabScreen):
    """A minimal Lab mode used to exercise the frame itself."""

    def __init__(self, app_instance, *, chips=(), **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self._chips = chips
        self.body_ready_calls = 0

    def lab_header_state(self) -> WorkbenchHeaderState:
        return WorkbenchHeaderState(title="Probe", subtitle="probe mode")

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        return self._chips

    def compose_lab_rail(self) -> ComposeResult:
        yield Static("rail row", id="probe-rail-row")

    def compose_lab_inspector(self) -> ComposeResult:
        yield Static("inspector row", id="probe-inspector-row")

    def build_lab_body(self) -> Widget:
        return _ProbeBody("body", id="probe-body")

    def on_lab_body_ready(self) -> None:
        self.body_ready_calls += 1


def _mount(screen_factory):
    """Build the test app and this probe screen.

    Deliberately does NOT load the CSS bundle: every assertion in this file
    is about behaviour and class membership, not rendered styling. Setting
    `app.CSS_PATH` after construction would be worse than useless anyway --
    `App.__init__` reads `css_path or self.CSS_PATH` once, at construction,
    so a post-hoc assignment silently does nothing. Styling assertions live
    in test_lab_workbench.py, which uses a class-level CSS_PATH.
    """
    app = _build_test_app()
    return app, screen_factory(app)


@pytest.mark.asyncio
async def test_the_body_is_absent_at_first_paint_and_present_after_deferral():
    """The lazy mount is the whole performance claim -- assert it directly.

    Without this, a frame that mounted the body inline would pass every
    other test in this file.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        assert not screen.query(_ProbeBody), "body mounted during first paint"
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one(_ProbeBody) is not None
        assert screen.body_ready_calls == 1


@pytest.mark.asyncio
async def test_mount_lab_body_is_a_silent_no_op_before_the_screen_is_mounted():
    """The deferred callback firing after the screen was torn down (the user
    navigated away during the deferral window) is a normal race, not the
    composition bug the next test checks for, and must not raise."""
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    assert not screen.is_mounted
    screen._mount_lab_body()
    assert screen.body_ready_calls == 0


@pytest.mark.asyncio
async def test_mount_lab_body_raises_when_the_region_is_missing_on_a_mounted_screen():
    """A missing `#lab-body` while the screen itself is alive means the
    frame's own composition is broken -- exactly the defect class a
    swallowed QueryError let ship as a permanently blank screen. Removing
    the region after mount reproduces "mounted but missing" without a
    second probe screen whose compose_content skips the workbench."""
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await screen.query_one("#lab-body").remove()
        await pilot.pause()

        with pytest.raises(QueryError):
            screen._mount_lab_body()


@pytest.mark.asyncio
async def test_a_mode_with_no_chips_renders_no_status_row_at_all():
    """A mode without status must not reserve a row of dead chrome.

    Models always supplies a chip, so this path has no real consumer until
    Speech and Evals adopt -- it would otherwise ship unexercised.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a, chips=()))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert not screen.query("#lab-status-row")


@pytest.mark.asyncio
async def test_chips_render_and_refresh_mutates_them_without_recomposing():
    """Refresh must update the same Static, not replace the row."""
    chips = [LabStatusChip(chip_id="servers", text="Servers: none running")]

    class _Screen(_ProbeLabScreen):
        def lab_status_chips(self):
            return tuple(chips)

    app, screen = _mount(lambda a: _Screen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        chip_widget = screen.query_one("#lab-status-chip-servers", Static)
        assert "Servers: none running" in str(chip_widget.renderable)

        chips[0] = LabStatusChip(chip_id="servers", text="Servers: 2 running")
        screen.refresh_lab_status()
        await pilot.pause()

        assert screen.query_one("#lab-status-chip-servers", Static) is chip_widget
        assert "Servers: 2 running" in str(chip_widget.renderable)


@pytest.mark.asyncio
async def test_the_frame_wires_its_route_into_the_mode_strip():
    """The existing strip suite mounts the strip standalone and would not
    notice the frame passing the wrong active_route."""
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        active = screen.query_one("#lab-mode-models")
        assert "is-active" in active.classes


@pytest.mark.asyncio
async def test_the_inspector_starts_collapsed_on_first_run():
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.rail_layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
        assert screen.query_one("#lab-inspector-handle") is not None


@pytest.mark.asyncio
async def test_pressing_the_inspector_handle_expands_it_and_shows_its_content(
    fake_rail_store, monkeypatch
):
    """Drive the actual Button.Pressed path, not `toggle_lab_rail` directly.

    Every other test in this module that exercises collapse calls
    `toggle_lab_rail` straight from Python -- exactly why the collapse
    handles shipped with no `@on` wiring behind them at all: calling the
    method directly can never notice a missing button handler.
    """
    _disable_splash_race(monkeypatch)
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        # First-run default: the inspector starts collapsed.
        assert screen.query_one("#lab-inspector").display is False
        assert screen.query_one("#lab-inspector-handle").display is True

        await pilot.click("#lab-inspector-open")
        await pilot.pause()

        assert screen.query_one("#lab-inspector").display is True
        assert screen.query_one("#lab-inspector-handle").display is False
        assert screen.query_one("#probe-inspector-row") is not None
        assert fake_rail_store, "expanding via the button did not persist the layout"


@pytest.mark.asyncio
async def test_pressing_the_inspector_collapse_button_collapses_it_again(
    fake_rail_store, monkeypatch
):
    """The collapse direction needs its own affordance, not just the handle.

    Before this fix there was no button in the expanded direction at all.
    """
    _disable_splash_race(monkeypatch)
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        await pilot.click("#lab-inspector-open")
        await pilot.pause()
        assert screen.query_one("#lab-inspector").display is True

        await pilot.click("#lab-inspector-collapse")
        await pilot.pause()

        assert screen.query_one("#lab-inspector").display is False
        assert screen.query_one("#lab-inspector-handle").display is True


@pytest.mark.asyncio
async def test_pressing_the_rail_collapse_button_collapses_the_catalog_rail(
    fake_rail_store, monkeypatch
):
    """The catalog rail starts expanded; only the collapse button is untested
    without this -- `#lab-rail-open` is exercised by the toggle test above."""
    _disable_splash_race(monkeypatch)
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one("#lab-rail").display is True
        assert screen.query_one("#lab-rail-handle").display is False

        await pilot.click("#lab-rail-collapse")
        await pilot.pause()

        assert screen.query_one("#lab-rail").display is False
        assert screen.query_one("#lab-rail-handle").display is True

        await pilot.click("#lab-rail-open")
        await pilot.pause()

        assert screen.query_one("#lab-rail").display is True
        assert screen.query_one("#lab-rail-handle").display is False


@pytest.fixture
def fake_rail_store(monkeypatch):
    """Prevent `toggle_lab_rail` from writing the user's real config.

    Patches the names `lab_frame` imported into its own module namespace,
    mirroring how `Tests/UI/test_lab_rail_store.py` patches the config
    accessors `lab_rail_store` imported. Pytest's autouse isolation fixture
    (`Tests/conftest.py`) already redirects config reads/writes to a temp
    XDG/HOME sandbox, but this monkeypatch belt-and-suspenders that: it
    proves `toggle_lab_rail` never even reaches the config layer, and it
    captures each persisted layout for assertions.
    """
    saved = []
    monkeypatch.setattr(lab_frame, "save_rail_layout", saved.append)
    return saved


@pytest.mark.asyncio
async def test_toggling_a_rail_preserves_the_mounted_mode_content(fake_rail_store):
    """Regression test: a rail toggle must not blow away mounted content.

    `toggle_lab_rail` used to end with `self.refresh(recompose=True)`,
    which rebuilds `compose_content()` -- including a brand-new
    `LabWorkbench` with fresh, empty regions -- without re-firing
    `on_mount()`. Since `_populate_regions()`/`_mount_lab_body()` only ever
    run from `on_mount()`, the mode's rail row and deferred body vanished
    for the life of the screen after any toggle. The fix applies the new
    layout to the existing workbench in place instead of recomposing.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one(_ProbeBody) is not None
        assert screen.query_one("#probe-rail-row") is not None
        assert screen.query_one("#lab-rail").display is True
        assert screen.query_one("#lab-rail-handle").display is False

        screen.toggle_lab_rail(LAB_RAIL_LEFT)
        await pilot.pause()

        assert screen.query_one(_ProbeBody) is not None, "body lost after toggle"
        assert (
            screen.query_one("#probe-rail-row") is not None
        ), "rail content lost after toggle"
        assert screen.query_one("#lab-rail").display is False
        assert screen.query_one("#lab-rail-handle").display is True

        screen.toggle_lab_rail(LAB_RAIL_LEFT)
        await pilot.pause()

        assert screen.query_one(_ProbeBody) is not None
        assert screen.query_one("#probe-rail-row") is not None
        assert screen.query_one("#lab-rail").display is True
        assert screen.query_one("#lab-rail-handle").display is False

        assert fake_rail_store, "toggle_lab_rail did not persist the new layout"
