"""DestinationHeader's height-triggered compact density (task-31825).

task-31419 measured the destination header block as 5 of 24 rows at the
80x24 floor, and found the widget already ships a `density="compact"` CSS
rule (`.density-compact .workbench-header`) that no caller ever triggers.
This wires a height-based trigger at the shared `BaseAppScreen` layer
(`VERTICAL_BREAKPOINTS`, native Textual mechanism) instead of reusing that
exact class, because `.density-compact` is also the user's own global
density preference and, on Console, an inner `#console-shell` wrapper it
manages itself -- colliding the new automatic trigger onto that class name
would let the two signals fight over the same selector. The new marker is
`shell-header-compact` / `shell-header-normal`
(`components/_workbench.tcss`).

`CSS_PATH = APP_STYLESHEETS` throughout: the `.workbench-header` /
`.shell-header-compact` rules live in the app CSS tier (the bundle), not in
any widget's `BUNDLED_CSS` -- a bare `ConsolidatedCSSApp` without that
`CSS_PATH` override measures an unstyled header (verified empirically while
writing this file: Study's header read height=3 with the subtitle always
visible under the bare harness, and only dropped to the real 5/4-row
behavior once `APP_STYLESHEETS` was on `CSS_PATH`, exactly like
`test_schedules_responsive_floor.py`'s own `BundledCSSWorkbenchApp`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.study_screen import StudyScreen

#: Same floor the responsive-floor suite pins, and the exact threshold
#: `BaseAppScreen._DESTINATION_HEADER_COMPACT_FLOOR_HEIGHT` uses.
FLOOR = (80, 24)
#: One row above the floor: the first height that must NOT be compact.
JUST_ABOVE_FLOOR = (80, 25)
#: A standard size, far from the threshold either way.
STANDARD = (120, 40)


class _SchedulesApp(ConsolidatedCSSApp):
    """Same harness `test_schedules_responsive_floor.py` uses."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]
    scheduling_service = None


def _schedules_app(tmp_path):
    app = _SchedulesApp()
    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    service = SchedulingService(db=db, runtime_source="local", app_getter=lambda: app)
    app.scheduling_service = service
    return app


async def _open_schedules(pilot):
    await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
    await pilot.pause()
    return pilot.app.screen


class _StudyApp(ConsolidatedCSSApp):
    """Study, the AC#2 second beneficiary -- no per-screen wiring of its own."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self) -> None:
        super().__init__()
        self._app_instance = SimpleNamespace(
            current_runtime_backend="local",
            runtime_backend=None,
            app_config={},
            notify=lambda *args, **kwargs: None,
            pending_handoffs=PendingHandoffStore(),
        )

    async def on_mount(self) -> None:
        await self.push_screen(StudyScreen(app_instance=self._app_instance))


# ---------------------------------------------------------------------------
# AC#1 / AC#3: compact at/below the floor, normal (pinned) at standard sizes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destination_header_is_compact_at_the_80x24_floor(tmp_path):
    """At the floor the header drops its subtitle row: 5 -> 4 of 24 rows."""
    app = _schedules_app(tmp_path)
    async with app.run_test(size=FLOOR) as pilot:
        screen = await _open_schedules(pilot)

        assert screen.has_class("shell-header-compact")
        header = screen.query_one("#schedules-destination-header")
        assert header.region.height == 4
        subtitle = screen.query_one("#workbench-header-subtitle")
        assert subtitle.styles.display == "none"


@pytest.mark.asyncio
async def test_destination_header_is_unchanged_at_a_standard_size(tmp_path):
    """Pin: normal density at a standard size is byte-identical to today."""
    app = _schedules_app(tmp_path)
    async with app.run_test(size=STANDARD) as pilot:
        screen = await _open_schedules(pilot)

        assert screen.has_class("shell-header-normal")
        header = screen.query_one("#schedules-destination-header")
        assert header.region.height == 5
        subtitle = screen.query_one("#workbench-header-subtitle")
        assert subtitle.styles.display == "block"


@pytest.mark.asyncio
async def test_destination_header_floor_threshold_is_exactly_24(tmp_path):
    """One row above the pinned floor must already be normal density."""
    app = _schedules_app(tmp_path)
    async with app.run_test(size=JUST_ABOVE_FLOOR) as pilot:
        screen = await _open_schedules(pilot)

        assert screen.has_class("shell-header-normal")
        header = screen.query_one("#schedules-destination-header")
        assert header.region.height == 5


# ---------------------------------------------------------------------------
# First-paint correctness: compact before any resize event fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destination_header_is_compact_from_first_paint(tmp_path):
    """A screen mounted directly at the floor is compact on its first
    layout pass -- `pilot.resize_terminal` is never called here, so this
    exercises Textual's own `_on_resize` firing from the screen's initial
    size assignment, not a later resize."""
    app = _schedules_app(tmp_path)
    async with app.run_test(size=FLOOR) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        # No pilot.pause() at all: the very next frame after push must
        # already carry the compact class.
        screen = pilot.app.screen
        assert screen.has_class("shell-header-compact")


# ---------------------------------------------------------------------------
# AC#2: a second, unmodified workbench screen benefits with no screen code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destination_header_is_compact_on_study_too():
    """Study never references `shell-header-compact`, `VERTICAL_BREAKPOINTS`,
    or density anywhere in its own source -- this is the shared-layer
    wiring alone."""
    app = _StudyApp()
    async with app.run_test(size=FLOOR) as pilot:
        await pilot.pause(0.1)
        screen = pilot.app.screen

        assert screen.has_class("shell-header-compact")
        header = screen.query_one("#study-destination-header")
        assert header.region.height == 4
        subtitle = screen.query_one(
            "#study-destination-header #workbench-header-subtitle"
        )
        assert subtitle.styles.display == "none"


@pytest.mark.asyncio
async def test_destination_header_is_normal_on_study_at_a_standard_size():
    app = _StudyApp()
    async with app.run_test(size=STANDARD) as pilot:
        await pilot.pause(0.1)
        screen = pilot.app.screen

        assert screen.has_class("shell-header-normal")
        header = screen.query_one("#study-destination-header")
        assert header.region.height == 5


# ---------------------------------------------------------------------------
# Hysteresis: repeated crossings settle cleanly, no leaked children/classes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shell_header_compact_toggles_cleanly_across_repeated_resizes(tmp_path):
    app = _schedules_app(tmp_path)
    async with app.run_test(size=FLOOR) as pilot:
        screen = await _open_schedules(pilot)
        header = screen.query_one("#schedules-destination-header")

        for step in range(6):
            size = FLOOR if step % 2 == 0 else STANDARD
            await pilot.resize_terminal(*size)
            expected_class = "shell-header-compact" if size == FLOOR else "shell-header-normal"
            other_class = "shell-header-normal" if size == FLOOR else "shell-header-compact"
            expected_height = 4 if size == FLOOR else 5

            assert screen.has_class(expected_class), (step, size, screen.classes)
            assert not screen.has_class(other_class), (step, size, screen.classes)
            assert header.region.height == expected_height, (step, size)
            # No accumulated/duplicate children from repeated syncs.
            assert len(header.children) == 3
