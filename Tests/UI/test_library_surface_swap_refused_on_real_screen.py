"""The Library's surface-swap seams refuse to be awaited on a pump they remove.

TASK-34000.4, AC#3, on the real Library screen. ``Tests/Architecture/
test_surface_swap_guard.py`` pins the predicate against a minimal app; this
file pins where the Library wires it, at every seam that awaits the removal
of the canvas host's children or of the screen's own children:

* the open-surface seam (``_apply_library_open_item_surface``) -- the one the
  freeze went through. Its structural branch awaits ``Screen.recompose``, and
  its targeted branch awaits the canvas-child swap. Either refusal becomes the
  seam's documented fallback, a whole-screen refresh on the screen's own
  pump, logged at ERROR naming the offending widget: the surface still
  arrives, and the bug is still loud;
* the canvas-child swap itself (``_replace_library_canvas_child``), the
  entry repair loop (``_repair_library_entry_canvas_owner``) and the snapshot
  reconcile (``_reconcile_library_entry_state``): refused by name, before
  anything is hidden or torn down, with the same widgets still mounted.

Not in the PR UI lane (``scripts/ui_pr_gate_census.txt``): each test boots the
real Library, and the AC#4 regression in
``test_library_media_export_no_freeze.py`` already spends that lane's budget.
Every wait is wall-clock bounded; a red run names and frees the parked pump
(``Tests/UI/pump_probe.py``).
"""

from __future__ import annotations

import pytest
from textual.widgets import Input

from Tests.UI.pump_probe import (
    describe_parked,
    free_parked_pumps,
    parked_pumps,
    until,
)
from Tests.UI.test_library_media_export_no_freeze import (
    OPEN_BUDGET_SECONDS,
    SCENARIO_TIMEOUT_SECONDS,
    _assert_nothing_is_parked,
    _enter_media_list,
    _export_canvas_is_up,
    _media_host,
    _media_list_is_up,
)
from Tests.UI.test_library_shell import LIBRARY_TEST_SIZE
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_INGEST_EXPORT,
)
from tldw_chatbook.UI.Navigation.surface_swap_guard import SurfaceSwapSelfAwaitError

pytestmark = pytest.mark.bootstrap_profile

REFUSAL_LOG_MARK = "Surface swap refused"


async def _export_open_as_the_review_ran_it(screen) -> None:
    """``_open_library_export_canvas``'s body at dev 2d34cbf80d, tail included:
    move the destination to Export, then AWAIT the projection inline."""
    screen._export_state.origin_row_id = screen._library_selected_row_id
    screen._set_library_destination_with_conversation_fence(LIBRARY_ROW_INGEST_EXPORT)
    screen._reset_library_export_transient_state(ExportScope(kind="media"))
    await screen._project_library_export_canvas()


def _run_on_pump(widget, operation, seen: dict) -> None:
    """Await ``operation()`` on ``widget``'s own message pump -- where a
    handler on that widget runs -- and record how it ended."""

    async def on_the_widgets_pump() -> None:
        try:
            seen["result"] = await operation()
        except SurfaceSwapSelfAwaitError as error:
            seen["refused"] = error
        seen["ended"] = True

    widget.call_later(on_the_widgets_pump)


async def _ended_or_name_the_parked(host, seen: dict, what: str) -> None:
    assert await until(lambda: "ended" in seen, OPEN_BUDGET_SECONDS), (
        f"{what} neither returned nor was refused; parked pumps: "
        f"{describe_parked(await parked_pumps(host, 0.5))}"
    )


def _refusal_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if REFUSAL_LOG_MARK in line]


@pytest.mark.timeout(SCENARIO_TIMEOUT_SECONDS)
@pytest.mark.asyncio
async def test_the_export_open_awaited_on_the_canvas_pump_lands_by_fallback(
    captured_lines,
) -> None:
    """The seam the freeze went through, awaited the way the review ran it.

    ``Screen.recompose`` refuses it while the Media canvas is still mounted;
    the open-surface seam takes its whole-screen fallback on the screen's own
    pump, so Export still opens, the app keeps answering, and the log names
    the canvas whose pump awaited its own removal.
    """
    host = _media_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media_list(host, pilot)
        canvas = screen.query_one("#library-media-canvas")
        seen: dict = {}
        passed = False
        try:
            _run_on_pump(
                canvas, lambda: _export_open_as_the_review_ran_it(screen), seen
            )
            await _ended_or_name_the_parked(
                host, seen, "an Export open awaited on the Media canvas's pump"
            )
            assert "refused" not in seen, (
                f"the open-surface seam let the refusal escape: {seen['refused']}"
            )
            refusals = _refusal_lines(captured_lines)
            assert len(refusals) == 1, captured_lines
            assert "LibraryScreen.recompose" in refusals[0], refusals[0]
            assert "LibraryMediaCanvas" in refusals[0], refusals[0]

            assert await until(
                lambda: _export_canvas_is_up(screen), OPEN_BUDGET_SECONDS
            ), "the fallback never landed Export"
            assert not canvas.is_attached, (
                "the fallback is a whole-screen rebuild; the old canvas is gone"
            )
            await _assert_nothing_is_parked(host, screen, "after the fallback")
            passed = True
        finally:
            if not passed:
                for line in await free_parked_pumps(host):
                    print(f"\nfreed for tear-down: {line}")


@pytest.mark.timeout(SCENARIO_TIMEOUT_SECONDS)
@pytest.mark.asyncio
async def test_a_canvas_child_swap_awaited_inside_the_canvas_is_refused_not_hung(
    captured_lines,
) -> None:
    """The region-scoped swap: one canvas host's children.

    A widget INSIDE the outgoing canvas awaiting that swap waits on its own
    removal exactly as the recompose case does. The child-swap seam refuses
    it with nothing torn down; through the open-surface seam the refusal
    becomes that seam's fallback and the surface still arrives.
    """
    host = _media_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media_list(host, pilot)
        canvas = screen.query_one("#library-media-canvas")
        nested = canvas.query_one("#library-media-filter", Input)
        seen: dict = {}
        passed = False
        try:
            _run_on_pump(
                nested,
                lambda: screen._replace_library_canvas_child(
                    screen._build_library_entry_active_child(),
                    generation=screen._library_snapshot_state_generation,
                    route_key=screen._library_entry_route_key(),
                ),
                seen,
            )
            await _ended_or_name_the_parked(
                host, seen, "a canvas-child swap awaited inside the outgoing canvas"
            )
            assert "refused" in seen, (
                "the child-swap seam let the removal be awaited on the pump of "
                f"a widget below an outgoing child (ended with {seen!r})"
            )
            assert "library-media-filter" in str(seen["refused"])
            assert screen.query_one("#library-media-canvas") is canvas
            assert canvas.is_attached and nested.is_attached
            assert canvas.display, "refused before the outgoing child was hidden"
            await _assert_nothing_is_parked(host, screen, "after the refusal")
            assert _refusal_lines(captured_lines) == [], (
                "the raw seam raises; only the open-surface seam logs a fallback"
            )

            seen.clear()
            _run_on_pump(
                nested,
                lambda: screen._apply_library_open_item_surface(
                    screen._build_library_entry_active_child
                ),
                seen,
            )
            await _ended_or_name_the_parked(
                host, seen, "the open-surface seam awaited inside the outgoing canvas"
            )
            assert "refused" not in seen, seen
            refusals = _refusal_lines(captured_lines)
            assert len(refusals) == 1 and "library-media-filter" in refusals[0], (
                captured_lines
            )
            assert await until(
                lambda: (
                    _media_list_is_up(screen)
                    and screen.query_one("#library-media-canvas") is not canvas
                ),
                OPEN_BUDGET_SECONDS,
            ), "the fallback recompose never repainted the Media list"
            await _assert_nothing_is_parked(host, screen, "after the fallback")
            passed = True
        finally:
            if not passed:
                for line in await free_parked_pumps(host):
                    print(f"\nfreed for tear-down: {line}")


def _aim_at_conversations(screen) -> tuple[int, tuple[object, ...]]:
    """Make the Conversations canvas the destination while the Media canvas
    is still the mounted owner, so the entry seams must replace it."""
    screen._set_library_destination_with_conversation_fence(
        LIBRARY_ROW_BROWSE_CONVERSATIONS
    )
    screen._library_entry_reconcile_dirty = True
    return screen._library_snapshot_state_generation, screen._library_entry_route_key()


@pytest.mark.timeout(SCENARIO_TIMEOUT_SECONDS)
@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["repair", "reconcile"])
async def test_the_entry_lifecycle_seams_refuse_a_swap_awaited_inside_the_canvas(
    seam: str,
) -> None:
    """The repair loop and the snapshot reconcile both await the removal of
    the canvas host's children. Awaited from a widget inside the Media
    canvas they are refused by name, not counted as a failed attempt and
    retried, and the Media canvas is still mounted and visible afterwards.
    """
    host = _media_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media_list(host, pilot)
        canvas = screen.query_one("#library-media-canvas")
        nested = canvas.query_one("#library-media-filter", Input)
        seen: dict = {}
        passed = False
        try:
            generation, route_key = _aim_at_conversations(screen)
            if seam == "repair":
                operation = screen._repair_library_entry_canvas_owner
            else:

                def operation():
                    return screen._reconcile_library_entry_state(generation, route_key)

            _run_on_pump(nested, operation, seen)
            await _ended_or_name_the_parked(
                host, seen, f"the {seam} seam awaited inside the outgoing canvas"
            )
            assert "refused" in seen, (
                f"the {seam} seam let the removal be awaited on the pump of a "
                f"widget below an outgoing child (ended with {seen!r})"
            )
            assert "library-media-filter" in str(seen["refused"])
            assert screen.query_one("#library-media-canvas") is canvas
            assert canvas.is_attached and canvas.display and nested.is_attached
            await _assert_nothing_is_parked(host, screen, "after the refusal")
            passed = True
        finally:
            if not passed:
                for line in await free_parked_pumps(host):
                    print(f"\nfreed for tear-down: {line}")
