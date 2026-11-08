"""Settings updates one region at a time, not the whole screen (task-15475).

Three defects from the 2026-08-11 input-latency audit, pinned here:

1. ``active_category`` was a screen-level ``recompose=True`` reactive, so every
   rail click rebuilt the nav bar, the footer, the category rail itself and
   the mode strip alongside the 60-150-widget detail pane.
2. The two sync-row reactives were screen-level ``recompose=True`` too, so
   landing on Overview recomposed the screen a SECOND time when the sync-rows
   worker reported, and "Sync preview"/"Run" each recomposed the whole screen
   to change three status lines.
3. ``_queue_sync_rows_refresh`` ran on both ``on_mount`` and the mount's OWN
   ``on_screen_resume`` (Textual posts ``ScreenResume`` when a screen is
   pushed), so the first visit paid for two full sync previews.

Identity is the evidence: a widget that survives is the same Python object.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

pytestmark = pytest.mark.asyncio

#: Chrome and rail: nothing here reads the active category's CONTENT.
_STABLE = (
    "#settings-shell",
    "#settings-workbench",
    "#settings-category-pane",
    "#settings-category-list",
    "#settings-category-search",
    "#settings-category-overview",
    "#settings-category-storage",
    "#settings-detail-pane",
    "#settings-impact-pane",
    "#settings-focus-help",
    "#screen-footer-status",
)


def _identities(screen, selectors) -> dict[str, int]:
    return {selector: id(screen.query_one(selector)) for selector in selectors}


def _text(widget) -> str:
    return str(getattr(widget.renderable, "plain", widget.renderable))


async def _settle(pilot) -> None:
    await pilot.app.workers.wait_for_complete()
    for _ in range(4):
        await pilot.pause()


async def _open_settings(pilot):
    await _settle(pilot)
    return _active_destination_screen(pilot.app)


@private_profile_test
async def test_category_switch_rebuilds_only_the_two_category_panes(request):
    """AC#1: a rail click leaves every widget that does not read the
    category's content exactly where it was."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        before = _identities(screen, _STABLE)
        nav_before = id(screen.query_one(MainNavigationBar))

        screen.query_one("#settings-category-storage", Button).focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)

        assert _identities(screen, _STABLE) == before, (
            "A Settings category switch rebuilt chrome that does not read the category."
        )
        assert id(screen.query_one(MainNavigationBar)) == nav_before
        # ...and the panes it DOES own actually repainted.
        assert screen.active_category == "storage"
        assert "Storage" in _text(screen.query_one("#settings-category-label", Static))


@private_profile_test
async def test_category_switch_moves_the_active_rail_marker(request):
    """The rail is no longer rebuilt, so the active marker has to be patched
    onto the surviving buttons -- with the same result as a fresh compose."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        assert screen.query_one("#settings-category-overview", Button).has_class(
            "settings-active-section"
        )

        screen.query_one("#settings-category-storage", Button).focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)

        assert screen.query_one("#settings-category-storage", Button).has_class(
            "settings-active-section"
        )
        assert not screen.query_one("#settings-category-overview", Button).has_class(
            "settings-active-section"
        )


@private_profile_test
async def test_category_switch_lands_focus_on_the_selected_rail_button(request):
    """What the recompose provided via `_pending_category_focus_value`: after
    a switch, focus is on the newly selected category button."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)

        screen.query_one("#settings-category-storage", Button).focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)

        assert screen.active_category == "storage"
        assert getattr(host.focused, "id", None) == "settings-category-storage"


@private_profile_test
@pytest.mark.parametrize("move", ("none", "pagedown", "key", "click"))
async def test_a_focus_the_user_moves_before_the_swap_lands_is_kept(request, move):
    """TASK-39135: a switch records a pending rail focus that its pane swap
    applies once the panes are rebuilt. A control the user focused in between,
    with a real key or click, kept focus only if the swap was quicker. The swap
    is held on its lock, so the input lands in that window on every run. The
    switch is the programmatic one a deep link makes, with focus on another
    rail row, so the landing has to move focus. Input that leaves focus where
    it was (PageDown) must not cost the landing (review r12s-rev1 I1)."""
    import asyncio

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        screen.query_one("#settings-category-storage", Button).focus()
        await _settle(pilot)

        moved = move in ("key", "click")
        await screen._category_swap_lock.acquire()
        try:
            screen._select_category("appearance", restore_focus=True)
            await pilot.pause()
            assert screen._category_pane_swap_pending
            if move == "pagedown":
                await pilot.press("pagedown")
            elif move == "key":
                await pilot.press("slash")
            elif move == "click":
                await pilot.click("#settings-category-search")
            assert getattr(host.focused, "id", None) == (
                "settings-category-search" if moved else "settings-category-storage"
            )
        finally:
            screen._category_swap_lock.release()
        for _ in range(100):
            if not screen._category_pane_swap_pending:
                break
            await pilot.pause(0.05)
        assert not screen._category_pane_swap_pending
        # The swap's own focus goes through the app's call_later queue; this
        # callback queues behind it.
        landed = asyncio.Event()
        host.call_later(landed.set)
        await asyncio.wait_for(landed.wait(), timeout=5)

        assert screen.active_category == "appearance"
        assert getattr(host.focused, "id", None) == (
            "settings-category-search" if moved else "settings-category-appearance"
        )


async def _settle_swap(host, pilot, screen) -> None:
    """Wait for the swap, then for its own focus (an app call_later)."""
    import asyncio

    for _ in range(100):
        if not screen._category_pane_swap_pending:
            break
        await pilot.pause(0.05)
    assert not screen._category_pane_swap_pending
    landed = asyncio.Event()
    host.call_later(landed.set)
    await asyncio.wait_for(landed.wait(), timeout=5)


@private_profile_test
async def test_a_focus_the_user_moves_while_the_panes_rebuild_is_kept(request):
    """TASK-39135 review r12s-rev2 I1': input keeps running while the swap
    awaits the panes' rebuild (81-1,348 ms measured), so the check at swap
    start alone missed a move made then. The detail pane's rebuild is held,
    "/" moves focus to the rail filter, and the filter keeps it."""
    import asyncio

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        screen.query_one("#settings-category-storage", Button).focus()
        await _settle(pilot)
        pane = screen.query_one("#settings-detail-pane")
        entered, go = asyncio.Event(), asyncio.Event()
        real_recompose = pane.recompose

        async def held_recompose():
            entered.set()
            await go.wait()
            return await real_recompose()

        pane.recompose = held_recompose
        try:
            screen._select_category("appearance", restore_focus=True)
            await asyncio.wait_for(entered.wait(), timeout=10)
            await pilot.press("slash")
            assert getattr(host.focused, "id", None) == "settings-category-search"
        finally:
            go.set()
        await _settle_swap(host, pilot, screen)

        assert screen.active_category == "appearance"
        assert getattr(host.focused, "id", None) == "settings-category-search"


@private_profile_test
async def test_a_tab_into_the_panes_the_swap_destroys_keeps_the_landing(request):
    """TASK-39135 review r12s-rev2 M-a: Overview's "Open Providers & Models"
    sits in the detail pane the switch rebuilds. A Tab mid-swap moved focus to
    the next button there, the swap destroyed it, and Textual's focus reset
    left the user on the Domain Defaults group toggle (where Space collapses
    the group). A move into the rebuilt panes is not the user's to keep, so
    the switch still lands on its rail row."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        assert screen.active_category == "overview"
        opener = screen.query_one("#settings-overview-open-providers-models", Button)
        opener.focus()
        await _settle(pilot)

        await screen._category_swap_lock.acquire()
        try:
            await pilot.press("enter")
            assert screen._category_pane_swap_pending
            await pilot.press("tab")
            assert host.focused is not opener
        finally:
            screen._category_swap_lock.release()
        await _settle_swap(host, pilot, screen)

        assert screen.active_category == "providers-models"
        assert getattr(host.focused, "id", None) == "settings-category-providers-models"


@private_profile_test
async def test_category_switch_keeps_compact_pane_classes_at_a_compact_size(request):
    """Compact geometry is only measured at a compact size.

    The compact classes used to be re-applied by `compose_content` on every
    category recompose. The panes are not recomposed by the screen any more,
    so the classes have to SURVIVE a switch instead -- and a pane that lost
    them would only be wrong below the 100-column breakpoint, where no
    170x48 test looks.
    """
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open_settings(pilot)
        assert screen._workbench_compact is True, "harness is not at a compact width"
        for pane in ("#settings-detail-pane", "#settings-impact-pane"):
            assert screen.query_one(pane).has_class("settings-workbench-compact-pane")

        screen._select_category("storage")
        await _settle(pilot)

        assert screen.active_category == "storage"
        for pane in ("#settings-detail-pane", "#settings-impact-pane"):
            assert screen.query_one(pane).has_class(
                "settings-workbench-compact-pane"
            ), f"{pane} lost its compact class across a category switch"
        assert screen.query("#settings-storage-card"), (
            "the new category did not render at a compact size"
        )


@private_profile_test
async def test_same_category_rebuild_keeps_focus_on_the_control_used(request):
    """A SAME-category pane rebuild must not throw focus into the rail.

    Workspaces rebuilds its pane in place (row click, "Show archived"), which
    destroys the control under the user. Textual's `_reset_focus` then lands
    them on the rail's Domain Defaults GROUP TOGGLE -- where the next Space
    collapses the group they are not even looking at. Pane control ids are
    stable across a same-category rebuild, so the swap restores the identity.
    """
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_settings(pilot)
        screen._select_category("workspaces")
        await _settle(pilot)
        assert screen.active_category == "workspaces"

        toggle = screen.query_one("#settings-workspaces-show-archived")
        toggle.focus()
        await pilot.pause()
        assert getattr(host.focused, "id", None) == "settings-workspaces-show-archived"

        # The real in-place rebuild path.
        screen._refresh_settings_workspaces_pane()
        await _settle(pilot)

        assert getattr(host.focused, "id", None) == (
            "settings-workspaces-show-archived"
        ), "focus escaped the pane on a same-category rebuild"


def _sync_row_texts(screen) -> list[str]:
    """The detail rows inside the two sync-row regions, in order.

    Read from the REGIONS, not from the whole screen: these are the widgets
    only `_sync_overview_sync_widgets`' region rebuild writes. Asserting on
    screen-wide text (or on the front-door summary Static alone) passes even
    with the rebuild removed, because a different path already keeps the
    summary current -- which is exactly how a neutered rebuild survived a
    first draft of this test.
    """
    rows: list[str] = []
    for region_id in (
        "#settings-overview-manual-sync-rows",
        "#settings-overview-handoff-rows",
    ):
        rows.extend(_text(row) for row in screen.query_one(region_id).query(Static))
    return rows


@private_profile_test
async def test_sync_row_refresh_patches_rows_without_rebuilding_the_screen(request):
    """AC#1: the sync rows update in place; nothing outside them moves."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        assert screen.active_category == "overview"
        before = _identities(screen, _STABLE)
        nav_before = id(screen.query_one(MainNavigationBar))
        card_before = id(screen.query_one("#settings-overview-card"))
        rows_before = _sync_row_texts(screen)
        assert rows_before, "the sync-row regions rendered nothing to begin with"

        screen.manual_sync_rows = (
            ("Manual sync status", "ready"),
            ("Manual sync preview", "Nothing pending."),
            ("Pending outgoing", "none"),
        )
        await _settle(pilot)

        assert _identities(screen, _STABLE) == before, (
            "A sync-rows refresh recomposed the screen."
        )
        assert id(screen.query_one(MainNavigationBar)) == nav_before
        assert id(screen.query_one("#settings-overview-card")) == card_before, (
            "The sync rows live in their own container; the Overview card "
            "around them must not be rebuilt."
        )
        # The load-bearing half: the ROW statics themselves carry the new
        # values. Only the region rebuild writes these.
        rows_after = _sync_row_texts(screen)
        assert rows_after != rows_before, "the sync-row regions never repainted"
        assert "Manual sync status: ready" in rows_after
        assert "Manual sync preview: Nothing pending." in rows_after
        assert "Pending outgoing: none" in rows_after
        # ...and the front door agrees.
        summary = _text(screen.query_one("#settings-overview-sync", Static))
        assert "ready" in summary and "none" in summary


@private_profile_test
async def test_sync_row_regions_repaint_when_the_row_SET_changes(request):
    """A run result renames a row and appends two more -- the region rebuild
    is what makes a variable row set possible at all (a per-row
    `Static.update` patch could not express it)."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)

        screen.manual_sync_rows = (
            ("Manual sync status", "failed"),
            ("Manual sync result", "Manual Sync failed: TimeoutError"),
            ("Pending outgoing", "notes: 2"),
            ("Conflict review", "notes | Note A | edited both sides"),
        )
        await _settle(pilot)

        rows = _sync_row_texts(screen)
        assert "Manual sync result: Manual Sync failed: TimeoutError" in rows
        assert "Conflict review: notes | Note A | edited both sides" in rows
        assert not any(row.startswith("Manual sync preview:") for row in rows), (
            "the replaced row survived -- the region did not rebuild"
        )


@private_profile_test
async def test_sync_rows_refresh_runs_once_per_visit(request, monkeypatch):
    """AC#3: on_mount and the mount's own ScreenResume must not both run it.

    Replays the two hooks against a fully MOUNTED screen deliberately. Their
    real firing order varies with how the screen was pushed, and on one of
    those orders `on_mount` runs before `is_mounted` flips and dispatches
    nothing at all -- which would let a naive dedupe suppress the only
    refresh. The replay pins the ordering the audit measured, and the second
    half pins that a genuinely later resume still refreshes.
    """
    calls: list[object] = []
    real = SettingsScreen._refresh_sync_rows

    def counting(self):
        calls.append(self)
        return real(self)

    monkeypatch.setattr(SettingsScreen, "_refresh_sync_rows", counting)

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        assert len(calls) >= 1, "The visit never refreshed the sync rows at all."

        calls.clear()
        screen.on_mount()
        screen.on_screen_resume()
        await _settle(pilot)
        assert len(calls) == 1, (
            f"Settings ran its sync-rows refresh {len(calls)}x for one visit."
        )

        screen.on_screen_resume()
        await _settle(pilot)
        assert len(calls) == 2, (
            "A later resume must refresh again: sync state can move while "
            "Settings is suspended."
        )


def test_focus_owner_check_survives_a_detached_focused_widget():
    """A popped screen's widget must not crash the sync-row refresh.

    TASK-24652. ``DOMNode.screen`` RAISES ``NoScreen`` for a detached node,
    so the original guard -- ``focused.screen is self`` -- exploded before it
    could protect anything. ``App.focused`` legitimately outlives its screen:
    push the first-run wizard over Settings and pop it, and the worker-driven
    ``_apply_sync_rows`` lands with ``app.focused`` still pointing at a
    NavigationButton that has left the tree. That surfaced as
    ``WorkerFailed: NoScreen('node has no screen')`` and failed
    ``test_rerun_over_settings_review_settings_returns_to_settings`` 3 times
    in 10 isolated runs; 0 in 12 after the guard.
    """
    from textual.dom import NoScreen

    from tldw_chatbook.UI.focus_ownership import (
        focus_is_on_screen,
        focused_id_on_screen,
    )

    screen = object()

    class _Detached:
        id = "nav-home"

        @property
        def screen(self):
            raise NoScreen("node has no screen")

    class _OnScreen:
        id = "nav-home"

        @property
        def screen(self):
            return screen

    class _Elsewhere:
        id = "nav-home"

        @property
        def screen(self):
            return object()

    # The regression: a detached widget must read as "not ours", not raise.
    assert focus_is_on_screen(_Detached(), screen) is False
    assert focused_id_on_screen(_Detached(), screen) is None
    # And the ordinary answers are unchanged.
    assert focus_is_on_screen(_OnScreen(), screen) is True
    assert focused_id_on_screen(_OnScreen(), screen) == "nav-home"
    assert focus_is_on_screen(_Elsewhere(), screen) is False
    assert focused_id_on_screen(_Elsewhere(), screen) is None
    assert focus_is_on_screen(None, screen) is False
    assert focused_id_on_screen(None, screen) is None
