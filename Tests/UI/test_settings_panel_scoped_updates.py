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


async def test_category_switch_rebuilds_only_the_two_category_panes():
    """AC#1: a rail click leaves every widget that does not read the
    category's content exactly where it was."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        before = _identities(screen, _STABLE)
        nav_before = id(screen.query_one(MainNavigationBar))

        await pilot.click("#settings-category-storage")
        await _settle(pilot)

        assert _identities(screen, _STABLE) == before, (
            "A Settings category switch rebuilt chrome that does not read the "
            "category."
        )
        assert id(screen.query_one(MainNavigationBar)) == nav_before
        # ...and the panes it DOES own actually repainted.
        assert screen.active_category == "storage"
        assert "Storage" in _text(screen.query_one("#settings-category-label", Static))


async def test_category_switch_moves_the_active_rail_marker():
    """The rail is no longer rebuilt, so the active marker has to be patched
    onto the surviving buttons -- with the same result as a fresh compose."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)
        assert screen.query_one("#settings-category-overview", Button).has_class(
            "settings-active-section"
        )

        await pilot.click("#settings-category-storage")
        await _settle(pilot)

        assert screen.query_one("#settings-category-storage", Button).has_class(
            "settings-active-section"
        )
        assert not screen.query_one("#settings-category-overview", Button).has_class(
            "settings-active-section"
        )


async def test_category_switch_lands_focus_on_the_selected_rail_button():
    """What the recompose provided via `_pending_category_focus_value`: after
    a switch, focus is on the newly selected category button."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(160, 45)) as pilot:
        screen = await _open_settings(pilot)

        await pilot.click("#settings-category-storage")
        await _settle(pilot)

        assert screen.active_category == "storage"
        assert getattr(host.focused, "id", None) == "settings-category-storage"


async def test_category_switch_keeps_compact_pane_classes_at_a_compact_size():
    """Compact geometry is only measured at a compact size.

    The compact classes used to be re-applied by `compose_content` on every
    category recompose. The panes are not recomposed by the screen any more,
    so the classes have to SURVIVE a switch instead -- and a pane that lost
    them would only be wrong below the 90-column breakpoint, where no
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


async def test_same_category_rebuild_keeps_focus_on_the_control_used():
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
        rows.extend(
            _text(row) for row in screen.query_one(region_id).query(Static)
        )
    return rows


async def test_sync_row_refresh_patches_rows_without_rebuilding_the_screen():
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


async def test_sync_row_regions_repaint_when_the_row_SET_changes():
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


async def test_sync_rows_refresh_runs_once_per_visit(monkeypatch):
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
