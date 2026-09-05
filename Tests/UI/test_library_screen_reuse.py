"""Library screen-instance reuse contracts (TASK-31521).

The library route is reusable (`ScreenRoute.reusable`): one LibraryScreen
per app run, suspended on switch-away, resumed on return. These pin the
four behaviors the 2026-09-04 audit gated enablement on:

1. same-instance resume (the reuse itself);
2. suspend stops every armed debounce timer -- Textual only auto-cancels
   timers on real removal, and three of Library's five relied entirely on
   that removal (which reuse removes);
3. resume is the per-visit refresh seam (revisits re-kick the active
   surface, so data changed elsewhere while hidden appears);
4. the ingest-registry listener's DOM/DB branches gate on the suspended
   flag with one resume-time reconciliation (the counting/toast half
   stays live -- it is a cross-tab signal).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    data = tmp_path / "data"
    config = tmp_path / "config"
    for sub in (home, data, config):
        sub.mkdir(parents=True, exist_ok=True)
    config_file = config / "tldw_cli" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "library_screen_reuse")
    return home


async def _boot_settled(app, pilot) -> None:
    while not getattr(app, "_ui_ready", False):
        await asyncio.sleep(0.01)
    for _ in range(20):
        await asyncio.sleep(0.05)
        await pilot.pause()


async def _press_until_screen(pilot, key: str, expected: str) -> None:
    deadline = asyncio.get_running_loop().time() + 30.0
    await pilot.press(key)
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if type(pilot.app.screen).__name__ == expected:
            break
    assert type(pilot.app.screen).__name__ == expected
    for _ in range(6):
        await asyncio.sleep(0.05)
        await pilot.pause()


def test_library_route_is_flagged_reusable() -> None:
    route = resolve_screen_route("library")
    assert route is not None and route.reusable is True


@pytest.mark.ui
@pytest.mark.asyncio
async def test_library_reuse_and_suspend_timer_quiescence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One journey pins reuse, timer quiescence, and the resume seam.

    A single boot exercises all three because they are one lifecycle: visit
    Library, arm a debounce timer, leave (suspend must stop it), return
    (same instance, visit surfaces re-kicked).
    """
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)

        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        library = app.screen
        assert library._library_visit_entered is True, (
            "the first ScreenResume must run the visit-surface kicks"
        )

        # Arm a debounce timer the way a mid-keystroke filter would.
        library._library_media_filter_timer = library.set_timer(
            60.0, lambda: None
        )

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert library._library_screen_suspended is True
        for attr in (
            "_library_media_filter_timer",
            "_library_media_selection_timer",
            "_library_prompts_debounce_timer",
            "_library_notes_autosave_timer",
            "_library_source_snapshot_timeout_timer",
            "_library_list_entry_focus_timer",
        ):
            assert getattr(library, attr, None) is None, (
                f"{attr} still armed on the suspended screen -- Textual "
                "does not auto-cancel a suspended installed screen's "
                "timers, so suspend must"
            )
        # (wave-5 merge) The ingest path-debounce timer is a
        # `LibraryIngestState` field, not a flat screen attribute -- the
        # screen's generated shim block was deleted in the ingest cleanup
        # PR, so a `getattr` on the old flat name passes VACUOUSLY.
        assert library._ingest_state.path_debounce_timer is None, (
            "the ingest path-debounce timer is still armed on the "
            "suspended screen -- Textual does not auto-cancel a suspended "
            "installed screen's timers, so suspend must"
        )

        # The resume seam: revisits must re-kick the visit surfaces.
        kicks: list[str] = []
        real_refresh = library._refresh_library_visit_surfaces
        monkeypatch.setattr(
            library,
            "_refresh_library_visit_surfaces",
            lambda: (kicks.append("visit"), real_refresh())[1],
        )
        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        assert app.screen is library, (
            "library is a reusable route: returning must resume the "
            "installed instance, not construct a new one"
        )
        assert library._library_screen_suspended is False
        assert kicks == ["visit"], (
            "on_screen_resume must dispatch the per-visit surface refresh "
            "-- without it a revisit shows the previous visit's data"
        )


@pytest.mark.ui
@pytest.mark.asyncio
async def test_suspended_library_gates_ingest_dom_work_until_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Registry events against a hidden Library defer DOM work to resume."""
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        library = app.screen
        library._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")

        dynamic = Mock()
        snapshot = Mock()
        monkeypatch.setattr(
            library, "_update_library_ingest_dynamic_regions", dynamic
        )
        monkeypatch.setattr(
            library, "_refresh_local_source_snapshot", snapshot
        )
        # A registry mutation lands while the screen is hidden.
        library._handle_library_ingest_registry_changed()
        assert dynamic.call_count == 0, (
            "a suspended screen must not rebuild ingest widgets per event"
        )
        assert library._library_ingest_suspended_activity is True

        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        assert dynamic.call_count >= 1, (
            "resume must run exactly one ingest-UI reconciliation for the "
            "events gated while suspended"
        )
        assert snapshot.call_count >= 1, (
            "resume's visit refresh must re-read the source snapshot the "
            "suspended gate skipped"
        )
        assert library._library_ingest_suspended_activity is False


class _RecordingTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_on_screen_suspend_stops_every_timer_in_isolation() -> None:
    """Unit contract for the suspend hook, no Textual app involved.

    (Qodo #2414 finding 1.) Every timer attribute the hook owns is armed
    with a recording stub; one call must stop and clear all seven and set
    the suspended flag. Enumerating them HERE too means a new timer added
    to the hook without updating this table fails loudly.
    """
    from tldw_chatbook.UI.Screens.library_screen import (
        LibraryIngestState,
        LibraryPromptsState,
        LibraryScreen,
    )

    screen = LibraryScreen.__new__(LibraryScreen)
    # (wave-6 task 1) `_library_prompts_debounce_timer` below is now a
    # generated `LibraryPromptsState` shim property, so setting it invokes a
    # setter that routes into `self._prompts_state` -- an attribute this
    # `__new__` bypass never constructed. Seeded here, exactly as the ingest
    # state object below already is (recipe §3's seventh bypass shape).
    screen._prompts_state = LibraryPromptsState()
    timer_attrs = (
        "_library_list_entry_focus_timer",
        "_library_media_selection_timer",
        "_library_media_filter_timer",
        "_library_prompts_debounce_timer",
        "_library_notes_autosave_timer",
        "_library_source_snapshot_timeout_timer",
    )
    timers = {}
    for attr in timer_attrs:
        timers[attr] = _RecordingTimer()
        setattr(screen, attr, timers[attr])
    # (wave-5 merge) The seventh timer is a `LibraryIngestState` field, not
    # a flat screen attribute -- the screen's generated shim block was
    # deleted in the ingest cleanup PR, so `setattr`/`getattr` on the old
    # flat name would arm and assert a field the hook never reads. An
    # `object.__new__` screen also skips `__init__`'s state construction,
    # hence the explicit seed.
    screen._ingest_state = LibraryIngestState()
    ingest_timer = _RecordingTimer()
    screen._ingest_state.path_debounce_timer = ingest_timer
    # State the focus-disarm helper resets alongside its timer.
    screen._library_screen_suspended = False
    screen._library_list_entry_focus_generation = 0
    screen._library_pending_list_entry_focus = False
    screen._library_pending_list_entry_media_return = None
    screen._library_pending_list_entry_focus_anchor = None
    screen._library_media_return_settlement = None
    screen._library_media_last_exact_settlement = None
    screen._library_media_last_successful_settlement = None

    LibraryScreen.on_screen_suspend(screen)

    assert screen._library_screen_suspended is True
    for attr in timer_attrs:
        assert timers[attr].stopped, f"{attr} was not stopped"
        assert getattr(screen, attr) is None, f"{attr} was not cleared"
    assert ingest_timer.stopped, "the ingest path-debounce timer was not stopped"
    assert screen._ingest_state.path_debounce_timer is None, (
        "the ingest path-debounce timer was not cleared"
    )
