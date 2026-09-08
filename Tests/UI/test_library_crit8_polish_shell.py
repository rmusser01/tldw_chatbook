"""Critique-8 polish/shell fixes for the Library landing, rail and Notes.

Covers tasks 32058, 32059, 32061, 32062, 32063, 32064, 32066, 32069, 32071
and 32072 -- the polish-shell group of the critique-8 fix wave.
"""

from __future__ import annotations

import pytest

from tldw_chatbook import config as app_config
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import LibraryLandingCanvas
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_library_shell,
)

#: The compact geometry the critique-8 live review ran at (register row 17).
COMPACT_TEST_SIZE = (100, 30)
WIDE_TEST_SIZE = (170, 48)


@pytest.mark.asyncio
async def test_library_landing_canvas_is_hidden_at_compact_widths():
    """task-32066: library.md says the rail owns navigation below 120 columns.

    At 100x30 the landing canvas ("Search everything…", counts, From your
    Library, Quick actions) was still painted beside the 22-column rail.
    """
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=COMPACT_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()

        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        rail = screen.query_one("#library-rail")

        assert rail.display is True
        assert landing.region.width == 0, (
            "the landing canvas must not paint at compact widths"
        )


@pytest.mark.asyncio
async def test_library_landing_canvas_paints_at_wide_widths():
    """The same route keeps the landing beside the rail above the breakpoint."""
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()

        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)

        assert landing.region.width > 0


# --- task-32059: Get started survives a relaunch before the first visit ---


def test_get_started_survives_a_relaunch_before_the_first_library_visit(
    tmp_path, monkeypatch
) -> None:
    """task-32059: complete setup, quit, relaunch -- Get started must survive.

    ``coerce_library_lifecycle`` reads an absent lifecycle as EXPANDED once the
    profile was not created in the current run, so a user who finished first-run
    setup and quit before ever opening Library never saw the documented compact
    Get started rail. The lifecycle is now stamped at profile creation.
    """
    config_path = tmp_path / "relaunch-profile" / "config.toml"
    config_path.parent.mkdir(parents=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(
        app_config, "_FIRST_PROFILE_CREATED_THIS_SESSION", False, raising=False
    )
    app_config._CONFIG_CACHE = None
    app_config._CONFIG_CACHE_SOURCE = None
    app_config._SETTINGS_CACHE = None
    app_config._SETTINGS_CACHE_SOURCE = None

    # Run 1: the profile is created. Library is never opened.
    app_config.load_settings(force_reload=True)
    first_run = _build_test_app(preserve_profile_admission=True)
    assert first_run.library_new_profile_admission is True

    # Run 2: same config file, no longer created by this process.
    monkeypatch.setattr(
        app_config, "_FIRST_PROFILE_CREATED_THIS_SESSION", False, raising=False
    )
    app_config._CONFIG_CACHE = None
    app_config._CONFIG_CACHE_SOURCE = None
    app_config._SETTINGS_CACHE = None
    app_config._SETTINGS_CACHE_SOURCE = None
    app_config.load_settings(force_reload=True)
    second_run = _build_test_app()
    assert second_run.library_new_profile_admission is False

    assert (
        config_path.read_text(encoding="utf-8").count('lifecycle = "unknown"') == 1
    ), "profile creation must stamp the lifecycle into [library.rail_state]"
    screen = LibraryScreen(second_run)
    assert screen._library_lifecycle is LibraryLifecycle.UNKNOWN
