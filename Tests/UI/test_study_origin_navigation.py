# test_study_origin_navigation.py
# Description: task-4011 regression -- StudyScreen's breadcrumb and Escape
# destination must reflect the screen the user actually arrived from.
#
# task-2854 gave Study a truthful navigation identity for the ONE origin it
# considered: Library's staging canvas ("Library ▸ Study", Escape back to the
# Library "Study decks" row). Home's "Review flashcards" button navigates
# straight to Study too (`open_home_flashcards_review` ->
# `open_study_screen(initial_section="flashcards")`), and for that origin
# both the breadcrumb and Escape lied: the header claimed "Library ▸ Study /
# Esc: back to Library" and Escape landed on Library's Study-decks staging
# canvas -- a screen the user never visited -- instead of back on Home.

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.study_screen import StudyScreen


async def _wait_for_screen(app, pilot, screen_type, tab: str):
    for _ in range(300):
        if app.current_tab == tab and isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"did not finish routing to {screen_type.__name__}.")


def _study_header_text(screen) -> tuple[str, str]:
    title = screen.query_one(
        "#study-destination-header #workbench-header-title", Static
    )
    subtitle = screen.query_one(
        "#study-destination-header #workbench-header-subtitle", Static
    )
    return str(title.renderable), str(subtitle.renderable)


@pytest.mark.asyncio
async def test_home_origin_study_breadcrumbs_home_and_escape_returns_home():
    """task-4011 AC#1/#2: reached via Home's Review flashcards entry, Study
    must say it came from Home and Escape must return there -- not to a
    Library staging canvas the user never visited.
    """
    app = _build_test_app()
    app._initial_tab_value = "home"
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_screen(app, pilot, HomeScreen, "home")

        app.open_home_flashcards_review()
        study = await _wait_for_screen(app, pilot, StudyScreen, "study")

        title, subtitle = _study_header_text(study)
        assert title == "Home ▸ Study", title
        assert "back to Home" in subtitle, subtitle
        assert "Library" not in subtitle, subtitle

        await pilot.press("escape")
        await _wait_for_screen(app, pilot, HomeScreen, "home")


@pytest.mark.asyncio
async def test_library_origin_study_round_trip_still_returns_to_library():
    """task-4011 AC#3: the task-2854 behaviour must not regress -- Study
    reached via the Library staging canvas still breadcrumbs Library and
    Escape still returns to the Library staging canvas.
    """
    app = _build_test_app()
    app._initial_tab_value = "library"
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_screen(app, pilot, LibraryScreen, "library")

        # The same call Library's "Continue in Study" handoff rows make
        # (`LibraryScreen._open_study_section` -> `open_study_screen`).
        app.open_study_screen(initial_section="flashcards")
        study = await _wait_for_screen(app, pilot, StudyScreen, "study")

        title, subtitle = _study_header_text(study)
        assert title == "Library ▸ Study", title
        assert "back to Library" in subtitle, subtitle

        await pilot.press("escape")
        await _wait_for_screen(app, pilot, LibraryScreen, "library")


@pytest.mark.asyncio
async def test_home_origin_does_not_leak_into_later_library_entry():
    """task-4011: origin is a single-use handoff, not sticky state -- after a
    Home-origin visit, a later Library-path entry must breadcrumb Library
    again.
    """
    app = _build_test_app()
    app._initial_tab_value = "home"
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_screen(app, pilot, HomeScreen, "home")

        app.open_home_flashcards_review()
        study = await _wait_for_screen(app, pilot, StudyScreen, "study")
        title, _ = _study_header_text(study)
        assert title == "Home ▸ Study", title

        await pilot.press("escape")
        await _wait_for_screen(app, pilot, HomeScreen, "home")

        app.open_study_screen(initial_section="dashboard")
        study = await _wait_for_screen(app, pilot, StudyScreen, "study")
        title, subtitle = _study_header_text(study)
        assert title == "Library ▸ Study", title
        assert "back to Library" in subtitle, subtitle
