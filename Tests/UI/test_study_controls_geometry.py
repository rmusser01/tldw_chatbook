"""Study dashboard / Flashcards control geometry under the shipping CSS (TASK-34000.6).

Split out of ``test_study_flashcards_real_service_contract.py`` (a pure move):
with these six parametrizations that file measured at the PR UI lane's 60 s
budget, so the real-service crash tests stay gated there and the geometry runs
here, NOT in ``scripts/ui_pr_gate_census.txt``.

The tests load the app bundle (``TldwCli.CSS_PATH``) because a harness screen
without it measures nothing -- see ``backlog/docs/lessons-textual.md`` ("A
geometry or `.display` test without `CSS_PATH = BUNDLED_STYLESHEET` measures
nothing"). Shared helpers are imported from the gated file, never copied.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_study_flashcards_real_service_contract import (  # noqa: F401
    _build_real_study_app,
    _disable_full_app_splash,  # autouse fixture, re-exported so it applies here
    _flashcards_view_up,
    _study_screen_up,
    _text,
    _wait_until,
)
from tldw_chatbook.UI.Screens.study_scope_models import (
    MATERIAL_SOURCE_LIBRARY,
    MATERIAL_TITLE_LIBRARY_SOURCES,
    StudyScopeContext,
)
from tldw_chatbook.UI.Study_Window import StudyWindow

# Same profile-selection marker as the gated file (full-app mounts).
pytestmark = pytest.mark.bootstrap_profile


# --- AC#2: the Dashboard's actions and the Flashcards controls render --------

_DASHBOARD_ACTION_IDS = (
    "#study-resume-last",
    "#study-open-flashcards",
    "#study-open-quizzes",
    "#study-generate-source-pack",
)


def _library_scope_context() -> StudyScopeContext:
    return StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_summary="Notes: 3",
        # Fix round 1 (Minor 2): one title that cleans to empty, so the test
        # proves Study and the Library hand-off drop it the same way.
        material_titles=("<draft>", *(f"Title {index}" for index in range(12))),
        return_hint=MATERIAL_SOURCE_LIBRARY,
    )


def _assert_inside_screen(widget, size: tuple[int, int]) -> None:
    region = widget.region
    width, height = size
    assert region.height > 0 and region.width > 0, (
        f"{widget.id or widget} has no painted area: {region}"
    )
    assert region.y >= 0 and region.bottom <= height, (
        f"{widget.id or widget} is outside the {width}x{height} screen: {region}"
    )
    assert region.x >= 0 and region.right <= width, (
        f"{widget.id or widget} is outside the {width}x{height} screen: {region}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (160, 70), (120, 36)])
async def test_dashboard_action_buttons_render_inside_the_screen(size):
    """RED on origin/dev: the dashboard's columns ``Horizontal`` kept Textual's
    default ``height: 1fr`` inside an auto-height card, took every remaining
    row, and the actions row below it never painted (capture
    ``verify/nl-v-s-05/08-dashboard-160x70.txt``)."""
    app, _db = _build_real_study_app(scope_context=_library_scope_context())

    async with app.run_test(size=size) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                _study_screen_up(app)
                and "Local Library Sources"
                in _text(app.screen.query_one("#study-scope-summary", Static))
            ),
            what="the Study dashboard with the staged Library scope",
        )
        for selector in _DASHBOARD_ACTION_IDS:
            _assert_inside_screen(app.screen.query_one(selector, Button), size)
        status = app.screen.query_one("#study-source-generation-status", Static)
        _assert_inside_screen(status, size)

        # The section bar is an unstyled Horizontal: once the dashboard
        # measured auto it took the freed `1fr` (18 rows at 160x45) and the
        # dashboard floated mid-screen. The bar stays one row of buttons and
        # the dashboard sits directly under it.
        bar = app.screen.query_one("#study-section-bar")
        dashboard = app.screen.query_one("#study-dashboard")
        assert bar.region.height <= 3, f"section bar ballooned: {bar.region}"
        assert dashboard.region.y == bar.region.bottom, (bar.region, dashboard.region)

        # AC#4: the banner describes the carried scope with the same names
        # and count the Library hand-off line showed (12 staged titles ->
        # Study keeps 10 -> 3 named and 7 more), via the shared describer.
        banner = _text(app.screen.query_one("#study-scope-summary", Static))
        assert (
            "Local Library Sources: Title 0, Title 1, Title 2 and 7 more" in banner
        ), banner
        # ... and the Library line for the SAME staged titles says the same
        # names and count (the `<draft>` title drops out on both sides).
        from tldw_chatbook.UI.Library_Modules.screen_helpers import (
            _library_carries_forward_line,
        )

        assert (
            _library_carries_forward_line(
                list(_library_scope_context().material_titles)
            )
            == "Carries forward: Title 0, Title 1, Title 2 and 7 more."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (160, 70), (120, 36)])
async def test_flashcards_tab_shows_deck_picker_and_card_editor_controls(size):
    """RED on origin/dev: ``.card-editor`` had no ``height: auto`` so it took
    ``1fr`` of the scroll container and hid everything after "Decks:"."""
    app, _db = _build_real_study_app()

    async with app.run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
        await pilot.click("#view-flashcards-btn")
        await _wait_until(
            pilot, lambda: _flashcards_view_up(app), what="the Flashcards view"
        )
        await _wait_until(
            pilot,
            lambda: app.screen.query_one("#create-deck-button").region.height > 0,
            what="the Flashcards editor to be laid out",
        )

        for selector in ("#deck-select", "#new-deck-name-input", "#create-deck-button"):
            _assert_inside_screen(app.screen.query_one(selector), size)

        # `StudyWindow { height: 100% }` overflowed the shell by the header
        # and section bar rows, so the scroll container's last rows were
        # never paintable: the window must end inside the shell.
        shell = app.screen.query_one("#study-shell")
        window = app.screen.query_one(StudyWindow)
        assert window.region.bottom <= shell.region.bottom, (
            shell.region,
            window.region,
        )

        # The editor's own controls: inside the editor's painted box, with a
        # non-empty region (a clipped child has region height 0).
        editor = app.screen.query_one(".card-editor")
        for selector in ("#card-front", "#card-back", "#card-tags", "#create-card-btn"):
            widget = app.screen.query_one(selector)
            assert widget.region.height > 0, f"{selector} is clipped: {widget.region}"
            assert editor.region.contains_region(widget.region), (
                f"{selector} {widget.region} is outside the editor {editor.region}"
            )

        # The row-mate buttons used to start AT the right edge (their Input
        # took 100%): every control must end inside the screen's width.
        for selector in (
            "#create-deck-button",
            "#flashcard-refresh-button",
            "#card-tags",
        ):
            widget = app.screen.query_one(selector)
            assert widget.region.right <= size[0], f"{selector} {widget.region}"

        # The editor lives in a scroll container; Create Card is reachable.
        create_button = app.screen.query_one("#create-card-btn", Button)
        create_button.scroll_visible(animate=False)
        await _wait_until(
            pilot,
            lambda: (
                0 <= create_button.region.y and create_button.region.bottom <= size[1]
            ),
            what="Create Card to scroll into view",
        )
        _assert_inside_screen(create_button, size)
