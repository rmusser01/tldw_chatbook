"""Study shows user text as typed and keeps it findable (TASK-34751).

A workspace name is user text. The Study scope banner drew it as Textual
markup, so a workspace named ``Bio [/b] plan`` raised ``MarkupError`` as soon
as the banner painted -- which exits the app (lessons-textual: "A name shaped
like markup exits the whole app").

Deck and quiz names reach their ``Select`` pickers as rich ``Text`` prompts
(TASK-33007). That choice matters beyond rendering: Textual 8.2.8's
type-to-search (``SelectOverlay._find_search_match``) reads only ``Text`` and
``str`` prompts and skips ``Content``, so a literal prompt built as
``Content`` would render correctly and silently stop being searchable. The
picker test types into the real dropdowns to pin that.
"""

from __future__ import annotations

import pytest
from textual.widgets import ListView, Select, Static

from Tests.UI.test_study_flashcards_real_service_contract import (  # noqa: F401
    _build_real_study_app,
    _card_list_labels,
    _create_deck,
    _disable_full_app_splash,
    _flashcards_view_up,
    _open_flashcards,
    _study_screen_up,
    _text,
    _wait_for_study_workers,
    _wait_until,
)
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeType

pytestmark = pytest.mark.bootstrap_profile


async def _type_into_open_picker(pilot, select: Select, keys: str) -> None:
    """Open ``select``'s dropdown, type ``keys`` and press Enter."""
    select.focus()
    select.expanded = True
    await pilot.pause()  # paint the dropdown's prompts
    for key in keys:
        await pilot.press(key)
    await pilot.press("enter")


@pytest.mark.asyncio
async def test_workspace_name_with_markup_brackets_paints_literally_in_the_scope_banner():
    """RED on dev 4de437a5c5: ``_sync_scope_banner`` hands
    ``f"Workspace: {name}"`` to a markup-on ``Static``, so painting the
    banner raises ``MarkupError: closing tag '[/b]' does not match any open
    tag`` (and ``[new]`` would vanish as a style tag)."""
    app, _db = _build_real_study_app()
    name = "Bio [/b] [new] plan"
    async with app.run_test(size=(160, 45)) as pilot:
        await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
        app.screen.enter_workspace_scope("workspace-1", name)
        await _wait_until(
            pilot,
            lambda: (
                app.screen.scope_state.scope_type == StudyScopeType.WORKSPACE
                and app.screen.scope_state.workspace_id == "workspace-1"
            ),
            what="the workspace scope to apply",
        )
        # The banner heads the Study window, which the Flashcards view shows.
        await pilot.click("#view-flashcards-btn")
        await _wait_until(
            pilot, lambda: _flashcards_view_up(app), what="the Flashcards view"
        )
        await pilot.pause()  # paint the banner
        assert app.is_running, "painting the scope banner exited the app"
        banner_name = app.screen.query_one("#study-scope-workspace-name", Static)
        assert banner_name.visible
        assert _text(banner_name) == f"Workspace: {name}"
        # Let the scope change's reloads finish before the app is torn down.
        await _wait_for_study_workers(pilot, app, what="the workspace scope's reloads")


@pytest.mark.asyncio
async def test_deck_and_quiz_pickers_find_a_markup_named_entry_by_typing():
    """Typing an entry's first letter in an open picker and pressing Enter
    selects it, and the picked name paints as typed, brackets included.
    RED on the pre-TASK-33007 code (75159f8843): ``MarkupError`` from the
    deck picker. RED with ``Content`` prompts: the search finds nothing and
    Enter keeps the previous deck/quiz."""
    app, _db = _build_real_study_app()
    graph_drill = "Graph [/b] Drill"
    app.study_quiz_scope_service.quizzes[1]["name"] = graph_drill
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        zebra = "Zebra [/b] deck"
        zebra_id = await _create_deck(app, pilot, controller, zebra)
        await _create_deck(app, pilot, controller, "Alpha deck")  # now selected

        deck_select = app.screen.query_one("#deck-select", Select)
        await _type_into_open_picker(pilot, deck_select, "z")
        await _wait_until(
            pilot,
            lambda: str(deck_select.value) == zebra_id,
            what="typing 'z' to pick the Zebra deck",
        )
        await _wait_for_study_workers(pilot, app, what="the deck switch")
        assert _text(deck_select.query_one("#label", Static)) == zebra
        assert _card_list_labels(app.screen.query_one("#card-list", ListView)) == [
            "No cards in this deck."
        ]

        await pilot.click("#view-quizzes-btn")
        await _wait_until(
            pilot,
            lambda: str(app.screen.query_one("#quiz-select", Select).value)
            == "quiz-1",
            what="the Quizzes view with its first quiz selected",
        )
        quiz_select = app.screen.query_one("#quiz-select", Select)
        await _type_into_open_picker(pilot, quiz_select, "g")
        await _wait_until(
            pilot,
            lambda: str(quiz_select.value) == "quiz-2",
            what="typing 'g' to pick the Graph quiz",
        )
        await pilot.pause()
        assert _text(quiz_select.query_one("#label", Static)) == graph_drill
        assert app.is_running
