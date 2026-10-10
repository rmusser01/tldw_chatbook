"""Study shows user text as typed, never as Textual markup (TASK-34751).

Card fronts/backs, deck names, quiz names and question text are user input.
They reached ``Label``/``Static``/``Select`` as markup-on strings, so the
``[new]`` queue state appended to every card row vanished as a style tag, and
a front such as ``What does [/b] do?`` raised ``MarkupError`` in the
compositor -- which exits the app (lessons-textual: "A name shaped like markup
exits the whole app"). Each test paints the surface it covers: an unpainted
dropdown or label never parses, so it would pass on the broken code.
"""

from __future__ import annotations

import pytest
from textual.widgets import ListView, Select, Static
from textual.widgets._select import SelectOverlay
from textual.widgets._toast import Toast

from Tests.UI.test_study_card_list_single_writer import (
    _add_cards,
    _create_deck,
    _open_flashcards,
)
from Tests.UI.test_study_flashcards_real_service_contract import (  # noqa: F401
    _build_real_study_app,
    _card_list_labels,
    _disable_full_app_splash,
    _study_screen_up,
    _study_workers_pending,
    _text,
    _wait_until,
)
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_card_text_with_markup_brackets_renders_literally():
    """A card front/back is user text. RED on dev: ``[/b]`` in a front raises
    ``MarkupError`` while the row renders, and the row's ``[new]`` queue state
    is swallowed as a style tag."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        deck_id = await _create_deck(app, pilot, controller, "Markup deck")
        front = "What does [/b] do? [new]"
        back = "It closes [b]bold[/b] -- [red]not[/red] a style here"
        await _add_cards(app, deck_id, [front], back=back)

        await controller.refresh_cards()
        await _wait_until(
            pilot, lambda: not _study_workers_pending(app), what="the re-list"
        )
        card_list = app.screen.query_one("#card-list", ListView)
        assert _card_list_labels(card_list) == [f"{front} [new]"]

        await controller.start_review()
        await pilot.pause()
        assert _text(app.screen.query_one("#review-front", Static)) == front
        controller.show_answer()
        await pilot.pause()
        assert _text(app.screen.query_one("#review-back", Static)) == back
        assert app.is_running


@pytest.mark.asyncio
async def test_deck_name_with_markup_brackets_renders_literally():
    """A deck name reaches the deck picker, the move-target dropdown, the
    "Deck '…' created." status line and the dashboard's Recent Decks. RED on
    dev: ``MarkupError`` out of ``create_deck`` as it selects the new deck
    (the picker's label parses the name)."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        first = "Bio [/b] [new]"
        await _create_deck(app, pilot, controller, first)
        deck_select = app.screen.query_one("#deck-select", Select)
        assert _text(deck_select.query_one("#label", Static)) == first
        # The deck switch's own rebuild overwrites this message, so set it
        # through the same helper `create_deck` uses and read it back.
        controller._set_review_status(f"Deck '{first}' created.")
        await pilot.pause()
        status = app.screen.query_one("#review-status", Static)
        assert _text(status) == f"Deck '{first}' created."

        second = "Chem [i]"
        await _create_deck(app, pilot, controller, second)
        target_select = app.screen.query_one("#move-card-target-select", Select)
        target_select.expanded = True  # paint the dropdown's prompts
        await pilot.pause()
        overlay = target_select.query_one(SelectOverlay)
        prompts = [
            str(overlay.get_option_at_index(index).prompt)
            for index in range(overlay.option_count)
        ]
        assert first in prompts, prompts
        assert app.is_running
        target_select.expanded = False

        await pilot.click("#view-dashboard-btn")
        # The snapshot reloads on scope application (screen open / scope
        # change), not on a section switch -- run that reload directly.
        await app.screen._refresh_dashboard_snapshot()
        await _wait_until(
            pilot,
            lambda: (
                "Chem" in _text(app.screen.query_one("#study-recent-decks", Static))
            ),
            what="the dashboard's Recent Decks",
        )
        recent = _text(app.screen.query_one("#study-recent-decks", Static))
        assert first in recent and second in recent, recent


@pytest.mark.asyncio
async def test_quiz_text_with_markup_brackets_renders_literally():
    """Quiz names and question text reach the quiz picker, the question list
    and the quiz session summary. RED on dev: ``MarkupError``."""
    app, _db = _build_real_study_app()
    quiz_name = "Tree [/b] Drill"
    question_text = "A balanced tree has height [/i] ____ [new]"
    app.study_quiz_scope_service.quizzes[0]["name"] = quiz_name
    app.study_quiz_scope_service.questions[0]["question_text"] = question_text
    async with app.run_test(size=(160, 45)) as pilot:
        await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
        await pilot.click("#view-quizzes-btn")
        await _wait_until(
            pilot,
            lambda: (
                _card_list_labels(app.screen.query_one("#quiz-question-list", ListView))
                == [question_text]
            ),
            what="the selected quiz's questions",
        )
        quiz_select = app.screen.query_one("#quiz-select", Select)
        assert _text(quiz_select.query_one("#label", Static)) == quiz_name
        quiz_select.expanded = True  # paint the dropdown's prompts
        await pilot.pause()
        quiz_select.expanded = False

        app.screen.sync_shell_from_window()
        await pilot.pause()
        summary = app.screen.query_one("#quiz-session-summary", Static)
        assert _text(summary) == f"Selected quiz: {quiz_name}"
        assert app.is_running


#: Every Study text surface that shows user text, by the view that mounts it.
#: (Labels in list rows and picker prompts are painted by the tests above.)
_DASHBOARD_TEXT_SURFACES = (
    "#study-scope-summary",
    "#study-recent-decks",
    "#study-recent-quizzes",
    "#study-source-generation-status",
    "#quiz-scope-summary",
    "#quiz-session-summary",
    "#quiz-session-status",
    "#study-scope-workspace-name",
)
_FLASHCARDS_TEXT_SURFACES = (
    "#review-status",
    "#review-front",
    "#review-back",
    "#review-next-intervals",
)
_QUIZZES_TEXT_SURFACES = (
    "#quiz-attempt-status",
    "#quiz-attempt-question",
    "#quiz-attempt-history-summary",
)


@pytest.mark.asyncio
async def test_every_study_text_surface_paints_markup_like_text_literally():
    """Each surface takes the same hostile text through ``Static.update`` (the
    path every Study helper uses) and paints it as typed; the dashboard's
    Resume button takes a deck title. RED on dev: ``MarkupError``."""
    hostile = "Bio [/b] [new] [@click=app.quit]x[/]"
    app, _db = _build_real_study_app()

    def assert_literal(selectors) -> None:
        for selector in selectors:
            widget = app.screen.query_one(selector, Static)
            widget.update(hostile)
            assert _text(widget) == hostile, selector

    async with app.run_test(size=(160, 45)) as pilot:
        await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
        assert_literal(_DASHBOARD_TEXT_SURFACES)
        app.screen.study_dashboard.update_resume_action(f"flashcards: {hostile}")
        resume = app.screen.query_one("#study-resume-last")
        assert str(resume.label) == f"Resume flashcards: {hostile}"

        await _open_flashcards(app, pilot)
        assert_literal(_FLASHCARDS_TEXT_SURFACES)

        await pilot.click("#view-quizzes-btn")
        await _wait_until(
            pilot,
            lambda: bool(app.screen.query("#quiz-attempt-status")),
            what="the Quizzes view",
        )
        assert_literal(_QUIZZES_TEXT_SURFACES)
        await pilot.pause()  # paint them
        assert app.is_running


@pytest.mark.asyncio
async def test_deck_created_toast_shows_the_deck_name_literally(tmp_path):
    """Creating a local deck raises a "deck created" toast through the app's
    notification dispatcher, wired as ``app_service_wiring`` wires it. Found
    live, after every widget above was fixed: ``Toast.render`` parses its
    message as markup, so a deck named ``bd9st Bio [/b] deck`` exited the app
    the moment it was created. No Pilot test drew that toast: ``run_test``
    disables notifications unless asked (``notifications=True``), and
    ``_build_real_study_app`` leaves the dispatcher unwired. RED on dev:
    ``MarkupError``."""
    app, _db = _build_real_study_app()
    local_study = app.study_scope_service.local_service
    local_study.notification_dispatch_service = NotificationDispatchService(
        store=ClientNotificationsDB(tmp_path / "notifications.db")
    )
    local_study.notification_app = app
    async with app.run_test(size=(160, 45), notifications=True) as pilot:
        controller = await _open_flashcards(app, pilot)
        name = "bd9st Bio [/b] deck"
        await _create_deck(app, pilot, controller, name)
        await _wait_until(
            pilot, lambda: bool(app.screen.query(Toast)), what="the deck-created toast"
        )
        await pilot.pause()  # paint it
        toast_text = " ".join(_text(toast) for toast in app.screen.query(Toast))
        assert name in toast_text, toast_text
        assert app.is_running
