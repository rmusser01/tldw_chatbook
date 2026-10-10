"""Study ▸ Flashcards: ``#card-list`` has one writer (TASK-34751).

``refresh_decks`` restores or picks the selected deck through
``Select.value``, which posts ``Select.Changed``;
``StudyWindow.handle_deck_select_changed`` then rebuilds ``#card-list`` in its
own worker while the caller (``initialize_view``, ``create_deck``,
``delete_selected_deck``) rebuilds it too -- and Create Card, Delete/Move Card
and the Refresh button are further writers on their own workers. Each rebuild
cleared the list and then appended row by row with an ``await`` per row, so
two rebuilds in flight interleaved: a four-card deck listed seven rows and the
create-deck path could list "No cards in this deck." twice. The same
interleave made the gated contract test
(``test_study_flashcards_real_service_contract.py``) read an empty list on a
loaded runner.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input, ListView, Static

from Tests.UI.test_study_flashcards_real_service_contract import (  # noqa: F401
    _build_real_study_app,
    _card_list_labels,
    _disable_full_app_splash,
    _flashcards_view_up,
    _study_screen_up,
    _study_workers_pending,
    _text,
    _wait_until,
)
from tldw_chatbook.UI.Study_Window import StudyWindow

pytestmark = pytest.mark.bootstrap_profile


def _card_fronts(app) -> list[str]:
    card_list = app.screen.query_one("#card-list", ListView)
    return [
        str((getattr(item, "study_card_record", None) or {}).get("front"))
        for item in card_list.children
    ]


async def _open_flashcards(app, pilot):
    await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
    await pilot.click("#view-flashcards-btn")
    await _wait_until(
        pilot, lambda: _flashcards_view_up(app), what="the Flashcards view"
    )
    controller = app.screen.query_one(StudyWindow).flashcards_controller
    await _wait_until(
        pilot,
        lambda: (
            "No study decks yet"
            in _text(app.screen.query_one("#review-status", Static))
        ),
        what="the initial deck load",
    )
    return controller


async def _create_deck(app, pilot, controller, name: str) -> str:
    app.screen.query_one("#new-deck-name-input", Input).value = name
    await controller.create_deck()
    await _wait_until(
        pilot, lambda: not _study_workers_pending(app), what="the deck switch"
    )
    deck_id = controller._selected_deck_id()
    assert deck_id is not None, "the created deck was not selected"
    return deck_id


def _slow_card_fetches(app, seconds: float = 0.05) -> None:
    """Make every card-list fetch take `seconds` (a server round trip), so
    two rebuilds started close together are both in flight at once."""
    service = app.study_scope_service
    real_list_flashcards = service.list_flashcards

    async def slow_list_flashcards(**kwargs):
        await asyncio.sleep(seconds)
        return await real_list_flashcards(**kwargs)

    service.list_flashcards = slow_list_flashcards


async def _add_cards(app, deck_id: str, fronts: list[str], back: str = "A") -> None:
    for front in fronts:
        await app.study_scope_service.create_flashcard(
            mode="local",
            deck_id=deck_id,
            front=front,
            back=back,
            tags=[],
            notes=None,
            extra=None,
        )


@pytest.mark.asyncio
async def test_relisting_a_selected_deck_lists_each_card_once():
    """The screen-resume / scope-reload path: ``initialize_view`` re-lists the
    decks with the selection restored (a ``Select.Changed`` rebuild) and then
    re-lists the cards itself. RED on dev: 7 rows for 4 cards."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        deck_id = await _create_deck(app, pilot, controller, "Deck A")
        fronts = [f"Q{index}" for index in range(4)]
        await _add_cards(app, deck_id, fronts)

        await controller.initialize_view()
        await _wait_until(
            pilot, lambda: not _study_workers_pending(app), what="the re-list"
        )
        assert _card_fronts(app) == fronts
        assert len(controller.current_cards) == len(fronts)


@pytest.mark.asyncio
async def test_overlapping_card_list_rebuilds_list_each_card_once():
    """Any two writers in flight at once -- Refresh pressed while Create Card's
    rebuild is still running, say -- leave one copy of each card, in order.
    RED on dev: the two rebuilds' per-row appends interleave (8 rows)."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        deck_id = await _create_deck(app, pilot, controller, "Deck B")
        fronts = [f"R{index}" for index in range(4)]
        await _add_cards(app, deck_id, fronts)

        await asyncio.gather(controller.refresh_cards(), controller.refresh_cards())
        await _wait_until(
            pilot, lambda: not _study_workers_pending(app), what="the rebuilds"
        )
        assert _card_fronts(app) == fronts
        assert [card["front"] for card in controller.current_cards] == fronts


@pytest.mark.asyncio
async def test_creating_a_deck_lists_its_empty_state_once():
    """Create Deck selects the new deck (a ``Select.Changed`` rebuild) and then
    rebuilds the list itself: once both settle, one empty-state row. With the
    fetch taking a server-like 50 ms both rebuilds are in flight together.
    RED on dev: "No cards in this deck." listed twice."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        _slow_card_fetches(app)
        await _create_deck(app, pilot, controller, "Deck C")
        card_list = app.screen.query_one("#card-list", ListView)
        assert _card_list_labels(card_list) == ["No cards in this deck."]


@pytest.mark.asyncio
async def test_rebuild_in_flight_when_the_list_leaves_stops_without_writing():
    """Switching sub-view removes ``#card-list`` (``watch_current_view``)
    while a rebuild is still fetching; the rebuild must stop, not write into
    a list -- or a review panel -- that is gone. The same check covers app
    shutdown, where a late mount raised ``MountError`` in teardown. RED on
    dev: the late writes raise once the fetch returns."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        deck_id = await _create_deck(app, pilot, controller, "Deck D")
        await _add_cards(app, deck_id, ["S0", "S1"])
        _slow_card_fetches(app, seconds=0.2)

        rebuild = asyncio.ensure_future(controller.refresh_cards())
        await pilot.pause(0.05)  # the rebuild is now waiting on its fetch
        await pilot.click("#view-quizzes-btn")
        await _wait_until(
            pilot,
            lambda: not app.screen.query("#card-list"),
            what="the Flashcards view to be removed",
        )
        await rebuild  # raises on dev: the late writes find no list / panel
        assert rebuild.exception() is None
