"""A card-list rebuild still running when the app quits ends quietly (TASK-34751).

Once the app is exiting, Textual reports every widget as detached
(``is_attached`` is False as soon as ``App._exit`` is set), so a rebuild that
resumes after its card fetch then appends into a list that "is not mounted":
``MountError`` out of the rebuild's worker. Under ``run_test`` that surfaces
as ``WorkerFailed`` at teardown -- the signature that made the gated contract
test (``test_study_flashcards_real_service_contract.py``) fail on a loaded
runner and block unrelated PRs in the merge queue. In the app it is a worker
failing during quit.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.test_study_flashcards_real_service_contract import (  # noqa: F401
    _build_real_study_app,
    _create_deck,
    _disable_full_app_splash,
    _open_flashcards,
    _wait_until,
)

pytestmark = pytest.mark.bootstrap_profile


async def _add_cards(app, deck_id: str, fronts: list[str]) -> None:
    for front in fronts:
        await app.study_scope_service.create_flashcard(
            mode="local",
            deck_id=deck_id,
            front=front,
            back="A",
            tags=[],
            notes=None,
            extra=None,
        )


def _hold_card_fetches(app) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold each card fetch's result; return (fetched, release) events.

    The real fetch runs first; only its return waits, so the test chooses
    the moment the rebuild resumes instead of racing a sleep.
    """
    fetched = asyncio.Event()
    release = asyncio.Event()
    service = app.study_scope_service
    real_list_flashcards = service.list_flashcards

    async def held_list_flashcards(**kwargs):
        cards = await real_list_flashcards(**kwargs)
        fetched.set()
        await release.wait()
        return cards

    service.list_flashcards = held_list_flashcards
    return fetched, release


@pytest.mark.asyncio
async def test_a_card_list_rebuild_resuming_after_quit_ends_quietly():
    """Refresh is pressed, the app quits while the card fetch is out, then
    the fetch returns. RED on dev 4de437a5c5: leaving ``run_test`` raises
    ``WorkerFailed: MountError("Can't mount widget(s) before
    ListView(id='card-list') is mounted")`` from the Refresh worker."""
    app, _db = _build_real_study_app()
    async with app.run_test(size=(160, 45)) as pilot:
        controller = await _open_flashcards(app, pilot)
        deck_id = await _create_deck(app, pilot, controller, "Quit deck")
        await _add_cards(app, deck_id, ["Q0", "Q1"])
        fetched, release = _hold_card_fetches(app)

        app.screen.query_one("#flashcard-refresh-button", Button).press()
        await _wait_until(pilot, fetched.is_set, what="the Refresh card fetch")
        # The fetch returns and the app quits in one step, in that order: the
        # rebuild resumes on the next loop turn with ``App._exit`` already
        # set, before the quit's own message is handled (both wake-ups queue
        # first in, first out).
        release.set()
        app.exit()
