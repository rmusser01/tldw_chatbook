"""Study ▸ Flashcards against the REAL ``StudyScopeService`` (TASK-34000.6, S-05).

Every earlier Flashcards test drove the handler through a hand-written fake
whose ``list_flashcards``/``create_flashcard`` accepted ``scope_type`` and
``workspace_id`` -- keywords the real ``StudyScopeService`` rejects. The fake
preserved the handler's wrong contract, so selecting or creating a deck
crashed the real app with ``TypeError`` for months while the suite stayed
green. These tests build the full ``TldwCli`` with the production
``StudyScopeService`` over a real in-memory ``CharactersRAGDB`` (local) and
over the real ``ServerStudyService`` (workspace scope), so a keyword the
service does not accept fails here first.

The dashboard/editor geometry tests (160x45, 160x70, 120x36) live in the
non-gated sibling ``Tests/UI/test_study_controls_geometry.py``, which imports
this file's helpers: with them this file measured at the PR UI lane's 60 s
budget.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import pytest
from textual.widgets import Input, ListView, Select, Static, TextArea

from Tests.app_module_patches import set_app_global
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_study_dashboard import DashboardQuizScopeService
import tldw_chatbook.app as app_module
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService
from tldw_chatbook.Study_Interop.server_study_service import ServerStudyService
from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Screens.study_scope_models import (
    StudyScopeContext,
    StudyScopeType,
)
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Study_Window import StudyWindow

# Every test here mounts the full app and rebuilds app config in its body, so
# the per-test sandbox's config re-read trips ADR-126's admission guard
# (`RecoveryRequired: raw_source_selection_changed`). The repo's own marker
# keeps the collection-time profile for these nodes (lessons-testing-evidence:
# "Tests/UI `RecoveryRequired` at setup is a profile-selection trip").
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(autouse=True)
def _disable_full_app_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    set_app_global(monkeypatch, "get_cli_setting", get_cli_setting_without_splash)


class _FakeServerFlashcardsClient:
    """The HTTP client under the real ``ServerStudyService``.

    This is the one seam that is faked: the wire. The service, the scope
    router and the handler are the production objects, so every keyword the
    handler passes is checked by the real signatures.
    """

    def __init__(self) -> None:
        self.decks: list[dict[str, Any]] = [
            {"id": 7, "name": "Global Biology", "workspace_id": None, "version": 1},
            {
                "id": 8,
                "name": "Workspace Biology",
                "workspace_id": "workspace-1",
                "version": 1,
            },
        ]
        self.cards: list[dict[str, Any]] = []
        self.calls: list[tuple[str, Any]] = []

    async def list_flashcard_decks(self, *, limit: int = 100, offset: int = 0):
        self.calls.append(("list_flashcard_decks", (limit, offset)))
        return list(self.decks[offset : offset + limit])

    async def create_flashcard_deck(self, request: Any):
        payload = request.model_dump(mode="json")
        self.calls.append(("create_flashcard_deck", payload))
        deck = {
            "id": 9,
            "name": payload["name"],
            "workspace_id": payload.get("workspace_id"),
            "version": 1,
        }
        self.decks.append(deck)
        return deck

    async def list_flashcards(
        self, *, deck_id=None, q=None, limit: int = 100, offset: int = 0
    ):
        self.calls.append(("list_flashcards", (deck_id, q, limit, offset)))
        items = [
            card for card in self.cards if deck_id is None or card["deck_id"] == deck_id
        ]
        return {"items": items[offset : offset + limit]}

    async def create_flashcard(self, request: Any):
        payload = request.model_dump(mode="json")
        self.calls.append(("create_flashcard", payload))
        card = {
            "uuid": f"card-{len(self.cards) + 1}",
            "deck_id": payload["deck_id"],
            "front": payload["front"],
            "back": payload["back"],
            "tags": payload.get("tags") or [],
            "version": 1,
        }
        self.cards.append(card)
        return card


def _build_real_study_app(
    *,
    runtime: str = "local",
    scope_context: StudyScopeContext | None = None,
    server_client: Any | None = None,
):
    """Full production app, real DB, real StudyScopeService."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "study"
    db = attach_chachanotes_db(app)
    app.study_scope_service = StudyScopeService(
        local_service=LocalStudyService(db=db),
        server_service=ServerStudyService(client=server_client),
    )
    app.study_quiz_scope_service = DashboardQuizScopeService()
    runtime_state = RuntimeSourceState(
        active_source=runtime,
        server_configured=runtime == "server",
    )
    app.runtime_policy.state = runtime_state
    app._publish_runtime_policy_projection(runtime_state)
    if scope_context is not None:
        app.pending_handoffs.stage(HandoffChannel.STUDY_SCOPE, scope_context)
    return app, db


def _text(widget) -> str:
    return str(widget.render())


#: Bound on every poll below; pytest-timeout still bounds the test as a whole.
_WAIT_SECONDS = 15.0


async def _wait_until(pilot, predicate: Callable[[], bool], *, what: str) -> None:
    """Poll ``predicate`` with a deadline instead of sleeping a fixed time.

    Review 1 (Minor 3): this file is in the PR UI lane, where a loaded runner
    turns a fixed ``pilot.pause(0.3)`` into a flake. A predicate that raises
    (widget not mounted yet) counts as "not yet".
    """
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception:
            pass
        await pilot.pause(0.02)
    raise AssertionError(f"timed out after {_WAIT_SECONDS:.0f}s waiting for {what}")


def _study_screen_up(app) -> bool:
    return isinstance(app.screen, StudyScreen) and bool(
        app.screen.query("#view-flashcards-btn")
    )


def _flashcards_view_up(app) -> bool:
    window = app.screen.query_one(StudyWindow)
    return (
        app.screen.current_section == "flashcards"
        and window.display
        and window.current_view == "flashcards"
        and bool(app.screen.query("#deck-select"))
    )


def _deck_options(app) -> list[str]:
    return [
        option[1]
        for option in app.screen.query_one("#deck-select", Select)._options
        if not str(option[1]).startswith("Select.")
    ]


def _is_blank(value: Any) -> bool:
    return value in {None, "", False} or str(value).startswith("Select.")


def _study_workers_pending(app) -> bool:
    """True while a Study worker (e.g. a ``#card-list`` rebuild) is queued or running.

    ``create_deck`` selects the new deck, and that ``Select.Changed`` makes
    ``StudyWindow.handle_deck_select_changed`` rebuild ``#card-list`` a second
    time in a "study-refresh-cards" worker, which is usually still running
    when ``create_deck()`` returns.
    """
    return any(
        str(worker.group).startswith("study-") and not worker.is_finished
        for worker in app.workers
    )


def _card_list_labels(list_view: ListView) -> list[str]:
    labels: list[str] = []
    for item in list_view.children:
        labels.extend(_text(child) for child in item.children)
    return labels


# --- AC#1 / AC#5: create deck ▸ select deck ▸ add card, real signatures -----


@pytest.mark.asyncio
async def test_real_service_create_deck_select_deck_and_add_card_in_local_mode():
    """RED on origin/dev: ``refresh_cards`` spreads ``_scope_arguments()`` into
    ``StudyScopeService.list_flashcards``, which has no ``scope_type`` ->
    ``TypeError`` the moment a deck is selected; ``create_card`` had the same
    call waiting behind it. GREEN: the deck's (empty) card list renders, the
    card lands in the DB and the list shows it.
    """
    app, db = _build_real_study_app()

    async with app.run_test(size=(160, 45)) as pilot:
        await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
        await pilot.click("#view-flashcards-btn")
        await _wait_until(
            pilot, lambda: _flashcards_view_up(app), what="the Flashcards view"
        )
        controller = app.screen.query_one(StudyWindow).flashcards_controller
        # The empty-DB deck load has settled once the status names the empty scope.
        await _wait_until(
            pilot,
            lambda: (
                "No study decks yet"
                in _text(app.screen.query_one("#review-status", Static))
            ),
            what="the initial deck load",
        )

        app.screen.query_one("#new-deck-name-input", Input).value = "Cell biology"
        await controller.create_deck()  # creates, selects, then refresh_cards()
        deck_select = app.screen.query_one("#deck-select", Select)
        await _wait_until(
            pilot,
            lambda: not _is_blank(deck_select.value),
            what="the new deck to be selected",
        )
        deck_id = str(deck_select.value)
        assert db.get_deck(deck_id) is not None, "deck row missing from the DB"
        assert db.get_deck(deck_id)["name"] == "Cell biology"

        card_list = app.screen.query_one("#card-list", ListView)
        # Not read the instant create_deck() returns: the deck switch's own
        # rebuild of #card-list is still in flight, and reading between its
        # clear and its append saw `[]` (CI UI Fast Lane, 2026-10-10).
        await _wait_until(
            pilot,
            lambda: not _study_workers_pending(app),
            what="the deck switch's card-list rebuild to finish",
        )
        assert _card_list_labels(card_list) == ["No cards in this deck."]

        app.screen.query_one("#card-front", TextArea).text = "What is a ribosome?"
        app.screen.query_one("#card-back", TextArea).text = "The protein factory."
        app.screen.query_one("#card-tags", Input).value = "cells organelles"
        await controller.create_card()
        await _wait_until(
            pilot,
            lambda: any(
                "What is a ribosome?" in label for label in _card_list_labels(card_list)
            ),
            what="the created card in the list",
        )

        rows = db.list_flashcards(deck_id=deck_id, q=None, limit=100, offset=0)
        assert [(row["front"], row["back"]) for row in rows] == [
            ("What is a ribosome?", "The protein factory.")
        ], "card row missing from the DB"

        labels = _card_list_labels(card_list)
        assert any("What is a ribosome?" in label for label in labels), labels

        # Shown, not merely appended: the app-wide `.card-list { height: 1fr }`
        # rule collapsed this list to its two border rows (live capture
        # 07-card-list-160x45 before the re-key), so the row had no paint.
        card_list.scroll_visible(animate=False)
        row = next(
            item
            for item in card_list.children
            if getattr(item, "study_card_record", None) is not None
        )
        await _wait_until(
            pilot, lambda: row.region.height > 0, what="the card row to be painted"
        )
        assert card_list.region.height >= 4, f"card list collapsed: {card_list.region}"
        assert card_list.region.contains_region(row.region), (
            card_list.region,
            row.region,
        )

        # Re-selecting the deck is the second crash path (refresh_cards on
        # Select.Changed): it must list the same card, not raise.
        await controller.refresh_cards()
        await _wait_until(
            pilot,
            lambda: any(
                "What is a ribosome?" in label for label in _card_list_labels(card_list)
            ),
            what="the card list after a re-list",
        )


@pytest.mark.asyncio
async def test_real_service_workspace_scope_lists_and_creates_through_real_signatures():
    """Workspace scope (server mode) goes through the real ``ServerStudyService``:
    decks are scope-filtered by ``list_decks``; the card list and create for the
    selected deck pass only what the real signatures accept."""
    client = _FakeServerFlashcardsClient()
    app, _db = _build_real_study_app(runtime="server", server_client=client)

    async with app.run_test(size=(160, 45)) as pilot:
        await _wait_until(pilot, lambda: _study_screen_up(app), what="the Study screen")
        app.screen.enter_workspace_scope("workspace-1", "Workspace One")
        await _wait_until(
            pilot,
            lambda: (
                app.screen.scope_state.scope_type == StudyScopeType.WORKSPACE
                and app.screen.scope_state.workspace_id == "workspace-1"
            ),
            what="the workspace scope to apply",
        )
        await pilot.click("#view-flashcards-btn")
        await _wait_until(
            pilot, lambda: _flashcards_view_up(app), what="the Flashcards view"
        )
        controller = app.screen.query_one(StudyWindow).flashcards_controller

        # the workspace deck only (scope-filtered by the real list_decks)
        await _wait_until(
            pilot, lambda: _deck_options(app) == ["8"], what="the workspace deck list"
        )
        deck_select = app.screen.query_one("#deck-select", Select)

        # Selecting the deck IS the crash path: the Select.Changed handler
        # runs refresh_cards (the TypeError site) on its own; wait for ITS
        # result rather than calling refresh_cards a second time alongside it.
        deck_select.value = "8"
        card_list = app.screen.query_one("#card-list", ListView)
        await _wait_until(
            pilot,
            lambda: (
                str(deck_select.value) == "8"
                and _card_list_labels(card_list) == ["No cards in this deck."]
            ),
            what="the selected deck's (empty) card list",
        )

        app.screen.query_one("#card-front", TextArea).text = "Workspace front"
        app.screen.query_one("#card-back", TextArea).text = "Workspace back"
        await controller.create_card()
        await _wait_until(
            pilot,
            lambda: any(
                "Workspace front" in label for label in _card_list_labels(card_list)
            ),
            what="the created workspace card in the list",
        )

        created = [call for call in client.calls if call[0] == "create_flashcard"]
        assert len(created) == 1 and created[0][1]["deck_id"] == 8
