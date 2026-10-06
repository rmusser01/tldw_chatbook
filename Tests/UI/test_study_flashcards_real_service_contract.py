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

The geometry tests load the app bundle (``TldwCli.CSS_PATH``) because a
harness screen without it measures nothing -- see
``backlog/docs/lessons-textual.md`` ("A geometry or `.display` test without
`CSS_PATH = BUNDLED_STYLESHEET` measures nothing").
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

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
    MATERIAL_SOURCE_LIBRARY,
    MATERIAL_TITLE_LIBRARY_SOURCES,
    StudyScopeContext,
)
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
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).flashcards_controller

        app.screen.query_one("#new-deck-name-input", Input).value = "Cell biology"
        await controller.create_deck()  # creates, selects, then refresh_cards()
        await pilot.pause(0.1)

        deck_select = app.screen.query_one("#deck-select", Select)
        deck_id = str(deck_select.value)
        assert db.get_deck(deck_id) is not None, "deck row missing from the DB"
        assert db.get_deck(deck_id)["name"] == "Cell biology"

        card_list = app.screen.query_one("#card-list", ListView)
        assert _card_list_labels(card_list) == ["No cards in this deck."]

        app.screen.query_one("#card-front", TextArea).text = "What is a ribosome?"
        app.screen.query_one("#card-back", TextArea).text = "The protein factory."
        app.screen.query_one("#card-tags", Input).value = "cells organelles"
        await controller.create_card()
        await pilot.pause(0.1)

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
        await pilot.pause(0.2)
        assert card_list.region.height >= 4, f"card list collapsed: {card_list.region}"
        row = next(
            item
            for item in card_list.children
            if getattr(item, "study_card_record", None) is not None
        )
        assert row.region.height > 0, f"card row has no paint: {row.region}"
        assert card_list.region.contains_region(row.region), (
            card_list.region,
            row.region,
        )

        # Re-selecting the deck is the second crash path (refresh_cards on
        # Select.Changed): it must list the same card, not raise.
        await controller.refresh_cards()
        await pilot.pause(0.1)
        assert any(
            "What is a ribosome?" in label for label in _card_list_labels(card_list)
        )


@pytest.mark.asyncio
async def test_real_service_workspace_scope_lists_and_creates_through_real_signatures():
    """Workspace scope (server mode) goes through the real ``ServerStudyService``:
    decks are scope-filtered by ``list_decks``; the card list and create for the
    selected deck pass only what the real signatures accept."""
    client = _FakeServerFlashcardsClient()
    app, _db = _build_real_study_app(runtime="server", server_client=client)

    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.2)
        app.screen.enter_workspace_scope("workspace-1", "Workspace One")
        await pilot.pause(0.5)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).flashcards_controller

        deck_select = app.screen.query_one("#deck-select", Select)
        option_values = [
            option[1]
            for option in deck_select._options
            if not str(option[1]).startswith("Select.")
        ]
        assert option_values == ["8"], option_values  # the workspace deck only

        deck_select.value = "8"
        await pilot.pause(0.1)
        await controller.refresh_cards()
        card_list = app.screen.query_one("#card-list", ListView)
        assert _card_list_labels(card_list) == ["No cards in this deck."]

        app.screen.query_one("#card-front", TextArea).text = "Workspace front"
        app.screen.query_one("#card-back", TextArea).text = "Workspace back"
        await controller.create_card()
        await pilot.pause(0.1)

        created = [call for call in client.calls if call[0] == "create_flashcard"]
        assert len(created) == 1 and created[0][1]["deck_id"] == 8
        assert any("Workspace front" in label for label in _card_list_labels(card_list))


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
        material_titles=tuple(f"Title {index}" for index in range(12)),
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
        await pilot.pause(0.3)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (160, 70), (120, 36)])
async def test_flashcards_tab_shows_deck_picker_and_card_editor_controls(size):
    """RED on origin/dev: ``.card-editor`` had no ``height: auto`` so it took
    ``1fr`` of the scroll container and hid everything after "Decks:"."""
    app, _db = _build_real_study_app()

    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

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
        await pilot.pause(0.2)
        _assert_inside_screen(create_button, size)
