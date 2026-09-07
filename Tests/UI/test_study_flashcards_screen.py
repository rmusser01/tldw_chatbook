import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_study_dashboard import DashboardQuizScopeService
import tldw_chatbook.app as app_module
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeState, StudyScopeType
from tldw_chatbook.UI.Study_Window import StudyWindow


class FakeStudyScopeService:
    def __init__(self):
        self.calls = []
        self.decks = [
            {
                "record_id": "local:study_deck:deck-local-1",
                "backing_id": "deck-local-1",
                "name": "Biology",
                "version": 3,
            },
            {
                "record_id": "local:study_deck:deck-local-2",
                "backing_id": "deck-local-2",
                "name": "Chemistry",
                "version": 5,
            },
        ]
        self.cards = [
            {
                "record_id": "local:study_flashcard:card-local-1",
                "backing_id": "card-local-1",
                "deck_record_id": "local:study_deck:deck-local-1",
                "front": "Question",
                "back": "Answer",
                "queue_state": "new",
                "version": 7,
            },
            {
                "record_id": "local:study_flashcard:card-local-2",
                "backing_id": "card-local-2",
                "deck_record_id": "local:study_deck:deck-local-1",
                "front": "Second Question",
                "back": "Second Answer",
                "queue_state": "new",
                "version": 11,
            },
        ]
        self.candidates = [
            {
                "card": {
                    "record_id": "server:study_flashcard:card-server-1",
                    "backing_id": "card-server-1",
                    "deck_record_id": "server:study_deck:7",
                    "front": "Question",
                    "back": "Answer",
                    "queue_state": "new",
                },
                "selection_reason": "new",
                "next_intervals": {"again": "10m", "good": "1d"},
                "review_session": None,
                "detail_available": True,
            },
            {
                "card": None,
                "selection_reason": "none",
                "next_intervals": None,
                "review_session": None,
                "detail_available": False,
            },
        ]

    async def list_decks(
        self, *, mode=None, scope_type=None, workspace_id=None, limit=100, offset=0
    ):
        self.calls.append(("list_decks", mode, scope_type, workspace_id, limit, offset))
        return list(self.decks)

    async def create_deck(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        name,
        description=None,
        scheduler_type=None,
    ):
        self.calls.append(
            (
                "create_deck",
                mode,
                scope_type,
                workspace_id,
                name,
                description,
                scheduler_type,
            )
        )
        created = {
            "record_id": f"{mode}:study_deck:new-deck",
            "backing_id": "new-deck",
            "name": name,
        }
        self.decks.append(created)
        return created

    async def list_flashcards(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        deck_id=None,
        q=None,
        limit=100,
        offset=0,
    ):
        self.calls.append(
            (
                "list_flashcards",
                mode,
                scope_type,
                workspace_id,
                deck_id,
                q,
                limit,
                offset,
            )
        )
        return [
            card
            for card in self.cards
            if deck_id is None or card["deck_record_id"].endswith(str(deck_id))
        ]

    async def create_flashcard(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        deck_id=None,
        front,
        back,
        tags=None,
        notes=None,
        extra=None,
    ):
        self.calls.append(
            (
                "create_flashcard",
                mode,
                scope_type,
                workspace_id,
                deck_id,
                front,
                back,
                tags,
                notes,
                extra,
            )
        )
        created = {
            "record_id": f"{mode}:study_flashcard:new-card",
            "backing_id": "new-card",
            "deck_record_id": f"{mode}:study_deck:{deck_id}",
            "front": front,
            "back": back,
            "queue_state": "new",
        }
        self.cards.append(created)
        return created

    async def move_flashcard(
        self, *, mode=None, card_id=None, target_deck_id=None, expected_version=None
    ):
        self.calls.append(
            ("move_flashcard", mode, card_id, target_deck_id, expected_version)
        )
        for card in self.cards:
            if card["backing_id"] == card_id:
                card["deck_record_id"] = f"{mode}:study_deck:{target_deck_id}"
                card["version"] = (card.get("version") or 0) + 1
                return card
        return None

    async def delete_flashcard(
        self, *, mode=None, card_id=None, expected_version=None, hard_delete=False
    ):
        self.calls.append(
            ("delete_flashcard", mode, card_id, expected_version, hard_delete)
        )
        self.cards = [card for card in self.cards if card["backing_id"] != card_id]
        return {"deleted": True}

    async def delete_deck(
        self, *, mode=None, deck_id=None, expected_version=None, hard_delete=False
    ):
        self.calls.append(("delete_deck", mode, deck_id, expected_version, hard_delete))
        self.decks = [deck for deck in self.decks if deck["backing_id"] != deck_id]
        self.cards = [
            card
            for card in self.cards
            if not str(card["deck_record_id"]).endswith(str(deck_id))
        ]
        return {"deleted": True}

    async def get_next_review_candidate(
        self, *, mode=None, scope_type=None, workspace_id=None, deck_id=None
    ):
        self.calls.append(
            ("get_next_review_candidate", mode, scope_type, workspace_id, deck_id)
        )
        if self.candidates:
            return self.candidates.pop(0)
        return {
            "card": None,
            "selection_reason": "none",
            "next_intervals": None,
            "review_session": None,
            "detail_available": False,
        }

    async def submit_flashcard_review(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        card_id=None,
        rating,
        current_card=None,
        answer_time_ms=None,
    ):
        self.calls.append(
            ("submit_flashcard_review", mode, scope_type, workspace_id, card_id, rating)
        )
        return {
            "card": {
                **(current_card or {}),
                "interval_days": 3,
                "queue_state": "review",
            },
            "rating": rating,
            "next_intervals": {"again": "10m", "good": "3d"},
            "review_session": {"review_session_id": 41},
            "detail_available": True,
        }

    async def end_review_session(
        self, *, mode=None, scope_type=None, workspace_id=None, review_session_id=None
    ):
        self.calls.append(
            ("end_review_session", mode, scope_type, workspace_id, review_session_id)
        )
        return {"id": review_session_id, "status": "completed"}


class EmptyStudyScopeService(FakeStudyScopeService):
    def __init__(self):
        super().__init__()
        self.decks = []
        self.cards = []
        self.candidates = []


class WorkspaceFilteredStudyScopeService(FakeStudyScopeService):
    def __init__(self):
        super().__init__()
        self.workspace_id = "workspace-1"
        self.decks = [
            {
                "record_id": "server:study_deck:deck-global-1",
                "backing_id": "deck-global-1",
                "name": "Global Biology",
                "version": 4,
            },
            {
                "record_id": "server:study_deck:deck-global-2",
                "backing_id": "deck-global-2",
                "name": "Global Chemistry",
                "version": 6,
            },
        ]
        self.cards = [
            {
                "record_id": "server:study_flashcard:card-global-1",
                "backing_id": "card-global-1",
                "deck_record_id": "server:study_deck:deck-global-1",
                "front": "Global question",
                "back": "Global answer",
                "queue_state": "new",
                "version": 4,
            },
            {
                "record_id": "server:study_flashcard:card-global-2",
                "backing_id": "card-global-2",
                "deck_record_id": "server:study_deck:deck-global-2",
                "front": "Second global question",
                "back": "Second global answer",
                "queue_state": "new",
                "version": 6,
            },
        ]
        self.workspace_decks = [
            {
                "record_id": "server:study_deck:deck-workspace-1",
                "backing_id": "deck-workspace-1",
                "name": "Workspace Biology",
                "workspace_id": self.workspace_id,
                "version": 9,
            }
        ]
        self.workspace_cards = [
            {
                "record_id": "server:study_flashcard:card-workspace-1",
                "backing_id": "card-workspace-1",
                "deck_record_id": "server:study_deck:deck-workspace-1",
                "front": "Workspace question",
                "back": "Workspace answer",
                "queue_state": "new",
                "version": 4,
            }
        ]

    async def list_decks(
        self, *, mode=None, scope_type=None, workspace_id=None, limit=100, offset=0
    ):
        self.calls.append(("list_decks", mode, scope_type, workspace_id, limit, offset))
        if scope_type == "workspace":
            assert workspace_id == self.workspace_id
            return list(self.workspace_decks)
        return list(self.decks)

    async def create_deck(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        name,
        description=None,
        scheduler_type=None,
    ):
        self.calls.append(
            (
                "create_deck",
                mode,
                scope_type,
                workspace_id,
                name,
                description,
                scheduler_type,
            )
        )
        created = {
            "record_id": "server:study_deck:new-workspace-deck",
            "backing_id": "new-workspace-deck",
            "name": name,
            "workspace_id": workspace_id if scope_type == "workspace" else None,
            "version": 1,
        }
        if scope_type == "workspace":
            self.workspace_decks.append(created)
        else:
            self.decks.append(created)
        return created

    async def list_flashcards(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        deck_id=None,
        q=None,
        limit=100,
        offset=0,
    ):
        self.calls.append(
            (
                "list_flashcards",
                mode,
                scope_type,
                workspace_id,
                deck_id,
                q,
                limit,
                offset,
            )
        )
        cards = self.workspace_cards if deck_id == "deck-workspace-1" else self.cards
        return [
            card
            for card in cards
            if deck_id is None or card["deck_record_id"].endswith(str(deck_id))
        ]


class FlakyEndReviewStudyScopeService(FakeStudyScopeService):
    def __init__(self):
        super().__init__()
        self.fail_end_review_calls = 1

    async def end_review_session(
        self, *, mode=None, scope_type=None, workspace_id=None, review_session_id=None
    ):
        self.calls.append(
            ("end_review_session", mode, scope_type, workspace_id, review_session_id)
        )
        if self.fail_end_review_calls > 0:
            self.fail_end_review_calls -= 1
            raise RuntimeError("failed to end review session")
        return {"id": review_session_id, "status": "completed"}


@pytest.fixture(autouse=True)
def _disable_full_app_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _build_full_study_app(app_instance):
    """Build the full production app with deterministic Study collaborators."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "study"
    app.study_scope_service = app_instance.study_scope_service
    app.study_quiz_scope_service = getattr(
        app_instance,
        "study_quiz_scope_service",
        DashboardQuizScopeService(),
    )
    app.notify = app_instance.notify
    source = str(getattr(app_instance, "current_runtime_backend", "local"))
    runtime_state = RuntimeSourceState(
        active_source=source,
        server_configured=source == "server",
    )
    app.runtime_policy.state = runtime_state
    app._publish_runtime_policy_projection(runtime_state)
    scope_context = getattr(app_instance, "scope_context", None)
    if scope_context is not None:
        app.pending_handoffs.stage(HandoffChannel.STUDY_SCOPE, scope_context)
    return app


def _text(widget) -> str:
    return str(widget.render())


def _is_blank(value) -> bool:
    return value in {None, "", False, Select.BLANK} or str(value).startswith("Select.")


def _non_blank_option_values(options: list[tuple]) -> list[str]:
    return [option[1] for option in options if not _is_blank(option[1])]


def _list_item_for_card(list_view: ListView, backing_id: str):
    for item in list_view.children:
        record = getattr(item, "study_card_record", None)
        if isinstance(record, dict) and record.get("backing_id") == backing_id:
            return item
    raise AssertionError(f"No list item found for card {backing_id}")


@pytest.mark.asyncio
async def test_study_screen_passes_app_instance_to_study_window():
    app_instance = SimpleNamespace(
        study_scope_service=FakeStudyScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        window = app.screen.query_one(StudyWindow)
        assert window.app_instance is app
        assert app.study_scope_service is app_instance.study_scope_service


@pytest.mark.asyncio
async def test_flashcards_view_loads_scope_backed_decks_without_default_fallback():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        deck_select = app.screen.query_one("#deck-select", Select)
        status = app.screen.query_one("#review-status", Static)

        assert all(
            getattr(option, "value", None) != "default"
            for option in deck_select._options
        )
        assert ("list_decks", "local", "global", None, 100, 0) in scope.calls
        assert "Select a deck" in _text(status)


@pytest.mark.asyncio
async def test_flashcards_view_creates_deck_and_card_through_scope_service():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).flashcards_controller

        app.screen.query_one("#new-deck-name-input", Input).value = "Chemistry"
        await controller.create_deck()

        deck_select = app.screen.query_one("#deck-select", Select)
        assert str(deck_select.value) == "new-deck"

        front = app.screen.query_one("#card-front", TextArea)
        back = app.screen.query_one("#card-back", TextArea)
        tags = app.screen.query_one("#card-tags", Input)
        front.text = "What is H2O?"
        back.text = "Water"
        tags.value = "chemistry water"

        await controller.create_card()

        card_list = app.screen.query_one("#card-list", ListView)

        assert (
            "create_deck",
            "local",
            "global",
            None,
            "Chemistry",
            None,
            None,
        ) in scope.calls
        assert (
            "create_flashcard",
            "local",
            "global",
            None,
            "new-deck",
            "What is H2O?",
            "Water",
            ["chemistry", "water"],
            None,
            None,
        ) in scope.calls
        assert card_list.children


@pytest.mark.asyncio
async def test_flashcards_view_exposes_delete_and_move_controls():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        deck_delete_button = app.screen.query_one("#delete-deck-button", Button)
        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_selected_button = app.screen.query_one(
            "#move-selected-card-button", Button
        )
        delete_selected_button = app.screen.query_one(
            "#delete-selected-card-button", Button
        )

        assert deck_delete_button is not None
        assert move_target_select is not None
        assert move_selected_button is not None
        assert delete_selected_button is not None


@pytest.mark.asyncio
async def test_server_mode_keeps_delete_deck_visible_but_disabled():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        delete_deck_button = app.screen.query_one("#delete-deck-button", Button)
        delete_note = app.screen.query_one("#delete-deck-note", Static)

        assert delete_deck_button.display is True
        assert delete_deck_button.disabled is True
        assert delete_note.display is True
        assert "server" in _text(delete_note).lower()
        assert "delete" in _text(delete_note).lower()
        assert "Server mode does not support deck deletion" in str(
            delete_deck_button.tooltip
        )


@pytest.mark.asyncio
async def test_flashcards_lifecycle_controls_noop_handlers_do_not_raise():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        window = app.screen.query_one(StudyWindow)
        await window.handle_delete_deck()
        await window.handle_move_selected_card()
        await window.handle_delete_selected_card()
        window.handle_move_card_target_changed(SimpleNamespace())


@pytest.mark.asyncio
async def test_flashcards_lifecycle_controls_disable_without_selected_card_or_target():
    scope = FakeStudyScopeService()
    scope.decks = [scope.decks[0]]
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        move_selected_button = app.screen.query_one(
            "#move-selected-card-button", Button
        )
        delete_selected_button = app.screen.query_one(
            "#delete-selected-card-button", Button
        )

        assert delete_selected_button.disabled is True
        assert move_selected_button.disabled is True
        assert "Select a flashcard before deleting it" in str(
            delete_selected_button.tooltip
        )
        assert "Select a flashcard and a different target deck" in str(
            move_selected_button.tooltip
        )


@pytest.mark.asyncio
async def test_delete_selected_card_uses_selected_card_version_and_refreshes_list():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()
        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(
            SimpleNamespace(item=_list_item_for_card(card_list, "card-local-1"))
        )
        controller.current_review_card = controller.selected_card_record
        controller.current_review_session_id = 41

        await controller.delete_selected_card()
        await pilot.pause(0.3)

        assert ("delete_flashcard", "server", "card-local-1", 7, False) in scope.calls
        assert ("end_review_session", "server", "global", None, 41) in scope.calls
        assert len(scope.cards) == 1
        assert scope.cards[0]["backing_id"] == "card-local-2"
        assert "No cards in this deck." not in _text(
            app.screen.query_one("#card-list", ListView).children[0].children[0]
        )


@pytest.mark.asyncio
async def test_move_selected_card_refreshes_current_deck_and_exits_review_when_needed():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()
        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(
            SimpleNamespace(item=_list_item_for_card(card_list, "card-local-1"))
        )
        controller.current_review_card = controller.selected_card_record
        controller.current_review_session_id = 41

        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_target_select.value = "deck-local-2"
        await controller.move_selected_card()
        await pilot.pause(0.3)

        assert (
            "move_flashcard",
            "server",
            "card-local-1",
            "deck-local-2",
            7,
        ) in scope.calls
        assert ("end_review_session", "server", "global", None, 41) in scope.calls
        assert any(
            card["backing_id"] == "card-local-1"
            and card["deck_record_id"].endswith("deck-local-2")
            for card in scope.cards
        )
        assert any(
            card["backing_id"] == "card-local-2"
            and card["deck_record_id"].endswith("deck-local-1")
            for card in scope.cards
        )
        assert _text(app.screen.query_one("#review-status", Static)) != ""


@pytest.mark.asyncio
async def test_delete_selected_card_preserves_unrelated_active_review_state():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(
            SimpleNamespace(item=_list_item_for_card(card_list, "card-local-2"))
        )

        controller.current_review_card = dict(
            next(card for card in scope.cards if card["backing_id"] == "card-local-1")
        )
        controller.current_review_session_id = 41
        controller._set_review_status("Next card (new).")
        controller._set_review_card(front="Question", back="Answer", show_back=False)

        await controller.delete_selected_card()
        await pilot.pause(0.3)

        assert ("delete_flashcard", "server", "card-local-2", 11, False) in scope.calls
        assert controller.current_review_card["backing_id"] == "card-local-1"
        assert controller.current_review_session_id == 41
        assert "Next card" in _text(app.screen.query_one("#review-status", Static))


@pytest.mark.asyncio
async def test_move_selected_card_preserves_unrelated_active_review_state():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(
            SimpleNamespace(item=_list_item_for_card(card_list, "card-local-2"))
        )

        controller.current_review_card = dict(
            next(card for card in scope.cards if card["backing_id"] == "card-local-1")
        )
        controller.current_review_session_id = 41
        controller._set_review_status("Next card (new).")
        controller._set_review_card(front="Question", back="Answer", show_back=False)

        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_target_select.value = "deck-local-2"
        await controller.move_selected_card()
        await pilot.pause(0.3)

        assert (
            "move_flashcard",
            "server",
            "card-local-2",
            "deck-local-2",
            11,
        ) in scope.calls
        assert controller.current_review_card["backing_id"] == "card-local-1"
        assert controller.current_review_session_id == 41
        assert "Next card" in _text(app.screen.query_one("#review-status", Static))


@pytest.mark.asyncio
async def test_lifecycle_actions_reconcile_live_deck_after_deck_change_before_refresh_finishes():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        window = app.screen.query_one(StudyWindow)
        controller = window.flashcards_controller

        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.handle_deck_changed()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(
            SimpleNamespace(item=_list_item_for_card(card_list, "card-local-1"))
        )
        assert controller.selected_card_record["backing_id"] == "card-local-1"

        deck_select.value = "deck-local-2"
        window.handle_deck_select_changed(SimpleNamespace())

        await controller.delete_selected_card()
        await controller.move_selected_card()
        await pilot.pause(0.3)

        assert not any(call[0] == "delete_flashcard" for call in scope.calls)
        assert not any(call[0] == "move_flashcard" for call in scope.calls)
        assert any(
            card["backing_id"] == "card-local-1"
            and card["deck_record_id"].endswith("deck-local-1")
            for card in scope.cards
        )
        assert controller.selected_card_record is None


@pytest.mark.asyncio
async def test_local_delete_deck_uses_selected_deck_version_and_resets_review_state():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        controller.current_review_card = dict(scope.cards[0])
        controller.current_review_session_id = 41
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()

        await controller.delete_selected_deck()
        await pilot.pause(0.3)

        assert ("delete_deck", "local", "deck-local-1", 3, False) in scope.calls
        assert controller.selected_deck_record is None
        assert controller.selected_card_record is None
        assert controller.current_review_card is None
        assert controller.current_review_session_id is None
        assert _is_blank(app.screen.query_one("#deck-select", Select).value)
        assert _text(app.screen.query_one("#review-front", Static)) == ""
        assert _text(app.screen.query_one("#review-back", Static)) == ""
        assert "Select a deck" in _text(app.screen.query_one("#review-status", Static))


@pytest.mark.asyncio
async def test_flashcards_review_flow_uses_scope_service_and_ends_server_session_when_queue_empties():
    scope = FakeStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        controller = app.screen.query_one(StudyWindow).flashcards_controller

        await controller.start_review()
        await pilot.pause(0.3)

        review_front = app.screen.query_one("#review-front", Static)
        review_back = app.screen.query_one("#review-back", Static)
        assert "Question" in _text(review_front)
        assert review_back.display is False

        controller.show_answer()
        await pilot.pause(0.1)
        assert review_back.display is True

        await controller.submit_rating(4)
        await pilot.pause(0.3)

        status = app.screen.query_one("#review-status", Static)

        assert (
            "submit_flashcard_review",
            "server",
            "global",
            None,
            "card-server-1",
            4,
        ) in scope.calls
        assert ("end_review_session", "server", "global", None, 41) in scope.calls
        assert "No cards due" in _text(status)


@pytest.mark.asyncio
async def test_flashcards_view_shows_explicit_empty_state_when_no_decks_exist():
    app_instance = SimpleNamespace(
        study_scope_service=EmptyStudyScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        status = app.screen.query_one("#review-status", Static)
        create_button = app.screen.query_one("#create-deck-button", Button)

        status_text = _text(status)
        assert "No study decks yet." in status_text
        assert "Create a deck" in status_text
        assert "add flashcards" in status_text
        assert "start reviewing" in status_text
        assert create_button.display is True


@pytest.mark.asyncio
async def test_workspace_flashcards_scope_uses_workspace_filtered_decks_and_server_scoped_create():
    scope = WorkspaceFilteredStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        app.screen.enter_workspace_scope("workspace-1", "Workspace One")
        await pilot.pause(0.5)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        create_button = app.screen.query_one("#create-deck-button", Button)

        assert (
            "list_decks",
            "server",
            "workspace",
            "workspace-1",
            100,
            0,
        ) in scope.calls
        assert _non_blank_option_values(deck_select._options) == ["deck-workspace-1"]
        assert _non_blank_option_values(move_target_select._options) == [
            "deck-workspace-1"
        ]
        assert create_button.disabled is False

        app.screen.query_one("#new-deck-name-input", Input).value = "New Workspace Deck"
        await controller.create_deck()
        await pilot.pause(0.1)

        assert (
            "create_deck",
            "server",
            "workspace",
            "workspace-1",
            "New Workspace Deck",
            None,
            None,
        ) in scope.calls
        assert str(deck_select.value) == "new-workspace-deck"
        assert _non_blank_option_values(move_target_select._options) == [
            "deck-workspace-1"
        ]


@pytest.mark.asyncio
async def test_workspace_flashcards_local_mode_fail_closed_ui_state():
    scope = WorkspaceFilteredStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        app.screen.enter_workspace_scope("workspace-1", "Workspace One")
        await pilot.pause(0.5)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        deck_select = app.screen.query_one("#deck-select", Select)
        review_status = app.screen.query_one("#review-status", Static)
        create_deck_button = app.screen.query_one("#create-deck-button", Button)
        create_card_button = app.screen.query_one("#create-card-btn", Button)
        start_review_button = app.screen.query_one("#start-review-btn", Button)
        move_selected_button = app.screen.query_one(
            "#move-selected-card-button", Button
        )
        delete_selected_button = app.screen.query_one(
            "#delete-selected-card-button", Button
        )
        delete_deck_button = app.screen.query_one("#delete-deck-button", Button)

        assert not any(
            call[0] == "list_decks" and call[2] == "workspace" for call in scope.calls
        )
        assert "server mode" in _text(review_status).lower()
        assert _is_blank(deck_select.value)
        assert create_deck_button.disabled is True
        assert create_card_button.disabled is True
        assert start_review_button.disabled is True
        assert move_selected_button.disabled is True
        assert delete_selected_button.disabled is True
        assert delete_deck_button.disabled is True
        for button in (
            create_deck_button,
            create_card_button,
            start_review_button,
            move_selected_button,
            delete_selected_button,
            delete_deck_button,
        ):
            assert "Workspace Flashcards require server mode" in str(button.tooltip)


@pytest.mark.asyncio
async def test_scope_transition_resets_review_state_and_clears_flashcards_panel():
    scope = WorkspaceFilteredStudyScopeService()
    scope.workspace_decks = []
    scope.workspace_cards = []
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-global-1"
        await controller.refresh_cards()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(
            SimpleNamespace(item=_list_item_for_card(card_list, "card-global-1"))
        )
        controller.current_review_card = dict(scope.cards[0])
        controller.current_review_session_id = 41
        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_target_select.value = "deck-global-2"
        controller._set_review_status("Next card (new).")
        controller._set_review_card(front="Question", back="Answer", show_back=False)

        app.screen.scope_state = StudyScopeState(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="workspace-1",
            workspace_name="Workspace One",
            backend="server",
            workspace_scope_available=True,
        )
        await controller.end_review_session_if_needed()
        controller.handle_scope_changed()
        await controller.refresh_decks()
        await controller.refresh_cards()

        review_status = app.screen.query_one("#review-status", Static)
        review_front = app.screen.query_one("#review-front", Static)
        review_back = app.screen.query_one("#review-back", Static)

        assert (
            "end_review_session",
            "server",
            "workspace",
            "workspace-1",
            41,
        ) in scope.calls
        assert controller.current_review_card is None
        assert controller.current_review_session_id is None
        assert controller.selected_deck_record is None
        assert controller.selected_card_record is None
        assert controller.current_cards == []
        assert controller.current_decks == []
        assert _is_blank(deck_select.value)
        assert _is_blank(move_target_select.value)
        assert _text(review_front) == ""
        assert _text(review_back) == ""
        status_text = _text(review_status)
        assert "No study decks in this workspace yet." in status_text
        assert "Create a workspace deck" in status_text
        assert "switch to Global Study" in status_text


@pytest.mark.asyncio
async def test_backend_flip_keeps_server_review_session_teardown_before_workspace_unavailable_reset():
    scope = WorkspaceFilteredStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        app.screen.enter_workspace_scope("workspace-1", "Workspace One")
        await pilot.pause(0.5)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        controller.current_review_card = dict(scope.workspace_cards[0])
        controller.current_review_session_id = 41
        controller._set_review_status("Next card (new).")
        controller._set_review_card(
            front="Workspace question", back="Workspace answer", show_back=False
        )

        await app.screen.handle_runtime_backend_changed("local")
        await pilot.pause(0.3)

        review_status = app.screen.query_one("#review-status", Static)

        assert (
            "end_review_session",
            "server",
            "workspace",
            "workspace-1",
            41,
        ) in scope.calls
        assert controller.current_review_session_id is None
        assert controller.current_review_card is None
        assert "server mode" in _text(review_status).lower()


@pytest.mark.asyncio
async def test_failed_end_review_session_is_retried_after_review_panel_reset():
    scope = FlakyEndReviewStudyScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        controller.current_review_card = {
            "record_id": "server:study_flashcard:card-server-1",
            "backing_id": "card-server-1",
            "deck_record_id": "server:study_deck:deck-local-1",
            "front": "Question",
            "back": "Answer",
            "queue_state": "new",
        }
        controller.current_review_session_id = 41
        controller.current_review_session_mode = "server"

        await controller.end_review_session_if_needed()
        controller.reset_review_panel("Selected flashcard moved.")

        assert controller.current_review_session_id is None

        await controller.end_review_session_if_needed()

        end_review_calls = [
            call
            for call in scope.calls
            if call == ("end_review_session", "server", "global", None, 41)
        ]
        assert len(end_review_calls) == 2


@pytest.mark.asyncio
async def test_start_review_blocks_when_pending_session_teardown_keeps_failing():
    scope = FlakyEndReviewStudyScopeService()
    scope.fail_end_review_calls = 2
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _build_full_study_app(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        controller.current_review_card = {
            "record_id": "server:study_flashcard:card-server-1",
            "backing_id": "card-server-1",
            "deck_record_id": "server:study_deck:deck-local-1",
            "front": "Question",
            "back": "Answer",
            "queue_state": "new",
        }
        controller.current_review_session_id = 41
        controller.current_review_session_mode = "server"

        await controller.end_review_session_if_needed()
        controller.reset_review_panel("Selected flashcard moved.")
        await controller.start_review()

        end_review_calls = [
            call
            for call in scope.calls
            if call == ("end_review_session", "server", "global", None, 41)
        ]
        assert len(end_review_calls) == 2
        assert not any(call[0] == "get_next_review_candidate" for call in scope.calls)
        assert controller._pending_review_session_teardown is not None



def _review_candidates(count: int = 6) -> list[dict]:
    """`count` distinct due cards, so a review queue can actually advance."""
    return [
        {
            "card": {
                "record_id": f"local:study_flashcard:card-local-{index}",
                "backing_id": f"card-local-{index}",
                "deck_record_id": "local:study_deck:deck-local-1",
                "front": f"Question {index}",
                "back": f"Answer {index}",
                "queue_state": "new",
            },
            "selection_reason": "new",
            "next_intervals": {"again": "10m", "good": "1d"},
            "review_session": {"review_session_id": 41},
            "detail_available": True,
        }
        for index in range(1, count + 1)
    ]


class GatedReviewStudyScopeService(FakeStudyScopeService):
    """A scope service whose review save can be held open mid-flight.

    `submit_flashcard_review` records the write only *after* the gate opens,
    so a submission that is cancelled at its await never appears in
    `persisted` -- which is exactly what "the rating did not reach the
    database" looks like from the user's side.
    """

    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()
        self.persisted: list[tuple[str | None, int]] = []
        self.candidates = _review_candidates()

    async def submit_flashcard_review(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        card_id=None,
        rating,
        current_card=None,
        answer_time_ms=None,
    ):
        await self.gate.wait()
        self.persisted.append((card_id, rating))
        return {
            "card": {
                **(current_card or {}),
                "interval_days": 3,
                "queue_state": "review",
            },
            "rating": rating,
            "next_intervals": {"again": "10m", "good": "3d"},
            "review_session": {"review_session_id": 41},
            "detail_available": True,
        }


class GatedCardListStudyScopeService(FakeStudyScopeService):
    """Holds `list_flashcards` open so two card-list rebuilds can interleave."""

    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()
        self.list_calls = 0
        self.candidates = _review_candidates()

    async def list_flashcards(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        deck_id=None,
        q=None,
        limit=100,
        offset=0,
    ):
        self.list_calls += 1
        await self.gate.wait()
        return [
            card
            for card in self.cards
            if deck_id is None or card["deck_record_id"].endswith(str(deck_id))
        ]


def _study_app_for(scope):
    app_instance = SimpleNamespace(
        study_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    return _build_full_study_app(app_instance)


async def _enter_review(pilot, app):
    """Open Flashcards, pick the deck, start a review and reveal the answer."""
    await pilot.pause(0.2)
    await pilot.click("#view-flashcards-btn")
    await pilot.pause(0.3)
    app.screen.query_one("#deck-select", Select).value = "deck-local-1"
    # Let the deck-change refresh settle before starting a review, so nothing
    # tears the card down underneath the test.
    await pilot.pause(0.5)
    window = app.screen.query_one(StudyWindow)
    controller = window.flashcards_controller
    await controller.start_review()
    await pilot.pause(0.2)
    controller.show_answer()
    await pilot.pause(0.1)
    return window, controller


@pytest.mark.asyncio
async def test_flashcard_rating_survives_a_sibling_study_worker():
    """TASK-19559(a): the named bug -- a sibling Study worker ate the save.

    `Study_Window.py:1007/1011` (create deck, refresh cards) and the
    `initialize_view` refresh at `914` were all `exclusive=True` with no
    `group=`, which put them in the shared "default" group alongside the
    rating submission. Pressing any of them while a rating was in flight
    cancelled the save, and `CancelledError` is a `BaseException` that
    `submit_rating`'s `except Exception:` cannot observe -- the rating simply
    vanished.

    Born red against the branch base: `persisted == []`.
    """
    scope = GatedReviewStudyScopeService()
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        window, _controller = await _enter_review(pilot, app)

        window.query_one("#review-rating-3", Button).press()
        await pilot.pause(0.2)
        assert scope.persisted == [], "the gate should still be holding the save"

        # A sibling Study worker starts while the save is in flight.
        window.query_one("#flashcard-refresh-button", Button).press()
        await pilot.pause(0.2)

        scope.gate.set()
        await pilot.pause(0.5)

        assert scope.persisted == [("card-local-1", 3)], (
            "a sibling Study worker cancelled the in-flight rating save; "
            f"persisted={scope.persisted}"
        )


@pytest.mark.asyncio
async def test_consecutive_ratings_on_distinct_cards_all_persist():
    """TASK-19559: rating card after card in quick succession loses nothing.

    This is the acceptance criterion's real content: *distinct* cards all
    persist. (Two presses on one card are a double-submit, not two reviews --
    see `test_double_press_on_one_card_applies_sm2_once`.)
    """
    scope = GatedReviewStudyScopeService()
    scope.gate.set()  # saves complete immediately; the user never waits
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        window, controller = await _enter_review(pilot, app)

        for _ in range(3):
            assert controller.current_review_card is not None
            window.query_one("#review-rating-4", Button).press()
            await pilot.pause(0.3)
            controller.show_answer()
            await pilot.pause(0.05)

        assert len(scope.persisted) == 3, f"persisted={scope.persisted}"
        assert [card_id for card_id, _rating in scope.persisted] == [
            "card-local-1",
            "card-local-2",
            "card-local-3",
        ]


@pytest.mark.asyncio
async def test_rating_in_flight_survives_leaving_the_flashcards_sub_view():
    """TASK-19559 review R1: switching sub-view mid-save must not crash.

    `StudyWindow.watch_current_view` calls `remove_children()` on the view
    container, destroying every widget `_set_review_status` /
    `_set_next_intervals` query with a bare `query_one`. Exclusivity used to
    hide this by cancelling the save first. Removing it exposed an unhandled
    `WorkerFailed(NoMatches(...))`, and the tail also re-assigned
    `current_review_session_id` for a session teardown had just ended.

    Base vs branch, identical probe -- base lost the rating and raised
    nothing; the pre-fix branch raised `WorkerFailed` and resurrected session
    41. The fix keeps the write *and* stays quiet.
    """
    scope = GatedReviewStudyScopeService()
    app = _study_app_for(scope)
    captured: list[str] = []

    async with app.run_test(size=(180, 60)) as pilot:
        app._handle_exception = lambda error: captured.append(repr(error))
        window, controller = await _enter_review(pilot, app)

        window.query_one("#review-rating-3", Button).press()
        await pilot.pause(0.2)

        # The user leaves Flashcards while the save is still in flight.
        window.current_view = "quizzes"
        await pilot.pause(0.4)

        scope.gate.set()
        await pilot.pause(0.6)

        assert captured == [], f"unhandled worker exception: {captured}"
        assert scope.persisted == [("card-local-1", 3)], (
            f"the rating did not survive the sub-view switch: {scope.persisted}"
        )
        assert controller.current_review_session_id is None, (
            "an ended review session was resurrected by the rating's tail"
        )


@pytest.mark.asyncio
async def test_deck_change_and_refresh_do_not_interleave_the_card_list():
    """TASK-19559 review R2: two card-list rebuilds must not interleave.

    `handle_deck_select_changed` was left ungrouped *and* non-exclusive, so it
    sat in the shared "default" group while `handle_refresh_cards` moved to
    `study-refresh-cards`. Base's ungrouped-exclusive refresh used to cancel
    the deck-change rebuild; afterwards both `refresh_cards()` bodies appended
    into `#card-list` together and the visible row count no longer matched
    `current_cards`.
    """
    scope = GatedCardListStudyScopeService()
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)
        window = app.screen.query_one(StudyWindow)
        controller = window.flashcards_controller
        scope.gate.set()
        await pilot.pause(0.3)

        # Hold both rebuilds open together.
        scope.gate.clear()
        app.screen.query_one("#deck-select", Select).value = "deck-local-1"
        await pilot.pause(0.1)
        window.query_one("#flashcard-refresh-button", Button).press()
        await pilot.pause(0.1)
        scope.gate.set()
        await pilot.pause(0.6)

        rows = len(window.query_one("#card-list", ListView).children)
        assert rows == len(controller.current_cards), (
            f"#card-list holds {rows} rows against "
            f"{len(controller.current_cards)} cards -- two rebuilds interleaved"
        )


@pytest.mark.asyncio
async def test_double_press_on_one_card_applies_sm2_once(tmp_path):
    """TASK-19559 review R3: SM-2 is compounding, so a double-submit doubles it.

    `ChaChaNotes_DB.update_flashcard_review` runs SM-2, which is *not*
    idempotent: applying it twice to one card moves `repetitions` 0 -> 1 -> 2
    and `interval` 1d -> 6d. Removing exclusivity turned a lost write into a
    doubled schedule, which is a different data defect, not a fix.

    Real DB, real Textual workers, two rapid presses on one card. Expected
    `repetitions=1, interval=1` (SM-2 applied exactly once, matching base);
    the pre-fix branch recorded `repetitions=2, interval=6`.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService

    db = CharactersRAGDB(str(tmp_path / "study.db"), "study-review-probe")
    deck_id = db.create_deck("Doubling deck")
    card_id = db.create_flashcard(
        {"deck_id": deck_id, "front": "Q", "back": "A"}
    )
    local = LocalStudyService(db)

    class RealDbStudyScopeService(FakeStudyScopeService):
        def __init__(self):
            super().__init__()
            self.gate = asyncio.Event()
            self.candidates = [
                {
                    "card": {
                        "record_id": f"local:study_flashcard:{card_id}",
                        "backing_id": card_id,
                        "deck_record_id": f"local:study_deck:{deck_id}",
                        "front": "Q",
                        "back": "A",
                        "queue_state": "new",
                    },
                    "selection_reason": "new",
                    "next_intervals": {"again": "10m", "good": "1d"},
                    "review_session": {"review_session_id": 41},
                    "detail_available": True,
                }
            ] * 4

        async def submit_flashcard_review(
            self,
            *,
            mode=None,
            scope_type=None,
            workspace_id=None,
            card_id=None,
            rating,
            current_card=None,
            answer_time_ms=None,
        ):
            await self.gate.wait()
            outcome = local.submit_flashcard_review(card_id, rating=rating)
            return {
                "card": outcome["card"],
                "rating": rating,
                "next_intervals": {"good": "3d"},
                "review_session": {"review_session_id": 41},
                "detail_available": True,
            }

    scope = RealDbStudyScopeService()
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        window, _controller = await _enter_review(pilot, app)

        window.query_one("#review-rating-3", Button).press()
        await pilot.pause(0.2)
        window.query_one("#review-rating-5", Button).press()
        await pilot.pause(0.2)
        scope.gate.set()
        await pilot.pause(0.8)

    row = db.get_flashcard(card_id)
    assert (row["repetitions"], row["interval"]) == (1, 1), (
        "SM-2 was applied more than once for a single card presentation: "
        f"repetitions={row['repetitions']} interval={row['interval']}"
    )


class RealDbReviewScopeService(FakeStudyScopeService):
    """Deals one real flashcard repeatedly and writes real SM-2 through it.

    The `gate` is a genuine suspension point *in front of* the SM-2 write. It
    stands in for the server backend, which is the only one where a rating can
    be cancelled with the write's fate unknown: the local backend reaches
    `ChaChaNotes_DB.update_flashcard_review` through `_maybe_await` without ever
    yielding to the loop, so a `CancelledError` delivered at that await means
    the write had not begun. Holding this gate open lets a test cancel the
    rating worker at a point where nothing has been written yet -- the case a
    retry must be able to recover.
    """

    def __init__(self, local, *, card_id: str, deck_id: str, deals: int = 4):
        super().__init__()
        self.local = local
        self.gate = asyncio.Event()
        self.submissions: list[tuple[str, int]] = []
        self.candidates = [
            {
                "card": {
                    "record_id": f"local:study_flashcard:{card_id}",
                    "backing_id": card_id,
                    "deck_record_id": f"local:study_deck:{deck_id}",
                    "front": "Q",
                    "back": "A",
                    "queue_state": "new",
                },
                "selection_reason": "relearn",
                "next_intervals": {"again": "10m", "good": "1d"},
                "review_session": {"review_session_id": 41},
                "detail_available": True,
            }
            for _ in range(deals)
        ]

    async def submit_flashcard_review(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        card_id=None,
        rating,
        current_card=None,
        answer_time_ms=None,
    ):
        await self.gate.wait()
        self.submissions.append((card_id, rating))
        outcome = self.local.submit_flashcard_review(card_id, rating=rating)
        return {
            "card": outcome["card"],
            "rating": rating,
            "next_intervals": {"good": "3d"},
            "review_session": {"review_session_id": 41},
            "detail_available": True,
        }


def _real_db_review_fixture(tmp_path, name: str):
    """A real ChaChaNotes DB holding one deck with one brand-new card."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService

    db = CharactersRAGDB(str(tmp_path / f"{name}.db"), f"study-{name}")
    deck_id = db.create_deck(f"{name} deck")
    card_id = db.create_flashcard({"deck_id": deck_id, "front": "Q", "back": "A"})
    return db, LocalStudyService(db), deck_id, card_id


def _rating_buttons_enabled(window) -> bool:
    return not any(
        window.query_one(f"#review-rating-{rating}", Button).disabled
        for rating in range(6)
    )


@pytest.mark.asyncio
async def test_cancelled_rating_leaves_the_card_retryable(tmp_path):
    """Qodo #1 on PR #1951: a cancelled rating locked the card out for good.

    `submit_rating` used to claim `_reviewed_presentation` and disable the
    rating buttons *before* awaiting the save. The `except asyncio.CancelledError:`
    branch -- added by this very branch, because `CancelledError` is a
    `BaseException` the `except Exception:` cannot see -- re-raised without
    undoing either. So a cancelled save left the panel frozen: buttons disabled,
    the presentation permanently marked reviewed, and no way to retry.

    Born red at 738bd6179: after the cancellation the rating buttons are still
    disabled, so the second press cannot even fire (`Button.press()` is a no-op
    on a disabled button) and the DB still shows `repetitions=0` -- the review
    the user made vanished with no way to make it again.
    """
    db, local, deck_id, card_id = _real_db_review_fixture(tmp_path, "cancel-retry")
    scope = RealDbReviewScopeService(local, card_id=card_id, deck_id=deck_id)
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        window, controller = await _enter_review(pilot, app)

        window.query_one("#review-rating-3", Button).press()
        await pilot.pause(0.2)
        assert scope.submissions == [], "the gate should still hold the save"

        # Cancel the in-flight rating exactly as an exclusive sibling would.
        app.workers.cancel_group(window, "study-flashcard-rating")
        await pilot.pause(0.3)

        assert _rating_buttons_enabled(window), (
            "a cancelled rating left the panel frozen: the rating buttons are "
            "still disabled while the review panel is mounted, so the user "
            "cannot retry the save that was just thrown away"
        )

        # The user rates the same card again. It must land exactly once.
        scope.gate.set()
        window.query_one("#review-rating-3", Button).press()
        await pilot.pause(0.6)

    row = db.get_flashcard(card_id)
    assert (row["repetitions"], row["interval"]) == (1, 1), (
        "the retry after a cancelled rating did not apply SM-2 exactly once: "
        f"repetitions={row['repetitions']} interval={row['interval']} "
        f"submissions={scope.submissions}"
    )
    assert scope.submissions == [(card_id, 3)], f"submissions={scope.submissions}"


@pytest.mark.asyncio
async def test_direct_submit_rating_call_cannot_double_apply_sm2(tmp_path):
    """Review property 2: the durable gate holds for callers that skip the UI.

    Disabling the rating buttons stops a second *press*, but the once-per-
    presentation check is the backstop that has to hold when something calls
    `submit_rating` directly. Here a real button press is in flight at the gate
    and a direct call queues behind it on the same presentation; SM-2 must
    still be applied once.
    """
    db, local, deck_id, card_id = _real_db_review_fixture(tmp_path, "direct-call")
    scope = RealDbReviewScopeService(local, card_id=card_id, deck_id=deck_id)
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        window, controller = await _enter_review(pilot, app)

        window.query_one("#review-rating-3", Button).press()
        await pilot.pause(0.2)
        # Bypass the (now disabled) buttons entirely.
        bypass = asyncio.ensure_future(controller.submit_rating(5))
        await pilot.pause(0.1)

        scope.gate.set()
        await pilot.pause(0.6)
        await bypass

    row = db.get_flashcard(card_id)
    assert (row["repetitions"], row["interval"]) == (1, 1), (
        "a direct submit_rating() call compounded SM-2 for one presentation: "
        f"repetitions={row['repetitions']} interval={row['interval']} "
        f"submissions={scope.submissions}"
    )
    assert scope.submissions == [(card_id, 3)], f"submissions={scope.submissions}"


@pytest.mark.asyncio
async def test_re_dealt_card_records_every_genuine_re_review(tmp_path):
    """Review property 3: the gate is per-*presentation*, never per-card.

    A relearn queue deals the same card again a few minutes later, and that
    second showing is a real recall event that must reach SM-2. Two sequential
    reviews of one re-dealt card therefore have to move it 0 -> 1 -> 2
    repetitions (interval 1d -> 6d) -- the exact state the double-press test
    forbids for a single presentation.
    """
    db, local, deck_id, card_id = _real_db_review_fixture(tmp_path, "re-deal")
    scope = RealDbReviewScopeService(local, card_id=card_id, deck_id=deck_id)
    scope.gate.set()  # saves complete immediately; the user never waits
    app = _study_app_for(scope)

    async with app.run_test(size=(180, 60)) as pilot:
        window, controller = await _enter_review(pilot, app)

        for _ in range(2):
            assert controller.current_review_card is not None
            window.query_one("#review-rating-3", Button).press()
            await pilot.pause(0.4)
            controller.show_answer()
            await pilot.pause(0.05)

    row = db.get_flashcard(card_id)
    assert (row["repetitions"], row["interval"]) == (2, 6), (
        "a re-dealt card lost one of its two genuine re-reviews: "
        f"repetitions={row['repetitions']} interval={row['interval']} "
        f"submissions={scope.submissions}"
    )
    assert scope.submissions == [(card_id, 3), (card_id, 3)], (
        f"submissions={scope.submissions}"
    )
