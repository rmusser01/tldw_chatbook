from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Study_Window import StudyWindow


class FakeStudyScopeService:
    def __init__(self):
        self.calls = []
        self.decks = [{"record_id": "local:study_deck:deck-local-1", "backing_id": "deck-local-1", "name": "Biology"}]
        self.cards = [{"record_id": "local:study_flashcard:card-local-1", "backing_id": "card-local-1", "deck_record_id": "local:study_deck:deck-local-1", "front": "Question", "back": "Answer", "queue_state": "new"}]
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
            {"card": None, "selection_reason": "none", "next_intervals": None, "review_session": None, "detail_available": False},
        ]

    async def list_decks(self, *, mode=None, limit=100, offset=0):
        self.calls.append(("list_decks", mode, limit, offset))
        return list(self.decks)

    async def create_deck(self, *, mode=None, name, description=None):
        self.calls.append(("create_deck", mode, name, description))
        created = {"record_id": f"{mode}:study_deck:new-deck", "backing_id": "new-deck", "name": name}
        self.decks.append(created)
        return created

    async def list_flashcards(self, *, mode=None, deck_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_flashcards", mode, deck_id, q, limit, offset))
        return [card for card in self.cards if deck_id is None or card["deck_record_id"].endswith(str(deck_id))]

    async def create_flashcard(self, *, mode=None, deck_id=None, front, back, tags=None, notes=None, extra=None):
        self.calls.append(("create_flashcard", mode, deck_id, front, back, tags, notes, extra))
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

    async def get_next_review_candidate(self, *, mode=None, deck_id=None):
        self.calls.append(("get_next_review_candidate", mode, deck_id))
        if self.candidates:
            return self.candidates.pop(0)
        return {"card": None, "selection_reason": "none", "next_intervals": None, "review_session": None, "detail_available": False}

    async def submit_flashcard_review(self, *, mode=None, card_id=None, rating, current_card=None, answer_time_ms=None):
        self.calls.append(("submit_flashcard_review", mode, card_id, rating))
        return {
            "card": {**(current_card or {}), "interval_days": 3, "queue_state": "review"},
            "rating": rating,
            "next_intervals": {"again": "10m", "good": "3d"},
            "review_session": {"review_session_id": 41},
            "detail_available": True,
        }

    async def end_review_session(self, *, mode=None, review_session_id=None):
        self.calls.append(("end_review_session", mode, review_session_id))
        return {"id": review_session_id, "status": "completed"}


class EmptyStudyScopeService(FakeStudyScopeService):
    def __init__(self):
        super().__init__()
        self.decks = []
        self.cards = []
        self.candidates = []


class StudyTestApp(App):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = StudyScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_study_screen_passes_app_instance_to_study_window():
    app_instance = SimpleNamespace(
        study_scope_service=FakeStudyScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        window = app.screen.query_one(StudyWindow)
        assert window.app_instance is app_instance


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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        deck_select = app.screen.query_one("#deck-select", Select)
        status = app.screen.query_one("#review-status", Static)

        assert all(getattr(option, "value", None) != "default" for option in deck_select._options)
        assert ("list_decks", "local", 100, 0) in scope.calls
        assert "Select a deck" in status.renderable


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
    app = StudyTestApp(app_instance)

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

        assert ("create_deck", "local", "Chemistry", None) in scope.calls
        assert ("create_flashcard", "local", "new-deck", "What is H2O?", "Water", ["chemistry", "water"], None, None) in scope.calls
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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        deck_delete_button = app.screen.query_one("#delete-deck-button", Button)
        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_selected_button = app.screen.query_one("#move-selected-card-button", Button)
        delete_selected_button = app.screen.query_one("#delete-selected-card-button", Button)

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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        delete_deck_button = app.screen.query_one("#delete-deck-button", Button)
        delete_note = app.screen.query_one("#delete-deck-note", Static)

        assert delete_deck_button.display is True
        assert delete_deck_button.disabled is True
        assert "server" in delete_note.renderable.lower()
        assert "delete" in delete_note.renderable.lower()


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
    app = StudyTestApp(app_instance)

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
        assert "Question" in review_front.renderable
        assert review_back.display is False

        controller.show_answer()
        await pilot.pause(0.1)
        assert review_back.display is True

        await controller.submit_rating(4)
        await pilot.pause(0.3)

        status = app.screen.query_one("#review-status", Static)

        assert ("submit_flashcard_review", "server", "card-server-1", 4) in scope.calls
        assert ("end_review_session", "server", 41) in scope.calls
        assert "No cards due" in status.renderable


@pytest.mark.asyncio
async def test_flashcards_view_shows_explicit_empty_state_when_no_decks_exist():
    app_instance = SimpleNamespace(
        study_scope_service=EmptyStudyScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        status = app.screen.query_one("#review-status", Static)
        create_button = app.screen.query_one("#create-deck-button", Button)

        assert "Create a deck" in status.renderable
        assert create_button.display is True
