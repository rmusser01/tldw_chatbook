from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeState, StudyScopeType
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Study_Window import StudyWindow


class FakeStudyScopeService:
    def __init__(self):
        self.calls = []
        self.decks = [
            {"record_id": "local:study_deck:deck-local-1", "backing_id": "deck-local-1", "name": "Biology", "version": 3},
            {"record_id": "local:study_deck:deck-local-2", "backing_id": "deck-local-2", "name": "Chemistry", "version": 5},
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
            }
            ,
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
            {"card": None, "selection_reason": "none", "next_intervals": None, "review_session": None, "detail_available": False},
        ]

    async def list_decks(self, *, mode=None, scope_type=None, workspace_id=None, limit=100, offset=0):
        self.calls.append(("list_decks", mode, scope_type, workspace_id, limit, offset))
        return list(self.decks)

    async def create_deck(self, *, mode=None, scope_type=None, workspace_id=None, name, description=None, scheduler_type=None):
        self.calls.append(("create_deck", mode, scope_type, workspace_id, name, description, scheduler_type))
        created = {"record_id": f"{mode}:study_deck:new-deck", "backing_id": "new-deck", "name": name}
        self.decks.append(created)
        return created

    async def list_flashcards(self, *, mode=None, scope_type=None, workspace_id=None, deck_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_flashcards", mode, scope_type, workspace_id, deck_id, q, limit, offset))
        return [card for card in self.cards if deck_id is None or card["deck_record_id"].endswith(str(deck_id))]

    async def create_flashcard(self, *, mode=None, scope_type=None, workspace_id=None, deck_id=None, front, back, tags=None, notes=None, extra=None):
        self.calls.append(("create_flashcard", mode, scope_type, workspace_id, deck_id, front, back, tags, notes, extra))
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

    async def move_flashcard(self, *, mode=None, card_id=None, target_deck_id=None, expected_version=None):
        self.calls.append(("move_flashcard", mode, card_id, target_deck_id, expected_version))
        for card in self.cards:
            if card["backing_id"] == card_id:
                card["deck_record_id"] = f"{mode}:study_deck:{target_deck_id}"
                card["version"] = (card.get("version") or 0) + 1
                return card
        return None

    async def delete_flashcard(self, *, mode=None, card_id=None, expected_version=None, hard_delete=False):
        self.calls.append(("delete_flashcard", mode, card_id, expected_version, hard_delete))
        self.cards = [card for card in self.cards if card["backing_id"] != card_id]
        return {"deleted": True}

    async def delete_deck(self, *, mode=None, deck_id=None, expected_version=None, hard_delete=False):
        self.calls.append(("delete_deck", mode, deck_id, expected_version, hard_delete))
        self.decks = [deck for deck in self.decks if deck["backing_id"] != deck_id]
        self.cards = [card for card in self.cards if not str(card["deck_record_id"]).endswith(str(deck_id))]
        return {"deleted": True}

    async def get_next_review_candidate(self, *, mode=None, scope_type=None, workspace_id=None, deck_id=None):
        self.calls.append(("get_next_review_candidate", mode, scope_type, workspace_id, deck_id))
        if self.candidates:
            return self.candidates.pop(0)
        return {"card": None, "selection_reason": "none", "next_intervals": None, "review_session": None, "detail_available": False}

    async def submit_flashcard_review(self, *, mode=None, scope_type=None, workspace_id=None, card_id=None, rating, current_card=None, answer_time_ms=None):
        self.calls.append(("submit_flashcard_review", mode, scope_type, workspace_id, card_id, rating))
        return {
            "card": {**(current_card or {}), "interval_days": 3, "queue_state": "review"},
            "rating": rating,
            "next_intervals": {"again": "10m", "good": "3d"},
            "review_session": {"review_session_id": 41},
            "detail_available": True,
        }

    async def end_review_session(self, *, mode=None, scope_type=None, workspace_id=None, review_session_id=None):
        self.calls.append(("end_review_session", mode, scope_type, workspace_id, review_session_id))
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
            {"record_id": "server:study_deck:deck-global-1", "backing_id": "deck-global-1", "name": "Global Biology", "version": 4},
            {"record_id": "server:study_deck:deck-global-2", "backing_id": "deck-global-2", "name": "Global Chemistry", "version": 6},
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

    async def list_decks(self, *, mode=None, scope_type=None, workspace_id=None, limit=100, offset=0):
        self.calls.append(("list_decks", mode, scope_type, workspace_id, limit, offset))
        if scope_type == "workspace":
            assert workspace_id == self.workspace_id
            return list(self.workspace_decks)
        return list(self.decks)

    async def create_deck(self, *, mode=None, scope_type=None, workspace_id=None, name, description=None, scheduler_type=None):
        self.calls.append(("create_deck", mode, scope_type, workspace_id, name, description, scheduler_type))
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

    async def list_flashcards(self, *, mode=None, scope_type=None, workspace_id=None, deck_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_flashcards", mode, scope_type, workspace_id, deck_id, q, limit, offset))
        cards = self.workspace_cards if deck_id == "deck-workspace-1" else self.cards
        return [card for card in cards if deck_id is None or card["deck_record_id"].endswith(str(deck_id))]


class FlakyEndReviewStudyScopeService(FakeStudyScopeService):
    def __init__(self):
        super().__init__()
        self.fail_end_review_calls = 1

    async def end_review_session(self, *, mode=None, scope_type=None, workspace_id=None, review_session_id=None):
        self.calls.append(("end_review_session", mode, scope_type, workspace_id, review_session_id))
        if self.fail_end_review_calls > 0:
            self.fail_end_review_calls -= 1
            raise RuntimeError("failed to end review session")
        return {"id": review_session_id, "status": "completed"}


class StudyTestApp(App):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = StudyScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


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

        assert ("create_deck", "local", "global", None, "Chemistry", None, None) in scope.calls
        assert ("create_flashcard", "local", "global", None, "new-deck", "What is H2O?", "Water", ["chemistry", "water"], None, None) in scope.calls
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
        assert delete_note.display is True
        assert "server" in _text(delete_note).lower()
        assert "delete" in _text(delete_note).lower()


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
    app = StudyTestApp(app_instance)

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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        move_selected_button = app.screen.query_one("#move-selected-card-button", Button)
        delete_selected_button = app.screen.query_one("#delete-selected-card-button", Button)

        assert delete_selected_button.disabled is True
        assert move_selected_button.disabled is True


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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()
        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(SimpleNamespace(item=_list_item_for_card(card_list, "card-local-1")))
        controller.current_review_card = controller.selected_card_record
        controller.current_review_session_id = 41

        await controller.delete_selected_card()
        await pilot.pause(0.3)

        assert ("delete_flashcard", "server", "card-local-1", 7, False) in scope.calls
        assert ("end_review_session", "server", "global", None, 41) in scope.calls
        assert len(scope.cards) == 1
        assert scope.cards[0]["backing_id"] == "card-local-2"
        assert "No cards in this deck." not in _text(app.screen.query_one("#card-list", ListView).children[0].children[0])


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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()
        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(SimpleNamespace(item=_list_item_for_card(card_list, "card-local-1")))
        controller.current_review_card = controller.selected_card_record
        controller.current_review_session_id = 41

        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_target_select.value = "deck-local-2"
        await controller.move_selected_card()
        await pilot.pause(0.3)

        assert ("move_flashcard", "server", "card-local-1", "deck-local-2", 7) in scope.calls
        assert ("end_review_session", "server", "global", None, 41) in scope.calls
        assert any(card["backing_id"] == "card-local-1" and card["deck_record_id"].endswith("deck-local-2") for card in scope.cards)
        assert any(card["backing_id"] == "card-local-2" and card["deck_record_id"].endswith("deck-local-1") for card in scope.cards)
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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(SimpleNamespace(item=_list_item_for_card(card_list, "card-local-2")))

        controller.current_review_card = dict(next(card for card in scope.cards if card["backing_id"] == "card-local-1"))
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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-local-1"
        await controller.refresh_cards()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(SimpleNamespace(item=_list_item_for_card(card_list, "card-local-2")))

        controller.current_review_card = dict(next(card for card in scope.cards if card["backing_id"] == "card-local-1"))
        controller.current_review_session_id = 41
        controller._set_review_status("Next card (new).")
        controller._set_review_card(front="Question", back="Answer", show_back=False)

        move_target_select = app.screen.query_one("#move-card-target-select", Select)
        move_target_select.value = "deck-local-2"
        await controller.move_selected_card()
        await pilot.pause(0.3)

        assert ("move_flashcard", "server", "card-local-2", "deck-local-2", 11) in scope.calls
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
    app = StudyTestApp(app_instance)

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
        await controller.handle_card_selected(SimpleNamespace(item=_list_item_for_card(card_list, "card-local-1")))
        assert controller.selected_card_record["backing_id"] == "card-local-1"

        deck_select.value = "deck-local-2"
        window.handle_deck_select_changed(SimpleNamespace())

        await controller.delete_selected_card()
        await controller.move_selected_card()
        await pilot.pause(0.3)

        assert not any(call[0] == "delete_flashcard" for call in scope.calls)
        assert not any(call[0] == "move_flashcard" for call in scope.calls)
        assert any(card["backing_id"] == "card-local-1" and card["deck_record_id"].endswith("deck-local-1") for card in scope.cards)
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
    app = StudyTestApp(app_instance)

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
        assert "Question" in _text(review_front)
        assert review_back.display is False

        controller.show_answer()
        await pilot.pause(0.1)
        assert review_back.display is True

        await controller.submit_rating(4)
        await pilot.pause(0.3)

        status = app.screen.query_one("#review-status", Static)

        assert ("submit_flashcard_review", "server", "global", None, "card-server-1", 4) in scope.calls
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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        status = app.screen.query_one("#review-status", Static)
        create_button = app.screen.query_one("#create-deck-button", Button)

        assert "Create a deck" in _text(status)
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
    app = StudyTestApp(app_instance)

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

        assert ("list_decks", "server", "workspace", "workspace-1", 100, 0) in scope.calls
        assert _non_blank_option_values(deck_select._options) == ["deck-workspace-1"]
        assert _non_blank_option_values(move_target_select._options) == ["deck-workspace-1"]
        assert create_button.disabled is False

        app.screen.query_one("#new-deck-name-input", Input).value = "New Workspace Deck"
        await controller.create_deck()
        await pilot.pause(0.1)

        assert ("create_deck", "server", "workspace", "workspace-1", "New Workspace Deck", None, None) in scope.calls
        assert str(deck_select.value) == "new-workspace-deck"
        assert _non_blank_option_values(move_target_select._options) == ["deck-workspace-1"]


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
    app = StudyTestApp(app_instance)

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
        move_selected_button = app.screen.query_one("#move-selected-card-button", Button)
        delete_selected_button = app.screen.query_one("#delete-selected-card-button", Button)
        delete_deck_button = app.screen.query_one("#delete-deck-button", Button)

        assert not any(call[0] == "list_decks" and call[2] == "workspace" for call in scope.calls)
        assert "server mode" in _text(review_status).lower()
        assert _is_blank(deck_select.value)
        assert create_deck_button.disabled is True
        assert create_card_button.disabled is True
        assert start_review_button.disabled is True
        assert move_selected_button.disabled is True
        assert delete_selected_button.disabled is True
        assert delete_deck_button.disabled is True


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
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-flashcards-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).flashcards_controller
        deck_select = app.screen.query_one("#deck-select", Select)
        deck_select.value = "deck-global-1"
        await controller.refresh_cards()

        card_list = app.screen.query_one("#card-list", ListView)
        await controller.handle_card_selected(SimpleNamespace(item=_list_item_for_card(card_list, "card-global-1")))
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

        assert ("end_review_session", "server", "workspace", "workspace-1", 41) in scope.calls
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
        assert "No study decks in this workspace yet." in _text(review_status)


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
    app = StudyTestApp(app_instance)

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
        controller._set_review_card(front="Workspace question", back="Workspace answer", show_back=False)

        await app.screen.handle_runtime_backend_changed("local")
        await pilot.pause(0.3)

        review_status = app.screen.query_one("#review-status", Static)

        assert ("end_review_session", "server", "workspace", "workspace-1", 41) in scope.calls
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
    app = StudyTestApp(app_instance)

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
            call for call in scope.calls if call == ("end_review_session", "server", "global", None, 41)
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
    app = StudyTestApp(app_instance)

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
            call for call in scope.calls if call == ("end_review_session", "server", "global", None, 41)
        ]
        assert len(end_review_calls) == 2
        assert not any(call[0] == "get_next_review_candidate" for call in scope.calls)
        assert controller._pending_review_session_teardown is not None
