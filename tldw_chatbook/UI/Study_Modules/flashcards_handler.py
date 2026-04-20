"""Screen-local flashcards controller for the Study window."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static, TextArea

from ...Study_Interop import LocalStudyService, ServerStudyService, StudyScopeService

if TYPE_CHECKING:
    from ..Study_Window import StudyWindow


class StudyFlashcardsController:
    """Own flashcards deck/card/review interactions inside the Study screen."""

    def __init__(self, window: "StudyWindow"):
        self.window = window
        self.app_instance = window.app_instance
        self.current_review_card: Optional[dict[str, Any]] = None
        self.current_review_session_id: Optional[int] = None
        self.current_decks: list[dict[str, Any]] = []
        self.current_cards: list[dict[str, Any]] = []
        self.selected_deck_record: Optional[dict[str, Any]] = None
        self.selected_card_record: Optional[dict[str, Any]] = None
        self._scope_service_cache: Optional[StudyScopeService] = None
        self.has_decks: bool = False

    def _current_mode(self) -> str:
        candidates = (
            getattr(self.window, "runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    def _notify(self, message: str, severity: str = "warning") -> None:
        notifier = getattr(self.window, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)
            return
        notifier = getattr(self.app_instance, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)

    def _scope_service(self) -> Optional[StudyScopeService]:
        service = getattr(self.app_instance, "study_scope_service", None)
        if service is not None:
            return service
        if self._scope_service_cache is not None:
            return self._scope_service_cache

        local_service = None
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is not None:
            local_service = LocalStudyService(db=db)

        try:
            server_service = ServerStudyService.from_config(getattr(self.app_instance, "app_config", {}) or {})
        except ValueError:
            server_service = ServerStudyService(client=None)

        if local_service is None and getattr(server_service, "client", None) is None:
            return None

        self._scope_service_cache = StudyScopeService(
            local_service=local_service,
            server_service=server_service,
        )
        return self._scope_service_cache

    def _selected_deck_id(self) -> Optional[str]:
        try:
            deck_select = self.window.query_one("#deck-select", Select)
        except Exception:
            return None
        value = getattr(deck_select, "value", None)
        if value in {None, "", Select.BLANK}:
            return None
        return str(value)

    def _selected_deck_record(self) -> Optional[dict[str, Any]]:
        deck_id = self._selected_deck_id()
        if deck_id is None:
            return None
        for deck in self.current_decks:
            if str(deck.get("backing_id") or "") == deck_id:
                return deck
        return None

    def _selected_target_deck_record(self) -> Optional[dict[str, Any]]:
        try:
            target_select = self.window.query_one("#move-card-target-select", Select)
        except Exception:
            return None
        value = getattr(target_select, "value", None)
        if value in {None, "", Select.BLANK}:
            return None
        selected_deck_id = self._selected_deck_id()
        for deck in self.current_decks:
            backing_id = str(deck.get("backing_id") or "")
            if backing_id == str(value) and backing_id != selected_deck_id:
                return deck
        return None

    @staticmethod
    def _card_row_index_from_widget(widget: ListItem) -> Optional[int]:
        widget_id = str(widget.id or "")
        if not widget_id.startswith("flashcard-row-"):
            return None
        try:
            return int(widget_id.replace("flashcard-row-", "", 1))
        except ValueError:
            return None

    def _sync_move_target_options(self) -> None:
        try:
            target_select = self.window.query_one("#move-card-target-select", Select)
        except Exception:
            return

        selected_deck_id = self._selected_deck_id()
        options = [
            (str(deck.get("name") or "Unnamed deck"), str(deck.get("backing_id")))
            for deck in self.current_decks
            if deck.get("backing_id") not in {None, "", selected_deck_id}
        ]
        if not options:
            target_select.set_options([("No target decks available", Select.BLANK)])
            target_select.value = Select.BLANK
            return

        target_select.set_options(options)
        valid_values = {option[1] for option in options}
        if getattr(target_select, "value", None) not in valid_values:
            target_select.value = Select.BLANK

    def _update_lifecycle_controls(self) -> None:
        try:
            move_selected_button = self.window.query_one("#move-selected-card-button", Button)
            delete_selected_button = self.window.query_one("#delete-selected-card-button", Button)
            delete_deck_button = self.window.query_one("#delete-deck-button", Button)
        except Exception:
            return

        selected_card = self.selected_card_record
        selected_target_deck = self._selected_target_deck_record()
        selected_deck = self.selected_deck_record

        delete_selected_button.disabled = selected_card is None
        move_selected_button.disabled = selected_card is None or selected_target_deck is None
        delete_deck_button.disabled = self._current_mode() == "server" or selected_deck is None

    def _parse_tags(self, text: str) -> list[str]:
        return [item for item in str(text or "").split() if item.strip()]

    def _set_review_status(self, message: str) -> None:
        self.window.query_one("#review-status", Static).update(message)

    def _set_review_card(self, *, front: str = "", back: str = "", show_back: bool = False) -> None:
        front_widget = self.window.query_one("#review-front", Static)
        back_widget = self.window.query_one("#review-back", Static)
        front_widget.update(front)
        back_widget.update(back)
        back_widget.display = show_back

    def _set_next_intervals(self, intervals: Optional[dict[str, Any]]) -> None:
        widget = self.window.query_one("#review-next-intervals", Static)
        if not intervals:
            widget.update("")
            return
        summary = "  ".join(f"{key}: {value}" for key, value in intervals.items())
        widget.update(summary)

    def _set_review_controls(self, *, show_answer_enabled: bool, ratings_enabled: bool) -> None:
        show_answer = self.window.query_one("#show-answer-button", Button)
        show_answer.disabled = not show_answer_enabled
        for rating in range(6):
            button = self.window.query_one(f"#review-rating-{rating}", Button)
            button.disabled = not ratings_enabled

    def reset_review_panel(self, message: str) -> None:
        self.current_review_card = None
        self._set_review_status(message)
        self._set_review_card(front="", back="", show_back=False)
        self._set_next_intervals(None)
        self._set_review_controls(show_answer_enabled=False, ratings_enabled=False)

    async def _wait_for_flashcards_widgets(self) -> bool:
        for _ in range(25):
            try:
                deck_select = self.window.query_one("#deck-select", Select)
            except Exception:
                await asyncio.sleep(0.01)
                continue

            if getattr(deck_select, "is_mounted", False) and list(deck_select.children):
                return True
            await asyncio.sleep(0.01)
        return False

    async def initialize_view(self) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_review_panel("Study flashcards backend is unavailable.")
            return
        if not await self._wait_for_flashcards_widgets():
            logger.warning("Study flashcards UI did not finish mounting before initialization")
            self.reset_review_panel("Study flashcards UI is still loading.")
            return
        await self.refresh_decks()
        await self.refresh_cards()

    async def refresh_decks(
        self,
        *,
        preserve_selection: bool = True,
        preferred_selection: Optional[str] = None,
    ) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_review_panel("Study flashcards backend is unavailable.")
            return

        selected_before = self._selected_deck_id() if preserve_selection else None
        mode = self._current_mode()
        decks = await service.list_decks(mode=mode)
        self.current_decks = list(decks)

        deck_select = self.window.query_one("#deck-select", Select)
        options = [
            (str(deck.get("name") or "Unnamed deck"), str(deck.get("backing_id")))
            for deck in decks
            if deck.get("backing_id") not in {None, ""}
        ]
        self.has_decks = bool(options)
        if not options:
            options = [("No decks available", Select.BLANK)]
        deck_select.set_options(options)

        available_values = {option[1] for option in options}
        if preferred_selection in available_values:
            deck_select.value = str(preferred_selection)
        elif selected_before in available_values:
            deck_select.value = selected_before
        else:
            deck_select.value = Select.BLANK

        self.selected_deck_record = self._selected_deck_record()
        self._sync_move_target_options()

        if options == [("No decks available", Select.BLANK)]:
            self.selected_deck_record = None
            self.selected_card_record = None
            self.reset_review_panel("Create a deck to begin studying.")
        elif self._selected_deck_id() is None:
            self.selected_card_record = None
            self.reset_review_panel("Select a deck to review cards.")

        self._update_lifecycle_controls()

    async def refresh_cards(self) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_review_panel("Study flashcards backend is unavailable.")
            return

        list_view = self.window.query_one("#card-list", ListView)
        await list_view.clear()
        self.current_cards = []
        self.selected_card_record = None

        deck_id = self._selected_deck_id()
        self.selected_deck_record = self._selected_deck_record()
        self._sync_move_target_options()
        if deck_id is None:
            if not self.has_decks:
                self.reset_review_panel("Create a deck to begin studying.")
            else:
                self.reset_review_panel("Select a deck to review cards.")
            self._update_lifecycle_controls()
            return

        search_value = self.window.query_one("#flashcard-search-input", Input).value
        cards = await service.list_flashcards(
            mode=self._current_mode(),
            deck_id=deck_id,
            q=search_value,
            limit=100,
            offset=0,
        )
        self.current_cards = list(cards)

        if not cards:
            empty_item = ListItem(Label("No cards in this deck."))
            empty_item.study_card_record = None
            await list_view.append(empty_item)
            self.reset_review_panel("No cards due for review.")
            self._update_lifecycle_controls()
            return

        for index, card in enumerate(cards):
            queue_state = str(card.get("queue_state") or "unknown")
            label = f"{card.get('front', '')} [{queue_state}]"
            list_item = ListItem(Label(label))
            list_item.study_card_record = card
            list_item.study_card_index = index
            await list_view.append(list_item)

        self._set_review_status("Ready to review selected deck.")
        self._update_lifecycle_controls()

    async def create_deck(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study flashcards backend is unavailable.")
            return

        name_input = self.window.query_one("#new-deck-name-input", Input)
        name = str(name_input.value or "").strip()
        if not name:
            self._notify("Deck name is required.")
            return

        try:
            created = await service.create_deck(mode=self._current_mode(), name=name, description=None)
        except Exception:
            logger.error("Failed to create study deck", exc_info=True)
            self._notify("Failed to create deck.", severity="error")
            return

        name_input.value = ""
        created_deck_id = str(created.get("backing_id") or created.get("record_id") or Select.BLANK)
        await self.refresh_decks(
            preserve_selection=False,
            preferred_selection=created_deck_id,
        )
        await self.refresh_cards()
        self._set_review_status(f"Deck '{created.get('name', name)}' created.")

    async def create_card(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study flashcards backend is unavailable.")
            return

        deck_id = self._selected_deck_id()
        if deck_id is None:
            self._notify("Select a deck before creating a card.")
            return

        front_widget = self.window.query_one("#card-front", TextArea)
        back_widget = self.window.query_one("#card-back", TextArea)
        tags_widget = self.window.query_one("#card-tags", Input)
        front = str(front_widget.text or "").strip()
        back = str(back_widget.text or "").strip()
        if not front:
            self._notify("Card front is required.")
            return
        if not back:
            self._notify("Card back is required.")
            return

        tags = self._parse_tags(tags_widget.value)
        try:
            await service.create_flashcard(
                mode=self._current_mode(),
                deck_id=deck_id,
                front=front,
                back=back,
                tags=tags,
                notes=None,
                extra=None,
            )
        except Exception:
            logger.error("Failed to create study flashcard", exc_info=True)
            self._notify("Failed to create flashcard.", severity="error")
            return

        front_widget.text = ""
        back_widget.text = ""
        tags_widget.value = ""
        await self.refresh_cards()
        self._set_review_status("Flashcard created.")

    async def handle_deck_changed(self) -> None:
        await self.end_review_session_if_needed()
        await self.refresh_cards()

    async def handle_card_selected(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            self.selected_card_record = None
            self._update_lifecycle_controls()
            return

        selected_record = getattr(item, "study_card_record", None)
        if not isinstance(selected_record, dict):
            row_index = self._card_row_index_from_widget(item)
            if row_index is None or row_index >= len(self.current_cards):
                self.selected_card_record = None
                self._update_lifecycle_controls()
                return
            selected_record = self.current_cards[row_index]

        self.selected_card_record = selected_record
        self._update_lifecycle_controls()

    def handle_move_target_changed(self) -> None:
        self._sync_move_target_options()
        self._update_lifecycle_controls()

    async def delete_selected_card(self) -> None:
        service = self._scope_service()
        selected_card = self.selected_card_record
        if service is None or selected_card is None:
            return

        try:
            await service.delete_flashcard(
                mode=self._current_mode(),
                card_id=str(selected_card.get("backing_id") or ""),
                expected_version=selected_card.get("version"),
                hard_delete=False,
            )
        except Exception:
            logger.error("Failed to delete study flashcard", exc_info=True)
            self._notify("Failed to delete flashcard.", severity="error")
            return

        if self.current_review_card and str(self.current_review_card.get("backing_id") or "") == str(selected_card.get("backing_id") or ""):
            await self.end_review_session_if_needed()
            self.reset_review_panel("Selected flashcard deleted.")

        self.selected_card_record = None
        await self.refresh_cards()

    async def move_selected_card(self) -> None:
        service = self._scope_service()
        selected_card = self.selected_card_record
        target_deck = self._selected_target_deck_record()
        if service is None or selected_card is None or target_deck is None:
            return

        try:
            await service.move_flashcard(
                mode=self._current_mode(),
                card_id=str(selected_card.get("backing_id") or ""),
                target_deck_id=str(target_deck.get("backing_id") or ""),
                expected_version=selected_card.get("version"),
            )
        except Exception:
            logger.error("Failed to move study flashcard", exc_info=True)
            self._notify("Failed to move flashcard.", severity="error")
            return

        if self.current_review_card and str(self.current_review_card.get("backing_id") or "") == str(selected_card.get("backing_id") or ""):
            await self.end_review_session_if_needed()
            self.reset_review_panel("Selected flashcard moved.")

        self.selected_card_record = None
        await self.refresh_cards()

    async def delete_selected_deck(self) -> None:
        service = self._scope_service()
        selected_deck = self.selected_deck_record
        if service is None or selected_deck is None:
            return
        if self._current_mode() == "server":
            return

        try:
            await service.delete_deck(
                mode=self._current_mode(),
                deck_id=str(selected_deck.get("backing_id") or ""),
                expected_version=selected_deck.get("version"),
                hard_delete=False,
            )
        except Exception:
            logger.error("Failed to delete study deck", exc_info=True)
            self._notify("Failed to delete deck.", severity="error")
            return

        if self.current_review_card:
            await self.end_review_session_if_needed()
        self.selected_deck_record = None
        self.selected_card_record = None
        await self.refresh_decks(preserve_selection=False)
        await self.refresh_cards()
        self.reset_review_panel("Create a deck to begin studying.")

    async def start_review(self) -> None:
        deck_id = self._selected_deck_id()
        if deck_id is None:
            self._notify("Select a deck before starting review.")
            return
        await self._load_next_review_candidate(deck_id=deck_id)

    def show_answer(self) -> None:
        if not self.current_review_card:
            return
        self._set_review_card(
            front=str(self.current_review_card.get("front") or ""),
            back=str(self.current_review_card.get("back") or ""),
            show_back=True,
        )
        self._set_review_controls(show_answer_enabled=False, ratings_enabled=True)

    async def submit_rating(self, rating: int) -> None:
        service = self._scope_service()
        if service is None or not self.current_review_card:
            return

        try:
            outcome = await service.submit_flashcard_review(
                mode=self._current_mode(),
                card_id=str(self.current_review_card.get("backing_id") or ""),
                rating=rating,
                current_card=self.current_review_card,
            )
        except Exception:
            logger.error("Failed to submit flashcard review", exc_info=True)
            self._notify("Failed to save review.", severity="error")
            return

        review_session = outcome.get("review_session") or {}
        session_id = review_session.get("review_session_id")
        if session_id is not None:
            self.current_review_session_id = int(session_id)

        self._set_review_status("Review saved.")
        self._set_next_intervals(outcome.get("next_intervals"))
        await self._load_next_review_candidate(deck_id=self._selected_deck_id())

    async def _load_next_review_candidate(self, *, deck_id: Optional[str]) -> None:
        service = self._scope_service()
        if service is None or deck_id is None:
            self.reset_review_panel("Select a deck to review cards.")
            return

        try:
            candidate = await service.get_next_review_candidate(
                mode=self._current_mode(),
                deck_id=deck_id,
            )
        except Exception:
            logger.error("Failed to fetch next review candidate", exc_info=True)
            self._notify("Failed to load review card.", severity="error")
            return

        card = candidate.get("card")
        if not card:
            await self.end_review_session_if_needed()
            self.reset_review_panel("No cards due for review.")
            return

        self.current_review_card = card
        self._set_review_status(f"Next card ({candidate.get('selection_reason', 'unknown')}).")
        self._set_review_card(
            front=str(card.get("front") or ""),
            back=str(card.get("back") or ""),
            show_back=False,
        )
        self._set_next_intervals(candidate.get("next_intervals"))
        self._set_review_controls(show_answer_enabled=True, ratings_enabled=False)

    async def end_review_session_if_needed(self) -> None:
        if self.current_review_session_id is None:
            return
        service = self._scope_service()
        if service is None:
            self.current_review_session_id = None
            return
        try:
            await service.end_review_session(
                mode=self._current_mode(),
                review_session_id=self.current_review_session_id,
            )
        except Exception:
            logger.warning("Failed to end review session {}", self.current_review_session_id, exc_info=True)
        finally:
            self.current_review_session_id = None
            self.current_review_card = None
