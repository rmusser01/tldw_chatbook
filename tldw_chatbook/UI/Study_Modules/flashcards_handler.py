"""Screen-local flashcards controller for the Study window."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static, TextArea

from ...Study_Interop import LocalStudyService, ServerStudyService, StudyScopeService

if TYPE_CHECKING:
    from ..Study_Window import StudyWindow


FLASHCARD_SCOPE_UNAVAILABLE_TOOLTIP = (
    "Workspace Flashcards require server mode. Switch to server mode or use Global Study to edit flashcards."
)
FLASHCARD_SELECT_DECK_TOOLTIP = "Select or create a deck before adding cards or starting review."
FLASHCARD_SELECT_CARD_DELETE_TOOLTIP = "Select a flashcard before deleting it."
FLASHCARD_SELECT_CARD_MOVE_TOOLTIP = "Select a flashcard and a different target deck before moving it."
FLASHCARD_SELECT_DECK_DELETE_TOOLTIP = "Select a deck before deleting it."
FLASHCARD_DELETE_DECK_SERVER_TOOLTIP = (
    "Server mode does not support deck deletion. Delete cards individually or switch to local mode."
)


class StudyFlashcardsController:
    """Own flashcards deck/card/review interactions inside the Study screen."""

    def __init__(self, window: "StudyWindow"):
        self.window = window
        self.app_instance = window.app_instance
        self.current_review_card: Optional[dict[str, Any]] = None
        self.current_review_session_id: Optional[int] = None
        self.current_review_session_mode: Optional[str] = None
        self._pending_review_session_teardown: Optional[dict[str, Any]] = None
        self.current_decks: list[dict[str, Any]] = []
        self.current_cards: list[dict[str, Any]] = []
        self.selected_deck_record: Optional[dict[str, Any]] = None
        self.selected_card_record: Optional[dict[str, Any]] = None
        self._scope_service_cache: Optional[StudyScopeService] = None
        self.has_decks: bool = False

    def _current_mode(self) -> str:
        getter = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        if callable(getter):
            normalized = str(getter() or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
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

    def _scope_state(self) -> Any:
        scope_state = getattr(self.window, "current_scope_state", None)
        if scope_state is not None:
            return scope_state
        screen = getattr(self.window, "screen", None)
        return getattr(screen, "current_scope", None)

    @staticmethod
    def _scope_type_value(scope_state: Any) -> str:
        scope_type = getattr(scope_state, "scope_type", None)
        scope_value = getattr(scope_type, "value", scope_type)
        return str(scope_value or "global").strip().lower()

    def _scope_type(self) -> str:
        return self._scope_type_value(self._scope_state())

    def _workspace_id(self) -> Optional[str]:
        scope_state = self._scope_state()
        if self._scope_type_value(scope_state) != "workspace":
            return None
        workspace_id = getattr(scope_state, "workspace_id", None)
        return str(workspace_id or "").strip() or None

    def _scope_is_available(self) -> bool:
        scope_state = self._scope_state()
        if scope_state is None:
            return True
        if self._scope_type_value(scope_state) != "workspace":
            return True
        return bool(getattr(scope_state, "workspace_scope_available", False)) and not bool(
            str(getattr(scope_state, "error_message", "") or "").strip()
        )

    def _scope_unavailable_message(self) -> str:
        scope_state = self._scope_state()
        message = str(getattr(scope_state, "error_message", "") or "").strip()
        if message:
            return message
        if self._scope_type() == "workspace":
            backend = str(getattr(scope_state, "backend", "") or "unknown")
            return f"Workspace study is unavailable on {backend}."
        return "Study flashcards backend is unavailable."

    def _scope_empty_message(self) -> str:
        if self._scope_type() == "workspace":
            return (
                "No study decks in this workspace yet. Create a workspace deck, "
                "or switch to Global Study to review existing decks."
            )
        return "No study decks yet. Create a deck, add flashcards, then start reviewing."

    def _scope_arguments(self) -> dict[str, Any]:
        scope_type = self._scope_type()
        return {
            "scope_type": scope_type,
            "workspace_id": self._workspace_id() if scope_type == "workspace" else None,
        }

    def _workspace_create_arguments(self) -> dict[str, Any]:
        if self._current_mode() != "server":
            return {"scope_type": self._scope_type(), "workspace_id": None}
        return self._scope_arguments()

    def _policy_action_allowed(self, action_id: str) -> bool:
        checker = getattr(self.app_instance, "require_ui_action_allowed", None)
        if not callable(checker):
            return True
        decision = checker(
            action_id=action_id,
            scope_type=self._scope_type(),
        )
        return bool(getattr(decision, "allowed", False))

    def _review_session_mode(self) -> str:
        stored_mode = str(self.current_review_session_mode or "").strip().lower()
        if stored_mode in {"local", "server"}:
            return stored_mode

        current_card = self.current_review_card or {}
        backend = str(current_card.get("backend") or "").strip().lower()
        if backend in {"local", "server"}:
            return backend

        if self.current_review_session_id is not None:
            if self._current_mode() == "server":
                return "server"

            record_id = str(current_card.get("record_id") or "").strip()
            if ":" in record_id:
                prefix = record_id.split(":", 1)[0].strip().lower()
                if prefix in {"local", "server"}:
                    return prefix

            # Local review responses do not currently create durable review session ids.
            return "server"

        record_id = str(current_card.get("record_id") or "").strip()
        if ":" in record_id:
            prefix = record_id.split(":", 1)[0].strip().lower()
            if prefix in {"local", "server"}:
                return prefix
        return self._current_mode()

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

    def _review_session_teardown_request(self) -> Optional[dict[str, Any]]:
        if self._pending_review_session_teardown is not None:
            return dict(self._pending_review_session_teardown)
        if self.current_review_session_id is None:
            return None
        return {
            "mode": self._review_session_mode(),
            **self._scope_arguments(),
            "review_session_id": self.current_review_session_id,
        }

    @staticmethod
    def _is_blank_select_value(value: Any) -> bool:
        return value in {None, "", False, Select.BLANK} or str(value).startswith("Select.")

    def _selected_deck_id(self) -> Optional[str]:
        try:
            deck_select = self.window.query_one("#deck-select", Select)
        except Exception:
            return None
        value = getattr(deck_select, "value", None)
        if self._is_blank_select_value(value):
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
        if self._is_blank_select_value(value):
            return None
        selected_deck_id = self._selected_deck_id()
        for deck in self.current_decks:
            backing_id = str(deck.get("backing_id") or "")
            if backing_id == str(value) and backing_id != selected_deck_id:
                return deck
        return None

    def _reconcile_live_selection_state(self) -> None:
        live_deck = self._selected_deck_record()
        self.selected_deck_record = live_deck

        if live_deck is None:
            self.selected_card_record = None
            return

        selected_card = self.selected_card_record
        if selected_card is None:
            return

        live_deck_record_id = str(live_deck.get("record_id") or "")
        selected_card_deck_record_id = str(selected_card.get("deck_record_id") or "")
        if selected_card_deck_record_id != live_deck_record_id:
            self.selected_card_record = None

    def handle_scope_changed(self) -> None:
        """Reset controller-local state before scoped study data reloads."""
        self.current_review_card = None
        self.current_review_session_id = None
        self.current_review_session_mode = None
        self.current_decks = []
        self.current_cards = []
        self.selected_deck_record = None
        self.selected_card_record = None
        self.has_decks = False
        try:
            deck_select = self.window.query_one("#deck-select", Select)
            deck_select.set_options([("No decks available", Select.BLANK)])
            deck_select.clear()
        except Exception:
            pass
        try:
            list_view = self.window.query_one("#card-list", ListView)
            list_view.remove_children()
        except Exception:
            pass
        self._sync_move_target_options()
        try:
            self.reset_review_panel(
                self._scope_unavailable_message() if not self._scope_is_available() else "Select a deck to review cards."
            )
        except Exception:
            pass
        self._update_lifecycle_controls()

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
            target_select.clear()
            return

        target_select.set_options(options)
        valid_values = {option[1] for option in options}
        if str(getattr(target_select, "value", "")) not in {str(value) for value in valid_values}:
            target_select.clear()

    def _update_lifecycle_controls(self) -> None:
        self._reconcile_live_selection_state()
        try:
            create_deck_button = self.window.query_one("#create-deck-button", Button)
            create_card_button = self.window.query_one("#create-card-btn", Button)
            start_review_button = self.window.query_one("#start-review-btn", Button)
            move_selected_button = self.window.query_one("#move-selected-card-button", Button)
            delete_selected_button = self.window.query_one("#delete-selected-card-button", Button)
            delete_deck_button = self.window.query_one("#delete-deck-button", Button)
        except Exception:
            return

        scope_enabled = self._scope_is_available()
        selected_card = self.selected_card_record
        selected_target_deck = self._selected_target_deck_record()
        selected_deck = self.selected_deck_record

        create_deck_button.disabled = not scope_enabled
        create_card_button.disabled = not scope_enabled or selected_deck is None
        start_review_button.disabled = not scope_enabled or selected_deck is None

        if not scope_enabled:
            for button in (
                create_deck_button,
                create_card_button,
                start_review_button,
                delete_selected_button,
                move_selected_button,
                delete_deck_button,
            ):
                button.tooltip = FLASHCARD_SCOPE_UNAVAILABLE_TOOLTIP
            delete_selected_button.disabled = True
            move_selected_button.disabled = True
            delete_deck_button.disabled = True
            return

        delete_selected_button.disabled = selected_card is None
        move_selected_button.disabled = selected_card is None or selected_target_deck is None
        delete_deck_button.disabled = self._current_mode() == "server" or selected_deck is None
        create_deck_button.tooltip = None
        create_card_button.tooltip = FLASHCARD_SELECT_DECK_TOOLTIP if selected_deck is None else None
        start_review_button.tooltip = FLASHCARD_SELECT_DECK_TOOLTIP if selected_deck is None else None
        delete_selected_button.tooltip = FLASHCARD_SELECT_CARD_DELETE_TOOLTIP if selected_card is None else None
        move_selected_button.tooltip = (
            FLASHCARD_SELECT_CARD_MOVE_TOOLTIP if selected_card is None or selected_target_deck is None else None
        )
        if self._current_mode() == "server":
            delete_deck_button.tooltip = FLASHCARD_DELETE_DECK_SERVER_TOOLTIP
        else:
            delete_deck_button.tooltip = FLASHCARD_SELECT_DECK_DELETE_TOOLTIP if selected_deck is None else None

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

    def _capture_review_panel_state(self) -> dict[str, Any]:
        front_widget = self.window.query_one("#review-front", Static)
        back_widget = self.window.query_one("#review-back", Static)
        intervals_widget = self.window.query_one("#review-next-intervals", Static)
        show_answer_button = self.window.query_one("#show-answer-button", Button)
        rating_buttons = [self.window.query_one(f"#review-rating-{rating}", Button) for rating in range(6)]
        return {
            "status": str(self.window.query_one("#review-status", Static).render()),
            "front": str(front_widget.render()),
            "back": str(back_widget.render()),
            "show_back": bool(back_widget.display),
            "intervals": str(intervals_widget.render()),
            "show_answer_enabled": not show_answer_button.disabled,
            "ratings_enabled": all(not button.disabled for button in rating_buttons),
        }

    def _restore_review_panel_state(self, state: dict[str, Any]) -> None:
        self._set_review_status(str(state.get("status") or ""))
        self._set_review_card(
            front=str(state.get("front") or ""),
            back=str(state.get("back") or ""),
            show_back=bool(state.get("show_back")),
        )
        intervals = str(state.get("intervals") or "")
        self.window.query_one("#review-next-intervals", Static).update(intervals)
        self._set_review_controls(
            show_answer_enabled=bool(state.get("show_answer_enabled")),
            ratings_enabled=bool(state.get("ratings_enabled")),
        )

    def _set_review_controls(self, *, show_answer_enabled: bool, ratings_enabled: bool) -> None:
        show_answer = self.window.query_one("#show-answer-button", Button)
        show_answer.disabled = not show_answer_enabled
        for rating in range(6):
            button = self.window.query_one(f"#review-rating-{rating}", Button)
            button.disabled = not ratings_enabled

    def reset_review_panel(self, message: str) -> None:
        self.current_review_card = None
        self.current_review_session_id = None
        self.current_review_session_mode = None
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
        if not self._scope_is_available():
            self.handle_scope_changed()
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
        if not self._scope_is_available():
            self.handle_scope_changed()
            return

        selected_before = self._selected_deck_id() if preserve_selection else None
        mode = self._current_mode()
        try:
            decks = await service.list_decks(mode=mode, **self._scope_arguments())
        except ValueError:
            self.handle_scope_changed()
            return
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
            deck_select.clear()

        self.selected_deck_record = self._selected_deck_record()
        self._sync_move_target_options()

        if options == [("No decks available", Select.BLANK)]:
            self.selected_deck_record = None
            self.selected_card_record = None
            self.reset_review_panel(self._scope_empty_message())
        elif self._selected_deck_id() is None:
            self.selected_card_record = None
            self.reset_review_panel("Select a deck to review cards.")

        self._update_lifecycle_controls()

    async def refresh_cards(self, *, preserve_review_panel: bool = False) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_review_panel("Study flashcards backend is unavailable.")
            return
        if not self._scope_is_available():
            self.handle_scope_changed()
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
                self.reset_review_panel(self._scope_empty_message())
            else:
                self.reset_review_panel("Select a deck to review cards.")
            self._update_lifecycle_controls()
            return

        search_value = self.window.query_one("#flashcard-search-input", Input).value
        cards = await service.list_flashcards(
            mode=self._current_mode(),
            **self._scope_arguments(),
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
            if not preserve_review_panel:
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

        if not preserve_review_panel:
            self._set_review_status("Ready to review selected deck.")
        self._update_lifecycle_controls()

    async def create_deck(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study flashcards backend is unavailable.")
            return
        if not self._scope_is_available():
            self.handle_scope_changed()
            return

        name_input = self.window.query_one("#new-deck-name-input", Input)
        name = str(name_input.value or "").strip()
        if not name:
            self._notify("Deck name is required.")
            return

        if not self._policy_action_allowed(f"study.deck.create.{self._current_mode()}"):
            return
        try:
            created = await service.create_deck(
                mode=self._current_mode(),
                name=name,
                description=None,
                scheduler_type=None,
                **self._workspace_create_arguments(),
            )
        except Exception:
            logger.opt(exception=True).error("Failed to create study deck")
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
        if not self._scope_is_available():
            self.handle_scope_changed()
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
                **self._scope_arguments(),
                deck_id=deck_id,
                front=front,
                back=back,
                tags=tags,
                notes=None,
                extra=None,
            )
        except Exception:
            logger.opt(exception=True).error("Failed to create study flashcard")
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
        self._reconcile_live_selection_state()
        selected_card = self.selected_card_record
        if service is None or selected_card is None or not self._scope_is_available():
            return

        try:
            await service.delete_flashcard(
                mode=self._current_mode(),
                card_id=str(selected_card.get("backing_id") or ""),
                expected_version=selected_card.get("version"),
                hard_delete=False,
            )
        except Exception:
            logger.opt(exception=True).error("Failed to delete study flashcard")
            self._notify("Failed to delete flashcard.", severity="error")
            return

        review_snapshot = None
        if self.current_review_card and str(self.current_review_card.get("backing_id") or "") == str(selected_card.get("backing_id") or ""):
            await self.end_review_session_if_needed()
            self.reset_review_panel("Selected flashcard deleted.")
            preserve_review_panel = False
        else:
            preserve_review_panel = True
            review_snapshot = self._capture_review_panel_state()
        self.selected_card_record = None
        await self.refresh_cards(preserve_review_panel=preserve_review_panel)
        if review_snapshot is not None:
            self._restore_review_panel_state(review_snapshot)

    async def move_selected_card(self) -> None:
        service = self._scope_service()
        self._reconcile_live_selection_state()
        selected_card = self.selected_card_record
        target_deck = self._selected_target_deck_record()
        if service is None or selected_card is None or target_deck is None or not self._scope_is_available():
            return

        try:
            await service.move_flashcard(
                mode=self._current_mode(),
                card_id=str(selected_card.get("backing_id") or ""),
                target_deck_id=str(target_deck.get("backing_id") or ""),
                expected_version=selected_card.get("version"),
            )
        except Exception:
            logger.opt(exception=True).error("Failed to move study flashcard")
            self._notify("Failed to move flashcard.", severity="error")
            return

        review_snapshot = None
        if self.current_review_card and str(self.current_review_card.get("backing_id") or "") == str(selected_card.get("backing_id") or ""):
            await self.end_review_session_if_needed()
            self.reset_review_panel("Selected flashcard moved.")
            preserve_review_panel = False
        else:
            preserve_review_panel = True
            review_snapshot = self._capture_review_panel_state()
        self.selected_card_record = None
        await self.refresh_cards(preserve_review_panel=preserve_review_panel)
        if review_snapshot is not None:
            self._restore_review_panel_state(review_snapshot)

    async def delete_selected_deck(self) -> None:
        service = self._scope_service()
        self._reconcile_live_selection_state()
        selected_deck = self.selected_deck_record
        if service is None or selected_deck is None or not self._scope_is_available():
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
            logger.opt(exception=True).error("Failed to delete study deck")
            self._notify("Failed to delete deck.", severity="error")
            return

        if self.current_review_card:
            await self.end_review_session_if_needed()
        self.selected_deck_record = None
        self.selected_card_record = None
        await self.refresh_decks(preserve_selection=False)
        await self.refresh_cards()
        message = "Select a deck to review cards." if self.has_decks else self._scope_empty_message()
        self.reset_review_panel(message)

    async def start_review(self) -> None:
        if not self._scope_is_available():
            return
        if self._pending_review_session_teardown is not None:
            await self.end_review_session_if_needed()
            if self._pending_review_session_teardown is not None:
                self._notify("Failed to close the previous review session.", severity="error")
                return
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
        if service is None or not self.current_review_card or not self._scope_is_available():
            return

        try:
            outcome = await service.submit_flashcard_review(
                mode=self._current_mode(),
                **self._scope_arguments(),
                card_id=str(self.current_review_card.get("backing_id") or ""),
                rating=rating,
                current_card=self.current_review_card,
            )
        except Exception:
            logger.opt(exception=True).error("Failed to submit flashcard review")
            self._notify("Failed to save review.", severity="error")
            return

        review_session = outcome.get("review_session") or {}
        session_id = review_session.get("review_session_id")
        if session_id is not None:
            self.current_review_session_id = int(session_id)
            self.current_review_session_mode = self._current_mode()

        self._set_review_status("Review saved.")
        self._set_next_intervals(outcome.get("next_intervals"))
        await self._load_next_review_candidate(deck_id=self._selected_deck_id())

    async def _load_next_review_candidate(self, *, deck_id: Optional[str]) -> None:
        service = self._scope_service()
        if service is None or deck_id is None or not self._scope_is_available():
            self.reset_review_panel(self._scope_empty_message() if deck_id is None and not self.has_decks else "Select a deck to review cards.")
            return

        try:
            candidate = await service.get_next_review_candidate(
                mode=self._current_mode(),
                **self._scope_arguments(),
                deck_id=deck_id,
            )
        except Exception:
            logger.opt(exception=True).error("Failed to fetch next review candidate")
            self._notify("Failed to load review card.", severity="error")
            return

        card = candidate.get("card")
        if not card:
            await self.end_review_session_if_needed()
            self.reset_review_panel(self._scope_empty_message() if not self.has_decks else "No cards due for review.")
            return

        self.current_review_card = card
        review_session = candidate.get("review_session") or {}
        session_id = review_session.get("review_session_id")
        if session_id is not None:
            self.current_review_session_id = int(session_id)
            self.current_review_session_mode = self._current_mode()
        self._set_review_status(f"Next card ({candidate.get('selection_reason', 'unknown')}).")
        self._set_review_card(
            front=str(card.get("front") or ""),
            back=str(card.get("back") or ""),
            show_back=False,
        )
        self._set_next_intervals(candidate.get("next_intervals"))
        self._set_review_controls(show_answer_enabled=True, ratings_enabled=False)

    async def end_review_session_if_needed(self) -> None:
        teardown_request = self._review_session_teardown_request()
        if teardown_request is None:
            return
        service = self._scope_service()
        if service is None:
            self._pending_review_session_teardown = teardown_request
            return
        try:
            await service.end_review_session(**teardown_request)
        except Exception:
            self._pending_review_session_teardown = teardown_request
            logger.opt(exception=True).warning("Failed to end review session {}", teardown_request.get("review_session_id"))
            return

        self._pending_review_session_teardown = None
        if self.current_review_session_id == teardown_request.get("review_session_id"):
            self.current_review_session_id = None
            self.current_review_card = None
            self.current_review_session_mode = None
