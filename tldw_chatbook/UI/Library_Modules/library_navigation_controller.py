"""Library route admission and first-use character-repair ownership.

The route-entry body moved from LibraryScreen for TASK-31243 fix I4. Shared
shell state and methods stay late-bound on the screen; only the media and
collections sibling calls cross named construction-time dependencies. This
preserves instance replacements and deferred admission/generation checks.
Repair's payload, service and modal imports remain first-use, not startup.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...Constants import (
    CHARACTER_NAV_CONTEXT_RETURN_FOCUS,
    LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE,
    LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION,
    LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR,
)
from ...Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_COLLECTIONS,
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SEARCH,
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_PROMPT,
    LIBRARY_ROW_CREATE_SKILL,
    LIBRARY_ROW_INGEST_EXPORT,
    LIBRARY_ROW_INGEST_MEDIA,
)
from .screen_constants import LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS

#: The list canvases a resume lands through the list-entry channel, as a
#: rail press onto them does (Collections arms it on a non-rail entry too).
_LIST_ENTRY_ROWS = frozenset(
    {
        LIBRARY_ROW_BROWSE_COLLECTIONS,
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_BROWSE_NOTES,
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_BROWSE_SKILLS,
    }
)

#: Review r11n I-2/M-4: each route's own entry control when no row or
#: remembered control takes focus -- a list's filter while its rows load,
#: an ordinary canvas's first field. The ordinary rows mirror
#: ``LibraryScreen._focus_library_ordinary_canvas_entry``'s table, which acts
#: only in the canvas-only stage; Media asks its own empty-list owner.
_ROUTE_ENTRY_SELECTORS = {
    LIBRARY_ROW_BROWSE_NOTES: "#library-notes-filter",
    LIBRARY_ROW_BROWSE_CONVERSATIONS: "#library-conversations-filter",
    LIBRARY_ROW_BROWSE_PROMPTS: "#library-prompts-filter",
    LIBRARY_ROW_BROWSE_SKILLS: "#library-skills-filter",
    LIBRARY_ROW_BROWSE_COLLECTIONS: "#library-collection-name-input",
    LIBRARY_ROW_BROWSE_SEARCH: "#library-rag-query-input",
    LIBRARY_ROW_INGEST_MEDIA: "#library-ingest-path",
    LIBRARY_ROW_INGEST_EXPORT: "#library-export-name",
    LIBRARY_ROW_CREATE_PROMPT: "#library-prompt-name",
    LIBRARY_ROW_CREATE_SKILL: "#library-skill-name",
    # The hub: where Esc from Ingest lands (``_focus_library_hub_entry``).
    "": "#library-hub-action-import",
}

if TYPE_CHECKING:
    from ..Navigation.character_conversation_navigation import (
        LibraryCharacterRepairContext,
        RoleplayReturnTarget,
    )
    from ..Screens.library_screen import LibraryScreen
    from .library_character_repair_controller import LibraryCharacterRepairController


class LibraryNavigationController:
    """Own route admission and the pending repair lifecycle for one Library."""

    def __init__(
        self,
        screen: LibraryScreen,
        *,
        invalidate_media_browse: Callable[[], None],
        unmount_collections_capture: Callable[[], None],
    ) -> None:
        self.screen = screen
        self._invalidate_media_browse = invalidate_media_browse
        self._unmount_collections_capture = unmount_collections_capture
        self.pending_repair_context: LibraryCharacterRepairContext | None = None
        self.repair_controller: LibraryCharacterRepairController | None = None
        self.repair_present_on_resume = False
        self.keyword_generation = 0
        self.semantic_generation = 0
        self.character_route = None
        self.character_candidate = None

    def apply_navigation_context(self, context: Mapping[str, Any]) -> None:
        """Admit shell routes without replacing a view before its save guards.

        Args:
            context: Ordinary mode/ID/create/ingest/source navigation or typed
                Character inspection, browse, or independently presented repair.
        """
        screen = self.screen
        if screen._prompts_state.mutation_in_flight or not isinstance(context, Mapping):
            return
        self.character_candidate = None
        screen._library_navigation_context_generation += 1
        if set(context) == {LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR}:
            from ..Navigation.character_conversation_navigation import (
                deserialize_library_character_repair_context,
            )

            payload = context.get(LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR)
            if not isinstance(payload, Mapping):
                return
            try:
                repair_context = deserialize_library_character_repair_context(payload)
            except (TypeError, ValueError):
                logger.warning("Rejected invalid Library character-repair context")
                return
            self.pending_repair_context = repair_context
            self.repair_present_on_resume = True
            if screen.is_mounted:
                screen.call_after_refresh(self.present_pending_repair)
            return
        character_admission = None
        if set(context) in (
            {LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION},
            {LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE},
        ):
            navigation = screen._unavailable_navigation
            character_admission = navigation._library_character_navigation_admission(
                screen,
                context,
                generation=screen._library_navigation_context_generation,
            )
            if character_admission is None:
                return
            target_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        else:
            target_row_id = screen._library_navigation_context_target_row(context)
            if target_row_id is None:
                return
        if target_row_id != LIBRARY_ROW_BROWSE_MEDIA:
            self._invalidate_media_browse()
        if target_row_id != LIBRARY_ROW_BROWSE_COLLECTIONS:
            self._unmount_collections_capture()
        generation = screen._library_navigation_context_generation
        if (
            screen.is_mounted
            and target_row_id == LIBRARY_ROW_BROWSE_PROMPTS
            and screen._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS
        ):
            return
        if screen.is_mounted:
            screen.run_worker(
                screen._apply_navigation_context_after_flush(
                    dict(context), target_row_id, generation, character_admission
                ),
                exclusive=True,
                group="library_nav_context",
            )
            return
        screen._apply_navigation_context_state(
            context, character_admission=character_admission
        )
        # task-32245: pre-mount there is no canvas to focus into yet, so the
        # destination's entry focus is deferred to ``on_mount`` -- the same
        # deferral the note/media/collections deep-link loads there use.
        screen._pending_library_entry_focus_row = target_row_id

    def present_pending_repair(self) -> None:
        """Present a typed context once the retained Library owns the screen."""
        screen = self.screen
        context = self.pending_repair_context
        if (
            context is None
            or not self.repair_present_on_resume
            or not screen.is_mounted
            or not screen.is_attached  # TASK-41715: the stack may be empty
            or screen.app.screen is not screen
        ):
            return
        database = getattr(screen.app_instance, "chachanotes_db", None)
        try:
            if (
                database.get_local_authority_id()
                != context.unresolved.data_authority_id
            ):
                self.pending_repair_context = None
                self.repair_present_on_resume = False
                screen._notify(
                    "The active Data Profile changed. Repair was not applied.",
                    "warning",
                )
                return
        except Exception:  # noqa: BLE001 - database boundary stays recoverable
            logger.opt(exception=True).warning(
                "Could not validate Library character-repair authority"
            )
            return
        from ...Character_Chat.character_conversation_navigation import (
            CharacterConversationNavigationService,
        )
        from .library_character_repair_controller import (
            LibraryCharacterRepairController,
            LibraryCharacterRepairDialog,
        )

        service = CharacterConversationNavigationService(database)
        controller = LibraryCharacterRepairController(
            service=service,
            invalidate_keyword=self.invalidate_keyword_candidates,
            invalidate_semantic=self.invalidate_semantic_candidates,
            return_to_anchor=self.return_from_repair,
            focus_refresh=lambda: None,
            source_revision=database.get_character_conversation_search_revision,
        )
        self.repair_controller = controller
        self.repair_present_on_resume = False
        screen.app.push_screen(LibraryCharacterRepairDialog(controller, context))

    def land_entry_focus_after_resume(self) -> None:
        """Land a Library that resumed with nothing focused inside Library.

        Review r11s (Probe B): a modal (F1, Ctrl+P) closing over a Library
        with nothing focused left focus to Textual's ``AUTO_FOCUS = "*"``,
        the nav bar's first tab, where a blind Enter leaves Library, and the
        list-entry expiry never took it back. The screen suppresses that
        auto-focus (Settings' TASK-33007.11 shape); this lands the route
        where arriving on it would, writing focus once, at that target.
        """
        screen = self.screen
        row = screen._library_selected_row_id
        files = screen._file_notes_active()
        region = "" if files else screen._library_notes_focus_region()
        if files:
            # Review r11n I-1: Folder files' own landing (its remembered
            # control, else Search, else Choose folder), never the toggle.
            screen._restore_library_notes_authority_focus("files")
        elif region == "import":
            # Review R0 M-7: the progress region, never Cancel import.
            self._land_route_entry("#note-import-body")
        elif region and region != "navigator":
            screen._restore_library_notes_focus_identity(
                screen._capture_library_notes_focus_identity()
            )
        elif row in _LIST_ENTRY_ROWS:
            screen._arm_library_list_entry_focus()
            # Review r11n I-2: rows that arrive after the arm's window end it
            # with nothing landed (it lands from None on Notes only).
            screen.set_timer(
                LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS + 0.1,
                self._land_content_focus_if_unfocused,
            )
        elif not row:
            self._land_route_entry()
        else:
            screen._arm_library_row_entry_focus(row)
        screen.call_after_refresh(self._land_content_focus_if_unfocused)

    def _land_route_entry(self, selector: str | None = None) -> bool:
        """Focus ``selector`` (default: the route's entry), marked as the app's."""
        screen = self.screen
        row = screen._library_selected_row_id
        if selector is None and row == LIBRARY_ROW_BROWSE_MEDIA:
            target = screen._library_media_empty_list_fallback_target()
        else:
            selector = selector or _ROUTE_ENTRY_SELECTORS.get(row)
            found = screen.query(selector) if selector else ()
            target = found.first() if found else None
        if target is None or target not in screen.focus_chain:
            return False
        # Marked like the list-entry channel's own landings: never user intent.
        screen._library_notes_programmatic_focus_target = target
        screen.set_focus(target, scroll_visible=False)
        return True

    def _land_content_focus_if_unfocused(self) -> None:
        """The last resort when nothing landed: the route's entry control,
        else the first content control -- which is Textual's first focusable
        (the nav bar) only when the content holds nothing focusable at all.
        """
        screen = self.screen
        if (
            not screen.is_attached
            or screen.app.screen is not screen
            or screen.focused is not None
            or screen._library_pending_list_entry_focus
        ):
            return
        if screen._file_notes_active():
            if screen._restore_library_notes_authority_focus("files"):
                return
        elif self._land_route_entry():
            return
        target = screen._first_focusable_in_content()
        if target is not None:
            screen._library_notes_programmatic_focus_target = target
            screen.set_focus(target, scroll_visible=False)

    def invalidate_keyword_candidates(self) -> None:
        self.keyword_generation += 1

    def invalidate_semantic_candidates(self) -> None:
        self.semantic_generation += 1

    def return_from_repair(self, target: RoleplayReturnTarget) -> None:
        """Clear only applied repair and emit the exact source return anchor."""
        from ..Navigation.main_navigation import NavigateToScreen

        self.pending_repair_context = None
        self.screen.post_message(
            NavigateToScreen(
                target.screen_id,
                {CHARACTER_NAV_CONTEXT_RETURN_FOCUS: target.focus_id},
            )
        )
