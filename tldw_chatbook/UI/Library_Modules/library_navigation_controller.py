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
    LIBRARY_ROW_BROWSE_PROMPTS,
)

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

    def present_pending_repair(self) -> None:
        """Present a typed context once the retained Library owns the screen."""
        screen = self.screen
        context = self.pending_repair_context
        if (
            context is None
            or not self.repair_present_on_resume
            or not screen.is_mounted
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
