"""Exact Library inspection preparation and single-use selection commit."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...Constants import LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION
from .library_unavailable_navigation import (
    _apply_navigation_context_state,
    _library_character_admission_is_current,
    _library_character_navigation_admission,
    _LibraryCharacterNavigationAdmission,
)

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


@dataclasses.dataclass
class PreparedLibraryInspection:
    """One display-neutral locator result and its request-owned source lease."""

    screen: LibraryScreen
    admission: _LibraryCharacterNavigationAdmission
    service: Any
    local_service: Any
    records: tuple[Mapping[str, Any], ...]
    page: int
    total: int
    has_more: bool
    source_is_current: Callable[[], bool]
    release: Callable[[], None] | None = None
    consumed: bool = False
    disposed: bool = False

    def is_current(self) -> bool:
        """Check the unconsumed preparation against its captured owners.

        Returns:
            True only while source, Library generation, and storage owners agree.
        """
        screen = self.screen
        return bool(
            not self.consumed
            and not self.disposed
            and self.source_is_current()
            and not screen._prompts_state.mutation_in_flight
            and self.admission.generation
            == screen._library_navigation_context_generation
            and getattr(screen.app_instance, "chachanotes_db", None)
            is self.admission.database
            and getattr(screen.app_instance, "chat_conversation_scope_service", None)
            is self.service
            and getattr(self.service, "local_service", None) is self.local_service
            and getattr(self.local_service, "db", None) is self.admission.database
        )

    def discard(self) -> None:
        """Invalidate this token and release only its own source lease once."""
        self.disposed = True
        release, self.release = self.release, None
        if release is not None:
            release()


async def _flush_library_navigation_sources(self, *, is_current) -> bool:
    """Share the retained note/prompt/skill barriers before route replacement."""
    from ...Library.library_notes_session import NoteFlushOutcomeKind

    if not is_current() or self._prompts_state.mutation_in_flight:
        return False
    note_flush = await self._flush_library_note_save()
    if not is_current() or note_flush.kind is not NoteFlushOutcomeKind.PERMITTED:
        return False
    prompt_allowed = await self._flush_library_prompt_save()
    if not is_current() or not prompt_allowed:
        return False
    skill_allowed = await self._flush_library_skill_save()
    if not is_current():
        return False
    if not skill_allowed:
        self._notify_skill_dirty_veto()
        return False
    return True


async def prepare_character_inspection(
    self: LibraryScreen, context: Mapping[str, Any], *, is_current: Callable[[], bool]
) -> PreparedLibraryInspection | None:
    """Prepare the existing bounded local locator without replacing any view.

    Args:
        self: Library screen owning the locator and retained save guards.
        context: Closed typed Character inspection navigation context.
        is_current: Source visit and cancellation validity callback across awaits.

    Returns:
        A display-neutral exact selection token, or None on rejected admission.
        The caller owns disposal of a returned token and its source lease.
    """
    from ...Library.library_conversation_reader_state import (
        LIBRARY_CONVERSATION_PAGE_SIZE,
    )

    if not isinstance(context, Mapping) or set(context) != {
        LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION
    }:
        return None
    if not is_current() or self._prompts_state.mutation_in_flight:
        return None
    self._library_navigation_context_generation += 1
    admission = _library_character_navigation_admission(
        self, context, generation=self._library_navigation_context_generation
    )
    if admission is None:
        return None
    service = getattr(self.app_instance, "chat_conversation_scope_service", None)
    prepared = PreparedLibraryInspection(
        self,
        admission,
        service,
        getattr(service, "local_service", None),
        (),
        0,
        0,
        False,
        is_current,
    )
    retained = False
    try:
        if not prepared.is_current():
            return None
        if self.is_mounted:
            allowed = await self._flush_active_file_notes()
            if not prepared.is_current() or not allowed:
                return None
            release = self._acquire_file_notes_transition("source")
            if release is False:
                return None
            prepared.release = release if callable(release) else None
            if not await _flush_library_navigation_sources(
                self, is_current=prepared.is_current
            ):
                return None
        authority = await asyncio.to_thread(admission.database.get_local_authority_id)
        if (
            not prepared.is_current()
            or authority != admission.route.unresolved.data_authority_id
        ):
            return None
        located = await self._run_library_service_call(
            service.locate_conversation_page,
            admission.route.unresolved.conversation_id,
            mode="local",
            scope_type="all",
            limit=LIBRARY_CONVERSATION_PAGE_SIZE,
        )
        if not prepared.is_current() or located is None:
            return None
        records, page, total, has_more = self._validate_library_conversation_locator(
            located, admission.route.unresolved.conversation_id
        )
        authority = await asyncio.to_thread(admission.database.get_local_authority_id)
        if (
            not prepared.is_current()
            or authority != admission.route.unresolved.data_authority_id
        ):
            return None
        prepared.records = tuple(dict(record) for record in records)
        prepared.page, prepared.total, prepared.has_more = page, total, has_more
        retained = True
        return prepared
    finally:
        if not retained:
            prepared.discard()


def commit_character_inspection(
    self: LibraryScreen, prepared: PreparedLibraryInspection
) -> bool:
    """Install one fully prepared exact selection synchronously, without I/O.

    Args:
        self: Library screen that owns the preparation.
        prepared: Unconsumed token carrying the exact bounded locator snapshot.

    Returns:
        True after installing the route and reader selection; False for a stale
        or foreign token. The caller remains responsible for lease disposal.
    """
    from ...Library.library_conversation_reader_state import ConversationReaderState

    if prepared.screen is not self or not prepared.is_current():
        return False
    admission = prepared.admission
    prepared.consumed = True
    self._navigation_controller.character_candidate = admission
    _apply_navigation_context_state(
        self, {}, recompose=False, character_admission=admission
    )
    self._pending_library_character_navigation = None
    state = self._conversations_state
    state.page_records, state.page = prepared.records, prepared.page
    state.total, state.has_more = prepared.total, prepared.has_more
    state.total_known = state.page_loaded = True
    state.query = state.requested_query = ""
    state.requested_page = prepared.page
    state.freshness, state.stale_copy, state.error = "fresh", "", ""
    state.loading = False
    self._selected_conversation_id = admission.route.unresolved.conversation_id
    state.reader_state = ConversationReaderState(
        selected_id=self._selected_conversation_id,
        generation=state.reader_state.generation + 1,
    )
    self._prepared_library_inspection_entry = admission
    return True


async def consume_prepared_character_inspection(self: LibraryScreen) -> None:
    """Render the admitted selection without a second locator admission.

    Args:
        self: Library screen whose pending admitted selection should be rendered.
    """
    admission = getattr(self, "_prepared_library_inspection_entry", None)
    if admission is None:
        return
    try:
        if not _library_character_admission_is_current(self, admission):
            return
        await self.recompose()
        if _library_character_admission_is_current(self, admission):
            self._ensure_library_conversation_reader_selection()
    finally:
        if getattr(self, "_prepared_library_inspection_entry", None) is admission:
            self._prepared_library_inspection_entry = None
