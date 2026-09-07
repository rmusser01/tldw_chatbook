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


_ABSENT = object()
# Exactly the synchronous inspection commit's write set, not a screen snapshot.
_SCREEN_FIELDS = (
    "_pending_library_character_navigation",
    "_library_unavailable_browse_scope",
    "_pending_library_source_open",
    "_library_selected_row_id",
    "_selected_conversation_id",
    "_library_workspace_depth_state_cache",
    "_prepared_library_inspection_entry",
    "_library_notes_work_session_phase",
    "_library_notes_work_session_activation_pending",
)
_CONVERSATION_FIELDS = (
    "projection",
    "page_records",
    "page",
    "total",
    "has_more",
    "total_known",
    "page_loaded",
    "query",
    "requested_query",
    "requested_page",
    "freshness",
    "stale_copy",
    "error",
    "loading",
    "reader_state",
)
_NAVIGATION_FIELDS = ("character_candidate", "character_route")


@dataclasses.dataclass
class _InspectionProjection:
    screen: LibraryScreen
    conversations: Any
    navigation: Any
    screen_values: tuple[Any, ...]
    conversation_values: tuple[Any, ...]
    navigation_values: tuple[Any, ...]

    @classmethod
    def capture(cls, screen: LibraryScreen) -> _InspectionProjection:
        return cls(
            screen,
            screen._conversations_state,
            screen._navigation_controller,
            tuple(getattr(screen, name, _ABSENT) for name in _SCREEN_FIELDS),
            tuple(
                getattr(screen._conversations_state, name)
                for name in _CONVERSATION_FIELDS
            ),
            tuple(
                getattr(screen._navigation_controller, name)
                for name in _NAVIGATION_FIELDS
            ),
        )

    def _groups(self):
        return (
            (self.screen, _SCREEN_FIELDS, self.screen_values),
            (self.conversations, _CONVERSATION_FIELDS, self.conversation_values),
            (self.navigation, _NAVIGATION_FIELDS, self.navigation_values),
        )

    def matches(self) -> bool:
        if (
            self.screen._conversations_state is not self.conversations
            or self.screen._navigation_controller is not self.navigation
        ):
            return False
        for owner, names, values in self._groups():
            for name, expected in zip(names, values, strict=True):
                actual = getattr(owner, name, _ABSENT)
                if actual is expected:
                    continue
                if (
                    type(expected) in (str, int, bool, float, type(None))
                    and actual == expected
                ):
                    continue
                return False
        return True

    def restore(self) -> None:
        for owner, names, values in self._groups():
            for name, value in zip(names, values, strict=True):
                if value is _ABSENT:
                    if hasattr(owner, name):
                        delattr(owner, name)
                else:
                    setattr(owner, name, value)


@dataclasses.dataclass
class _LibraryInspectionCommitReceipt:
    before: _InspectionProjection
    after: _InspectionProjection
    request_generation: int


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
    receipt: _LibraryInspectionCommitReceipt | None = None

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

    def finish(self, *, target_owned: bool) -> None:
        """Retire a transfer, restoring only an unowned, unsuperseded projection.

        Args:
            target_owned: Whether this request ever acquired destination ownership.
                Later mounting/bookkeeping failure must not revoke that ownership.
        """
        receipt, self.receipt = self.receipt, None
        try:
            if target_owned or receipt is None:
                return
            screen = self.screen
            if (
                self.admission.generation
                == screen._library_navigation_context_generation
                and receipt.request_generation
                == screen._conversations_state.request_generation
                and getattr(screen.app_instance, "chachanotes_db", None)
                is self.admission.database
                and getattr(
                    screen.app_instance, "chat_conversation_scope_service", None
                )
                is self.service
                and getattr(self.service, "local_service", None) is self.local_service
                and getattr(self.local_service, "db", None) is self.admission.database
                and receipt.after.matches()
            ):
                receipt.before.restore()
            elif (
                getattr(screen, "_prepared_library_inspection_entry", None)
                is self.admission
            ):
                # Newer Library work wins; retire only this request's marker.
                screen._prepared_library_inspection_entry = None
        finally:
            self.discard()


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
    before = _InspectionProjection.capture(self)
    try:
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
        prepared.receipt = _LibraryInspectionCommitReceipt(
            before, _InspectionProjection.capture(self), state.request_generation
        )
    except BaseException:
        # No await or ownership transfer occurred: undo a partial synchronous write.
        before.restore()
        raise
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
