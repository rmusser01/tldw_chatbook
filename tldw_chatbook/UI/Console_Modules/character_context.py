"""Read-only controller for the Console Character context browser."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ...Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationKey,
    CharacterConversationNavigationService,
    CharacterConversationRow,
    CharacterKeywordIndexStatus,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)

if TYPE_CHECKING:
    from ...Chat.console_conversation_activation import (
        CharacterConversationActivationRequest,
        ConsoleConversationActivationResult,
    )
    from ..Navigation.character_conversation_navigation import (
        LibraryCharacterRepairContext,
        LibraryUnavailableConversationInspection,
        LibraryUnavailableConversationsBrowse,
        RoleplayCharacterConversationLink,
    )

CONSOLE_CHARACTER_GROUP_LIMIT = 4
CONSOLE_CHARACTER_ROW_LIMIT = 5
CONSOLE_CHARACTER_SEARCH_LIMIT = 8
CONSOLE_CHARACTER_REPAIR_CANDIDATE_LIMIT = 20
_SCOPE_CAPTURE_ATTEMPTS = 3


class ConsoleCharacterOperationPhase(StrEnum):
    """One explicit asynchronous presentation phase."""

    IDLE = "idle"
    REFRESHING = "refreshing"
    SEARCHING = "searching"
    OPENING = "opening"
    REPAIRING = "repairing"


@dataclass(frozen=True)
class ConsoleCharacterScopeFingerprint:
    """Stable authority and ambient Console identity for one projection."""

    database_identity: int | None
    data_authority_id: str
    data_revision: int
    current_character_id: int | None
    current_character_label: str = ""
    open_conversation_id: str = ""


@dataclass(frozen=True)
class _ConsoleCharacterScopeSnapshot:
    """One self-validated ambient scope plus its exact database handle."""

    database: Any
    fingerprint: ConsoleCharacterScopeFingerprint


class _ConsoleCharacterScopeChanged(RuntimeError):
    """Raised when ambient scope cannot settle within the bounded capture."""


class _ConsoleCharacterScopeReadError(RuntimeError):
    """Stable database metadata failure, paired with its ambient identities."""

    def __init__(
        self,
        database: Any,
        current: tuple[int, str] | None,
        open_conversation_id: str,
    ) -> None:
        super().__init__("Character context scope metadata is unavailable")
        self.database = database
        self.current = current
        self.open_conversation_id = open_conversation_id


@dataclass(frozen=True)
class ConsoleCharacterFocusIdentity:
    """Semantic focus target, independent of projection ordering."""

    role: str
    group_key: CharacterConversationKey | None = None
    row_key: str = ""


@dataclass(frozen=True)
class ConsoleCharacterBrowseSnapshot:
    """Stable browse presentation restored after leaving search."""

    expanded_key: CharacterConversationKey | None = None
    focus: ConsoleCharacterFocusIdentity | None = None
    scroll_offset: int = 0


@dataclass(frozen=True)
class ConsoleCharacterUnavailableDetail:
    """Bounded, same-authority recovery evidence for one unavailable row."""

    row_key: str
    reason_copy: str
    context: LibraryCharacterRepairContext | None
    candidate_count: int

    @property
    def can_repair(self) -> bool:
        return self.context is not None and self.candidate_count > 0


@dataclass(frozen=True)
class ConsoleCharacterQueryHandoffCapability:
    """Dormant Task 5 installation capability."""

    available: bool = False


@dataclass(frozen=True)
class ConsoleCharacterQueryHandoff:
    """Validated query transferred to the future Task 5 mode."""

    query: str

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query handoff requires nonblank text")


@dataclass(frozen=True)
class ConsoleCharacterContextState:
    """Complete render snapshot for the bounded Character section."""

    groups: tuple[CharacterConversationGroup, ...] = ()
    query: str = ""
    search_rows: tuple[CharacterConversationRow, ...] = ()
    expanded_key: CharacterConversationKey | None = None
    loading: bool = False
    error: str = ""
    data_revision: int = 0
    keyword_status: CharacterKeywordIndexStatus | None = None
    restore_focus: ConsoleCharacterFocusIdentity | None = None
    restore_scroll_offset: int | None = None
    scope_fingerprint: ConsoleCharacterScopeFingerprint | None = None
    phase: ConsoleCharacterOperationPhase = ConsoleCharacterOperationPhase.IDLE
    operation_row_key: str = ""
    unavailable_details: tuple[ConsoleCharacterUnavailableDetail, ...] = ()
    selected_unavailable_row_key: str = ""

    @property
    def has_context(self) -> bool:
        return bool(self.groups)

    @property
    def chat_count(self) -> int:
        return sum(group.total for group in self.groups)

    def unavailable_detail(
        self, row_key: str
    ) -> ConsoleCharacterUnavailableDetail | None:
        return next(
            (
                detail
                for detail in self.unavailable_details
                if detail.row_key == row_key
            ),
            None,
        )


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


_UNAVAILABLE_REASON_COPY = {
    UnavailableCharacterReason.MISSING_CARD: "Card missing",
    UnavailableCharacterReason.DELETED_CARD: "Card deleted",
    UnavailableCharacterReason.MISSING_CHARACTER_AUTHORITY_LINK: (
        "Character source changed"
    ),
    UnavailableCharacterReason.AMBIGUOUS_LEGACY_LINK: (
        "Historical identity incomplete"
    ),
}


def console_character_unavailable_reason_copy(
    reason: UnavailableCharacterReason | None,
) -> str:
    """Map repository reason identity to concise visible recovery copy."""

    return _UNAVAILABLE_REASON_COPY.get(reason, "Character unavailable")


class ConsoleCharacterContextController:
    """Own bounded Character reads, search state, and typed action routing."""

    def __init__(
        self,
        *,
        database_accessor: Callable[[], Any | None],
        current_character_accessor: Callable[[], tuple[int, str] | None],
        open_conversation_accessor: Callable[[], str | None],
        activate_target: Callable[
            [CharacterConversationActivationRequest, asyncio.Event],
            Awaitable[ConsoleConversationActivationResult],
        ],
        navigate_roleplay: Callable[[RoleplayCharacterConversationLink], None],
        navigate_repair: Callable[[LibraryCharacterRepairContext], None],
        navigate_inspection: Callable[[LibraryUnavailableConversationInspection], None],
        navigate_unavailable_browse: Callable[
            [LibraryUnavailableConversationsBrowse], None
        ],
        navigate_roleplay_home: Callable[[], None],
        navigate_library_home: Callable[[], None],
        start_console: Callable[
            [ResolvedLocalCharacterKey, Any, str, Callable[[], bool]], Any
        ],
        state_changed: Callable[[ConsoleCharacterContextState], None] | None = None,
        service_factory: Callable[..., CharacterConversationNavigationService] = (
            CharacterConversationNavigationService
        ),
        query_handoff_capability: ConsoleCharacterQueryHandoffCapability | None = None,
        query_handoff: Callable[[ConsoleCharacterQueryHandoff], None] | None = None,
    ) -> None:
        self._database_accessor = database_accessor
        self._current_character_accessor = current_character_accessor
        self._open_conversation_accessor = open_conversation_accessor
        self._activate_target = activate_target
        self._navigate_roleplay = navigate_roleplay
        self._navigate_repair = navigate_repair
        self._navigate_inspection = navigate_inspection
        self._navigate_unavailable_browse = navigate_unavailable_browse
        self._navigate_roleplay_home = navigate_roleplay_home
        self._navigate_library_home = navigate_library_home
        self._start_console = start_console
        self._state_changed = state_changed or (lambda _state: None)
        self._service_factory = service_factory
        self._query_handoff_capability = (
            query_handoff_capability or ConsoleCharacterQueryHandoffCapability()
        )
        self._query_handoff = query_handoff
        self._generation = 0
        self.return_reveal = False
        self._browse_snapshot: ConsoleCharacterBrowseSnapshot | None = None
        self._activation_cancellation: asyncio.Event | None = None
        self.state = ConsoleCharacterContextState()

    def _publish(self, state: ConsoleCharacterContextState) -> None:
        self.state = state
        self._state_changed(state)

    def _begin(
        self,
        phase: ConsoleCharacterOperationPhase,
        *,
        row_key: str = "",
        **changes: Any,
    ) -> int:
        self._generation += 1
        self._publish(
            replace(
                self.state,
                phase=phase,
                loading=phase
                in {
                    ConsoleCharacterOperationPhase.REFRESHING,
                    ConsoleCharacterOperationPhase.SEARCHING,
                },
                operation_row_key=row_key,
                error="",
                **changes,
            )
        )
        return self._generation

    def _current_character_identity(self) -> tuple[int, str] | None:
        current = self._current_character_accessor()
        if current is None:
            return None
        return int(current[0]), str(current[1])

    def _open_conversation_identity(self) -> str:
        conversation_id = self._open_conversation_accessor()
        return str(conversation_id) if conversation_id else ""

    def _ambient_scope_matches(
        self,
        database: Any,
        current: tuple[int, str] | None,
        open_conversation_id: str,
    ) -> bool:
        return (
            self._database_accessor() is database
            and self._current_character_identity() == current
            and self._open_conversation_identity() == open_conversation_id
        )

    @staticmethod
    def _read_database_scope_metadata(database: Any) -> tuple[str, int]:
        return (
            str(database.get_local_authority_id()),
            int(database.get_character_conversation_search_revision()),
        )

    async def _capture_scope(self) -> _ConsoleCharacterScopeSnapshot:
        """Capture DB/current identity atomically across off-thread metadata reads."""

        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            database = self._database_accessor()
            current = self._current_character_identity()
            open_conversation_id = self._open_conversation_identity()
            if database is None:
                if (
                    self._database_accessor() is database
                    and self._current_character_identity() == current
                    and self._open_conversation_identity() == open_conversation_id
                ):
                    return _ConsoleCharacterScopeSnapshot(
                        database,
                        ConsoleCharacterScopeFingerprint(
                            database_identity=None,
                            data_authority_id="",
                            data_revision=0,
                            current_character_id=(
                                None if current is None else current[0]
                            ),
                            current_character_label=(
                                "" if current is None else current[1]
                            ),
                            open_conversation_id=open_conversation_id,
                        ),
                    )
                continue
            try:
                metadata_before = await asyncio.to_thread(
                    self._read_database_scope_metadata, database
                )
            except Exception:  # noqa: BLE001 - DB adapters have no shared error base.
                if not self._ambient_scope_matches(
                    database, current, open_conversation_id
                ):
                    continue
                raise _ConsoleCharacterScopeReadError(
                    database, current, open_conversation_id
                ) from None
            if not self._ambient_scope_matches(database, current, open_conversation_id):
                continue
            try:
                metadata_after = await asyncio.to_thread(
                    self._read_database_scope_metadata, database
                )
            except Exception:  # noqa: BLE001 - DB adapters have no shared error base.
                if not self._ambient_scope_matches(
                    database, current, open_conversation_id
                ):
                    continue
                raise _ConsoleCharacterScopeReadError(
                    database, current, open_conversation_id
                ) from None
            if metadata_before != metadata_after:
                continue
            if self._ambient_scope_matches(database, current, open_conversation_id):
                authority, revision = metadata_after
                return _ConsoleCharacterScopeSnapshot(
                    database,
                    ConsoleCharacterScopeFingerprint(
                        database_identity=id(database),
                        data_authority_id=authority,
                        data_revision=revision,
                        current_character_id=(None if current is None else current[0]),
                        current_character_label=("" if current is None else current[1]),
                        open_conversation_id=open_conversation_id,
                    ),
                )
        raise _ConsoleCharacterScopeChanged("Character context scope did not settle")

    async def _fingerprint(self) -> ConsoleCharacterScopeFingerprint:
        return (await self._capture_scope()).fingerprint

    async def _scope_is_current(self, snapshot: _ConsoleCharacterScopeSnapshot) -> bool:
        try:
            current = await self._capture_scope()
        except (_ConsoleCharacterScopeChanged, _ConsoleCharacterScopeReadError):
            return False
        return (
            current.database is snapshot.database
            and current.fingerprint == snapshot.fingerprint
        )

    async def _operation_scope_is_current(self, snapshot, generation: int) -> bool:
        """Fence both sides of the final asynchronous authority validation."""
        current = await self._scope_is_current(snapshot)
        return current and generation == self._generation

    def invalidate_scope(self) -> None:
        """Fence work and force the next lifecycle check to reload."""

        self._generation += 1
        self._publish(replace(self.state, scope_fingerprint=None))

    async def refresh_if_scope_changed(self, *, force: bool = False) -> bool:
        try:
            snapshot = await self._capture_scope()
        except _ConsoleCharacterScopeChanged:
            self.invalidate_scope()
        except _ConsoleCharacterScopeReadError:
            await self.refresh()
            return True
        else:
            if not force and snapshot.fingerprint == self.state.scope_fingerprint:
                return False
        await self.refresh()
        return True

    @staticmethod
    def _unavailable_rows(
        groups: Iterable[CharacterConversationGroup],
    ) -> tuple[CharacterConversationRow, ...]:
        return tuple(
            row for group in groups for row in group.rows if row.unresolved is not None
        )

    def _load_unavailable_details_sync(
        self,
        service: CharacterConversationNavigationService,
        groups: Iterable[CharacterConversationGroup],
    ) -> tuple[ConsoleCharacterUnavailableDetail, ...]:
        details: list[ConsoleCharacterUnavailableDetail] = []
        for row in self._unavailable_rows(groups):
            key = row.unresolved
            if key is None:
                continue
            evidence = service.refresh_unresolved_evidence(key)
            context = None
            candidate_count = 0
            if evidence is not None:
                from ..Navigation.character_conversation_navigation import (
                    LibraryCharacterRepairContext,
                    RoleplayReturnTarget,
                )

                version, snapshot = evidence
                context = LibraryCharacterRepairContext(
                    unresolved=key,
                    expected_conversation_version=version,
                    historical_display_snapshot=snapshot,
                    return_target=RoleplayReturnTarget.console_context_character(),
                )
                page = service.repair_candidates(
                    key, limit=CONSOLE_CHARACTER_REPAIR_CANDIDATE_LIMIT
                )
                candidate_count = page.total
            details.append(
                ConsoleCharacterUnavailableDetail(
                    row_key=row.row_key,
                    reason_copy=console_character_unavailable_reason_copy(
                        row.unavailable_reason
                    ),
                    context=context,
                    candidate_count=candidate_count,
                )
            )
        return tuple(details)

    def _load_recent_sync(
        self,
        database: Any,
        fingerprint: ConsoleCharacterScopeFingerprint,
    ) -> tuple[
        tuple[CharacterConversationGroup, ...],
        tuple[ConsoleCharacterUnavailableDetail, ...],
    ]:
        current = (
            ResolvedLocalCharacterKey(
                fingerprint.data_authority_id,
                fingerprint.current_character_id,
            )
            if fingerprint.current_character_id is not None
            else None
        )
        service = self._service_factory(database, current_character=current)
        groups = service.recent_groups(
            group_limit=CONSOLE_CHARACTER_GROUP_LIMIT,
            row_limit=CONSOLE_CHARACTER_ROW_LIMIT,
        )
        return groups, self._load_unavailable_details_sync(service, groups)

    async def refresh(self) -> None:
        """Refresh the bounded projection under one complete scope fence."""

        generation = self._begin(ConsoleCharacterOperationPhase.REFRESHING)
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterScopeReadError as error:
                if generation != self._generation:
                    return
                if not self._ambient_scope_matches(
                    error.database, error.current, error.open_conversation_id
                ):
                    continue
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        error="Could not load local character chats · Retry",
                        scope_fingerprint=None,
                    )
                )
                return
            database = snapshot.database
            fingerprint = snapshot.fingerprint
            if database is None:
                if (
                    generation == self._generation
                    and await self._operation_scope_is_current(snapshot, generation)
                ):
                    self._publish(
                        replace(
                            self.state,
                            phase=ConsoleCharacterOperationPhase.IDLE,
                            loading=False,
                            error="Local character data is unavailable · Retry",
                            scope_fingerprint=fingerprint,
                        )
                    )
                return
            try:
                groups, details = await asyncio.to_thread(
                    self._load_recent_sync, database, fingerprint
                )
            except Exception:  # noqa: BLE001 - DB boundary becomes visible recovery
                if generation != self._generation:
                    return
                if not await self._operation_scope_is_current(snapshot, generation):
                    continue
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        error="Could not load local character chats · Retry",
                    )
                )
                return
            if generation != self._generation:
                return
            if not await self._operation_scope_is_current(snapshot, generation):
                continue
            expanded = self.state.expanded_key
            keys = {group.key for group in groups}
            if expanded not in keys:
                expanded = next((g.key for g in groups if g.is_current), None)
                expanded = expanded or (groups[0].key if groups else None)
            selected = self.state.selected_unavailable_row_key
            row_keys = {detail.row_key for detail in details}
            if selected not in row_keys:
                selected = ""
            self._publish(
                ConsoleCharacterContextState(
                    groups=groups,
                    expanded_key=expanded,
                    data_revision=fingerprint.data_revision,
                    scope_fingerprint=fingerprint,
                    unavailable_details=details,
                    selected_unavailable_row_key=selected,
                )
            )
            return
        if generation == self._generation:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    loading=False,
                    error="Local character chats changed · Retry",
                    scope_fingerprint=None,
                )
            )

    async def refresh_unavailable_details(
        self, groups: Iterable[CharacterConversationGroup]
    ) -> None:
        generation = self._generation
        bounded_groups = tuple(groups)
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                continue
            if snapshot.database is None:
                return
            try:
                service = self._service_factory(
                    snapshot.database, current_character=None
                )
                details = await asyncio.to_thread(
                    self._load_unavailable_details_sync, service, bounded_groups
                )
            except Exception:  # noqa: BLE001 - stale DB failures fail closed
                if generation != self._generation:
                    return
                if not await self._operation_scope_is_current(snapshot, generation):
                    continue
                return
            if generation != self._generation:
                return
            if not await self._operation_scope_is_current(snapshot, generation):
                continue
            self._publish(
                replace(
                    self.state,
                    unavailable_details=details,
                    scope_fingerprint=snapshot.fingerprint,
                )
            )
            return

    def toggle_group(self, key: CharacterConversationKey) -> None:
        self._publish(
            replace(
                self.state,
                expanded_key=None if self.state.expanded_key == key else key,
                restore_focus=None,
                restore_scroll_offset=None,
            )
        )

    def select_unavailable(self, row_key: str) -> None:
        self._publish(replace(self.state, selected_unavailable_row_key=row_key))

    def capture_browse(
        self,
        *,
        focus: ConsoleCharacterFocusIdentity | None,
        scroll_offset: int,
    ) -> None:
        if self._browse_snapshot is None:
            self._browse_snapshot = ConsoleCharacterBrowseSnapshot(
                expanded_key=self.state.expanded_key,
                focus=focus,
                scroll_offset=max(0, int(scroll_offset)),
            )

    def _search_sync(
        self,
        database: Any,
        fingerprint: ConsoleCharacterScopeFingerprint,
        query: str,
    ) -> tuple[tuple[CharacterConversationRow, ...], CharacterKeywordIndexStatus]:
        current = (
            ResolvedLocalCharacterKey(
                fingerprint.data_authority_id,
                fingerprint.current_character_id,
            )
            if fingerprint.current_character_id is not None
            else None
        )
        service = self._service_factory(database, current_character=current)
        status = service.ensure_keyword_index()
        page = service.keyword_search(query, limit=CONSOLE_CHARACTER_SEARCH_LIMIT)
        rows = tuple(replace(row, selected_excerpt="") for row in page.rows)
        return rows, page.keyword_status or status

    async def search(self, query: str) -> None:
        """Search at most eight local rows and restore semantic browse state."""

        normalized = str(query or "").strip()
        if not normalized:
            self._generation += 1
            snapshot = self._browse_snapshot
            self._browse_snapshot = None
            self._publish(
                replace(
                    self.state,
                    query="",
                    search_rows=(),
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    loading=False,
                    error="",
                    operation_row_key="",
                    expanded_key=(
                        snapshot.expanded_key
                        if snapshot is not None
                        else self.state.expanded_key
                    ),
                    restore_focus=(snapshot.focus if snapshot else None),
                    restore_scroll_offset=(
                        snapshot.scroll_offset if snapshot else None
                    ),
                )
            )
            return
        generation = self._begin(
            ConsoleCharacterOperationPhase.SEARCHING,
            query=normalized,
            search_rows=(),
            restore_focus=None,
            restore_scroll_offset=None,
        )
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterScopeReadError as error:
                if generation != self._generation:
                    return
                if not self._ambient_scope_matches(
                    error.database, error.current, error.open_conversation_id
                ):
                    continue
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        operation_row_key="",
                        error="Could not search local character chats · Retry",
                        scope_fingerprint=None,
                    )
                )
                return
            if snapshot.database is None:
                if (
                    generation == self._generation
                    and await self._operation_scope_is_current(snapshot, generation)
                ):
                    self._publish(
                        replace(
                            self.state,
                            phase=ConsoleCharacterOperationPhase.IDLE,
                            loading=False,
                            error="Local character search is unavailable · Retry",
                            scope_fingerprint=snapshot.fingerprint,
                        )
                    )
                return
            try:
                rows, status = await asyncio.to_thread(
                    self._search_sync,
                    snapshot.database,
                    snapshot.fingerprint,
                    normalized,
                )
            except Exception:  # noqa: BLE001 - DB boundary becomes visible recovery
                if generation != self._generation:
                    return
                if not await self._operation_scope_is_current(snapshot, generation):
                    continue
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        error="Could not search local character chats · Retry",
                    )
                )
                return
            if generation != self._generation:
                return
            if not await self._operation_scope_is_current(snapshot, generation):
                continue
            fingerprint = snapshot.fingerprint
            keyword_error = {
                CharacterKeywordIndexStatus.ABSENT: "Character source changed · Retry",
                CharacterKeywordIndexStatus.BUILDING: (
                    "Character chat search is rebuilding · Retry"
                ),
                CharacterKeywordIndexStatus.FAILED: (
                    "Character chat index needs repair · Retry"
                ),
            }.get(status, "")
            self._publish(
                replace(
                    self.state,
                    search_rows=(
                        rows[:CONSOLE_CHARACTER_SEARCH_LIMIT]
                        if not keyword_error
                        else ()
                    ),
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    loading=False,
                    error=keyword_error,
                    data_revision=fingerprint.data_revision,
                    scope_fingerprint=fingerprint,
                    keyword_status=status,
                )
            )
            return
        if generation == self._generation:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    loading=False,
                    error="Local character chats changed · Retry",
                    scope_fingerprint=None,
                )
            )

    async def activate(
        self,
        target: LocalCharacterConversationTarget,
        *,
        row_key: str = "",
    ) -> ConsoleConversationActivationResult:
        """Activate one exact target while preserving its visible row."""
        from ...Chat.console_conversation_activation import (
            CharacterConversationActivationRequest,
            ConsoleActivationResultKind,
            ConsoleConversationActivationResult,
        )

        if self._activation_cancellation is not None:
            return ConsoleConversationActivationResult(
                ConsoleActivationResultKind.FAILED, target, False
            )
        generation = self._begin(
            ConsoleCharacterOperationPhase.OPENING, row_key=row_key
        )
        cancellation = asyncio.Event()
        self._activation_cancellation = cancellation
        request = CharacterConversationActivationRequest(
            target=target,
            data_authority_id=target.character.data_authority_id,
            data_revision=self.state.data_revision,
        )
        try:
            result = await self._activate_target(request, cancellation)
        except Exception:  # noqa: BLE001 - typed UI failure at navigation boundary
            result = ConsoleConversationActivationResult(
                ConsoleActivationResultKind.FAILED, target, False
            )
        finally:
            if self._activation_cancellation is cancellation:
                self._activation_cancellation = None
        failure = {
            ConsoleActivationResultKind.NOT_FOUND: "Chat no longer exists · Refresh",
            ConsoleActivationResultKind.DATA_PROFILE_CHANGED: (
                "Data Profile changed · Refresh"
            ),
            ConsoleActivationResultKind.CHARACTER_UNAVAILABLE: (
                "Character unavailable · Open in Library"
            ),
            ConsoleActivationResultKind.FAILED: (
                "Could not open character chat · Retry"
            ),
        }.get(result.kind, "")
        if generation == self._generation:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    loading=False,
                    operation_row_key="",
                    error=failure,
                )
            )
        if (
            result.kind is ConsoleActivationResultKind.OPENED
            and generation == self._generation
        ):
            await self.refresh_if_scope_changed(force=True)
        return result

    def cancel_activation(self) -> None:
        if self._activation_cancellation is not None:
            self._activation_cancellation.set()

    def open_roleplay(self, link: RoleplayCharacterConversationLink) -> None:
        self._navigate_roleplay(link)

    def open_repair(self, context: LibraryCharacterRepairContext) -> None:
        self._navigate_repair(context)

    async def _prepare_unavailable_repair(
        self,
        key: UnresolvedConversationKey,
        *,
        row_key: str,
    ) -> bool:
        generation = self._begin(
            ConsoleCharacterOperationPhase.REPAIRING, row_key=row_key
        )
        snapshot: _ConsoleCharacterScopeSnapshot | None = None
        evidence = None
        candidates: tuple[Any, ...] = ()
        candidate_total = 0
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterScopeReadError as error:
                if generation != self._generation:
                    return False
                if not self._ambient_scope_matches(
                    error.database, error.current, error.open_conversation_id
                ):
                    continue
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        operation_row_key="",
                        error="Could not refresh Library details · Retry",
                        scope_fingerprint=None,
                    )
                )
                return False
            if snapshot.database is None:
                if (
                    generation == self._generation
                    and await self._operation_scope_is_current(snapshot, generation)
                ):
                    self._publish(
                        replace(
                            self.state,
                            phase=ConsoleCharacterOperationPhase.IDLE,
                            error="Local character data is unavailable",
                        )
                    )
                return False
            try:
                service = self._service_factory(
                    snapshot.database, current_character=None
                )
                evidence, page = await asyncio.to_thread(
                    lambda service=service: (
                        service.refresh_unresolved_evidence(key),
                        service.repair_candidates(
                            key, limit=CONSOLE_CHARACTER_REPAIR_CANDIDATE_LIMIT
                        ),
                    )
                )
                candidates = page.candidates
                candidate_total = page.total
            except Exception:  # noqa: BLE001 - repair evidence is a DB boundary
                if generation != self._generation:
                    return False
                if not await self._operation_scope_is_current(snapshot, generation):
                    continue
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        error="Could not refresh Library details · Retry",
                    )
                )
                return False
            if generation != self._generation:
                return False
            if not await self._operation_scope_is_current(snapshot, generation):
                continue
            break
        else:
            if generation == self._generation:
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        error="Data Profile changed · Refresh",
                        scope_fingerprint=None,
                    )
                )
            return False
        fingerprint = snapshot.fingerprint
        if fingerprint.data_authority_id != key.data_authority_id:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    error="Data Profile changed · Refresh",
                    scope_fingerprint=None,
                )
            )
            return False
        if evidence is None:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    error="Chat changed · Refresh",
                )
            )
            return False
        same_authority = tuple(
            candidate
            for candidate in candidates
            if candidate.key.data_authority_id == key.data_authority_id
        )
        if not same_authority:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    operation_row_key="",
                    error="No compatible local character cards",
                )
            )
            return False
        version, snapshot = evidence
        from ..Navigation.character_conversation_navigation import (
            LibraryCharacterRepairContext,
            RoleplayReturnTarget,
        )

        context = LibraryCharacterRepairContext(
            unresolved=key,
            expected_conversation_version=version,
            historical_display_snapshot=snapshot,
            return_target=RoleplayReturnTarget.console_context_character(),
        )
        updated = tuple(
            replace(
                item,
                context=context,
                candidate_count=candidate_total,
            )
            if item.row_key == row_key
            else item
            for item in self.state.unavailable_details
        )
        if not any(item.row_key == row_key for item in updated):
            updated = (
                *updated,
                ConsoleCharacterUnavailableDetail(
                    row_key=row_key,
                    reason_copy="Character unavailable",
                    context=context,
                    candidate_count=candidate_total,
                ),
            )
        self._publish(
            replace(
                self.state,
                phase=ConsoleCharacterOperationPhase.IDLE,
                operation_row_key="",
                error="",
                unavailable_details=updated,
                scope_fingerprint=fingerprint,
            )
        )
        self.open_repair(context)
        return True

    async def open_unavailable(
        self, key: UnresolvedConversationKey, *, row_key: str
    ) -> bool:
        """Open exact Library detail whether or not repair candidates exist."""

        generation = self._begin(
            ConsoleCharacterOperationPhase.REPAIRING, row_key=row_key
        )
        try:
            snapshot = await self._capture_scope()
        except Exception:  # noqa: BLE001 - scope boundary becomes visible recovery
            if generation == self._generation:
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        operation_row_key="",
                        error="Could not open Library details · Retry",
                    )
                )
            return False
        if (
            generation != self._generation
            or snapshot.fingerprint.data_authority_id != key.data_authority_id
            or not await self._operation_scope_is_current(snapshot, generation)
        ):
            if generation == self._generation:
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        operation_row_key="",
                        error="Data Profile changed · Refresh",
                        scope_fingerprint=None,
                    )
                )
            return False
        self._publish(
            replace(
                self.state,
                phase=ConsoleCharacterOperationPhase.IDLE,
                operation_row_key="",
                error="",
                scope_fingerprint=snapshot.fingerprint,
            )
        )
        from ..Navigation.character_conversation_navigation import (
            LibraryUnavailableConversationInspection,
            RoleplayReturnTarget,
        )

        self._navigate_inspection(
            LibraryUnavailableConversationInspection(
                unresolved=key,
                return_target=RoleplayReturnTarget.console_context_character(),
            )
        )
        return True

    async def repair_unavailable(
        self, key: UnresolvedConversationKey, *, row_key: str = ""
    ) -> bool:
        """Navigate to repair only with fresh same-authority candidates."""

        return await self._prepare_unavailable_repair(key, row_key=row_key)

    async def view_group(self, group: CharacterConversationGroup) -> None:
        generation = self._generation
        from ..Navigation.character_conversation_navigation import (
            LibraryUnavailableConversationsBrowse,
            RoleplayCharacterConversationLink,
            RoleplayReturnTarget,
        )

        if isinstance(group.key, ResolvedLocalCharacterKey):
            self.open_roleplay(
                RoleplayCharacterConversationLink(
                    character=group.key,
                    return_target=RoleplayReturnTarget.console_context_character(),
                )
            )
            return
        selected = self.state.selected_unavailable_row_key
        row = next(
            (
                item
                for item in group.rows
                if item.row_key == selected and item.unresolved is not None
            ),
            next((item for item in group.rows if item.unresolved is not None), None),
        )
        if row is not None and row.unresolved is not None:
            try:
                snapshot = await self._capture_scope()
            except Exception:  # noqa: BLE001 - route fails closed on scope churn
                return
            if (
                snapshot.fingerprint.data_authority_id
                != row.unresolved.data_authority_id
                or not await self._operation_scope_is_current(snapshot, generation)
            ):
                return
            self._navigate_unavailable_browse(
                LibraryUnavailableConversationsBrowse(
                    selected=row.unresolved,
                    return_target=RoleplayReturnTarget.console_context_character(),
                )
            )
        else:
            self._navigate_library_home()

    def open_roleplay_home(self) -> None:
        self._navigate_roleplay_home()

    async def start_current(self, group: CharacterConversationGroup) -> None:
        if isinstance(group.key, ResolvedLocalCharacterKey):
            generation = self._generation
            try:
                snapshot = await self._capture_scope()
            except (_ConsoleCharacterScopeChanged, _ConsoleCharacterScopeReadError):
                return
            if (
                snapshot.database is None
                or snapshot.fingerprint.data_authority_id != group.key.data_authority_id
                or not await self._operation_scope_is_current(snapshot, generation)
            ):
                return
            self.invalidate_scope()
            generation = self._generation

            def is_current() -> bool:
                return (
                    generation == self._generation
                    and self._database_accessor() is snapshot.database
                )

            await _maybe_await(
                self._start_console(
                    group.key, snapshot.database, group.character_label, is_current
                )
            )
            if is_current():
                await self.refresh_if_scope_changed(force=True)

    def handoff_query(self, query: str) -> bool:
        """Invoke Task 5's typed handoff only after capability installation."""

        if not self._query_handoff_capability.available or self._query_handoff is None:
            return False
        self._query_handoff(ConsoleCharacterQueryHandoff(query.strip()))
        return True

    @property
    def query_handoff_available(self) -> bool:
        """Return whether Task 5 installed both halves of the dormant seam."""

        return (
            self._query_handoff_capability.available and self._query_handoff is not None
        )
