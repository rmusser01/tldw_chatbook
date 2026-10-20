"""Read-only controller for the Console Character context browser."""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from enum import StrEnum
from types import CodeType, FunctionType, MethodType
from typing import TYPE_CHECKING, Any

from ...Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationKey,
    CharacterConversationNavigationService,
    CharacterConversationPage,
    CharacterConversationRow,
    CharacterKeywordIndexStatus,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)
from ...DB.base_db import run_owned_db_call
from ...Utils.input_validation import validate_console_character_query

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
_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS = 10.0
_CHARACTER_PRESENTATION_TTL_SECONDS = 2.0
#: TASK-33620.15.1: during a run, check the scope's database at most this
#: often. Every 0.2 s tick made two or more worker DB calls, each worker opening
#: a fresh connection (a private SQLite helper process) -- the busiest worker
#: work, live, between a send's acceptance and its provider call.
SCOPE_RECHECK_DURING_RUN_SECONDS = 1.0


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


class _ConsoleCharacterAmbientError(RuntimeError):
    """Carry the exact original loop-accessor failure out of a finite pair."""

    def __init__(self, error: BaseException) -> None:
        super().__init__("Character ambient owner check failed")
        self.error = error


def _character_reader_current(record: tuple, receiver: Any = None) -> bool:
    """Check defining provenance before looking up a dynamic bound callback."""
    (
        owner,
        name,
        descriptor,
        function,
        code,
        namespace,
        defaults,
        keyword_defaults,
        keyword_items,
        closure,
        cells,
        filename,
        spec,
        origin,
    ) = record
    module = sys.modules.get(namespace.get("__name__"))
    if (
        module is None
        or vars(module) is not namespace
        or type(namespace.get("__file__")) is not type(filename)
        or namespace.get("__file__") != filename
        or namespace.get("__spec__") is not spec
        or type(getattr(spec, "origin", None)) is not type(origin)
        or getattr(spec, "origin", None) != origin
        or type(function) is not FunctionType
        or function.__code__ is not code
        or function.__globals__ is not namespace
        or function.__defaults__ is not defaults
        or function.__kwdefaults__ is not keyword_defaults
        or function.__closure__ is not closure
        or any(cell.cell_contents is not value for cell, value in cells)
    ):
        return False
    actual_items = tuple((keyword_defaults or {}).items())
    if len(actual_items) != len(keyword_items) or any(
        actual_name != expected_name or actual_value is not expected_value
        for (actual_name, actual_value), (expected_name, expected_value) in zip(
            actual_items, keyword_items
        )
    ):
        return False
    if owner is None:
        return namespace.get(name) is descriptor
    if (
        namespace.get(owner.__name__) is not owner
        or vars(owner).get(name) is not descriptor
    ):
        return False
    if receiver is None:
        return True
    if (
        type(receiver) is not owner
        or inspect.getattr_static(receiver, name) is not descriptor
    ):
        return False
    callback = getattr(receiver, name)
    if type(descriptor) is staticmethod:
        return callback is function
    return (
        type(callback) is MethodType
        and callback.__self__ is receiver
        and callback.__func__ is function
    )


def _stock_character_scope_pair(controller: Any) -> bool:
    """Keep custom direct readers on their preceding two-await route."""
    try:
        return (
            ConsoleCharacterContextController is _CHARACTER_SCOPE_OWNER
            and type(controller) is _CHARACTER_SCOPE_OWNER
            and inspect.getattr_static(_CHARACTER_SCOPE_OWNER, "__getattribute__")
            is object.__getattribute__
            and all(
                _character_reader_current(record, controller)
                for record in _CHARACTER_SCOPE_READERS
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _stock_character_presentation(controller: Any) -> bool:
    """Optional display reuse never changes a custom direct callback's route."""
    try:
        return (
            _stock_character_scope_pair(controller)
            and controller._service_factory is CharacterConversationNavigationService
            and all(
                _character_reader_current(record, controller)
                for record in _CHARACTER_REFRESH_READERS
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


class _ConsoleCharacterReaderChanged(RuntimeError):
    """A previously stock finite reader changed during its original body."""


@dataclass(frozen=True)
class _CharacterRefreshReaders:
    """Strong, short-lived source references for one display callback."""

    controller: Any
    database: Any
    current: tuple[int, str] | None
    conversation: str
    loop: asyncio.AbstractEventLoop
    generation: int
    presentation: Callable[[], bool]
    accessors: tuple
    records: tuple
    owned_call: Callable
    pair: Callable
    recent: Callable
    scope: MethodType
    batch: MethodType
    presentation_closure: tuple
    presentation_cells: tuple
    #: TASK-33620.15.6: a display check's settled fingerprint. When the
    #: batch's first scope read still matches it, the batch ends there.
    unchanged_from: ConsoleCharacterScopeFingerprint | None = None

    def require_current(self) -> None:
        controller = self.controller
        if (
            type(controller) is not ConsoleCharacterContextController
            or controller._service_factory is not CharacterConversationNavigationService
            or any(
                getattr(controller, name) is not callback
                for name, callback in self.accessors
            )
            or run_owned_db_call is not self.owned_call
            or vars(self.database).get("is_memory_db") is not False
            or type(self.presentation) is not FunctionType
            or self.presentation.__code__ is not _CHARACTER_PRESENTATION_OWNER_CODE
            or self.presentation.__globals__ is not globals()
            or self.presentation.__defaults__ is not None
            or self.presentation.__kwdefaults__ is not None
            or self.presentation.__closure__ is not self.presentation_closure
            or any(
                cell.cell_contents is not value
                for cell, value in self.presentation_cells
            )
            or any(
                type(callback) is not MethodType
                or callback.__self__ is not controller
                or callback.__func__
                is not next(
                    record[3]
                    for record in _CHARACTER_REFRESH_READERS
                    if record[1] == name
                )
                for name, callback in (
                    ("_read_refresh_scope_sync", self.scope),
                    ("_read_recent_scope_sync", self.batch),
                )
            )
            or any(
                not _character_reader_current(record, receiver)
                for record, receiver in self.records
            )
        ):
            raise _ConsoleCharacterReaderChanged("Character display reader changed")

    def ambient_current(self) -> bool:
        """Called only on the captured loop, including the paired handoffs."""
        self.require_current()
        current = (
            self.generation == self.controller._generation
            and self.presentation()
            and self.controller._ambient_scope_matches(
                self.database, self.current, self.conversation
            )
        )
        self.require_current()
        return current

    def ambient_on_loop(self) -> bool:
        """Retain the original post-pair check without GUI publication."""
        check: Future[bool] = Future()

        def validate() -> None:
            if not check.set_running_or_notify_cancel():
                return
            try:
                check.set_result(self.ambient_current())
            except _ConsoleCharacterReaderChanged as error:
                check.set_exception(error)
            except BaseException as error:  # noqa: BLE001 - transport the original loop-accessor outcome after finite retirement.
                check.set_exception(_ConsoleCharacterAmbientError(error))

        try:
            self.loop.call_soon_threadsafe(validate)
            return check.result(timeout=_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS)
        finally:
            check.cancel()


def _stock_character_refresh_readers(
    controller, presentation, generation, unchanged_from=None
):
    """Select only the original display closure and concrete file DB route.

    ``unchanged_from`` (TASK-33620.15.6) is a display check's settled
    fingerprint: the batch then ends after its first scope read when that
    still matches it.
    """
    if (
        type(controller) is not ConsoleCharacterContextController
        or type(presentation) is not FunctionType
        or presentation.__code__ is not _CHARACTER_PRESENTATION_OWNER_CODE
        or presentation.__globals__ is not globals()
        or controller._service_factory is not CharacterConversationNavigationService
        or any(
            not _character_reader_current(record, controller)
            for record in _CHARACTER_REFRESH_READERS
        )
    ):
        return None
    # The nested original closure must belong to this exact controller. A
    # borrowed same-code closure cannot authorize another controller's batch.
    closure = dict(
        zip(presentation.__code__.co_freevars, presentation.__closure__ or ())
    )
    if closure.get("self") is None or closure["self"].cell_contents is not controller:
        return None
    from ...Character_Chat import character_conversation_navigation as navigation
    from ...DB import ChaChaNotes_DB, base_db

    db_records = vars(ChaChaNotes_DB).get("_CHARACTER_REFRESH_READERS")
    service_records = vars(navigation).get("_CHARACTER_REFRESH_READERS")
    owned_records = vars(base_db).get("_CHARACTER_REFRESH_READERS")
    if (
        not db_records
        or not service_records
        or not owned_records
        or CharacterConversationNavigationService is not service_records[0][0]
        or run_owned_db_call is not owned_records[0][3]
        or any(
            not _character_reader_current(record)
            for record in (*db_records, *service_records, *owned_records)
        )
    ):
        return None
    database = controller._database_accessor()
    if (
        type(database) is not db_records[0][0]
        or vars(database).get("is_memory_db") is not False
    ):
        return None
    records = (
        *((record, controller) for record in _CHARACTER_REFRESH_READERS),
        *((record, database) for record in db_records),
        *((record, None) for record in service_records),
        *((record, None) for record in owned_records),
    )
    readers = _CharacterRefreshReaders(
        controller,
        database,
        controller._current_character_identity(),
        controller._open_conversation_identity(),
        asyncio.get_running_loop(),
        generation,
        presentation,
        tuple(
            (name, getattr(controller, name))
            for name in (
                "_database_accessor",
                "_current_character_accessor",
                "_open_conversation_accessor",
                "_state_changed",
                "_service_factory",
            )
        ),
        records,
        run_owned_db_call,
        controller._read_database_scope_metadata_pair,
        controller._load_recent_sync,
        controller._read_refresh_scope_sync,
        controller._read_recent_scope_sync,
        presentation.__closure__,
        tuple(
            (cell, cell.cell_contents)
            for name, cell in closure.items()
            if name != "cancelled"
        ),
        unchanged_from,
    )
    try:
        readers.require_current()
    except _ConsoleCharacterReaderChanged:
        return None
    return readers


@dataclass(frozen=True)
class _CharacterRecentRead:
    snapshot: _ConsoleCharacterScopeSnapshot
    groups: tuple
    details: tuple
    load_failed: bool
    scope_current: bool
    unchanged: bool = False


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
        query = validate_console_character_query(self.query)
        if not query.strip():
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

    def pending_progress_count(self, row: CharacterConversationRow) -> int:
        """Read metadata only for the currently displayed database authority."""
        scope = self.state.scope_fingerprint
        identity = row.target.character if row.target else row.unresolved
        if (
            self._progress_counts is None
            or scope is None
            or id(self._database_accessor()) != scope.database_identity
            or identity.data_authority_id != scope.data_authority_id
        ):
            return 0
        conversation_id = (
            row.target.conversation_id
            if row.target
            else row.unresolved.conversation_id
        )
        return max(0, int(self._progress_counts().get(conversation_id, 0)))

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
        progress_counts: Callable[[], dict[str, int]] | None = None,
        state_changed: Callable[[ConsoleCharacterContextState], None] | None = None,
        service_factory: Callable[..., CharacterConversationNavigationService] = (
            CharacterConversationNavigationService
        ),
        query_handoff_capability: ConsoleCharacterQueryHandoffCapability | None = None,
        query_handoff: Callable[[ConsoleCharacterQueryHandoff], None] | None = None,
        run_active: Callable[[], bool] | None = None,
    ) -> None:
        self._progress_counts = progress_counts
        self._run_active = run_active
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
        self._preparation_reads: set[Any] = set()
        self._presentation_scope_lock = asyncio.Lock()
        self._presentation_scope_key: tuple | None = None
        self._presentation_scope_at = 0.0
        self._successful_refresh_generation: int | None = None
        self.return_reveal = False
        self._browse_snapshot: ConsoleCharacterBrowseSnapshot | None = None
        self._activation_cancellation: asyncio.Event | None = None
        self._scope_checked: tuple[float, Any, tuple[int, str] | None, str] | None = None
        self._scope_invalidations = 0
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

    def _read_database_scope_metadata_pair(
        self,
        database: Any,
        current: tuple[int, str] | None,
        open_conversation_id: str,
        loop: asyncio.AbstractEventLoop,
        *,
        _reader_check: Callable[[], None] | None = None,
    ) -> tuple[tuple[str, int], tuple[str, int]] | None:
        """Keep the original pair on one handle and its ambient check on-loop."""
        if not _stock_character_scope_pair(self):
            return None
        if _reader_check is not None:
            _reader_check()
        before = self._read_database_scope_metadata(database)
        if _reader_check is not None:
            _reader_check()
        if not _stock_character_scope_pair(self):
            return None
        ambient: Future[bool] = Future()

        def validate() -> None:
            if not ambient.set_running_or_notify_cancel():
                return
            try:
                if _reader_check is not None:
                    _reader_check()
                matches = _stock_character_scope_pair(self) and self._ambient_scope_matches(
                    database, current, open_conversation_id
                )
                if _reader_check is not None:
                    _reader_check()
                ambient.set_result(matches and _stock_character_scope_pair(self))
            except _ConsoleCharacterReaderChanged as error:
                ambient.set_exception(error)
            except BaseException as error:  # noqa: BLE001 - transport the original loop-accessor outcome after finite retirement.
                ambient.set_exception(_ConsoleCharacterAmbientError(error))

        try:
            loop.call_soon_threadsafe(validate)
            if not ambient.result(timeout=_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS):
                return None
        finally:
            ambient.cancel()
        if not _stock_character_scope_pair(self):
            return None
        if _reader_check is not None:
            _reader_check()
        after = self._read_database_scope_metadata(database)
        if _reader_check is not None:
            _reader_check()
        return (before, after) if _stock_character_scope_pair(self) else None

    async def _await_database_work(self, database, operation, /, *args, **kwargs):
        """Retain an exact file callback until its existing native scopes retire."""
        from ...DB.ChaChaNotes_DB import CharactersRAGDB

        if type(database) is not CharactersRAGDB or database.is_memory_db:
            return await run_owned_db_call(database, operation, *args, **kwargs)
        from ...Backup_Recovery.participants import _core_operation
        from ...Chat.console_preparation_reads import run_preparation_read
        from ...DB.base_db import operation_owned_connection

        def invoke():
            # The same scopes as run_owned_db_call, on this exact receiver.
            # A borrower retains its handle; only a newly opened handle retires.
            with operation_owned_connection(database), _core_operation(database):
                return operation(*args, **kwargs)

        return await run_preparation_read(
            invoke,
            creator=self,
            session_id=None,
            reads=self._preparation_reads,
            require_current=lambda: None,
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
            from ...DB.ChaChaNotes_DB import CharactersRAGDB

            if (
                type(database) is CharactersRAGDB
                and not database.is_memory_db
                and _stock_character_scope_pair(self)
            ):
                ambient_error = None
                try:
                    pair = await self._await_database_work(
                        database,
                        self._read_database_scope_metadata_pair,
                        database,
                        current,
                        open_conversation_id,
                        asyncio.get_running_loop(),
                    )
                except _ConsoleCharacterAmbientError as failure:
                    ambient_error = failure.error
                except Exception:  # noqa: BLE001 - DB adapters have no shared error base.
                    if not self._ambient_scope_matches(
                        database, current, open_conversation_id
                    ):
                        continue
                    raise _ConsoleCharacterScopeReadError(
                        database, current, open_conversation_id
                    ) from None
                # Outside the metadata catch, preserve the exact original
                # accessor exception, cause and traceback through retirement.
                if ambient_error is not None:
                    raise ambient_error
                if pair is None or not _stock_character_scope_pair(self):
                    continue
                metadata_before, metadata_after = pair
            else:
                try:
                    metadata_before = await self._await_database_work(
                        database, self._read_database_scope_metadata, database
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
                    metadata_after = await self._await_database_work(
                        database, self._read_database_scope_metadata, database
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
        self._scope_invalidations += 1
        self._scope_checked = None
        self._publish(replace(self.state, scope_fingerprint=None))

    def _mark_checked(self, started: tuple, invalidations: int) -> None:
        """Start the run recheck interval unless the scope was invalidated since."""
        if self._scope_invalidations == invalidations:
            self._scope_checked = started

    def _checked_recently(self) -> bool:
        """Whether a run's sync may skip the database scope check (TASK-33620.15.1).

        During a run the database scope is checked at most once per
        ``SCOPE_RECHECK_DURING_RUN_SECONDS``, whatever the last check found:
        every message write bumps the conversation revision, so each 0.2 s
        tick of a send reloaded (or failed to settle) the browser. A skip
        needs the ambient scope (database handle, current character, open
        chat) the last check started from. Outside a run, after any ambient
        change or an invalidated scope, every sync checks.
        """
        checked = self._scope_checked
        if checked is None or self._run_active is None or not self._run_active():
            return False
        checked_at, database, current, open_conversation_id = checked
        return time.monotonic() - checked_at < (
            SCOPE_RECHECK_DURING_RUN_SECONDS
        ) and self._ambient_scope_matches(database, current, open_conversation_id)

    def _presentation_owner_key(self, screen: Any) -> tuple:
        """Name the local display owner without obtaining storage authority."""
        from tldw_chatbook import config

        from ..Screens.chat_screen import _console_screen_is_torn_down

        app = getattr(screen, "app_instance", None)
        store = getattr(screen, "_console_chat_store", None)
        session_id = getattr(store, "active_session_id", None)
        owner = (
            next((item for item in store.sessions() if item.id == session_id), None)
            if store is not None
            else None
        )
        return (
            self._generation,
            config.current_config_identity(),
            app,
            id(getattr(app, "app_config", None)),
            getattr(app, "app_config", None),
            getattr(app, "chachanotes_db", None),
            self._database_accessor(),
            self._current_character_identity(),
            self._open_conversation_identity(),
            store,
            session_id,
            id(owner),
            owner,
            getattr(owner, "workspace_id", None),
            getattr(owner, "conversation_id", None),
            getattr(owner, "conversation_binding_revision", None),
            store.session_settings_revision(session_id) if owner is not None else None,
            bool(getattr(screen, "_console_chat_tearing_down", False) or _console_screen_is_torn_down(screen)),
        )

    async def refresh_presentation_if_scope_changed(
        self, screen: Any, *, force_fresh: bool = False
    ) -> bool:
        """Reuse only a current stock display observation; live actions stay fresh."""
        from ..Screens.chat_screen import _console_screen_is_torn_down

        if getattr(screen, "_console_chat_tearing_down", False) or _console_screen_is_torn_down(screen):
            return False
        if not _stock_character_presentation(self):
            return await self.refresh_if_scope_changed()

        def owner_is_current() -> bool:
            return (
                _stock_character_presentation(self)
                and key[1:] == self._presentation_owner_key(screen)[1:]
            )

        if not force_fresh and self._run_active is not None and self._run_active():
            if not _stock_character_presentation(self):
                return await self.refresh_if_scope_changed()
            key = self._presentation_owner_key(screen)
            if not _stock_character_presentation(self):
                return await self.refresh_if_scope_changed()
            return await self.refresh_if_scope_changed(
                _presentation_is_current=owner_is_current
            )
        async with self._presentation_scope_lock:
            if getattr(screen, "_console_chat_tearing_down", False) or _console_screen_is_torn_down(screen):
                return False
            if not _stock_character_presentation(self):
                return await self.refresh_if_scope_changed()
            if not force_fresh and self._run_active is not None and self._run_active():
                if not _stock_character_presentation(self):
                    return await self.refresh_if_scope_changed()
                key = self._presentation_owner_key(screen)
                if not _stock_character_presentation(self):
                    return await self.refresh_if_scope_changed()
                return await self.refresh_if_scope_changed(
                    _presentation_is_current=owner_is_current
                )
            key = self._presentation_owner_key(screen)
            if (
                not force_fresh
                and key == self._presentation_scope_key
                and self._successful_refresh_generation == self._generation
                and not self.state.loading
                and self.state.phase is ConsoleCharacterOperationPhase.IDLE
                and not self.state.error
                and time.monotonic() - self._presentation_scope_at
                < _CHARACTER_PRESENTATION_TTL_SECONDS
            ):
                return False
            self._presentation_scope_key = None

            try:
                settled = self.state.scope_fingerprint
                expected_generation = self._generation + 1
                # TASK-33620.15.6: decide and reload on one handle (one helper);
                # None takes the original probe-then-refresh route below.
                checked = await self._check_recent_batch(
                    owner_is_current,
                    settled if type(settled) is ConsoleCharacterScopeFingerprint else None,
                )
                if checked is False:
                    if not owner_is_current() or key != self._presentation_owner_key(screen):
                        return False
                elif checked is None:
                    try:
                        snapshot = await self._capture_scope()
                    except (_ConsoleCharacterScopeChanged, _ConsoleCharacterScopeReadError):
                        if not owner_is_current() or key != self._presentation_owner_key(screen):
                            return False
                    else:
                        if not owner_is_current() or key != self._presentation_owner_key(screen):
                            return False
                        if snapshot.fingerprint == self.state.scope_fingerprint:
                            checked = False
                if checked is False:
                    if (
                        self._successful_refresh_generation == self._generation
                        and not self.state.loading
                        and self.state.phase is ConsoleCharacterOperationPhase.IDLE
                        and not self.state.error
                    ):
                        self._presentation_scope_key = key
                        self._presentation_scope_at = time.monotonic()
                    return False
                if checked is None:
                    expected_generation = self._generation + 1
                    await self.refresh(_presentation_is_current=owner_is_current)
                if (
                    owner_is_current()
                    and self._successful_refresh_generation == expected_generation
                    and self._generation == expected_generation
                    and not self.state.error
                    and self.state.scope_fingerprint is not None
                ):
                    self._presentation_scope_key = self._presentation_owner_key(screen)
                    self._presentation_scope_at = time.monotonic()
                return True
            except asyncio.CancelledError:
                self._presentation_scope_key = None
                raise

    async def refresh_if_scope_changed(
        self,
        *,
        force: bool = False,
        _presentation_is_current: Callable[[], bool] | None = None,
    ) -> bool:
        """Reload the projection when its scope changed.

        During a run the database scope is checked at most once per
        ``SCOPE_RECHECK_DURING_RUN_SECONDS`` (``_checked_recently``).

        Args:
            force: Reload even when the scope is unchanged.
            _presentation_is_current: Original display owner; direct actions stay fresh.

        Returns:
            True when a reload ran.
        """
        if not force and self._checked_recently():
            # Keep the yield the skipped worker read gave the sync tick: the
            # tick's synchronous stretch must not grow by the read's absence.
            await asyncio.sleep(0)
            return False
        started = (
            time.monotonic(),
            self._database_accessor(),
            self._current_character_identity(),
            self._open_conversation_identity(),
        )
        self._scope_checked = None
        invalidations = self._scope_invalidations
        fingerprint = self.state.scope_fingerprint
        # A changed open chat already requires the stock fully checked refresh.
        known_changed = (
            not force
            and _presentation_is_current is not None
            and type(fingerprint) is ConsoleCharacterScopeFingerprint
            and fingerprint.open_conversation_id != started[3]
            and _stock_character_refresh_readers(
                self, _presentation_is_current, self._generation
            )
            is not None
        )
        if (
            not known_changed
            and not force
            and _presentation_is_current is not None
            and _stock_character_presentation(self)
        ):
            # TASK-33620.15.6: decide and reload on one handle (one helper).
            checked = await self._check_recent_batch(
                _presentation_is_current,
                fingerprint
                if type(fingerprint) is ConsoleCharacterScopeFingerprint
                else None,
            )
            if checked is not None:
                self._mark_checked(started, invalidations)
                return checked
        if not known_changed:
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                self.invalidate_scope()
                invalidations = self._scope_invalidations
            except _ConsoleCharacterScopeReadError:
                if (
                    _presentation_is_current is not None
                    and _stock_character_presentation(self)
                ):
                    await self.refresh(_presentation_is_current=_presentation_is_current)
                else:
                    await self.refresh()
                self._mark_checked(started, invalidations)
                return True
            else:
                if not force and snapshot.fingerprint == self.state.scope_fingerprint:
                    self._mark_checked(started, invalidations)
                    return False
        if (
            _presentation_is_current is not None
            and _stock_character_presentation(self)
        ):
            await self.refresh(_presentation_is_current=_presentation_is_current)
        else:
            await self.refresh()
        self._mark_checked(started, invalidations)
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
        *,
        _reader_check: Callable[[], None] | None = None,
    ) -> tuple[ConsoleCharacterUnavailableDetail, ...]:
        details: list[ConsoleCharacterUnavailableDetail] = []
        for row in self._unavailable_rows(groups):
            key = row.unresolved
            if key is None:
                continue
            if _reader_check is not None:
                _reader_check()
            evidence = service.refresh_unresolved_evidence(key)
            if _reader_check is not None:
                _reader_check()
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
                if _reader_check is not None:
                    _reader_check()
                page = service.repair_candidates(
                    key, limit=CONSOLE_CHARACTER_REPAIR_CANDIDATE_LIMIT
                )
                if _reader_check is not None:
                    _reader_check()
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
        *,
        _reader_check: Callable[[], None] | None = None,
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
        if _reader_check is not None:
            _reader_check()
        service = self._service_factory(database, current_character=current)
        if _reader_check is not None:
            _reader_check()
        groups = service.recent_groups(
            group_limit=CONSOLE_CHARACTER_GROUP_LIMIT,
            row_limit=CONSOLE_CHARACTER_ROW_LIMIT,
        )
        if _reader_check is not None:
            _reader_check()
            details = self._load_unavailable_details_sync(
                service, groups, _reader_check=_reader_check
            )
            _reader_check()
            return groups, details
        return groups, self._load_unavailable_details_sync(service, groups)

    def _read_refresh_scope_sync(
        self,
        readers: _CharacterRefreshReaders,
        *,
        settled: ConsoleCharacterScopeFingerprint | None = None,
    ) -> _ConsoleCharacterScopeSnapshot:
        """Run all original pair reads and post-pair ambient checks fresh.

        A display check's first read passes the fingerprint the display shows
        as ``settled`` (TASK-33620.15.6). A settled pair that still matches it
        returns before the full post-pair owner check: nothing is published
        from it, and the caller confirms the ambient scope on the loop, as
        after the original probe.
        """
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            readers.require_current()
            try:
                pair = readers.pair(
                    readers.database,
                    readers.current,
                    readers.conversation,
                    readers.loop,
                    _reader_check=readers.require_current,
                )
            except (_ConsoleCharacterReaderChanged, _ConsoleCharacterAmbientError):
                raise
            except Exception:  # noqa: BLE001 - preserve the declared display error/refusal boundary.
                if not readers.ambient_on_loop():
                    raise _ConsoleCharacterScopeChanged(
                        "Character display owner changed"
                    ) from None
                raise _ConsoleCharacterScopeReadError(
                    readers.database, readers.current, readers.conversation
                ) from None
            readers.require_current()
            if pair is None:
                if not readers.ambient_on_loop():
                    raise _ConsoleCharacterScopeChanged(
                        "Character display owner changed"
                    )
                continue
            if pair[0] != pair[1]:
                continue
            authority, revision = pair[1]
            snapshot = _ConsoleCharacterScopeSnapshot(
                readers.database,
                ConsoleCharacterScopeFingerprint(
                    database_identity=id(readers.database),
                    data_authority_id=authority,
                    data_revision=revision,
                    current_character_id=(
                        None if readers.current is None else readers.current[0]
                    ),
                    current_character_label=(
                        "" if readers.current is None else readers.current[1]
                    ),
                    open_conversation_id=readers.conversation,
                ),
            )
            if settled is not None and snapshot.fingerprint == settled:
                return snapshot
            if not readers.ambient_on_loop():
                raise _ConsoleCharacterScopeChanged("Character display owner changed")
            return snapshot
        raise _ConsoleCharacterScopeChanged("Character context scope did not settle")

    def _read_recent_scope_sync(
        self, readers: _CharacterRefreshReaders
    ) -> _CharacterRecentRead:
        """Only the inner pair/groups/pair share this existing finite interval."""
        readers.require_current()
        snapshot = readers.scope.__func__(
            readers.controller, readers, settled=readers.unchanged_from
        )
        if (
            readers.unchanged_from is not None
            and snapshot.fingerprint == readers.unchanged_from
        ):
            # A settled display check ends after its first scope read.
            readers.require_current()
            return _CharacterRecentRead(snapshot, (), (), False, True, unchanged=True)
        groups, details = (), ()
        load_failed = False
        try:
            readers.require_current()
            groups, details = readers.recent(
                readers.database,
                snapshot.fingerprint,
                _reader_check=readers.require_current,
            )
        except _ConsoleCharacterReaderChanged:
            raise
        except Exception:  # noqa: BLE001 - preserve the declared display error/refusal boundary.
            load_failed = True
        readers.require_current()
        # The original coroutine checks generation/presentation after the
        # groups callback, before it starts the final metadata callback.
        if not readers.ambient_on_loop():
            return _CharacterRecentRead(snapshot, groups, details, load_failed, False)
        try:
            readers.require_current()
            final = readers.scope.__func__(readers.controller, readers)
        except (_ConsoleCharacterScopeChanged, _ConsoleCharacterScopeReadError):
            scope_current = False
        else:
            scope_current = (
                final.database is snapshot.database
                and final.fingerprint == snapshot.fingerprint
            )
        readers.require_current()
        return _CharacterRecentRead(
            snapshot, groups, details, load_failed, scope_current
        )

    async def _check_recent_batch(
        self,
        presentation: Callable[[], bool],
        settled: ConsoleCharacterScopeFingerprint | None,
    ) -> bool | None:
        """Check the stock display's scope and reload it in one finite callback.

        TASK-33620.15.6: a display check that found its scope changed made
        two finite callbacks, the scope probe and then the reload batch,
        which reads that scope again. Each callback's fresh worker handle
        starts an ADR-125 validation helper process. This runs the existing
        inner batch once instead: it ends after its first scope read when
        that read still matches ``settled``, and otherwise loads and fences
        as a refresh does. Every original check stays inside the callback
        (``_await_database_work`` admission and retirement, the on-loop
        ambient midpoint and post-pair checks, reader provenance). An
        unchanged check keeps the probe's cost: one on-loop midpoint inside
        the pair, then the ambient scope confirmed on the loop after
        retirement, as ``_capture_scope`` does. Nothing is published until
        the callback has retired; a reload then advances the generation, as
        ``_begin`` would, and publishes its result.

        Args:
            presentation: The original display-owner closure.
            settled: Fingerprint the display shows, or None when it has none.

        Returns:
            False when the scope is unchanged, or when the display owner
            moved on before the first scope read settled (as after the
            original probe, it gets nothing). True when the scope changed:
            the reload or its load error is published, except that, as in
            the original batch, an owner that moved on or a reader that
            changed during the read gets no publication, and a scope that
            moved again during the load is reloaded by the original
            ``refresh``. None when the check reached no decision for a
            current owner (no stock readers, an unchanged read whose ambient
            scope or displayed fingerprint changed by its retirement, a first
            scope read that did not settle or failed, a refresh dispatch
            customized during the read): the caller then takes the original
            probe-then-refresh route, which owns those outcomes.

        Raises:
            BaseException: An original ambient accessor failure, unchanged.
        """
        readers = _stock_character_refresh_readers(
            self, presentation, self._generation, settled
        )
        if readers is None:
            return None

        def read_current() -> _CharacterRecentRead:
            readers.require_current()
            return readers.batch.__func__(readers.controller, readers)

        ambient_error = None
        try:
            result = await self._await_database_work(readers.database, read_current)
        except _ConsoleCharacterReaderChanged:
            # As in the original batch, a changed qualified reader authorizes
            # no further read or publication. A no-argument refresh dispatch
            # customized meanwhile is honored at once, as after the probe.
            customized = any(
                inspect.getattr_static(self, name)
                is not vars(ConsoleCharacterContextController)[name]
                for name in ("refresh", "refresh_if_scope_changed")
            )
            return None if customized else True
        except _ConsoleCharacterAmbientError as failure:
            ambient_error = failure.error
        except Exception:  # noqa: BLE001 - an unsettled or unreadable first scope.
            # As after the original probe: a display owner that moved on gets
            # nothing; a current one takes the original route, which decides
            # and presents the read error.
            try:
                current = readers.ambient_current()
            except _ConsoleCharacterReaderChanged:
                current = False
            return None if current else False
        # Outside the catch, as _capture_scope does: the exact original
        # accessor exception, cause and traceback, after retirement.
        if ambient_error is not None:
            raise ambient_error
        if result.unchanged:
            # As after the original probe: the ambient scope is confirmed on
            # the loop, and a display invalidated meanwhile no longer shows
            # ``settled``; either change takes the original route.
            if not self._ambient_scope_matches(
                readers.database, readers.current, readers.conversation
            ):
                return None
            return False if self.state.scope_fingerprint == settled else None
        try:
            if not readers.ambient_current():
                # As the original batch: an owner that moved on during the
                # read gets no publication; its next check reloads.
                return True
        except _ConsoleCharacterReaderChanged:
            return True
        if not result.scope_current:
            # The first read already differed from ``settled``, so the scope
            # changed; it moved again during the load. The original refresh
            # retries it and presents a scope that never settles.
            if _stock_character_presentation(self):
                await self.refresh(_presentation_is_current=presentation)
            else:
                await self.refresh()
            return True
        # Publish as the original batch does after its ``_begin``.
        self._generation += 1
        if result.load_failed:
            self._publish(
                replace(
                    self.state,
                    phase=ConsoleCharacterOperationPhase.IDLE,
                    loading=False,
                    operation_row_key="",
                    error="Could not load local character chats · Retry",
                )
            )
            return True
        self._publish_recent_groups(
            result.snapshot.fingerprint, result.groups, result.details
        )
        self._successful_refresh_generation = self._generation
        return True

    async def _refresh_recent_batch(
        self, readers: _CharacterRefreshReaders, generation: int
    ) -> bool | None:
        def read_current() -> _CharacterRecentRead:
            # Queueing and original native admission happen before this entry.
            # Check again before invoking either captured mutable function body.
            readers.require_current()
            return readers.batch.__func__(readers.controller, readers)

        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            try:
                if not readers.ambient_current():
                    return
            except _ConsoleCharacterReaderChanged:
                return
            try:
                result = await self._await_database_work(
                    readers.database, read_current
                )
            except _ConsoleCharacterReaderChanged:
                return
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterAmbientError as error:
                raise error.error
            except Exception:  # noqa: BLE001 - preserve the declared display error/refusal boundary.
                try:
                    if not readers.ambient_current():
                        return
                except _ConsoleCharacterReaderChanged:
                    return
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
            # Keep unexpected owner/accessor errors outside the DB recovery
            # catch, as in the original post-await presentation checks.
            try:
                if not readers.ambient_current():
                    return
            except _ConsoleCharacterReaderChanged:
                return
            # The finite callback has exited; its new handle is retired.
            # No GUI state is published while that callback owns custody.
            if not result.scope_current:
                continue
            if result.load_failed:
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        error="Could not load local character chats · Retry",
                    )
                )
                return
            self._publish_recent_groups(
                result.snapshot.fingerprint, result.groups, result.details
            )
            return True
        try:
            if readers.ambient_current():
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        error="Local character chats changed · Retry",
                        scope_fingerprint=None,
                    )
                )
        except _ConsoleCharacterReaderChanged:
            return

    def _publish_recent_groups(self, fingerprint, groups, details) -> None:
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

    async def refresh(
        self, *, _presentation_is_current: Callable[[], bool] | None = None
    ) -> None:
        """Refresh the bounded projection under one complete scope fence."""

        def presentation_is_current() -> bool:
            return _presentation_is_current is None or _presentation_is_current()

        self._successful_refresh_generation = None
        if not presentation_is_current():
            return
        generation = self._begin(ConsoleCharacterOperationPhase.REFRESHING)
        readers = _stock_character_refresh_readers(self, _presentation_is_current, generation)
        if readers is not None:
            published = await self._refresh_recent_batch(readers, generation)
            if published is True:
                try:
                    readers.require_current()
                    if generation == self._generation:
                        self._successful_refresh_generation = generation
                except _ConsoleCharacterReaderChanged:
                    pass
            return
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            if not presentation_is_current():
                return
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterScopeReadError as error:
                if generation != self._generation or not presentation_is_current():
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
                    and presentation_is_current()
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
                groups, details = await self._await_database_work(
                    database, self._load_recent_sync, database, fingerprint
                )
            except Exception:  # noqa: BLE001 - DB boundary becomes visible recovery
                if generation != self._generation or not presentation_is_current():
                    return
                if not await self._operation_scope_is_current(snapshot, generation):
                    continue
                if not presentation_is_current():
                    return
                self._publish(
                    replace(
                        self.state,
                        phase=ConsoleCharacterOperationPhase.IDLE,
                        loading=False,
                        error="Could not load local character chats · Retry",
                    )
                )
                return
            if generation != self._generation or not presentation_is_current():
                return
            if not await self._operation_scope_is_current(snapshot, generation):
                continue
            if not presentation_is_current():
                return
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
            self._successful_refresh_generation = generation
            return
        if generation == self._generation and presentation_is_current():
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
                details = await self._await_database_work(
                    snapshot.database, self._load_unavailable_details_sync, service, bounded_groups
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

    def _keyword_page_sync(
        self,
        database: Any,
        fingerprint: ConsoleCharacterScopeFingerprint,
        query: str,
        offset: int,
        limit: int,
    ) -> CharacterConversationPage:
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
        page = service.keyword_search(query, offset=offset, limit=limit)
        return (
            page
            if page.keyword_status is not None
            else replace(page, keyword_status=status)
        )

    async def keyword_page(
        self, *, query: str, offset: int, limit: int
    ) -> CharacterConversationPage:
        """Load one authority-fenced Keyword page for the installed switcher.

        Args:
            query: Literal Keyword text, bounded to 200 raw characters.
            offset: Nonnegative result offset in repository relevance order.
            limit: Maximum page size, from 1 through 50.

        Returns:
            The current authority's page, or an absent page if scope is unavailable
            or cannot remain stable across the bounded capture attempts.

        Raises:
            ValueError: Query, offset, or limit violates its input boundary.
        """

        query = validate_console_character_query(query)
        for _attempt in range(_SCOPE_CAPTURE_ATTEMPTS):
            try:
                snapshot = await self._capture_scope()
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterScopeReadError:
                return CharacterConversationPage(
                    (), 0, None, 0, CharacterKeywordIndexStatus.ABSENT
                )
            if snapshot.database is None:
                return CharacterConversationPage(
                    (), 0, None, 0, CharacterKeywordIndexStatus.ABSENT
                )
            page = await self._await_database_work(
                snapshot.database, self._keyword_page_sync,
                snapshot.database,
                snapshot.fingerprint,
                query,
                offset,
                limit,
            )
            if await self._scope_is_current(snapshot):
                return page
        return CharacterConversationPage(
            (), 0, None, 0, CharacterKeywordIndexStatus.ABSENT
        )

    async def search(self, query: str) -> None:
        """Search at most eight local rows and restore semantic browse state.

        Args:
            query: Raw text validated before trimming or changing state.

        Raises:
            ValueError: If query is not text or exceeds the shared Console limit.
        """

        from ...Utils.input_validation import validate_console_switcher_query

        normalized = validate_console_switcher_query(query).strip()
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
                rows, status = await self._await_database_work(
                    snapshot.database, self._search_sync,
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
                evidence, page = await self._await_database_work(
                    snapshot.database, lambda service=service: (
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

    def open_library_home(self) -> None:
        """Open Library for generic unavailable-character recovery."""

        self._navigate_library_home()

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
        try:
            handoff = ConsoleCharacterQueryHandoff(query)
        except ValueError as exc:
            self._publish(replace(self.state, error=str(exc)))
            return False
        self._query_handoff(handoff)
        return True

    @property
    def query_handoff_available(self) -> bool:
        """Return whether Task 5 installed both halves of the dormant seam."""

        return (
            self._query_handoff_capability.available and self._query_handoff is not None
        )


# Capture only the defining callbacks consumed by the stock finite pair.
_CHARACTER_SCOPE_OWNER = ConsoleCharacterContextController
_CHARACTER_SCOPE_READERS = tuple(
    (
        _CHARACTER_SCOPE_OWNER, name, descriptor, function, function.__code__,
        function.__globals__, function.__defaults__, function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()), function.__closure__,
        tuple((cell, cell.cell_contents) for cell in (function.__closure__ or ())),
        globals().get("__file__"), globals().get("__spec__"),
        getattr(globals().get("__spec__"), "origin", None),
    )
    for name in (
        "_read_database_scope_metadata", "_ambient_scope_matches",
        "_read_database_scope_metadata_pair", "_await_database_work",
    )
    for descriptor in (vars(_CHARACTER_SCOPE_OWNER)[name],)
    for function in (descriptor.__func__ if type(descriptor) is staticmethod else descriptor,)
)


_CHARACTER_REFRESH_READERS = (*_CHARACTER_SCOPE_READERS, *tuple(
    (
        _CHARACTER_SCOPE_OWNER, name, descriptor, function, function.__code__,
        function.__globals__, function.__defaults__, function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()), function.__closure__,
        tuple((cell, cell.cell_contents) for cell in (function.__closure__ or ())),
        globals().get("__file__"), globals().get("__spec__"),
        getattr(globals().get("__spec__"), "origin", None),
    )
    for name in (
        "refresh", "_presentation_owner_key", "refresh_presentation_if_scope_changed",
        "refresh_if_scope_changed", "_load_recent_sync", "_load_unavailable_details_sync",
        "_begin", "_publish", "_current_character_identity", "_open_conversation_identity",
        "_capture_scope", "_scope_is_current", "_operation_scope_is_current", "_unavailable_rows",
        "_read_refresh_scope_sync", "_read_recent_scope_sync", "_refresh_recent_batch", "_publish_recent_groups",
        "_check_recent_batch",
    )
    for descriptor in (vars(_CHARACTER_SCOPE_OWNER)[name],)
    for function in (descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor,)
))


_CHARACTER_PRESENTATION_OWNER_CODE = next(
    code for code in ConsoleCharacterContextController.refresh_presentation_if_scope_changed.__code__.co_consts
    if type(code) is CodeType and code.co_name == "owner_is_current"
)


def character_presentation_work(controller: Any, screen: Any) -> Awaitable[bool]:
    """Use the stock display facade while retaining custom no-argument callers."""
    if type(controller) is _CHARACTER_SCOPE_OWNER:
        return controller.refresh_presentation_if_scope_changed(screen)
    return controller.refresh_if_scope_changed()


def _stock_character_view_resume(controller, screen):
    """Select only the defining stock facade and original screen accessors."""
    from types import CellType, GetSetDescriptorType, MemberDescriptorType, ModuleType

    from ...DB import ChaChaNotes_DB
    from . import wiring

    def field(receiver, name):
        owner = type(receiver)
        if (
            inspect.getattr_static(owner, "__getattribute__")
            is not object.__getattribute__
        ):
            return _CHARACTER_VIEW_MISSING
        if (
            inspect.getattr_static(owner, name, _CHARACTER_VIEW_MISSING)
            is not _CHARACTER_VIEW_MISSING
        ):
            return _CHARACTER_VIEW_MISSING
        descriptor = inspect.getattr_static(owner, "__dict__", None)
        if (
            type(descriptor) is not GetSetDescriptorType
            and type(descriptor) is not MemberDescriptorType
        ):
            return _CHARACTER_VIEW_MISSING
        values = descriptor.__get__(receiver, owner)
        return (
            values.get(name, _CHARACTER_VIEW_MISSING)
            if type(values) is dict
            else _CHARACTER_VIEW_MISSING
        )

    try:
        if (
            type(controller) is not _CHARACTER_VIEW_CONTROLLER
            or inspect.getattr_static(_CHARACTER_VIEW_CONTROLLER, "__getattribute__")
            is not object.__getattribute__
            or type(wiring) is not ModuleType
            or sys.modules.get(__package__ + ".wiring") is not wiring
            or type(_CHARACTER_REFRESH_READERS) is not tuple
            or any(
                type(record) is not tuple
                or len(record) != 14
                or not _character_reader_current(record, controller)
                for record in _CHARACTER_REFRESH_READERS
            )
        ):
            return None
        metadata = vars(wiring).get("_CHARACTER_VIEW_ACCESSORS")
        if type(metadata) is not tuple or len(metadata) != 2:
            return None
        build_record, callbacks = metadata
        if (
            type(build_record) is not tuple
            or len(build_record) != 14
            or not _character_reader_current(build_record)
            or type(callbacks) is not tuple
            or len(callbacks) != 3
        ):
            return None
        namespace = build_record[5]
        values = vars(controller)
        for entry, expected_name in zip(
            callbacks,
            (
                "_database_accessor",
                "_current_character_accessor",
                "_open_conversation_accessor",
            ),
        ):
            if type(entry) is not tuple or len(entry) != 2:
                return None
            name, codes = entry
            if (
                type(name) is not str
                or name != expected_name
                or type(codes) is not tuple
                or not codes
                or any(type(code) is not CodeType for code in codes)
            ):
                return None
            callback = values.get(name)
            if (
                type(callback) is not FunctionType
                or type(vars(callback)) is not dict
                or callback.__globals__ is not namespace
                or not any(callback.__code__ is code for code in codes)
                or callback.__defaults__ is not None
                or callback.__kwdefaults__ is not None
                or type(callback.__closure__) is not tuple
                or len(callback.__closure__) != 1
                or type(callback.__closure__[0]) is not CellType
                or callback.__closure__[0].cell_contents is not screen
            ):
                return None
        if (
            type(ChaChaNotes_DB) is not ModuleType
            or sys.modules.get("tldw_chatbook.DB.ChaChaNotes_DB") is not ChaChaNotes_DB
        ):
            return None
        records = vars(ChaChaNotes_DB).get("_CHARACTER_REFRESH_READERS")
        if (
            type(records) is not tuple
            or not records
            or any(
                type(record) is not tuple
                or len(record) != 14
                or not _character_reader_current(record)
                for record in records
            )
        ):
            return None
        app = field(screen, "app_instance")
        database = (
            field(app, "chachanotes_db")
            if app is not _CHARACTER_VIEW_MISSING
            else _CHARACTER_VIEW_MISSING
        )
        if (
            type(database) is not records[0][0]
            or field(database, "is_memory_db") is not False
        ):
            return None
        facade_record = next(
            record
            for record in _CHARACTER_REFRESH_READERS
            if record[1] == "refresh_presentation_if_scope_changed"
        )
        if _CHARACTER_VIEW_FACADE is not facade_record[3]:
            return None
        return tuple(
            values[name]
            for name in (
                "_database_accessor",
                "_current_character_accessor",
                "_open_conversation_accessor",
            )
        ) + (app, database, _CHARACTER_VIEW_FACADE)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


async def _resume_stock_character_view(controller, screen, selected):
    """Recheck a queued selection before entering its original finite facade."""
    if (
        _stock_character_view_resume is not _CHARACTER_VIEW_SELECTOR
        or _CHARACTER_VIEW_SELECTOR.__code__ is not _CHARACTER_VIEW_SELECTOR_CODE
    ):
        return
    current = _CHARACTER_VIEW_SELECTOR(controller, screen)
    if current is None or any(
        actual is not expected for actual, expected in zip(current, selected)
    ):
        return
    await selected[-1](controller, screen, force_fresh=True)


def character_view_resume_work(
    controller: ConsoleCharacterContextController, screen: Any
) -> Awaitable[bool | None]:
    """Keep custom/direct callbacks; route stock resume through physical retirement."""
    selected = _stock_character_view_resume(controller, screen)
    if selected is None:
        return controller.refresh_if_scope_changed()
    return _resume_stock_character_view(controller, screen, selected)


_CHARACTER_VIEW_MISSING = object()


_CHARACTER_VIEW_CONTROLLER = ConsoleCharacterContextController


_CHARACTER_VIEW_FACADE = (
    ConsoleCharacterContextController.refresh_presentation_if_scope_changed
)


_CHARACTER_VIEW_SELECTOR = _stock_character_view_resume


_CHARACTER_VIEW_SELECTOR_CODE = _stock_character_view_resume.__code__
