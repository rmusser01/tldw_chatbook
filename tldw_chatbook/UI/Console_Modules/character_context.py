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


class _ConsoleCharacterReaderChanged(RuntimeError):
    """A previously stock finite reader changed during its original body."""


class _ConsoleCharacterAmbientError(RuntimeError):
    """Carry an unexpected original loop-check error through physical retirement."""

    def __init__(self, error: Exception) -> None:
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
            except Exception as error:  # noqa: BLE001 - original accessor contract.
                check.set_exception(_ConsoleCharacterAmbientError(error))

        try:
            self.loop.call_soon_threadsafe(validate)
            return check.result(timeout=_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS)
        finally:
            check.cancel()


def _stock_character_refresh_readers(controller, presentation, generation):
    """Select only the original display closure and concrete file DB route."""
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
    from ...DB import ChaChaNotes_DB, base_db
    from ...Character_Chat import character_conversation_navigation as navigation

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
            row.target.conversation_id if row.target else row.unresolved.conversation_id
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
    ) -> None:
        self._progress_counts = progress_counts
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
        self._successful_refresh_generation: int | None = None
        self._presentation_scope_lock = asyncio.Lock()
        self._presentation_scope_key: tuple | None = None
        self._presentation_scope_at = 0.0
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
    def _read_database_scope_metadata(
        database: Any, *, _reader_check: Callable[[], None] | None = None
    ) -> tuple[str, int]:
        if _reader_check is not None:
            _reader_check()
        authority = str(database.get_local_authority_id())
        if _reader_check is not None:
            _reader_check()
        revision = int(database.get_character_conversation_search_revision())
        if _reader_check is not None:
            _reader_check()
        return authority, revision

    def _read_database_scope_metadata_pair(
        self,
        database: Any,
        current: tuple[int, str] | None,
        open_conversation_id: str,
        loop: asyncio.AbstractEventLoop,
        *,
        _reader_check: Callable[[], None] | None = None,
    ) -> tuple[tuple[str, int], tuple[str, int]] | None:
        """Keep paired reads on one finite handle and ambient checks on the loop."""
        metadata_before = (
            self._read_database_scope_metadata(database)
            if _reader_check is None
            else self._read_database_scope_metadata(
                database, _reader_check=_reader_check
            )
        )
        ambient_check: Future[bool] = Future()

        def validate_ambient_on_loop() -> None:
            if not ambient_check.set_running_or_notify_cancel():
                return
            try:
                if _reader_check is not None:
                    _reader_check()
                matches = self._ambient_scope_matches(
                    database, current, open_conversation_id
                )
                if _reader_check is not None:
                    _reader_check()
                ambient_check.set_result(matches)
            except Exception as exc:  # noqa: BLE001 - propagate accessor failures.
                ambient_check.set_exception(exc)

        # Awaiter cancellation leaves this finite callback in charge of its
        # connection. A stopped/closed event loop must not strand that handle;
        # an expired queued check is cancelled before it can inspect UI state.
        try:
            loop.call_soon_threadsafe(validate_ambient_on_loop)
            if not ambient_check.result(timeout=_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS):
                return None
        finally:
            ambient_check.cancel()
        return metadata_before, (
            self._read_database_scope_metadata(database)
            if _reader_check is None
            else self._read_database_scope_metadata(
                database, _reader_check=_reader_check
            )
        )

    async def _await_database_work(self, database: Any, work: Awaitable[Any]) -> Any:
        """Keep native Character callbacks owned until their physical return."""
        from ...DB.ChaChaNotes_DB import CharactersRAGDB

        if type(database) is not CharactersRAGDB or database.is_memory_db:
            return await work

        async def invoke() -> Any:
            return await work

        # This callback already existed without a configured Task-factory call.
        # A private standard Task retains it even if the caller is cancelled.
        owned = asyncio.Task(invoke(), loop=asyncio.get_running_loop())
        try:
            return await asyncio.shield(owned)
        except asyncio.CancelledError:
            while not owned.done():
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not owned.cancelled():
                owned.exception()
            raise

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
                metadata_pair = await self._await_database_work(
                    database,
                    run_owned_db_call(
                        database,
                        self._read_database_scope_metadata_pair,
                        database,
                        current,
                        open_conversation_id,
                        asyncio.get_running_loop(),
                    ),
                )
            except Exception:  # noqa: BLE001 - DB adapters have no shared error base.
                if not self._ambient_scope_matches(
                    database, current, open_conversation_id
                ):
                    continue
                raise _ConsoleCharacterScopeReadError(
                    database, current, open_conversation_id
                ) from None
            if metadata_pair is None:
                continue
            metadata_before, metadata_after = metadata_pair
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

    def _presentation_owner_key(self, screen: Any) -> tuple:
        """Name the local display owner without obtaining storage authority."""
        from tldw_chatbook import config

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
            getattr(screen, "_console_chat_tearing_down", False),
        )

    async def refresh_presentation_if_scope_changed(
        self, screen: Any, *, force_fresh: bool = False
    ) -> bool:
        """Bound display observations while preserving fresh live actions.

        Args:
            screen: Exact screen owning the ambient profile and active Chat.
            force_fresh: Bypass the display memo for the stock resume's fresh read.

        Returns:
            Whether a fresh Character state refresh was attempted.
        """
        async with self._presentation_scope_lock:
            # A waiter must recapture its owner after acquiring the lock.
            key = self._presentation_owner_key(screen)
            if getattr(screen, "_console_chat_tearing_down", False):
                return False
            if (
                not force_fresh
                and key == self._presentation_scope_key
                and time.monotonic() - self._presentation_scope_at
                < _CHARACTER_PRESENTATION_TTL_SECONDS
            ):
                return False
            self._presentation_scope_key = None
            cancelled = False

            def owner_is_current() -> bool:
                return (
                    not cancelled
                    and key[1:] == self._presentation_owner_key(screen)[1:]
                )

            async def observe() -> bool:
                fingerprint = self.state.scope_fingerprint
                readers = (
                    _stock_character_refresh_readers(
                        self, owner_is_current, self._generation
                    )
                    if fingerprint is None
                    or type(fingerprint) is ConsoleCharacterScopeFingerprint
                    else None
                )
                # A resident mismatch already determines the comparison; refresh
                # still captures and validates its complete original DB scope.
                known_changed = readers is not None and (
                    readers.database is key[6]
                    and readers.current == key[7]
                    and readers.conversation == key[8]
                    and (
                        fingerprint is None
                        or fingerprint.database_identity != id(readers.database)
                        or fingerprint.current_character_id
                        != (None if readers.current is None else readers.current[0])
                        or fingerprint.current_character_label
                        != ("" if readers.current is None else readers.current[1])
                        or fingerprint.open_conversation_id != readers.conversation
                    )
                )
                if known_changed:
                    if cancelled or key != self._presentation_owner_key(screen):
                        return False
                else:
                    try:
                        snapshot = await self._capture_scope()
                    except (
                        _ConsoleCharacterScopeChanged,
                        _ConsoleCharacterScopeReadError,
                    ):
                        if cancelled or key != self._presentation_owner_key(screen):
                            return False
                    else:
                        if cancelled or key != self._presentation_owner_key(screen):
                            return False
                        if snapshot.fingerprint == self.state.scope_fingerprint:
                            if not self.state.error:
                                self._presentation_scope_key = key
                                self._presentation_scope_at = time.monotonic()
                            return False
                # refresh owns its generation increment. Preserve every other
                # ambient owner and its fresh generation/commit checks.
                expected_generation = self._generation + 1
                refresh_record = None
                original_refresh = False
                try:
                    if type(_CHARACTER_REFRESH_READERS) is tuple:
                        refresh_records = tuple(
                            record
                            for record in _CHARACTER_REFRESH_READERS
                            if type(record) is tuple
                            and len(record) == 14
                            and type(record[1]) is str  # noqa: E721 - exact optional metadata shape.
                            and record[1] == "refresh"
                        )
                        if len(refresh_records) == 1:
                            refresh_record = refresh_records[0]
                            original_refresh = _character_reader_current(
                                refresh_record, self
                            )
                except (AttributeError, TypeError, ValueError):
                    pass  # Unknown optional metadata never establishes a memo.
                await self.refresh(_presentation_is_current=owner_is_current)
                # Only this exact invocation's accepted stock publication can
                # establish a memo; an earlier successful state is insufficient.
                if (
                    original_refresh
                    and _character_reader_current(refresh_record, self)
                    and self._successful_refresh_generation == expected_generation
                    and self._generation == expected_generation
                    and owner_is_current()
                    and not self.state.error
                    and self.state.scope_fingerprint is not None
                ):
                    self._presentation_scope_key = self._presentation_owner_key(screen)
                    self._presentation_scope_at = time.monotonic()
                return True

            owned = asyncio.create_task(observe())
            try:
                return await asyncio.shield(owned)
            except asyncio.CancelledError:
                cancelled = True
                self._presentation_scope_key = None
                # Keep the coalescing lock until the finite callback retires.
                # Cancellation neither publishes a memo nor closes its handle.
                while not owned.done():
                    try:
                        await asyncio.shield(owned)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 - cancellation takes precedence.
                        break
                if not owned.cancelled():
                    owned.exception()
                raise

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
        self, readers: _CharacterRefreshReaders
    ) -> _ConsoleCharacterScopeSnapshot:
        """Run all original pair reads and post-pair ambient checks fresh."""
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
            except _ConsoleCharacterReaderChanged:
                raise
            except Exception:  # noqa: BLE001 - preserve metadata recovery semantics.
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
            if not readers.ambient_on_loop():
                raise _ConsoleCharacterScopeChanged("Character display owner changed")
            authority, revision = pair[1]
            return _ConsoleCharacterScopeSnapshot(
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
        raise _ConsoleCharacterScopeChanged("Character context scope did not settle")

    def _read_recent_scope_sync(
        self, readers: _CharacterRefreshReaders
    ) -> _CharacterRecentRead:
        """Only the inner pair/groups/pair share this existing finite interval."""
        readers.require_current()
        snapshot = readers.scope.__func__(readers.controller, readers)
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
        except Exception:  # noqa: BLE001 - final pair still gates load-error publication.
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
                    readers.database, readers.owned_call(readers.database, read_current)
                )
            except _ConsoleCharacterReaderChanged:
                return
            except _ConsoleCharacterScopeChanged:
                continue
            except _ConsoleCharacterAmbientError as error:
                raise error.error from None
            except Exception:  # noqa: BLE001 - initial scope admission/metadata error.
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
        readers = _stock_character_refresh_readers(
            self, _presentation_is_current, generation
        )
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
                    database,
                    run_owned_db_call(
                        database, self._load_recent_sync, database, fingerprint
                    ),
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
            self._publish_recent_groups(fingerprint, groups, details)
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
                    snapshot.database,
                    run_owned_db_call(
                        snapshot.database,
                        self._load_unavailable_details_sync,
                        service,
                        bounded_groups,
                    ),
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
                snapshot.database,
                run_owned_db_call(
                    snapshot.database,
                    self._keyword_page_sync,
                    snapshot.database,
                    snapshot.fingerprint,
                    query,
                    offset,
                    limit,
                ),
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
                    snapshot.database,
                    run_owned_db_call(
                        snapshot.database,
                        self._search_sync,
                        snapshot.database,
                        snapshot.fingerprint,
                        normalized,
                    ),
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
                    snapshot.database,
                    run_owned_db_call(
                        snapshot.database,
                        lambda service=service: (
                            service.refresh_unresolved_evidence(key),
                            service.repair_candidates(
                                key, limit=CONSOLE_CHARACTER_REPAIR_CANDIDATE_LIMIT
                            ),
                        ),
                    ),
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


# Definition-time identities for the finite Character display reader only.
_CHARACTER_REFRESH_READERS = tuple(
    (
        ConsoleCharacterContextController,
        name,
        descriptor,
        function,
        function.__code__,
        function.__globals__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
        __file__,
        __spec__,
        getattr(__spec__, "origin", None),
    )
    for name in (
        "refresh_presentation_if_scope_changed",
        "refresh_if_scope_changed",
        "_presentation_owner_key",
        "refresh",
        "_begin",
        "_publish",
        "_current_character_identity",
        "_open_conversation_identity",
        "_ambient_scope_matches",
        "_read_database_scope_metadata",
        "_read_database_scope_metadata_pair",
        "_await_database_work",
        "_capture_scope",
        "_scope_is_current",
        "_operation_scope_is_current",
        "_load_recent_sync",
        "_load_unavailable_details_sync",
        "_unavailable_rows",
        "_read_refresh_scope_sync",
        "_read_recent_scope_sync",
        "_refresh_recent_batch",
        "_publish_recent_groups",
    )
    for descriptor in (vars(ConsoleCharacterContextController)[name],)
    for function in (
        descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor,
    )
)

_CHARACTER_PRESENTATION_OWNER_CODE = next(
    code
    for code in ConsoleCharacterContextController.refresh_presentation_if_scope_changed.__code__.co_consts
    if type(code) is CodeType and code.co_name == "owner_is_current"
)


def _stock_character_view_resume(controller, screen):
    """Select only the defining stock facade and original screen accessors."""
    from types import CellType, GetSetDescriptorType, MemberDescriptorType, ModuleType
    from . import wiring
    from ...DB import ChaChaNotes_DB

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
            if type(values) is dict  # noqa: E721 - refuse custom mapping dispatch.
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
                type(name) is not str  # noqa: E721 - immutable defining field names.
                or name != expected_name
                or type(codes) is not tuple
                or not codes
                or any(type(code) is not CodeType for code in codes)
            ):
                return None
            callback = values.get(name)
            if (
                type(callback) is not FunctionType
                or type(vars(callback)) is not dict  # noqa: E721 - refuse custom metadata.
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
