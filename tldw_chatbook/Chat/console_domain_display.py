"""Domain owner of the original Console display producers and finite FULL reads.

One original producer per section. No App/Screen stand-in, transport authority,
second policy cache or timer is constructed here.

"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Never, TypeVar

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.character_context import (
        ConsoleCharacterContextController,
        ConsoleCharacterContextState,
        _CharacterRecentRead,
        _CharacterRefreshReaders,
    )
    from .console_chat_store import ConsoleChatSession
    from .console_domain_services import ConsoleDomainServices

    _ReadResult = TypeVar("_ReadResult")
    _ScopeMetadataPair = tuple[tuple[str, int], tuple[str, int]]
    _CharacterPhaseResult = _CharacterRecentRead | _ScopeMetadataPair | None

from .console_display_phases import (
    ConsoleCharacterDisplayPhases,
    ConsoleDisplayPhaseRefused,
    ConsoleDisplaySessionChanged,
    ConsoleDisplayPhysicalOwner,
)
from .console_preparation_reads import (
    ConsolePreparationRead,
    _READS_LOCK,
    _observe_locked,
    _retire,  # noqa: F401 - Retain the original source import and initialization order.
)


@dataclass(slots=True, eq=False)
class _DisplayCheckpoint:
    full: object
    runner: object
    ordinal: int
    kind: str
    settled: bool
    source: object
    session: object
    incarnation: str
    stamp: object
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _DisplayDeclined:
    reason: str


@dataclass(frozen=True, slots=True)
class _DisplaySelection:
    owner: object
    task: object
    controls: object
    sources: object
    source_revision: int
    session: object
    incarnation: str
    character: object
    context: object
    appearance: object
    port: object
    stamp: object
    parent_entry: object
    command_source: object
    loop: object
    task_factory: object
    backend_checkpoint: object = None


def _display_stock_eligible(owner):
    """Original finite stock predicates, before selected admission only."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.character_context import (
        _stock_character_presentation,
    )
    from tldw_chatbook.UI.Console_Modules.config_sync import _stock_main_context_read
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleContextReadSnapshot,
        _CONTEXT_SNAPSHOT_WARM_READERS,
    )
    from .console_received_dispatch import stock_native_methods

    database = owner.sources.chat_database
    return (
        type(database) is CharactersRAGDB
        and vars(database).get("is_memory_db") is False
        and _stock_character_presentation(owner.character)
        and _stock_main_context_read(owner.controller)
        and type(owner.context) is ConsoleContextReadSnapshot
        and stock_native_methods(owner.context, _CONTEXT_SNAPSHOT_WARM_READERS)
        and type(owner.appearance) is ConsoleAppearanceDisplay
    )


def _caller_owns_display_native_scope(database):
    """Read existing custody only; never acquire a connection to inspect it."""
    import sqlite3
    from tldw_chatbook.DB.base_db import _OPERATION_OWNERS
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    if (
        id(database) in getattr(_OPERATION_OWNERS, "ids", ())
        or getattr(storage._operation_local, "operation", None) is not None
    ):
        return True
    connection = getattr(database._local, "conn", None)
    if connection is None or not database._connection_quiescence.is_registered(
        connection
    ):
        return False
    try:
        return sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True  # Inconsistent registered custody stays with its original route.


def preselect_display(
    owner: ConsoleDomainDisplayOwner,
) -> _DisplaySelection | _DisplayDeclined:
    """Decline unsupported composition before any FULL/effect/native issuance."""
    from .console_control_dispatch import _CommandSource, _command_source

    owner.require_current()
    controls = owner.runtime._execution_service._local_controls
    from .console_control_dispatch import _display_control_sources_current

    if not _display_control_sources_current(controls):
        return _DisplayDeclined("custom-display-control-definitions")
    controls._require_graph()
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    if factory not in (None, asyncio.eager_task_factory):
        return _DisplayDeclined("custom-task-factory")
    session = owner.active_session()
    if session is None:
        return _DisplayDeclined("no-original-session")
    # These are unsupported pre-existing routes, not failed selected reads.
    if not _display_stock_eligible(owner):
        return _DisplayDeclined("custom-or-memory-producer")
    if _caller_owns_display_native_scope(owner.sources.chat_database):
        return _DisplayDeclined("original-caller-native-scope")
    port = owner.domain.presentation
    stamp = port.capture_display(owner)
    entry = port._display_full_entries.get(task)
    actual = controls._settings_command_tasks.get(task)
    inherited = _command_source.get()
    if actual is not None:
        if type(actual) is not _CommandSource or inherited is not actual:
            raise ConsoleDisplayPhaseRefused("Display command Task/source changed")
    # An unrelated inherited context grants nothing. Autonomous work must
    # independently possess the exact original FULL entry checked below.
    if stamp is None and actual is None:
        raise ConsoleDisplayPhaseRefused("Detached display requires its actual command")
    if stamp is not None:
        port.require_display_full_entry(entry, owner, task)
    from .console_backend_presentation import ConsoleBackendPresentation

    backend_checkpoint = (
        port.capture_display_reuse_checkpoint(owner)
        if type(port) is ConsoleBackendPresentation else None
    )
    return _DisplaySelection(
        owner,
        task,
        controls,
        owner.sources,
        controls.source_revision,
        session,
        session.incarnation_id,
        owner.character,
        owner.context,
        owner.appearance,
        port,
        stamp,
        entry,
        actual,
        loop,
        factory,
        backend_checkpoint,
    )


def require_selected_display_sources(full: ConsoleDisplayFull) -> None:
    """Once admitted, false stock/source checks refuse; they never select legacy."""
    selected = full.selection
    owner = full.owner
    if (
        type(selected) is not _DisplaySelection
        or selected.owner is not owner
        or owner.sources is not selected.sources
        or owner.character is not selected.character
        or owner.context is not selected.context
        or owner.appearance is not selected.appearance
        or selected.loop.get_task_factory() is not selected.task_factory
        or owner.store._sessions.get(selected.session.id) is not selected.session
        or selected.session.incarnation_id != selected.incarnation
        or not _display_stock_eligible(owner)
    ):
        raise ConsoleDisplayPhaseRefused("Admitted display defining source changed")
    owner.require_current()
    owner.require_character_sources(selected.character)
    # Checks the same retained effect, not fresh _effects_open/source_revision.
    # A source fence stops NEW FULLs; the admitted one settles before publication.
    from .console_control_dispatch import _display_control_sources_current

    if not _display_control_sources_current(selected.controls):
        raise ConsoleDisplayPhaseRefused("Admitted display control source changed")
    selected.controls._require_display_effect(full)


class ConsoleDisplayUnsupported(RuntimeError):
    """Selected composition refused before native entry."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Selected Console display is unsupported: {reason}")
        self.reason = reason


class ConsoleDomainDisplayOwner:
    """Actual source/store/controller owner of the three original producers.

    The selected view refers to these same domain-owned producer instances.
    The legacy constructor is unchanged. Appearance state has one owner,
    preserving the original map, coverage and age.
    """

    def __init__(self, domain: ConsoleDomainServices) -> None:
        self.domain = domain
        self.runtime = domain.runtime
        self.sources = domain.sources
        self.store = None
        self.controller = None
        self._graph_bound = False
        self._full_tasks = {}
        self._preparation_reads = set()
        self._full = None
        self._character_reuse_checkpoint = None
        self.character = None
        self.context = None
        self.appearance = None

    def initialize(self) -> None:
        """Create the original domain producers once, without a view."""
        from tldw_chatbook.UI.Console_Modules.character_context import (
            ConsoleCharacterContextController,
        )
        from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
            ConsoleContextReadSnapshot,
        )

        _require_definitions(self)
        self.domain.require_current(self.sources)
        if self.domain.runtime is not self.runtime:
            raise ConsoleDisplayPhaseRefused("Display construction runtime changed")
        if self.character is not None:
            return
        self.character = ConsoleCharacterContextController(
            database_accessor=self.database,
            current_character_accessor=self.current_character,
            open_conversation_accessor=self.open_conversation,
            activate_target=self.no_view_action,
            navigate_roleplay=self.no_view_action,
            navigate_repair=self.no_view_action,
            navigate_inspection=self.no_view_action,
            navigate_unavailable_browse=self.no_view_action,
            navigate_roleplay_home=self.no_view_action,
            navigate_library_home=self.no_view_action,
            start_console=self.no_view_action,
            state_changed=self.publish_character,
            run_active=self.run_active,
        )
        self.character._domain_display_owner = self
        self.context = ConsoleContextReadSnapshot(max_age=1.0)
        self.context._domain_display_owner = self
        self.appearance = ConsoleAppearanceDisplay(self)

    def require_current(self) -> None:
        _require_definitions(self)
        self.domain.require_current(self.sources)
        if (
            not self._graph_bound
            or self.domain.runtime is not self.runtime
            or self.runtime._disposed
            or self.runtime._chat_store is not self.store
            or self.runtime._chat_controller is not self.controller
            or self.controller.store is not self.store
            or self.controller._disposed
            or self.store.persistence.db is not self.sources.chat_database
        ):
            raise ConsoleDisplayPhaseRefused("Domain display owner changed")

    def require_character_sources(
        self, character: ConsoleCharacterContextController
    ) -> None:
        self.require_current()
        if character is not self.character:
            raise ConsoleDisplayPhaseRefused("Character domain owner changed")
        for name, method in (
            ("_database_accessor", "database"),
            ("_current_character_accessor", "current_character"),
            ("_open_conversation_accessor", "open_conversation"),
            ("_state_changed", "publish_character"),
            ("_run_active", "run_active"),
        ):
            bound = getattr(character, name)
            if (
                type(bound) is not MethodType
                or bound.__self__ is not self
                or bound.__func__ is not vars(ConsoleDomainDisplayOwner)[method]
            ):
                raise ConsoleDisplayPhaseRefused("Character defining accessor changed")

    def active_session(self) -> ConsoleChatSession | None:
        self.require_current()
        return next(
            (
                item
                for item in self.store.sessions()
                if item.id == self.store.active_session_id
            ),
            None,
        )

    def database(self) -> CharactersRAGDB:
        self.require_current()
        return self.sources.chat_database

    def current_character(self) -> tuple[int, str] | None:
        # The original native-session rail getters, without their Screen route.
        session = self.active_session()
        character_id = session.local_character_id() if session is not None else None
        if character_id is None:
            return None
        name = getattr(session, "character_name", None)
        return character_id, str(name) if name else "Current character"

    def open_conversation(self) -> str | None:
        session = self.active_session()
        conversation = (
            session.persisted_conversation_id if session is not None else None
        )
        return str(conversation) if conversation else None

    def run_active(self) -> bool:
        self.require_current()
        # This fixed port body calls the actual original _console_run_active:
        # acknowledged Enter and image editing are not inferred from run_state.
        # With no view it returns None, never a positive attached witness.
        active = self.domain.presentation.display_run_active(self)
        return active is True

    def character_key(self) -> tuple[object, ...]:
        from tldw_chatbook import config

        session = self.active_session()
        session_id = self.store.active_session_id
        active = self.run_active()
        return (
            self.character._generation,
            config.current_config_identity(),
            self.sources,
            id(self.sources.configuration),
            self.sources.configuration,
            self.sources.chat_database,
            self.database(),
            self.character._current_character_identity(),
            self.character._open_conversation_identity(),
            self.store,
            session_id,
            id(session),
            session,
            getattr(session, "workspace_id", None),
            getattr(session, "conversation_id", None),
            getattr(session, "conversation_binding_revision", None),
            self.store.session_settings_revision(session_id)
            if session is not None
            else None,
            self.store.payload_revision(session_id)
            if not active and session is not None
            else None,
            self.runtime._disposed,
            # Exact App/config/DB/store/view/visit/teardown and parent-run facts
            # are separately retained by the actual port, never synthesized.
            self.domain.presentation.display_character_key(self),
        )

    def no_view_action(self, *args: object, **kwargs: object) -> Never:
        raise RuntimeError("Character navigation requires its original attached action")

    def publish_character(self, state: ConsoleCharacterContextState) -> None:
        self.require_current()
        self.domain.presentation.publish_display_character(self, state)

    def full_for_current_task(self, section: str) -> ConsoleDisplayFull | None:
        task = asyncio.current_task()
        association = self._full_tasks.get(task)
        if association is None:
            return None
        full, expected_section = association
        if (
            full is not self._full
            or expected_section != section
            or full.tasks.get(section) is not task
            or task.done()
        ):
            raise ConsoleDisplayPhaseRefused("Display producer Task changed")
        full.require_current()
        return full

    async def read(self) -> None:
        from .console_control_settings import wait_settings_settled

        if not self._graph_bound:
            raise ConsoleDisplayUnsupported("display-graph-unbound")
        self.initialize()
        session = self.active_session()
        if session is None:
            raise ConsoleDisplayUnsupported("no-original-session")
        await wait_settings_settled(self.store, session.id)
        self.require_current()
        if self.active_session() is not session:
            raise ConsoleDisplayPhaseRefused("Display session changed before selection")
        selection = preselect_display(self)
        if type(selection) is _DisplayDeclined:
            # Applies to BOTH selected ReadProjection and selected autonomous UI.
            # No old-helper call, fake snapshot, new native work or mixed graph.
            raise ConsoleDisplayUnsupported(selection.reason)
        full = ConsoleDisplayFull(self, selection)
        # The private adapter validates the same real graph and registers
        # owner._full + the effect in one controls-lock acquisition. No await,
        # worker scheduling or producer exposure occurs before this admission.
        selection.controls._begin_display_effect(full, selection)
        # A failed/uncertain FULL cannot qualify a later observation. The
        # original character cache remains its sole value/TTL owner.
        self._character_reuse_checkpoint = None
        try:
            await full.run()
        finally:
            if full.logical is None and not full.tasks and not full.retirements:
                # Failure/cancellation before run's logical observer/producer
                # entry. No native could have been issued: every issued phase
                # is recorded before exposure and all producers are retained.
                full.phase_issuance_closed = True
            from .console_control_dispatch import finish_admitted_display

            finish_admitted_display(full)
            # Do NOT unconditionally clear self._full. Uncertain original
            # owners remain retained by the effect/observer/parent registration.
        if (
            selection.backend_checkpoint is not None
            and full.effect.completed.result() is True
            and selection.port.display_reuse_checkpoint_current(
                self, selection.backend_checkpoint,
            )
        ):
            self._character_reuse_checkpoint = full.backend_character_key

    def retired_session_change(self, entry: object, error: BaseException) -> bool:
        """Recognize a failed old-session FULL only after its actual settlement.

        Args:
            entry: The caller's original, still-registered FULL entry.
            error: The specific active-session refusal from that FULL.

        Returns:
            Whether this exact refusal may use the original UI deferred outcome.

        Raises:
            ConsoleDisplayPhaseRefused: Actual defining sources changed.
        """
        from .console_control_dispatch import _display_control_sources_current
        from .console_display_phases import require_display_phase_definitions
        from .console_presentation_port import _DisplayFullEntry

        self.require_current()
        if (
            type(error) is not ConsoleDisplaySessionChanged
            or type(entry) is not _DisplayFullEntry
        ):
            return False
        full = error.full
        if type(full) is not ConsoleDisplayFull:
            return False
        _require_definitions(full)
        require_display_phase_definitions(full)
        selected, effect, logical = full.selection, full.effect, full.logical
        if (
            full.owner is not self
            or entry.full is not full
            or entry.port is not self.domain.presentation
            or entry.task is not asyncio.current_task()
            or entry.port._display_full_entries.get(entry.task) is not entry
            or entry.body_returned
            or selected.parent_entry is not entry
            or selected.task is not entry.task
            or full._failed is not error
            or not full.phase_issuance_closed
            or not full.physically_retired
            or not all(task.done() for task in full.tasks.values())
            or logical is None
            or logical.task is not entry.task
            or logical.callback_returned
            or not logical.retired.done()
            or logical.retired.cancelled()
            or logical.retired.exception() is not None
            or effect is None
            or not effect.completed.done()
            or effect.completed.cancelled()
            or effect.completed.exception() is not None
            or effect.completed.result() is not True
            or effect.physically_retired is not True
            or effect.owner.full is not full
            or effect.owner.selection is not selected
            or full.store._sessions.get(full.session.id) is not full.session
            or full.session.incarnation_id != full.incarnation
        ):
            return False
        selected.controls._require_graph()
        self.domain.require_current(selected.sources)
        if (
            self.sources is not selected.sources
            or self.character is not selected.character
            or self.context is not selected.context
            or self.appearance is not selected.appearance
            or self.runtime._execution_service._local_controls is not selected.controls
            or self.domain.presentation is not selected.port
            or full.session is not selected.session
            or full.incarnation != selected.incarnation
            or not _display_stock_eligible(self)
            or not _display_control_sources_current(selected.controls)
            or selected.loop.get_task_factory() is not selected.task_factory
        ):
            raise ConsoleDisplayPhaseRefused("Retired display defining source changed")
        self.require_character_sources(selected.character)
        view = entry.port._require_display_owner(self)
        return entry.port._current_visit(entry.stamp) and view is entry.view

    def bind_original_graph(self) -> None:
        """Bind once, only after the original graph construction has completed."""
        _require_definitions(self)
        self.domain.require_current(self.sources)
        store = self.runtime._chat_store
        controller = self.runtime._chat_controller
        if (
            self.domain.runtime is not self.runtime
            or self.runtime.domain_services is not self.domain
            or store is None
            or controller is None
            or controller.store is not store
            or store.persistence.db is not self.sources.chat_database
            or self.runtime._disposed
            or controller._disposed
        ):
            raise ConsoleDisplayPhaseRefused("Original display graph is unavailable")
        if self._graph_bound:
            if self.store is not store or self.controller is not controller:
                raise ConsoleDisplayPhaseRefused("Display graph identity changed")
        else:
            self.store, self.controller = store, controller
            self._graph_bound = True
        self.require_current()


class ConsoleDisplayFull(ConsoleDisplayPhysicalOwner):
    """One retained logical read across fresh phases and late native fallback."""

    def __init__(
        self, owner: ConsoleDomainDisplayOwner, selection: _DisplaySelection
    ) -> None:
        self.owner = owner
        self.selection = selection
        self.effect = None
        self.sources = selection.sources
        self.session = selection.session
        self.incarnation = selection.incarnation
        from .console_control_dispatch import ConsoleLocalControls

        self.controls = selection.controls
        if type(self.controls) is not ConsoleLocalControls:
            raise ConsoleDisplayPhaseRefused(
                "Display requires its actual selected controls"
            )
        self.source_revision = selection.source_revision
        self.port = selection.port
        self.stamp = selection.stamp
        self.backend_character_key = (
            selection.backend_checkpoint, selection.session.id,
            selection.incarnation, selection.source_revision,
        ) if selection.backend_checkpoint is not None else None
        self.reuse_backend_character = (
            self.backend_character_key is not None
            and self.backend_character_key == owner._character_reuse_checkpoint
        )
        self.collecting = True
        self.sections = {}
        self.tasks = {}
        self._ready = set()
        self._publication = asyncio.get_running_loop().create_future()
        self._initial_phases = asyncio.get_running_loop().create_future()
        self.sections_retired = False
        self.character_readers = None
        self.character_demand = (
            selection.parent_entry.character_demand
            if selection.parent_entry is not None
            else None
        )
        self.character_result = None
        self._initial_character_scope_only = False
        self.pending_checkpoint = None
        self._runner_ordinals = {}
        self.logical = None
        self._failed = None
        super().__init__(
            runtime=owner.runtime,
            store=owner.store,
            database=owner.sources.chat_database,
            creator=owner,
            observers=[owner._preparation_reads, owner.runtime._preparation_reads],
            require_current=self.require_current,
        )

    def require_current(self) -> None:
        from .console_display_phases import require_display_phase_definitions

        _require_definitions(self)
        require_display_phase_definitions(self)
        self.owner.require_current()
        require_selected_display_sources(self)
        if (
            self.owner._full is not self
            or self.runtime._execution_service._local_controls is not self.controls
            or self.controls._domain is not self.owner.domain
            or self.owner.domain.sources is not self.sources
            or self.store._sessions.get(self.session.id) is not self.session
            or self.session.incarnation_id != self.incarnation
            or self._failed is not None
        ):
            raise ConsoleDisplayPhaseRefused("Logical display source/session changed")
        if self.store.active_session_id != self.session.id:
            raise ConsoleDisplaySessionChanged(self)
        if self.reuse_backend_character and not self.port.display_reuse_checkpoint_current(
            self.owner, self.selection.backend_checkpoint,
        ):
            raise ConsoleDisplayPhaseRefused("Backend display visibility changed")

    def require_phase_task(self) -> None:
        current = asyncio.current_task()
        if self.logical is None or (
            current is not self.logical.task
            and (
                current is not self.tasks.get("character")
                or self.owner._full_tasks.get(current) != (self, "character")
            )
        ):
            raise ConsoleDisplayPhaseRefused("Display phase has no exact owning Task")

    def _section_task(self, section):
        if self.tasks.get(section) is not asyncio.current_task():
            raise ConsoleDisplayPhaseRefused(
                "Display capture has no exact producer Task"
            )
        self.require_current()

    def capture(
        self,
        section: str,
        database: CharactersRAGDB,
        read: Callable[[], _ReadResult],
        *,
        reads: set[ConsolePreparationRead],
        require_current: Callable[[], None] | None = None,
    ) -> asyncio.Future[_ReadResult] | None:
        self._section_task(section)
        if (
            not self.collecting
            or database is not self.database
            or section not in {"context", "appearance"}
            or section in self.sections
        ):
            return None
        future = asyncio.get_running_loop().create_future()
        self.sections[section] = read, require_current, future
        with _READS_LOCK:
            _observe_locked(self.logical, reads)
        if not any(reads is observer for observer in self.observers):
            self.observers.append(reads)
        return future

    def capture_character(
        self, readers: _CharacterRefreshReaders
    ) -> asyncio.Future[_CharacterPhaseResult] | None:
        self._section_task("character")
        if (
            not self.collecting
            or "character" in self.sections
            or readers.controller is not self.owner.character
            or readers.database is not self.database
        ):
            return None
        future = asyncio.get_running_loop().create_future()
        self.character_readers = readers
        self.sections["character"] = readers, None, future
        reads = readers.controller._preparation_reads
        with _READS_LOCK:
            _observe_locked(self.logical, reads)
        if not any(reads is observer for observer in self.observers):
            self.observers.append(reads)
        return future

    async def character_fallback(
        self, readers: _CharacterRefreshReaders
    ) -> _CharacterRecentRead:
        """Original finite fallback, same exact section Task and logical owner."""
        self._section_task("character")
        if readers.controller is not self.owner.character:
            raise ConsoleDisplayPhaseRefused("Character fallback changed its FULL")
        if self.collecting:
            captured = self.capture_character(readers)
            if captured is None:
                raise ConsoleDisplayPhaseRefused(
                    "Character initial read was already captured"
                )
            return await captured
        await asyncio.shield(self._initial_phases)
        self.require_current()
        self.character_readers = readers
        return await ConsoleCharacterDisplayPhases(self, readers).read()

    async def character_scope_pair(
        self,
        controller: ConsoleCharacterContextController,
        database: CharactersRAGDB,
        current: tuple[int, str] | None,
        conversation: str,
    ) -> _ScopeMetadataPair | None:
        """Split the original late probe's stock pair without replacing its loop."""
        from dataclasses import replace
        from tldw_chatbook.UI.Console_Modules import character_context as source

        self._section_task("character")
        self.require_current()
        captured = self.character_readers
        if (
            captured is None
            or captured.controller is not controller
            or controller is not self.owner.character
            or database is not self.database
            or not source._stock_character_scope_pair(controller)
        ):
            raise ConsoleDisplayPhaseRefused(
                "Late character probe has no original reader"
            )
        # These are the original _capture_scope attempt's own accessor results;
        # no source authority, retry count or settled fingerprint is substituted.
        readers = replace(
            captured,
            current=current,
            conversation=conversation,
            generation=controller._generation,
            unchanged_from=None,
        )
        readers.require_current()
        if self.collecting:
            self._initial_character_scope_only = True
            pending = self.capture_character(readers)
            if pending is None:
                raise ConsoleDisplayPhaseRefused("Character scope already captured")
            return await pending
        await asyncio.shield(self._initial_phases)
        self.require_current()
        return await ConsoleCharacterDisplayPhases(self, readers)._pair(settled=False)

    async def character_checkpoint(
        self,
        runner: ConsoleCharacterDisplayPhases,
        kind: str,
        ordinal: int,
        *,
        settled: bool,
    ) -> bool:
        from tldw_chatbook.UI.Console_Modules import character_context as source

        self.require_current()
        self.require_phase_task()
        if (
            not self.physically_retired
            or self.pending_checkpoint is not None
            or runner.full is not self
            or runner.readers.controller is not self.owner.character
            or kind
            not in {"pair-ambient", "scope-owner", "scope-error", "post-groups-owner"}
            or ordinal != self._runner_ordinals.get(runner, 0) + 1
        ):
            raise ConsoleDisplayPhaseRefused("Display checkpoint phase changed")
        self._runner_ordinals[runner] = ordinal
        request = _DisplayCheckpoint(
            self,
            runner,
            ordinal,
            kind,
            settled,
            self.sources,
            self.session,
            self.incarnation,
            self.stamp,
        )
        self.pending_checkpoint = request
        try:
            # Fixed concrete port: no callback or receiver supplied by a caller.
            # It validates the real optional attachment; None stays no witness.
            self.port.require_display_checkpoint(self.owner, request)
            readers = runner.readers
            try:
                if kind == "pair-ambient":
                    check = (
                        readers.require_loop_readers
                        if settled
                        else readers.require_current
                    )
                    check()
                    matches = source._stock_character_scope_pair(
                        readers.controller
                    ) and (
                        readers.controller._ambient_scope_matches(
                            readers.database, readers.current, readers.conversation
                        )
                    )
                    check()
                    outcome = matches and source._stock_character_scope_pair(
                        readers.controller
                    )
                else:
                    outcome = readers.ambient_current()
            except source._ConsoleCharacterReaderChanged:
                raise
            except BaseException as error:
                raise source._ConsoleCharacterAmbientError(error) from error
            self.require_current()
            self.port.require_display_checkpoint(self.owner, request)
            request.consumed = True
            return outcome
        finally:
            if self.pending_checkpoint is request:
                self.pending_checkpoint = None

    async def publication_ready(self, section: str) -> None:
        """Called after this section's last native/fallback decision, before age."""
        self._section_task(section)
        self._ready.add(section)
        self._release_publication_if_ready()
        # Character may still take an original late fallback after an unchanged
        # provisional result. Its exact Task owns that decision. Context and
        # appearance wait for that Task's actual completion, not a ready flag.
        if section != "character":
            await asyncio.shield(self._publication)
        self.require_current()
        if not self.physically_retired:
            raise ConsoleDisplayPhaseRefused(
                "Display publication still owns native work"
            )
        self.port.require_display_publication(self.owner, self.stamp, self)

    def _release_publication_if_ready(self):
        if self.collecting or not self.sections_retired or self._publication.done():
            return
        character = self.tasks.get("character")
        if character is None or not character.done():
            return
        if not all(
            name in self._ready or task.done() for name, task in self.tasks.items()
        ):
            return
        if self._failed is not None:
            self._publication.set_exception(self._failed)
        elif not self.physically_retired:
            self._publication.set_exception(
                ConsoleDisplayPhaseRefused("Display FULL retirement is unproven")
            )
        else:
            self._publication.set_result(None)

    def _producer_done(self, task):
        self._release_publication_if_ready()

    def fail(self, error: BaseException) -> None:
        if not self._initial_phases.done():
            self._initial_phases.set_exception(error)
        if self._failed is None:
            self._failed = error
        if not self._publication.done():
            self._publication.set_exception(error)

    async def _producer(self, section, started):
        await started
        self._section_task(section)
        try:
            if section == "character":
                self.character_result = (
                    await self.owner.character.refresh_presentation_if_scope_changed(
                        None,
                        force_fresh=self.stamp is None and not self.reuse_backend_character,
                        _display_read=self,
                    )
                )
            elif section == "context":
                await self.owner.context.warm(
                    self.owner.controller,
                    self.session.id,
                    _display_read=self,
                )
            else:
                await self.owner.appearance.warm(self)
        except BaseException as error:
            self.fail(error)
            raise
        finally:
            self._ready.add(section)
            self._release_publication_if_ready()

    async def run(self) -> None:
        self.require_current()
        task = asyncio.current_task()
        self.logical = ConsolePreparationRead(
            self.owner,
            self.run,
            self.sources,
            task,
            self.session.id,
            asyncio.get_running_loop().create_future(),
        )
        with _READS_LOCK:
            for observer in self.observers:
                _observe_locked(self.logical, observer)
        try:
            for section in ("character", "context", "appearance"):
                require_selected_display_sources(self)
                started = asyncio.get_running_loop().create_future()
                operation = self._producer(section, started)
                try:
                    producer = asyncio.create_task(operation)
                except BaseException:
                    operation.close()
                    raise
                self.tasks[section] = producer
                producer.add_done_callback(self._producer_done)
                self.owner._full_tasks[producer] = self, section
                if not started.done():
                    started.set_result(None)
            # Original capture yield; a cold/busy/late section cannot extend it.
            await asyncio.sleep(0)
            self.collecting = False
            results = {}
            if "character" in self.sections:
                readers, _, _future = self.sections["character"]
                try:
                    phases = ConsoleCharacterDisplayPhases(self, readers)
                    value = (
                        await phases._pair(settled=False)
                        if self._initial_character_scope_only
                        else await phases.read()
                    )
                except ConsoleDisplayPhaseRefused:
                    raise
                except Exception as error:
                    from tldw_chatbook.UI.Console_Modules.character_context import (
                        _ConsoleCharacterReaderChanged,
                    )

                    if isinstance(error, _ConsoleCharacterReaderChanged):
                        raise ConsoleDisplayPhaseRefused(
                            "Selected character reader changed"
                        ) from error
                    results["character"] = False, error
                else:
                    if value is None:
                        raise ConsoleDisplayPhaseRefused(
                            "Selected character read was refused"
                        )
                    results["character"] = True, value
            remaining = tuple(
                (name, self.sections[name])
                for name in ("context", "appearance")
                if name in self.sections
            )
            if remaining:

                def read_remaining():
                    values = {}
                    for name, (read, validate, _future) in remaining:
                        try:
                            if validate is not None:
                                validate()
                            value = read()
                            if validate is not None:
                                validate()
                        except Exception as error:
                            values[name] = False, error
                        else:
                            values[name] = True, value
                    return values

                results.update(await self.native_phase(read_remaining))
            self.require_current()
            if not self.physically_retired:
                raise ConsoleDisplayPhaseRefused("Display sections have not retired")
            self.sections_retired = True
            if not self._initial_phases.done():
                self._initial_phases.set_result(None)
            for name, (_read, _check, future) in self.sections.items():
                if not future.done():
                    ok, value = results[name]
                    if ok:
                        future.set_result(value)
                    else:
                        future.set_exception(value)
            self._release_publication_if_ready()
            outcomes = await asyncio.gather(
                *self.tasks.values(), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            self.require_current()
        except BaseException as error:
            self.fail(error)
            raise
        finally:
            self.collecting = False
            for _, _, future in self.sections.values():
                if not future.done():
                    future.cancel()
            for producer in self.tasks.values():
                if not producer.done():
                    producer.cancel()
            settled = asyncio.gather(*self.tasks.values(), return_exceptions=True)
            cancelled = None
            while not settled.done():
                try:
                    await asyncio.shield(settled)
                except asyncio.CancelledError as error:
                    cancelled = error
            for producer in self.tasks.values():
                self.owner._full_tasks.pop(producer, None)
            # Consume a failure even when every original publisher refused early.
            if self._publication.done() and not self._publication.cancelled():
                self._publication.exception()
            if self._initial_phases.done() and not self._initial_phases.cancelled():
                self._initial_phases.exception()
            self.logical.callback_returned = self._failed is None
            self.phase_issuance_closed = True
            if cancelled is not None:
                raise cancelled


class ConsoleAppearanceDisplay:
    """Own the original appearance map, coverage and age."""

    def __init__(self, owner: ConsoleDomainDisplayOwner) -> None:
        self.owner = owner
        self.cache = None
        self.cached_at = 0.0
        self.service = None
        self.queried_ids = set()
        self.refresh_for = None
        self._view_fields_installed = False
        self._preparation_reads = set()

    async def warm(self, full: ConsoleDisplayFull) -> None:
        from .chat_conversation_service import ChatConversationService
        from .console_received_dispatch import stock_native_methods
        from tldw_chatbook.UI.Console_Modules.workspace import (
            _APPEARANCE_DISPLAY_READERS,
            _APPEARANCE_DB_READERS,
            CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS,
        )

        if self.owner.domain._conversation_owner is None:
            return
        scope = self.owner.domain.conversation_scope_service()
        service, database = scope.local_service, full.database
        captured, cache_at = self.cache, self.cached_at
        wanted = tuple(sorted(self.queried_ids))
        if (
            type(service) is not ChatConversationService
            or not stock_native_methods(service, _APPEARANCE_DISPLAY_READERS)
            or not stock_native_methods(database, _APPEARANCE_DB_READERS)
            or service.db is not database
            or self.service is not service
            or captured is None
            or not wanted
            or self.refresh_for is captured
            or time.monotonic() - cache_at < CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS
        ):
            return
        if full.stamp is not None and not full.port.display_appearance_current(
            self.owner, full.stamp
        ):
            return
        get_appearances = service.get_conversation_appearances
        self.refresh_for = captured

        def current():
            self.owner.require_current()
            if (
                type(get_appearances) is not MethodType
                or get_appearances.__self__ is not service
                or self.owner.domain.conversation_scope_service().local_service
                is not service
                or service.db is not database
                or not stock_native_methods(service, _APPEARANCE_DISPLAY_READERS)
                or not stock_native_methods(database, _APPEARANCE_DB_READERS)
            ):
                raise ConsoleDisplayPhaseRefused(
                    "Console appearance display owner changed."
                )
            if (
                self.service is not service
                or self.cache is not captured
                or self.cached_at != cache_at
                or self.refresh_for is not captured
            ):
                raise RuntimeError("Console appearance cache changed during read.")

        try:
            shared = full.capture(
                "appearance",
                database,
                lambda: get_appearances(list(wanted)),
                require_current=current,
                reads=self._preparation_reads,
            )
            if shared is None:
                return  # Late work keeps its original finite scheduled owner.
            appearances = await shared
            current()
            await full.publication_ready("appearance")
            current()
            if full.stamp is not None and not full.port.display_appearance_current(
                self.owner, full.stamp
            ):
                return
            merged = dict(captured)
            for conversation_id in wanted:
                merged.pop(conversation_id, None)
            merged.update(
                {
                    str(conversation_id): (
                        str(value.icon or ""),
                        str(value.color or ""),
                    )
                    for conversation_id, value in appearances.items()
                }
            )
            changed = merged != captured
            self.cache = merged if changed else captured
            now = time.monotonic()
            if now - cache_at < CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS:
                self.queried_ids.update(wanted)
            else:
                self.cached_at = now
                self.queried_ids = set(wanted)
            if changed:
                self.owner.domain.presentation.publish_display_appearance(
                    self.owner, full.stamp
                )
        except ConsoleDisplayPhaseRefused:
            raise
        except Exception:
            return  # The original independent next-read retry policy.
        finally:
            if self.refresh_for is captured:
                self.refresh_for = None


# Same full provenance shape as the actual character definitions, qualified on
# the new defining class, receiver and module. Installed records are not grants.
_DISPLAY_DEFINITIONS = tuple(
    (
        owner,
        name,
        function,
        function,
        function.__code__,
        function.__globals__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
        globals().get("__file__"),
        globals().get("__spec__"),
        getattr(globals().get("__spec__"), "origin", None),
    )
    for owner in (
        ConsoleDomainDisplayOwner,
        ConsoleDisplayFull,
        ConsoleAppearanceDisplay,
    )
    for name, function in vars(owner).items()
    if type(function) is FunctionType
)


def _require_definitions(receiver=None):
    from tldw_chatbook.UI.Console_Modules.character_context import (
        _character_reader_current,
    )
    from . import console_display_phases

    if (
        ConsoleDisplaySessionChanged
        is not console_display_phases.ConsoleDisplaySessionChanged
    ):
        raise ConsoleDisplayPhaseRefused("Display origin refusal source changed")

    if not all(
        _character_reader_current(
            record,
            receiver if receiver is not None and type(receiver) is record[0] else None,
        )
        for record in _DISPLAY_DEFINITIONS
    ):
        raise ConsoleDisplayPhaseRefused("Domain display defining body changed")


from .console_domain_services import _definition

_DOMAIN_SOURCE_DEFINITIONS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "_display_stock_eligible",
        "_caller_owns_display_native_scope",
        "preselect_display",
        "require_selected_display_sources",
        "_require_definitions",
    )
)
