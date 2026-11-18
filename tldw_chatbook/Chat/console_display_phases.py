"""Fresh physical phases for one sealed Console display observation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.character_context import (
        _CharacterRecentRead,
        _CharacterRefreshReaders,
    )
    from .console_chat_store import ConsoleChatStore
    from .console_domain_display import ConsoleDisplayFull
    from .console_preparation_reads import ConsolePreparationRead
    from .console_runtime import ConsoleRuntime

    _ReadResult = TypeVar("_ReadResult")

from .console_native_commit import (
    _ConsoleNativeBinding,
    _ConsolePlanNativeRetirement,
    _invoke_plan_native,
)
from .console_preparation_reads import run_preparation_read


class ConsoleDisplayPhaseRefused(RuntimeError):
    """No section of this logical FULL may publish after uncertain ownership."""


class ConsoleDisplaySessionChanged(ConsoleDisplayPhaseRefused):
    """The registered original session remains valid but is no longer active."""

    def __init__(self, full: object) -> None:
        super().__init__("Logical display active session changed")
        self.full = full


class ConsoleDisplayVisibilityChanged(ConsoleDisplayPhaseRefused):
    """This exact display lost its authenticated backend visibility checkpoint."""

    def __init__(self, full: object) -> None:
        super().__init__("Backend display visibility changed")
        self.full = full


class ConsoleCharacterDisplayPhases:
    """Preserve the original pair/groups/pair algorithm across retired phases."""

    def __init__(
        self, full: ConsoleDisplayFull, readers: _CharacterRefreshReaders
    ) -> None:
        self.full, self.readers = full, readers
        self.ordinal = 0

    async def _native(self, read):
        return await self.full.native_phase(read, self.readers.require_current)

    async def _checkpoint(self, kind, *, settled=False):
        self.ordinal += 1
        return await self.full.character_checkpoint(
            self, kind, self.ordinal, settled=settled
        )

    async def _pair(self, *, settled):
        from tldw_chatbook.UI.Console_Modules import character_context as source

        readers = self.readers

        def metadata():
            if not source._stock_character_scope_pair(readers.controller):
                return None
            readers.require_current()
            value = readers.controller._read_database_scope_metadata(readers.database)
            readers.require_current()
            return (
                value
                if source._stock_character_scope_pair(readers.controller)
                else None
            )

        before = await self._native(metadata)
        if before is None:
            return None
        if not await self._checkpoint("pair-ambient", settled=settled):
            return None
        after = await self._native(metadata)
        return (before, after) if after is not None else None

    async def _scope(self, *, settled=None):
        from tldw_chatbook.UI.Console_Modules import character_context as source

        readers = self.readers
        for _attempt in range(source._SCOPE_CAPTURE_ATTEMPTS):
            readers.require_current()
            try:
                pair = await self._pair(settled=settled is not None)
            except (
                source._ConsoleCharacterReaderChanged,
                source._ConsoleCharacterAmbientError,
                ConsoleDisplayPhaseRefused,
            ):
                raise
            except Exception:
                if not await self._checkpoint("scope-error"):
                    raise source._ConsoleCharacterScopeChanged(
                        "Character display owner changed"
                    ) from None
                raise source._ConsoleCharacterScopeReadError(
                    readers.database, readers.current, readers.conversation
                ) from None
            readers.require_current()
            if pair is None:
                if not await self._checkpoint("scope-owner"):
                    raise source._ConsoleCharacterScopeChanged(
                        "Character display owner changed"
                    )
                continue
            if pair[0] != pair[1]:
                continue
            authority, revision = pair[1]
            snapshot = source._ConsoleCharacterScopeSnapshot(
                readers.database,
                source.ConsoleCharacterScopeFingerprint(
                    database_identity=id(readers.database),
                    data_authority_id=authority,
                    data_revision=revision,
                    current_character_id=None
                    if readers.current is None
                    else readers.current[0],
                    current_character_label=""
                    if readers.current is None
                    else readers.current[1],
                    open_conversation_id=readers.conversation,
                ),
            )
            if settled is not None and snapshot.fingerprint == settled:
                return snapshot
            if not await self._checkpoint("scope-owner"):
                raise source._ConsoleCharacterScopeChanged(
                    "Character display owner changed"
                )
            return snapshot
        raise source._ConsoleCharacterScopeChanged(
            "Character context scope did not settle"
        )

    async def read(self) -> _CharacterRecentRead:
        from tldw_chatbook.UI.Console_Modules import character_context as source

        readers = self.readers
        if readers.unchanged_from is None:
            readers.require_current()
        else:
            try:
                readers.require_current()
            except source._ConsoleCharacterReaderChanged as error:
                raise ConsoleDisplayPhaseRefused(
                    "Selected character reader changed"
                ) from error
        snapshot = await self._scope(settled=readers.unchanged_from)
        if (
            readers.unchanged_from is not None
            and snapshot.fingerprint == readers.unchanged_from
        ):
            readers.require_current()
            return source._CharacterRecentRead(
                snapshot, (), (), False, True, unchanged=True
            )
        groups, details = (), ()
        load_failed = False
        try:
            groups, details = await self._native(
                lambda: readers.recent(
                    readers.database,
                    snapshot.fingerprint,
                    _reader_check=readers.require_current,
                )
            )
        except (source._ConsoleCharacterReaderChanged, ConsoleDisplayPhaseRefused):
            raise
        except Exception:
            load_failed = True
        readers.require_current()
        if not await self._checkpoint("post-groups-owner"):
            return source._CharacterRecentRead(
                snapshot, groups, details, load_failed, False
            )
        try:
            readers.require_current()
            final = await self._scope()
        except (
            source._ConsoleCharacterScopeChanged,
            source._ConsoleCharacterScopeReadError,
        ):
            scope_current = False
        else:
            scope_current = (
                final.database is snapshot.database
                and final.fingerprint == snapshot.fingerprint
            )
        readers.require_current()
        return source._CharacterRecentRead(
            snapshot, groups, details, load_failed, scope_current
        )


@dataclass(slots=True)
class _DisplayPhaseObservation:
    retirement: _ConsolePlanNativeRetirement
    native_attempted: bool = False
    joined: bool = False


class ConsoleDisplayPhysicalOwner:
    """Observe original physical owners; never borrow a caller's lifetime."""

    def __init__(
        self,
        *,
        runtime: ConsoleRuntime,
        store: ConsoleChatStore,
        database: CharactersRAGDB,
        creator: object,
        observers: list[set[ConsolePreparationRead]],
        require_current: Callable[[], None],
    ) -> None:
        self.runtime, self.store, self.database = runtime, store, database
        self.creator, self.observers = creator, observers
        self._require_current = require_current
        self.retirements = []
        self.refusal = None
        self.phase_pending = False
        self.phase_issuance_closed = False

    @property
    def physically_retired(self) -> bool:
        # Only resource evidence. require_current/refusal still governs whether
        # a new phase or publication is eligible; retirement never grants that.
        return self.native_retirement_confirmed()

    async def native_phase(
        self,
        read: Callable[[], _ReadResult],
        validate: Callable[[], None] | None = None,
    ) -> _ReadResult:
        self.require_phase_task()
        self._require_current()
        if (
            self.phase_issuance_closed
            or self.refusal is not None
            or not self.native_retirement_confirmed()
        ):
            raise ConsoleDisplayPhaseRefused("Display physical continuation refused")
        phase = _DisplayPhaseObservation(_ConsolePlanNativeRetirement())
        self.retirements.append(phase)  # BEFORE permit wait/executor exposure.
        self.phase_pending = True
        binding = _ConsoleNativeBinding(
            self.store,
            self,
            self.store.persistence,
            self.database,
            retirement=phase.retirement,
        )

        def invoke():
            from tldw_chatbook.DB.base_db import _OPERATION_OWNERS
            from tldw_chatbook.Backup_Recovery import storage_admission as storage

            self._require_current()
            if (
                id(self.database) in getattr(_OPERATION_OWNERS, "ids", ())
                or getattr(self.database._local, "conn", None) is not None
                or getattr(storage._operation_local, "operation", None) is not None
            ):
                # Worker custody was unknowable on the issuing loop. The FULL
                # is ALREADY admitted: refuse, never run legacy on this worker.
                # No handle opens/closes, and no borrowed resource is claimed.
                raise ConsoleDisplayPhaseRefused(
                    "Fresh phase found original borrowed custody"
                )

            def body():
                try:
                    if validate is not None:
                        validate()
                    result = read()
                    if validate is not None:
                        validate()
                    return True, result
                except Exception as error:
                    return False, error

            phase.native_attempted = True
            return _invoke_plan_native(binding, body)

        try:
            ok, value = await run_preparation_read(
                invoke,
                creator=self.creator,
                session_id=None,
                reads=self.observers[0],
                observers=tuple(self.observers[1:]),
                require_current=self._require_current,
                submission_slot=self.runtime._display_submission_slot,
            )
        except asyncio.CancelledError:
            self.refusal = ConsoleDisplayPhaseRefused(
                "Display canceled after physical join"
            )
            raise
        except (ConsoleDisplaySessionChanged, ConsoleDisplayVisibilityChanged) as error:
            self.refusal = error
            raise
        except BaseException as error:
            self.refusal = ConsoleDisplayPhaseRefused("Display phase refused")
            raise self.refusal from error
        finally:
            # The QUALIFIED ORIGINAL run_preparation_read has joined its exact
            # submitted producer or guaranteed its original no-entry path.
            # It is not task.done()/a global empty observer census.
            phase.joined = True
            self.phase_pending = False
        if not self.native_retirement_confirmed():
            self.refusal = ConsoleDisplayPhaseRefused(
                "Display native retirement is unproven"
            )
            raise self.refusal
        self._require_current()
        if not ok:
            raise value
        return value

    def native_retirement_confirmed(self) -> bool:
        """Exact issued-phase outcomes, independent of success/cancellation."""
        return not self.phase_pending and all(
            phase.joined
            and (
                not phase.native_attempted
                or (phase.retirement.confirmed and phase.retirement.borrowed is None)
            )
            for phase in self.retirements
        )


# Qualify the actual new defining owners, including the retirement property.
from .console_domain_services import _definition, _definition_current
from types import FunctionType, MethodType
import inspect
import sys

_PHASE_MODULE = (
    globals(),
    globals().get("__file__"),
    globals().get("__spec__"),
    getattr(globals().get("__spec__"), "origin", None),
)
_PHASE_DEFINITIONS = tuple(
    (owner, name, descriptor, _definition(function))
    for owner in (
        ConsoleCharacterDisplayPhases,
        ConsoleDisplayPhysicalOwner,
        ConsoleDisplaySessionChanged,
        ConsoleDisplayVisibilityChanged,
    )
    for name, descriptor in vars(owner).items()
    for function in (
        descriptor.fget if isinstance(descriptor, property) else descriptor,
    )
    if type(function) is FunctionType
)


def require_display_phase_definitions(receiver: object) -> None:
    from . import console_native_commit, console_preparation_reads

    for module, name, record in _PHASE_NATIVE_DEFINITIONS:
        if getattr(module, name, None) is not record[0] or not _definition_current(
            record
        ):
            raise ConsoleDisplayPhaseRefused("Display native defining source changed")
    if (
        _invoke_plan_native is not console_native_commit._invoke_plan_native
        or run_preparation_read is not console_preparation_reads.run_preparation_read
    ):
        raise ConsoleDisplayPhaseRefused("Display native import changed")
    namespace, filename, spec, origin = _PHASE_MODULE
    module = sys.modules.get(namespace.get("__name__"))
    if (
        module is None
        or vars(module) is not namespace
        or namespace.get("__file__") != filename
        or namespace.get("__spec__") is not spec
        or getattr(spec, "origin", None) != origin
    ):
        raise ConsoleDisplayPhaseRefused("Display phase source module changed")
    for owner, name, descriptor, record in _PHASE_DEFINITIONS:
        if (
            namespace.get(owner.__name__) is not owner
            or vars(owner).get(name) is not descriptor
            or not _definition_current(record)
        ):
            raise ConsoleDisplayPhaseRefused("Display phase definition changed")
        if isinstance(receiver, owner):
            if name == "__init__" and type(receiver) is not owner:
                from .console_domain_display import (
                    ConsoleDisplayFull,
                    _require_definitions,
                )

                if type(receiver) is not ConsoleDisplayFull:
                    raise ConsoleDisplayPhaseRefused(
                        "Display phase constructor receiver changed"
                    )
                _require_definitions(receiver)
                continue  # Exact Full constructor is separately defined and qualified.
            if inspect.getattr_static(receiver, name) is not descriptor:
                raise ConsoleDisplayPhaseRefused("Display phase receiver changed")
            if type(descriptor) is FunctionType:
                bound = getattr(receiver, name)
                if (
                    type(bound) is not MethodType
                    or bound.__self__ is not receiver
                    or bound.__func__ is not descriptor
                ):
                    raise ConsoleDisplayPhaseRefused("Display phase binding changed")


from . import console_native_commit as _native_source
from . import console_preparation_reads as _preparation_source

_PHASE_NATIVE_DEFINITIONS = tuple(
    (module, name, _definition(getattr(module, name)))
    for module, name in (
        (_native_source, "_invoke_plan_native"),
        (_preparation_source, "run_preparation_read"),
    )
)
_DOMAIN_SOURCE_DEFINITIONS = (
    (
        "require_display_phase_definitions",
        _definition(require_display_phase_definitions),
    ),
)
