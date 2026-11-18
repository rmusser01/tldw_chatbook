"""Fixed parent archive checkpoints around the original owned archive read."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .console_domain_services import _definition, _definition_current
from .console_execution_protocol import ARCHIVE_SEND_REFUSALS


@dataclass(eq=False, slots=True)
class _ArchiveCheckpoint:
    presentation: object
    record: object
    source: object
    conversation_id: str
    read_owner: object
    ordinal: int = 0
    generation: int | None = None
    refusal: str | None = None
    answer: asyncio.Future | None = None
    registration: tuple | None = None


def _require(checkpoint):
    from .console_backend_presentation import ConsoleBackendPresentation
    from .console_execution_worker import _BackendService
    from .console_control_dispatch import ReceiveTurn, _CommandSource
    from .console_runtime import _ConsoleTurnCustodyRecord
    from .console_control_sessions import _SavedOpenOwner

    if (
        type(checkpoint) is not _CHECKPOINT_TYPE
        or _ArchiveCheckpoint is not _CHECKPOINT_TYPE
        or _ArchiveCheckpoint.__init__ is not _CHECKPOINT_INIT[0]
        or not _definition_current(_CHECKPOINT_INIT)
        or checkpoint.registration
        != (
            checkpoint,
            checkpoint.presentation,
            checkpoint.record,
            checkpoint.source,
            checkpoint.conversation_id,
            checkpoint.read_owner,
        )
        or any(
            globals().get(name) is not saved[0] or not _definition_current(saved)
            for name, saved in _DEFINITIONS
        )
    ):
        raise RuntimeError("Console archive checkpoint source changed")
    presentation, record, source = (
        checkpoint.presentation,
        checkpoint.record,
        checkpoint.source,
    )
    if type(presentation) is not ConsoleBackendPresentation:
        raise RuntimeError("Console archive checkpoint owner changed")
    presentation.require_association(presentation.association)
    backend = presentation.backend
    backend._require_owner_loop()
    controls = backend.controls
    service = backend._worker_service
    entry = controls._turns.get(record.turn_id)
    session = backend.store._sessions.get(record.session_id)
    received = backend.adapter._receipts.get(source.identity.request_id)
    command = backend.adapter._commands.get(source.identity.request_id)
    if (
        type(record) is not _ConsoleTurnCustodyRecord
        or record.archive_checkpoint is not checkpoint
        or backend.runtime._turn_custody.get(record.turn_id) is not record
        or source.controls is not controls
        or type(source) is not _CommandSource
        or source.sources is not backend.domain.sources
        or source.turn_retirement is not controls._turn_retirements.get(record.turn_id)
        or source.turn_retirement is None
        or source.turn_retirement.source is not source
        or type(checkpoint.read_owner) is not _SavedOpenOwner
        or checkpoint.read_owner.controls is not controls
        or checkpoint.read_owner.sources is not source.sources
        or source.identity.source_revision != controls.source_revision
        or entry is None
        or entry[0] is not source.identity
        or entry[1] is not record
        or entry[2] is not record.task
        or record.task is None
        or record.task.done()
        or session is None
        or session.incarnation_id != source.identity.incarnation_id
        or session.persisted_conversation_id != checkpoint.conversation_id
        or record.archive_conversation_id != checkpoint.conversation_id
        or type(service) is not _BackendService
        or service.backend is not backend
        or received is None
        or type(command) is not ReceiveTurn
        or command.identity is not source.identity
        or command.turn_id != record.turn_id
        or received[1].work is not record.task
        or source.identity.request_id not in service._received_requests
    ):
        raise RuntimeError("Console archive received owner changed")
    return backend, received[0]


async def _ask(checkpoint, ordinal, archived):
    backend, message = _require(checkpoint)
    if (
        asyncio.current_task() is not checkpoint.record.task
        or checkpoint.answer is not None
    ):
        raise RuntimeError("Console archive checkpoint task changed")
    checkpoint.ordinal = ordinal
    checkpoint.answer = backend._loop.create_future()
    try:
        backend._worker_service.emit(
            message.identity,
            "archive_checkpoint",
            "ReceiveTurn",
            {
                "turn_id": checkpoint.record.turn_id,
                "ordinal": ordinal,
                "stage": "prepare" if ordinal == 1 else "finish",
                "conversation_id": checkpoint.conversation_id,
                "archived": archived,
            },
        )
        generation, refusal = await asyncio.shield(checkpoint.answer)
        _require(checkpoint)
        if ordinal == 2 and generation != checkpoint.generation:
            raise RuntimeError("Console archive parent generation changed")
        checkpoint.generation = generation
        checkpoint.refusal = refusal
        return None if refusal is None else ARCHIVE_SEND_REFUSALS[refusal]
    finally:
        checkpoint.answer = None


async def prepare_archive_send(presentation, record, *, read_owner):
    from .console_control_dispatch import require_control_source

    source = require_control_source(presentation.backend.runtime)
    if record.archive_checkpoint is not None or not record.archive_conversation_id:
        raise RuntimeError("Console archive preparation changed")
    checkpoint = _ArchiveCheckpoint(
        presentation, record, source, record.archive_conversation_id, read_owner
    )
    checkpoint.registration = (
        checkpoint,
        presentation,
        record,
        source,
        record.archive_conversation_id,
        read_owner,
    )
    record.archive_checkpoint = checkpoint
    try:
        refusal = await _ask(checkpoint, 1, None)
        return checkpoint, refusal
    except BaseException:
        record.archive_checkpoint = None
        raise


async def finish_archive_send(checkpoint, record, states):
    if (
        checkpoint.record is not record
        or checkpoint.ordinal != 1
        or checkpoint.refusal is not None
        or not checkpoint.read_owner.reads
        or not checkpoint.read_owner.retired
    ):
        raise RuntimeError("Console archive finish changed")
    if type(states) is not dict or any(
        key != checkpoint.conversation_id or type(value) is not bool
        for key, value in states.items()
    ):
        raise RuntimeError("Console archive state result changed")
    return await _ask(checkpoint, 2, states.get(checkpoint.conversation_id))


def acknowledge_archive_send(presentation, command) -> bool:
    from .console_control_dispatch import ArchiveSendCheckpointAck

    if type(command) is not ArchiveSendCheckpointAck:
        raise TypeError("Original archive acknowledgement is required")
    record = presentation.backend.runtime._turn_custody.get(command.turn_id)
    checkpoint = None if record is None else record.archive_checkpoint
    _require(checkpoint)
    identity = command.identity
    original = checkpoint.source.identity
    if (
        checkpoint.presentation is not presentation
        or command.received_request_id != original.request_id
        or command.conversation_id != checkpoint.conversation_id
        or command.ordinal != checkpoint.ordinal
        or command.stage != ("prepare" if checkpoint.ordinal == 1 else "finish")
        or any(
            getattr(identity, name) != getattr(original, name)
            for name in (
                "profile_epoch",
                "service_epoch",
                "session_id",
                "incarnation_id",
                "source_revision",
            )
        )
        or checkpoint.answer is None
        or checkpoint.answer.done()
        or checkpoint.ordinal == 2
        and command.generation != checkpoint.generation
        or checkpoint.ordinal == 1
        and command.refusal == "archived"
    ):
        raise RuntimeError("Console archive acknowledgement changed")
    checkpoint.answer.set_result((command.generation, command.refusal))
    return True


_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in tuple(globals().items())
    if name
    in {
        "_require",
        "_ask",
        "prepare_archive_send",
        "finish_archive_send",
        "acknowledge_archive_send",
    }
)
_CHECKPOINT_TYPE = _ArchiveCheckpoint
_CHECKPOINT_INIT = _definition(_ArchiveCheckpoint.__init__)
