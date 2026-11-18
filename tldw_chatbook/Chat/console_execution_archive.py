"""The real parent's archive reservation around one original received turn."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import FunctionType

from . import conversation_archive_actions as archive
from . import console_execution_protocol as protocol
from .console_domain_services import _definition, _definition_current
from .console_execution_protocol import ConsoleProtocolError


@dataclass(eq=False, slots=True)
class _ArchiveSend:
    client: object
    app: object
    receipt: object
    conversation_id: str
    reservations: dict
    generation: int | None = None
    ordinal: int = 0
    terminal: bool = False
    acknowledgements: list = field(default_factory=list)


def _require_sources() -> None:
    if _ArchiveSend is not _RECORD_TYPE or not _definition_current(_RECORD_INIT):
        raise RuntimeError("Console parent archive record source changed")
    if (
        protocol.ARCHIVE_SEND_REFUSALS is not _REFUSAL_SOURCE
        or tuple(_REFUSAL_SOURCE.items()) != _REFUSAL_ITEMS
    ):
        raise RuntimeError("Console archive refusal mapping changed")
    for name, definition in _SOURCES:
        if getattr(archive, name) is not definition[0] or not _definition_current(
            definition
        ):
            raise RuntimeError("Original parent archive source changed")
    for name, definition in _DEFINITIONS:
        if globals().get(name) is not definition[0] or not _definition_current(
            definition
        ):
            raise RuntimeError("Console parent archive checkpoint source changed")


def _require_parent(client, app) -> None:
    from .console_execution_process import ConsoleExecutionClient

    _require_sources()
    if (
        type(client) is not ConsoleExecutionClient
        or client._app is not app
        or app is None
        or getattr(app, "console_execution_client", None) is not client
        or getattr(app, "console_runtime", None) is not None
        or client._loop is not asyncio.get_running_loop()
        or client._disposed
        or client._channel_error is not None
    ):
        raise ConsoleProtocolError("archive_parent_owner_changed")


def reserve_archive_send(client, receipt) -> None:
    """Reserve the actual published saved conversation before transmission."""
    from .console_execution_projection_view import ConsoleProjectionView

    _require_sources()
    view = client.projection_view
    if (
        type(view) is not ConsoleProjectionView
        or view.source_revision != receipt.identity.source_revision
    ):
        raise ConsoleProtocolError("archive_send_publication_missing")
    row = view.session(receipt.identity.session_id)
    if row.incarnation_id != receipt.identity.incarnation_id:
        raise ConsoleProtocolError("archive_send_session_changed")
    conversation_id = row.persisted_conversation_id
    if not conversation_id:
        return
    app = client._app
    _require_parent(client, app)
    reservations = getattr(app, "_conversation_send_inflight", None)
    if reservations is None:
        reservations = app._conversation_send_inflight = {}
    if type(reservations) is not dict:
        raise ConsoleProtocolError("archive_send_reservations_changed")
    count = reservations.get(conversation_id, 0)
    if (
        type(count) is not int
        or count < 0
        or receipt.identity.request_id in client._archive_sends
    ):
        raise ConsoleProtocolError("archive_send_reservation_changed")
    record = _ArchiveSend(client, app, receipt, conversation_id, reservations)
    reservations[conversation_id] = count + 1
    client._archive_sends[receipt.identity.request_id] = record


def _record(client, receipt):
    record = client._archive_sends.get(receipt.identity.request_id)
    if (
        type(record) is not _ArchiveSend
        or record.client is not client
        or record.receipt is not receipt
        or getattr(record.app, "_conversation_send_inflight", None)
        is not record.reservations
        or type(record.reservations.get(record.conversation_id)) is not int
        or record.reservations[record.conversation_id] <= 0
    ):
        raise ConsoleProtocolError("archive_send_reservation_changed")
    return record


def receive_archive_checkpoint(client, receipt, message) -> None:
    """Run only the two original parent operations on the channel owner loop."""
    record = _record(client, receipt)
    _require_parent(client, record.app)
    payload = message.payload
    if (
        receipt.operation != "ReceiveTurn"
        or not receipt.received.done()
        or receipt.received.cancelled()
        or receipt.received.exception() is not None
        or client._requests.get(receipt.identity.request_id) is not receipt
        or receipt.identity.source_revision != client._source_revision
        or payload["turn_id"] != receipt.turn_id
        or payload["conversation_id"] != record.conversation_id
        or payload["ordinal"] != record.ordinal + 1
        or record.terminal
        or receipt.retired.done()
    ):
        raise ConsoleProtocolError("archive_send_checkpoint_changed")
    if payload["stage"] == "prepare":
        refusal = archive.archive_send_inflight_refusal(
            record.app, record.conversation_id
        )
        generation = getattr(record.app, "_conversation_archive_generation", 0)
        if type(generation) is not int or generation < 0:
            raise ConsoleProtocolError("archive_send_generation_changed")
        record.generation = generation
    else:
        if record.generation is None:
            raise ConsoleProtocolError("archive_send_prepare_missing")
        states = (
            {}
            if payload["archived"] is None
            else {record.conversation_id: payload["archived"]}
        )
        refusal = archive.publish_archive_send_state(
            record.app,
            record.conversation_id,
            record.generation,
            states,
        )
    refusal_code = next(
        (code for code, copy in _REFUSAL_ITEMS if copy == refusal), None
    )
    if refusal is not None and refusal_code is None:
        raise ConsoleProtocolError("archive_send_refusal_changed")
    record.ordinal = payload["ordinal"]
    acknowledgement = client.submit(
        "ArchiveSendCheckpointAck",
        {
            "received_request_id": receipt.identity.request_id,
            "turn_id": receipt.turn_id,
            "ordinal": record.ordinal,
            "stage": payload["stage"],
            "conversation_id": record.conversation_id,
            "generation": record.generation,
            "refusal": refusal_code,
        },
        session_id=receipt.identity.session_id,
        incarnation_id=receipt.identity.incarnation_id,
    )
    record.acknowledgements.append(acknowledgement)
    task = asyncio.create_task(_join_acknowledgement(record, acknowledgement))
    client._archive_tasks.add(task)
    task.add_done_callback(lambda done: _acknowledgement_finished(client, done))


async def _join_acknowledgement(record, acknowledgement) -> None:
    from .console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure

    try:
        applied = await asyncio.shield(acknowledgement.applied)
        if applied.get("applied") is not True:
            raise ConsoleChannelLost("Console archive checkpoint was not acknowledged")
    finally:
        admission_refused = False
        try:
            await asyncio.shield(acknowledgement.received)
        except ConsoleRemoteFailure as error:
            admission_refused = error.stage == "admission"
            if admission_refused:
                acknowledgement.applied.exception()
        if not admission_refused:
            if await asyncio.shield(acknowledgement.retired) is not True:
                raise ConsoleChannelLost(
                    "Console archive checkpoint retirement is unproven"
                )
    settle_archive_send(record.client, record.receipt)


def _acknowledgement_finished(client, task) -> None:
    from .console_execution_process import ConsoleChannelLost

    client._archive_tasks.discard(task)
    if task.cancelled():
        client._fail(
            ConsoleChannelLost("Console archive checkpoint observer cancelled")
        )
    elif task.exception() is not None:
        client._fail(task.exception())


def settle_archive_send(
    client, receipt, *, terminal=False, admission_refused=False
) -> None:
    """Release only this exact reservation after its proven custody outcome."""
    if receipt.identity.request_id not in client._archive_sends:
        return
    _require_sources()
    record = _record(client, receipt)
    _require_parent(client, record.app)
    if terminal:
        record.terminal = True
    if admission_refused:
        from .console_execution_process import ConsoleRemoteFailure

        error = receipt.received.exception()
        if (
            type(error) is not ConsoleRemoteFailure
            or error.stage != "admission"
            or error.code != "command_refused"
            or receipt.applied.exception() is not error
            or record.ordinal
            or record.terminal
            or receipt.potentially_committed
        ):
            raise ConsoleProtocolError("archive_send_refusal_unproven")
    elif (
        not record.terminal
        or not receipt.retired.done()
        or receipt.retired.cancelled()
        or receipt.retired.exception() is not None
        or receipt.retired.result() is not True
        or any(
            not ack.retired.done()
            or ack.retired.cancelled()
            or ack.retired.exception() is not None
            or ack.retired.result() is not True
            or not ack.applied.done()
            or ack.applied.cancelled()
            or ack.applied.exception() is not None
            or ack.applied.result().get("applied") is not True
            for ack in record.acknowledgements
        )
    ):
        return
    remaining = record.reservations[record.conversation_id] - 1
    if remaining:
        record.reservations[record.conversation_id] = remaining
    else:
        record.reservations.pop(record.conversation_id)
    client._archive_sends.pop(receipt.identity.request_id)


_REFUSAL_SOURCE = protocol.ARCHIVE_SEND_REFUSALS
_REFUSAL_ITEMS = tuple(_REFUSAL_SOURCE.items())
_RECORD_TYPE = _ArchiveSend
_RECORD_INIT = _definition(_ArchiveSend.__init__)
_SOURCES = tuple(
    (name, _definition(getattr(archive, name)))
    for name in ("archive_send_inflight_refusal", "publish_archive_send_state")
)
_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in tuple(globals().items())
    if type(value) is FunctionType and value.__module__ == __name__
)
