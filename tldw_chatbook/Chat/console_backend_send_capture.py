"""One physically owned configuration read for an original received press."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from inspect import getattr_static

from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import config_participants
from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current,
)

from . import console_backend_send_selection as selection_source
from . import console_execution_worker as wire_source
from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle
from .console_chat_store import ConsoleChatStore
from .console_control_dispatch import ReceiveTurn, require_control_source
from .console_domain_services import _definition
from .console_execution_protocol import _BACKEND_SEND_SELECTION, _detach
from .console_execution_retirement import _ReadResources
from .console_preparation_reads import preparation_read_for, run_preparation_read


@dataclass(eq=False, slots=True)
class _CapturedSendSelection:
    controls: object
    command: object
    source: object
    session: object
    inputs: object
    config_identity: tuple
    association: object
    lifecycle: object
    checkpoint: object
    attachment: tuple
    task: object
    reads: set = field(default_factory=set)
    observations: list = field(default_factory=list)
    effect: object = None
    selection: object = None
    registration: object = None


def _require_readers(controls):
    backend = controls._domain.presentation.backend
    if not definitions_current(_SOURCES) or any(
        type(receiver) is not expected
        or any(
            getattr_static(receiver, name, None) is not getattr(expected, name)
            for name in names
        )
        for receiver, expected, names in (
            (
                backend.store,
                ConsoleChatStore,
                ("session_input_snapshot", "session_inputs_are_current"),
            ),
            (
                backend.presentation.view_lifecycle,
                ConsoleBackendViewLifecycle,
                ("capture_display_checkpoint", "display_checkpoint_current"),
            ),
        )
    ):
        raise RuntimeError("Console Send configuration reader changed")


def _current(capture):
    controls, command = capture.controls, capture.command
    backend = controls._domain.presentation.backend
    _require_readers(controls)
    controls._require_graph_identity()
    backend.require_sources()
    backend.domain.require_current(capture.source.sources)
    backend.presentation.require_association(capture.association)
    if (
        type(capture) is not _CapturedSendSelection
        or capture.registration != (capture, controls, command, capture.source)
        or type(command) is not ReceiveTurn
        or not command.backend_selection
        or command.selection is not None
        or require_control_source(backend.runtime) is not capture.source
        or capture.source.identity is not command.identity
        or controls._settings_command_tasks.get(capture.task) is not capture.source
        or backend.adapter._commands.get(command.identity.request_id) is not command
        or backend.store._sessions.get(command.identity.session_id)
        is not capture.session
        or capture.session.incarnation_id != command.identity.incarnation_id
        or backend.store.active_session_id != capture.session.id
        or controls.source_revision != command.identity.source_revision
        or not controls._intake_open
        or not controls._effects_open
        or config.current_config_identity() != capture.config_identity
        or type(capture.lifecycle) is not ConsoleBackendViewLifecycle
        or backend.presentation.view_lifecycle is not capture.lifecycle
        or backend.presentation.parent_attachment != capture.attachment
        or not capture.lifecycle.display_checkpoint_current(capture.checkpoint)
        or not backend.store.session_inputs_are_current(capture.inputs)
    ):
        raise RuntimeError("Console Send configuration capture changed")


async def capture_backend_send_selection(controls, command):
    """Retain the checked config read without requesting a display projection."""
    controls._require_graph()
    _require_readers(controls)
    backend = controls._domain.presentation.backend
    source = require_control_source(backend.runtime)
    wire = command.wire_inputs
    if (
        type(command) is not ReceiveTurn
        or not command.backend_selection
        or command.selection is not None
        or wire is None
        or backend.adapter._receipts[command.identity.request_id][0] is not wire
    ):
        raise RuntimeError("Console Send configuration command changed")
    request = wire.payload["selection"]
    _BACKEND_SEND_SELECTION(request, "selection")
    lifecycle = backend.presentation.view_lifecycle
    with backend.store._preparation_lock:
        session = backend.store._sessions[command.identity.session_id]
        capture = _CapturedSendSelection(
            controls,
            command,
            source,
            session,
            backend.store.session_input_snapshot(session.id),
            config.current_config_identity(),
            backend.presentation.association,
            lifecycle,
            lifecycle.capture_display_checkpoint(),
            (request["attachment_id"], request["visit"]),
            asyncio.current_task(),
        )
    capture.registration = (capture, controls, command, source)
    _current(capture)
    capture.effect = controls._begin_effect("send_selection", capture, source)

    def read():
        handoff = preparation_read_for(capture)
        if (
            handoff is None
            or handoff.creator is not capture
            or handoff.task is not capture.task
        ):
            raise RuntimeError("Console Send configuration handoff changed")
        observation = _ReadResources(None, handoff)
        capture.observations.append(observation)
        with observation.observe():
            _current(capture)
            with config_participants.operation(config) as active:
                before = config_participants.checked_config_identity(config, active)
                loaded = copy.deepcopy(config.load_settings())
                _current(capture)
                after = config_participants.checked_config_identity(config, active)
                if before != capture.config_identity or after != before:
                    raise RuntimeError("Console Send configuration changed during read")
            _current(capture)
            return loaded

    try:
        loaded = await run_preparation_read(
            read,
            creator=capture,
            session_id=session.id,
            reads=capture.reads,
            observers=(backend.runtime._preparation_reads,),
            require_current=lambda: _current(capture),
        )
        _current(capture)
        if not capture.observations or not all(
            item.confirmed() for item in capture.observations
        ):
            raise RuntimeError("Console Send configuration retirement is unproven")
        backend._require_owner_loop()
        with backend.store._preparation_lock:
            _current(capture)
            values = selection_source.capture_send_selection(backend, session, loaded)
            values["rag_defaults"] = {"source_types": list(request["rag_source_types"])}
            capture.selection = wire_source._selection(
                _detach(values, "selection", 0, set())
            )
            _current(capture)
    finally:
        controls._finish_effect(
            capture.effect,
            retired=all(item.confirmed() for item in capture.observations),
        )
    return capture


def require_backend_send_selection(capture, controls, command):
    """Consume only this command's positively retired capture before its claim."""
    if (
        type(capture) is not _CapturedSendSelection
        or capture.controls is not controls
        or capture.command is not command
    ):
        raise RuntimeError("Console Send configuration receipt changed")
    _current(capture)
    effect = capture.effect
    if (
        effect not in capture.source.effects
        or effect.owner is not capture
        or effect.kind != "send_selection"
        or not effect.completed.done()
        or effect.completed.result() is not True
        or not effect.physically_retired
        or capture.selection is None
        or not capture.observations
        or not all(item.confirmed() for item in capture.observations)
    ):
        raise RuntimeError("Console Send configuration capture did not retire")
    return capture.selection


def publish_backend_send_claim(controls, command, receipt):
    """Publish the existing Received checkpoint before the new task can run."""
    service = controls._domain.presentation.backend._worker_service
    if service is None:
        return  # In-process original controls have no wire observer.
    if (
        type(service) is not wire_source._BackendService
        or getattr_static(service, "publish_received_claim", None) is not _CLAIM[0]
        or not definitions_current((_CLAIM_RECORD,))
    ):
        raise RuntimeError("Console received checkpoint producer changed")
    service.publish_received_claim(command, receipt)


_SOURCES = capture_definitions(
    (
        *(
            (globals(), name, globals()[name])
            for name in (
                "_current",
                "_require_readers",
                "capture_backend_send_selection",
                "require_backend_send_selection",
                "publish_backend_send_claim",
            )
        ),
        *(
            (owner, name, getattr(owner, name))
            for owner, names in (
                (config, ("load_settings", "current_config_identity")),
                (config_participants, ("operation", "checked_config_identity")),
                (selection_source, ("capture_send_selection",)),
                (wire_source, ("_selection",)),
                (
                    ConsoleChatStore,
                    ("session_input_snapshot", "session_inputs_are_current"),
                ),
                (
                    ConsoleBackendViewLifecycle,
                    ("capture_display_checkpoint", "display_checkpoint_current"),
                ),
            )
            for name in names
        ),
    )
)
_CAPTURE = _definition(capture_backend_send_selection)
_REQUIRE = _definition(require_backend_send_selection)
_PUBLISH = _definition(publish_backend_send_claim)
_CLAIM = _definition(wire_source._BackendService.publish_received_claim)
(_CLAIM_RECORD,) = capture_definitions(
    (
        (
            wire_source._BackendService,
            "publish_received_claim",
            _CLAIM[0],
        ),
    )
)
