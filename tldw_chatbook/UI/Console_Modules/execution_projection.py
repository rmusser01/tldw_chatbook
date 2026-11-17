"""Present one retired process snapshot through the original mounted transcript."""

from __future__ import annotations

import inspect
from types import MethodType

from textual._context import NoActiveAppError
from textual.css.query import QueryError
from textual.dom import NoScreen

from ...Chat import console_control_dispatch as _controls
from ...Chat import console_domain_services as _definitions
from ...Chat import console_execution_process as _process
from ...Chat.console_control_dispatch import ConsoleProjection
from ...Chat.console_execution_process import ConsoleExecutionClient
from ...Widgets.Console import console_transcript as _transcript_module
from ...Widgets.Console.console_transcript import ConsoleTranscript
from ..Screens import chat_screen as _screen_module
from ..Screens.chat_screen import ChatScreen
from . import message as _message_module
from .message import ConsoleMessageController
from . import send_acknowledgement as _ack_module
from . import execution_sessions as _session_presentation

_definition = _definitions._definition
_definition_current = _definitions._definition_current
_TORN_DOWN = _definition(_screen_module._console_screen_is_torn_down)
_ACK_OWNER = _definition(_session_presentation._require_ack)
_ACK_PROJECT = tuple(
    (name, _definition(getattr(_ack_module.ConsoleSendAcknowledgement, name)))
    for name in ("project", "_with_stand_in")
)
_TYPES = (
    (_screen_module, "ChatScreen", ChatScreen),
    (_process, "ConsoleExecutionClient", ConsoleExecutionClient),
    (_controls, "ConsoleProjection", ConsoleProjection),
    (_transcript_module, "ConsoleTranscript", ConsoleTranscript),
    (_message_module, "ConsoleMessageController", ConsoleMessageController),
)
_SCREEN_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in ("query_one", "_show_model_thinking")
)
_TRANSCRIPT_METHODS = tuple(
    (name, _definition(getattr(ConsoleTranscript, name)))
    for name in (
        "set_model_thinking_visible",
        "set_messages",
        "sync_jump_indicator",
    )
)
_PROPERTIES = tuple(
    (
        owner,
        name,
        descriptor,
        tuple(
            _definition(function)
            for function in (descriptor.fget, descriptor.fset)
            if function is not None
        ),
    )
    for owner, names in (
        (
            ChatScreen,
            ("app", "is_current", "is_mounted", "_pending_console_swipe_selection"),
        ),
        (ConsoleTranscript, ("screen", "is_mounted")),
    )
    for name in names
    for descriptor in (inspect.getattr_static(owner, name),)
)


def _require_methods(receiver: object, records: tuple) -> None:
    for name, record in records:
        method = getattr(receiver, name, None)
        if (
            type(method) is not MethodType
            or method.__self__ is not receiver
            or method.__func__ is not record[0]
            or inspect.getattr_static(type(receiver), name) is not record[0]
            or not _definition_current(record)
        ):
            raise RuntimeError("Console projection presentation definition changed")


def _require_sources() -> None:
    if (
        _definitions._definition_current is not _definition_current
        or any(
            getattr(module, name) is not original for module, name, original in _TYPES
        )
        or _screen_module._console_screen_is_torn_down is not _TORN_DOWN[0]
        or not _definition_current(_TORN_DOWN)
        or any(
            inspect.getattr_static(owner, name) is not descriptor
            or any(not _definition_current(record) for record in records)
            for owner, name, descriptor, records in _PROPERTIES
        )
    ):
        raise RuntimeError("Console projection presentation source changed")


def apply_console_process_projection(
    screen: ChatScreen,
    client: ConsoleExecutionClient,
    projection: ConsoleProjection,
    *,
    visit: int,
) -> bool:
    """Apply transcript values for the client's exact current visible visit.

    Args:
        screen: Original mounted Console screen selected by the parent.
        client: Exact process owner already bound to that screen and visit.
        projection: Snapshot published by the client after positive retirement.
        visit: Parent attachment visit captured for this presentation request.

    Returns:
        Whether the synchronous original presentation calls ran. This does not
        acknowledge native retirement, row mounting, paint, or other UI surfaces.

    Raises:
        TypeError: A receiver is outside the fixed original presentation cohort.
        RuntimeError: An original presentation definition or receiver changed.
    """
    if (
        type(screen) is not ChatScreen
        or type(client) is not ConsoleExecutionClient
        or type(projection) is not ConsoleProjection
        or type(visit) is not int
    ):
        raise TypeError("Original Console projection presentation types are required")
    _require_sources()
    if (
        client._view is not screen
        or client._visit != visit
        or client._projection is not projection
        or projection.source_revision != client._source_revision
        or client._disposed
        or client._shutdown_task is not None
        or client._channel_error is not None
        or _TORN_DOWN[0](screen)
    ):
        return False
    try:
        if (
            screen.app is not client._app
            or screen.app_instance is not client._app
            or not screen.is_current
            or not screen.is_mounted
        ):
            return False
    except NoActiveAppError:
        return False
    _require_methods(screen, _SCREEN_METHODS)
    try:
        transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
    except QueryError:
        return False
    if type(transcript) is not ConsoleTranscript:
        raise TypeError("Original Console transcript is required")
    try:
        if (
            not transcript.is_mounted
            or transcript._closing
            or transcript._closed
            or transcript.screen is not screen
        ):
            return False
    except NoScreen:
        return False
    _require_methods(transcript, _TRANSCRIPT_METHODS)
    message = screen._message
    if (
        type(message) is not ConsoleMessageController
        or message._screen is not screen
        or message.app_instance is not client._app
    ):
        raise RuntimeError("Console swipe-selection presentation receiver changed")

    # Preserve the original UI-only handoff immediately before message ingest.
    if screen._pending_console_swipe_selection is not None:
        transcript.pending_selection_id = screen._pending_console_swipe_selection
        screen._pending_console_swipe_selection = None
    transcript.set_model_thinking_visible(screen._show_model_thinking())
    rows = projection.messages
    ack = screen._console_send_ack
    if ack is not None:
        if (
            _session_presentation._require_ack is not _ACK_OWNER[0]
            or not _definition_current(_ACK_OWNER)
        ):
            raise RuntimeError("Console acknowledgement owner source changed")
        _ACK_OWNER[0](screen, ack)
        _require_methods(ack, _ACK_PROJECT)
        rows = ack.project(projection.session_id, rows)
    transcript.set_messages(rows, session_id=projection.session_id)
    transcript.sync_jump_indicator(projection.run_state.status.value)
    return True
