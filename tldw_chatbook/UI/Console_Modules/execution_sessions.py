"""Present retired session values through the original mounted tab surface.

Import lazily after ChatScreen is defined; this module pins its original methods.
"""

from __future__ import annotations

import asyncio
import inspect
from functools import partial
from types import MethodType

from textual._context import NoActiveAppError
from textual.css.query import QueryError
from textual.dom import NoScreen

from ...Chat import console_domain_services as _definitions
from ...Chat import console_execution_process as _process
from ...Chat import console_execution_projection_view as _views
from ...Chat import console_execution_session_surface as _values
from ...Chat.console_execution_process import ConsoleExecutionClient
from ...Chat.console_execution_projection_view import ConsoleProjectionView
from ...Chat.console_execution_session_surface import ConsoleSessionSurfaceProjection
from ...Widgets.Console import console_session_surface as _surface_module
from ...Widgets.Console.console_session_surface import ConsoleSessionSurface
from ..Screens import chat_screen as _screen_module
from ..Screens.chat_screen import ChatScreen
from . import send_acknowledgement as _ack_module
from .send_acknowledgement import ConsoleSendAcknowledgement

_definition = _definitions._definition
_definition_current = _definitions._definition_current
_TORN_DOWN = _definition(_screen_module._console_screen_is_torn_down)
_LOCK_TYPE = asyncio.Lock
_TYPES = (
    (asyncio, "Lock", _LOCK_TYPE),
    (_screen_module, "ChatScreen", ChatScreen),
    (_process, "ConsoleExecutionClient", ConsoleExecutionClient),
    (_views, "ConsoleProjectionView", ConsoleProjectionView),
    (_values, "ConsoleSessionSurfaceProjection", ConsoleSessionSurfaceProjection),
    (_surface_module, "ConsoleSessionSurface", ConsoleSessionSurface),
    (_ack_module, "ConsoleSendAcknowledgement", ConsoleSendAcknowledgement),
)
_SCREEN_METHODS = (("query_one", _definition(ChatScreen.query_one)),)
_VIEW_METHODS = (("sessions", _definition(ConsoleProjectionView.sessions)),)
_SURFACE_METHODS = (("sync_sessions", _definition(ConsoleSessionSurface.sync_sessions)),)
_ACK_METHODS = (
    ("overlay_run_markers", _definition(ConsoleSendAcknowledgement.overlay_run_markers)),
)
_ACK_CALLBACKS = (
    ("_on_release", "_request_resync", _definition(_ack_module._request_resync)),
    ("_message_ids", "_session_message_ids", _definition(_ack_module._session_message_ids)),
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
        (ChatScreen, ("app", "is_current", "is_mounted")),
        (ConsoleSessionSurface, ("app", "screen", "is_mounted", "is_attached")),
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
            raise RuntimeError("Console session presentation definition changed")


def _require_sources() -> None:
    if (
        _definitions._definition_current is not _definition_current
        or any(getattr(module, name) is not original for module, name, original in _TYPES)
        or _screen_module._console_screen_is_torn_down is not _TORN_DOWN[0]
        or not _definition_current(_TORN_DOWN)
        or any(
            inspect.getattr_static(owner, name) is not descriptor
            or any(not _definition_current(record) for record in records)
            for owner, name, descriptor, records in _PROPERTIES
        )
    ):
        raise RuntimeError("Console session presentation source changed")


def _require_ack(screen: ChatScreen, ack: ConsoleSendAcknowledgement) -> None:
    if type(ack) is not ConsoleSendAcknowledgement:
        raise TypeError("Original Console send acknowledgement is required")
    _require_methods(ack, _ACK_METHODS)
    for attribute, name, record in _ACK_CALLBACKS:
        callback = getattr(ack, attribute, None)
        if (
            getattr(_ack_module, name) is not record[0]
            or not _definition_current(record)
            or type(callback) is not partial
            or callback.func is not record[0]
            or len(callback.args) != 1
            or callback.args[0] is not screen
            or callback.keywords
        ):
            raise RuntimeError("Console send acknowledgement screen changed")


def _current(screen, client, publication, visit) -> bool:
    _require_sources()
    if (
        client._view is not screen
        or client._visit != visit
        or getattr(client, "session_surface", None) is not publication
        or publication.view is not client.projection_view
        or publication.view.source_revision != client._source_revision
        or client._disposed
        or client._physically_closed
        or client._shutdown_task is not None
        or client._channel_error is not None
        or not client._intake_open
        or _TORN_DOWN[0](screen)
    ):
        return False
    try:
        if (
            screen.app is not client._app
            or screen.app_instance is not client._app
            or screen.app._exit
            or not screen.is_current
            or not screen.is_mounted
        ):
            return False
    except NoActiveAppError:
        return False
    _require_methods(screen, _SCREEN_METHODS)
    _require_methods(publication.view, _VIEW_METHODS)
    return True


def _surface(screen: ChatScreen) -> ConsoleSessionSurface | None:
    try:
        surface = screen.query_one("#console-session-surface", ConsoleSessionSurface)
    except QueryError:
        return None
    if type(surface) is not ConsoleSessionSurface:
        raise TypeError("Original Console session surface is required")
    try:
        if (
            not surface.is_mounted
            or not surface.is_attached
            or surface._closing
            or surface._closed
            or surface._pruning
            or surface.screen is not screen
            or surface.app is not screen.app
        ):
            return None
    except (NoScreen, NoActiveAppError):
        return None
    _require_methods(surface, _SURFACE_METHODS)
    return surface


async def apply_console_process_sessions(
    screen: ChatScreen,
    client: ConsoleExecutionClient,
    publication: ConsoleSessionSurfaceProjection,
    *,
    visit: int,
) -> bool:
    """Apply one current publication under the original screen's tabs lock.

    Args:
        screen: Actual mounted Console screen selected by the parent.
        client: Original client bound to that screen and visit.
        publication: Exact current session surface record adopted after retirement.
        visit: Original attachment visit captured for this presentation request.

    Returns:
        Whether the original surface call completed for the still-current record.
        This is not native retirement, paint or complete screen reconciliation.

    Raises:
        TypeError: A receiver is outside the fixed original presentation cohort.
        RuntimeError: An original presentation definition or receiver changed.
        asyncio.CancelledError: Cancellation, after the original counter cleanup.
    """
    if (
        type(screen) is not ChatScreen
        or type(client) is not ConsoleExecutionClient
        or type(publication) is not ConsoleSessionSurfaceProjection
        or type(visit) is not int
    ):
        raise TypeError("Original Console session presentation types are required")
    if not _current(screen, client, publication, visit):
        return False
    lock = screen._console_session_tabs_sync_lock
    if type(lock) is not _LOCK_TYPE:
        raise TypeError("Original Console session tabs lock is required")
    screen._console_session_tabs_sync_calls += 1
    try:
        async with lock:
            if (
                screen._console_session_tabs_sync_lock is not lock
                or not _current(screen, client, publication, visit)
            ):
                return False
            surface = _surface(screen)
            if surface is None:
                return False
            ack = getattr(screen, "_console_send_ack", None)
            markers = (
                None if publication.run_markers is None
                else dict(publication.run_markers)
            )
            if ack is not None:
                _require_ack(screen, ack)
                markers = ack.overlay_run_markers(markers)
            try:
                await surface.sync_sessions(
                    sessions=publication.view.sessions(),
                    active_session_id=publication.view.active_session_id,
                    streaming_session_id=publication.streaming_session_id,
                    run_markers=markers,
                    queue_counts=(
                        None if publication.queue_counts is None
                        else dict(publication.queue_counts)
                    ),
                )
            except Exception:
                if not _current(screen, client, publication, visit):
                    return False
                raise
            if (
                not _current(screen, client, publication, visit)
                or screen._console_session_tabs_sync_lock is not lock
                or getattr(screen, "_console_send_ack", None) is not ack
            ):
                return False
            if ack is not None:
                _require_ack(screen, ack)
            return _surface(screen) is surface
    finally:
        screen._console_session_tabs_sync_calls -= 1
