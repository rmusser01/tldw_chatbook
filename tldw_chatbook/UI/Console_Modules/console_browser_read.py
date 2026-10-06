"""One source-qualified, finite stock Console conversation-browser read.

The optional callback owns only the native handle it creates. It never caches
authority or changes the original service dispatch for custom receivers.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import (
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
    SimpleNamespace,
)
from typing import Any, Callable

from tldw_chatbook.Chat import chat_conversation_service as _service_module
from tldw_chatbook.DB import ChaChaNotes_DB as _notes_module

_OWNER_CHECK_TIMEOUT_SECONDS = 10.0
_MISSING = object()
_SERVICE_SOURCE = _service_module._CONSOLE_BROWSER_SERVICE_SOURCE
_NOTES_SOURCE = _notes_module._CONSOLE_BROWSER_NOTES_SOURCE
_CORE_SOURCE = _notes_module._CONSOLE_BROWSER_NOTES_CORE_SOURCE
_CLOSE_SOURCE = _notes_module._CONSOLE_BROWSER_NOTES_CLOSE_SOURCE
_CLOSING_SOURCE = _notes_module._CONSOLE_BROWSER_NOTES_CLOSING_SOURCE
_RETIREMENT_SOURCE = _notes_module._CONSOLE_BROWSER_NOTES_RETIREMENT_SOURCE
_SERVICE_TYPE = _SERVICE_SOURCE[5][0][1]
_NOTES_TYPE = _NOTES_SOURCE[5][0][1]
_REGISTRY_TYPE = _RETIREMENT_SOURCE[5][0][1]
_CORE_OPERATION = dict(_NOTES_SOURCE[5])["_core_operation"]
_RETIRE_HANDLE = dict(
    (name, descriptor)
    for owner, name, descriptor in _NOTES_SOURCE[4]
    if owner is _NOTES_TYPE
)["_close_connection_handle"]
_GET_HANDLE = dict(
    (name, descriptor)
    for owner, name, descriptor in _NOTES_SOURCE[4]
    if owner is _NOTES_TYPE
)["get_connection"]
_SERVICE_READER = dict(
    (name, descriptor)
    for owner, name, descriptor in _SERVICE_SOURCE[4]
    if owner is _SERVICE_TYPE
)["list_conversations"]


class _BrowserSourceChanged(RuntimeError):
    """An optional source changed; no rows from that source may be accepted."""


def _function_current(record: tuple) -> bool:
    function, code, defining, defaults, kwdefaults, items, closure, cells, wrapped = (
        record
    )
    try:
        return (
            type(function) is FunctionType
            and function.__code__ is code
            and function.__globals__ is defining
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and len(function.__kwdefaults__ or {}) == len(items)
            and all(
                (function.__kwdefaults__ or {}).get(key) is value
                for key, value in items
            )
            and function.__closure__ is closure
            and all(cell.cell_contents is value for cell, value in cells)
            and vars(function).get("__wrapped__") is wrapped
        )
    except (AttributeError, ValueError):
        return False


def _source_current(source: tuple) -> bool:
    defining, path, spec, origin, descriptors, bindings, records = source
    module = sys.modules.get(defining.get("__name__"))
    return (
        type(module) is ModuleType
        and vars(module) is defining
        and defining.get("__file__") == path
        and defining.get("__spec__") is spec
        and getattr(spec, "origin", None) == origin
        and all(
            inspect.getattr_static(owner, name, _MISSING) is descriptor
            for owner, name, descriptor in descriptors
        )
        and all(defining.get(name, _MISSING) is value for name, value in bindings)
        and all(_function_current(record) for record in records)
    )


def _plain_fields(receiver: object, owner: type, names: tuple[str, ...]) -> bool:
    """Decline a custom lookup/descriptor before inspecting instance fields."""
    expected = object.__getattribute__
    if owner is SimpleNamespace:
        expected = SimpleNamespace.__getattribute__
    return (
        type(receiver) is owner
        and inspect.getattr_static(owner, "__getattribute__") is expected
        and inspect.getattr_static(owner, "__getattr__", _MISSING) is _MISSING
        and all(
            inspect.getattr_static(owner, name, _MISSING) is _MISSING for name in names
        )
        and type(inspect.getattr_static(owner, "__dict__", _MISSING))
        in (GetSetDescriptorType, MemberDescriptorType)
        and type(vars(receiver)) is dict  # noqa: E721 -- no custom mapping dispatch
    )


def _receiver_current(receiver: object, owner: type, source: tuple) -> bool:
    # Class lookup custody is checked before vars(receiver), which itself reads
    # __dict__. A changed custom lookup cannot participate in optional admission.
    if type(receiver) is not owner or not _source_current(source):
        return False
    lookup = next(
        descriptor
        for recorded_owner, name, descriptor in source[4]
        if recorded_owner is owner and name == "__getattribute__"
    )
    if (
        inspect.getattr_static(owner, "__getattribute__") is not lookup
        or inspect.getattr_static(owner, "__getattr__", _MISSING) is not _MISSING
    ):
        return False
    fields = vars(receiver)
    if type(fields) is not dict:  # noqa: E721 -- no custom mapping dispatch
        return False
    return all(
        name not in fields
        for recorded_owner, name, _descriptor in source[4]
        if recorded_owner is owner
    )


def _sources_current() -> bool:
    return (
        _service_module._CONSOLE_BROWSER_SERVICE_SOURCE is _SERVICE_SOURCE
        and _notes_module._CONSOLE_BROWSER_NOTES_SOURCE is _NOTES_SOURCE
        and _notes_module._CONSOLE_BROWSER_NOTES_CORE_SOURCE is _CORE_SOURCE
        and _notes_module._CONSOLE_BROWSER_NOTES_CLOSE_SOURCE is _CLOSE_SOURCE
        and _notes_module._CONSOLE_BROWSER_NOTES_CLOSING_SOURCE is _CLOSING_SOURCE
        and _notes_module._CONSOLE_BROWSER_NOTES_RETIREMENT_SOURCE is _RETIREMENT_SOURCE
        and all(
            _source_current(source)
            for source in (
                _SERVICE_SOURCE,
                _NOTES_SOURCE,
                _CORE_SOURCE,
                _CLOSE_SOURCE,
                _CLOSING_SOURCE,
                _RETIREMENT_SOURCE,
                _HELPER_SOURCE,
            )
        )
    )


@dataclass
class _StockBrowserRead:
    """Strong references that last for exactly one synchronous native callback."""

    controller: Any
    controller_type: type
    controller_source: tuple
    app: Any
    app_type: type
    service: Any
    local_service: Any
    scope_service: Any
    database: Any
    local: Any
    registry: Any
    path: Any
    path_text: str
    reader: MethodType
    loop: asyncio.AbstractEventLoop
    factory_is_current: Callable[[], bool]
    source_changed: bool = False
    cancelled: bool = False
    actor: tuple | None = None
    connection: Any = None
    previous: Any = None
    _results: list[tuple[Any, Exception | None]] = field(default_factory=list)

    def _changed(self) -> None:
        self.source_changed = True
        raise _BrowserSourceChanged("console_browser_source_changed")

    def require_source_current(self) -> None:
        if (
            not self.factory_is_current()
            or not _sources_current()
            or not _receiver_current(self.service, _SERVICE_TYPE, _SERVICE_SOURCE)
            or not _receiver_current(self.database, _NOTES_TYPE, _NOTES_SOURCE)
            or not _plain_fields(self.service, _SERVICE_TYPE, ("db",))
            or not _plain_fields(
                self.database,
                _NOTES_TYPE,
                (
                    "_local",
                    "_connection_quiescence",
                    "db_path",
                    "db_path_str",
                    "is_memory_db",
                ),
            )
            or vars(self.service).get("db") is not self.database
            or vars(self.database).get("_local") is not self.local
            or type(self.local) is not threading.local
            or vars(self.database).get("_connection_quiescence") is not self.registry
            or vars(self.database).get("db_path") != self.path
            or vars(self.database).get("db_path_str") != self.path_text
            or vars(self.database).get("is_memory_db") is not False
            or not _receiver_current(self.registry, _REGISTRY_TYPE, _RETIREMENT_SOURCE)
            or type(self.reader) is not MethodType
            or self.reader.__self__ is not self.service
            or self.reader.__func__ is not _SERVICE_READER
        ):
            self._changed()

    def require_loop_current(self) -> None:
        """Inspect ambient controller/App ownership only on the captured loop."""
        self.require_source_current()
        if (
            asyncio.get_running_loop() is not self.loop
            or self.cancelled
            or not _receiver_current(
                self.controller, self.controller_type, self.controller_source
            )
            or not _plain_fields(
                self.controller, self.controller_type, ("app_instance",)
            )
            or vars(self.controller).get("app_instance") is not self.app
            or not _plain_fields(
                self.app,
                self.app_type,
                (
                    "local_chat_conversation_service",
                    "chat_conversation_scope_service",
                ),
            )
            or vars(self.app).get("local_chat_conversation_service")
            is not self.local_service
            or vars(self.app).get("chat_conversation_scope_service")
            is not self.scope_service
        ):
            self._changed()
        # The scope-service adapter route is optional only when its local field
        # is an ordinary captured instance field. Its original custom route stays
        # outside this finite callback.
        if self.service is not self.local_service:
            if (
                not _plain_fields(
                    self.scope_service, type(self.scope_service), ("local_service",)
                )
                or vars(self.scope_service).get("local_service") is not self.service
            ):
                self._changed()
        self.require_source_current()

    def _owner_on_loop(self) -> None:
        check: Future[None] = Future()

        def validate() -> None:
            if not check.set_running_or_notify_cancel():
                return
            try:
                self.require_loop_current()
            except BaseException as error:
                check.set_exception(error)
            else:
                check.set_result(None)

        try:
            self.loop.call_soon_threadsafe(validate)
            check.result(timeout=_OWNER_CHECK_TIMEOUT_SECONDS)
        except BaseException:
            self.source_changed = True
            raise
        finally:
            # A timed-out queued callback must never read GUI ownership later.
            check.cancel()

    def _require_worker_current(self) -> None:
        self.require_source_current()
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if self.actor != (os.getpid(), threading.current_thread(), task):
            self._changed()
        if self.connection is not None and (
            getattr(self.local, "conn", None) is not self.connection
            or not self.registry.is_registered(self.connection)
        ):
            self._changed()
        self._owner_on_loop()
        self.require_source_current()

    def _retire_created_handle(self) -> None:
        connection = self.connection
        if connection is None or connection is self.previous:
            return
        # Service, App or cache drift must not prevent A's captured cleanup.
        # Retirement still requires its original defining boundary and actor.
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if (
            self.actor != (os.getpid(), threading.current_thread(), task)
            or _notes_module._CONSOLE_BROWSER_NOTES_CLOSE_SOURCE is not _CLOSE_SOURCE
            or _notes_module._CONSOLE_BROWSER_NOTES_CLOSING_SOURCE
            is not _CLOSING_SOURCE
            or not _source_current(_CLOSE_SOURCE)
            or not _source_current(_CLOSING_SOURCE)
            or not _source_current(_RETIREMENT_SOURCE)
            or type(self.database) is not _NOTES_TYPE
            or not _plain_fields(
                self.database,
                _NOTES_TYPE,
                (
                    "is_memory_db",
                    "_connection_quiescence",
                    "db_path",
                    "db_path_str",
                ),
            )
            or vars(self.database).get("is_memory_db") is not False
            or vars(self.database).get("_connection_quiescence") is not self.registry
            or vars(self.database).get("db_path") != self.path
            or vars(self.database).get("db_path_str") != self.path_text
            or not _receiver_current(self.registry, _REGISTRY_TYPE, _RETIREMENT_SOURCE)
        ):
            raise RuntimeError("console_browser_retirement_source_changed")
        _RETIRE_HANDLE(
            self.database, connection, self.local, self.registry, strict=True
        )

    def _read_pair(
        self, kwargs_pair: tuple[dict, ...]
    ) -> list[tuple[Any, Exception | None]]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        self.actor = (os.getpid(), threading.current_thread(), task)
        self._require_worker_current()
        self.previous = getattr(self.local, "conn", None)
        if self.previous is not None and not self.registry.is_registered(self.previous):
            self._changed()
        operation_error = None
        body_error = None
        try:
            with _CORE_OPERATION(self.database):
                self._require_worker_current()  # Admission can invoke callbacks.
                connection = _GET_HANDLE(self.database)
                # A foreign cache value is never evidence of this operation's
                # native ownership. Keep it out of the retirement target.
                if not self.registry.is_registered(connection):
                    self._changed()
                self.connection = connection
                self._require_worker_current()  # Native opening can invoke callbacks.
                for kwargs in kwargs_pair:
                    self._require_worker_current()
                    try:
                        result = self.reader(**kwargs)
                    except Exception as error:
                        body_error = error
                        self._results.append((None, error))
                        # Original ordinary errors stop the second query. Source
                        # drift takes precedence over publishing partial rows.
                        self._require_worker_current()
                        break
                    self._require_worker_current()
                    self._results.append((result, None))
        except BaseException as error:
            operation_error = error
        retirement_error = None
        try:
            self._retire_created_handle()
            self._require_worker_current_after_retirement()
        except BaseException as error:
            retirement_error = error
        if self.source_changed:
            error = (
                operation_error
                or retirement_error
                or _BrowserSourceChanged("console_browser_source_changed")
            )
            if retirement_error is not None and retirement_error is not error:
                raise error from retirement_error
            raise error
        if body_error is not None:
            secondary = retirement_error or operation_error
            if secondary is not None:
                # Preserve the original declared error as primary. The UI keeps
                # its existing partial-row/error ABI while cleanup stays visible.
                body_error.__cause__ = secondary
            return self._results
        if operation_error is not None:
            if retirement_error is not None:
                raise operation_error from retirement_error
            raise operation_error
        if retirement_error is not None:
            raise retirement_error
        return self._results

    def _require_worker_current_after_retirement(self) -> None:
        self.require_source_current()
        # A successfully retired owned cache is empty; a caller borrower stays
        # selected. Drift checks still compare the original local object.
        expected = self.previous if self.connection is self.previous else None
        if getattr(self.local, "conn", None) is not expected:
            self._changed()
        self._owner_on_loop()
        self.require_source_current()

    async def run(
        self, kwargs_pair: tuple[dict, ...]
    ) -> list[tuple[Any, Exception | None]]:
        self.require_loop_current()
        worker = asyncio.create_task(asyncio.to_thread(self._read_pair, kwargs_pair))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self.cancelled = True
            # Repeated cancellation cannot let an owning waiter abandon native
            # work before its captured handle has reached its retirement boundary.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if worker.done() and not worker.cancelled():
                worker.exception()
            raise


def capture_stock_browser_read(
    controller: object,
    service: object,
    local_service: object,
    scope_service: object,
    *,
    controller_source: tuple,
    factory_is_current: Callable[[], bool],
) -> _StockBrowserRead | None:
    """Decline all preinstalled/custom/memory routes before optional selection."""
    controller_type = controller_source[5][0][1]
    if (
        not factory_is_current()
        or not _sources_current()
        or not _receiver_current(controller, controller_type, controller_source)
        or not _receiver_current(service, _SERVICE_TYPE, _SERVICE_SOURCE)
        or not _plain_fields(controller, controller_type, ("app_instance",))
        or not _plain_fields(service, _SERVICE_TYPE, ("db",))
    ):
        return None
    database = vars(service).get("db")
    if (
        not _receiver_current(database, _NOTES_TYPE, _NOTES_SOURCE)
        or not _plain_fields(
            database,
            _NOTES_TYPE,
            (
                "_local",
                "_connection_quiescence",
                "db_path",
                "db_path_str",
                "is_memory_db",
            ),
        )
        or vars(database).get("is_memory_db") is not False
        or type(vars(database).get("_local")) is not threading.local
    ):
        return None
    app = vars(controller).get("app_instance")
    if not _plain_fields(
        app,
        type(app),
        (
            "local_chat_conversation_service",
            "chat_conversation_scope_service",
        ),
    ):
        return None
    if service is not local_service and (
        scope_service is None
        or not _plain_fields(scope_service, type(scope_service), ("local_service",))
        or vars(scope_service).get("local_service") is not service
    ):
        return None
    registry = vars(database).get("_connection_quiescence")
    if not _receiver_current(registry, _REGISTRY_TYPE, _RETIREMENT_SOURCE):
        return None
    captured = _StockBrowserRead(
        controller,
        controller_type,
        controller_source,
        app,
        type(app),
        service,
        local_service,
        scope_service,
        database,
        vars(database).get("_local"),
        registry,
        vars(database).get("db_path"),
        vars(database).get("db_path_str"),
        MethodType(_SERVICE_READER, service),
        asyncio.get_running_loop(),
        factory_is_current,
    )
    captured.require_loop_current()
    return captured


_HELPER_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    tuple(
        (_StockBrowserRead, name, descriptor)
        for name, descriptor in vars(_StockBrowserRead).items()
        if type(descriptor) is FunctionType
    ),
    tuple(
        (name, globals()[name])
        for name in (
            "_StockBrowserRead",
            "_BrowserSourceChanged",
            "_function_current",
            "_source_current",
            "_plain_fields",
            "_receiver_current",
            "_sources_current",
            "capture_stock_browser_read",
            "_SERVICE_SOURCE",
            "_NOTES_SOURCE",
            "_CORE_SOURCE",
            "_CLOSE_SOURCE",
            "_CLOSING_SOURCE",
            "_RETIREMENT_SOURCE",
            "_SERVICE_TYPE",
            "_NOTES_TYPE",
            "_REGISTRY_TYPE",
            "_CORE_OPERATION",
            "_RETIRE_HANDLE",
            "_GET_HANDLE",
            "_SERVICE_READER",
        )
    ),
    tuple(
        (
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for function in (
            _function_current,
            _source_current,
            _plain_fields,
            _receiver_current,
            _sources_current,
            capture_stock_browser_read,
            *(
                descriptor
                for descriptor in vars(_StockBrowserRead).values()
                if type(descriptor) is FunctionType
            ),
        )
    ),
)
