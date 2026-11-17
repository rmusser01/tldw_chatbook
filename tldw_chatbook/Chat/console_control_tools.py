"""Original tool invocation handoffs, independent of the response lifetime."""

from __future__ import annotations

import threading
import contextvars
from contextlib import contextmanager


_mcp_invocation = contextvars.ContextVar("console_mcp_invocation", default=None)


class _SelectedToolThread:
    """Only the originating command and this original physical tool Thread."""

    def __init__(self, source, runner):
        self.source, self.runner = source, runner
        self.controls = source.controls
        self.thread = None
        self.start_resolved = threading.Event()
        self.never_started = False
        self.runner_returned = False

    def require_current(self):
        self.controls._require_graph_identity()
        with self.controls._lock:
            if (
                self.source.sources is not self.controls._domain.sources
                or self.source.identity.source_revision
                != self.controls._source_revision
                or not self.controls._effects_open
            ):
                raise RuntimeError("Console tool Thread source is fenced")

    @contextmanager
    def scope(self):
        from .console_control_dispatch import _command_source

        if threading.current_thread() is not self.thread:
            raise RuntimeError("Console tool Thread receiver changed")
        token = _command_source.set(self.source)
        try:
            yield
        finally:
            _command_source.reset(token)

    def bind_thread(self, thread):
        if self.thread is not None or thread._target is not self.runner:
            raise RuntimeError("Console tool Thread invocation changed")
        self.thread = thread
        with self.controls._lock:
            self.controls._tool_threads.add(self)

    def resolve_start(self):
        thread = self.thread
        self.never_started = (
            thread is None
            or thread.ident is None
            and not thread._started.is_set()
            and not thread.is_alive()
        )
        self.start_resolved.set()
        if self.never_started:
            with self.controls._lock:
                self.controls._tool_threads.discard(self)


def capture_selected_tool_thread(runner):
    from .console_control_dispatch import _command_source, require_control_source

    source = _command_source.get()
    if source is None:
        return None
    if require_control_source(source.controls._runtime) is not source:
        raise RuntimeError("Console tool Thread source changed")
    worker = _SelectedToolThread(source, runner)
    worker.require_current()
    return worker


async def drain_tool_threads(controls):
    import asyncio

    with controls._lock:
        workers = tuple(controls._tool_threads)
    for worker in workers:

        def join(worker=worker):
            worker.start_resolved.wait()
            if not worker.never_started:
                worker.thread.join()

        task = asyncio.create_task(asyncio.to_thread(join))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if not worker.never_started and (
            worker.thread.is_alive() or not worker.runner_returned
        ):
            raise RuntimeError("Console tool Thread retirement is unproven")
        with controls._lock:
            controls._tool_threads.discard(worker)
        if cancelled:
            raise asyncio.CancelledError


def bind_control_provider(controller, provider):
    domain = getattr(controller, "domain_services", None)
    if domain is None or domain.runtime._execution_service._local_controls is None:
        return provider
    from .console_control_dispatch import require_control_source

    source = require_control_source(domain.runtime)
    if source.controls._controller is not controller:
        raise RuntimeError("Console tool composition owner changed")
    previous = getattr(provider, "_console_local_controls", None)
    if previous is not None and previous is not source.controls:
        raise RuntimeError("Console provider cannot adopt another control owner")
    provider._console_local_controls = source.controls
    return provider


def provider_control_source(provider):
    controls = getattr(provider, "_console_local_controls", None)
    if controls is None:
        return None
    from .console_control_dispatch import require_control_source

    source = require_control_source(controls._runtime)
    if source.controls is not controls:
        raise RuntimeError("Console tool invocation owner changed")
    return source


@contextmanager
def tool_permission_scope(provider):
    """Observe the original complete synchronous permission callback only."""
    source = provider_control_source(provider)
    if source is None:
        yield
        return
    with _permission_scope(
        source,
        getattr(provider, "_service", source.controls._permissions._control_plane),
    ):
        yield


@contextmanager
def service_permission_scope(service):
    """Fixed captured catalog callback, on its original native worker."""
    from .console_control_dispatch import _command_source

    source = _command_source.get()
    if source is None:
        yield
        return
    with _permission_scope(source, service):
        yield


@contextmanager
def _permission_scope(source, service):
    from tldw_chatbook.MCP.local_permission_operation import (
        LocalPermissionOperation,
        active,
    )
    from tldw_chatbook.MCP import permission_process  # noqa: F401

    operations = source.controls._permissions
    if service is not operations._control_plane:
        raise RuntimeError("Console tool permission receiver changed")
    existing = active()
    if existing is not None:
        if existing.source is not operations._store:
            raise RuntimeError("Console tool permission source changed")
        existing.require_current()
        yield
        return
    operation = LocalPermissionOperation(operations._store, operations._require_current)
    with source.controls.permission_effect(operations, operation, _source=source):
        try:
            with operation:
                yield
        finally:
            if not operation.retired:
                operations._unresolved.append(operation)


def begin_tool_dispatch(provider):
    source = provider_control_source(provider)
    return None if source is None else _ToolDispatch(provider, source)


def begin_mcp_dispatch(provider, tool, observation):
    source = provider_control_source(provider)
    if source is None:
        return None
    if provider._service is not source.controls._permissions._control_plane:
        raise RuntimeError("Console MCP invocation service changed")
    return _MCPDispatch(provider, source, tool, observation)


@contextmanager
def mcp_invocation_scope(invocation):
    token = _mcp_invocation.set(invocation)
    try:
        yield
    finally:
        _mcp_invocation.reset(token)


def bind_mcp_transport(receiver, method, params, observation):
    invocation = _mcp_invocation.get()
    if invocation is None or method != "tools/call":
        return None
    if not isinstance(params, dict) or params.get("name") != invocation.tool.name:
        raise RuntimeError("Console MCP transport tool changed")
    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection
    from tldw_chatbook.MCP.streamable_http import StreamableHTTPConnection

    if type(receiver) not in (_StdioJSONRPCConnection, StreamableHTTPConnection):
        raise RuntimeError("Console MCP transport receiver changed")
    invocation.bind_receiver(receiver, observation)
    return invocation


def mcp_transport_handoff(invocation, receiver, observation):
    if invocation is not None:
        invocation.handoff(receiver, observation)


def mcp_builtin_handoff(delegate, tool_name, *, handler=None):
    invocation = _mcp_invocation.get()
    if invocation is None:
        return
    from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate

    if (
        invocation.provider._service.local_service.runtime_delegate is not delegate
        or type(delegate) is not LocalMCPRuntimeDelegate
        or invocation.tool.name != tool_name
        or not invocation.tool.server_key.startswith("builtin:")
    ):
        raise RuntimeError("Console MCP builtin invocation changed")
    if handler is not None and getattr(handler, "__self__", None) is not delegate:
        raise RuntimeError("Console MCP builtin handler changed")
    invocation.bind_receiver(delegate, invocation.observation)
    invocation.handoff(delegate, invocation.observation)


class _ToolDispatch:
    def __init__(self, provider, source):
        self.provider, self.source = provider, source
        self.controls = source.controls
        self._lock = threading.Lock()
        self.finished = False
        self.effect = self.controls._begin_effect("tool_dispatch", self, source)

    def local_handoff(self, provider):
        with self._lock:
            if provider is not self.provider or self.finished:
                raise RuntimeError("Console local tool handoff changed")
            self.finished = True
        self.controls._finish_effect(self.effect, retired=True)

    def returned_without_handoff(self, *, retired):
        with self._lock:
            if self.finished:
                return
            self.finished = True
        self.controls._finish_effect(self.effect, retired=retired)


class _MCPDispatch(_ToolDispatch):
    def __init__(self, provider, source, tool, observation):
        super().__init__(provider, source)
        self.tool, self.observation = tool, observation
        self.receiver = None
        self.bridge = None
        self.handed_off = False

    def bind_receiver(self, receiver, observation):
        with self._lock:
            if (
                self.finished
                or observation is not self.observation
                or self.receiver is not None
                and self.receiver is not receiver
            ):
                raise RuntimeError("Console MCP dispatch receiver changed")
            self.receiver = receiver

    def handoff(self, receiver, observation):
        with self._lock:
            if (
                self.receiver is not receiver
                or observation is not self.observation
                or self.finished
                and not self.handed_off
            ):
                raise RuntimeError("Console MCP dispatch handoff changed")
            if self.handed_off:
                return  # The same HTTP invocation may retry its headers.
            self.handed_off = True
            self.finished = True
        self.controls._finish_effect(self.effect, retired=True)
