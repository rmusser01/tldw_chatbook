"""Exact Task and native-worker observations for one original MCP submission."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import Future
from contextlib import contextmanager

from .console_control_tools import _mcp_invocation


_worker = contextvars.ContextVar("console_mcp_native_worker", default=None)


def observe_activation(state):
    invocation = _mcp_invocation.get()
    if invocation is None:
        return
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = _worker.get() or invocation.bridge
    with storage._lock:
        leases = tuple(
            (lease, storage._holds.get(lease._key)) for lease in state.leases.values()
        )
    owner.states.append((state, leases))


def failed_activation_entry():
    invocation = _mcp_invocation.get()
    if invocation is not None:
        # The original guard did not yield its state. Its partial admission
        # cannot be certified by a coroutine exception or an empty census.
        (_worker.get() or invocation.bridge).unproven = True


def _states_retired(owner):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.MCP.local_permission_operation import _hold_retired_or_shared

    with storage._lock:
        return not owner.unproven and all(
            not state.live
            and all(
                lease not in storage._live_leases
                and lease._key is None
                and _hold_retired_or_shared(hold, storage)
                for lease, hold in leases
            )
            for state, leases in owner.states
        )


class MCPBridgeSubmission:
    """The single original scheduled coroutine, its Task and actual workers."""

    def __init__(self, invocation, coroutine, loop):
        self.invocation, self.coroutine, self.loop = invocation, coroutine, loop
        self.controls = invocation.controls
        self.future = Future()
        self.retired = Future()
        self.task = None
        self.states = []
        self.workers = []
        self.unproven = False
        self.task_finished = False
        self._lock = threading.Lock()
        invocation.bridge = self
        with self.controls._lock:
            self.controls._mcp_submissions.add(self)

    def submit(self):
        # This is the original run_coroutine_threadsafe callback and chaining
        # contract, with the exact created Task retained before it can start.
        def callback():
            try:
                self.task = asyncio.ensure_future(self.coroutine, loop=self.loop)
                self.task.add_done_callback(self._task_done)
                asyncio.futures._chain_future(self.task, self.future)
            except BaseException as error:
                if self.task is None:
                    self.coroutine.close()
                    self.task_finished = True
                    self.refresh()
                if not self.future.cancelled():
                    self.future.set_exception(error)

        try:
            self.loop.call_soon_threadsafe(callback)
        except BaseException:
            self.coroutine.close()
            self.task_finished = True
            self.refresh()
            raise
        return self.future

    def _task_done(self, task):
        if task is not self.task:
            raise RuntimeError("Console MCP scheduled Task changed")
        self.task_finished = True
        self.refresh()

    def refresh(self):
        with self._lock:
            if (
                self.retired.done()
                or not self.task_finished
                or any(not worker.finished for worker in self.workers)
            ):
                return
            retired = _states_retired(self) and all(
                worker.retired for worker in self.workers
            )
            self.invocation.returned_without_handoff(retired=retired)
            self.retired.set_result(retired)
        if retired:
            with self.controls._lock:
                self.controls._mcp_submissions.discard(self)


class _NativeWorker:
    def __init__(self, invocation, service, function, args):
        self.invocation, self.service, self.function, self.args = (
            invocation,
            service,
            function,
            args,
        )
        self.states = []
        self.unproven = False
        self.finished = False
        self.native_returned = False
        self.retired = False
        self.task = None
        self.native_binding = None

    @contextmanager
    def observe(self):
        token = _worker.set(self)
        try:
            yield
        except BaseException:
            self.unproven = True
            raise
        finally:
            _worker.reset(token)
            self.retired = _states_retired(self) and (
                self.native_binding is None or self.native_binding.retirement.confirmed
            )
            self.native_returned = True

    def invoke(self):
        if self.native_binding is None:
            return self.function(*self.args)
        from .console_native_commit import _invoke_plan_native

        _require_character_worker(self)
        return _invoke_plan_native(
            self.native_binding, lambda: self.function(*self.args)
        )

    def task_done(self, task):
        if task is not self.task:
            raise RuntimeError("Console MCP native worker Task changed")
        if not task.cancelled():
            task.exception()
        self.retired = self.retired and self.native_returned
        self.finished = True
        self.invocation.bridge.refresh()

    def library_handoff(self):
        from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS

        invocation = self.invocation
        delegate = invocation.provider._service.local_service.runtime_delegate
        if (
            self.service is not delegate
            or invocation.tool.name not in LIBRARY_TOOL_DESCRIPTORS
        ):
            return
        library = delegate._library_service
        if (
            getattr(self.function, "__self__", None) is not library
            or getattr(self.function, "__func__", None) is not type(library).invoke
            or len(self.args) != 2
            or self.args[0] != invocation.tool.name
        ):
            raise RuntimeError("Console MCP Library invocation changed")
        invocation.bind_receiver(library, invocation.observation)
        invocation.handoff(library, invocation.observation)


def capture_worker(service, function, args):
    invocation = _mcp_invocation.get()
    if invocation is None:
        return None
    worker = _NativeWorker(invocation, service, function, args)
    from tldw_chatbook.MCP.tools import MCPTools

    if type(service) is MCPTools:
        from .console_native_commit import (
            _ConsoleNativeBinding,
            _ConsolePlanNativeRetirement,
        )

        _require_character_worker(worker)
        controls = invocation.controls
        worker.native_binding = _ConsoleNativeBinding(
            controls._store,
            invocation,
            controls._store.persistence,
            service.chachanotes_db,
            retirement=_ConsolePlanNativeRetirement(),
        )
    invocation.bridge.workers.append(worker)
    return worker


def _require_character_worker(worker):
    from tldw_chatbook.MCP.tools import MCPTools
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    invocation = worker.invocation
    delegate = invocation.provider._service.local_service.runtime_delegate
    tools = worker.service
    database = tools.chachanotes_db
    name = invocation.tool.name
    if (
        type(tools) is not MCPTools
        or delegate._tools is not tools
        or invocation.receiver is not delegate
        or not invocation.handed_off
        or type(database) is not CharactersRAGDB
        or database is not invocation.source.sources.chat_database
        or database is not invocation.controls._domain.sources.chat_database
        or delegate._require_chachanotes_db() is not database
        or (
            name == "list_characters"
            and (
                worker.args
                or getattr(worker.function, "__self__", None) is not database
                or getattr(worker.function, "__func__", None)
                is not CharactersRAGDB.list_character_cards
            )
        )
        or (
            name in {"create_character", "update_character"}
            and (
                getattr(worker.function, "__self__", None) is not tools
                or getattr(worker.function, "__func__", None)
                is not MCPTools._write_character
            )
        )
        or name not in {"list_characters", "create_character", "update_character"}
    ):
        raise RuntimeError("Console MCP character worker owner changed")


async def selected_character_list(tools):
    """Keep the original single character read and its native resource owner."""
    worker = capture_worker(tools, tools.chachanotes_db.list_character_cards, ())
    if worker is None:
        return await asyncio.to_thread(tools.chachanotes_db.list_character_cards)

    def run():
        with worker.observe():
            return worker.invoke()

    return await await_worker(worker, run)


async def await_worker(worker, run):
    # Keep the actual to_thread Task alive when its async caller is cancelled.
    # Completion is observed inside run, after the original native scopes exit.
    coroutine = asyncio.to_thread(run)
    try:
        worker.task = asyncio.create_task(coroutine)
    except BaseException:
        coroutine.close()
        worker.finished = True
        worker.retired = True  # No Task ever took ownership or entered the worker.
        worker.invocation.bridge.refresh()
        raise
    worker.task.add_done_callback(worker.task_done)
    return await asyncio.shield(worker.task)
