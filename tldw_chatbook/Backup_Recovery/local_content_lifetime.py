"""Finite maintenance lifetimes for installed local Skills and Chatbooks calls."""

import asyncio
import inspect
import os
import threading
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from . import storage_admission as storage
from .bootstrap import RecoveryRequired
from .profile_paths import lexical_path
from .runtime_producer_lifetime import ProducerLifetime


class LocalContentLifetime:
    """Drain installed content operations before their database owners pause."""

    def __init__(self):
        self.producer = ProducerLifetime()

    def _maintenance_close_admission(self):
        self.producer.close()

    async def _maintenance_drain(self, deadline):
        return await self.producer.drain(deadline)

    def _maintenance_resume(self):
        self.producer.resume()


participant = LocalContentLifetime()
_current = ContextVar("local_content_operation", default=None)


def _identity():
    return os.getpid(), threading.get_ident(), storage._task_identity()


@dataclass
class _Call:
    identity: tuple
    leases: dict
    cleanup: ExitStack
    parent: object = None
    live: bool = True

    def check(self):
        if not self.live or self.identity[0] != os.getpid():
            raise RecoveryRequired("local_content_operation_retired")
        if self.parent is not None:
            self.parent.check()


@contextmanager
def operation(paths=(), *, _parent=None):
    """Hold exact selected sources through a complete installed call chain."""
    previous = _parent or _current.get()
    if previous is not None:
        previous.check()
        if _parent is None and previous.identity != _identity():
            raise RecoveryRequired("local_content_operation_context_changed")
    with ExitStack() as stack:
        if previous is None:
            stack.enter_context(participant.producer.operation())
        leases = dict(previous.leases) if previous is not None else {}
        for path in paths:
            if path is None or str(path) == ":memory:":
                continue
            selected = lexical_path(path)
            if selected not in leases:
                leases[selected] = stack.enter_context(
                    storage.acquire_storage(selected)
                )
            leases[selected].execution_context(selected)
        scope = _Call(_identity(), leases, stack, previous)
        token = _current.set(scope)
        try:
            yield
        finally:
            # Database cleanup registered below runs before source leases close.
            try:
                stack.close()
            finally:
                scope.live = False
                _current.reset(token)


def call(sources, *, detached=False):
    """Wrap a named owner boundary using its explicit source selector."""

    def decorate(function):
        signature = inspect.signature(function)

        def selected(args, kwargs):
            values = signature.bind(*args, **kwargs).arguments
            return sources(values)

        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def asynchronous(*args, **kwargs):
                async def admitted():
                    with operation(selected(args, kwargs)):
                        return await function(*args, **kwargs)

                # The shipped script caller has no enclosing content operation.
                # Its retained task owns the scope independently of its waiter.
                if detached and _current.get() is None:
                    completion = asyncio.create_task(admitted())
                    try:
                        return await asyncio.shield(completion)
                    except asyncio.CancelledError:
                        completion.add_done_callback(_report_detached_failure)
                        raise
                return await admitted()

            return asynchronous

        @wraps(function)
        def synchronous(*args, **kwargs):
            with operation(selected(args, kwargs)):
                return function(*args, **kwargs)

        return synchronous

    return decorate


def own_database(database):
    """Close an exact newly constructed Chatbook DB on its creating thread."""
    scope = _current.get()
    if scope is None or scope.identity != _identity():
        raise RecoveryRequired("local_content_operation_missing")
    scope.check()
    scope.cleanup.callback(database.close_connection)
    return database


@contextmanager
def worker_databases(databases):
    """Retire only new current-thread handles of supplied Library source DBs."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    with ExitStack() as stack:
        for database in databases:
            if type(database) not in (CharactersRAGDB, MediaDatabase, PromptsDatabase):
                continue
            if (
                not database.is_memory_db
                and getattr(database._local, "conn", None) is None
            ):
                stack.callback(database.close_connection)
        yield


def run_async(coroutine):
    """Transfer the synchronous Library export span to its explicit child loop."""
    parent = _current.get()
    if parent is None or parent.identity != _identity():
        coroutine.close()
        raise RecoveryRequired("local_content_operation_missing")
    parent.check()

    async def child():
        with operation(_parent=parent):
            return await coroutine

    return asyncio.run(child())


def native_worker(function):
    """Transfer an accepted script call only to its explicitly awaited worker."""
    parent = _current.get()
    if parent is None or parent.identity != _identity():
        raise RecoveryRequired("local_content_operation_missing")
    parent.check()

    @wraps(function)
    def worker(*args, **kwargs):
        with operation(_parent=parent):
            return function(*args, **kwargs)

    return worker


def _report_detached_failure(completion):
    """Observe a retained script failure after its original waiter detached."""
    from loguru import logger

    if not completion.cancelled():
        error = completion.exception()
        if error is not None:
            logger.error(
                "Retained local content operation failed: {}", type(error).__name__
            )
