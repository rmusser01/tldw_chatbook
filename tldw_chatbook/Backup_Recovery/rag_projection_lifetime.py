"""Source-local Chroma borrowers; no engine construction during maintenance."""

import asyncio
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from loguru import logger

from ..TTS._async_lifecycle import join_retained_task
from .bootstrap import RecoveryRequired
from .storage_admission import acquire_storage


@dataclass(eq=False)
class _Borrower:
    lease: object
    path: object
    client: object = None
    system: object = None
    server: object = None
    store: object = None
    closed: bool = False
    failed: bool = False
    qualified: bool = False
    close_lock: object = field(default_factory=threading.Lock)

    def attach(self, client, store=None):
        """Observe only the installed dependency's exact native implementation."""
        self.client = client
        self.store = weakref.ref(store) if store is not None else None
        loaded = sys.modules.get("chromadb.api.client")
        package = sys.modules.get("chromadb")
        native = sys.modules.get("chromadb.api.rust")
        server = getattr(client, "_server", None)
        self.qualified = (
            loaded is not None
            and type(client) is vars(loaded).get("Client")
            and vars(package).get("__version__") == "1.5.8"
            and native is not None
            and type(server) is vars(native).get("RustBindingsAPI")
            and server._system is client._system
            and "bindings" in vars(server)
        )
        if self.qualified:
            self.system = client._system
            # Retain the Python server, never the native bindings themselves.
            self.server = server


class ChromaLifetime:
    """Count accepted source work and retain native holds until proven stopped."""

    def __init__(self):
        self._lock = threading.RLock()
        self._tokens = set()
        self._borrowers = set()
        self._accepted = ContextVar("rag_projection_accepted", default=None)
        self._closed = False
        self._retirement = None

    def reserve(self):
        """Reserve an ingestion queue item before it can reach its worker."""
        with self._lock:
            if self._closed and self._accepted.get() not in self._tokens:
                raise RecoveryRequired("rag_projection_operations_paused")
            token = object()
            self._tokens.add(token)
            return token

    def release(self, token):
        with self._lock:
            self._tokens.remove(token)

    @contextmanager
    def accepted(self, token):
        """Transfer an exact live queued item's acceptance to its source worker."""
        with self._lock:
            if token not in self._tokens:
                raise RecoveryRequired("rag_projection_operation_inactive")
        context = self._accepted.set(token)
        try:
            yield
        finally:
            self._accepted.reset(context)

    @contextmanager
    def operation(self):
        """Every entered native/async scope holds its own accepted lifetime."""
        token = self.reserve()
        try:
            with self.accepted(token):
                yield
        finally:
            self.release(token)

    def sync_operation(self, function):
        @wraps(function)
        def invoke(*args, **kwargs):
            with self.operation():
                return function(*args, **kwargs)

        return invoke

    def async_operation(self, function):
        """Retain the actual coroutine through native work and its bookkeeping."""

        @wraps(function)
        async def invoke(*args, **kwargs):
            with self.operation():
                completion = asyncio.create_task(function(*args, **kwargs))
                await join_retained_task(completion)
                return completion.result()

        return invoke

    @contextmanager
    def opening(self, path):
        """Acquire storage before calling the installed native constructor."""
        with self.operation():
            borrower = _Borrower(acquire_storage(path), path)
            with self._lock:
                self._borrowers.add(borrower)
            try:
                yield borrower
            except BaseException:
                # A failing native constructor may already have acquired resources.
                # Keep its hold; constructor failure is not native retirement.
                borrower.failed = True
                raise

    def retained_lease(self, store):
        """Borrow only this exact store's live installed native lifetime."""
        with self._lock:
            for item in self._borrowers:
                if (
                    item.store is not None
                    and item.store() is store
                    and item.client is store._client
                    and item.path == store.persist_directory
                    and not item.closed
                    and not item.failed
                ):
                    item.lease.execution_context(item.path)
                    return item.lease
        raise RecoveryRequired("projection_native_lease_unavailable")

    def close_client(self, client):
        """Preserve ordinary cleanup behavior while retaining ambiguous holds."""
        with self._lock:
            borrower = next(
                (item for item in self._borrowers if item.client is client), None
            )
        close_lock = borrower.close_lock if borrower is not None else threading.Lock()
        with close_lock:
            if borrower is not None and (borrower.closed or borrower.failed):
                with self._lock:
                    self._release_stopped()
                return borrower.closed and not borrower.failed
            try:
                close = getattr(client, "close", None)
                if not callable(close):
                    raise TypeError("public close unavailable")
                close()
            except Exception:  # noqa: BLE001 - preserve operation result and retain native evidence.
                if borrower is not None:
                    borrower.failed = True
                logger.warning("chroma_retirement_unqualified: client close failed")
                return False
            if borrower is not None:
                borrower.closed = True
                if not borrower.qualified:
                    logger.warning(
                        "chroma_retirement_unqualified: installed version unsupported"
                    )
                with self._lock:
                    self._release_stopped()
                return not borrower.failed
            return True

    def _release_stopped(self):
        """Require actual Rust binding retirement, not System's early stop flag."""
        for borrower in tuple(self._borrowers):
            system = borrower.system
            group = [item for item in self._borrowers if item.system is system]
            if (
                system is None
                or any(not item.closed or item.failed for item in group)
                or system._running is not False
                # In 1.5.8 System.stop clears _running BEFORE component stop.
                # RustBindingsAPI.stop's `del self.bindings` retires the native
                # SQLite/HNSW owner; an external last-client failure may leave
                # it alive even though all our own clients closed successfully.
                or any(
                    item.server is None
                    or item.client._server is not item.server
                    or "bindings" in vars(item.server)
                    for item in group
                )
            ):
                continue
            for item in group:
                store = item.store() if item.store is not None else None
                if store is not None and store._client is item.client:
                    store._client = None
                    store._collection = None
                item.lease.close()
                self._borrowers.discard(item)

    def _maintenance_close_admission(self):
        with self._lock:
            self._closed = True

    async def _maintenance_drain(self, deadline):
        while True:
            with self._lock:
                if not self._closed:
                    raise RecoveryRequired("rag_projection_intake_open")
                if not self._tokens:
                    if not self._borrowers:
                        return True
                    if self._retirement is None:
                        self._retirement = threading.Thread(
                            target=self._close_borrowers,
                            name="rag-native-retirement",
                            daemon=True,
                        )
                        self._retirement.start()
                    elif not self._retirement.is_alive():
                        self._release_stopped()
                        return not self._borrowers
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.01))

    def _close_borrowers(self):
        with self._lock:
            borrowers = tuple(self._borrowers)
        for borrower in borrowers:
            if borrower.client is not None:
                self.close_client(borrower.client)

    async def _maintenance_resume(self):
        # A timed-out waiter does not authorize reopening during native stop.
        while self._retirement is not None and self._retirement.is_alive():
            await asyncio.sleep(0.01)
        with self._lock:
            self._retirement = None
            self._closed = False


participant = ChromaLifetime()


def store_operation(function):
    """Serialize a store's native operation with ordinary explicit close."""

    @wraps(function)
    def invoke(self, *args, **kwargs):
        with participant.operation(), self._projection_lock:
            return function(self, *args, **kwargs)

    return invoke
