"""Console durable writes that app teardown must wait for (TASK-33628.5).

A Console Delete or Undo saves its durable half off the event loop
(``console_message_delete.run_durable_off_loop``), and it holds the store's
fork-source and voice-promotion fences from before the save starts until its
result is applied. ``ConsoleChatStore.end_app_runtime`` replaces the store's
volatile state under a voice fence that refuses while any such admission is
held. So when Quit anyway ended the app mid-save, the store refused, and
``ConsoleRuntime.dispose`` skipped the whole store teardown: the
trace-settlement drain, both executor shutdowns and the teardown retries.

Here each off-loop write registers its settled outcome against the store it
writes for. At app exit :func:`end_store_after_writes` closes admission (a
Delete or Undo started after that is refused before it takes any fence),
waits a bounded time for the writes already running, and then ends the
store. If the bound passes first, it says so without content and ends the
store without replacing its state, so every other teardown step still runs.
Each write is one SQLite transaction, so a quit never half-applies one.

Every call runs on the event-loop thread. Imported at the first Delete or at
app exit, never before first paint (ADR-097).
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field
from typing import Any, Callable
from weakref import WeakKeyDictionary

from loguru import logger


class ConsoleClosingError(RuntimeError):
    """The app is closing, so a new durable Console write was refused."""


@dataclass
class _StoreWrites:
    closed: bool = False
    pending: set[asyncio.Future[Any]] = field(default_factory=set)


_BY_STORE: WeakKeyDictionary[Any, _StoreWrites] = WeakKeyDictionary()


def _writes(store: Any) -> _StoreWrites | None:
    """The store's registry entry, made on first use.

    ``None`` for a store double that cannot be weakly referenced: nothing
    can be registered against it, so nothing is waited for or refused.
    """
    try:
        writes = _BY_STORE.get(store)
        if writes is None:
            writes = _BY_STORE[store] = _StoreWrites()
    except TypeError:
        return None
    return writes


def admit(store: Any) -> None:
    """Refuse a new durable write once the store's teardown has begun.

    Args:
        store: The Console store the write is for.

    Raises:
        ConsoleClosingError: The app is closing; nothing was started.
    """
    writes = _writes(store)
    if writes is not None and writes.closed:
        raise ConsoleClosingError("Chatbook is closing.")


def track(store: Any, settled: asyncio.Future[Any]) -> None:
    """Make app teardown wait for ``settled`` before it ends ``store``.

    Args:
        store: The Console store the write is for.
        settled: Resolves once the write's outcome has been applied on the
            event loop and its fences released.
    """
    writes = _writes(store)
    if writes is None or settled.done():
        return
    writes.pending.add(settled)
    settled.add_done_callback(writes.pending.discard)


async def _close_and_settle(store: Any, timeout: float) -> int:
    """Close admission, then wait up to ``timeout`` s for the writes running.

    Returns:
        How many writes were still running when the bound passed.
    """
    writes = _writes(store)
    if writes is None:
        return 0
    writes.closed = True
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    while writes.pending and (remaining := deadline - loop.time()) > 0:
        await asyncio.wait(set(writes.pending), timeout=remaining)
    return len(writes.pending)


async def end_store_after_writes(
    store: Any, end_app_runtime: Callable[..., None], timeout: float
) -> None:
    """End ``store`` for app exit once its durable writes have settled.

    Args:
        store: The Console store.
        end_app_runtime: Its ``end_app_runtime``; it runs off the event loop.
        timeout: How long to wait for writes still running, in seconds.

    Raises:
        Exception: Whatever ``end_app_runtime`` raised.
    """
    running = await _close_and_settle(store, timeout)
    if running:
        # Their fences are still held, so the voice-fenced state swap would
        # refuse; every other step does not depend on it.
        logger.warning(
            "Console runtime: {} message write(s) from Delete, Undo or Done "
            "still saving after {:.1f} s at dispose; ending the store without "
            "replacing its state.",
            running,
            timeout,
        )
        end_app_runtime = functools.partial(end_app_runtime, replace_state=False)
    await asyncio.to_thread(end_app_runtime)
