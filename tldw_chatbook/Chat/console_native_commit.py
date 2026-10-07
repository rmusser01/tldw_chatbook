"""Retain ordinary Console save and terminal work through cancellation."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .console_chat_store import (
        ConsoleChatStore,
        ConsoleDurableAcceptanceFingerprint,
        ConsoleDurableTurnCommit,
    )
    from .console_dispatch_checkpoint import ConsoleDurableTurnAcceptance


@dataclass(frozen=True, slots=True)
class ConsoleNativeCommitCompletion:
    """Actual callback outcome after its operation-owned cleanup has retired."""

    commit: ConsoleDurableTurnCommit | None
    error: BaseException | None
    caller_cancelled: bool


@dataclass(frozen=True, slots=True)
class ConsoleNativeSettlementCompletion:
    """Actual accepted settlement outcome after native cleanup has retired."""

    settled: bool
    error: BaseException | None
    caller_cancelled: bool


@dataclass(frozen=True, slots=True)
class _ConsoleNativeBinding:
    """Pure original-source binding for one synchronous Console callback."""

    store: ConsoleChatStore
    turn_owner: object
    persistence: object
    database: object


_native_commit_binding: contextvars.ContextVar[_ConsoleNativeBinding | None] = (
    contextvars.ContextVar("console_native_commit_binding", default=None)
)


async def _run_owned_native_call[_Result](
    binding: _ConsoleNativeBinding,
    callback: Callable[[], _Result],
) -> tuple[_Result | None, BaseException | None, bool]:
    """Retire one of the two named Console callbacks on its original source."""
    store, persistence, database = (
        binding.store,
        binding.persistence,
        binding.database,
    )

    def invoke_bound() -> _Result:
        if (
            store.persistence is not persistence
            or getattr(persistence, "db", None) is not database
        ):
            raise RuntimeError("Durable Console persistence owner changed.")
        token = _native_commit_binding.set(binding)
        try:
            return callback()
        finally:
            _native_commit_binding.reset(token)

    try:
        memory_backed = bool(getattr(database, "is_memory_db", False))
    except BaseException as error:
        return None, error, False
    if memory_backed:
        try:
            return invoke_bound(), None, False
        except BaseException as error:
            return None, error, False

    submit_resolved = threading.Event()
    submitted = False

    def invoke() -> _Result | None:
        # submit() can enqueue a wrapper before thread creation raises. Only a
        # successfully returned Future permits this wrapper to touch storage.
        submit_resolved.wait()
        if not submitted:
            return None
        if (
            store.persistence is not persistence
            or getattr(persistence, "db", None) is not database
        ):
            raise RuntimeError("Durable Console persistence owner changed.")
        from tldw_chatbook.DB.base_db import operation_owned_connection
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        with operation_owned_connection(database):
            if type(database) is CharactersRAGDB and not database.is_memory_db:
                from tldw_chatbook.Backup_Recovery.participants import _core_operation

                with _core_operation(database):
                    return invoke_bound()
            return invoke_bound()

    try:
        context = contextvars.copy_context()
        future = asyncio.get_running_loop().run_in_executor(None, context.run, invoke)
    except BaseException as error:
        submit_resolved.set()
        return None, error, False
    submitted = True
    submit_resolved.set()

    caller_cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            # A callback can raise this exact error even if the task retains an
            # earlier cancellation count. Only cancellation of the await marks
            # the caller; consume the original callback exception below.
            if future.done() and not future.cancelled() and future.exception() is error:
                break
            caller_cancelled = True
        except BaseException:
            break
    try:
        return future.result(), None, caller_cancelled
    except BaseException as error:
        return None, error, caller_cancelled


async def commit_durable_turn_owned(
    store: ConsoleChatStore,
    acceptance: ConsoleDurableTurnAcceptance,
) -> ConsoleNativeCommitCompletion:
    """Finish the captured save before reporting any caller cancellation.

    Args:
        store: Original store whose callback and persistence binding are captured.
        acceptance: Exact already-admitted ordinary turn.

    Returns:
        The original callback result or error, plus observed caller cancellation.
        This outcome does not establish rollback or permission authority.
    """
    try:
        persistence = store.persistence
        database = getattr(persistence, "db", None)
        callback = store.commit_durable_turn
    except BaseException as error:
        return ConsoleNativeCommitCompletion(None, error, False)
    binding = _ConsoleNativeBinding(store, acceptance, persistence, database)
    return ConsoleNativeCommitCompletion(
        *await _run_owned_native_call(binding, lambda: callback(acceptance))
    )


async def settle_accepted_durable_turn_owned(
    store: ConsoleChatStore,
    preparation_id: str,
    *,
    fingerprint: ConsoleDurableAcceptanceFingerprint,
    terminal_state: str,
    content: str,
    persistence: object,
    database: object,
    metadata_json: str | None = None,
) -> ConsoleNativeSettlementCompletion:
    """Retire the exact accepted terminal write without blocking the owner loop.

    Args:
        store: Original store retaining the accepted turn.
        preparation_id: Exact accepted preparation.
        fingerprint: Immutable original acceptance owner.
        terminal_state: Only stopped or failed.
        content: Assistant terminal content.
        persistence: Original persistence captured by the controller owner.
        database: Original database captured by the controller owner.
        metadata_json: Optional terminal metadata object.

    Returns:
        Actual settlement/refusal or error, plus observed caller cancellation.
    """
    try:
        callback = store.settle_accepted_durable_turn
    except BaseException as error:
        return ConsoleNativeSettlementCompletion(False, error, False)

    def settle() -> bool:
        if metadata_json is None:
            return callback(
                preparation_id,
                fingerprint=fingerprint,
                terminal_state=terminal_state,
                content=content,
            )
        return callback(
            preparation_id,
            fingerprint=fingerprint,
            terminal_state=terminal_state,
            content=content,
            metadata_json=metadata_json,
        )

    binding = _ConsoleNativeBinding(store, fingerprint, persistence, database)
    result, error, cancelled = await _run_owned_native_call(binding, settle)
    return ConsoleNativeSettlementCompletion(bool(result), error, cancelled)
