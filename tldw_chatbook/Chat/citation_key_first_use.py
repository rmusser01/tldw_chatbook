"""First-use citation key custody: finite DB phases around key-only work.

ADR024: provision only unpublished key IDs, then publish by identity CAS.
The retained daemon thread cannot be physically cancelled; it owns no DB/UI
resources and can only settle its future, never publish repository state.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Any, Callable

from loguru import logger

from .citation_trace_identity import (
    CitationFingerprintCodec,
    CitationFingerprintKeyProvider,
    CitationFingerprintKeyUnavailable,
    KeyringCitationFingerprintKeyProvider,
    LocalCitationIdentityContext,
    load_fingerprint_codec,
    new_opaque_id,
)


class CitationKeyState(str, Enum):
    """Bounded availability reasons; pending is not a failed fingerprint."""

    PENDING = "pending"
    READY = "ready"
    WRITES_DISABLED = "writes_disabled"
    KEYRING_UNAVAILABLE = "keyring_unavailable"
    KEY_UNREADABLE = "key_unreadable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class _KeyAcquisition:
    identity: LocalCitationIdentityContext
    future: Future = field(repr=False)
    thread: threading.Thread = field(repr=False)


def _load(
    provider: CitationFingerprintKeyProvider, key_id: str
) -> CitationFingerprintCodec | None:
    try:
        return load_fingerprint_codec(provider, key_id)
    except CitationFingerprintKeyUnavailable:
        return None


def _acquire_key_candidate(provider, key_id, candidate_id, future) -> None:
    """Key-only target: deliberately has no repository or DB reference."""
    try:
        codec = _load(provider, key_id)
        state = CitationKeyState.READY if codec else CitationKeyState.KEY_UNREADABLE
        if codec is None and candidate_id is not None:
            try:
                provider.provision_key(candidate_id)
                codec = _load(provider, candidate_id)
                key_id = candidate_id
                state = (
                    CitationKeyState.READY
                    if codec
                    else CitationKeyState.KEYRING_UNAVAILABLE
                )
            except CitationFingerprintKeyUnavailable:
                state = CitationKeyState.KEYRING_UNAVAILABLE
        future.set_result((codec, state, key_id))
    except Exception:
        future.set_result((None, CitationKeyState.UNAVAILABLE, key_id))


def _snapshot(repository):
    from .citation_trace_repository import load_local_citation_identity_context

    # Never turn a caller's write transaction into an interactive wait.
    if repository.db.get_connection().in_transaction:
        raise RuntimeError("fingerprint_key_acquisition_inside_transaction")
    return (
        load_local_citation_identity_context(repository.db),
        repository.fingerprint_bearing_rows_exist(),
    )


def _start(repository, provider, snapshot, *, replace=None):
    identity, protected = snapshot
    with repository._key_lock:
        if not repository.fingerprint_key_pending or identity is None:
            return None
        current = repository._key_acquisition
        if current is not None and current is not replace:
            return current
        candidate_id = (
            new_opaque_id("citation-key")
            if not protected
            and isinstance(provider, KeyringCitationFingerprintKeyProvider)
            else None
        )
        future = Future()
        thread = threading.Thread(
            target=_acquire_key_candidate,
            args=(provider, identity.fingerprint_key_id, candidate_id, future),
            daemon=True,
            name="citation-key-first-use",
        )
        operation = _KeyAcquisition(identity, future, thread)
        repository._key_acquisition = operation
        thread.start()
        return operation


def _publish(repository, operation):
    """Finite CAS; return a changed identity for one load-only retry."""
    from .citation_trace_repository import load_local_citation_identity_context

    codec, state, key_id = operation.future.result()
    with repository.db.transaction(immediate=True) as cursor:
        if repository._fingerprint_codec is not None:
            return None
        current = load_local_citation_identity_context(repository.db)
        if current is None:
            raise RuntimeError("citation_identity_context_unavailable")
        if current != operation.identity:
            return current
        if codec is not None and key_id != current.fingerprint_key_id:
            if repository.fingerprint_bearing_rows_exist():
                return current
            cursor.execute(
                """UPDATE rag_identity_context SET fingerprint_key_id = ?
                   WHERE context_name = 'default' AND profile_id = ?
                     AND local_authority_id = ? AND fingerprint_key_id = ?""",
                (
                    key_id,
                    current.profile_id,
                    current.local_authority_id,
                    current.fingerprint_key_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("citation_identity_context_mismatch")
            current = current.model_copy(update={"fingerprint_key_id": key_id})
    with repository._key_lock:
        # Publish the pair in one assignment. A ready pair is never replaced.
        if repository._fingerprint_codec is None:
            repository._key_context = (current, codec)
            repository._key_state = state.value
            if state is CitationKeyState.UNAVAILABLE:
                repository._key_acquisition = None
    return None


def acquire_local_fingerprint_codec(repository, key_provider):
    """Acquire synchronously for eager batch callers, outside transactions."""
    if not repository.policy.canonical_writes_enabled:
        return None, CitationKeyState.WRITES_DISABLED
    if repository.identity_context is None:
        return None, CitationKeyState.UNAVAILABLE
    operation = _start(repository, key_provider, _snapshot(repository))
    for attempt in range(2):
        if operation is None:
            break
        operation.thread.join()
        retry_identity = _publish(repository, operation)
        if retry_identity is None:
            break
        if attempt == 0:
            operation = _start(
                repository,
                key_provider,
                (retry_identity, True),
                replace=operation,
            )
    return repository._fingerprint_codec, CitationKeyState(
        repository.fingerprint_key_state
    )


async def run_citation_db_call(database: Any, operation: Callable, /, *args):
    """Retain an issued finite citation DB callback through cancellation."""
    from tldw_chatbook.DB.base_db import run_owned_db_call

    # Memory fixtures own their connection on this thread.
    if database is None or getattr(database, "is_memory_db", False):
        return operation(*args)
    worker = asyncio.create_task(run_owned_db_call(database, operation, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


async def ensure_fingerprint_key_off_loop(repository: Any) -> None:
    """Await key-only work cancellably, retaining each finite DB outcome."""
    if not getattr(repository, "fingerprint_key_pending", False):
        return
    try:
        provider = repository._key_provider
        operation = repository._key_acquisition
        if operation is None:
            snapshot = await run_citation_db_call(repository.db, _snapshot, repository)
            operation = _start(repository, provider, snapshot)
        for attempt in range(2):
            if operation is None:
                return
            # Cancelling this waiter must not cancel the repository's shared Future.
            await asyncio.shield(asyncio.wrap_future(operation.future))
            # Future completion precedes the final thread return. Retain the
            # exact thread as well; it remains observable even after cancellation.
            await asyncio.to_thread(operation.thread.join)
            retry_identity = await run_citation_db_call(
                repository.db,
                _publish,
                repository,
                operation,
            )
            if retry_identity is None:
                return
            if attempt == 0:
                operation = _start(
                    repository,
                    provider,
                    (retry_identity, True),
                    replace=operation,
                )
    except Exception as exc:
        logger.warning(
            "Citation key unavailable; reason=key_acquisition_failure; "
            "exception_type={}",
            type(exc).__name__,
        )


__all__ = [
    "CitationKeyState",
    "acquire_local_fingerprint_codec",
    "ensure_fingerprint_key_off_loop",
    "run_citation_db_call",
]
