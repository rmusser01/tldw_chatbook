"""Finite citation-footer database work owned by its Console view worker."""

import asyncio
from collections.abc import Callable
from typing import Any

#: TASK-33620.8 (AC#4): a saved reply whose citation record exists but cannot
#: be verified, because citation records are off or the fingerprint key is
#: unavailable. The transcript shows 'Cited sources unavailable' for it, and
#: the Sources window names the cause and a next step.
CITATIONS_UNVERIFIABLE = -1

#: Repository key states under which an existing record cannot be verified.
_UNVERIFIABLE_KEY_STATES = frozenset(
    {"writes_disabled", "keyring_unavailable", "key_unreadable", "unavailable"}
)


def _read_with_first_use_key(
    reader: Callable[..., dict[str, int]],
    repository: Any,
    eligible: tuple[tuple[str, str, str, str], ...],
) -> dict[str, int]:
    """Acquire the deferred key, read counts, and mark unverifiable records.

    Runs on the owned worker thread (an in-memory database runs where it
    lives), so the first-use keyring call never blocks the UI loop.
    """
    if getattr(repository, "fingerprint_key_pending", False):
        repository.ensure_local_fingerprint_key()
    counts = reader(repository, eligible)
    if getattr(repository, "fingerprint_key_state", "ready") not in (
        _UNVERIFIABLE_KEY_STATES
    ):
        return counts
    uncounted = [item for item in eligible if counts.get(item[0], 0) == 0]
    owner_ids = getattr(repository, "active_owner_candidate_message_ids", None)
    if not uncounted or not callable(owner_ids):
        return counts
    try:
        owned = owner_ids([persisted_id for _native, persisted_id, *_ in uncounted])
    except Exception:  # noqa: BLE001 - an unreadable owner set marks nothing
        return counts
    for native_id, persisted_id, *_ in uncounted:
        if persisted_id in owned:
            counts[native_id] = CITATIONS_UNVERIFIABLE
    return counts


async def read_citation_counts(
    repository: Any,
    reader: Callable[..., dict[str, int]],
    eligible: tuple[tuple[str, str, str, str], ...],
) -> dict[str, int]:
    """Run the captured reader, retiring stock file-backed work before cancellation."""
    from ...Chat.citation_trace_repository import CitationTraceRepository
    from ...DB.base_db import run_owned_db_call
    from ...DB.ChaChaNotes_DB import CharactersRAGDB

    database = getattr(repository, "db", None)
    operation = run_owned_db_call(
        database, _read_with_first_use_key, reader, repository, eligible
    )
    if (
        type(repository) is not CitationTraceRepository
        or type(database) is not CharactersRAGDB
        or database.is_memory_db
    ):
        return await operation
    worker = asyncio.Task(operation)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - cancellation wins.
                break
        if not worker.cancelled():
            worker.exception()
        raise
