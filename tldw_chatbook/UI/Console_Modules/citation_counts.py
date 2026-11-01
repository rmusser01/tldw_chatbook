"""Finite citation-footer database work owned by its Console view worker."""

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
    """Read counts after readiness and mark genuinely unverifiable records.

    Runs on the owned worker thread (an in-memory database runs where it
    lives). Interactive key acquisition has already left DB custody.
    """
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
    from ...Chat.citation_key_first_use import (
        ensure_fingerprint_key_off_loop,
        run_citation_db_call,
    )

    await ensure_fingerprint_key_off_loop(repository)
    return await run_citation_db_call(
        getattr(repository, "db", None),
        _read_with_first_use_key,
        reader,
        repository,
        eligible,
    )
