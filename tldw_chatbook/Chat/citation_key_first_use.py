"""First-use custody of the local citation fingerprint key (TASK-33620.8).

Owner decision 2026-10-27: citation records persist by default, and the key
that binds a saved record to its reply is created in the system keyring the
first time it is needed. Boot never touches the keyring. A keyring call can
take tens of milliseconds, and on some desktops it can prompt to unlock, so
the app composes its citation repository without the key. Each caller that
needs the key acquires it off the UI loop: the Console capture paths before a
send, the citation-count reader, the Sources window, and the deferred startup
units.

Provisioning keeps the factory's original guard (TASK-553.17). A missing key
is created only inside a write transaction that proves no fingerprint-bearing
row exists. A key that saved records depend on is never silently replaced:
the records stay unverifiable until the original key is readable again.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    CitationFingerprintKeyProvider,
    CitationFingerprintKeyUnavailable,
    KeyringCitationFingerprintKeyProvider,
    load_fingerprint_codec,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository


class CitationKeyState(str, Enum):
    """Why local citation records can or cannot be written and verified."""

    PENDING = "pending"
    READY = "ready"
    WRITES_DISABLED = "writes_disabled"
    KEYRING_UNAVAILABLE = "keyring_unavailable"
    KEY_UNREADABLE = "key_unreadable"
    UNAVAILABLE = "unavailable"


def _load(
    provider: CitationFingerprintKeyProvider, key_id: str
) -> CitationFingerprintCodec | None:
    try:
        return load_fingerprint_codec(provider, key_id)
    except CitationFingerprintKeyUnavailable:
        return None


def acquire_local_fingerprint_codec(
    repository: CitationTraceRepository,
    key_provider: CitationFingerprintKeyProvider,
) -> tuple[CitationFingerprintCodec | None, CitationKeyState]:
    """Load the fingerprint key, or create it when that is provably safe.

    Blocking: it calls the keyring and, to provision, holds a database write
    transaction. Run it off the UI loop.

    Args:
        repository: The repository whose identity and rows decide the outcome.
        key_provider: Loads (and, for the system keyring, creates) the key.

    Returns:
        The codec (``None`` unless the state is ``READY``) and the state.
    """

    if not repository.policy.canonical_writes_enabled:
        return None, CitationKeyState.WRITES_DISABLED
    identity = repository.identity_context
    if identity is None:
        return None, CitationKeyState.UNAVAILABLE
    key_id = identity.fingerprint_key_id
    codec = _load(key_provider, key_id)
    if codec is not None:
        return codec, CitationKeyState.READY
    if not isinstance(key_provider, KeyringCitationFingerprintKeyProvider):
        return None, CitationKeyState.KEY_UNREADABLE
    try:
        return _provision_under_write_guard(repository, key_provider, key_id)
    except Exception as exc:  # noqa: BLE001 - fail closed, never half-provision
        logger.warning(
            "Citation key unavailable; reason=key_provisioning_failure; "
            "exception_type={}",
            type(exc).__name__,
        )
        return None, CitationKeyState.UNAVAILABLE


def _provision_under_write_guard(
    repository: CitationTraceRepository,
    key_provider: KeyringCitationFingerprintKeyProvider,
    key_id: str,
) -> tuple[CitationFingerprintCodec | None, CitationKeyState]:
    with repository.db.transaction() as cursor:
        # Take the write lock first: no other writer can add a row that
        # depends on this key between the check below and the provisioning.
        cursor.execute(
            """
            UPDATE rag_identity_context
            SET fingerprint_key_id = fingerprint_key_id
            WHERE context_name = 'default'
            """
        )
        codec = _load(key_provider, key_id)
        if codec is not None:
            return codec, CitationKeyState.READY
        if repository.fingerprint_bearing_rows_exist():
            return None, CitationKeyState.KEY_UNREADABLE
        try:
            key_provider.provision_key(key_id)
        except CitationFingerprintKeyUnavailable:
            return None, CitationKeyState.KEYRING_UNAVAILABLE
    codec = _load(key_provider, key_id)
    if codec is None:
        return None, CitationKeyState.KEYRING_UNAVAILABLE
    return codec, CitationKeyState.READY


async def ensure_fingerprint_key_off_loop(repository: Any) -> None:
    """Acquire a pending fingerprint key without blocking the UI loop.

    A file-backed database is read on a worker thread through the owned-call
    seam; an in-memory one is per-thread, so it is read where it lives.

    Args:
        repository: The app's citation repository, or ``None``.
    """

    if not getattr(repository, "fingerprint_key_pending", False):
        return
    ensure = repository.ensure_local_fingerprint_key
    database = getattr(repository, "db", None)
    try:
        if database is None or getattr(database, "is_memory_db", True):
            ensure()
            return
        from tldw_chatbook.DB.base_db import run_owned_db_call

        await run_owned_db_call(database, ensure)
    except Exception as exc:  # noqa: BLE001 - capture falls back to no record
        logger.warning(
            "Citation key unavailable; reason=key_acquisition_failure; "
            "exception_type={}",
            type(exc).__name__,
        )


__all__ = [
    "CitationKeyState",
    "acquire_local_fingerprint_codec",
    "ensure_fingerprint_key_off_loop",
]
