"""Shared local citation conversation-service composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.citation_legacy_migration import (
    CitationLegacyMigrationService,
)
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintKeyProvider,
    KeyringCitationFingerprintKeyProvider,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationTraceRepository,
    load_local_citation_identity_context,
)


def build_local_citation_conversation_service(
    db: Any,
    *,
    sidecar_path: str | Path,
    policy: CitationProvenanceRuntimePolicy | None = None,
    key_provider: CitationFingerprintKeyProvider | None = None,
    repository: CitationTraceRepository | None = None,
    defer_key: bool = False,
) -> tuple[
    ChatConversationService,
    CitationTraceRepository,
    CitationLegacyMigrationService,
]:
    """Compose one policy/identity/key context for local citation reads and writes.

    Args:
        db: The ChaChaNotes database that owns messages and citation records.
        sidecar_path: The legacy RAG-context sidecar read during migration.
        policy: The canonical-write switch; read from config when omitted.
        key_provider: The fingerprint key source; the system keyring by default.
        repository: An already composed repository to reuse as-is.
        defer_key: Compose without touching the key provider. The app's boot
            composition passes ``True`` (TASK-33620.8): the key is acquired,
            or provisioned, on first use through
            ``CitationTraceRepository.ensure_local_fingerprint_key``. Batch
            callers (Chatbook export/import) keep the eager default.

    Returns:
        The conversation service, its citation repository and its migration.
    """

    if repository is None:
        repository = CitationTraceRepository(
            db,
            policy=policy or CitationProvenanceRuntimePolicy.from_config(),
            identity_context=load_local_citation_identity_context(db),
            fingerprint_codec=None,
            key_provider=key_provider or KeyringCitationFingerprintKeyProvider(),
        )
        if not defer_key:
            repository.ensure_local_fingerprint_key()
    migration = CitationLegacyMigrationService(
        db=db,
        repository=repository,
        sidecar_path=sidecar_path,
    )
    service = ChatConversationService(
        db,
        rag_context_store_path=sidecar_path,
        citation_legacy_migration=migration,
    )
    return service, repository, migration


__all__ = ["build_local_citation_conversation_service"]
