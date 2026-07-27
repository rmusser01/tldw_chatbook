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
    CitationFingerprintKeyUnavailable,
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
) -> tuple[
    ChatConversationService,
    CitationTraceRepository,
    CitationLegacyMigrationService,
]:
    """Compose one policy/identity/key context for local citation reads and writes."""

    if repository is None:
        runtime_policy = policy or CitationProvenanceRuntimePolicy.from_config()
        identity_context = load_local_citation_identity_context(db)
        runtime_key_provider = key_provider or KeyringCitationFingerprintKeyProvider()
        repository = CitationTraceRepository.from_key_provider(
            db,
            policy=runtime_policy,
            identity_context=identity_context,
            key_provider=runtime_key_provider,
        )
        if (
            runtime_policy.canonical_writes_enabled
            and identity_context is not None
            and isinstance(
                runtime_key_provider,
                KeyringCitationFingerprintKeyProvider,
            )
            and not repository.local_citation_writes_ready
        ):
            with db.transaction() as cursor:
                cursor.execute(
                    """
                    UPDATE rag_identity_context
                    SET fingerprint_key_id = fingerprint_key_id
                    WHERE context_name = 'default'
                    """
                )
                repository = CitationTraceRepository.from_key_provider(
                    db,
                    policy=runtime_policy,
                    identity_context=identity_context,
                    key_provider=runtime_key_provider,
                )
                if (
                    not repository.local_citation_writes_ready
                    and not repository.fingerprint_bearing_rows_exist()
                ):
                    try:
                        runtime_key_provider.provision_key(
                            identity_context.fingerprint_key_id
                        )
                    except CitationFingerprintKeyUnavailable:
                        pass
                    else:
                        repository = CitationTraceRepository.from_key_provider(
                            db,
                            policy=runtime_policy,
                            identity_context=identity_context,
                            key_provider=runtime_key_provider,
                        )
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
