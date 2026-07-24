from __future__ import annotations

import pytest

from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_service_factory import (
    build_local_citation_conversation_service,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _StaticKeyProvider:
    def __init__(self, key: bytes) -> None:
        self.key = key
        self.calls = 0

    def load_key(self, _fingerprint_key_id: str) -> bytes:
        self.calls += 1
        return self.key


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(
        tmp_path / "citation-factory.sqlite",
        client_id="citation-factory-test",
    )
    yield database
    database.close_connection()


def test_enabled_factory_shares_ready_repository_and_blocks_sidecar_writes(
    db,
    tmp_path,
) -> None:
    provider = _StaticKeyProvider(b"f" * 32)

    service, repository, migration = build_local_citation_conversation_service(
        db,
        sidecar_path=tmp_path / "chat_rag_context.json",
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        key_provider=provider,
    )

    assert provider.calls == 1
    assert migration.repository is repository
    assert service.citation_legacy_migration is migration
    assert migration.ready is True
    with pytest.raises(RuntimeError, match="legacy_rag_context_writes_disabled"):
        service.record_message_rag_context("conversation", "message")


def test_disabled_factory_does_not_load_key_and_retains_sidecar_writer(
    db,
    tmp_path,
) -> None:
    provider = _StaticKeyProvider(b"f" * 32)
    conversation_id = db.add_conversation(
        {"title": "Recovery mode", "character_id": None}
    )
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Legacy answer",
        }
    )

    service, repository, migration = build_local_citation_conversation_service(
        db,
        sidecar_path=tmp_path / "chat_rag_context.json",
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        key_provider=provider,
    )
    record = service.record_message_rag_context(conversation_id, message_id)

    assert provider.calls == 0
    assert migration.repository is repository
    assert migration.ready is False
    assert record["message_id"] == message_id
