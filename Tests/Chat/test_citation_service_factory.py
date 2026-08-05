from __future__ import annotations

import base64

import pytest

from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_service_factory import (
    build_local_citation_conversation_service,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CITATION_FINGERPRINT_KEYRING_SERVICE,
    KeyringCitationFingerprintKeyProvider,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    load_local_citation_identity_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _StaticKeyProvider:
    def __init__(self, key: bytes) -> None:
        self.key = key
        self.calls = 0

    def load_key(self, _fingerprint_key_id: str) -> bytes:
        self.calls += 1
        return self.key


class _SecureKeyring:
    __module__ = "keyring.backends.macOS"
    priority = 5

    def __init__(self) -> None:
        self.value: str | None = None
        self.set_calls: list[tuple[str, str, str]] = []

    def get_password(self, _service: str, _account: str) -> str | None:
        return self.value

    def set_password(self, service: str, account: str, value: str) -> None:
        self.set_calls.append((service, account, value))
        self.value = value


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


def test_enabled_factory_provisions_and_reuses_key_for_fresh_database(
    db,
    tmp_path,
) -> None:
    backend = _SecureKeyring()
    provider = KeyringCitationFingerprintKeyProvider(keyring_backend=backend)
    policy = CitationProvenanceRuntimePolicy(canonical_writes_enabled=True)

    _, first_repository, _ = build_local_citation_conversation_service(
        db,
        sidecar_path=tmp_path / "chat_rag_context.json",
        policy=policy,
        key_provider=provider,
    )
    first_secret = provider.load_key(
        load_local_citation_identity_context(db).fingerprint_key_id
    )
    _, second_repository, _ = build_local_citation_conversation_service(
        db,
        sidecar_path=tmp_path / "chat_rag_context.json",
        policy=policy,
        key_provider=provider,
    )

    assert first_repository.local_citation_writes_ready is True
    assert second_repository.local_citation_writes_ready is True
    assert len(first_secret) == 32
    assert backend.set_calls == [
        (
            CITATION_FINGERPRINT_KEYRING_SERVICE,
            load_local_citation_identity_context(db).fingerprint_key_id,
            base64.b64encode(first_secret).decode("ascii"),
        )
    ]


def test_enabled_factory_does_not_replace_missing_key_for_existing_rows(
    db,
    tmp_path,
) -> None:
    identity = load_local_citation_identity_context(db)
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_payload_tombstones VALUES (
                ?, 'local_payload_v1', 'prior-payload', 'prior-scope',
                'source_revoked', 'policy-1',
                '2026-07-27T00:00:00Z', '2027-07-27T00:00:00Z'
            )
            """,
            (identity.profile_id,),
        )
    backend = _SecureKeyring()

    _, repository, _ = build_local_citation_conversation_service(
        db,
        sidecar_path=tmp_path / "chat_rag_context.json",
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        key_provider=KeyringCitationFingerprintKeyProvider(
            keyring_backend=backend
        ),
    )

    assert repository.local_citation_writes_ready is False
    assert backend.set_calls == []
