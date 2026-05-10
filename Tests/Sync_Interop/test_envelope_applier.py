from __future__ import annotations

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder


class RecordingLocalStore:
    def __init__(self) -> None:
        self.note_hashes: dict[str, str] = {}
        self.note_content: dict[str, dict] = {}
        self.note_metadata: dict[str, dict] = {}
        self.chat_hashes: dict[str, str] = {}
        self.chat_messages: dict[str, dict] = {}
        self.workspace_links: set[tuple[str, str]] = set()
        self.source_cache: dict[str, dict] = {}
        self.conflicts: list[dict] = []

    def get_note_content_hash(self, note_id: str) -> str | None:
        return self.note_hashes.get(note_id)

    def upsert_note_content(self, note_id: str, payload: dict, payload_hash: str) -> None:
        self.note_content[note_id] = payload
        self.note_hashes[note_id] = payload_hash

    def upsert_note_metadata(self, note_id: str, metadata: dict) -> None:
        self.note_metadata[note_id] = metadata

    def get_chat_message_hash(self, stable_key: str) -> str | None:
        return self.chat_hashes.get(stable_key)

    def append_chat_message(self, stable_key: str, payload: dict, payload_hash: str) -> None:
        self.chat_messages[stable_key] = payload
        self.chat_hashes[stable_key] = payload_hash

    def link_workspace_source(self, workspace_id: str, source_id: str) -> None:
        self.workspace_links.add((workspace_id, source_id))

    def unlink_workspace_source(self, workspace_id: str, source_id: str) -> None:
        self.workspace_links.discard((workspace_id, source_id))

    def upsert_source_cache(self, stable_key: str, payload: dict, metadata: dict) -> None:
        self.source_cache[stable_key] = {"payload": payload, "metadata": metadata}

    def record_conflict(self, conflict: dict) -> None:
        self.conflicts.append(conflict)


def test_note_applier_records_conflict_instead_of_overwriting_divergent_content() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)
    store = RecordingLocalStore()
    store.note_hashes["note-1"] = "sha256:local-dirty"
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Remote",
        body="remote body",
        base_version="sha256:remote-base",
    )
    result = applier.apply(envelope)

    assert result["status"] == "conflict"
    assert store.note_content == {}
    assert store.conflicts[0]["domain"] == "notes"
    assert store.conflicts[0]["conflict_type"] == "encrypted_content_edit"


def test_note_applier_merges_safe_metadata_without_content_overwrite() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    envelope = builder.build_note_metadata_update(
        note_id="note-1",
        status="archived",
        tag_ids=["tag-1"],
    )
    result = applier.apply(envelope)

    assert result["status"] == "applied"
    assert store.note_metadata["note-1"] == {"status": "archived", "tag_ids": ["tag-1"]}
    assert store.note_content == {}


def test_chat_applier_appends_by_stable_id_and_conflicts_on_hash_mismatch() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="user",
        content="hello",
    )
    first = applier.apply(envelope)
    second = applier.apply(envelope)
    changed = envelope.model_copy(update={"payload_hash": "sha256:other"})
    conflict = applier.apply(changed)

    assert first["status"] == "applied"
    assert second["status"] == "noop"
    assert store.chat_messages["conversation-1:message-1"] == {"content": "hello", "role": "user"}
    assert conflict["status"] == "conflict"
    assert store.conflicts[-1]["domain"] == "chat"


def test_workspace_and_source_cache_appliers_route_to_local_store() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    link = builder.build_workspace_source_ref(
        workspace_id="workspace-1",
        source_id="source-1",
        operation="link",
    )
    cache = builder.build_source_cache(
        source_id="source-1",
        content_hash="sha256:content",
        cache_kind="transcript",
        content="private transcript",
    )

    assert applier.apply(link)["status"] == "applied"
    assert ("workspace-1", "source-1") in store.workspace_links
    assert applier.apply(cache)["status"] == "applied"
    assert store.source_cache["source-1:sha256:content"]["payload"] == {
        "content": "private transcript"
    }
