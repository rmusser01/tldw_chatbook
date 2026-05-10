from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key, wrap_dataset_key_for_recovery
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.restore_service import SyncRestoreService
from tldw_chatbook.tldw_api import SyncV2Envelope

pytestmark = pytest.mark.asyncio


class FakeRestoreServer:
    def __init__(self, *, envelopes=None, conflicts=None, recovery_records=None) -> None:
        self.calls: list[tuple] = []
        self.envelopes = envelopes or []
        self.conflicts = conflicts or []
        self.recovery_records = recovery_records or []

    async def get_v2_restore_manifest(self, *, dataset_ids=None, domains=None):
        self.calls.append(("manifest", dataset_ids, domains))
        return {
            "datasets": [
                {
                    "dataset_id": "locked-dataset",
                    "scope_type": "personal",
                    "encryption_policy": "client_private_v1",
                    "domains": ["notes"],
                    "approximate_counts": {"notes": 3},
                    "byte_estimates": {"small": 128},
                    "unresolved_conflicts": 2,
                    "attachment_size_classes": {"small": 1},
                    "key_recovery_available": False,
                },
                {
                    "dataset_id": "recoverable-dataset",
                    "scope_type": "personal",
                    "encryption_policy": "client_private_v1",
                    "domains": ["notes", "chat"],
                    "approximate_counts": {"notes": 1},
                    "byte_estimates": {"small": 64},
                    "unresolved_conflicts": 0,
                    "attachment_size_classes": {"small": 0},
                    "key_recovery_available": True,
                },
            ],
            "devices": [{"device_id": "device-1", "display_name": "Laptop"}],
            "generated_at": "2026-05-10T00:00:00Z",
        }

    async def pull_v2_envelopes(
        self,
        *,
        dataset_id,
        device_id,
        cursor=None,
        domains=None,
        page_size=None,
        include_own_changes=False,
    ):
        self.calls.append(("pull", dataset_id, device_id, cursor, domains, page_size, include_own_changes))
        return {"dataset_id": dataset_id, "envelopes": self.envelopes, "next_cursor": "cursor-2", "has_more": False}

    async def list_v2_conflicts(self, *, dataset_id, status="unresolved"):
        self.calls.append(("conflicts", dataset_id, status))
        return self.conflicts

    async def list_v2_recovery_bundles(
        self,
        *,
        dataset_id,
        device_id=None,
        key_purpose="dataset_recovery",
    ):
        self.calls.append(("recovery_bundles", dataset_id, device_id, key_purpose))
        return {"dataset_id": dataset_id, "key_records": self.recovery_records}


class RecordingLocalStore:
    def __init__(self) -> None:
        self.note_hashes: dict[str, str] = {}
        self.note_content: dict[str, dict] = {}
        self.note_metadata: dict[str, dict] = {}
        self.conflicts: list[dict] = []

    def get_note_content_hash(self, note_id: str) -> str | None:
        return self.note_hashes.get(note_id)

    def upsert_note_content(self, note_id: str, payload: dict, payload_hash: str) -> None:
        self.note_content[note_id] = payload
        self.note_hashes[note_id] = payload_hash

    def upsert_note_metadata(self, note_id: str, metadata: dict) -> None:
        self.note_metadata[note_id] = metadata

    def record_conflict(self, conflict: dict) -> None:
        self.conflicts.append(conflict)


async def test_restore_service_previews_locked_and_recoverable_datasets():
    service = SyncRestoreService(
        server_service=FakeRestoreServer(),
        local_store=RecordingLocalStore(),
        dataset_keys={"local-key-dataset": generate_dataset_key()},
    )

    preview = await service.preview_restore(dataset_ids=["locked-dataset"], domains=["notes"])

    assert preview["devices"] == [{"device_id": "device-1", "display_name": "Laptop"}]
    assert preview["datasets"][0]["dataset_id"] == "locked-dataset"
    assert preview["datasets"][0]["restore_status"] == "locked"
    assert preview["datasets"][0]["unresolved_conflicts"] == 2
    assert preview["datasets"][0]["attachment_size_classes"] == {"small": 1}
    assert preview["datasets"][1]["dataset_id"] == "recoverable-dataset"
    assert preview["datasets"][1]["restore_status"] == "recovery_available"


async def test_restore_selection_filters_pull_and_decrypts_before_local_apply():
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="recoverable-dataset", device_id="device-1", dataset_key=dataset_key)
    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Restored",
        body="private restored body",
        status="active",
    )
    store = RecordingLocalStore()
    server = FakeRestoreServer(envelopes=[envelope.model_dump(mode="json")])
    service = SyncRestoreService(
        server_service=server,
        local_store=store,
        dataset_keys={"recoverable-dataset": dataset_key},
    )

    result = await service.restore_selection(
        dataset_id="recoverable-dataset",
        device_id="device-1",
        domains=["notes"],
        page_size=25,
    )

    assert server.calls[-1] == ("pull", "recoverable-dataset", "device-1", None, ["notes"], 25, False)
    assert result["applied"] == 1
    assert store.note_content["note-1"] == {"body": "private restored body", "title": "Restored"}
    assert store.note_metadata["note-1"] == {"status": "active"}


async def test_restore_selection_recovers_dataset_key_with_recovery_secret():
    dataset_key = generate_dataset_key()
    recovery_bundle = wrap_dataset_key_for_recovery(
        dataset_key,
        recovery_secret="correct horse battery staple",
        recovery_hint="laptop",
    )
    builder = SyncEnvelopeBuilder(dataset_id="recoverable-dataset", device_id="device-1", dataset_key=dataset_key)
    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Restored",
        body="private restored body",
        status="active",
    )
    store = RecordingLocalStore()
    server = FakeRestoreServer(
        envelopes=[envelope.model_dump(mode="json")],
        recovery_records=[
            {
                **recovery_bundle.model_dump(mode="json"),
                "key_record_id": "key-record-1",
                "dataset_id": "recoverable-dataset",
                "device_id": "device-1",
                "created_at": "2026-05-10T00:00:00Z",
                "revoked_at": None,
            }
        ],
    )
    service = SyncRestoreService(server_service=server, local_store=store)

    result = await service.restore_selection(
        dataset_id="recoverable-dataset",
        device_id="device-1",
        domains=["notes"],
        recovery_secret="correct horse battery staple",
    )

    assert server.calls[0] == (
        "recovery_bundles",
        "recoverable-dataset",
        None,
        "dataset_recovery",
    )
    assert server.calls[1] == ("pull", "recoverable-dataset", "device-1", None, ["notes"], None, False)
    assert result["applied"] == 1
    assert store.note_content["note-1"] == {"body": "private restored body", "title": "Restored"}
    assert "correct horse battery staple" not in str(result)
    assert "wrapped_key_blob" not in str(result)
    assert "kdf_metadata" not in str(result)
    assert "dataset_key" not in str(result)


async def test_restore_selection_tries_later_recovery_bundle_when_first_fails():
    dataset_key = generate_dataset_key()
    stale_bundle = wrap_dataset_key_for_recovery(
        generate_dataset_key(),
        recovery_secret="old recovery secret",
        recovery_hint="old laptop",
    )
    valid_bundle = wrap_dataset_key_for_recovery(
        dataset_key,
        recovery_secret="correct horse battery staple",
        recovery_hint="current laptop",
    )
    builder = SyncEnvelopeBuilder(
        dataset_id="recoverable-dataset",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Restored",
        body="private restored body",
        status="active",
    )
    store = RecordingLocalStore()
    server = FakeRestoreServer(
        envelopes=[envelope.model_dump(mode="json")],
        recovery_records=[
            {
                **stale_bundle.model_dump(mode="json"),
                "key_record_id": "key-record-old",
                "dataset_id": "recoverable-dataset",
                "device_id": "old-device",
                "created_at": "2026-05-09T00:00:00Z",
                "revoked_at": None,
            },
            {
                **valid_bundle.model_dump(mode="json"),
                "key_record_id": "key-record-current",
                "dataset_id": "recoverable-dataset",
                "device_id": "device-1",
                "created_at": "2026-05-10T00:00:00Z",
                "revoked_at": None,
            },
        ],
    )
    service = SyncRestoreService(server_service=server, local_store=store)

    result = await service.restore_selection(
        dataset_id="recoverable-dataset",
        device_id="device-1",
        domains=["notes"],
        recovery_secret="correct horse battery staple",
    )

    assert server.calls[0] == (
        "recovery_bundles",
        "recoverable-dataset",
        None,
        "dataset_recovery",
    )
    assert server.calls[1] == (
        "pull",
        "recoverable-dataset",
        "device-1",
        None,
        ["notes"],
        None,
        False,
    )
    assert result["applied"] == 1
    assert store.note_content["note-1"] == {"body": "private restored body", "title": "Restored"}


async def test_restore_selection_surfaces_adapter_rejections():
    dataset_key = generate_dataset_key()
    rejected_workspace_envelope = SyncV2Envelope(
        client_envelope_id="remote-device:workspaces:workspace-1:missing-source",
        dataset_id="recoverable-dataset",
        device_id="remote-device",
        domain="workspaces",
        entity_id="workspace-1:missing-source",
        operation="link",
        adapter_version=1,
        stable_key="workspace-1:missing-source",
        payload_clear={"workspace_id": "workspace-1"},
        payload_hash="sha256:missing-source",
    )
    store = RecordingLocalStore()
    server = FakeRestoreServer(
        envelopes=[rejected_workspace_envelope.model_dump(mode="json")]
    )
    service = SyncRestoreService(
        server_service=server,
        local_store=store,
        dataset_keys={"recoverable-dataset": dataset_key},
    )

    result = await service.restore_selection(
        dataset_id="recoverable-dataset",
        device_id="device-1",
        domains=["workspaces"],
    )

    assert result["applied"] == 0
    assert result["conflicts"] == []
    assert result["rejected"] == [
        {
            "status": "rejected",
            "error_code": "missing_workspace_source_ref",
            "client_envelope_id": rejected_workspace_envelope.client_envelope_id,
            "domain": "workspaces",
            "entity_id": "workspace-1:missing-source",
            "stable_key": "workspace-1:missing-source",
            "operation": "link",
        }
    ]
    assert result["results"] == [
        {"status": "rejected", "error_code": "missing_workspace_source_ref"}
    ]


async def test_restore_selection_recovery_failure_does_not_pull_or_apply():
    dataset_key = generate_dataset_key()
    recovery_bundle = wrap_dataset_key_for_recovery(
        dataset_key,
        recovery_secret="correct horse battery staple",
    )
    store = RecordingLocalStore()
    server = FakeRestoreServer(
        envelopes=[{"client_envelope_id": "should-not-pull"}],
        recovery_records=[
            {
                **recovery_bundle.model_dump(mode="json"),
                "key_record_id": "key-record-1",
                "dataset_id": "recoverable-dataset",
                "device_id": "device-1",
                "created_at": "2026-05-10T00:00:00Z",
                "revoked_at": None,
            }
        ],
    )
    service = SyncRestoreService(server_service=server, local_store=store)

    with pytest.raises(ValueError, match="Failed to recover dataset key"):
        await service.restore_selection(
            dataset_id="recoverable-dataset",
            device_id="device-1",
            domains=["notes"],
            recovery_secret="wrong secret",
        )

    assert server.calls == [("recovery_bundles", "recoverable-dataset", None, "dataset_recovery")]
    assert store.note_content == {}
    assert store.note_metadata == {}


async def test_restore_service_keeps_unresolved_conflicts_visible():
    server = FakeRestoreServer(
        conflicts=[
            {
                "conflict_id": "conflict-1",
                "dataset_id": "dataset-1",
                "domain": "notes",
                "entity_id": "note-1",
                "conflict_type": "encrypted_content_edit",
                "status": "unresolved",
            }
        ]
    )
    service = SyncRestoreService(server_service=server, local_store=RecordingLocalStore())

    conflicts = await service.list_conflicts(dataset_id="dataset-1")

    assert conflicts[0]["status"] == "unresolved"
    assert server.calls[-1] == ("conflicts", "dataset-1", "unresolved")
