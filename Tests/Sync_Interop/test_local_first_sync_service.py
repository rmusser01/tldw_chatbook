from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

pytestmark = pytest.mark.asyncio


class FakeLocalFirstServer:
    def __init__(self, *, pull_envelopes=None) -> None:
        self.calls: list[tuple] = []
        self.pull_envelopes = pull_envelopes or []

    async def push_v2_envelopes(
        self,
        *,
        dataset_id,
        device_id,
        envelopes,
        idempotency_key=None,
        last_known_cursor=None,
    ):
        self.calls.append(
            ("push", dataset_id, device_id, envelopes, idempotency_key, last_known_cursor)
        )
        return {
            "dataset_id": dataset_id,
            "accepted": [{"client_envelope_id": "outgoing-1"}],
            "next_cursor": "8",
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
        self.calls.append(
            ("pull", dataset_id, device_id, cursor, domains, page_size, include_own_changes)
        )
        return {
            "dataset_id": dataset_id,
            "envelopes": self.pull_envelopes,
            "next_cursor": "9",
            "has_more": False,
        }


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


def _repo_with_profile(tmp_path, *, profile_mode="local_first") -> SyncStateRepository:
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode=profile_mode,
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "7"},
        capabilities={"supported_domains": ["notes"]},
        dry_run_metadata={"dry_run": True},
    )
    repo.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
        cursor="7",
    )
    return repo


async def test_local_first_sync_once_pushes_pulls_applies_and_persists_cursor(tmp_path):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-2",
        dataset_key=dataset_key,
    )
    incoming = builder.build_note_upsert(
        note_id="note-1",
        title="Remote title",
        body="remote private body",
        status="active",
    )
    outgoing = builder.build_note_metadata_update(note_id="note-2", status="archived")
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(pull_envelopes=[incoming.model_dump(mode="json")])
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
        outgoing_envelopes=[outgoing],
        page_size=25,
    )

    assert result["pushed_envelopes"] == 1
    assert result["pulled_envelopes"] == 1
    assert result["applied_envelopes"] == 1
    assert result["next_cursor"] == "9"
    assert store.note_content["note-1"] == {
        "body": "remote private body",
        "title": "Remote title",
    }
    assert server.calls[0][0] == "push"
    assert server.calls[0][3][0]["client_envelope_id"] == outgoing.client_envelope_id
    assert server.calls[1] == ("pull", "dataset-1", "device-1", "7", ["notes"], 25, False)
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor == "9"
    assert repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )["dataset_cursors"]["sync_v2"] == "9"


async def test_local_first_sync_once_requires_local_first_profile(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path, profile_mode="server_frontend")
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="local_first"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )

    assert server.calls == []


async def test_local_first_sync_once_requires_profile_device_dataset_and_dataset_key(tmp_path):
    dataset_key = generate_dataset_key()
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first",
        device_id=None,
        dataset_id="dataset-1",
        dataset_cursors={},
    )
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="device_id and dataset_id"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )

    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={},
    )
    service_without_key = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={},
    )

    with pytest.raises(ValueError, match="dataset key"):
        await service_without_key.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )

    assert server.calls == []
