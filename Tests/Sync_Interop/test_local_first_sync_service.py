from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope

pytestmark = pytest.mark.asyncio


class FakeLocalFirstServer:
    def __init__(
        self,
        *,
        pull_envelopes=None,
        push_response=None,
        pull_response=None,
        push_error: Exception | None = None,
        pull_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self.pull_envelopes = pull_envelopes or []
        self.push_response = push_response
        self.pull_response = pull_response
        self.push_error = push_error
        self.pull_error = pull_error

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
        if self.push_error is not None:
            raise self.push_error
        if self.push_response is not None:
            return self.push_response
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
        if self.pull_error is not None:
            raise self.pull_error
        if self.pull_response is not None:
            return self.pull_response
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
        self.workspace_links: set[tuple[str, str]] = set()
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


def _repo_with_profile(
    tmp_path,
    *,
    profile_mode="local_first",
    last_error: str | None = None,
) -> SyncStateRepository:
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
        last_error=last_error,
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


async def test_local_first_sync_once_drains_persisted_outbox_and_records_push_failures(tmp_path):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    accepted = builder.build_note_metadata_update(note_id="note-1", status="archived")
    rejected = builder.build_note_metadata_update(note_id="note-2", status="active")
    conflicted = builder.build_note_metadata_update(note_id="note-3", status="draft")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=accepted,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=rejected,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=conflicted,
    )
    server = FakeLocalFirstServer(
        push_response={
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": accepted.client_envelope_id}],
            "rejected": [
                {
                    "client_envelope_id": rejected.client_envelope_id,
                    "error_code": "stale_base",
                    "message": "Local base is stale.",
                }
            ],
            "conflicts": [
                {
                    "client_envelope_id": conflicted.client_envelope_id,
                    "conflict_id": "conflict-1",
                    "message": "Needs manual review.",
                }
            ],
            "next_cursor": "8",
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
    )

    pending_after = repo.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )
    dispatched = repo.list_sync_v2_outbox_entries(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        status="dispatched",
    )

    assert [envelope["client_envelope_id"] for envelope in server.calls[0][3]] == [
        accepted.client_envelope_id,
        rejected.client_envelope_id,
        conflicted.client_envelope_id,
    ]
    assert result["outbox_drained"] == 3
    assert result["outbox_dispatched"] == 1
    assert result["outbox_retained"] == 2
    assert result["rejected_envelopes"][0]["error_code"] == "stale_base"
    assert result["push_conflicts"][0]["conflict_id"] == "conflict-1"
    assert [entry["client_envelope_id"] for entry in dispatched] == [accepted.client_envelope_id]
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        rejected.client_envelope_id,
        conflicted.client_envelope_id,
    ]
    assert pending_after[0]["last_error"]["error_code"] == "stale_base"
    assert pending_after[1]["last_error"]["error_code"] == "conflict"
    assert repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )["last_error"] == "push_partial_failure: stale_base,conflict"


async def test_local_first_sync_once_records_push_failure_without_advancing_cursor(tmp_path):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    outgoing = builder.build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer(push_error=RuntimeError("upstream unavailable"))
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
            outgoing_envelopes=[outgoing],
        )

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert profile["last_error"] == "push_failed: upstream unavailable"
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor == "7"


async def test_local_first_sync_once_records_pull_failure_without_advancing_cursor(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer(pull_error=RuntimeError("server offline"))
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(RuntimeError, match="server offline"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert profile["last_error"] == "pull_failed: server offline"
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor == "7"


async def test_local_first_sync_once_records_apply_failure_without_advancing_cursor(tmp_path):
    dataset_key = generate_dataset_key()
    wrong_key = generate_dataset_key()
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
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(pull_envelopes=[incoming.model_dump(mode="json")])
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": wrong_key},
    )

    with pytest.raises(ValueError, match="Failed to decrypt sync payload"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert profile["last_error"] == "apply_failed: Failed to decrypt sync payload"
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor == "7"
    assert store.note_content == {}


async def test_local_first_sync_once_treats_adapter_rejection_as_failed_apply(tmp_path):
    dataset_key = generate_dataset_key()
    rejected_workspace_envelope = SyncV2Envelope(
        client_envelope_id="remote-device:workspaces:workspace-1:missing-source",
        dataset_id="dataset-1",
        device_id="remote-device",
        domain="workspaces",
        entity_id="workspace-1:missing-source",
        operation="link",
        adapter_version=1,
        stable_key="workspace-1:missing-source",
        payload_clear={"workspace_id": "workspace-1"},
        payload_hash="sha256:missing-source",
    )
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(
        pull_envelopes=[rejected_workspace_envelope.model_dump(mode="json")]
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="apply rejected"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["workspaces"],
        )

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert profile["last_error"] == "apply_rejected: missing_workspace_source_ref"
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor == "7"
    assert store.workspace_links == set()


async def test_local_first_sync_once_success_clears_prior_last_error_without_new_cursor(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path, last_error="pull_failed: server offline")
    server = FakeLocalFirstServer(
        pull_response={
            "dataset_id": "dataset-1",
            "envelopes": [],
            "next_cursor": None,
            "has_more": False,
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
    )
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert result["next_cursor"] == "7"
    assert profile["last_error"] is None
    assert profile["device_id"] == "device-1"
    assert profile["dataset_id"] == "dataset-1"
    assert profile["capabilities"] == {"supported_domains": ["notes"]}
    assert profile["dry_run_metadata"] == {"dry_run": True}


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
