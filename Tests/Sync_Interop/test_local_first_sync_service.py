from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileManifest,
    ProfileProposal,
    ProfileScope,
    ProposalOperation,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.reconciliation import (
    CanonicalBootstrapSnapshot,
    build_reconciliation_plan,
)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    RecordMutation,
)
from tldw_chatbook.Personal_Context.sync_outbox import ProfileSyncOutbox
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.personal_context_adapter import (
    PersonalContextSyncAdapter,
)
from tldw_chatbook.Sync_Interop.personal_context_dispatcher import (
    PersonalContextOutboxDispatcher,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope, SyncV2PushResponse

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
        self.personal_context_complete_pushes = 0
        self.personal_context_complete_pulls = 0
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
        domains=None,
    ):
        self.calls.append(
            (
                "push",
                dataset_id,
                device_id,
                envelopes,
                idempotency_key,
                last_known_cursor,
                domains,
            )
        )
        if self.push_error is not None:
            raise self.push_error
        if self.push_response is not None:
            return self.push_response
        return {
            "dataset_id": dataset_id,
            "accepted": [
                {"client_envelope_id": envelope["client_envelope_id"]}
                for envelope in envelopes
            ],
            "next_cursor": "8",
        }

    async def _push_v2_personal_context_complete(self, **kwargs):
        self.personal_context_complete_pushes += 1
        return await self.push_v2_envelopes(**kwargs)

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
            (
                "pull",
                dataset_id,
                device_id,
                cursor,
                domains,
                page_size,
                include_own_changes,
            )
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

    async def _pull_v2_personal_context_complete(self, **kwargs):
        self.personal_context_complete_pulls += 1
        return await self.pull_v2_envelopes(**kwargs)


class RecordingLocalStore:
    def __init__(self) -> None:
        self.note_hashes: dict[str, str] = {}
        self.note_content: dict[str, dict] = {}
        self.note_metadata: dict[str, dict] = {}
        self.workspace_links: set[tuple[str, str]] = set()
        self.conflicts: list[dict] = []

    def get_note_content_hash(self, note_id: str) -> str | None:
        return self.note_hashes.get(note_id)

    def upsert_note_content(
        self, note_id: str, payload: dict, payload_hash: str
    ) -> None:
        self.note_content[note_id] = payload
        self.note_hashes[note_id] = payload_hash

    def upsert_note_metadata(self, note_id: str, metadata: dict) -> None:
        self.note_metadata[note_id] = metadata

    def record_conflict(self, conflict: dict) -> None:
        self.conflicts.append(conflict)


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _repo_with_profile(
    tmp_path,
    *,
    profile_mode="local_first",
    last_error: str | None = None,
    capabilities: dict | None = None,
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
        capabilities=capabilities or {"supported_domains": ["notes"]},
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


async def test_local_first_sync_service_observes_key_added_to_empty_shared_cache(
    tmp_path: Path,
) -> None:
    """Retain an empty shared key cache so later key loads reach Sync.

    Args:
        tmp_path: Private root for the real file-backed Sync repository.
    """
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer()
    shared_dataset_keys: dict[str, bytes] = {}
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys=shared_dataset_keys,
    )

    assert service.dataset_keys is shared_dataset_keys
    shared_dataset_keys["dataset-1"] = dataset_key

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
    )

    assert result["pulled_envelopes"] == 0
    assert server.calls[0][0] == "pull"


async def test_local_first_sync_service_injects_notes_organization_repository(
    tmp_path: Path,
) -> None:
    dataset_key = generate_dataset_key()
    object_id = "00000000-0000-4000-8000-000000000101"
    envelope = SyncV2Envelope(
        client_envelope_id="remote:notes.keyword:101:1",
        dataset_id="dataset-1",
        device_id="remote-device",
        domain="notes.keyword",
        object_id=object_id,
        operation="upsert",
        schema_version=1,
        object_revision=1,
        server_cursor=8,
        payload={"keyword": "Research"},
        payload_hash="a" * 64,
        encryption_policy="server_trusted_v1",
    )
    state = _repo_with_profile(
        tmp_path, capabilities={"supported_domains": ["notes.keyword"]}
    )
    notes_db = CharactersRAGDB(
        tmp_path / "notes.sqlite", client_id="local-first-tests"
    )
    try:
        service = LocalFirstSyncService(
            server_service=FakeLocalFirstServer(
                pull_envelopes=[envelope.model_dump(mode="json")]
            ),
            state_repository=state,
            local_store=None,
            dataset_keys={"dataset-1": dataset_key},
            notes_organization_repository=NotesOrganizationRepository(
                notes_db, server_profile_id="server-a"
            ),
        )

        result = await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes.keyword"],
        )

        assert result["applied_envelopes"] == 1
        row = notes_db.get_connection().execute(
            "SELECT keyword FROM keywords WHERE sync_id = ?", (object_id,)
        ).fetchone()
        assert row["keyword"] == "Research"
    finally:
        notes_db.close_connection()


async def test_organization_only_sync_does_not_require_dataset_key(
    tmp_path: Path,
) -> None:
    object_id = "00000000-0000-4000-8000-000000000102"
    envelope = SyncV2Envelope(
        client_envelope_id="remote:notes.keyword:102:1",
        dataset_id="dataset-1",
        device_id="remote-device",
        domain="notes.keyword",
        object_id=object_id,
        operation="upsert",
        schema_version=1,
        object_revision=1,
        server_cursor=8,
        payload={"keyword": "No key required"},
        payload_hash="b" * 64,
        encryption_policy="server_trusted_v1",
    )
    state = _repo_with_profile(tmp_path)
    notes_db = CharactersRAGDB(tmp_path / "notes-no-key.sqlite", client_id="tests")
    try:
        service = LocalFirstSyncService(
            server_service=FakeLocalFirstServer(
                pull_envelopes=[envelope.model_dump(mode="json")]
            ),
            state_repository=state,
            local_store=None,
            dataset_keys={},
            notes_organization_repository=NotesOrganizationRepository(notes_db),
        )

        result = await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes.keyword"],
        )

        assert result["applied_envelopes"] == 1
    finally:
        notes_db.close_connection()


async def test_organization_sync_scopes_heads_and_reviews_by_runtime_profile(
    tmp_path: Path,
) -> None:
    state = _repo_with_profile(tmp_path)
    state.set_sync_v2_profile_state(
        server_profile_id="server-b",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first",
        device_id="device-2",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "7"},
        capabilities={"supported_domains": ["notes.keyword"]},
    )
    state.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-b",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
        cursor="7",
    )
    first_id = "00000000-0000-4000-8000-000000000103"
    second_id = "00000000-0000-4000-8000-000000000104"

    def envelope(
        object_id: str, device_id: str, *, revision: int = 1, cursor: int = 8
    ) -> SyncV2Envelope:
        return SyncV2Envelope(
            client_envelope_id=f"remote:notes.keyword:{object_id}:1",
            dataset_id="dataset-1",
            device_id=device_id,
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            schema_version=1,
            object_revision=revision,
            server_cursor=cursor,
            payload={"keyword": "Profile collision"},
            payload_hash=(object_id.replace("-", "") + f"{revision:032x}"),
            encryption_policy="server_trusted_v1",
        )

    server = FakeLocalFirstServer(
        pull_envelopes=[envelope(first_id, "remote-a").model_dump(mode="json")]
    )
    notes_db = CharactersRAGDB(tmp_path / "notes-profiles.sqlite", client_id="tests")
    try:
        service = LocalFirstSyncService(
            server_service=server,
            state_repository=state,
            local_store=None,
            dataset_keys={"dataset-1": generate_dataset_key()},
            notes_organization_repository=NotesOrganizationRepository(notes_db),
        )
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes.keyword"],
        )
        server.pull_envelopes = [
            envelope(first_id, "remote-b", revision=2, cursor=9).model_dump(
                mode="json"
            ),
            envelope(second_id, "remote-b", cursor=10).model_dump(mode="json"),
        ]
        result = await service.sync_once(
            server_profile_id="server-b",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes.keyword"],
        )

        assert result["conflicts"][0]["conflict_type"] == "local_representation_collision"
        heads = notes_db.get_connection().execute(
            "SELECT server_profile_id, object_id FROM notes_organization_heads "
            "ORDER BY server_profile_id"
        ).fetchall()
        assert [tuple(row) for row in heads] == [
            ("server-a", first_id),
            ("server-b", first_id),
        ]
        reviews = notes_db.get_connection().execute(
            "SELECT server_profile_id, remote_object_id "
            "FROM notes_organization_adoption_reviews"
        ).fetchall()
        assert [tuple(row) for row in reviews] == [("server-b", second_id)]
    finally:
        notes_db.close_connection()


async def test_local_first_sync_once_pushes_pulls_applies_and_persists_cursor(tmp_path):
    dataset_key = generate_dataset_key()
    local_builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    remote_builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-2",
        dataset_key=dataset_key,
    )
    incoming = remote_builder.build_note_upsert(
        note_id="note-1",
        title="Remote title",
        body="remote private body",
        status="active",
    )
    outgoing = local_builder.build_note_metadata_update(
        note_id="note-2", status="archived"
    )
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
    assert server.calls[1] == (
        "pull",
        "dataset-1",
        "device-1",
        "7",
        ["notes"],
        25,
        False,
    )
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "9"
    )
    assert (
        repo.get_sync_v2_profile_state(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
        )["dataset_cursors"]["sync_v2"]
        == "9"
    )


async def test_local_first_sync_once_accepts_canonical_profile_mode(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path, profile_mode="local_first_sync")
    store = RecordingLocalStore()
    server = FakeLocalFirstServer()
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
    )

    assert result["dataset_id"] == "dataset-1"
    assert server.calls[-1][0] == "pull"


async def test_local_first_sync_once_chunks_pushes_by_server_max_batch_size(tmp_path):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    outgoing = [
        builder.build_note_metadata_update(note_id=f"note-{index}", status="archived")
        for index in range(5)
    ]
    repo = _repo_with_profile(
        tmp_path,
        capabilities={"supported_domains": ["notes"], "max_batch_size": 2},
    )
    server = FakeLocalFirstServer()
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
        outgoing_envelopes=outgoing,
    )

    push_calls = [call for call in server.calls if call[0] == "push"]
    assert [len(call[3]) for call in push_calls] == [2, 2, 1]
    assert [call[6] for call in push_calls] == [["notes"], ["notes"], ["notes"]]
    assert len({call[4] for call in push_calls}) == 3
    assert result["pushed_envelopes"] == 5


async def test_local_first_sync_once_drains_persisted_outbox_and_records_push_failures(
    tmp_path,
):
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
    reviews = repo.list_sync_v2_conflict_reviews(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
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
    assert [entry["client_envelope_id"] for entry in dispatched] == [
        accepted.client_envelope_id
    ]
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        rejected.client_envelope_id,
        conflicted.client_envelope_id,
    ]
    assert pending_after[0]["last_error"]["error_code"] == "stale_base"
    assert pending_after[1]["last_error"]["error_code"] == "conflict"
    assert reviews[0]["source_conflict_key"] == conflicted.client_envelope_id
    assert reviews[0]["item_label"] == "notes note-3"
    assert reviews[0]["recovery_options"]["accept-remote"] == "available"
    assert (
        repo.get_sync_v2_profile_state(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
        )["last_error"]
        == "push_partial_failure: stale_base,conflict"
    )


async def test_local_first_sync_preserves_accepted_materialization_failure(
    tmp_path,
) -> None:
    dataset_key = generate_dataset_key()
    pending = SyncV2Envelope(
        client_envelope_id="organization-intent-1",
        dataset_id="dataset-1",
        domain="notes.keyword",
        object_id="00000000-0000-4000-8000-000000000001",
        operation="upsert",
        device_id="device-1",
        payload={"keyword": "Agent lesson"},
        payload_hash="a" * 64,
        encryption_policy="server_trusted_v1",
    )
    repo = _repo_with_profile(
        tmp_path,
        capabilities={"supported_domains": ["notes.keyword"]},
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    server = FakeLocalFirstServer(
        push_response=SyncV2PushResponse.model_validate(
            {
                "dataset_id": "dataset-1",
                "accepted": [
                    {
                        "client_envelope_id": pending.client_envelope_id,
                        "server_cursor": 17,
                        "object_revision": 1,
                        "apply_status": "failed",
                        "apply_error_code": "projection_failed",
                        "apply_error_message": "folder parent is missing",
                    }
                ],
                "next_cursor": "17",
            }
        )
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
        domains=["notes.keyword"],
    )

    row = repo.list_sync_v2_outbox_entries(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        status="pending",
    )[0]
    assert result["outbox_dispatched"] == 0
    assert result["outbox_retained"] == 1
    assert row["accepted_result"] == {
        "client_envelope_id": pending.client_envelope_id,
        "server_cursor": 17,
        "object_revision": 1,
        "apply_status": "failed",
        "apply_error_code": "projection_failed",
        "apply_error_message": "folder parent is missing",
    }


async def test_local_first_sync_once_rejects_duplicate_outgoing_ids_before_push(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    outgoing = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=outgoing,
    )
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="duplicate client_envelope_id"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
            outgoing_envelopes=[outgoing],
        )

    pending_after = repo.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert server.calls == []
    assert pending_after[0]["attempt_count"] == 0
    assert pending_after[0]["last_error"] is None
    assert profile["last_error"] == (
        "push_failed: outgoing Sync v2 batch contained duplicate client_envelope_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_rejects_mismatched_push_response_dataset_before_dispatch(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    pending = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    server = FakeLocalFirstServer(
        push_response={
            "dataset_id": "other-dataset",
            "accepted": [{"client_envelope_id": pending.client_envelope_id}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "8",
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="dataset_id"):
        await service.sync_once(
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
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert [call[0] for call in server.calls] == ["push"]
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        pending.client_envelope_id
    ]
    assert pending_after[0]["attempt_count"] == 0
    assert pending_after[0]["last_error"] is None
    assert dispatched == []
    assert profile["last_error"] == (
        "push_failed: Sync v2 push response dataset_id must match requested dataset_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_rejects_unknown_push_response_envelope_ids_before_dispatch(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    pending = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    server = FakeLocalFirstServer(
        push_response={
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": "unknown-envelope"}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "8",
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="client_envelope_id"):
        await service.sync_once(
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
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert [call[0] for call in server.calls] == ["push"]
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        pending.client_envelope_id
    ]
    assert pending_after[0]["attempt_count"] == 0
    assert pending_after[0]["last_error"] is None
    assert dispatched == []
    assert profile["last_error"] == (
        "push_failed: Sync v2 push response referenced unknown client_envelope_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_rejects_incomplete_push_response_before_dispatch(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    acknowledged = builder.build_note_metadata_update(
        note_id="note-1", status="archived"
    )
    omitted = builder.build_note_metadata_update(note_id="note-2", status="active")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=acknowledged,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=omitted,
    )
    server = FakeLocalFirstServer(
        push_response={
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": acknowledged.client_envelope_id}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "8",
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="omitted submitted client_envelope_id"):
        await service.sync_once(
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
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert [call[0] for call in server.calls] == ["push"]
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        acknowledged.client_envelope_id,
        omitted.client_envelope_id,
    ]
    assert [entry["attempt_count"] for entry in pending_after] == [0, 0]
    assert [entry["last_error"] for entry in pending_after] == [None, None]
    assert dispatched == []
    assert profile["last_error"] == (
        "push_failed: Sync v2 push response omitted submitted client_envelope_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_rejects_duplicate_push_response_envelope_ids_before_dispatch(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    pending = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    server = FakeLocalFirstServer(
        push_response={
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": pending.client_envelope_id}],
            "rejected": [
                {
                    "client_envelope_id": pending.client_envelope_id,
                    "error_code": "stale_base",
                    "message": "Local base is stale.",
                }
            ],
            "conflicts": [],
            "next_cursor": "8",
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="duplicate client_envelope_id"):
        await service.sync_once(
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
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert [call[0] for call in server.calls] == ["push"]
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        pending.client_envelope_id
    ]
    assert pending_after[0]["attempt_count"] == 0
    assert pending_after[0]["last_error"] is None
    assert dispatched == []
    assert profile["last_error"] == (
        "push_failed: Sync v2 push response contained duplicate client_envelope_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_preserves_push_and_apply_attention_statuses(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    local_builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    remote_builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="remote-device",
        dataset_key=dataset_key,
    )
    pending = local_builder.build_note_metadata_update(
        note_id="note-1", status="archived"
    )
    incoming = remote_builder.build_note_upsert(
        note_id="note-2",
        title="Remote title",
        body="remote private body",
        status="active",
        base_version="sha256:remote-base",
    )
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    store = RecordingLocalStore()
    store.note_hashes["note-2"] = "sha256:local-dirty"
    server = FakeLocalFirstServer(
        pull_envelopes=[incoming.model_dump(mode="json")],
        push_response={
            "dataset_id": "dataset-1",
            "accepted": [],
            "rejected": [
                {
                    "client_envelope_id": pending.client_envelope_id,
                    "error_code": "stale_base",
                    "message": "Local base is stale.",
                }
            ],
            "conflicts": [],
            "next_cursor": "8",
        },
    )
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
    )
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )
    reviews = repo.list_sync_v2_conflict_reviews(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )

    assert result["outbox_retained"] == 1
    assert result["conflicts"][0]["conflict_type"] == "encrypted_content_edit"
    assert reviews[0]["conflict_kind"] == "encrypted_content_edit"
    assert reviews[0]["item_label"] == "notes note-2"
    assert reviews[0]["recovery_options"]["keep-local"] == "available"
    assert profile["last_error"] == (
        "push_partial_failure: stale_base; apply_conflict: encrypted_content_edit"
    )


async def test_local_first_sync_apply_conflict_review_uses_safe_fallback_key(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )
    service = LocalFirstSyncService(
        server_service=FakeLocalFirstServer(),
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    service._record_conflict_reviews(
        profile=profile,
        dataset_id="dataset-1",
        outbox_entries=[],
        push_conflicts=[],
        apply_conflicts=[
            {
                "domain": "notes",
                "conflict_type": "encrypted_content_edit",
                "message": "Malformed apply conflict.",
            }
        ],
    )
    reviews = repo.list_sync_v2_conflict_reviews(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )

    assert reviews[0]["source_conflict_key"] != "None"
    assert reviews[0]["source_conflict_key"].startswith(
        "apply-conflict:notes:encrypted_content_edit:"
    )
    assert reviews[0]["item_label"] != "notes None"


async def test_local_first_sync_once_uses_stable_push_idempotency_key_for_retry(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    pending = builder.build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    server = FakeLocalFirstServer(push_error=RuntimeError("temporary network split"))
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(RuntimeError, match="temporary network split"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )
    first_key = server.calls[0][4]
    server.push_error = None
    server.push_response = {
        "dataset_id": "dataset-1",
        "accepted": [{"client_envelope_id": pending.client_envelope_id}],
        "rejected": [],
        "conflicts": [],
        "next_cursor": "8",
    }

    await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
    )
    second_key = server.calls[1][4]

    assert first_key
    assert first_key.startswith("sync-v2-push:")
    assert second_key == first_key


async def test_local_first_sync_once_records_outbox_transport_failure_attempt(tmp_path):
    dataset_key = generate_dataset_key()
    pending = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    server = FakeLocalFirstServer(push_error=RuntimeError("temporary network split"))
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(RuntimeError, match="temporary network split"):
        await service.sync_once(
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

    assert [entry["client_envelope_id"] for entry in pending_after] == [
        pending.client_envelope_id
    ]
    assert pending_after[0]["attempt_count"] == 1
    assert pending_after[0]["last_error"] == {
        "client_envelope_id": pending.client_envelope_id,
        "error_code": "push_failed",
        "message": "temporary network split",
        "retryable": True,
    }


async def test_local_first_sync_once_changes_push_idempotency_key_when_batch_changes(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    first = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    second = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-2", status="active")
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first_server = FakeLocalFirstServer()
    first_service = LocalFirstSyncService(
        server_service=first_server,
        state_repository=_repo_with_profile(first_path),
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )
    second_server = FakeLocalFirstServer()
    second_service = LocalFirstSyncService(
        server_service=second_server,
        state_repository=_repo_with_profile(second_path),
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    await first_service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
        outgoing_envelopes=[first],
    )
    await second_service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
        outgoing_envelopes=[first, second],
    )

    assert first_server.calls[0][4] != second_server.calls[0][4]


async def test_local_first_sync_once_rejects_outgoing_domain_outside_requested_domains_before_push(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    outgoing = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="user",
        content="local chat content",
    )
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="domain"):
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

    assert server.calls == []
    assert profile["last_error"] == (
        "push_failed: outgoing Sync v2 envelope domain must be included in requested domains"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_rejects_outgoing_dataset_mismatch_before_push(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    outgoing = SyncEnvelopeBuilder(
        dataset_id="other-dataset",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="dataset_id"):
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

    assert server.calls == []
    assert profile["last_error"] == (
        "push_failed: outgoing Sync v2 envelope dataset_id must match profile dataset_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_rejects_outgoing_device_mismatch_before_push(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    outgoing = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-2",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="device_id"):
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

    assert server.calls == []
    assert profile["last_error"] == (
        "push_failed: outgoing Sync v2 envelope device_id must match profile device_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_records_push_failure_without_advancing_cursor(
    tmp_path,
):
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
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_records_pull_failure_without_advancing_cursor(
    tmp_path,
):
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
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )


async def test_local_first_sync_once_records_apply_failure_without_advancing_cursor(
    tmp_path,
):
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
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_content == {}


async def test_local_first_sync_once_rejects_wrong_dataset_pull_before_apply(tmp_path):
    dataset_key = generate_dataset_key()
    incoming = SyncEnvelopeBuilder(
        dataset_id="other-dataset",
        device_id="remote-device",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(pull_envelopes=[incoming.model_dump(mode="json")])
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="dataset_id"):
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

    assert profile["last_error"] == (
        "apply_failed: pulled Sync v2 envelope dataset_id must match requested dataset_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_metadata == {}


async def test_local_first_sync_once_rejects_out_of_scope_pull_domain_before_apply(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    incoming = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="remote-device",
        dataset_key=dataset_key,
    ).build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="user",
        content="remote chat content",
    )
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(pull_envelopes=[incoming.model_dump(mode="json")])
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="domain"):
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

    assert profile["last_error"] == (
        "apply_failed: pulled Sync v2 envelope domain must be included in requested domains"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_content == {}
    assert store.note_metadata == {}


async def test_local_first_sync_once_rejects_duplicate_pull_envelope_ids_before_apply(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    incoming = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="remote-device",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(
        pull_envelopes=[
            incoming.model_dump(mode="json"),
            incoming.model_dump(mode="json"),
        ]
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="duplicate client_envelope_id"):
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

    assert profile["last_error"] == (
        "apply_failed: pulled Sync v2 response contained duplicate client_envelope_id"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_metadata == {}


async def test_local_first_sync_once_rejects_has_more_pull_without_next_cursor_before_apply(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    incoming = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="remote-device",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(
        pull_response={
            "dataset_id": "dataset-1",
            "envelopes": [incoming.model_dump(mode="json")],
            "next_cursor": None,
            "has_more": True,
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="has_more.*next_cursor"):
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

    assert profile["last_error"] == (
        "apply_failed: Sync v2 pull response has_more requires next_cursor"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_metadata == {}


async def test_local_first_sync_once_rejects_nonempty_pull_without_next_cursor_before_apply(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    incoming = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="remote-device",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(
        pull_response={
            "dataset_id": "dataset-1",
            "envelopes": [incoming.model_dump(mode="json")],
            "next_cursor": None,
            "has_more": False,
        }
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="envelopes.*next_cursor"):
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

    assert profile["last_error"] == (
        "apply_failed: Sync v2 pull response with envelopes requires next_cursor"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_metadata == {}


async def test_local_first_sync_once_rejects_own_device_pull_before_apply(tmp_path):
    dataset_key = generate_dataset_key()
    incoming = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="archived")
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    server = FakeLocalFirstServer(pull_envelopes=[incoming.model_dump(mode="json")])
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=store,
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(ValueError, match="own device"):
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

    assert server.calls[-1] == (
        "pull",
        "dataset-1",
        "device-1",
        "7",
        ["notes"],
        None,
        False,
    )
    assert profile["last_error"] == (
        "apply_failed: pulled Sync v2 envelope from own device is not allowed in incremental sync"
    )
    assert profile["dataset_cursors"]["sync_v2"] == "7"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.note_metadata == {}


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
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "7"
    )
    assert store.workspace_links == set()


async def test_local_first_sync_once_persists_apply_conflict_status_and_advances_cursor(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="remote-device",
        dataset_key=dataset_key,
    )
    incoming = builder.build_note_upsert(
        note_id="note-1",
        title="Remote title",
        body="remote private body",
        status="active",
        base_version="sha256:remote-base",
    )
    repo = _repo_with_profile(tmp_path)
    store = RecordingLocalStore()
    store.note_hashes["note-1"] = "sha256:local-dirty"
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
    )
    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert result["conflicts"][0]["conflict_type"] == "encrypted_content_edit"
    assert profile["last_error"] == "apply_conflict: encrypted_content_edit"
    assert profile["dataset_cursors"]["sync_v2"] == "9"
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "9"
    )


async def test_local_first_sync_once_success_clears_prior_last_error_without_new_cursor(
    tmp_path,
):
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


async def test_local_first_personal_context_sync_fails_closed_without_composition(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
    )

    with pytest.raises(
        ValueError, match="personal_context_sync_transport_unavailable"
    ):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["personal_context.record"],
        )

    assert server.calls == []


async def test_personal_context_sync_once_lazy_loads_exact_dataset_key_after_restart(
    tmp_path,
) -> None:
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(
        tmp_path,
        capabilities={"supported_domains": ["personal_context.record"]},
    )
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "7"},
        capabilities={"supported_domains": ["personal_context.record"]},
    )
    repo.set_personal_context_link_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        state="complete",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="integrity-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        sync_transport_cursor="transport-bootstrap",
        confirmed_cursor="7",
        bootstrap_heads={},
        expected_heads={},
        reviewed_lineage=[],
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )

    class Dispatcher:
        adapter = object()

        @staticmethod
        def dispatch_pending(**_kwargs):
            return {"dispatched": 0, "quarantined": 0}

    keys: dict[str, bytes] = {}
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys=keys,
    )
    loader_calls = []

    def load_runtime(**binding) -> None:
        loader_calls.append(binding)
        keys["other-dataset"] = generate_dataset_key()
        keys["dataset-1"] = dataset_key
        service.personal_context_outbox_dispatcher = Dispatcher()
        service.personal_context_service = object()

    service.personal_context_runtime_loader = load_runtime

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        domains=["personal_context.record"],
    )

    assert loader_calls == [
        {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
        }
    ]
    assert result["pulled_envelopes"] == 0
    assert server.personal_context_complete_pulls == 1


async def test_sync_once_does_not_lazy_load_for_non_personal_context_dataset(
    tmp_path,
) -> None:
    repo = _repo_with_profile(tmp_path)
    calls = []
    service = LocalFirstSyncService(
        server_service=FakeLocalFirstServer(),
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={},
        personal_context_runtime_loader=lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="dataset key is required"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["notes"],
        )

    assert calls == []


async def test_personal_context_sync_once_rejects_loader_key_for_other_dataset(
    tmp_path,
) -> None:
    repo = _repo_with_profile(
        tmp_path,
        capabilities={"supported_domains": ["personal_context.record"]},
    )
    keys: dict[str, bytes] = {}
    service = LocalFirstSyncService(
        server_service=FakeLocalFirstServer(),
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys=keys,
    )
    loader_calls = []

    def load_runtime(**binding) -> None:
        loader_calls.append(binding)
        keys["other-dataset"] = generate_dataset_key()
        service.personal_context_outbox_dispatcher = object()
        service.personal_context_service = object()

    service.personal_context_runtime_loader = load_runtime

    with pytest.raises(ValueError, match="dataset key is required"):
        await service.sync_once(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domains=["personal_context.record"],
        )

    assert loader_calls == [
        {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
        }
    ]


async def test_local_first_complete_binding_uses_private_personal_context_transport(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(
        tmp_path,
        capabilities={"supported_domains": ["personal_context.record"]},
    )
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "7"},
        capabilities={"supported_domains": ["personal_context.record"]},
    )
    repo.set_personal_context_link_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        state="complete",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="integrity-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        confirmed_cursor="7",
        bootstrap_heads={},
        expected_heads={},
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )

    class _Adapter:
        @staticmethod
        def restore_from_storage(envelope, *, storage_key):
            return envelope

    class _Dispatcher:
        adapter = _Adapter()

        @staticmethod
        def dispatch_pending(**kwargs):
            return {"dispatched": 0, "quarantined": 0}

    envelope = SyncV2Envelope(
        client_envelope_id="pc:record-1:v1",
        dataset_id="dataset-1",
        domain="personal_context.record",
        object_id="record-1",
        parent_id="scope-global",
        operation="upsert",
        device_id="device-1",
        base_version=None,
        entity_version="version-1",
        payload={"schema_version": 1},
        payload_hash="hmac-sha256-v1:" + "a" * 64,
        encryption_policy="server_trusted_v1",
    )
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": dataset_key},
        personal_context_outbox_dispatcher=_Dispatcher(),
        personal_context_service=object(),
    )

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["personal_context.record"],
        outgoing_envelopes=[envelope],
    )

    assert result["pushed_envelopes"] == 1
    assert server.personal_context_complete_pushes == 1
    assert server.personal_context_complete_pulls == 1


async def test_unlinked_workspace_edit_stays_local_until_explicitly_mapped(
    tmp_path,
) -> None:
    profile_path = tmp_path / "profile.db"
    protector = InMemoryProfileKeyProtector()
    ids = _Ids()
    profile_repository = PersonalContextRepository(
        profile_path,
        key_protector=protector,
    )
    profile = PersonalContextService(profile_repository, id_factory=ids)
    local_manifest = profile.create_profile()
    local_global = profile.list_scopes()[0]
    local_workspace = profile.create_workspace_scope("workspace-local", "Project")
    record = profile.create_manual_record(
        scope_id=local_workspace.scope_id,
        payload=PreferencePayload(
            subject="project.goal",
            polarity="like",
            value="ship",
        ),
        semantic_key=SemanticKey(namespace="preference", subject="project.goal"),
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
    )
    local_manifest = profile.get_manifest()
    remote_manifest = ProfileManifest(
        profile_id="profile-server",
        revision=0,
        purge_generation=0,
        created_at=local_manifest.created_at,
        updated_at=local_manifest.updated_at,
        current_version_id="manifest-server-v1",
    )
    remote_global = ProfileScope(
        profile_id=remote_manifest.profile_id,
        scope_id="scope-server-global",
        kind=ScopeKind.GLOBAL,
        version_id="scope-server-global-v1",
        created_at=local_global.created_at,
        updated_at=local_global.updated_at,
    )
    remote = CanonicalBootstrapSnapshot(
        dataset_id="dataset-1",
        authority_id="authority-1",
        manifest=remote_manifest,
        scopes=(remote_global,),
        records=(),
        proposals=(),
        purge_generation=0,
        schema_version=1,
        quotas={"max_record_bytes": 16_384},
        cursor="cursor-bootstrap",
        sync_transport_cursor="transport-bootstrap",
        integrity_key_id="integrity-1",
        key_record_id="key-record-1",
        wrapped_key_blob="wrapped",
    )
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=(local_global, local_workspace),
        local_records=(record,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings=profile.list_workspace_bindings(),
    )
    profile.acquire_first_link_freeze(
        plan_id=plan.plan_id,
        snapshot_token=plan.local_snapshot_token,
    )
    profile.apply_reviewed_link(
        plan=plan,
        remote=remote,
        decisions={f"workspace:{local_workspace.scope_id}": "unlinked"},
        integrity_key=b"i" * 32,
    )
    profile.release_first_link_freeze(plan_id=plan.plan_id)
    profile_outbox = ProfileSyncOutbox(profile_repository)
    for entry in profile_outbox.list_pending():
        profile_outbox.acknowledge(entry.outbox_id, f"first-link:{entry.outbox_id}")
    profile_repository.close()
    profile_repository = PersonalContextRepository(
        profile_path,
        key_protector=protector,
    )
    profile = PersonalContextService(profile_repository, id_factory=ids)
    profile_outbox = ProfileSyncOutbox(profile_repository)

    retained = profile.get_record(record.record_id)
    assert retained is not None
    updated = profile.update_record(
        retained.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="project.goal",
                polarity="like",
                value="ship safely",
            )
        ),
        expected_version_id=retained.version_id,
    )
    retained_scope = next(
        scope
        for scope in profile.list_scopes()
        if scope.scope_id == local_workspace.scope_id
    )
    updated_scope = retained_scope.model_copy(
        update={
            "version_id": "scope-unlinked-later-v2",
            "updated_at": retained_scope.updated_at + timedelta(seconds=1),
        }
    )
    profile_repository.commit_scope(
        updated_scope,
        expected_version_id=retained_scope.version_id,
    )
    profile_repository.commit_outbox_body(
        object_type="scope",
        object_id=updated_scope.scope_id,
        version_id=updated_scope.version_id,
        body={"version": 1, "scope": updated_scope.model_dump(mode="json")},
    )
    proposed_record = updated.model_copy(
        update={
            "record_id": "record-proposed-unlinked",
            "version_id": "record-proposed-unlinked-v1",
            "parent_version_id": None,
            "payload": PreferencePayload(
                subject="project.constraint",
                polarity="like",
                value="preserve local context",
            ),
            "semantic_key": SemanticKey(
                namespace="preference",
                subject="project.constraint",
            ),
        }
    )
    proposal = ProfileProposal(
        proposal_id="proposal-unlinked-later",
        profile_id=updated.profile_id,
        scope_id=updated.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed_record,
        provenance=updated.provenance,
        confidence=0.8,
        created_at=updated.updated_at,
        expires_at=updated.updated_at + timedelta(days=90),
    )
    profile_repository.commit_proposal(proposal)
    sync_repository = SyncStateRepository(tmp_path / "sync-state.db")
    sync_repository.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "9"},
        capabilities={
            "supported_domains": [
                "personal_context.manifest",
                "personal_context.scope",
                "personal_context.record",
                "personal_context.proposal",
            ]
        },
    )
    sync_repository.set_personal_context_link_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        state="complete",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id=remote_manifest.profile_id,
        integrity_key_id="integrity-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        confirmed_cursor="9",
        expected_heads=profile.first_link_sync_heads(),
        plan_id=plan.plan_id,
        rebaseline_version=2,
        attention_code=None,
    )
    adapter = PersonalContextSyncAdapter(
        integrity_key=b"i" * 32,
        integrity_key_id="integrity-1",
    )
    dispatcher = PersonalContextOutboxDispatcher(
        profile_outbox=profile_outbox,
        state_repository=sync_repository,
        adapter=adapter,
    )
    server = FakeLocalFirstServer()
    local_first = LocalFirstSyncService(
        server_service=server,
        state_repository=sync_repository,
        local_store=RecordingLocalStore(),
        dataset_keys={"dataset-1": b"s" * 32},
        personal_context_outbox_dispatcher=dispatcher,
        personal_context_service=profile,
    )
    domains = [
        "personal_context.manifest",
        "personal_context.scope",
        "personal_context.record",
        "personal_context.proposal",
    ]

    await local_first.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        domains=domains,
    )

    pushed_domains = {
        envelope["domain"]
        for call in server.calls
        if call[0] == "push"
        for envelope in call[3]
    }
    assert pushed_domains.isdisjoint(
        {
            "personal_context.scope",
            "personal_context.record",
            "personal_context.proposal",
        }
    )
    pending_unlinked = {
        entry.object_type: entry
        for entry in profile_outbox.list_pending()
        if entry.object_type in {"scope", "record", "proposal"}
    }
    assert set(pending_unlinked) == {"scope", "record", "proposal"}
    assert all(
        profile_outbox.read_body(entry.outbox_id) is not None
        for entry in pending_unlinked.values()
    )
    assert profile.get_record(record.record_id) == updated
    assert updated_scope in profile.list_scopes()
    assert proposal in profile_repository.list_proposals()

    profile.map_workspace_scope("workspace-remapped", local_workspace.scope_id)
    await local_first.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        domains=domains,
    )
    retained_head_retry = next(
        entry
        for entry in profile_outbox.list_pending()
        if entry.object_type == "record" and entry.version_id == updated.version_id
    )
    assert profile_outbox.read_body(retained_head_retry.outbox_id) is not None
    await local_first.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        domains=domains,
    )

    pushed_after_mapping = {
        envelope["domain"]
        for call in server.calls
        if call[0] == "push"
        for envelope in call[3]
    }
    assert {
        "personal_context.scope",
        "personal_context.record",
        "personal_context.proposal",
    } <= pushed_after_mapping
    pushed_record_versions = [
        (envelope["base_version"], envelope["entity_version"])
        for call in server.calls
        if call[0] == "push"
        for envelope in call[3]
        if envelope["domain"] == "personal_context.record"
        and envelope["object_id"] == record.record_id
    ]
    assert pushed_record_versions == [
        (None, retained.version_id),
        (retained.version_id, updated.version_id),
    ]
    assert all(
        entry.object_type not in {"scope", "record", "proposal"}
        for entry in profile_outbox.list_pending()
    )


async def test_local_first_sync_once_requires_profile_device_dataset_and_dataset_key(
    tmp_path,
):
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
