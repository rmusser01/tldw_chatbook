from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.agent_lessons import AgentLessonsSeedResult
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Sync_Interop.conflict_review import SyncV2ConflictReviewService
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.notes_organization import NOTES_ORGANIZATION_DOMAINS
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope


DEPENDENCIES = ("notes.note", "chat.conversation")
COMPLETE_GROUP = (*DEPENDENCIES, *NOTES_ORGANIZATION_DOMAINS)


def _profile(
    *,
    domains=COMPLETE_GROUP,
    state="ready",
    policy="server_trusted_v1",
    captured=3,
    expected=3,
):
    return {
        "profile_bootstrapped": True,
        "user_id": "user-1",
        "active_dataset_id": "dataset-1",
        "device": {"device_id": "device-1", "registered": True},
        "dataset": {
            "dataset_id": "dataset-1",
            "scope": "personal",
            "default_personal": True,
            "domains": list(domains),
            "encryption_policy": policy,
            "notes_organization": {
                "state": state,
                "captured_count": captured,
                "expected_count": expected,
                "error_code": None,
            },
        },
        "server_cursor": 3,
        "capabilities": {
            "protocol_version": "sync-v2-m1",
            "domains": list(COMPLETE_GROUP),
            "supported_adapter_versions": {domain: [1] for domain in COMPLETE_GROUP},
        },
    }


class ProfileClient:
    def __init__(self, profiles, *, bootstrap_response=None):
        self.profiles = [deepcopy(profile) for profile in profiles]
        self.bootstrap_response = deepcopy(bootstrap_response)
        self.calls = []

    async def get_sync_v2_profile(self, *, device_id=None):
        self.calls.append(("get", device_id))
        return self.profiles.pop(0)

    async def bootstrap_sync_v2_profile(self, request):
        self.calls.append(("bootstrap", request.model_dump(mode="json")))
        response = deepcopy(self.bootstrap_response)
        if isinstance(response, dict):
            response["device"] = {
                "device_id": request.device_id,
                "registered": True,
                "client_profile_id": request.client_profile_id,
            }
        return response


@pytest.mark.asyncio
async def test_new_dataset_bootstrap_submits_one_complete_versioned_group(tmp_path):
    absent = _profile(domains=())
    absent["profile_bootstrapped"] = False
    absent["active_dataset_id"] = None
    absent["dataset"] = None
    absent["device"] = None
    ready = _profile()
    client = ProfileClient([absent], bootstrap_response=ready)
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    service = ServerSyncService(client, state_repository=state)

    result = await service.bootstrap_notes_organization_profile(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
    )

    bootstrap_calls = [call for call in client.calls if call[0] == "bootstrap"]
    assert len(bootstrap_calls) == 1
    request = bootstrap_calls[0][1]
    assert request["requested_domains"] == list(COMPLETE_GROUP)
    assert request["supported_adapter_versions"] == {
        domain: [1] for domain in COMPLETE_GROUP
    }
    assert "encryption_policy" not in request
    assert result["dataset"]["notes_organization"]["state"] == "ready"
    persisted = state.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )
    assert persisted["device_id"] == request["device_id"]
    assert persisted["dataset_id"] == "dataset-1"


@pytest.mark.asyncio
async def test_existing_complete_dataset_resumes_without_new_bootstrap(tmp_path):
    client = ProfileClient([_profile()])
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id=None,
    )
    service = ServerSyncService(client, state_repository=state)

    await service.bootstrap_notes_organization_profile(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
    )

    assert [call[0] for call in client.calls] == ["get"]


@pytest.mark.asyncio
async def test_partial_group_advertisement_and_existing_enrollment_are_refused(
    tmp_path,
):
    for profile in (
        _profile(domains=("notes.keyword",)),
        _profile(),
    ):
        if tuple(profile["dataset"]["domains"]) == COMPLETE_GROUP:
            profile["capabilities"]["domains"].remove("notes.folder_link")
        client = ProfileClient([profile], bootstrap_response=_profile())
        service = ServerSyncService(
            client,
            state_repository=SyncStateRepository(
                tmp_path / f"{len(profile['capabilities']['domains'])}.sqlite"
            ),
        )

        with pytest.raises(ValueError, match="complete Notes organization group"):
            await service.bootstrap_notes_organization_profile(
                server_profile_id="server-a",
                authenticated_principal_id="user-1",
                workspace_scope=None,
                display_name="Laptop",
            )
        assert not any(call[0] == "bootstrap" for call in client.calls)


@pytest.mark.asyncio
async def test_incompatible_dataset_encryption_policy_is_rejected(tmp_path):
    client = ProfileClient([_profile(policy="client_private_v1")])
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id=None,
    )
    service = ServerSyncService(
        client,
        state_repository=state,
    )

    with pytest.raises(ValueError, match="server_trusted_v1"):
        await service.bootstrap_notes_organization_profile(
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
            display_name="Laptop",
        )


@pytest.mark.asyncio
async def test_bootstrap_identity_is_persisted_before_network_and_is_device_local(
    tmp_path,
):
    class InterruptedClient:
        def __init__(self):
            self.device_ids = []

        async def get_sync_v2_profile(self, *, device_id=None):
            self.device_ids.append(device_id)
            raise RuntimeError("offline")

    first_state = SyncStateRepository(tmp_path / "first.sqlite")
    interrupted = InterruptedClient()
    first_service = ServerSyncService(interrupted, state_repository=first_state)
    scope = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "user-1",
        "workspace_scope": None,
    }

    with pytest.raises(RuntimeError, match="offline"):
        await first_service.bootstrap_notes_organization_profile(
            **scope, display_name="Laptop"
        )

    persisted = first_state.get_sync_v2_profile_state(**scope)
    assert persisted is not None
    first_device_id = persisted["device_id"]
    first_profile_id = persisted["dry_run_metadata"][
        "notes_organization_bootstrap_identity"
    ]
    assert interrupted.device_ids == [first_device_id]
    assert first_device_id
    assert first_profile_id

    absent = _profile(domains=())
    absent.update(
        profile_bootstrapped=False,
        active_dataset_id=None,
        device=None,
        dataset=None,
    )

    class EchoBootstrapClient(ProfileClient):
        async def bootstrap_sync_v2_profile(self, request):
            response = _profile()
            response["device"] = {
                "device_id": request.device_id,
                "registered": True,
                "client_profile_id": request.client_profile_id,
            }
            self.calls.append(("bootstrap", request.model_dump(mode="json")))
            return response

    retry = EchoBootstrapClient([absent])
    first_service.client = retry
    await first_service.bootstrap_notes_organization_profile(
        **scope,
        display_name="Laptop",
        client_profile_id="replacement-must-not-win",
    )
    request = retry.calls[-1][1]
    assert retry.calls[0] == ("get", first_device_id)
    assert request["device_id"] == first_device_id
    assert request["client_profile_id"] == first_profile_id

    second_state = SyncStateRepository(tmp_path / "second.sqlite")
    second_interrupted = InterruptedClient()
    with pytest.raises(RuntimeError, match="offline"):
        await ServerSyncService(
            second_interrupted, state_repository=second_state
        ).bootstrap_notes_organization_profile(**scope, display_name="Other laptop")
    second = second_state.get_sync_v2_profile_state(**scope)
    assert second is not None
    assert second["device_id"] != first_device_id
    assert (
        second["dry_run_metadata"]["notes_organization_bootstrap_identity"]
        != first_profile_id
    )


@pytest.mark.asyncio
async def test_existing_dataset_is_adopted_only_for_matching_registered_device(
    tmp_path,
):
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    scope = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "user-1",
        "workspace_scope": None,
    }

    class InterruptOnce:
        async def get_sync_v2_profile(self, *, device_id=None):
            raise RuntimeError("seed identity")

    service = ServerSyncService(InterruptOnce(), state_repository=state)
    with pytest.raises(RuntimeError, match="seed identity"):
        await service.bootstrap_notes_organization_profile(
            **scope, display_name="Laptop"
        )
    local = state.get_sync_v2_profile_state(**scope)
    assert local is not None
    mismatched = _profile()
    mismatched["device"] = {"device_id": "some-other-device", "registered": True}

    class EchoBootstrapClient(ProfileClient):
        async def bootstrap_sync_v2_profile(self, request):
            response = _profile()
            response["device"] = {
                "device_id": request.device_id,
                "registered": True,
                "client_profile_id": request.client_profile_id,
            }
            self.calls.append(("bootstrap", request.model_dump(mode="json")))
            return response

    client = EchoBootstrapClient([mismatched])
    service.client = client

    await service.bootstrap_notes_organization_profile(**scope, display_name="Laptop")

    assert [call[0] for call in client.calls] == ["get", "bootstrap"]
    assert client.calls[-1][1]["device_id"] == local["device_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "versions",
    [None, [], {domain: [1] for domain in COMPLETE_GROUP[:-1]}],
)
async def test_capabilities_require_mapping_with_v1_for_every_required_domain(
    tmp_path, versions
):
    profile = _profile()
    profile["capabilities"]["supported_adapter_versions"] = versions
    service = ServerSyncService(
        ProfileClient([profile]),
        state_repository=SyncStateRepository(tmp_path / "sync.sqlite"),
    )

    with pytest.raises(ValueError, match="adapter version 1"):
        await service.bootstrap_notes_organization_profile(
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
            display_name="Laptop",
        )


def _checkpoint(db, *, state="adoption_review"):
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, bootstrap_id, "
            "captured_count, expected_count, pull_cursor, inventory_phase, updated_at) "
            "VALUES ('server-a', 'dataset-1', ?, 'ready', 'bootstrap-1', 3, 3, "
            "'3', 'not_started', '2026-08-29T00:00:00Z')",
            (state,),
        )


def test_adoption_review_is_content_free_and_resolution_is_idempotent(tmp_path: Path):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    _checkpoint(notes)
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO keywords(keyword, deleted, client_id, version, sync_id) "
            "VALUES ('Visible keyword', 0, 'local', 1, "
            "'00000000-0000-4000-8000-000000000001')"
        )
        local_id = str(cursor.lastrowid)
        cursor.execute(
            "INSERT INTO notes_organization_adoption_reviews("
            "review_id, server_profile_id, dataset_id, domain, local_object_id, "
            "remote_object_id, collision_key, display_name, portable_path, state, "
            "created_at, updated_at) VALUES ('review-1', 'server-a', 'dataset-1', "
            "'notes.keyword', ?, '00000000-0000-4000-8000-000000000002', "
            "'visible keyword', 'Visible keyword', NULL, 'open', "
            "'2026-08-29T00:00:00Z', '2026-08-29T00:00:00Z')",
            (local_id,),
        )
    service = SyncV2ConflictReviewService(
        state_repository=SyncStateRepository(tmp_path / "sync.sqlite"),
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
    )

    items = service.build_notes_organization_adoption_items(dataset_id="dataset-1")
    assert len(items) == 1
    assert items[0].conflict_review_id == "review-1"
    assert items[0].item_label == "Visible keyword"
    assert "private" not in str(items).lower()
    assert set(items[0].recovery_options) == {
        "merge",
        "rename_local",
        "keep_local",
    }
    assert (
        service.resolve_notes_organization_adoption(
            review_id="review-1", action="keep_local"
        )
        is True
    )
    assert (
        service.resolve_notes_organization_adoption(
            review_id="review-1", action="keep_local"
        )
        is False
    )
    notes.close_connection()


def test_ready_gate_requires_complete_inventory_and_zero_open_reviews(tmp_path: Path):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    _checkpoint(notes)
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(notes),
        state_repository=SyncStateRepository(tmp_path / "sync.sqlite"),
    )

    assert (
        service.notes_organization_ready(
            server_profile_id="server-a", dataset_id="dataset-1"
        )
        is False
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET inventory_phase = 'complete', "
            "local_state = 'ready' WHERE server_profile_id = 'server-a' "
            "AND dataset_id = 'dataset-1'"
        )
    assert (
        service.notes_organization_ready(
            server_profile_id="server-a", dataset_id="dataset-1"
        )
        is True
    )
    notes.close_connection()


class PullServer:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    async def pull_v2_envelopes(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages.pop(0)


def _keyword_envelope() -> dict[str, object]:
    object_id = str(uuid.UUID("00000000-0000-4000-8000-000000000010"))
    payload = {"keyword": "Remote keyword"}
    content = json.dumps(
        {"operation": "upsert", "payload": payload, "revision": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SyncV2Envelope(
        client_envelope_id="remote-keyword-1",
        dataset_id="dataset-1",
        device_id="remote-device",
        domain="notes.keyword",
        object_id=object_id,
        operation="upsert",
        object_revision=1,
        server_cursor=1,
        payload=payload,
        payload_hash=hashlib.sha256(content).hexdigest(),
        encryption_policy="server_trusted_v1",
    ).model_dump(mode="json")


def _seed_local_keyword(notes: CharactersRAGDB, *, name: str = "Remote keyword") -> int:
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO keywords(keyword, deleted, client_id, version, sync_id) "
            "VALUES (?, 0, 'local', 1, '00000000-0000-4000-8000-000000000011')",
            (name,),
        )
        return int(cursor.lastrowid)


@pytest.mark.asyncio
async def test_bootstrap_pull_applies_complete_history_without_publishing(
    tmp_path: Path,
):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    server = PullServer(
        [
            {
                "dataset_id": "dataset-1",
                "envelopes": [_keyword_envelope()],
                "next_cursor": "1",
                "has_more": False,
            }
        ]
    )
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=state,
        local_store=None,
        notes_organization_repository=NotesOrganizationRepository(notes),
    )

    result = await service.pull_notes_organization_history(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )

    assert result["applied_envelopes"] == 1
    assert result["next_cursor"] == "1"
    assert server.calls[0]["domains"] == list(NOTES_ORGANIZATION_DOMAINS)
    assert (
        notes.get_connection()
        .execute(
            "SELECT keyword FROM keywords WHERE sync_id = ?",
            ("00000000-0000-4000-8000-000000000010",),
        )
        .fetchone()["keyword"]
        == "Remote keyword"
    )
    notes.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid_case", "message"),
    (
        ("response_dataset", "pull response dataset_id"),
        ("envelope_dataset", "envelope dataset_id"),
        ("domain", "envelope domain"),
        ("duplicate", "duplicate client_envelope_id"),
    ),
)
async def test_bootstrap_history_rejects_wrong_scope_before_apply_or_cursor_advance(
    tmp_path: Path,
    invalid_case: str,
    message: str,
) -> None:
    notes = CharactersRAGDB(tmp_path / "scope.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "scope-sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "pre-page"},
    )
    state.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        domain="sync_v2",
        remote_collection="dataset-1",
        cursor="pre-page",
    )
    envelope = _keyword_envelope()
    page = {
        "dataset_id": "dataset-1",
        "envelopes": [envelope],
        "next_cursor": "post-page",
        "has_more": False,
    }
    if invalid_case == "response_dataset":
        page["dataset_id"] = "other-dataset"
    elif invalid_case == "envelope_dataset":
        envelope["dataset_id"] = "other-dataset"
    elif invalid_case == "domain":
        envelope["domain"] = "notes.note"
    else:
        page["envelopes"] = [envelope, deepcopy(envelope)]
    server = PullServer([page])
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=state,
        local_store=None,
        notes_organization_repository=NotesOrganizationRepository(notes),
    )

    with pytest.raises(ValueError, match=message):
        await service.pull_notes_organization_history(
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
        )

    assert server.calls[0]["cursor"] == "pre-page"
    assert (
        state.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "pre-page"
    )
    assert state.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )["dataset_cursors"]["sync_v2"] == "pre-page"
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) AS count FROM keywords WHERE deleted = 0")
        .fetchone()["count"]
        == 0
    )
    notes.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page",
    (
        {
            "dataset_id": "dataset-1",
            "envelopes": [],
            "next_cursor": None,
            "has_more": True,
        },
        {
            "dataset_id": "dataset-1",
            "envelopes": [_keyword_envelope()],
            "next_cursor": None,
            "has_more": False,
        },
    ),
)
async def test_bootstrap_history_rejects_malformed_pagination_before_apply(
    tmp_path: Path,
    page: dict[str, object],
) -> None:
    notes = CharactersRAGDB(tmp_path / "pagination.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "pagination-sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "pre-page"},
    )
    state.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        domain="sync_v2",
        remote_collection="dataset-1",
        cursor="pre-page",
    )
    service = LocalFirstSyncService(
        server_service=PullServer([deepcopy(page)]),
        state_repository=state,
        local_store=None,
        notes_organization_repository=NotesOrganizationRepository(notes),
    )

    with pytest.raises(ValueError, match="next_cursor"):
        await service.pull_notes_organization_history(
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
        )

    assert (
        state.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "pre-page"
    )
    assert state.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )["dataset_cursors"]["sync_v2"] == "pre-page"
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) AS count FROM keywords WHERE deleted = 0")
        .fetchone()["count"]
        == 0
    )
    notes.close_connection()


@pytest.mark.asyncio
async def test_bootstrap_history_rejected_envelope_fails_without_cursor_advance(
    tmp_path: Path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "rejected.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "rejected-sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "pre-page"},
    )
    state.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        domain="sync_v2",
        remote_collection="dataset-1",
        cursor="pre-page",
    )
    rejected = _keyword_envelope()
    rejected["schema_version"] = 2
    service = LocalFirstSyncService(
        server_service=PullServer(
            [
                {
                    "dataset_id": "dataset-1",
                    "envelopes": [rejected],
                    "next_cursor": "post-page",
                    "has_more": False,
                }
            ]
        ),
        state_repository=state,
        local_store=None,
        notes_organization_repository=NotesOrganizationRepository(notes),
    )

    with pytest.raises(
        ValueError,
        match="apply rejected: notes_organization_schema_version_invalid",
    ):
        await service.pull_notes_organization_history(
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
        )

    assert (
        state.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-1",
            workspace_scope=None,
            domain="sync_v2",
            remote_collection="dataset-1",
        ).cursor
        == "pre-page"
    )
    assert state.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )["dataset_cursors"]["sync_v2"] == "pre-page"
    notes.close_connection()


@pytest.mark.asyncio
async def test_adoption_collision_holds_cursor_then_replays_after_keep_local(tmp_path):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    _seed_local_keyword(notes)
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    page = {
        "dataset_id": "dataset-1",
        "envelopes": [_keyword_envelope()],
        "next_cursor": "1",
        "has_more": False,
    }
    server = PullServer([deepcopy(page), deepcopy(page)])
    repository = NotesOrganizationRepository(
        notes, server_profile_id="wrong-default-profile"
    )
    local = LocalFirstSyncService(
        server_service=server,
        state_repository=state,
        local_store=None,
        notes_organization_repository=repository,
    )

    first = await local.pull_notes_organization_history(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )

    assert len(first["conflicts"]) == 1
    assert first["next_cursor"] is None
    cursor = state.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        domain="sync_v2",
        remote_collection="dataset-1",
    )
    assert cursor.cursor is None
    review_id = (
        notes.get_connection()
        .execute(
            "SELECT review_id FROM notes_organization_adoption_reviews WHERE state = 'open'"
        )
        .fetchone()["review_id"]
    )
    review = SyncV2ConflictReviewService(
        state_repository=state,
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
    )
    assert review.resolve_notes_organization_adoption(
        review_id=review_id, action="keep_local"
    )

    second = await local.pull_notes_organization_history(
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
    )

    assert second["conflicts"] == []
    assert second["next_cursor"] == "1"
    assert [call["cursor"] for call in server.calls] == [None, None]
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) AS count FROM keywords WHERE deleted = 0")
        .fetchone()["count"]
        == 1
    )
    notes.close_connection()


def test_keep_local_blocks_later_resource_and_dependent_link_intents(tmp_path):
    notes = CharactersRAGDB(tmp_path / "keep-local.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-nondefault",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, "
            "inventory_phase, updated_at) VALUES "
            "('server-nondefault', 'dataset-1', 'ready', 'ready', 'complete', "
            "'2026-08-29T00:00:00Z')"
        )
        cursor.execute(
            "INSERT INTO keywords(keyword, deleted, client_id, version, sync_id) "
            "VALUES ('Kept keyword', 0, 'local', 1, "
            "'00000000-0000-4000-8000-000000000011')"
        )
        keyword_id = int(cursor.lastrowid)
        cursor.execute(
            "INSERT INTO notes_organization_adoption_reviews("
            "review_id, server_profile_id, dataset_id, domain, local_object_id, "
            "remote_object_id, collision_key, display_name, state, resolution, "
            "created_at, updated_at, resolved_at) VALUES "
            "('kept-review', 'server-nondefault', 'dataset-1', 'notes.keyword', ?, "
            "'00000000-0000-4000-8000-000000000010', 'kept keyword', "
            "'Kept keyword', 'resolved', 'keep_local', "
            "'2026-08-29T00:00:00Z', '2026-08-29T00:00:00Z', "
            "'2026-08-29T00:00:00Z')",
            (str(keyword_id),),
        )
    note_id = notes.add_note(
        "Local note", "private body", note_id="00000000-0000-4000-8000-000000000020"
    )
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="wrong-default-profile"
        ),
        state_repository=state,
    )
    scope = {
        "server_profile_id": "server-nondefault",
        "authenticated_principal_id": None,
        "workspace_scope": None,
    }

    assert service.mutate_keyword(
        keyword_id=keyword_id,
        expected_version=1,
        keyword="Kept keyword renamed",
        **scope,
    )
    assert service.sync_subject_keywords(
        subject_type="note",
        subject_id=note_id,
        keywords=("Kept keyword renamed",),
        **scope,
    ) == ["Kept keyword renamed"]

    assert (
        notes.get_connection()
        .execute("SELECT keyword FROM keywords WHERE id = ?", (keyword_id,))
        .fetchone()["keyword"]
        == "Kept keyword renamed"
    )
    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) AS count FROM notes_organization_sync_intents "
            "WHERE server_profile_id = 'server-nondefault'"
        )
        .fetchone()["count"]
        == 0
    )
    notes.close_connection()


@pytest.mark.parametrize(
    ("action", "new_name", "expected_name", "expected_sync_id"),
    [
        (
            "merge",
            None,
            "Remote keyword",
            "00000000-0000-4000-8000-000000000010",
        ),
        (
            "rename_local",
            "Local keyword",
            "Local keyword",
            "00000000-0000-4000-8000-000000000011",
        ),
        (
            "keep_local",
            None,
            "Remote keyword",
            "00000000-0000-4000-8000-000000000011",
        ),
    ],
)
def test_each_adoption_decision_is_durable_and_idempotent(
    tmp_path: Path,
    action: str,
    new_name: str | None,
    expected_name: str,
    expected_sync_id: str,
):
    notes = CharactersRAGDB(tmp_path / f"{action}.sqlite", client_id="enrollment")
    local_id = _seed_local_keyword(notes)
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_adoption_reviews("
            "review_id, server_profile_id, dataset_id, domain, local_object_id, "
            "remote_object_id, collision_key, display_name, portable_path, state, "
            "created_at, updated_at) VALUES ('review-1', 'server-a', 'dataset-1', "
            "'notes.keyword', ?, '00000000-0000-4000-8000-000000000010', "
            "'remote keyword', 'Remote keyword', NULL, 'open', "
            "'2026-08-29T00:00:00Z', '2026-08-29T00:00:00Z')",
            (str(local_id),),
        )
    service = SyncV2ConflictReviewService(
        state_repository=SyncStateRepository(tmp_path / "sync.sqlite"),
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
    )

    assert service.resolve_notes_organization_adoption(
        review_id="review-1", action=action, new_name=new_name
    )
    assert not service.resolve_notes_organization_adoption(
        review_id="review-1", action=action, new_name=new_name
    )

    row = (
        notes.get_connection()
        .execute("SELECT keyword, sync_id FROM keywords WHERE id = ?", (local_id,))
        .fetchone()
    )
    assert tuple(row) == (expected_name, expected_sync_id)
    resolution = (
        notes.get_connection()
        .execute("SELECT state, resolution FROM notes_organization_adoption_reviews")
        .fetchone()
    )
    assert tuple(resolution) == ("resolved", action)
    notes.close_connection()


class EnrollmentServer:
    def __init__(self, profile):
        self.profile = profile
        self.calls = 0

    async def bootstrap_notes_organization_profile(self, **_kwargs):
        self.calls += 1
        return deepcopy(self.profile)


class EmptyHistory:
    def __init__(self):
        self.calls = 0

    async def pull_notes_organization_history(self, **_kwargs):
        self.calls += 1
        return {
            "applied_envelopes": 0,
            "conflicts": [],
            "next_cursor": "3",
            "has_more": False,
        }


class EmptyBootstrapHistory:
    async def pull_notes_organization_history(self, **_kwargs):
        return {
            "applied_envelopes": 0,
            "conflicts": [],
            "next_cursor": None,
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_empty_bootstrap_history_reaches_ready_without_a_cursor(tmp_path) -> None:
    notes = CharactersRAGDB(tmp_path / "empty.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "empty-sync.sqlite")
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
        state_repository=state,
    )

    result = await service.advance_enrollment(
        server_service=EnrollmentServer(_profile(captured=0, expected=0)),
        local_first_service=EmptyBootstrapHistory(),
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
        enrolled_note_ids=set(),
        enrolled_conversation_ids=set(),
    )

    assert result == {"status": "ready", "dataset_id": "dataset-1"}
    checkpoint = notes.get_connection().execute(
        "SELECT local_state, inventory_phase, pull_cursor "
        "FROM notes_organization_sync_checkpoints"
    ).fetchone()
    assert tuple(checkpoint) == ("ready", "complete", None)
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_folders WHERE name = 'Agent_Lessons' AND deleted = 0"
    ).fetchone()[0] == 1
    notes.close_connection()


@pytest.mark.asyncio
async def test_upgraded_ready_unknown_profile_replays_history_before_seeding(
    tmp_path: Path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "upgraded.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "upgraded-sync.sqlite")
    from tldw_chatbook.Sync_Interop.notes_organization_inventory import (
        LegacyNotesOrganizationInventory,
    )

    repository = NotesOrganizationRepository(notes, server_profile_id="server-a")
    with notes.transaction() as cursor:
        snapshot = LegacyNotesOrganizationInventory(
            repository,
            dataset_id="dataset-1",
            enrolled_note_ids=set(),
            enrolled_conversation_ids=set(),
        )._snapshot(cursor)
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, "
            "inventory_phase, last_inventory_key, updated_at) VALUES "
            "('server-a', 'dataset-1', 'ready', 'ready', 'complete', ?, "
            "'2026-08-30T00:00:00Z')",
            (json.dumps({"baseline": snapshot.digest, "key": None}),),
        )
    history = EmptyHistory()
    service = NotesOrganizationSyncService(
        notes_repository=repository,
        state_repository=state,
    )

    result = await service.advance_enrollment(
        server_service=EnrollmentServer(_profile()),
        local_first_service=history,
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
        enrolled_note_ids=set(),
        enrolled_conversation_ids=set(),
    )

    assert result == {"status": "ready", "dataset_id": "dataset-1"}
    assert history.calls == 1
    seed = notes.get_connection().execute(
        "SELECT state, scope_mode FROM agent_lessons_seed_state WHERE "
        "profile_id = 'server-a' AND dataset_id = 'dataset-1'"
    ).fetchone()
    assert tuple(seed) == ("seeded", "synchronized")
    notes.close_connection()


@pytest.mark.asyncio
async def test_ready_transition_seeds_before_pending_receipt_finalization(
    tmp_path: Path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "order.sqlite", client_id="enrollment")
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
        state_repository=SyncStateRepository(tmp_path / "order-sync.sqlite"),
    )
    order: list[str] = []

    def seed(**_kwargs):
        order.append("seed")
        return AgentLessonsSeedResult("created")

    def finalize(**_kwargs):
        order.append("finalize")
        return {"finalized": 0}

    service.initialize_agent_lessons_seed = seed
    service.finalize_pending_note_organization_receipts = finalize

    result = await service.advance_enrollment(
        server_service=EnrollmentServer(_profile(captured=0, expected=0)),
        local_first_service=EmptyBootstrapHistory(),
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
        enrolled_note_ids=set(),
        enrolled_conversation_ids=set(),
    )

    assert result["status"] == "ready"
    assert order == ["seed", "finalize"]
    notes.close_connection()


@pytest.mark.asyncio
async def test_enrollment_persists_progress_and_resumes_each_durable_checkpoint(
    tmp_path,
):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
        state_repository=state,
    )
    server = EnrollmentServer(_profile(state="ready"))
    history = EmptyHistory()
    interrupted = set()

    def interrupt_once(stage, _cursor=None):
        if stage not in interrupted:
            interrupted.add(stage)
            raise RuntimeError(stage)

    while True:
        try:
            result = await service.advance_enrollment(
                server_service=server,
                local_first_service=history,
                server_profile_id="server-a",
                authenticated_principal_id="user-1",
                workspace_scope=None,
                display_name="Laptop",
                enrolled_note_ids=set(),
                enrolled_conversation_ids=set(),
                after_checkpoint=interrupt_once,
            )
            break
        except RuntimeError:
            continue

    assert result["status"] == "ready"
    assert (
        service.notes_organization_ready(
            server_profile_id="server-a", dataset_id="dataset-1"
        )
        is True
    )
    assert interrupted >= {"server_status", "pull_complete", "adoption_review", "ready"}
    checkpoint = (
        notes.get_connection()
        .execute(
            "SELECT server_state, captured_count, expected_count, pull_cursor, "
            "inventory_phase, local_state FROM notes_organization_sync_checkpoints"
        )
        .fetchone()
    )
    assert tuple(checkpoint) == ("ready", 3, 3, "3", "complete", "ready")
    notes.close_connection()


@pytest.mark.asyncio
async def test_missing_note_dependency_holds_link_local_and_prevents_ready(tmp_path):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    note_id = notes.add_note("Local note", "private body")
    keyword_id = notes.add_keyword("Local keyword")
    assert note_id is not None and keyword_id is not None
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE keywords SET sync_id = "
            "'00000000-0000-4000-8000-000000000020' WHERE id = ?",
            (keyword_id,),
        )
    notes.link_note_to_keyword(note_id, keyword_id)
    state = SyncStateRepository(tmp_path / "sync.sqlite")
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
        state_repository=state,
    )

    result = await service.advance_enrollment(
        server_service=EnrollmentServer(_profile(state="ready")),
        local_first_service=EmptyHistory(),
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
        enrolled_note_ids=set(),
        enrolled_conversation_ids=set(),
    )

    assert result == {
        "status": "adoption_review",
        "dataset_id": "dataset-1",
        "error_code": "notes_organization_dependency_missing",
    }
    assert not service.notes_organization_ready(
        server_profile_id="server-a", dataset_id="dataset-1"
    )
    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) AS count FROM notes_organization_sync_intents "
            "WHERE domain = 'notes.keyword_link'"
        )
        .fetchone()["count"]
        == 0
    )
    checkpoint = (
        notes.get_connection()
        .execute(
            "SELECT local_state, inventory_phase, error_code "
            "FROM notes_organization_sync_checkpoints"
        )
        .fetchone()
    )
    assert tuple(checkpoint) == (
        "adoption_review",
        "complete",
        "notes_organization_dependency_missing",
    )
    notes.close_connection()


@pytest.mark.asyncio
async def test_server_failure_replaces_local_ready_and_persists_safe_status(tmp_path):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="enrollment")
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, bootstrap_id, "
            "captured_count, expected_count, pull_cursor, inventory_phase, updated_at) "
            "VALUES ('server-a', 'dataset-1', 'ready', 'ready', 'bootstrap-1', 3, 3, "
            "'3', 'complete', '2026-08-29T00:00:00Z')"
        )
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="server-a"
        ),
        state_repository=SyncStateRepository(tmp_path / "sync.sqlite"),
    )
    failed = _profile(state="failed", captured=2, expected=3)
    failed["dataset"]["notes_organization"]["error_code"] = "capture_failed"

    result = await service.advance_enrollment(
        server_service=EnrollmentServer(failed),
        local_first_service=EmptyHistory(),
        server_profile_id="server-a",
        authenticated_principal_id="user-1",
        workspace_scope=None,
        display_name="Laptop",
        enrolled_note_ids=set(),
        enrolled_conversation_ids=set(),
    )

    assert result["status"] == "failed"
    checkpoint = (
        notes.get_connection()
        .execute(
            "SELECT local_state, server_state, captured_count, expected_count, error_code "
            "FROM notes_organization_sync_checkpoints"
        )
        .fetchone()
    )
    assert tuple(checkpoint) == ("failed", "failed", 2, 3, "capture_failed")
    notes.close_connection()
