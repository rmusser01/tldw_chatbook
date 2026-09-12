from __future__ import annotations

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.manual_sync_control import ManualSyncControlService
from tldw_chatbook.Sync_Interop.notes_organization import NOTES_ORGANIZATION_DOMAINS
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

pytestmark = pytest.mark.asyncio


class RecordingLocalFirstSync:
    def __init__(
        self, result: dict | None = None, exc: Exception | None = None
    ) -> None:
        self.calls: list[dict] = []
        self.local_store = object()
        self.result = result or {
            "pushed_envelopes": 0,
            "pulled_envelopes": 0,
            "applied_envelopes": 0,
            "outbox_dispatched": 0,
            "outbox_retained": 0,
            "rejected_envelopes": [],
            "push_conflicts": [],
            "conflicts": [],
        }
        self.exc = exc

    async def sync_once(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return dict(self.result)


class LocalFirstSyncWithoutStore(RecordingLocalFirstSync):
    def __init__(self) -> None:
        super().__init__()
        self.local_store = None


class MutatingLocalFirstSync(RecordingLocalFirstSync):
    def __init__(
        self,
        repo: SyncStateRepository,
        *,
        accepted_client_envelope_ids: list[str],
    ) -> None:
        super().__init__(
            {
                "pushed_envelopes": len(accepted_client_envelope_ids),
                "pulled_envelopes": 0,
                "applied_envelopes": 0,
                "outbox_dispatched": len(accepted_client_envelope_ids),
                "outbox_retained": 0,
                "rejected_envelopes": [],
                "push_conflicts": [],
                "conflicts": [],
            }
        )
        self.repo = repo
        self.accepted_client_envelope_ids = accepted_client_envelope_ids

    async def sync_once(self, **kwargs):
        self.calls.append(kwargs)
        self.repo.mark_sync_v2_outbox_push_results(
            server_profile_id=kwargs["server_profile_id"],
            authenticated_principal_id=kwargs["authenticated_principal_id"],
            workspace_scope=kwargs["workspace_scope"],
            dataset_id="dataset-1",
            accepted=[
                {"client_envelope_id": client_envelope_id}
                for client_envelope_id in self.accepted_client_envelope_ids
            ],
            rejected=[],
            conflicts=[],
        )
        return dict(self.result)


class FailingMutatingLocalFirstSync(RecordingLocalFirstSync):
    def __init__(
        self, repo: SyncStateRepository, *, failed_client_envelope_id: str
    ) -> None:
        super().__init__(exc=RuntimeError("temporary network split"))
        self.repo = repo
        self.failed_client_envelope_id = failed_client_envelope_id

    async def sync_once(self, **kwargs):
        self.calls.append(kwargs)
        self.repo.mark_sync_v2_outbox_push_results(
            server_profile_id=kwargs["server_profile_id"],
            authenticated_principal_id=kwargs["authenticated_principal_id"],
            workspace_scope=kwargs["workspace_scope"],
            dataset_id="dataset-1",
            accepted=[],
            rejected=[
                {
                    "client_envelope_id": self.failed_client_envelope_id,
                    "error_code": "push_failed",
                    "message": "temporary network split",
                    "retryable": True,
                }
            ],
            conflicts=[],
        )
        raise self.exc


def _repo_with_profile(
    tmp_path, *, dataset_id: str | None = "dataset-1"
) -> SyncStateRepository:
    repo = SyncStateRepository(tmp_path / "manual_sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        profile_mode="local_first_sync",
        device_id="device-a",
        dataset_id=dataset_id,
        dataset_cursors={"sync_v2": "cursor-1"},
        capabilities={"supported_domains": ["notes", "chat"]},
        dry_run_metadata={},
    )
    return repo


def _enqueue_note_and_chat(repo: SyncStateRepository, dataset_key: bytes) -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-a",
        dataset_key=dataset_key,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        envelope=builder.build_note_upsert(
            note_id="note-1",
            title="Research note",
            body="Local note content",
        ),
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        envelope=builder.build_chat_message(
            conversation_id="conv-1",
            message_id="msg-1",
            role="user",
            content="Local chat message",
        ),
    )


async def test_manual_sync_preview_counts_notes_and_chat_without_dispatch(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    _enqueue_note_and_chat(repo, dataset_key)
    sync_runner = RecordingLocalFirstSync()
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
    )

    preview = service.preview(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert preview.status == "ready"
    assert preview.can_run is True
    assert preview.pending_total == 2
    assert preview.pending_by_domain == {"notes": 1, "chat": 1}
    assert "2 pending" in preview.user_message
    assert sync_runner.calls == []


async def test_manual_sync_run_is_explicit_and_maps_partial_failure(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    _enqueue_note_and_chat(repo, dataset_key)
    sync_runner = RecordingLocalFirstSync(
        {
            "pushed_envelopes": 1,
            "pulled_envelopes": 0,
            "applied_envelopes": 0,
            "outbox_dispatched": 1,
            "outbox_retained": 1,
            "rejected_envelopes": [
                {"client_envelope_id": "msg-1", "error_code": "policy"}
            ],
            "push_conflicts": [],
            "conflicts": [],
        }
    )
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
    )

    preview = service.preview(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )
    assert sync_runner.calls == []

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert preview.status == "ready"
    assert result.status == "partial-failure"
    assert "partial" in result.user_message.lower()
    assert result.summary["outbox_retained"] == 1
    assert sync_runner.calls == [
        {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-a",
            "domains": ["notes", "chat"],
        }
    ]


async def test_manual_sync_run_blocks_without_dataset_key(tmp_path):
    repo = _repo_with_profile(tmp_path)
    sync_runner = RecordingLocalFirstSync()
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={},
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "blocked"
    assert "dataset key" in result.user_message.lower()
    assert sync_runner.calls == []


async def test_manual_sync_run_blocks_without_local_apply_store(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    sync_runner = LocalFirstSyncWithoutStore()
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "blocked"
    assert "local apply store" in result.user_message.lower()
    assert sync_runner.calls == []


async def test_manual_sync_preview_uses_shared_dataset_key_updates(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    _enqueue_note_and_chat(repo, dataset_key)
    shared_dataset_keys: dict[str, bytes] = {}
    sync_runner = RecordingLocalFirstSync()
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys=shared_dataset_keys,
    )

    blocked_preview = service.preview(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )
    shared_dataset_keys["dataset-1"] = dataset_key
    ready_preview = service.preview(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert blocked_preview.status == "blocked"
    assert ready_preview.status == "ready"
    assert ready_preview.pending_by_domain == {"notes": 1, "chat": 1}


async def test_manual_sync_run_blocks_without_local_first_sync_service(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=None,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "blocked"
    assert "local apply store" in result.user_message.lower()


async def test_manual_sync_run_returns_post_run_pending_counts(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    _enqueue_note_and_chat(repo, dataset_key)
    pending_before = repo.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
    )
    accepted_id = str(pending_before[0]["client_envelope_id"])
    sync_runner = MutatingLocalFirstSync(
        repo,
        accepted_client_envelope_ids=[accepted_id],
    )
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "success"
    assert result.preview.pending_total == 1


async def test_manual_sync_run_surfaces_conflict_review_items(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    repo.record_sync_v2_conflict_review(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        domain="chat",
        item_label="Chat message msg-1",
        cause="Remote assistant variant conflicts with local variant.",
        local_summary="Local chat message retained.",
        remote_summary="Remote chat message changed.",
        source_conflict_key="msg-1:remote",
        conflict_kind="chat_variant_conflict",
        recovery_options={
            "retry": "available",
            "keep-local": "available",
            "accept-remote": "available",
            "duplicate-fork": "available",
            "defer-later": "available",
        },
    )
    sync_runner = RecordingLocalFirstSync(
        {
            "pushed_envelopes": 0,
            "pulled_envelopes": 0,
            "applied_envelopes": 0,
            "outbox_dispatched": 0,
            "outbox_retained": 1,
            "rejected_envelopes": [],
            "push_conflicts": [{"client_envelope_id": "msg-1"}],
            "conflicts": [],
        }
    )
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "conflict"
    assert result.conflict_reviews[0].item_label == "Chat message msg-1"
    assert result.conflict_reviews[0].recovery_options["duplicate-fork"] == "available"


async def test_manual_sync_failed_run_surfaces_retained_outbox_failure(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    _enqueue_note_and_chat(repo, dataset_key)
    pending_before = repo.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
    )
    failed_id = str(pending_before[0]["client_envelope_id"])
    sync_runner = FailingMutatingLocalFirstSync(
        repo,
        failed_client_envelope_id=failed_id,
    )
    service = ManualSyncControlService(
        state_repository=repo,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "failed"
    assert result.conflict_reviews
    assert result.conflict_reviews[0].cause == "push_failed: temporary network split"
    assert result.conflict_reviews[0].recovery_options["retry"] == "available"


async def test_manual_sync_advances_and_resumes_notes_adoption_review(tmp_path):
    dataset_key = generate_dataset_key()
    state = _repo_with_profile(tmp_path)
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="manual-enrollment")
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
            "remote_object_id, collision_key, display_name, state, created_at, updated_at) "
            "VALUES ('review-1', 'server-a', 'dataset-1', 'notes.keyword', ?, "
            "'00000000-0000-4000-8000-000000000002', 'visible keyword', "
            "'Visible keyword', 'open', '2026-08-29T00:00:00Z', "
            "'2026-08-29T00:00:00Z')",
            (local_id,),
        )

    class Enrollment:
        def __init__(self):
            self.calls = []

        async def advance_enrollment(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "status": "adoption_review" if len(self.calls) == 1 else "ready",
                "dataset_id": "dataset-1",
            }

        def for_server_profile(self, server_profile_id):
            assert server_profile_id == "server-a"
            return self

    enrollment = Enrollment()
    sync_runner = RecordingLocalFirstSync()
    sync_runner.server_service = object()
    service = ManualSyncControlService(
        state_repository=state,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
        notes_organization_sync_service=enrollment,
        notes_repository=NotesOrganizationRepository(
            notes, server_profile_id="wrong-default-profile"
        ),
    )
    arguments = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "user-a",
        "workspace_scope": "workspace-a",
    }

    first = await service.run_once(**arguments)

    assert first.status == "conflict"
    assert first.conflict_reviews[0].conflict_review_id == "review-1"
    assert service.list_conflict_reviews(**arguments)[0].item_label == "Visible keyword"
    assert service.resolve_notes_organization_adoption(
        **arguments, review_id="review-1", action="keep_local"
    )
    assert service.list_conflict_reviews(**arguments) == ()

    second = await service.run_once(**arguments)

    assert second.status == "success"
    assert len(enrollment.calls) == 2
    assert len(sync_runner.calls) == 1
    assert enrollment.calls[0]["server_profile_id"] == "server-a"
    notes.close_connection()


async def test_production_manual_sync_derives_dependencies_and_syncs_complete_group(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    state = _repo_with_profile(tmp_path)
    notes = CharactersRAGDB(tmp_path / "production-notes.sqlite", client_id="device-a")
    note_id = "00000000-0000-4000-8000-000000000041"
    conversation_id = "00000000-0000-4000-8000-000000000042"
    notes.add_note("Linked note", "private body", note_id=note_id)
    notes.add_conversation(
        {"id": conversation_id, "root_id": conversation_id, "title": "Conversation"}
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, "
            "inventory_phase, updated_at) VALUES "
            "('server-a', 'dataset-1', 'ready', 'ready', 'complete', "
            "'2026-08-29T00:00:00+00:00')"
        )
    repository = NotesOrganizationRepository(notes, server_profile_id="server-a")
    organization = NotesOrganizationSyncService(
        notes_repository=repository,
        state_repository=state,
    )
    organization.sync_subject_keywords(
        subject_type="note",
        subject_id=note_id,
        keywords=("agent-lesson",),
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )
    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents "
            "WHERE acknowledged_at IS NULL"
        )
        .fetchone()[0]
        == 2
    )

    class Enrollment:
        def __init__(self):
            self.calls = []

        def for_server_profile(self, server_profile_id):
            assert server_profile_id == "server-a"
            return self

        async def advance_enrollment(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "ready", "dataset_id": "dataset-1"}

    enrollment = Enrollment()
    sync_runner = RecordingLocalFirstSync()
    sync_runner.server_service = object()
    service = ManualSyncControlService(
        state_repository=state,
        local_first_sync_service=sync_runner,
        dataset_keys={"dataset-1": dataset_key},
        notes_organization_sync_service=enrollment,
        notes_repository=repository,
    )

    result = await service.run_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
    )

    assert result.status == "success"
    assert enrollment.calls[0]["enrolled_note_ids"] == {note_id}
    assert enrollment.calls[0]["enrolled_conversation_ids"] == {conversation_id}
    assert sync_runner.calls[0]["domains"] == [
        "notes",
        "chat",
        *NOTES_ORGANIZATION_DOMAINS,
    ]
    notes.close_connection()
