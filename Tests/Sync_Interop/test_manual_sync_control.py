from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.manual_sync_control import ManualSyncControlService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

pytestmark = pytest.mark.asyncio


class RecordingLocalFirstSync:
    def __init__(self, result: dict | None = None, exc: Exception | None = None) -> None:
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


def _repo_with_profile(tmp_path, *, dataset_id: str | None = "dataset-1") -> SyncStateRepository:
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
            "rejected_envelopes": [{"client_envelope_id": "msg-1", "error_code": "policy"}],
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
