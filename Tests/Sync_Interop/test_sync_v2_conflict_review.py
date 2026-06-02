from __future__ import annotations

from tldw_chatbook.Sync_Interop.conflict_review import SyncV2ConflictReviewService
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


def test_sync_v2_conflict_review_records_are_durable_and_user_facing(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)

    repo.record_sync_v2_conflict_review(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        domain="notes",
        item_label="Research note",
        cause="Remote edit conflicts with local edit.",
        local_summary="Local: title changed locally.",
        remote_summary="Remote: body changed on server.",
        source_conflict_key="note-1:remote-2",
        conflict_kind="encrypted_content_edit",
        recovery_options={
            "retry": "available",
            "keep-local": "available",
            "accept-remote": "available",
            "duplicate-fork": "available",
            "defer-later": "available",
        },
        details={"client_envelope_id": "remote-envelope-1"},
    )
    repo.close()

    reopened = SyncStateRepository(db_path)
    reviews = reopened.list_sync_v2_conflict_reviews(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
    )

    assert len(reviews) == 1
    assert reviews[0]["domain"] == "notes"
    assert reviews[0]["item_label"] == "Research note"
    assert reviews[0]["cause"] == "Remote edit conflicts with local edit."
    assert reviews[0]["local_summary"] == "Local: title changed locally."
    assert reviews[0]["remote_summary"] == "Remote: body changed on server."
    assert reviews[0]["recovery_options"] == {
        "retry": "available",
        "keep-local": "available",
        "accept-remote": "available",
        "duplicate-fork": "available",
        "defer-later": "available",
    }
    assert reviews[0]["resolution_status"] == "open"


def test_conflict_review_service_maps_retained_outbox_failures_without_plaintext(tmp_path):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-a",
        dataset_key=dataset_key,
    )
    note = builder.build_note_upsert(
        note_id="note-1",
        title="Private title",
        body="Private body",
    )
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        envelope=note,
    )
    repo.mark_sync_v2_outbox_push_results(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        accepted=[],
        rejected=[
            {
                "client_envelope_id": note.client_envelope_id,
                "error_code": "stale_base",
                "message": "Remote base changed.",
            }
        ],
        conflicts=[],
    )
    service = SyncV2ConflictReviewService(state_repository=repo)

    reviews = service.build_review_items(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-a",
        dataset_id="dataset-1",
        domains=["notes"],
    )

    assert len(reviews) == 1
    assert reviews[0].domain == "notes"
    assert reviews[0].item_label == "notes note-1"
    assert reviews[0].cause == "stale_base: Remote base changed."
    assert reviews[0].local_summary == "Local pending notes change retained for retry."
    assert reviews[0].remote_summary == "Remote state unavailable until retry or conflict review."
    assert reviews[0].recovery_options["retry"] == "available"
    assert reviews[0].recovery_options["keep-local"] == "unavailable"
    assert reviews[0].recovery_options["accept-remote"] == "unavailable"
    assert reviews[0].recovery_options["duplicate-fork"] == "unavailable"
    assert reviews[0].recovery_options["defer-later"] == "available"
    assert "Private" not in str(reviews[0])

