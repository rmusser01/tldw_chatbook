"""Durable Copy progress retains versions, ownership, and safe recovery."""

from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_copy_models import LOCAL, SERVER, TIME, item
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyOwnerRef,
    ResearchCopyConfirmation,
    ResearchCopyManifest,
    ResearchCopyStatus,
)
from tldw_chatbook.Research_Workspace.copy_receipts import (
    CopyItemStage,
    CopyReceiptConflict,
    CopyReceiptFailure,
    ResearchCopyReceiptStore,
)


@pytest.fixture
def setup_copy(tmp_path):
    path = tmp_path / "workspace.sqlite"
    db = WorkspaceDB(path)
    tick = [100]
    store = ResearchCopyReceiptStore(db, clock=lambda: tick[0])
    manifest = ResearchCopyManifest(LOCAL, SERVER, (item("41"), item("42")), TIME)
    confirmation = ResearchCopyConfirmation(manifest.manifest_hash, confirmed=True)
    try:
        yield path, db, store, tick, manifest, confirmation
    finally:
        db.close()


def test_confirmed_receipt_is_idempotent_and_scoped_on_restart(setup_copy):
    path, db, store, _, manifest, confirmation = setup_copy
    with pytest.raises(ValueError, match="Confirm"):
        store.create_from_manifest(manifest, replace(confirmation, confirmed=False))
    assert store.get(manifest.transfer_id) is None
    first = store.create_from_manifest(manifest, confirmation)
    assert store.create_from_manifest(manifest, confirmation) == first
    assert first.status is ResearchCopyStatus.CONFIRMED and first.revision == 1
    assert len(first.items) == 2
    with db.connection() as conn:
        row = conn.execute(
            "SELECT manifest_json FROM research_copy_operations"
        ).fetchone()
        assert "content" not in row[0].replace("copy_content", "")
    reopened = WorkspaceDB(path)
    try:
        active = ResearchCopyReceiptStore(reopened)
        assert active.get(manifest.transfer_id) == first
        assert active.list_recoverable(LOCAL) == (first,)
        assert (
            active.list_recoverable(replace(SERVER, principal_id="other-account")) == ()
        )
    finally:
        reopened.close()


def test_copy_history_uses_both_scope_indexes_without_statistics(setup_copy):
    _, db, store, _, manifest, confirmation = setup_copy
    store.create_from_manifest(manifest, confirmation)
    statements = []
    with db.connection() as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'sqlite_stat1'").fetchone() is None
        connection.set_trace_callback(statements.append)
        try:
            assert len(store.list_history(LOCAL)) == 1
        finally:
            connection.set_trace_callback(None)
        select = next(statement for statement in statements if statement.startswith("SELECT transfer_id FROM research_copy_operations"))
        details = "\n".join(row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + select))
        assert "idx_research_copy_source" in details
        assert "idx_research_copy_destination" in details


def test_copy_lease_expiry_fences_the_previous_worker(setup_copy):
    _, _, store, tick, manifest, confirmation = setup_copy
    created = store.create_from_manifest(manifest, confirmation)
    with pytest.raises(CopyReceiptConflict, match="revision"):
        store.claim(manifest.transfer_id, expected_revision=None)
    claimed = store.claim(
        manifest.transfer_id, expected_revision=created.revision, lease_seconds=10
    )
    with pytest.raises(CopyReceiptConflict, match="worker"):
        store.claim(
            manifest.transfer_id, expected_revision=claimed.revision, lease_seconds=10
        )
    tick[0] = 111
    recovered = store.claim(
        manifest.transfer_id, expected_revision=claimed.revision, lease_seconds=10
    )
    assert recovered.lease_token != claimed.lease_token
    with pytest.raises(CopyReceiptConflict, match="worker"):
        store.record_server_migration(
            manifest.transfer_id,
            "server-session",
            expected_revision=recovered.revision,
            lease_token=claimed.lease_token,
        )


def test_rolled_back_status_requires_owner_acknowledgement_and_remains_in_history(
    setup_copy,
):
    _, _, store, _, manifest, confirmation = setup_copy
    created = store.create_from_manifest(manifest, confirmation)
    held = store.claim(manifest.transfer_id, expected_revision=created.revision)
    item = held.items[0]
    destination = CopyOwnerRef(SERVER, "media", "900", version=1)
    saved = store.record_item_stage(
        manifest.transfer_id,
        item.item_key,
        CopyItemStage.DESTINATION_SAVED,
        destination=destination,
        expected_revision=held.revision,
        lease_token=held.lease_token,
    )
    with pytest.raises(CopyReceiptConflict, match="acknowledgement"):
        store.finish(
            manifest.transfer_id,
            ResearchCopyStatus.ROLLED_BACK,
            expected_revision=saved.revision,
            lease_token=held.lease_token,
        )
    undone = store.record_item_stage(
        manifest.transfer_id,
        item.item_key,
        CopyItemStage.DESTINATION_SAVED,
        rolled_back=True,
        expected_revision=saved.revision,
        lease_token=held.lease_token,
    )
    finished = store.finish(
        manifest.transfer_id,
        ResearchCopyStatus.ROLLED_BACK,
        expected_revision=undone.revision,
        lease_token=held.lease_token,
    )
    assert finished.items[0].rolled_back
    assert store.list_history(LOCAL) == (finished,)
    assert not store.list_recoverable(LOCAL)


def test_item_stage_cas_preserves_exact_owner_then_recovers_partial_copy(setup_copy):
    _, _, store, _, manifest, confirmation = setup_copy
    receipt = store.create_from_manifest(manifest, confirmation)
    receipt = store.claim(manifest.transfer_id, expected_revision=receipt.revision)
    lease = receipt.lease_token
    first, second = receipt.items
    owner = CopyOwnerRef(SERVER, "media", "900", version=1)
    with pytest.raises(CopyReceiptConflict, match="next stage"):
        store.record_item_stage(
            manifest.transfer_id,
            first.item_key,
            CopyItemStage.ASSOCIATED,
            lease_token=lease,
            expected_revision=receipt.revision,
            destination=owner,
        )
    with pytest.raises(ValueError, match="destination"):
        store.record_item_stage(
            manifest.transfer_id,
            first.item_key,
            CopyItemStage.DESTINATION_SAVED,
            lease_token=lease,
            expected_revision=receipt.revision,
            destination=replace(owner, workspace=LOCAL),
        )
    for stage in (
        CopyItemStage.DESTINATION_SAVED,
        CopyItemStage.ASSOCIATED,
        CopyItemStage.VERIFIED,
    ):
        before = receipt.revision
        receipt = store.record_item_stage(
            manifest.transfer_id,
            first.item_key,
            stage,
            lease_token=lease,
            expected_revision=receipt.revision,
            destination=owner,
        )
        assert receipt.revision == before + 1
    with pytest.raises(CopyReceiptConflict, match="revision"):
        store.record_item_stage(
            manifest.transfer_id,
            second.item_key,
            CopyItemStage.PLANNED,
            failure=CopyReceiptFailure.OWNER_UNAVAILABLE,
            retryable=True,
            lease_token=lease,
            expected_revision=before,
        )
    receipt = store.record_item_stage(
        manifest.transfer_id,
        second.item_key,
        CopyItemStage.PLANNED,
        failure=CopyReceiptFailure.OWNER_UNAVAILABLE,
        retryable=True,
        lease_token=lease,
        expected_revision=receipt.revision,
    )
    with pytest.raises(CopyReceiptConflict, match="verified"):
        store.finish(
            manifest.transfer_id,
            ResearchCopyStatus.COMPLETED,
            lease_token=lease,
            expected_revision=receipt.revision,
        )
    partial = store.finish(
        manifest.transfer_id,
        ResearchCopyStatus.PARTIALLY_COMPLETED,
        lease_token=lease,
        expected_revision=receipt.revision,
    )
    resumed = store.claim(manifest.transfer_id, expected_revision=partial.revision)
    assert resumed.items[0].stage is CopyItemStage.VERIFIED
    assert resumed.items[0].destination == owner
    assert resumed.items[1].failure is CopyReceiptFailure.OWNER_UNAVAILABLE


def test_copy_receipt_rejects_an_indexed_authority_that_does_not_match_its_manifest(
    setup_copy,
):
    _, db, store, _, manifest, confirmation = setup_copy
    store.create_from_manifest(manifest, confirmation)
    wrong = replace(SERVER, principal_id="other-account")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE research_copy_operations SET destination_principal_id = ? WHERE transfer_id = ?",
            (wrong.principal_id, manifest.transfer_id),
        )
    with pytest.raises(CopyReceiptConflict, match="workspace identity"):
        store.list_recoverable(wrong)


def test_verified_server_copy_waits_for_the_final_server_receipt(setup_copy):
    _, _, store, _, manifest, confirmation = setup_copy
    current = store.create_from_manifest(manifest, confirmation)
    current = store.claim(manifest.transfer_id, expected_revision=current.revision)
    for entry in current.items:
        target = CopyOwnerRef(
            SERVER, "media", "target-" + entry.item.source.owner_id, version=2
        )
        for stage in (
            CopyItemStage.DESTINATION_SAVED,
            CopyItemStage.ASSOCIATED,
            CopyItemStage.VERIFIED,
        ):
            current = store.record_item_stage(
                manifest.transfer_id,
                entry.item_key,
                stage,
                destination=target,
                expected_revision=current.revision,
                lease_token=current.lease_token,
            )
    with pytest.raises(CopyReceiptConflict, match="finalized Server migration"):
        store.finish(
            manifest.transfer_id,
            ResearchCopyStatus.COMPLETED,
            expected_revision=current.revision,
            lease_token=current.lease_token,
        )
    current = store.record_server_migration(
        manifest.transfer_id,
        "server-receipt",
        finalized=True,
        expected_revision=current.revision,
        lease_token=current.lease_token,
    )
    completed = store.finish(
        manifest.transfer_id,
        ResearchCopyStatus.COMPLETED,
        expected_revision=current.revision,
        lease_token=current.lease_token,
    )
    assert completed.lease_token == "" and completed.server_migration_finalized
    assert store.list_recoverable(SERVER) == ()
