from __future__ import annotations

from datetime import UTC, datetime

from tldw_profile_core import ProposalState, SyncMode

from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    RecordMutation,
)
from tldw_chatbook.Personal_Context.sync_outbox import ProfileSyncOutbox


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _service(tmp_path, memory_protector) -> PersonalContextService:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    return PersonalContextService(repository, clock=lambda: NOW, id_factory=_Ids())


def test_profile_creation_journals_exact_manifest_and_global_scope_atomically(
    tmp_path, memory_protector
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    scope = service.list_scopes()[0]

    outbox = ProfileSyncOutbox(service._repository)
    entries = outbox.list_pending()

    assert {(entry.object_type, entry.object_id) for entry in entries} == {
        ("manifest", manifest.profile_id),
        ("scope", scope.scope_id),
    }
    bodies = {entry.object_type: outbox.read_body(entry.outbox_id) for entry in entries}
    assert bodies["manifest"] == {
        "version": 1,
        "manifest": manifest.model_dump(mode="json"),
    }
    assert bodies["scope"] == {
        "version": 1,
        "scope": scope.model_dump(mode="json"),
    }


def test_profile_outbox_uses_monotonic_order_when_timestamps_tie(
    tmp_path, memory_protector, monkeypatch
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.repository._now_text",
        lambda: "2026-08-30T12:00:00.000Z",
    )
    service = _service(tmp_path, memory_protector)
    service.create_profile()

    with service._repository._connect() as connection:
        rows = connection.execute(
            "SELECT object_type, sequence FROM encrypted_outbox ORDER BY sequence"
        ).fetchall()

    assert [(row["object_type"], row["sequence"]) for row in rows] == [
        ("manifest", 1),
        ("scope", 2),
    ]


def test_device_only_record_never_enters_profile_sync_outbox(
    tmp_path, memory_protector, record_factory
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    outbox = ProfileSyncOutbox(service._repository)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")

    record = record_factory(manifest.profile_id, sync_mode=SyncMode.DEVICE_ONLY)
    record = record.model_copy(update={"scope_id": service.list_scopes()[0].scope_id})
    service.create_record(record)

    assert all(
        entry.object_id != record.record_id for entry in outbox.list_pending()
    )


def test_pending_device_only_proposal_never_enters_profile_sync_outbox(
    tmp_path, memory_protector, proposal_factory
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    outbox = ProfileSyncOutbox(service._repository)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    proposal = proposal_factory(manifest.profile_id)
    proposal = proposal.model_copy(
        update={
            "proposed_record": proposal.proposed_record.model_copy(
                update={
                    "controls": proposal.proposed_record.controls.model_copy(
                        update={"sync_mode": SyncMode.DEVICE_ONLY}
                    )
                }
            )
        }
    )

    service._repository.commit_proposal(proposal)

    assert service._repository.get_proposal(proposal.proposal_id) == proposal
    assert all(
        entry.object_id != proposal.proposal_id for entry in outbox.list_pending()
    )


def test_interview_batch_journals_advanced_manifest_with_records(
    tmp_path, memory_protector, record_factory
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    outbox = ProfileSyncOutbox(service._repository)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    record = record_factory(manifest.profile_id).model_copy(
        update={"scope_id": scope.scope_id}
    )
    next_manifest = manifest.model_copy(
        update={
            "revision": manifest.revision + 1,
            "current_version_id": "manifest-interview-next",
        }
    )

    service._repository.commit_interview_batch(
        (record,),
        next_manifest,
        expected_record_versions={record.record_id: None},
        expected_manifest_version=manifest.current_version_id,
    )

    entries = outbox.list_pending()
    assert {(entry.object_type, entry.object_id) for entry in entries} == {
        ("manifest", manifest.profile_id),
        ("record", record.record_id),
    }
    manifest_entry = next(
        entry for entry in entries if entry.object_type == "manifest"
    )
    assert outbox.read_body(manifest_entry.outbox_id) == {
        "version": 1,
        "manifest": next_manifest.model_dump(mode="json"),
    }


def test_device_only_split_journals_manifest_and_only_shared_tombstone(
    tmp_path, memory_protector, record_factory
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    outbox = ProfileSyncOutbox(service._repository)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    record = record_factory(manifest.profile_id).model_copy(
        update={"scope_id": scope.scope_id}
    )
    service.create_record(record)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"created:{entry.outbox_id}")

    private_record = service.update_record(
        record.record_id,
        RecordMutation(
            controls=record.controls.model_copy(
                update={"sync_mode": SyncMode.DEVICE_ONLY}
            )
        ),
        expected_version_id=record.version_id,
    )

    entries = outbox.list_pending()
    tombstone = service.get_record(record.record_id)
    current_manifest = service.get_manifest()
    assert tombstone is not None
    assert {(entry.object_type, entry.object_id) for entry in entries} == {
        ("manifest", current_manifest.profile_id),
        ("record", tombstone.record_id),
    }
    assert private_record.record_id not in {entry.object_id for entry in entries}
    assert next(
        outbox.read_body(entry.outbox_id)
        for entry in entries
        if entry.object_type == "record"
    ) == {"version": 1, "record": tombstone.model_dump(mode="json")}


def test_terminal_proposal_receipt_replaces_pending_outbox_snapshot(
    tmp_path, memory_protector, proposal_factory
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    outbox = ProfileSyncOutbox(service._repository)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    proposal = proposal_factory(manifest.profile_id)
    service._repository.commit_proposal(proposal)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"pending:{entry.outbox_id}")

    resolved = service._repository.resolve_proposal(
        proposal.proposal_id, ProposalState.REJECTED
    )

    entries = outbox.list_pending()
    assert len(entries) == 1
    assert (entries[0].object_type, entries[0].object_id) == (
        "proposal",
        proposal.proposal_id,
    )
    assert outbox.read_body(entries[0].outbox_id) == {
        "version": 1,
        "proposal": resolved.model_dump(mode="json"),
    }


def test_record_tombstone_keeps_acknowledged_receipt_and_collapses_unsent_parent(
    tmp_path, memory_protector, record_factory
) -> None:
    service = _service(tmp_path, memory_protector)
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    outbox = ProfileSyncOutbox(service._repository)
    for entry in outbox.list_pending():
        outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    record = record_factory(manifest.profile_id).model_copy(
        update={"scope_id": scope.scope_id}
    )
    service.create_record(record)
    first_entry = next(
        entry
        for entry in outbox.list_pending()
        if entry.object_type == "record" and entry.object_id == record.record_id
    )
    outbox.acknowledge(first_entry.outbox_id, "remote-record-v1")
    archived = service.archive_record(
        record.record_id, expected_version_id=record.version_id
    )

    tombstone = service.delete_record(
        record.record_id, expected_version_id=archived.version_id
    )

    pending_record_entries = [
        entry
        for entry in outbox.list_pending()
        if entry.object_type == "record" and entry.object_id == record.record_id
    ]
    assert [entry.version_id for entry in pending_record_entries] == [
        tombstone.version_id
    ]
    assert outbox.get_receipt(first_entry.outbox_id) == "remote-record-v1"


def test_acknowledgement_records_receipt_and_crypto_shreds_body(
    tmp_path, memory_protector
) -> None:
    service = _service(tmp_path, memory_protector)
    service.create_profile()
    outbox = ProfileSyncOutbox(service._repository)
    entry = outbox.list_pending(limit=1)[0]

    outbox.acknowledge(entry.outbox_id, "client-envelope-1")

    assert outbox.read_body(entry.outbox_id) is None
    assert entry.outbox_id not in {
        pending.outbox_id for pending in outbox.list_pending()
    }
    assert outbox.get_receipt(entry.outbox_id) == "client-envelope-1"
    assert "manifest" not in repr(entry)


def test_poisoned_outbox_body_is_quarantined_content_free_and_shredded(
    tmp_path, memory_protector
) -> None:
    service = _service(tmp_path, memory_protector)
    service.create_profile()
    outbox = ProfileSyncOutbox(service._repository)
    outbox_id = service._repository.commit_outbox_body(
        object_type="record",
        object_id="poison-record",
        version_id="poison-version",
        body={"version": 1, "record": {"private": "DO-NOT-PERSIST"}},
    )

    outbox.quarantine(outbox_id, "invalid_canonical_object")

    assert outbox.read_body(outbox_id) is None
    assert outbox.get_quarantine_reason(outbox_id) == "invalid_canonical_object"
    durable = service._repository.db_path.read_bytes()
    assert b"DO-NOT-PERSIST" not in durable
