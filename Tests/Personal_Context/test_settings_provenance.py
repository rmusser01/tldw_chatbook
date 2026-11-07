"""Settings provenance must describe retained metadata without inventing history."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from threading import Event, Thread

import pytest
from tldw_profile_core import ActorType, PreferencePayload, ProfileProposeRequest

from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    RecordMutation,
)
from tldw_chatbook.Personal_Context.settings_provenance import (
    SettingsProfileIdentity,
    project_provenance,
    provenance_subject,
)

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)


@pytest.fixture
def owner(tmp_path, memory_protector):
    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    service = PersonalContextService(repo, clock=lambda: NOW)
    service.create_profile()
    return service, repo


def manual(service, value="PRIVATE_PAYLOAD_MARKER"):
    return service.create_manual_record(
        scope_id=service.list_scopes()[0].scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value=value
        ),
        semantic_key=None,
        controls={"sync_mode": "syncable", "agent_visibility": "user_only"},
    )


def subject_for(service, value):
    manifest = service.get_manifest()
    return provenance_subject(
        SettingsProfileIdentity(manifest.profile_id, manifest.purge_generation), value
    )


def fields(projection):
    return {item.label: item.value for item in projection.fields}


@pytest.mark.parametrize("enabled", [False, True])
def test_private_metadata_is_readable_in_settings_without_canonical_writes(
    owner, enabled
):
    service, repo = owner
    record = manual(service)
    service.set_runtime_enabled(enabled)
    subject = subject_for(service, record)
    before = repo.db_path.read_bytes()
    result = service.settings_provenance(subject)
    assert result.state == "available"
    assert result.projection.reference_status == "No source reference retained"
    assert fields(result.projection)["Recorded source"] == "manual"
    assert fields(result.projection)["Current version ID"] == record.version_id
    assert "PRIVATE_PAYLOAD_MARKER" not in " ".join(fields(result.projection).values())
    assert repo.db_path.read_bytes() == before
    assert record.record_id not in repr(result)
    assert record.profile_id not in repr(subject)
    with pytest.raises(FrozenInstanceError):
        result.state = "changed"


@pytest.mark.parametrize("edited", [False, True])
def test_approval_and_subsequent_edit_do_not_invent_history(owner, edited):
    service, _ = owner
    service.set_runtime_enabled(True)
    proposals = service.proposal_service()
    proposal = proposals.create(
        ProfileProposeRequest(
            operation="create",
            proposed_payload=PreferencePayload(
                subject="response.detail", polarity="like", value="concise"
            ),
        ),
        profile_id=service.get_manifest().profile_id,
        scope_id=service.list_scopes()[0].scope_id,
        turn_id="turn",
        session_id="session",
        evidence_reference="message-legacy",
    )
    record = proposals.accept(
        proposal.proposal_id,
        user_actor=ActorType.USER,
        edited_payload=PreferencePayload(
            subject="response.detail", polarity="like", value="edited"
        )
        if edited
        else None,
    )
    for current in (
        record,
        service.update_record(
            record.record_id,
            RecordMutation(
                payload=PreferencePayload(
                    subject="response.detail",
                    polarity="like",
                    value="later Settings edit",
                )
            ),
            expected_version_id=record.version_id,
        ),
    ):
        detail = project_provenance(subject_for(service, current), current)
        assert (
            detail.reference_status
            == "Legacy source reference — quotation not verified"
        )
        assert fields(detail)["Edit history"] == "Edit history not recorded"
        assert (
            fields(detail)["Inference classification"]
            == "Inference classification not recorded"
        )
        assert fields(detail)["Recorded actor"] == "user"
        assert all("confidence" not in label.lower() for label in fields(detail))
        assert detail.source_references == ("message-legacy",)


@pytest.mark.parametrize(
    "source,actor,reason,origin",
    [
        ("manual", "user", "settings_edit", None),
        ("migration", "system", "legacy_import", None),
        ("agent", "user", "promotion_approved", "source-record"),
    ],
)
def test_recorded_provenance_is_not_reclassified(
    record_factory, source, actor, reason, origin
):
    record = record_factory("profile-synthetic")
    record = type(record).model_validate(
        {
            **record.model_dump(mode="python"),
            "provenance": {
                "source": source,
                "actor": actor,
                "reason_code": reason,
                "derived_from_record_id": origin,
            },
        }
    )
    detail = project_provenance(
        provenance_subject(SettingsProfileIdentity(record.profile_id, 0), record),
        record,
    )
    assert fields(detail)["Recorded source"] == source
    assert fields(detail)["Recorded reason"] == reason
    assert fields(detail)["Promotion origin record ID"] == (origin or "Not recorded")


@pytest.mark.parametrize(
    "refs,hashes,status",
    [
        ((), (), "No source reference retained"),
        ((), ("0" * 64,), "No source reference retained"),
        (
            ("source-a", "source-b"),
            ("0" * 64,),
            "Legacy source reference — quotation not verified",
        ),
    ],
)
def test_references_and_hashes_remain_independent(record_factory, refs, hashes, status):
    record = record_factory("profile")
    record = record.model_copy(
        update={
            "provenance": record.provenance.model_copy(
                update={"source_references": refs, "source_hashes": hashes}
            )
        }
    )
    detail = project_provenance(
        provenance_subject(SettingsProfileIdentity("profile", 0), record), record
    )
    assert (
        detail.source_references,
        detail.source_hashes,
        detail.reference_status,
    ) == (refs, hashes, status)


def test_proposal_fingerprint_covers_envelope_without_base_version_change(
    owner, proposal_factory, monkeypatch
):
    service, repo = owner
    proposal = proposal_factory(service.get_manifest().profile_id)
    repo.commit_proposal(proposal)
    subject = subject_for(service, proposal)
    detail = service.settings_provenance(subject).projection
    assert fields(detail)["Recorded source"] == "agent"  # Proposed record says manual.
    assert (
        fields(detail)["Proposal revision"]
        == "Proposal revision not retained as a canonical field"
    )
    changed = proposal.model_copy(update={"confidence": 0.2})
    assert changed.base_version_id == proposal.base_version_id
    assert subject_for(service, changed) != subject
    # Current proposals are immutable; perturb a read to test the guard itself.
    monkeypatch.setattr(repo, "get_proposal", lambda _identifier: changed)
    assert service.settings_provenance(subject).state == "changed"


def test_deleted_metadata_does_not_load_retired_content(owner, monkeypatch):
    service, repo = owner
    record = manual(service)
    deleted = service.delete_record(
        record.record_id, expected_version_id=record.version_id
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata inspection must not recover Undo")

    monkeypatch.setattr(repo, "get_undo", forbidden)
    monkeypatch.setattr(repo, "list_undo_ids", forbidden)
    snapshot = service.settings_snapshot()
    assert snapshot.records == ()
    assert len(snapshot.deleted_records) == 1
    row = snapshot.deleted_records[0]
    assert row.subject.object_id == record.record_id
    assert not hasattr(row, "payload")
    result = service.settings_provenance(row.subject)
    assert result.state == "available"
    assert fields(result.projection)["State"] == "deleted"
    assert fields(result.projection)["Current version ID"] == deleted.version_id
    assert "PRIVATE_PAYLOAD_MARKER" not in " ".join(fields(result.projection).values())
    assert service.settings_provenance(subject_for(service, record)).state == "changed"


@pytest.mark.parametrize(
    "change", ["version", "identity", "purge", "missing", "quarantine", "locked"]
)
def test_unavailable_or_changed_selection_never_returns_old_metadata(owner, change):
    service, repo = owner
    record = manual(service)
    subject = subject_for(service, record)
    assert service.settings_provenance(subject).state == "available"
    if change == "version":
        service.update_record(
            record.record_id,
            RecordMutation(payload=record.payload),
            expected_version_id=record.version_id,
        )
    elif change == "identity":
        subject = replace(
            subject, profile=replace(subject.profile, profile_id="another-profile")
        )
    elif change == "purge":
        subject = replace(
            subject, profile=replace(subject.profile, purge_generation=99)
        )
    elif change == "missing":
        subject = replace(subject, object_id="missing")
    elif change == "quarantine":
        repo.quarantine_object(
            "record", record.record_id, record.version_id, "integrity_failure"
        )
    else:
        service = PersonalContextService.locked()
    result = service.settings_provenance(subject)
    assert result.state != "available"
    assert result.projection is None


def test_proposal_expiry_check_does_not_run_mutating_sweep(
    owner, proposal_factory, monkeypatch
):
    service, repo = owner
    proposal = proposal_factory(service.get_manifest().profile_id)
    repo.commit_proposal(proposal)
    subject = subject_for(service, proposal)
    service.clock = lambda: proposal.expires_at

    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not run an expiry sweep")

    monkeypatch.setattr(repo, "expire_due_proposals", forbidden)
    assert service.settings_provenance(subject).projection is None
    assert repo.get_proposal(proposal.proposal_id) == proposal


def test_mutation_during_read_is_fenced_and_concurrent_reads_do_not_queue(
    owner, monkeypatch
):
    service, repo = owner
    record = manual(service)
    subject = subject_for(service, record)
    entered, release = Event(), Event()
    original = repo.get_record

    def delayed(record_id):
        value = original(record_id)
        entered.set()
        assert release.wait(5)
        return value

    monkeypatch.setattr(repo, "get_record", delayed)
    results = []
    thread = Thread(target=lambda: results.append(service.settings_provenance(subject)))
    thread.start()
    try:
        assert entered.wait(5)
        assert service.settings_provenance(subject).projection is None
        monkeypatch.setattr(repo, "get_record", original)
        service.update_record(
            record.record_id,
            RecordMutation(payload=record.payload),
            expected_version_id=record.version_id,
        )
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert results[0].state == "changed"
    assert results[0].projection is None


@pytest.mark.parametrize("change", ["foreign", "locked", "purge"])
def test_identity_or_lock_change_during_selected_read_is_rejected(
    owner, monkeypatch, change
):
    service, repo = owner
    record = manual(service)
    subject = subject_for(service, record)
    original = repo.get_record
    calls = 0

    def changing(record_id):
        nonlocal calls
        calls += 1
        value = original(record_id)
        if calls == 2:
            if change == "foreign":
                return value.model_copy(update={"profile_id": "foreign-profile"})
            if change == "locked":
                service._locked_reason = "profile_locked"
            else:
                manifest = service.get_manifest()
                replacement = manifest.model_copy(
                    update={
                        "purge_generation": manifest.purge_generation + 1,
                        "revision": manifest.revision + 1,
                        "current_version_id": "manifest-next",
                    }
                )
                monkeypatch.setattr(repo, "get_manifest", lambda: replacement)
        return value

    monkeypatch.setattr(repo, "get_record", changing)
    assert service.settings_provenance(subject).projection is None
