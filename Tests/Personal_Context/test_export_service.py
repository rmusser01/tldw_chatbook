from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from loguru import logger
from tldw_profile_core import PreferencePayload, ProfileRecord
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_chatbook.Personal_Context.export_service import (
    ExportRequest,
    PersonalContextExportError,
    RecoveryExportRequest,
    RecoverySnapshot,
    _decode_snapshot,
    load_recovery_export,
)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    RecordMutation,
)


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def _service(tmp_path, memory_protector, record_factory):
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    record = record_factory(manifest.profile_id, value="EXPORT-CONTENT-CANARY")
    record = ProfileRecord.model_validate(
        {**record.model_dump(mode="python"), "scope_id": scope.scope_id}
    )
    service.create_record(record)
    return service, record


def test_plaintext_export_requires_explicit_absolute_destination_and_confirmation(
    tmp_path, memory_protector, record_factory, monkeypatch
) -> None:
    service, _ = _service(tmp_path, memory_protector, record_factory)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PersonalContextExportError, match="absolute"):
        service.export_plaintext(
            ExportRequest(destination="profile.json", confirm_plaintext=True)
        )


@pytest.mark.parametrize("recovery", [False, True])
def test_export_rejects_existing_destination_without_explicit_overwrite(
    tmp_path, memory_protector, record_factory, recovery: bool
) -> None:
    service, _ = _service(tmp_path, memory_protector, record_factory)
    destination = tmp_path / "existing-profile.json"
    destination.write_text("do-not-replace", encoding="utf-8")

    with pytest.raises(PersonalContextExportError, match="overwrite"):
        if recovery:
            service.export_recovery(
                RecoveryExportRequest(
                    destination=destination,
                    passphrase="export passphrase",
                )
            )
        else:
            service.export_plaintext(
                ExportRequest(destination=destination, confirm_plaintext=True)
            )

    assert destination.read_text(encoding="utf-8") == "do-not-replace"


@pytest.mark.parametrize("recovery", [False, True])
def test_export_replaces_existing_destination_only_with_explicit_overwrite(
    tmp_path, memory_protector, record_factory, recovery: bool
) -> None:
    service, _ = _service(tmp_path, memory_protector, record_factory)
    destination = tmp_path / "existing-profile.json"
    destination.write_text("replace-me", encoding="utf-8")

    if recovery:
        service.export_recovery(
            RecoveryExportRequest(
                destination=destination,
                passphrase="export passphrase",
                confirm_overwrite=True,
            )
        )
    else:
        service.export_plaintext(
            ExportRequest(
                destination=destination,
                confirm_plaintext=True,
                confirm_overwrite=True,
            )
        )

    assert destination.read_text(encoding="utf-8") != "replace-me"
    with pytest.raises(PersonalContextExportError, match="confirmation"):
        service.export_plaintext(
            ExportRequest(
                destination=tmp_path / "profile.json", confirm_plaintext=False
            )
        )


def test_plaintext_export_contains_only_canonical_eligible_data(
    tmp_path, memory_protector, record_factory, proposal_factory
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    service.set_runtime_enabled(True)
    service._repository.commit_outbox_body(
        object_type="record",
        object_id=record.record_id,
        version_id=record.version_id,
        body={"secret": "OUTBOX-EXCLUDED-CANARY"},
    )
    proposal = proposal_factory(record.profile_id)
    proposal = type(proposal).model_validate(
        {
            **proposal.model_dump(mode="python"),
            "scope_id": record.scope_id,
            "proposed_record": {
                **proposal.proposed_record.model_dump(mode="python"),
                "scope_id": record.scope_id,
            },
        }
    )
    service._repository.commit_proposal(proposal)
    service.update_record(
        record.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value="UPDATED-CANARY"
            )
        ),
        expected_version_id=record.version_id,
    )
    destination = tmp_path / "profile.json"

    service.export_plaintext(
        ExportRequest(destination=destination, confirm_plaintext=True)
    )

    payload = json.loads(destination.read_text())
    text = destination.read_text()
    assert payload["format"] == "tldw-personal-context-plaintext-v1"
    assert payload["manifest"]["revision"] == 2
    assert payload["records"][0]["payload"]["value"] == "UPDATED-CANARY"
    assert payload["proposals"][0]["proposal_id"] == proposal.proposal_id
    for excluded in (
        "runtime_policy",
        "scope_binding",
        "undo",
        "outbox",
        "quarantine",
        "OUTBOX-EXCLUDED-CANARY",
    ):
        assert excluded not in text


def test_export_snapshot_is_one_sqlite_read_generation(
    tmp_path, memory_protector, record_factory
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    with sqlite3.connect(service._repository.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    concurrent_service = PersonalContextService(
        PersonalContextRepository(
            service._repository.db_path,
            key_protector=memory_protector,
        ),
        clock=lambda: NOW,
    )
    snapshot_started = threading.Event()
    mutation_finished = threading.Event()
    original_decrypt = service._repository._decrypt_row

    def pause_after_manifest(row):
        plaintext = original_decrypt(row)
        if row["object_type"] == "manifest" and not snapshot_started.is_set():
            snapshot_started.set()
            assert mutation_finished.wait(5)
        return plaintext

    service._repository._decrypt_row = pause_after_manifest

    def mutate_after_snapshot_starts():
        assert snapshot_started.wait(5)
        try:
            return concurrent_service.update_record(
                record.record_id,
                RecordMutation(
                    payload=PreferencePayload(
                        subject="response.detail",
                        polarity="like",
                        value="CONCURRENT-EXPORT-CANARY",
                    )
                ),
                expected_version_id=record.version_id,
            )
        finally:
            mutation_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot_future = executor.submit(service.snapshot_for_export)
        mutation_future = executor.submit(mutate_after_snapshot_starts)
        snapshot = snapshot_future.result(timeout=5)
        changed = mutation_future.result(timeout=5)

    manifest, scopes, records, proposals = snapshot
    assert manifest.revision == 1
    assert scopes == service.list_scopes()
    assert records == (record,)
    assert proposals == ()
    assert changed.payload.value == "CONCURRENT-EXPORT-CANARY"


def test_plaintext_export_can_select_scopes(
    tmp_path, memory_protector, record_factory
) -> None:
    service, global_record = _service(tmp_path, memory_protector, record_factory)
    workspace = service.create_workspace_scope("workspace-1", "Workspace One")
    workspace_record = ProfileRecord.model_validate(
        {
            **global_record.model_dump(mode="python"),
            "record_id": "workspace-record",
            "version_id": "workspace-record-v1",
            "scope_id": workspace.scope_id,
            "semantic_key": {
                "namespace": "preference",
                "subject": "workspace.detail",
            },
            "payload": {
                "kind": "preference",
                "subject": "workspace.detail",
                "polarity": "like",
                "value": "workspace-only",
            },
        }
    )
    service.create_record(workspace_record)
    destination = tmp_path / "workspace.json"

    service.export_plaintext(
        ExportRequest(
            destination=destination,
            confirm_plaintext=True,
            scope_ids=(workspace.scope_id,),
        )
    )
    payload = json.loads(destination.read_text())
    assert [item["scope_id"] for item in payload["records"]] == [workspace.scope_id]


def test_recovery_export_is_independently_encrypted_and_round_trips(
    tmp_path, memory_protector, record_factory
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    destination = tmp_path / "profile.tldw-profile-recovery"

    service.export_recovery(
        RecoveryExportRequest(destination=destination, passphrase="export passphrase")
    )

    durable = destination.read_bytes()
    assert b"EXPORT-CONTENT-CANARY" not in durable
    snapshot = load_recovery_export(destination, "export passphrase")
    assert snapshot.manifest.profile_id == record.profile_id
    assert snapshot.records == (record,)
    assert snapshot.scopes == service.list_scopes()
    with pytest.raises(PersonalContextExportError, match="unlock"):
        load_recovery_export(destination, "wrong passphrase")


def test_recovery_preserves_current_tombstones_while_plaintext_omits_them(
    tmp_path, memory_protector, record_factory
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    tombstone = service.delete_record(
        record.record_id, expected_version_id=record.version_id
    )
    recovery_path = tmp_path / "deleted.tldw-profile-recovery"
    plaintext_path = tmp_path / "deleted.json"

    service.export_recovery(
        RecoveryExportRequest(destination=recovery_path, passphrase="recovery secret")
    )
    service.export_plaintext(
        ExportRequest(destination=plaintext_path, confirm_plaintext=True)
    )

    recovered = load_recovery_export(recovery_path, "recovery secret")
    assert recovered.records == (tombstone,)
    assert recovered.records[0].payload is None
    assert json.loads(plaintext_path.read_text())["records"] == []


def test_recovery_round_trip_preserves_pending_proposal_with_stale_base(
    tmp_path, memory_protector, record_factory, proposal_factory
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    proposed_record = ProfileRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "payload": {
                "kind": "preference",
                "subject": "response.detail",
                "polarity": "like",
                "value": "PROPOSED-STALE-CANARY",
            },
            "version_id": "proposed-stale-version",
            "parent_version_id": record.version_id,
        }
    )
    proposal = proposal_factory(record.profile_id)
    proposal = type(proposal).model_validate(
        {
            **proposal.model_dump(mode="python"),
            "scope_id": record.scope_id,
            "operation": "update",
            "target_record_id": record.record_id,
            "base_version_id": record.version_id,
            "proposed_record": proposed_record,
        }
    )
    service._repository.commit_proposal(proposal)
    service.update_record(
        record.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail",
                polarity="like",
                value="LATER-CURRENT-CANARY",
            )
        ),
        expected_version_id=record.version_id,
    )
    destination = tmp_path / "stale-proposal.tldw-profile-recovery"

    service.export_recovery(
        RecoveryExportRequest(destination=destination, passphrase="recovery secret")
    )

    recovered = load_recovery_export(destination, "recovery secret")
    assert recovered.proposals == (proposal,)
    assert recovered.records[0].version_id != proposal.base_version_id


def test_malformed_recovery_and_symlink_destination_fail_closed(
    tmp_path, memory_protector, record_factory
) -> None:
    service, _ = _service(tmp_path, memory_protector, record_factory)
    malformed = tmp_path / "malformed.tldw-profile-recovery"
    malformed.write_text('{"version":1,"ciphertext":"bad"}')
    with pytest.raises(PersonalContextExportError):
        load_recovery_export(malformed, "passphrase")

    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    destination = tmp_path / "linked.tldw-profile-recovery"
    destination.symlink_to(target)
    with pytest.raises(Exception):
        service.export_recovery(
            RecoveryExportRequest(destination=destination, passphrase="passphrase")
        )
    assert target.read_bytes() == b"unchanged"


def test_export_never_logs_content_paths_or_passphrases(
    tmp_path, memory_protector, record_factory
) -> None:
    service, _ = _service(tmp_path, memory_protector, record_factory)
    destination = tmp_path / "PRIVATE-PATH-CANARY.tldw-profile-recovery"
    messages: list[str] = []
    sink = logger.add(messages.append)
    try:
        service.export_recovery(
            RecoveryExportRequest(
                destination=destination, passphrase="PASSPHRASE-CANARY"
            )
        )
    finally:
        logger.remove(sink)
    rendered = "".join(messages)
    assert "EXPORT-CONTENT-CANARY" not in rendered
    assert "PRIVATE-PATH-CANARY" not in rendered
    assert "PASSPHRASE-CANARY" not in rendered


def test_sensitive_export_and_mutation_dataclasses_have_content_free_repr(
    tmp_path, memory_protector, record_factory
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    mutation = RecordMutation(
        payload=PreferencePayload(
            subject="response.detail",
            polarity="like",
            value="MUTATION-BODY-CANARY",
        )
    )
    export_request = ExportRequest(
        destination="/PRIVATE-DESTINATION-CANARY",
        confirm_plaintext=True,
    )
    recovery_request = RecoveryExportRequest(
        destination="/RECOVERY-DESTINATION-CANARY",
        passphrase="RECOVERY-PASSPHRASE-CANARY",
    )
    snapshot = RecoverySnapshot(
        service.get_manifest(), service.list_scopes(), (record,), ()
    )

    rendered = "\n".join(
        map(repr, (mutation, export_request, recovery_request, snapshot))
    )
    for secret in (
        "MUTATION-BODY-CANARY",
        "PRIVATE-DESTINATION-CANARY",
        "RECOVERY-DESTINATION-CANARY",
        "RECOVERY-PASSPHRASE-CANARY",
        "EXPORT-CONTENT-CANARY",
    ):
        assert secret not in rendered


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body.update(scopes=[]),
        lambda body: body["scopes"].append(body["scopes"][0]),
        lambda body: body["scopes"].append(
            {
                **body["scopes"][0],
                "scope_id": "second-global",
                "version_id": "second-global-version",
            }
        ),
        lambda body: body["records"].append(body["records"][0]),
        lambda body: body["proposals"].append(body["proposals"][0]),
        lambda body: body["records"][0].update(profile_id="other-profile"),
        lambda body: body["proposals"][0].update(scope_id="other-scope"),
        lambda body: body["records"][0].update(
            parent_version_id=body["records"][0]["version_id"]
        ),
        lambda body: body["records"].append(
            {
                **body["records"][0],
                "record_id": "duplicate-key-record",
                "version_id": "duplicate-key-version",
            }
        ),
        lambda body: body["records"].append(
            {
                **body["records"][0],
                "record_id": "duplicate-version-record",
                "semantic_key": {
                    "namespace": "preference",
                    "subject": "different.subject",
                },
                "payload": {
                    "kind": "preference",
                    "subject": "different.subject",
                    "polarity": "like",
                    "value": "different",
                },
            }
        ),
        lambda body: body["records"].append(
            {
                **body["records"][0],
                "record_id": "cross-parent-record",
                "version_id": "cross-parent-version",
                "parent_version_id": body["records"][0]["version_id"],
                "semantic_key": {
                    "namespace": "preference",
                    "subject": "cross.parent.subject",
                },
                "payload": {
                    "kind": "preference",
                    "subject": "cross.parent.subject",
                    "polarity": "like",
                    "value": "cross-parent",
                },
            }
        ),
    ],
)
def test_recovery_snapshot_rejects_missing_or_duplicate_and_cross_linked_objects(
    tmp_path, memory_protector, record_factory, proposal_factory, mutation
) -> None:
    service, record = _service(tmp_path, memory_protector, record_factory)
    proposal = proposal_factory(record.profile_id)
    proposal = type(proposal).model_validate(
        {
            **proposal.model_dump(mode="python"),
            "scope_id": record.scope_id,
            "proposed_record": {
                **proposal.proposed_record.model_dump(mode="python"),
                "scope_id": record.scope_id,
            },
        }
    )
    service._repository.commit_proposal(proposal)
    snapshot = {
        "format": "tldw-personal-context-snapshot-v1",
        "manifest": service.get_manifest().model_dump(mode="json"),
        "scopes": [scope.model_dump(mode="json") for scope in service.list_scopes()],
        "records": [record.model_dump(mode="json")],
        "proposals": [proposal.model_dump(mode="json")],
    }
    mutation(snapshot)

    with pytest.raises(PersonalContextExportError, match="invalid"):
        _decode_snapshot(canonical_json_bytes(snapshot))
