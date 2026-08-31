from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
    NotesOrganizationRepositoryError,
)
from tldw_chatbook.Sync_Interop.conflict_review import SyncV2ConflictReviewService
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.notes_outbox_producer import NotesSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


PROFILE = "server-a"
DATASET = "dataset-a"
SCOPE = {
    "server_profile_id": PROFILE,
    "authenticated_principal_id": None,
    "workspace_scope": None,
}


def _services(tmp_path: Path, *, ready: bool = False, failure_injector=None):
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="receipt-tests")
    state = SyncStateRepository(tmp_path / "sync.sqlite", client_id="receipt-tests")
    key = generate_dataset_key()
    state.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id=DATASET,
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, "
            "inventory_phase, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                PROFILE,
                DATASET,
                "ready" if ready else "initializing",
                "ready" if ready else "initializing",
                "complete" if ready else "not_started",
                "2026-08-30T00:00:00+00:00",
            ),
        )
    producer = NotesSyncV2OutboxProducer(
        state_repository=state,
        dataset_keys={DATASET: key},
        notes_db=notes,
    )
    organization = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(notes, server_profile_id=PROFILE),
        state_repository=state,
        notes_producer=producer,
        failure_injector=failure_injector,
    )
    library = NotesInteropService(
        tmp_path,
        "receipt-tests",
        global_db_to_use=notes,
    )
    library._db_instances["user-a"] = notes
    return notes, state, library, organization, producer


def _save_pending(library: NotesInteropService, *, receipt_id: str = "receipt-a"):
    return library.save_note_with_organization(
        "user-a",
        title="Pending lesson",
        content="Verified content",
        folder="Agent_Lessons",
        ensure_keywords=("agent-lesson",),
        receipt_id=receipt_id,
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )


def _make_ready(notes: CharactersRAGDB) -> None:
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET "
            "local_state = 'ready', server_state = 'ready', "
            "inventory_phase = 'complete', error_code = NULL "
            "WHERE server_profile_id = ? AND dataset_id = ?",
            (PROFILE, DATASET),
        )


def test_blocking_receipt_excludes_direct_producer_and_legacy_sync_log_after_restart(
    tmp_path: Path,
) -> None:
    notes, state, library, _organization, producer = _services(tmp_path)
    saved = _save_pending(library)

    direct = producer.enqueue_note_upsert(
        **SCOPE,
        note_id=saved["id"],
        title="Pending lesson",
        content="Verified content",
        entity_version=1,
    )
    assert direct == {"status": "skipped", "reason": "pending_organization"}
    # ManualSyncControl only previews dependencies from this legacy projection;
    # it cannot itself enqueue or publish a note.
    assert notes.get_sync_log_entries(entity_type="notes") == []
    assert state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET) == []

    restarted = NotesSyncV2OutboxProducer(
        state_repository=state,
        dataset_keys={DATASET: generate_dataset_key()},
        notes_db=notes,
    )
    assert restarted.enqueue_note_upsert(
        **SCOPE,
        note_id=saved["id"],
        title="Pending lesson",
        content="Verified content",
        entity_version=1,
    ) == {"status": "skipped", "reason": "pending_organization"}


def test_ready_finalization_commits_note_resource_and_link_intents_then_clears_receipt(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library)
    _make_ready(notes)

    first = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    second = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    connection = notes.get_connection()
    assert first == {"finalized": 1, "placement_review": 0, "cancelled": 0}
    assert second == {"finalized": 0, "placement_review": 0, "cancelled": 0}
    assert connection.execute(
        "SELECT COUNT(*) FROM note_organization_receipts"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_sync_publication_intents WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 1
    assert {
        str(row[0])
        for row in connection.execute(
            "SELECT domain FROM notes_organization_sync_intents ORDER BY domain"
        )
    } == {
        "notes.folder",
        "notes.folder_link",
        "notes.keyword",
        "notes.keyword_link",
    }
    assert connection.execute(
        "SELECT COUNT(*) FROM note_keywords WHERE note_id = ?", (saved["id"],)
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ? AND deleted = 0",
        (saved["id"],),
    ).fetchone()[0] == 1
    assert state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET) == []

    copied = organization.drain_pending_note_intents(**SCOPE)
    repeated = organization.drain_pending_note_intents(**SCOPE)
    entries = state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET)
    assert copied == {"copied": 1, "already_copied": 0}
    assert repeated == {"copied": 0, "already_copied": 1}
    assert len([entry for entry in entries if entry["domain"] == "notes"]) == 1


def test_direct_drain_after_ready_restart_finalizes_before_dispatch(
    tmp_path: Path,
) -> None:
    notes, state, library, _organization, producer = _services(tmp_path)
    saved = _save_pending(library, receipt_id="receipt-direct-drain-restart")
    notes_path = notes.db_path
    notes.close_connection()

    reopened = CharactersRAGDB(notes_path, client_id="receipt-direct-drain-restart")
    _make_ready(reopened)
    restarted_producer = NotesSyncV2OutboxProducer(
        state_repository=state,
        dataset_keys=producer.dataset_keys,
        notes_db=reopened,
    )
    restarted = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            reopened, server_profile_id=PROFILE
        ),
        state_repository=state,
        notes_producer=restarted_producer,
    )

    assert restarted.drain_pending_note_intents(**SCOPE) == {
        "copied": 1,
        "already_copied": 0,
    }
    connection = reopened.get_connection()
    assert connection.execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_sync_publication_intents WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 1
    entries = state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET)
    assert len([entry for entry in entries if entry["domain"] == "notes"]) == 1


def test_scoped_note_intent_survives_restart_and_cannot_drain_to_another_profile(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, producer = _services(tmp_path)
    state.set_sync_v2_profile_state(
        server_profile_id="server-b",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-b",
        dataset_id="dataset-b",
    )
    producer.dataset_keys["dataset-b"] = generate_dataset_key()
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    notes_path = notes.db_path
    notes.close_connection()
    reopened = CharactersRAGDB(notes_path, client_id="receipt-tests-restarted")
    restarted_producer = NotesSyncV2OutboxProducer(
        state_repository=state,
        dataset_keys=producer.dataset_keys,
        notes_db=reopened,
    )
    restarted = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            reopened, server_profile_id=PROFILE
        ),
        state_repository=state,
        notes_producer=restarted_producer,
    )

    wrong_scope = restarted.drain_pending_note_intents(
        server_profile_id="server-b",
        authenticated_principal_id=None,
        workspace_scope=None,
    )
    right_scope = restarted.drain_pending_note_intents(**SCOPE)
    replay = restarted.drain_pending_note_intents(**SCOPE)

    assert wrong_scope == {"copied": 0, "already_copied": 0}
    assert state.list_sync_v2_outbox_entries(
        server_profile_id="server-b",
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id="dataset-b",
    ) == []
    assert right_scope == {"copied": 1, "already_copied": 0}
    assert replay == {"copied": 0, "already_copied": 1}
    assert state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET)[0][
        "envelope"
    ]["object_id"] == saved["id"]
    reopened.close_connection()


def test_finalization_does_not_lose_scoped_intent_to_same_version_legacy_row(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library)
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, "
            "version, payload) VALUES ('notes', ?, 'create', ?, ?, 1, ?)",
            (
                saved["id"],
                "2026-08-30T00:00:00+00:00",
                "legacy-client",
                json.dumps({"id": saved["id"], "version": 1}),
            ),
        )
    _make_ready(notes)

    assert organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    ) == {"finalized": 1, "placement_review": 0, "cancelled": 0}
    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 1,
        "already_copied": 0,
    }
    envelope = state.list_sync_v2_outbox_entries(
        **SCOPE, dataset_id=DATASET
    )[0]["envelope"]
    assert envelope["object_id"] == saved["id"]


def test_finalized_intent_survives_note_mutation_and_restart_before_drain(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, producer = _services(tmp_path)
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    assert library.update_note(
        "user-a",
        saved["id"],
        {"title": "Later local edit", "content": "Not the finalized payload"},
        expected_version=1,
    )
    notes_path = notes.db_path
    notes.close_connection()
    reopened = CharactersRAGDB(notes_path, client_id="receipt-tests-restarted")
    restarted = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(
            reopened, server_profile_id=PROFILE
        ),
        state_repository=state,
        notes_producer=NotesSyncV2OutboxProducer(
            state_repository=state,
            dataset_keys=producer.dataset_keys,
            notes_db=reopened,
        ),
    )

    assert restarted.drain_pending_note_intents(**SCOPE) == {
        "copied": 1,
        "already_copied": 0,
    }
    envelope = state.list_sync_v2_outbox_entries(
        **SCOPE, dataset_id=DATASET
    )[0]["envelope"]
    assert envelope["object_id"] == saved["id"]
    assert envelope["entity_version"] == 1
    reopened.close_connection()


def test_finalized_intent_is_retained_until_general_outbox_acknowledgement(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    organization.drain_pending_note_intents(**SCOPE)
    outbox = state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET)[0]

    state.mark_sync_v2_outbox_push_results(
        **SCOPE,
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": outbox["client_envelope_id"],
                "server_cursor": 1,
                "object_revision": 1,
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )
    assert organization.reconcile_acknowledgements(
        **SCOPE,
        dataset_id=DATASET,
    ) == 1
    row = notes.get_connection().execute(
        "SELECT acknowledged_at FROM note_sync_publication_intents WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()
    assert row["acknowledged_at"] is not None
    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 0,
        "already_copied": 0,
    }


def test_finalized_update_preserves_immutable_base_and_entity_versions(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    note_id = str(
        library.add_note(
            "user-a",
            "Before",
            "Before body",
            note_id="00000000-0000-4000-8000-000000000077",
        )
    )
    current = library.get_library_note_text(
        "user-a", note_id, start=0, max_chars=100
    )
    updated = library.save_note_with_organization(
        "user-a",
        note_id=note_id,
        expected_version=1,
        expected_organization_version=current["organization_version"],
        title="After",
        content="After body",
        folder="Agent_Lessons",
        ensure_keywords=("agent-lesson",),
        receipt_id="receipt-update",
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    _make_ready(notes)

    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    organization.drain_pending_note_intents(**SCOPE)

    envelope = state.list_sync_v2_outbox_entries(
        **SCOPE, dataset_id=DATASET
    )[0]["envelope"]
    assert updated["version"] == 2
    assert envelope["object_id"] == note_id
    assert envelope["base_version"] == 1
    assert envelope["entity_version"] == 2


def _finalize_two_versions_with_reversed_clocks(
    notes: CharactersRAGDB,
    library: NotesInteropService,
    organization: NotesOrganizationSyncService,
) -> dict:
    created = _save_pending(library, receipt_id="receipt-version-1")
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    current = library.get_library_note_text(
        "user-a", created["id"], start=0, max_chars=100
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET "
            "local_state = 'pulling', server_state = 'initializing' "
            "WHERE server_profile_id = ? AND dataset_id = ?",
            (PROFILE, DATASET),
        )
    library.save_note_with_organization(
        "user-a",
        note_id=created["id"],
        expected_version=1,
        expected_organization_version=current["organization_version"],
        title="Pending lesson",
        content="Verified content",
        folder="Agent_Lessons",
        ensure_keywords=("agent-lesson",),
        receipt_id="receipt-version-2",
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE note_sync_publication_intents SET created_at = CASE "
            "WHEN entity_version = 1 THEN '2026-08-30T00:00:02+00:00' "
            "ELSE '2026-08-30T00:00:01+00:00' END WHERE note_id = ?",
            (created["id"],),
        )
    return created


def test_reversed_clocks_drain_same_note_versions_once_in_lineage_order(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    created = _finalize_two_versions_with_reversed_clocks(
        notes, library, organization
    )
    assert [
        int(row[0])
        for row in notes.get_connection().execute(
            "SELECT entity_version FROM note_sync_publication_intents "
            "WHERE note_id = ? ORDER BY created_at, intent_id",
            (created["id"],),
        )
    ] == [2, 1]

    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 2,
        "already_copied": 0,
    }
    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 0,
        "already_copied": 2,
    }
    entries = state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET)
    note_entries = [entry for entry in entries if entry["domain"] == "notes"]
    assert [entry["envelope"]["entity_version"] for entry in note_entries] == [1, 2]
    assert [entry["envelope"]["base_version"] for entry in note_entries] == [None, 1]
    assert len({entry["client_envelope_id"] for entry in note_entries}) == 2


def test_dispatch_order_is_deterministic_across_notes_and_scope_safe(
    tmp_path: Path,
) -> None:
    notes, _state, library, _organization, _producer = _services(tmp_path, ready=True)
    note_a = str(
        library.add_note(
            "user-a", "A", "A body", note_id="00000000-0000-4000-8000-000000000001"
        )
    )
    note_b = str(
        library.add_note(
            "user-a", "B", "B body", note_id="00000000-0000-4000-8000-000000000002"
        )
    )
    with notes.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO note_sync_publication_intents("
            "intent_id, server_profile_id, dataset_id, note_id, operation, "
            "base_version, entity_version, request_fingerprint, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)",
            (
                ("b-v1", PROFILE, DATASET, note_b, "create", None, 1, "a" * 64, "00"),
                ("a-v2", PROFILE, DATASET, note_a, "update", 1, 2, "b" * 64, "01"),
                ("a-v1", PROFILE, DATASET, note_a, "create", None, 1, "c" * 64, "02"),
                ("other", "server-b", DATASET, note_a, "update", 2, 3, "d" * 64, "03"),
            ),
        )

    rows = notes.list_latest_dispatchable_note_sync_entries(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    assert [(row["entity_id"], row["version"]) for row in rows] == [
        (note_a, 1),
        (note_a, 2),
        (note_b, 1),
    ]


def test_finalized_create_receipt_retry_returns_original_note_without_mutation(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    created = _save_pending(library, receipt_id="receipt-create-replay")
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    replayed = _save_pending(library, receipt_id="receipt-create-replay")

    assert replayed["id"] == created["id"]
    assert replayed["version"] == created["version"] == 1
    assert replayed["receipt_state"] is None
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM notes WHERE deleted = 0"
    ).fetchone()[0] == 1
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_sync_publication_intents WHERE intent_id = ?",
        ("receipt-create-replay",),
    ).fetchone()[0] == 1


def test_finalized_create_receipt_rejects_mismatched_retry_before_mutation(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    created = _save_pending(library, receipt_id="receipt-create-conflict")
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    with pytest.raises(NotesOrganizationRepositoryError) as error:
        library.save_note_with_organization(
            "user-a",
            title="Different lesson",
            content="Different content",
            folder="Agent_Lessons",
            ensure_keywords=("agent-lesson",),
            receipt_id="receipt-create-conflict",
            server_profile_id=PROFILE,
            dataset_id=DATASET,
        )

    assert error.value.reason_code == "receipt_conflict"
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM notes WHERE deleted = 0"
    ).fetchone()[0] == 1
    note_row = notes.get_connection().execute(
        "SELECT id, version FROM notes WHERE deleted = 0"
    ).fetchone()
    assert tuple(note_row) == (created["id"], 1)


def test_finalization_failure_rolls_back_every_note_owner_change(tmp_path: Path) -> None:
    def fail(stage: str) -> None:
        if stage == "after_receipt_finalization_intents":
            raise RuntimeError(stage)

    notes, _state, library, organization, _producer = _services(
        tmp_path, failure_injector=fail
    )
    saved = _save_pending(library)
    _make_ready(notes)

    with pytest.raises(RuntimeError, match="after_receipt_finalization_intents"):
        organization.finalize_pending_note_organization_receipts(
            server_profile_id=PROFILE,
            dataset_id=DATASET,
        )

    connection = notes.get_connection()
    assert connection.execute(
        "SELECT state FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == "pending_organization"
    assert connection.execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_keywords WHERE note_id = ?", (saved["id"],)
    ).fetchone()[0] == 0


def test_finalizer_refuses_changed_note_or_organization_state(tmp_path: Path) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library)
    with notes.transaction() as cursor:
        keyword_id = notes.add_keyword("concurrent-user-keyword", cursor=cursor)
        notes.link_note_to_keyword(saved["id"], int(keyword_id), cursor=cursor)
    _make_ready(notes)

    result = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    assert result == {"finalized": 0, "placement_review": 0, "cancelled": 0}
    connection = notes.get_connection()
    assert connection.execute(
        "SELECT state FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == "pending_organization"
    assert connection.execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0


def test_profile_scoped_finalizer_leaves_another_profiles_receipt_untouched(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library)
    with notes.transaction() as cursor:
        raw = cursor.execute(
            "SELECT requested_keywords_json FROM note_organization_receipts "
            "WHERE note_id = ?",
            (saved["id"],),
        ).fetchone()[0]
        stored = json.loads(raw)
        stored[-1]["_request"]["server_profile_id"] = "server-b"
        stored[-1]["_request"]["dataset_id"] = "dataset-b"
        cursor.execute(
            "UPDATE note_organization_receipts SET requested_keywords_json = ? "
            "WHERE note_id = ?",
            (json.dumps(stored), saved["id"]),
        )
    _make_ready(notes)

    result = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    assert result == {"finalized": 0, "placement_review": 0, "cancelled": 0}
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 1


def test_finalizer_fails_closed_for_malformed_request_fingerprint(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library)
    with notes.transaction() as cursor:
        raw = cursor.execute(
            "SELECT requested_keywords_json FROM note_organization_receipts "
            "WHERE note_id = ?",
            (saved["id"],),
        ).fetchone()[0]
        stored = json.loads(raw)
        stored[-1]["_request"]["fingerprint"] = "not-a-sha256"
        cursor.execute(
            "UPDATE note_organization_receipts SET requested_keywords_json = ? "
            "WHERE note_id = ?",
            (json.dumps(stored), saved["id"]),
        )
    _make_ready(notes)

    result = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    assert result == {"finalized": 0, "placement_review": 0, "cancelled": 0}
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 1
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_sync_publication_intents WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0


def test_folder_collision_transitions_same_receipt_to_nonblocking_review(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, producer = _services(tmp_path)
    collision = LocalNoteFolderRepository(notes).create_folder(
        name="agent_lessons", parent_id=None
    )
    saved = _save_pending(library)
    _make_ready(notes)

    result = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    receipt = notes.get_connection().execute(
        "SELECT receipt_id, state, review_id, collision_ids_json "
        "FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()

    assert result == {"finalized": 0, "placement_review": 1, "cancelled": 0}
    assert receipt["receipt_id"] == "receipt-a"
    assert receipt["state"] == "placement_review"
    assert receipt["review_id"]
    assert collision.folder_id in receipt["collision_ids_json"]
    assert producer.enqueue_note_upsert(
        **SCOPE,
        note_id=saved["id"],
        title="Pending lesson",
        content="Verified content",
        entity_version=1,
    )["status"] == "enqueued"


@pytest.mark.parametrize("action", ("merge", "rename_local", "keep_local"))
def test_resolved_or_dismissed_placement_review_retires_receipt(
    tmp_path: Path, action: str
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    collision = LocalNoteFolderRepository(notes).create_folder(
        name="agent_lessons", parent_id=None
    )
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    receipt = notes.get_connection().execute(
        "SELECT review_id FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()
    resolved = SyncV2ConflictReviewService(
        state_repository=object(),
        notes_repository=NotesOrganizationRepository(notes, server_profile_id=PROFILE),
        notes_organization_sync_service=organization,
    ).resolve_notes_organization_adoption(
        review_id=str(receipt["review_id"]),
        action=action,
        new_name="Former_Agent_Lessons" if action == "rename_local" else None,
    )

    result = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    assert resolved is True
    assert result == {"finalized": 0, "placement_review": 0, "cancelled": 0}
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0
    memberships = notes.get_connection().execute(
        "SELECT folder_id FROM note_folder_memberships WHERE note_id = ? AND deleted = 0",
        (saved["id"],),
    ).fetchall()
    assert bool(memberships) is (action != "keep_local")
    if action == "merge":
        assert memberships[0][0] == collision.folder_id


def test_resolved_placement_review_waits_when_organization_loses_readiness(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    collision = LocalNoteFolderRepository(notes).create_folder(
        name="agent_lessons", parent_id=None
    )
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    receipt = notes.get_connection().execute(
        "SELECT review_id FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET local_state = 'pulling' "
            "WHERE server_profile_id = ? AND dataset_id = ?",
            (PROFILE, DATASET),
        )

    resolver = SyncV2ConflictReviewService(
        state_repository=object(),
        notes_repository=NotesOrganizationRepository(notes, server_profile_id=PROFILE),
        notes_organization_sync_service=organization,
    )
    with pytest.raises(ValueError, match="not ready"):
        resolver.resolve_notes_organization_adoption(
            review_id=str(receipt["review_id"]), action="merge"
        )

    result = organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    assert result == {"finalized": 0, "placement_review": 0, "cancelled": 0}
    assert notes.get_connection().execute(
        "SELECT state FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == "placement_review"
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_folder_memberships "
        "WHERE note_id = ? AND folder_id = ? AND deleted = 0",
        (saved["id"], collision.folder_id),
    ).fetchone()[0] == 0
    assert notes.get_connection().execute(
        "SELECT state FROM notes_organization_adoption_reviews WHERE review_id = ?",
        (receipt["review_id"],),
    ).fetchone()[0] == "open"


def test_rename_local_resolution_records_both_folder_resource_intents(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    LocalNoteFolderRepository(notes).create_folder(
        name="agent_lessons", parent_id=None
    )
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    review_id = notes.get_connection().execute(
        "SELECT review_id FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0]
    SyncV2ConflictReviewService(
        state_repository=object(),
        notes_repository=NotesOrganizationRepository(notes, server_profile_id=PROFILE),
        notes_organization_sync_service=organization,
    ).resolve_notes_organization_adoption(
        review_id=str(review_id),
        action="rename_local",
        new_name="Former_Agent_Lessons",
    )

    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )

    folder_payloads = {
        __import__("json").loads(str(row[0]))["name"]
        for row in notes.get_connection().execute(
            "SELECT payload_json FROM notes_organization_sync_intents "
            "WHERE domain = 'notes.folder'"
        )
    }
    assert folder_payloads == {"Former_Agent_Lessons", "Agent_Lessons"}


def test_soft_delete_atomically_cancels_pending_or_review_receipt_and_review(
    tmp_path: Path,
) -> None:
    notes, _state, library, organization, _producer = _services(tmp_path)
    LocalNoteFolderRepository(notes).create_folder(name="agent_lessons", parent_id=None)
    saved = _save_pending(library)
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    review_id = notes.get_connection().execute(
        "SELECT review_id FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0]

    assert notes.soft_delete_note(saved["id"], 1) is True
    connection = notes.get_connection()
    assert connection.execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0
    review = connection.execute(
        "SELECT state, resolution FROM notes_organization_adoption_reviews "
        "WHERE review_id = ?",
        (review_id,),
    ).fetchone()
    assert tuple(review) == ("resolved", "keep_local")
    assert connection.execute(
        "SELECT cancelled_at FROM note_sync_publication_intents WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] is not None
    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 0,
        "already_copied": 0,
    }


def test_soft_deleted_pending_create_keeps_receipt_id_terminal_on_retry(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library, receipt_id="receipt-deleted-pending-create")

    assert notes.soft_delete_note(saved["id"], 1) is True
    with pytest.raises(NotesOrganizationRepositoryError) as error:
        _save_pending(library, receipt_id="receipt-deleted-pending-create")

    assert error.value.reason_code == "receipt_conflict"
    connection = notes.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM notes WHERE deleted = 0"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_organization_receipts"
    ).fetchone()[0] == 0
    _make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 0,
        "already_copied": 0,
    }
    assert state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET) == []


def test_finalizer_cancellation_keeps_deleted_create_receipt_id_terminal(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, _producer = _services(tmp_path)
    saved = _save_pending(library, receipt_id="receipt-finalizer-cancelled-create")
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE notes SET deleted = 1, version = 2 WHERE id = ?",
            (saved["id"],),
        )
    _make_ready(notes)

    assert organization.finalize_pending_note_organization_receipts(
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    ) == {"finalized": 0, "placement_review": 0, "cancelled": 1}
    with pytest.raises(NotesOrganizationRepositoryError) as error:
        _save_pending(library, receipt_id="receipt-finalizer-cancelled-create")

    assert error.value.reason_code == "receipt_conflict"
    with pytest.raises(NotesOrganizationRepositoryError) as mismatch:
        library.save_note_with_organization(
            "user-a",
            title="Different pending lesson",
            content="Different verified content",
            folder="Agent_Lessons",
            ensure_keywords=("agent-lesson",),
            receipt_id="receipt-finalizer-cancelled-create",
            server_profile_id=PROFILE,
            dataset_id=DATASET,
        )

    assert mismatch.value.reason_code == "receipt_conflict"
    connection = notes.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
    assert connection.execute(
        "SELECT cancelled_at FROM note_sync_publication_intents WHERE intent_id = ?",
        ("receipt-finalizer-cancelled-create",),
    ).fetchone()[0] is not None
    assert organization.drain_pending_note_intents(**SCOPE) == {
        "copied": 0,
        "already_copied": 0,
    }
    assert state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET) == []


class _RecordingServer:
    def __init__(self) -> None:
        self.pushed: list[dict] = []

    async def push_v2_envelopes(self, **kwargs):
        self.pushed.extend(kwargs["envelopes"])
        return {
            "dataset_id": kwargs["dataset_id"],
            "accepted": [
                {"client_envelope_id": item["client_envelope_id"]}
                for item in kwargs["envelopes"]
            ],
            "next_cursor": "1",
        }

    async def pull_v2_envelopes(self, **kwargs):
        return {
            "dataset_id": kwargs["dataset_id"],
            "envelopes": [],
            "next_cursor": "1",
            "has_more": False,
        }


class _LocalStore:
    conflicts: list[dict] = []


@pytest.mark.asyncio
async def test_general_outbox_drain_filters_an_already_enqueued_blocking_note(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, producer = _services(tmp_path)
    note_id = library.add_note("user-a", "Earlier", "Earlier body", note_id="note-a")
    assert producer.enqueue_note_upsert(
        **SCOPE,
        note_id=str(note_id),
        title="Earlier",
        content="Earlier body",
        entity_version=1,
    )["status"] == "enqueued"
    current = library.get_library_note_text("user-a", str(note_id), start=0, max_chars=20)
    library.save_note_with_organization(
        "user-a",
        note_id=str(note_id),
        expected_version=1,
        expected_organization_version=current["organization_version"],
        title="Now pending",
        content="Now pending body",
        folder="Agent_Lessons",
        ensure_keywords=("agent-lesson",),
        receipt_id="receipt-existing-outbox",
        server_profile_id=PROFILE,
        dataset_id=DATASET,
    )
    server = _RecordingServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=state,
        local_store=_LocalStore(),
        dataset_keys=producer.dataset_keys,
        notes_organization_repository=organization.notes_repository,
        notes_organization_sync_service=organization,
    )

    await service.sync_once(**SCOPE, domains=["notes"])

    assert server.pushed == []
    assert len(state.list_pending_sync_v2_outbox_envelopes(
        **SCOPE, dataset_id=DATASET, domains=["notes"]
    )) == 1


@pytest.mark.asyncio
async def test_ready_receipt_finalizes_before_the_same_normal_drain(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, producer = _services(tmp_path)
    saved = _save_pending(library)
    _make_ready(notes)
    server = _RecordingServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=state,
        local_store=_LocalStore(),
        dataset_keys=producer.dataset_keys,
        notes_organization_repository=organization.notes_repository,
        notes_organization_sync_service=organization,
    )

    await service.sync_once(**SCOPE, domains=["notes"])

    assert [envelope["object_id"] for envelope in server.pushed] == [saved["id"]]
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_local_first_submits_reversed_clock_lineage_once_in_version_order(
    tmp_path: Path,
) -> None:
    notes, state, library, organization, producer = _services(tmp_path)
    created = _finalize_two_versions_with_reversed_clocks(
        notes, library, organization
    )
    server = _RecordingServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=state,
        local_store=_LocalStore(),
        dataset_keys=producer.dataset_keys,
        notes_organization_repository=organization.notes_repository,
        notes_organization_sync_service=organization,
    )

    first = await service.sync_once(**SCOPE, domains=["notes"])
    second = await service.sync_once(**SCOPE, domains=["notes"])

    pushed = [item for item in server.pushed if item["object_id"] == created["id"]]
    assert [item["entity_version"] for item in pushed] == [1, 2]
    assert [item["base_version"] for item in pushed] == [None, 1]
    assert len({item["client_envelope_id"] for item in pushed}) == 2
    dispatched = state.list_sync_v2_outbox_entries(
        **SCOPE,
        dataset_id=DATASET,
        status="dispatched",
    )
    assert [item["envelope"]["entity_version"] for item in dispatched] == [1, 2]
    assert first["outbox_dispatched"] == 2
    assert second["outbox_drained"] == 0
