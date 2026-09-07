from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.notes_organization import organization_link_id
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


PROFILE = "server-a"
DATASET = "dataset-a"
DEVICE = "device-a"


def _id(number: int) -> str:
    return str(uuid.UUID(f"00000000-0000-4000-8000-{number:012d}"))


def _stores(tmp_path: Path):
    notes_path = tmp_path / "notes.sqlite"
    sync_path = tmp_path / "sync.sqlite"
    notes = CharactersRAGDB(notes_path, client_id="intent-dispatch")
    state = SyncStateRepository(sync_path, client_id="intent-dispatch")
    state.set_sync_v2_profile_state(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id=DEVICE,
        dataset_id=DATASET,
    )
    _set_group_state(notes)
    return notes_path, sync_path, notes, state


def _record_keyword_intent(notes: CharactersRAGDB, *, keyword: str = "Old") -> str:
    repository = NotesOrganizationRepository(notes, server_profile_id=PROFILE)
    with notes.transaction() as cursor:
        return repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword",
            object_id=_id(1),
            operation="upsert",
            payload={"keyword": keyword},
            source_version=1,
        )


def _service(notes: CharactersRAGDB, state: SyncStateRepository, **kwargs):
    return NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(notes, server_profile_id=PROFILE),
        state_repository=state,
        **kwargs,
    )


def _set_group_state(notes: CharactersRAGDB, state_name: str = "ready") -> None:
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES (?, ?, ?, 'ready', 'complete', ?)
            ON CONFLICT(server_profile_id, dataset_id) DO UPDATE SET
                local_state = excluded.local_state,
                server_state = excluded.server_state,
                inventory_phase = excluded.inventory_phase,
                updated_at = excluded.updated_at
            """,
            (PROFILE, DATASET, state_name, "2026-08-29T00:00:00+00:00"),
        )


@pytest.mark.parametrize(
    ("local_state", "inventory_phase"),
    [
        ("initializing", "not_started"),
        ("pulling", "not_started"),
        ("adoption_review", "resources"),
        ("failed", "not_started"),
    ],
)
def test_drain_fails_closed_until_server_and_local_checkpoint_are_fully_ready(
    tmp_path, local_state: str, inventory_phase: str
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _record_keyword_intent(notes)
    with notes.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET local_state = ?, "
            "inventory_phase = ? WHERE server_profile_id = ? AND dataset_id = ?",
            (local_state, inventory_phase, PROFILE, DATASET),
        )

    with pytest.raises(ValueError, match="organization group is not ready"):
        _service(notes, state).drain_pending_intents(
            server_profile_id=PROFILE,
            authenticated_principal_id=None,
            workspace_scope=None,
            dataset_id=DATASET,
            device_id=DEVICE,
        )

    assert _outbox(state) == []
    assert (
        _intent(
            notes,
            notes.get_connection()
            .execute("SELECT intent_id FROM notes_organization_sync_intents")
            .fetchone()["intent_id"],
        )["copied_at"]
        is None
    )


def _scope() -> dict[str, object]:
    return {
        "server_profile_id": PROFILE,
        "authenticated_principal_id": None,
        "workspace_scope": None,
    }


def _intent(notes: CharactersRAGDB, intent_id: str):
    return (
        notes.get_connection()
        .execute(
            "SELECT * FROM notes_organization_sync_intents WHERE intent_id = ?",
            (intent_id,),
        )
        .fetchone()
    )


def _outbox(state: SyncStateRepository):
    return state.list_sync_v2_outbox_entries(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
    )


def test_restart_copies_exact_committed_intent_after_crash_before_outbox(
    tmp_path,
) -> None:
    notes_path, sync_path, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes, keyword="Immutable old value")

    def fail(stage: str) -> None:
        if stage == "before_outbox_insert":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        _service(notes, state, failure_injector=fail).drain_pending_intents(
            server_profile_id=PROFILE,
            authenticated_principal_id=None,
            workspace_scope=None,
            dataset_id=DATASET,
            device_id=DEVICE,
        )
    assert _outbox(state) == []
    assert _intent(notes, intent_id)["payload_json"] == json.dumps(
        {"keyword": "Immutable old value"}, separators=(",", ":"), sort_keys=True
    )
    notes.close_connection()
    state.close()

    reopened_notes = CharactersRAGDB(notes_path, client_id="intent-dispatch-reopen")
    reopened_state = SyncStateRepository(sync_path, client_id="intent-dispatch-reopen")
    result = _service(reopened_notes, reopened_state).drain_pending_intents(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        device_id=DEVICE,
    )

    assert result == {"copied": 1, "already_copied": 0}
    rows = _outbox(reopened_state)
    assert len(rows) == 1
    assert rows[0]["client_envelope_id"] == intent_id
    envelope = rows[0]["envelope"]
    assert envelope["payload"] == {"keyword": "Immutable old value"}
    assert envelope["payload_clear"] == {"keyword": "Immutable old value"}
    assert envelope["payload_ciphertext"] is None
    assert envelope["encryption_policy"] == "server_trusted_v1"
    assert _intent(reopened_notes, intent_id)["copied_at"] is not None


def test_restart_after_outbox_insert_reuses_one_row_and_marks_copied(tmp_path) -> None:
    notes_path, sync_path, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes, keyword="Immutable boundary value")

    def fail(stage: str) -> None:
        if stage == "after_outbox_insert":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        _service(notes, state, failure_injector=fail).drain_pending_intents(
            server_profile_id=PROFILE,
            authenticated_principal_id=None,
            workspace_scope=None,
            dataset_id=DATASET,
            device_id=DEVICE,
        )
    assert len(_outbox(state)) == 1
    assert _intent(notes, intent_id)["copied_at"] is None
    notes.close_connection()
    state.close()
    reopened_notes = CharactersRAGDB(notes_path, client_id="outbox-reopen")
    reopened_state = SyncStateRepository(sync_path, client_id="outbox-reopen")

    assert _service(reopened_notes, reopened_state).drain_pending_intents(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        device_id=DEVICE,
    ) == {"copied": 0, "already_copied": 1}
    assert len(_outbox(reopened_state)) == 1
    assert _outbox(reopened_state)[0]["envelope"]["payload"] == {
        "keyword": "Immutable boundary value"
    }
    assert _intent(reopened_notes, intent_id)["copied_at"] is not None


def test_missing_general_row_is_recopied_with_same_logical_operation(tmp_path) -> None:
    _, _, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes)
    service = _service(notes, state)
    arguments = dict(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        device_id=DEVICE,
    )
    service.drain_pending_intents(**arguments)
    state.clear_server_profile_state(server_profile_id=PROFILE)
    state.set_sync_v2_profile_state(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id=DEVICE,
        dataset_id=DATASET,
    )

    service.drain_pending_intents(**arguments)

    rows = _outbox(state)
    assert len(rows) == 1
    assert rows[0]["client_envelope_id"] == intent_id
    assert _intent(notes, intent_id)["outbox_client_envelope_id"] == intent_id


def test_pending_successor_waits_for_ack_then_binds_exact_base_after_restart(
    tmp_path,
) -> None:
    notes_path, sync_path, notes, state = _stores(tmp_path)
    repository = NotesOrganizationRepository(notes, server_profile_id=PROFILE)
    object_id = _id(31)
    with notes.transaction() as cursor:
        first_id = repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            payload={"keyword": "First"},
            source_version=1,
        )
        second_id = repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            payload={"keyword": "Second"},
            source_version=2,
        )

    service = _service(notes, state)
    assert service.drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 1, "already_copied": 0}
    assert [row["client_envelope_id"] for row in _outbox(state)] == [first_id]
    second_before = _intent(notes, second_id)
    assert (
        second_before["base_server_cursor"],
        second_before["base_object_revision"],
        second_before["base_object_hash"],
        second_before["copied_at"],
    ) == (None, None, None, None)

    first_envelope = _outbox(state)[0]["envelope"]
    state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": first_id,
                "server_cursor": 17,
                "object_revision": 1,
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )
    assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 1
    notes.close_connection()
    state.close()

    reopened_notes = CharactersRAGDB(notes_path, client_id="successor-reopen")
    reopened_state = SyncStateRepository(sync_path, client_id="successor-reopen")

    def fail_before_enqueue(stage: str) -> None:
        if stage == "before_outbox_insert":
            raise RuntimeError("crash after base bind")

    with pytest.raises(RuntimeError, match="crash after base bind"):
        _service(
            reopened_notes,
            reopened_state,
            failure_injector=fail_before_enqueue,
        ).drain_pending_intents(**_scope(), dataset_id=DATASET, device_id=DEVICE)
    second_after_crash = _intent(reopened_notes, second_id)
    assert (
        second_after_crash["base_server_cursor"],
        second_after_crash["base_object_revision"],
        second_after_crash["base_object_hash"],
    ) == ("17", 1, first_envelope["payload_hash"])
    assert [row["client_envelope_id"] for row in _outbox(reopened_state)] == [first_id]
    reopened_notes.close_connection()
    reopened_state.close()

    retried_notes = CharactersRAGDB(notes_path, client_id="successor-retry")
    retried_state = SyncStateRepository(sync_path, client_id="successor-retry")
    retried_service = _service(retried_notes, retried_state)
    assert retried_service.drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 1, "already_copied": 0}
    second_after = _intent(retried_notes, second_id)
    assert (
        second_after["base_server_cursor"],
        second_after["base_object_revision"],
        second_after["base_object_hash"],
    ) == ("17", 1, first_envelope["payload_hash"])
    second_outbox = next(
        row for row in _outbox(retried_state) if row["client_envelope_id"] == second_id
    )
    assert (
        second_outbox["envelope"]["base_server_cursor"],
        second_outbox["envelope"]["base_object_revision"],
        second_outbox["envelope"]["base_object_hash"],
    ) == (17, 1, first_envelope["payload_hash"])
    assert retried_service.drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 0, "already_copied": 1}


def test_intent_sequence_orders_resource_before_link_with_identical_timestamps(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    repository = NotesOrganizationRepository(notes, server_profile_id=PROFILE)
    keyword_id = _id(1)
    note_id = _id(101)
    link_payload = {
        "subject_type": "note",
        "subject_id": note_id,
        "keyword_sync_id": keyword_id,
    }
    link_id = organization_link_id("notes.keyword_link", ("note", note_id, keyword_id))
    with notes.transaction() as cursor:
        resource_intent_id = repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword",
            object_id=keyword_id,
            operation="upsert",
            payload={"keyword": "Ordered"},
            source_version=1,
        )
        link_intent_id = repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword_link",
            object_id=link_id,
            operation="upsert",
            payload=link_payload,
            source_version=1,
        )
        assert resource_intent_id > link_intent_id
        cursor.execute("UPDATE notes_organization_sync_intents SET created_at = 'same'")

    assert _service(notes, state).drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 2, "already_copied": 0}
    assert [row["envelope"]["domain"] for row in _outbox(state)] == [
        "notes.keyword",
        "notes.keyword_link",
    ]


def test_missing_or_mismatched_profile_scope_fails_closed(tmp_path) -> None:
    _, _, notes, state = _stores(tmp_path)
    service = _service(notes, state)
    folders = LocalNoteFolderRepository(notes)

    with pytest.raises(ValueError, match="persisted Notes profile"):
        service.create_folder(
            folder_repository=folders,
            name="No partial write",
            parent_id=None,
            server_profile_id="missing",
            authenticated_principal_id=None,
            workspace_scope=None,
        )
    with pytest.raises(ValueError, match="persisted Notes profile"):
        service.create_folder(
            folder_repository=folders,
            name="Wrong principal",
            parent_id=None,
            server_profile_id=PROFILE,
            authenticated_principal_id="other",
            workspace_scope=None,
        )
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM note_folders")
        .fetchone()[0]
        == 0
    )


def test_manual_attach_clears_suppression_and_publishes_effective_union(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes)
    service = _service(notes, state)
    folders = LocalNoteFolderRepository(notes)
    folder = service.create_folder(
        folder_repository=folders, name="Root", parent_id=None, **_scope()
    )
    note_id = notes.add_note("Note", "Body", note_id=_id(25))
    service.mutate_managed_folder_links(
        folder_repository=folders,
        mutation_method="reconcile_managed",
        owner_id="source-a",
        desired=((folder.folder_id, note_id),),
        **_scope(),
    )
    connection = notes.get_connection()
    folder_sync_id = connection.execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (folder.folder_id,)
    ).fetchone()[0]
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at) VALUES (?, ?, ?)",
            (note_id, folder_sync_id, "2026-08-29T00:00:00+00:00"),
        )
    before = connection.execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents WHERE domain = 'notes.folder_link'"
    ).fetchone()[0]

    service.attach_folder_link(
        folder_repository=folders,
        folder_id=folder.folder_id,
        note_id=note_id,
        **_scope(),
    )

    assert (
        connection.execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions WHERE note_id = ? AND folder_sync_id = ?",
            (note_id, folder_sync_id),
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents WHERE domain = 'notes.folder_link'"
        ).fetchone()[0]
        == before + 1
    )


def test_acknowledgement_restarts_without_losing_finalization(tmp_path) -> None:
    notes_path, sync_path, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes, keyword="Immutable ack value")
    service = _service(notes, state)
    arguments = dict(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        device_id=DEVICE,
    )
    service.drain_pending_intents(**arguments)
    state.mark_sync_v2_outbox_push_results(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": intent_id,
                "server_cursor": 17,
                "object_revision": 1,
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )

    def fail(stage: str) -> None:
        if stage == "after_server_acknowledgement":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        _service(notes, state, failure_injector=fail).reconcile_acknowledgements(
            server_profile_id=PROFILE,
            authenticated_principal_id=None,
            workspace_scope=None,
            dataset_id=DATASET,
        )
    assert _intent(notes, intent_id)["acknowledged_at"] is None
    notes.close_connection()
    state.close()
    reopened_notes = CharactersRAGDB(notes_path, client_id="ack-reopen")
    reopened_state = SyncStateRepository(sync_path, client_id="ack-reopen")

    assert (
        _service(reopened_notes, reopened_state).reconcile_acknowledgements(
            server_profile_id=PROFILE,
            authenticated_principal_id=None,
            workspace_scope=None,
            dataset_id=DATASET,
        )
        == 1
    )
    assert _intent(reopened_notes, intent_id)["acknowledged_at"] is not None
    assert _outbox(reopened_state)[0]["envelope"]["payload"] == {
        "keyword": "Immutable ack value"
    }


def test_acknowledgement_persists_the_exact_accepted_intent_as_local_head(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes, keyword="Accepted head")
    service = _service(notes, state)
    service.drain_pending_intents(**_scope(), dataset_id=DATASET, device_id=DEVICE)
    envelope = _outbox(state)[0]["envelope"]
    state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": intent_id,
                "server_cursor": 77,
                "object_revision": envelope["object_revision"],
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )

    assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 1
    head = (
        notes.get_connection()
        .execute(
            "SELECT operation, payload_json, payload_hash, object_revision, "
            "object_hash, server_cursor, apply_state FROM notes_organization_heads "
            "WHERE server_profile_id = ? AND dataset_id = ? AND domain = ? "
            "AND object_id = ?",
            (PROFILE, DATASET, envelope["domain"], envelope["object_id"]),
        )
        .fetchone()
    )
    assert tuple(head) == (
        envelope["operation"],
        json.dumps(envelope["payload"], separators=(",", ":"), sort_keys=True),
        envelope["payload_hash"],
        envelope["object_revision"],
        envelope["payload_hash"],
        "77",
        "applied",
    )


def test_superseded_predecessor_is_terminal_without_synthesizing_lineage(
    tmp_path,
) -> None:
    notes_path, sync_path, notes, state = _stores(tmp_path)
    repository = NotesOrganizationRepository(notes, server_profile_id=PROFILE)
    object_id = _id(32)
    with notes.transaction() as cursor:
        predecessor_id = repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            payload={"keyword": "Superseded"},
            source_version=1,
        )
        successor_id = repository.record_intent(
            cursor,
            profile=PROFILE,
            dataset=DATASET,
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            payload={"keyword": "Blocked successor"},
            source_version=2,
        )
    service = _service(notes, state)
    assert service.drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 1, "already_copied": 0}

    assert state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": predecessor_id,
                "server_cursor": 91,
                "object_revision": 1,
                "apply_status": "superseded",
            }
        ],
        rejected=[],
        conflicts=[],
    ) == {"dispatched": 1, "retained": 0}
    terminal = _outbox(state)[0]
    assert terminal["status"] == "dispatched"
    assert terminal["accepted_result"] == {
        "client_envelope_id": predecessor_id,
        "server_cursor": 91,
        "object_revision": 1,
        "apply_status": "superseded",
    }
    assert terminal["last_error"] == {
        "error_code": "notes_organization_superseded",
        "message": "server superseded the intent without proving object state",
        "retryable": False,
        "review_required": True,
    }
    assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 0
    assert _intent(notes, predecessor_id)["acknowledged_at"] is None
    assert _intent(notes, successor_id)["acknowledged_at"] is None
    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM notes_organization_heads WHERE object_id = ?",
            (object_id,),
        )
        .fetchone()[0]
        == 0
    )
    notes.close_connection()
    state.close()

    reopened_notes = CharactersRAGDB(notes_path, client_id="superseded-reopen")
    reopened_state = SyncStateRepository(sync_path, client_id="superseded-reopen")
    assert _service(reopened_notes, reopened_state).drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 0, "already_copied": 1}
    reopened_terminal = _outbox(reopened_state)[0]
    assert reopened_terminal["accepted_result"]["apply_status"] == "superseded"
    assert reopened_terminal["last_error"]["error_code"] == (
        "notes_organization_superseded"
    )
    assert reopened_terminal["last_error"]["retryable"] is False
    assert [
        row["client_envelope_id"]
        for row in _outbox(reopened_state)
        if row["status"] == "pending"
    ] == []
    assert _intent(reopened_notes, predecessor_id)["acknowledged_at"] is None
    assert _intent(reopened_notes, successor_id)["outbox_client_envelope_id"] is None


def test_acknowledgement_without_complete_server_lineage_fails_closed(tmp_path) -> None:
    _, _, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes, keyword="Incomplete acknowledgement")
    service = _service(notes, state)
    service.drain_pending_intents(**_scope(), dataset_id=DATASET, device_id=DEVICE)
    state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": intent_id,
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )

    assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 0
    assert _intent(notes, intent_id)["acknowledged_at"] is None
    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM notes_organization_heads WHERE object_id = ?",
            (_outbox(state)[0]["envelope"]["object_id"],),
        )
        .fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("apply_status", [None, "pending", "failed"])
def test_unsuccessful_organization_acceptance_remains_retryable_until_applied(
    tmp_path, apply_status: str | None
) -> None:
    notes_path, sync_path, notes, state = _stores(tmp_path)
    intent_id = _record_keyword_intent(notes, keyword="Retry materialization")
    service = _service(notes, state)
    service.drain_pending_intents(**_scope(), dataset_id=DATASET, device_id=DEVICE)
    accepted = {
        "client_envelope_id": intent_id,
        "server_cursor": 81,
        "object_revision": 1,
    }
    if apply_status is not None:
        accepted.update(
            {
                "apply_status": apply_status,
                "apply_error_code": "projection_failed",
                "apply_error_message": "retryable projection failure",
            }
        )

    assert state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[accepted],
        rejected=[],
        conflicts=[],
    ) == {"dispatched": 0, "retained": 1}
    pending = _outbox(state)[0]
    assert pending["status"] == "pending"
    assert pending["accepted_result"].get("apply_status") == apply_status
    if apply_status is not None:
        assert pending["accepted_result"]["apply_error_code"] == "projection_failed"
        assert (
            pending["accepted_result"]["apply_error_message"]
            == "retryable projection failure"
        )
    assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 0
    assert _intent(notes, intent_id)["acknowledged_at"] is None
    notes.close_connection()
    state.close()

    reopened_notes = CharactersRAGDB(notes_path, client_id="materialize-reopen")
    reopened_state = SyncStateRepository(sync_path, client_id="materialize-reopen")
    assert reopened_state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": intent_id,
                "server_cursor": 82,
                "object_revision": 1,
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    ) == {"dispatched": 1, "retained": 0}
    assert (
        _service(reopened_notes, reopened_state).reconcile_acknowledgements(
            **_scope(), dataset_id=DATASET
        )
        == 1
    )
    assert _intent(reopened_notes, intent_id)["acknowledged_at"] is not None


def test_general_outbox_same_id_requires_exact_same_envelope(tmp_path) -> None:
    _, _, notes, state = _stores(tmp_path)
    _record_keyword_intent(notes)
    service = _service(notes, state)
    service.drain_pending_intents(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        device_id=DEVICE,
    )
    original = _outbox(state)[0]
    identical = state.enqueue_sync_v2_outbox_envelope(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id=DATASET,
        envelope=original["envelope"],
    )
    changed = dict(original["envelope"])
    changed["payload"] = {"keyword": "Different"}
    changed["payload_clear"] = {"keyword": "Different"}
    changed["payload_hash"] = "0" * 64

    assert identical["outbox_id"] == original["outbox_id"]
    with pytest.raises(ValueError, match="different envelope"):
        state.enqueue_sync_v2_outbox_envelope(
            server_profile_id=PROFILE,
            authenticated_principal_id=None,
            workspace_scope=None,
            dataset_id=DATASET,
            envelope=changed,
        )
    assert json.dumps(_outbox(state)[0]["envelope"], sort_keys=True) == json.dumps(
        original["envelope"], sort_keys=True
    )


@pytest.mark.parametrize(
    "state_name", ["initializing", "pulling", "adoption_review", "failed"]
)
def test_keyword_and_collection_mutations_reject_pre_ready_without_partial_rows(
    tmp_path, state_name: str
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes, state_name)
    service = _service(notes, state)

    with pytest.raises(ValueError, match="organization group is not ready"):
        service.create_keyword(keyword="Blocked", **_scope())
    with pytest.raises(ValueError, match="organization group is not ready"):
        service.create_keyword_collection(name="Blocked", **_scope())

    assert (
        notes.get_connection().execute("SELECT COUNT(*) FROM keywords").fetchone()[0]
        == 0
    )
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM keyword_collections")
        .fetchone()[0]
        == 0
    )
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


def test_ready_keyword_resources_and_note_conversation_links_share_intent_transaction(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes)
    service = _service(notes, state)
    note_id = notes.add_note("Note", "Body", note_id=_id(20))
    conversation_id = notes.add_conversation({"title": "Conversation"})

    assert service.sync_subject_keywords(
        subject_type="note", subject_id=note_id, keywords=("Lesson",), **_scope()
    ) == ["Lesson"]
    assert service.sync_subject_keywords(
        subject_type="conversation",
        subject_id=conversation_id,
        keywords=("Lesson",),
        **_scope(),
    ) == ["Lesson"]
    service.sync_subject_keywords(
        subject_type="note", subject_id=note_id, keywords=(), **_scope()
    )

    rows = (
        notes.get_connection()
        .execute(
            "SELECT domain, operation, payload_json FROM notes_organization_sync_intents ORDER BY intent_sequence"
        )
        .fetchall()
    )
    assert [(row["domain"], row["operation"]) for row in rows] == [
        ("notes.keyword", "upsert"),
        ("notes.keyword_link", "upsert"),
        ("notes.keyword_link", "upsert"),
        ("notes.keyword_link", "tombstone"),
    ]
    assert notes.get_keywords_for_note(note_id) == []
    assert len(notes.get_keywords_for_conversation(conversation_id)) == 1


def test_ready_collection_resource_and_link_operations_are_journaled(tmp_path) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes)
    service = _service(notes, state)
    keyword_id = service.create_keyword(keyword="Portable", **_scope())
    collection_id = service.create_keyword_collection(name="Lessons", **_scope())

    assert service.set_collection_keyword_link(
        collection_id=collection_id, keyword_id=keyword_id, linked=True, **_scope()
    )
    assert service.set_collection_keyword_link(
        collection_id=collection_id, keyword_id=keyword_id, linked=False, **_scope()
    )
    assert service.mutate_keyword_collection(
        collection_id=collection_id,
        expected_version=1,
        update_data={"name": "Renamed"},
        **_scope(),
    )
    assert service.mutate_keyword_collection(
        collection_id=collection_id, expected_version=2, delete=True, **_scope()
    )

    operations = (
        notes.get_connection()
        .execute(
            "SELECT domain, operation FROM notes_organization_sync_intents ORDER BY intent_sequence"
        )
        .fetchall()
    )
    assert {(row["domain"], row["operation"]) for row in operations} >= {
        ("notes.keyword_collection", "upsert"),
        ("notes.keyword_collection", "tombstone"),
        ("notes.keyword_collection_link", "upsert"),
        ("notes.keyword_collection_link", "tombstone"),
    }


def test_folder_link_intents_follow_effective_union_and_not_owner_provenance(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes)
    service = _service(notes, state)
    folders = LocalNoteFolderRepository(notes)
    folder = service.create_folder(
        folder_repository=folders, name="Root", parent_id=None, **_scope()
    )
    note_id = notes.add_note("Note", "Body", note_id=_id(21))

    service.mutate_managed_folder_links(
        folder_repository=folders,
        mutation_method="reconcile_managed",
        owner_id="source-a",
        desired=((folder.folder_id, note_id),),
        **_scope(),
    )
    service.attach_folder_link(
        folder_repository=folders,
        folder_id=folder.folder_id,
        note_id=note_id,
        **_scope(),
    )
    manual = folders.get_exact_manual_membership(
        folder_id=folder.folder_id, note_id=note_id
    )
    service.detach_folder_link(
        folder_repository=folders,
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_version=manual[0].version,
        **_scope(),
    )
    service.mutate_managed_folder_links(
        folder_repository=folders,
        mutation_method="remove_owner_memberships",
        owner_id="source-a",
        **_scope(),
    )

    link_operations = (
        notes.get_connection()
        .execute(
            "SELECT operation FROM notes_organization_sync_intents WHERE domain = 'notes.folder_link' ORDER BY source_version"
        )
        .fetchall()
    )
    assert [row["operation"] for row in link_operations] == ["upsert", "tombstone"]


def test_folder_link_restore_uses_repository_lineage_and_source_version_allocator(
    tmp_path, monkeypatch
) -> None:
    _, _, notes, state = _stores(tmp_path)
    repository = NotesOrganizationRepository(notes, server_profile_id=PROFILE)
    service = NotesOrganizationSyncService(
        notes_repository=repository,
        state_repository=state,
    )
    folders = LocalNoteFolderRepository(notes)
    folder = service.create_folder(
        folder_repository=folders, name="Root", parent_id=None, **_scope()
    )
    note_id = notes.add_note("Note", "Body", note_id=_id(211))
    service.attach_folder_link(
        folder_repository=folders,
        folder_id=folder.folder_id,
        note_id=note_id,
        **_scope(),
    )
    membership = folders.get_exact_manual_membership(
        folder_id=folder.folder_id, note_id=note_id
    )
    service.detach_folder_link(
        folder_repository=folders,
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_version=membership[0].version,
        **_scope(),
    )

    original_lineage = repository._intent_lineage_with_cursor

    def allocate_from_repository(cursor, **identity):
        lineage = original_lineage(cursor, **identity)
        if identity["domain"] == "notes.folder_link":
            return replace(lineage, next_source_version=41)
        return lineage

    monkeypatch.setattr(
        repository, "_intent_lineage_with_cursor", allocate_from_repository
    )

    service.attach_folder_link(
        folder_repository=folders,
        folder_id=folder.folder_id,
        note_id=note_id,
        **_scope(),
    )

    restored = notes.get_connection().execute(
        "SELECT source_version, routing_metadata_json "
        "FROM notes_organization_sync_intents "
        "WHERE domain = 'notes.folder_link' ORDER BY intent_sequence DESC LIMIT 1"
    ).fetchone()
    assert tuple(restored) == (41, '{"restore_intent":true}')


def test_local_only_profile_keeps_direct_organization_behavior(tmp_path) -> None:
    notes = CharactersRAGDB(tmp_path / "local-notes.sqlite", client_id="local-only")
    state = SyncStateRepository(tmp_path / "local-sync.sqlite", client_id="local-only")
    state.set_sync_v2_profile_state(
        server_profile_id=PROFILE,
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_only",
        device_id=DEVICE,
        dataset_id=DATASET,
    )
    service = _service(notes, state)

    keyword_id = service.create_keyword(keyword="Local", **_scope())
    collection_id = service.create_keyword_collection(
        name="Local collection", **_scope()
    )

    assert notes.get_keyword_by_id(keyword_id)["keyword"] == "Local"
    assert (
        notes.get_keyword_collection_by_id(collection_id)["name"] == "Local collection"
    )
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


def test_explicit_folder_lifecycle_never_emits_derived_descendant_intents(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes)
    service = _service(notes, state)
    folders = LocalNoteFolderRepository(notes)
    parent = service.create_folder(
        folder_repository=folders, name="Parent", parent_id=None, **_scope()
    )
    child = service.create_folder(
        folder_repository=folders, name="Child", parent_id=parent.folder_id, **_scope()
    )
    destination = service.create_folder(
        folder_repository=folders, name="Destination", parent_id=None, **_scope()
    )
    connection = notes.get_connection()
    parent_sync_id = connection.execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (parent.folder_id,)
    ).fetchone()[0]
    child_sync_id = connection.execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (child.folder_id,)
    ).fetchone()[0]
    connection.execute("DELETE FROM notes_organization_sync_intents")
    connection.commit()

    renamed = service.rename_folder(
        folder_repository=folders,
        folder_id=parent.folder_id,
        name="Renamed",
        expected_version=1,
        **_scope(),
    )
    moved = service.move_folder(
        folder_repository=folders,
        folder_id=parent.folder_id,
        parent_id=destination.folder_id,
        expected_version=renamed.folder.version,
        **_scope(),
    )
    deleted = service.delete_folder(
        folder_repository=folders,
        folder_id=parent.folder_id,
        expected_version=moved.folder.version,
        **_scope(),
    )
    pending_intents = connection.execute(
        "SELECT intent_id, source_version FROM notes_organization_sync_intents "
        "WHERE object_id = ? ORDER BY intent_sequence",
        (parent_sync_id,),
    ).fetchall()
    for row in pending_intents:
        assert service.drain_pending_intents(
            **_scope(), dataset_id=DATASET, device_id=DEVICE
        ) == {"copied": 1, "already_copied": 0}
        state.mark_sync_v2_outbox_push_results(
            **_scope(),
            dataset_id=DATASET,
            accepted=[
                {
                    "client_envelope_id": row["intent_id"],
                    "server_cursor": 68 + row["source_version"],
                    "object_revision": row["source_version"],
                    "apply_status": "applied",
                }
            ],
            rejected=[],
            conflicts=[],
        )
        assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 1
    restored = service.restore_folder(
        folder_repository=folders,
        folder_id=parent.folder_id,
        expected_version=deleted.folder.version,
        **_scope(),
    )
    service.rename_folder(
        folder_repository=folders,
        folder_id=parent.folder_id,
        name="After restore",
        expected_version=restored.folder.version,
        **_scope(),
    )

    rows = connection.execute(
        "SELECT object_id, operation FROM notes_organization_sync_intents ORDER BY source_version"
    ).fetchall()
    assert [(row["object_id"], row["operation"]) for row in rows] == [
        (parent_sync_id, "upsert"),
        (parent_sync_id, "upsert"),
        (parent_sync_id, "tombstone"),
        (parent_sync_id, "upsert"),
        (parent_sync_id, "upsert"),
    ]
    assert child_sync_id not in {row["object_id"] for row in rows}
    assert folders.get_folder(child.folder_id, include_deleted=False) is not None

    assert service.drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 1, "already_copied": 0}
    restore_intent = connection.execute(
        "SELECT intent_id, source_version FROM notes_organization_sync_intents "
        "WHERE object_id = ? AND routing_metadata_json = ?",
        (parent_sync_id, '{"restore_intent":true}'),
    ).fetchone()
    state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": restore_intent["intent_id"],
                "server_cursor": 68 + restore_intent["source_version"],
                "object_revision": restore_intent["source_version"],
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )
    assert service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET) == 1
    assert service.drain_pending_intents(
        **_scope(), dataset_id=DATASET, device_id=DEVICE
    ) == {"copied": 1, "already_copied": 0}
    envelopes = [row["envelope"] for row in _outbox(state)]
    assert [envelope["routing_metadata"] for envelope in envelopes] == [
        {},
        {},
        {},
        {"restore_intent": True},
        {},
    ]
    assert (
        envelopes[3]["base_server_cursor"],
        envelopes[3]["base_object_revision"],
        envelopes[3]["base_object_hash"],
    ) == (
        68 + deleted.folder.version,
        deleted.folder.version,
        envelopes[2]["payload_hash"],
    )


def test_convert_managed_provenance_to_manual_emits_nothing_and_keeps_suppression(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes)
    service = _service(notes, state)
    folders = LocalNoteFolderRepository(notes)
    folder = service.create_folder(
        folder_repository=folders, name="Root", parent_id=None, **_scope()
    )
    note_id = notes.add_note("Note", "Body", note_id=_id(22))
    service.mutate_managed_folder_links(
        folder_repository=folders,
        mutation_method="reconcile_managed",
        owner_id="source-a",
        desired=((folder.folder_id, note_id),),
        **_scope(),
    )
    sync_id = (
        notes.get_connection()
        .execute("SELECT sync_id FROM note_folders WHERE id = ?", (folder.folder_id,))
        .fetchone()[0]
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at) VALUES (?, ?, ?)",
            (note_id, sync_id, "2026-08-29T00:00:00+00:00"),
        )
    before = (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents WHERE domain = 'notes.folder_link'"
        )
        .fetchone()[0]
    )

    assert (
        service.mutate_managed_folder_links(
            folder_repository=folders,
            mutation_method="convert_owner_to_manual",
            owner_id="source-a",
            **_scope(),
        )
        == 1
    )

    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents WHERE domain = 'notes.folder_link'"
        )
        .fetchone()[0]
        == before
    )
    assert (
        notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions WHERE note_id = ? AND folder_sync_id = ?",
            (note_id, sync_id),
        )
        .fetchone()[0]
        == 1
    )
    manual = folders.get_exact_manual_membership(
        folder_id=folder.folder_id, note_id=note_id
    )
    assert manual is not None and manual[1] is False


def test_pending_agent_lesson_hook_is_content_only_default_closed_and_write_free(
    tmp_path,
) -> None:
    _, _, notes, state = _stores(tmp_path)
    _set_group_state(notes, "initializing")
    service = _service(notes, state)

    with pytest.raises(ValueError, match="content-only"):
        service.pending_agent_lesson_content_scope(**_scope())
    assert service.pending_agent_lesson_content_scope(
        content_only=True, **_scope()
    ) == {
        "server_profile_id": PROFILE,
        "dataset_id": DATASET,
    }
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )
