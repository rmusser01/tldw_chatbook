from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import uuid

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.agent_lessons import initialize_agent_lessons_folder
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.notes_organization import NOTES_ORGANIZATION_DOMAINS
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope


PROFILE = "server-a"
DATASET = "dataset-a"
NOTE_ID = str(uuid.UUID("00000000-0000-4000-8000-000000000001"))
SCOPE = {
    "server_profile_id": PROFILE,
    "authenticated_principal_id": None,
    "workspace_scope": None,
}


@dataclass
class _Device:
    notes: CharactersRAGDB
    state: SyncStateRepository
    service: NotesOrganizationSyncService
    folders: LocalNoteFolderRepository
    device_id: str
    cursor: int = 0

    def close(self) -> None:
        self.notes.close_connection()
        self.state.close()


@dataclass
class _FakeOrganizationTransport:
    history: list[SyncV2Envelope] = field(default_factory=list)
    heads: dict[tuple[str, str], SyncV2Envelope] = field(default_factory=dict)
    next_cursor: int = 1

    def push(self, device: _Device) -> list[SyncV2Envelope]:
        device.service.drain_pending_intents(
            **SCOPE,
            dataset_id=DATASET,
            device_id=device.device_id,
        )
        pending = device.state.list_sync_v2_outbox_entries(
            **SCOPE,
            dataset_id=DATASET,
            status="pending",
        )
        accepted: list[dict[str, object]] = []
        pushed: list[SyncV2Envelope] = []
        submitted_keys: set[tuple[str, str]] = set()
        for row in pending:
            submitted = SyncV2Envelope.model_validate(row["envelope"])
            key = (submitted.domain, str(submitted.object_id))
            assert key not in submitted_keys
            submitted_keys.add(key)
            current = self.heads.get(key)
            restore_intent = submitted.routing_metadata.get("restore_intent")
            assert restore_intent is None or restore_intent is True
            if restore_intent is True:
                assert submitted.operation == "upsert"
            base = (
                submitted.base_server_cursor,
                submitted.base_object_revision,
                submitted.base_object_hash,
            )
            if current is None:
                assert base == (None, None, None)
                assert restore_intent is None
            else:
                assert base == (
                    current.server_cursor,
                    current.object_revision,
                    current.payload_hash,
                )
                if current.operation == "tombstone" and submitted.operation == "upsert":
                    assert restore_intent is True
                else:
                    assert restore_intent is None
            envelope = submitted.model_copy(update={"server_cursor": self.next_cursor})
            self.next_cursor += 1
            self.history.append(envelope)
            self.heads[key] = envelope
            pushed.append(envelope)
            accepted.append(
                {
                    "client_envelope_id": envelope.client_envelope_id,
                    "server_cursor": envelope.server_cursor,
                    "object_revision": envelope.object_revision,
                    "apply_status": "applied",
                }
            )
        device.state.mark_sync_v2_outbox_push_results(
            **SCOPE,
            dataset_id=DATASET,
            accepted=accepted,
            rejected=[],
            conflicts=[],
        )
        device.service.reconcile_acknowledgements(**SCOPE, dataset_id=DATASET)
        return pushed

    def pull(self, device: _Device) -> list[dict[str, object]]:
        applier = SyncEnvelopeApplier(
            local_store=None,
            notes_organization_repository=NotesOrganizationRepository(
                device.notes, server_profile_id=PROFILE
            ),
        )
        results: list[dict[str, object]] = []
        for envelope in self.history:
            cursor = int(envelope.server_cursor or 0)
            if cursor <= device.cursor or envelope.device_id == device.device_id:
                continue
            results.append(applier.apply(envelope))
            device.cursor = cursor
        return results


def _open_device(root: Path, device_id: str) -> _Device:
    root.mkdir(exist_ok=True)
    notes = CharactersRAGDB(root / "notes.sqlite", client_id=device_id)
    state = SyncStateRepository(root / "sync.sqlite", client_id=device_id)
    state.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        device_id=device_id,
        dataset_id=DATASET,
    )
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT OR REPLACE INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES (?, ?, 'ready', 'ready', 'complete', ?)
            """,
            (PROFILE, DATASET, "2026-08-30T00:00:00+00:00"),
        )
    repository = NotesOrganizationRepository(notes, server_profile_id=PROFILE)
    return _Device(
        notes=notes,
        state=state,
        service=NotesOrganizationSyncService(
            notes_repository=repository,
            state_repository=state,
        ),
        folders=LocalNoteFolderRepository(notes),
        device_id=device_id,
    )


def _resource_id(device: _Device, table: str, *, sync_id: str) -> str | int:
    row = (
        device.notes.get_connection()
        .execute(f"SELECT id FROM {table} WHERE sync_id = ?", (sync_id,))
        .fetchone()
    )
    assert row is not None
    return row["id"]


def _pending_folder_link_count(device: _Device) -> int:
    return int(
        device.notes.get_connection()
        .execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents "
            "WHERE domain = 'notes.folder_link' AND acknowledged_at IS NULL"
        )
        .fetchone()[0]
    )


def test_simultaneous_untouched_agent_lessons_seeds_converge_on_server_winner(
    tmp_path: Path,
) -> None:
    first = _open_device(tmp_path / "seed-first", "device-a")
    second = _open_device(tmp_path / "seed-second", "device-b")
    transport = _FakeOrganizationTransport()
    try:
        for device in (first, second):
            initialize_agent_lessons_folder(
                device.notes,
                scope_mode="synchronized",
                profile_id=PROFILE,
                dataset_id=DATASET,
                organization_repository=NotesOrganizationRepository(
                    device.notes, server_profile_id=PROFILE
                ),
            )

        winner = transport.push(first)[0]
        assert transport.pull(second) == [{"status": "applied"}]
        active = second.notes.get_connection().execute(
            "SELECT sync_id FROM note_folders WHERE name = 'Agent_Lessons' "
            "AND deleted = 0"
        ).fetchall()
        state = second.notes.get_connection().execute(
            "SELECT folder_sync_id FROM agent_lessons_seed_state WHERE "
            "profile_id = ? AND dataset_id = ?",
            (PROFILE, DATASET),
        ).fetchone()
        assert [row["sync_id"] for row in active] == [winner.object_id]
        assert state["folder_sync_id"] == winner.object_id
        assert second.notes.get_connection().execute(
            "SELECT COUNT(*) FROM notes_organization_adoption_reviews"
        ).fetchone()[0] == 0
    finally:
        first.close()
        second.close()


def test_created_folder_intent_uses_the_persisted_portable_identity(
    tmp_path: Path,
) -> None:
    device = _open_device(tmp_path / "identity-device", "device-a")
    try:
        folder = device.service.create_folder(
            folder_repository=device.folders,
            name="Identity invariant",
            parent_id=None,
            **SCOPE,
        )
        connection = device.notes.get_connection()
        persisted_sync_id = connection.execute(
            "SELECT sync_id FROM note_folders WHERE id = ?",
            (folder.folder_id,),
        ).fetchone()[0]
        intent_object_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT object_id FROM notes_organization_sync_intents "
                "WHERE domain = 'notes.folder' ORDER BY intent_sequence"
            )
        ]

        assert intent_object_ids == [persisted_sync_id]
    finally:
        device.close()


def _base(envelope: SyncV2Envelope) -> tuple[int | None, int | None, str | None]:
    return (
        envelope.base_server_cursor,
        envelope.base_object_revision,
        envelope.base_object_hash,
    )


def test_server_shaped_transport_accepts_consecutive_offline_folder_mutations(
    tmp_path: Path,
) -> None:
    device = _open_device(tmp_path / "device", "device-a")
    transport = _FakeOrganizationTransport()
    try:
        destination = device.service.create_folder(
            folder_repository=device.folders,
            name="Destination",
            parent_id=None,
            **SCOPE,
        )
        established = device.service.create_folder(
            folder_repository=device.folders,
            name="Established",
            parent_id=None,
            **SCOPE,
        )
        created = transport.push(device)
        established_sync_id = str(
            device.notes.get_connection()
            .execute(
                "SELECT sync_id FROM note_folders WHERE id = ?",
                (established.folder_id,),
            )
            .fetchone()[0]
        )
        established_head = next(
            envelope
            for envelope in created
            if envelope.object_id == established_sync_id
        )

        renamed = device.service.rename_folder(
            folder_repository=device.folders,
            folder_id=established.folder_id,
            name="Renamed",
            expected_version=established.version,
            **SCOPE,
        )
        moved = device.service.move_folder(
            folder_repository=device.folders,
            folder_id=established.folder_id,
            parent_id=destination.folder_id,
            expected_version=renamed.folder.version,
            **SCOPE,
        )
        deleted = device.service.delete_folder(
            folder_repository=device.folders,
            folder_id=established.folder_id,
            expected_version=moved.folder.version,
            **SCOPE,
        )
        assert deleted.folder.version == moved.folder.version + 1
        rename_push = transport.push(device)
        assert [item.operation for item in rename_push] == ["upsert"]
        assert _base(rename_push[0]) == (
            established_head.server_cursor,
            established_head.object_revision,
            established_head.payload_hash,
        )
        device.close()
        device = _open_device(tmp_path / "device", "device-a")
        move_push = transport.push(device)
        assert [item.operation for item in move_push] == ["upsert"]
        assert _base(move_push[0]) == (
            rename_push[0].server_cursor,
            rename_push[0].object_revision,
            rename_push[0].payload_hash,
        )
        delete_push = transport.push(device)
        assert [item.operation for item in delete_push] == ["tombstone"]
        assert _base(delete_push[0]) == (
            move_push[0].server_cursor,
            move_push[0].object_revision,
            move_push[0].payload_hash,
        )

        offline = device.service.create_folder(
            folder_repository=device.folders,
            name="Offline",
            parent_id=None,
            **SCOPE,
        )
        offline_renamed = device.service.rename_folder(
            folder_repository=device.folders,
            folder_id=offline.folder_id,
            name="Offline renamed",
            expected_version=offline.version,
            **SCOPE,
        )
        offline_moved = device.service.move_folder(
            folder_repository=device.folders,
            folder_id=offline.folder_id,
            parent_id=destination.folder_id,
            expected_version=offline_renamed.folder.version,
            **SCOPE,
        )
        device.service.delete_folder(
            folder_repository=device.folders,
            folder_id=offline.folder_id,
            expected_version=offline_moved.folder.version,
            **SCOPE,
        )
        first_drain = device.service.drain_pending_intents(
            **SCOPE, dataset_id=DATASET, device_id=device.device_id
        )
        second_drain = device.service.drain_pending_intents(
            **SCOPE, dataset_id=DATASET, device_id=device.device_id
        )
        assert first_drain == {"copied": 1, "already_copied": 0}
        assert second_drain == {"copied": 0, "already_copied": 1}

        offline_mutations: list[SyncV2Envelope] = []
        for index in range(4):
            pushed = transport.push(device)
            assert len(pushed) == 1
            offline_mutations.extend(pushed)
            if index == 1:
                device.close()
                device = _open_device(tmp_path / "device", "device-a")
        assert [item.operation for item in offline_mutations] == [
            "upsert",
            "upsert",
            "upsert",
            "tombstone",
        ]
        assert _base(offline_mutations[0]) == (None, None, None)
        for predecessor, mutation in zip(
            offline_mutations[:-1], offline_mutations[1:], strict=True
        ):
            assert _base(mutation) == (
                predecessor.server_cursor,
                predecessor.object_revision,
                predecessor.payload_hash,
            )
    finally:
        device.close()


def test_two_devices_converge_without_cascading_tombstones_or_provenance_leaks(
    tmp_path: Path,
) -> None:
    device_a = _open_device(tmp_path / "a", "device-a")
    device_b = _open_device(tmp_path / "b", "device-b")
    transport = _FakeOrganizationTransport()
    try:
        assert device_a.notes.add_note("Lesson", "private", note_id=NOTE_ID) == NOTE_ID
        assert device_b.notes.add_note("Lesson", "private", note_id=NOTE_ID) == NOTE_ID

        # Force device-local integer identities apart before remote materialization.
        device_b.notes.add_keyword("Device B only")
        device_b.notes.add_keyword_collection("Device B only")
        device_b.folders.create_folder(name="Device B only", parent_id=None)

        keyword_id = device_a.service.create_keyword(keyword="agent-lesson", **SCOPE)
        collection_id = device_a.service.create_keyword_collection(
            name="Lessons", **SCOPE
        )
        assert keyword_id is not None
        assert collection_id is not None
        parent = device_a.service.create_folder(
            folder_repository=device_a.folders,
            name="Agent_Lessons",
            parent_id=None,
            **SCOPE,
        )
        child = device_a.service.create_folder(
            folder_repository=device_a.folders,
            name="Python",
            parent_id=parent.folder_id,
            **SCOPE,
        )
        device_a.service.sync_subject_keywords(
            subject_type="note",
            subject_id=NOTE_ID,
            keywords=("agent-lesson",),
            **SCOPE,
        )
        assert device_a.service.set_collection_keyword_link(
            collection_id=collection_id,
            keyword_id=keyword_id,
            linked=True,
            **SCOPE,
        )
        device_a.service.attach_folder_link(
            folder_repository=device_a.folders,
            folder_id=child.folder_id,
            note_id=NOTE_ID,
            **SCOPE,
        )

        created = transport.push(device_a)
        assert {envelope.domain for envelope in created} == set(
            NOTES_ORGANIZATION_DOMAINS
        )
        assert {result["status"] for result in transport.pull(device_b)} == {"applied"}

        connection_a = device_a.notes.get_connection()
        keyword_sync_id = str(
            connection_a.execute(
                "SELECT sync_id FROM keywords WHERE id = ?", (keyword_id,)
            ).fetchone()[0]
        )
        collection_sync_id = str(
            connection_a.execute(
                "SELECT sync_id FROM keyword_collections WHERE id = ?",
                (collection_id,),
            ).fetchone()[0]
        )
        parent_sync_id = str(
            connection_a.execute(
                "SELECT sync_id FROM note_folders WHERE id = ?", (parent.folder_id,)
            ).fetchone()[0]
        )
        created_parent_head = transport.heads[("notes.folder", parent_sync_id)]
        child_sync_id = str(
            connection_a.execute(
                "SELECT sync_id FROM note_folders WHERE id = ?", (child.folder_id,)
            ).fetchone()[0]
        )
        assert _resource_id(
            device_a, "keywords", sync_id=keyword_sync_id
        ) != _resource_id(device_b, "keywords", sync_id=keyword_sync_id)
        assert _resource_id(
            device_a, "keyword_collections", sync_id=collection_sync_id
        ) != _resource_id(device_b, "keyword_collections", sync_id=collection_sync_id)
        assert _resource_id(
            device_a, "note_folders", sync_id=parent_sync_id
        ) != _resource_id(device_b, "note_folders", sync_id=parent_sync_id)
        assert NotesOrganizationRepository(
            device_b.notes, server_profile_id=PROFILE
        ).effective_folder_sync_ids(NOTE_ID) == (child_sync_id,)

        deleted = device_a.service.delete_folder(
            folder_repository=device_a.folders,
            folder_id=parent.folder_id,
            expected_version=1,
            **SCOPE,
        )
        deletion = transport.push(device_a)
        assert [(item.domain, item.object_id, item.operation) for item in deletion] == [
            ("notes.folder", parent_sync_id, "tombstone")
        ]
        assert _base(deletion[0]) == (
            created_parent_head.server_cursor,
            created_parent_head.object_revision,
            created_parent_head.payload_hash,
        )
        assert transport.pull(device_b) == [{"status": "applied"}]

        connection_b = device_b.notes.get_connection()
        assert (
            connection_b.execute(
                "SELECT deleted FROM note_folders WHERE sync_id = ?", (parent_sync_id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection_b.execute(
                "SELECT deleted FROM note_folders WHERE sync_id = ?", (child_sync_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection_b.execute(
                "SELECT operation FROM notes_organization_heads "
                "WHERE domain = 'notes.folder' AND object_id = ?",
                (child_sync_id,),
            ).fetchone()[0]
            == "upsert"
        )
        assert (
            connection_b.execute(
                "SELECT m.deleted FROM note_folder_memberships m "
                "JOIN note_folders f ON f.id = m.folder_id "
                "WHERE f.sync_id = ? AND m.note_id = ?",
                (child_sync_id, NOTE_ID),
            ).fetchone()[0]
            == 0
        )
        assert (
            NotesOrganizationRepository(
                device_b.notes, server_profile_id=PROFILE
            ).effective_folder_sync_ids(NOTE_ID)
            == ()
        )

        device_a.service.restore_folder(
            folder_repository=device_a.folders,
            folder_id=parent.folder_id,
            expected_version=deleted.folder.version,
            **SCOPE,
        )
        restored = transport.push(device_a)
        assert restored[0].routing_metadata == {"restore_intent": True}
        assert _base(restored[0]) == (
            deletion[0].server_cursor,
            deletion[0].object_revision,
            deletion[0].payload_hash,
        )
        assert transport.pull(device_b) == [{"status": "applied"}]
        assert NotesOrganizationRepository(
            device_b.notes, server_profile_id=PROFILE
        ).effective_folder_sync_ids(NOTE_ID) == (child_sync_id,)

        initial_link_intents = _pending_folder_link_count(device_a)
        device_a.service.mutate_managed_folder_links(
            folder_repository=device_a.folders,
            mutation_method="reconcile_managed",
            owner_id="source-a",
            desired=((child.folder_id, NOTE_ID),),
            **SCOPE,
        )
        assert _pending_folder_link_count(device_a) == initial_link_intents
        manual = device_a.folders.get_exact_manual_membership(
            folder_id=child.folder_id, note_id=NOTE_ID
        )
        assert manual is not None
        device_a.service.detach_folder_link(
            folder_repository=device_a.folders,
            folder_id=child.folder_id,
            note_id=NOTE_ID,
            expected_version=manual[0].version,
            **SCOPE,
        )
        assert _pending_folder_link_count(device_a) == initial_link_intents

        device_a.service.mutate_managed_folder_links(
            folder_repository=device_a.folders,
            mutation_method="remove_owner_memberships",
            owner_id="source-a",
            **SCOPE,
        )
        assert _pending_folder_link_count(device_a) == initial_link_intents + 1
        unlink = transport.push(device_a)
        assert [(item.domain, item.operation) for item in unlink] == [
            ("notes.folder_link", "tombstone")
        ]
        assert transport.pull(device_b) == [{"status": "applied"}]
        assert (
            NotesOrganizationRepository(
                device_b.notes, server_profile_id=PROFILE
            ).effective_folder_sync_ids(NOTE_ID)
            == ()
        )
    finally:
        device_a.close()
        device_b.close()
