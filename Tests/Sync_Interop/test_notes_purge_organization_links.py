"""Delete forever cancels a note's Sync v2 organization links (PR #3423 Qodo 2).

Qodo, on the PR's first head: a purge removed the note's local relationships
but left its pending ``notes.keyword_link`` and ``notes.folder_link`` intents
dispatchable, while it cancelled the note's own publication. A later drain
could then publish a link to a note the server never receives. Review R0's
m-2 (8639dcae88) had kept those intents on purpose -- "A pending link intent
must still reach the server, in order" -- and the controller's ruling now
reverses that, mirroring the publication treatment at purge time:

- the note's unacknowledged link intents are cancelled in the purge's own
  ``BEGIN IMMEDIATE`` transaction, so the organization drain never selects
  them;
- a link upsert the drain already copied to the Sync v2 outbox, unsent, is
  discarded after the purge, as the note's own upserts are;
- an acknowledged link intent stays as lineage, its payload scrubbed to
  ``'{}'``;
- ``notes.keyword`` resource intents are never touched: the keyword outlives
  the note.

And the race the purge-time discard cannot close: the drain reads its
pending intents, the purge commits and discards, then the drain inserts. The
push holds a link upsert for a note with no row back, and heals it, as final
review C-1 does for the note's content.

Real stores throughout: a ``CharactersRAGDB``, a ``SyncStateRepository``
with a local-first profile, the real ``NotesOrganizationSyncService`` on a
ready group, the real ``NotesScopeService.purge_note``, and
``LocalFirstSyncService.sync_once`` against the in-memory server fake the C-1
pins use.

Base (``734ce8022f``): the drain after a purge copies both links; a copied
link survives the purge in the outbox; an acknowledged link keeps the note's
id in its payload; the raced link is pushed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Tests.Sync_Interop.test_local_first_sync_service import (
    FakeLocalFirstServer,
    RecordingLocalStore,
    _repo_with_profile,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SCOPE = {
    "server_profile_id": "server-a",
    "authenticated_principal_id": "user-a",
    "workspace_scope": "workspace-1",
}
DATASET = "dataset-1"
DEVICE = "device-1"
USER = "default_user"
DOMAINS = [
    "notes",
    "notes.note",
    "notes.keyword",
    "notes.keyword_link",
    "notes.folder",
    "notes.folder_link",
]
LINKS = {"notes.keyword_link", "notes.folder_link"}


def _subject(domain: str, payload: dict) -> str | None:
    if domain == "notes.folder_link":
        return payload.get("note_id")
    if domain == "notes.keyword_link" and payload.get("subject_type") == "note":
        return payload.get("subject_id")
    return None


class _Linked:
    """Two notes with a keyword link and a folder link each; one in the Trash."""

    def __init__(self, tmp_path, monkeypatch) -> None:
        from tldw_chatbook.RAG_Search import ingestion_indexing

        monkeypatch.setattr(
            ingestion_indexing, "remove_purged_note_projection", lambda note_id: 1
        )
        self.state = _repo_with_profile(
            tmp_path, capabilities={"supported_domains": DOMAINS}
        )
        self.db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="qodo-2")
        with self.db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO notes_organization_sync_checkpoints(server_profile_id, "
                "dataset_id, local_state, server_state, inventory_phase, updated_at) "
                "VALUES (?, ?, 'ready', 'ready', 'complete', '2026-11-06T00:00:00Z')",
                (SCOPE["server_profile_id"], DATASET),
            )
        self.org = NotesOrganizationSyncService(
            notes_repository=NotesOrganizationRepository(
                self.db, server_profile_id=SCOPE["server_profile_id"]
            ),
            state_repository=self.state,
        )
        self.service = NotesScopeService(
            local_notes_service=NotesInteropService(
                base_db_directory=tmp_path,
                api_client_id="qodo-2",
                global_db_to_use=self.db,
            ),
            server_service=None,
            organization_sync_service=self.org,
            sync_v2_notes_producer=SimpleNamespace(state_repository=self.state),
        )
        folders = LocalNoteFolderRepository(self.db)
        folder = self.org.create_folder(
            folder_repository=folders, name="Projects", parent_id=None, **SCOPE
        )
        self.note = self.db.add_note("Gone note", "secret body")
        self.keeper = self.db.add_note("Keeper", "stays")
        for note_id in (self.note, self.keeper):
            self.org.sync_subject_keywords(
                subject_type="note", subject_id=note_id, keywords=("Lesson",), **SCOPE
            )
            self.org.attach_folder_link(
                folder_repository=folders,
                folder_id=folder.folder_id,
                note_id=note_id,
                **SCOPE,
            )
        version = self.db.get_note_by_id(self.note)["version"]
        self.db.soft_delete_note(self.note, version)
        self.version = version + 1

    def drain(self) -> None:
        self.org.drain_pending_intents(**SCOPE, dataset_id=DATASET, device_id=DEVICE)

    async def purge(self) -> bool:
        return await self.service.purge_note(
            scope=ScopeType.LOCAL_NOTE,
            note_id=self.note,
            version=self.version,
            user_id=USER,
            sync_runtime=None,
        )

    def outbox(self) -> set[tuple[str, str, str | None]]:
        """``(domain, operation, subject note)`` for every outbox row."""
        return {
            (
                row["envelope"]["domain"],
                row["envelope"]["operation"],
                _subject(row["envelope"]["domain"], row["envelope"]["payload"]),
            )
            for row in self.state.list_sync_v2_outbox_entries(
                **SCOPE, dataset_id=DATASET
            )
        }

    def intents(self) -> list[dict]:
        return [
            dict(row)
            for row in self.db.get_connection()
            .execute(
                "SELECT intent_id, domain, operation, payload_json, payload_hash, "
                "acknowledged_at FROM notes_organization_sync_intents "
                "ORDER BY intent_sequence"
            )
            .fetchall()
        ]

    def links_of(self, note_id: str) -> set[tuple[str, str, str]]:
        return {
            (domain, operation, subject)
            for domain, operation, subject in self.outbox()
            if domain in LINKS and subject == note_id
        }


async def test_after_a_purge_the_drain_sends_no_link_to_the_note(
    tmp_path, monkeypatch
) -> None:
    linked = _Linked(tmp_path, monkeypatch)
    resources_before = [
        row
        for row in linked.intents()
        if row["domain"] in {"notes.keyword", "notes.folder"}
    ]

    assert await linked.purge() is True
    linked.drain()

    assert linked.links_of(linked.note) == set(), linked.outbox()
    assert linked.links_of(linked.keeper) == {
        ("notes.keyword_link", "upsert", linked.keeper),
        ("notes.folder_link", "upsert", linked.keeper),
    }
    # The keyword and the folder outlive the note: their resource intents
    # are untouched and still go out.
    assert [
        row
        for row in linked.intents()
        if row["domain"] in {"notes.keyword", "notes.folder"}
    ] == resources_before
    assert {("notes.keyword", "upsert", None), ("notes.folder", "upsert", None)} <= (
        linked.outbox()
    )


async def test_a_link_the_drain_already_copied_is_discarded_by_the_purge(
    tmp_path, monkeypatch
) -> None:
    linked = _Linked(tmp_path, monkeypatch)
    linked.drain()
    assert linked.links_of(linked.note) == {
        ("notes.keyword_link", "upsert", linked.note),
        ("notes.folder_link", "upsert", linked.note),
    }

    assert await linked.purge() is True

    assert linked.links_of(linked.note) == set(), linked.outbox()
    assert len(linked.links_of(linked.keeper)) == 2
    # Nothing left for a later drain to copy again.
    linked.drain()
    assert linked.links_of(linked.note) == set()


async def test_an_acknowledged_link_intent_stays_as_lineage_without_the_note_id(
    tmp_path, monkeypatch
) -> None:
    linked = _Linked(tmp_path, monkeypatch)
    linked.drain()
    accepted = [
        {
            "client_envelope_id": row["client_envelope_id"],
            "server_cursor": cursor,
            "object_revision": row["envelope"]["object_revision"],
            "apply_status": "applied",
        }
        for cursor, row in enumerate(
            linked.state.list_sync_v2_outbox_entries(**SCOPE, dataset_id=DATASET),
            start=10,
        )
    ]
    linked.state.mark_sync_v2_outbox_push_results(
        **SCOPE, dataset_id=DATASET, accepted=accepted, rejected=[], conflicts=[]
    )
    assert linked.org.reconcile_acknowledgements(**SCOPE, dataset_id=DATASET) == len(
        accepted
    )
    note_links = {
        row["intent_id"]: row
        for row in linked.intents()
        if row["domain"] in LINKS
        and _subject(row["domain"], json.loads(row["payload_json"])) == linked.note
    }
    assert len(note_links) == 2

    assert await linked.purge() is True

    after = {row["intent_id"]: row for row in linked.intents()}
    for intent_id, before in note_links.items():
        kept = after[intent_id]
        assert kept["acknowledged_at"] == before["acknowledged_at"] is not None
        assert kept["payload_json"] == "{}"
        assert kept["payload_hash"] == before["payload_hash"]
    assert not [row for row in after.values() if linked.note in row["payload_json"]]
    # The keeper's acknowledged links keep their payloads.
    assert sum(linked.keeper in row["payload_json"] for row in after.values()) == 2


async def test_a_link_the_drain_copies_after_the_purge_is_never_pushed(
    tmp_path, monkeypatch
) -> None:
    """The race: the drain read the note's pending links, then the purge
    committed and ran its outbox discard, then the drain inserted them."""
    linked = _Linked(tmp_path, monkeypatch)
    raced: list[str] = []

    def purge_mid_drain(stage: str) -> None:
        if stage == "before_outbox_insert" and not raced:
            raced.append(stage)
            # ``purge_note``'s two steps, between the drain's read and insert.
            assert linked.db.hard_delete_note(
                linked.note, expected_version=linked.version
            )
            linked.state.discard_sync_v2_note_history(linked.note)

    linked.org.failure_injector = purge_mid_drain
    linked.drain()
    linked.org.failure_injector = None
    assert raced, "the drain never reached its outbox insert"
    assert linked.links_of(linked.note) == {
        ("notes.keyword_link", "upsert", linked.note),
        ("notes.folder_link", "upsert", linked.note),
    }, "the race this pin needs did not happen"
    server = FakeLocalFirstServer()
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=linked.state,
        local_store=RecordingLocalStore(),
        dataset_keys={DATASET: generate_dataset_key()},
        note_exists=linked.db.note_ids_present,
    )

    await service.sync_once(**SCOPE, domains=DOMAINS)

    pushed = {
        (
            envelope["domain"],
            envelope["operation"],
            _subject(envelope["domain"], envelope["payload"]),
        )
        for envelope in server.calls[0][3]
    }
    assert not [entry for entry in pushed if entry[2] == linked.note], pushed
    assert {
        ("notes.keyword_link", "upsert", linked.keeper),
        ("notes.folder_link", "upsert", linked.keeper),
    } <= pushed
    # Healed, so the next cycle does not hold them back again.
    assert linked.links_of(linked.note) == set()
