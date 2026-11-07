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

PR #3423 re-review 3 (I-1, m-1): cancelling is for links the server may
never have had. A pending link TOMBSTONE whose predecessor is acknowledged,
or absent (its base is then the server's head), is the user's unlink of a
link the server holds -- review R0 m-2's "a pending server-side delete must
still propagate" -- so the purge keeps it, id-only, and it is sent and
acknowledged like any other. Unacknowledged upserts, and tombstones with no
acknowledged state behind them, are still cancelled. An upsert the drain
already copied is cancelled too, together with its outbox row (keeping the
intent would let every later drain copy it again); a copied tombstone is
kept, so its acknowledgement finds it and moves the head.

Real stores throughout: a ``CharactersRAGDB``, a ``SyncStateRepository``
with a local-first profile, the real ``NotesOrganizationSyncService`` on a
ready group, the real ``NotesScopeService.purge_note``, and
``LocalFirstSyncService.sync_once`` against the in-memory server fake the C-1
pins use (the re-review 3 pins with a fake that applies and acknowledges
what it is pushed, and one two-device pin on the server-shaped organization
transport of ``test_notes_organization_two_device``, which checks every
pushed base against the server's head).

Base (``734ce8022f``): the drain after a purge copies both links; a copied
link survives the purge in the outbox; an acknowledged link keeps the note's
id in its payload; the raced link is pushed. Base for the re-review 3 pins
(``67ddead5e9``): the purge deletes every pending unlink, so an uncopied one
never reaches the server, and a copied one's acknowledgement leaves the head
on ``upsert``; and a link held back from a ``payload_clear``-only envelope
is never healed (re-review 3 m-3).
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
from Tests.Sync_Interop.test_notes_organization_two_device import (
    NOTE_ID as TWO_DEVICE_NOTE_ID,
    SCOPE as TWO_DEVICE_SCOPE,
    _FakeOrganizationTransport,
    _open_device,
    _resource_id,
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
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

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
        self.folders = LocalNoteFolderRepository(self.db)
        self.folder = self.org.create_folder(
            folder_repository=self.folders, name="Projects", parent_id=None, **SCOPE
        )
        self.note = self.db.add_note("Gone note", "secret body")
        self.keeper = self.db.add_note("Keeper", "stays")
        for note_id in (self.note, self.keeper):
            self.link(note_id)
        version = self.db.get_note_by_id(self.note)["version"]
        self.db.soft_delete_note(self.note, version)
        self.version = version + 1

    def link(self, note_id: str) -> None:
        """Tag the note "Lesson" and file it under Projects."""
        self.org.sync_subject_keywords(
            subject_type="note", subject_id=note_id, keywords=("Lesson",), **SCOPE
        )
        self.org.attach_folder_link(
            folder_repository=self.folders,
            folder_id=self.folder.folder_id,
            note_id=note_id,
            **SCOPE,
        )

    def unlink(self, note_id: str) -> None:
        """Remove the note's keyword and take it out of Projects."""
        self.org.sync_subject_keywords(
            subject_type="note", subject_id=note_id, keywords=(), **SCOPE
        )
        manual = self.folders.get_exact_manual_membership(
            folder_id=self.folder.folder_id, note_id=note_id
        )
        assert manual is not None
        assert self.org.detach_folder_link(
            folder_repository=self.folders,
            folder_id=self.folder.folder_id,
            note_id=note_id,
            expected_version=manual[0].version,
            **SCOPE,
        )

    def restore(self) -> None:
        """Take the note out of the Trash (the fixture puts it there)."""
        assert self.db.restore_note(self.note, self.version)
        self.version += 1

    def trash(self) -> None:
        assert self.db.soft_delete_note(self.note, self.version)
        self.version += 1

    def sync_service(self, server: FakeLocalFirstServer) -> LocalFirstSyncService:
        """The app's push path: the organization drain, the purge guard and
        the acknowledgement, around one push to ``server``."""
        return LocalFirstSyncService(
            server_service=server,
            state_repository=self.state,
            local_store=RecordingLocalStore(),
            dataset_keys={DATASET: generate_dataset_key()},
            notes_organization_sync_service=self.org,
            note_exists=self.db.note_ids_present,
        )

    def heads_of(self, note_id: str) -> set[tuple[str, str, int]]:
        """``(domain, operation, deleted)`` of every head naming the note."""
        return {
            (str(row["domain"]), str(row["operation"]), int(row["deleted"]))
            for row in self.db.get_connection().execute(
                "SELECT domain, operation, deleted, payload_json "
                "FROM notes_organization_heads"
            )
            if _subject(str(row["domain"]), json.loads(row["payload_json"]))
            == note_id
        }

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
                _subject(
                    row["envelope"]["domain"],
                    row["envelope"].get("payload")
                    or row["envelope"].get("payload_clear")
                    or {},
                ),
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


# -- PR #3423 re-review 3: an unlink still reaches the server (I-1, m-1) ------

#: The organization domains only: no dataset key, no note content.
ORG_DOMAINS = [
    "notes.keyword",
    "notes.keyword_link",
    "notes.folder",
    "notes.folder_link",
]


class _AcknowledgingServer(FakeLocalFirstServer):
    """Applies every envelope it is pushed, as the real server acknowledges
    one: its own cursor, the envelope's revision, ``applied``."""

    def __init__(self) -> None:
        super().__init__()
        self.cursor = 100

    async def push_v2_envelopes(self, *, dataset_id, device_id, envelopes, **kwargs):
        await super().push_v2_envelopes(
            dataset_id=dataset_id, device_id=device_id, envelopes=envelopes, **kwargs
        )
        accepted = []
        for envelope in envelopes:
            self.cursor += 1
            accepted.append(
                {
                    "client_envelope_id": envelope["client_envelope_id"],
                    "server_cursor": self.cursor,
                    "object_revision": envelope.get("object_revision"),
                    "apply_status": "applied",
                }
            )
        return {
            "dataset_id": dataset_id,
            "accepted": accepted,
            "next_cursor": str(self.cursor),
        }

    def pushes(self) -> int:
        return len([entry for entry in self.calls if entry[0] == "push"])

    def pushed(self, since: int = 0) -> set[tuple[str, str, str | None]]:
        """``(domain, operation, subject note)`` of every envelope pushed in
        the push calls after the first ``since``."""
        pushes = [entry for entry in self.calls if entry[0] == "push"]
        return {
            (
                envelope["domain"],
                envelope["operation"],
                _subject(envelope["domain"], envelope.get("payload") or {}),
            )
            for call in pushes[since:]
            for envelope in call[3]
        }


async def _synced_then_unlinked(tmp_path, monkeypatch) -> tuple[_Linked, _AcknowledgingServer]:
    """Both notes' links reach the server and are acknowledged; then the note
    comes back from the Trash, loses its keyword and its folder, and goes
    back in the Trash. Its two unlinks are pending, each after an
    acknowledged link."""
    linked = _Linked(tmp_path, monkeypatch)
    server = _AcknowledgingServer()
    await linked.sync_service(server).sync_once(**SCOPE, domains=ORG_DOMAINS)
    assert linked.heads_of(linked.note) == {
        ("notes.keyword_link", "upsert", 0),
        ("notes.folder_link", "upsert", 0),
    }
    linked.restore()
    linked.unlink(linked.note)
    linked.trash()
    pending = [
        row
        for row in linked.intents()
        if row["domain"] in LINKS
        and row["acknowledged_at"] is None
        and _subject(row["domain"], json.loads(row["payload_json"])) == linked.note
    ]
    assert sorted((row["domain"], row["operation"]) for row in pending) == [
        ("notes.folder_link", "tombstone"),
        ("notes.keyword_link", "tombstone"),
    ]
    return linked, server


async def test_an_unlink_of_a_link_the_server_holds_still_reaches_it_after_a_purge(
    tmp_path, monkeypatch
) -> None:
    """I-1: the note's keyword and folder were removed after its links were
    acknowledged, then the note was deleted for good before the next sync.
    Both unlinks are pushed, and the heads move to the tombstones."""
    linked, server = await _synced_then_unlinked(tmp_path, monkeypatch)
    before = server.pushes()

    assert await linked.purge() is True
    await linked.sync_service(server).sync_once(**SCOPE, domains=ORG_DOMAINS)

    pushed = server.pushed(since=before)
    unlinks = {
        ("notes.keyword_link", "tombstone", linked.note),
        ("notes.folder_link", "tombstone", linked.note),
    }
    assert unlinks <= pushed, f"the unlinks never left the device: pushed={pushed}"
    assert not [entry for entry in pushed if entry[1] == "upsert"], pushed
    assert linked.heads_of(linked.note) == {
        ("notes.keyword_link", "tombstone", 1),
        ("notes.folder_link", "tombstone", 1),
    }
    # Sent and acknowledged: nothing for the note is left to send.
    assert not [
        row
        for row in linked.intents()
        if row["acknowledged_at"] is None and linked.note in row["payload_json"]
    ]
    # The keeper's links were never touched.
    assert linked.heads_of(linked.keeper) == {
        ("notes.keyword_link", "upsert", 0),
        ("notes.folder_link", "upsert", 0),
    }


async def test_a_copied_unlink_is_acknowledged_and_moves_the_head_after_a_purge(
    tmp_path, monkeypatch
) -> None:
    """m-1: the drain copied the unlinks to the outbox, then the note was
    deleted for good, then they were pushed and acknowledged. Their intents
    are still there to acknowledge, so the heads become the tombstones."""
    linked, server = await _synced_then_unlinked(tmp_path, monkeypatch)
    before = server.pushes()
    linked.drain()
    # Copied, unsent (the outbox also keeps the links' dispatched upserts).
    assert {
        ("notes.keyword_link", "tombstone", linked.note),
        ("notes.folder_link", "tombstone", linked.note),
    } <= linked.links_of(linked.note)

    assert await linked.purge() is True
    await linked.sync_service(server).sync_once(**SCOPE, domains=ORG_DOMAINS)

    assert {
        ("notes.keyword_link", "tombstone", linked.note),
        ("notes.folder_link", "tombstone", linked.note),
    } <= server.pushed(since=before)
    assert linked.heads_of(linked.note) == {
        ("notes.keyword_link", "tombstone", 1),
        ("notes.folder_link", "tombstone", 1),
    }, "the acknowledged unlinks left the heads on the old links"


async def test_a_link_the_server_never_had_is_still_never_sent_after_a_purge(
    tmp_path, monkeypatch
) -> None:
    """Qodo 2's guarantee, against an over-wide I-1: links made since the
    last sync -- linked, unlinked and linked again, none acknowledged -- are
    all cancelled, tombstone included, and nothing for the note is copied or
    pushed. The keeper's links still go."""
    linked = _Linked(tmp_path, monkeypatch)
    linked.restore()
    linked.unlink(linked.note)
    linked.link(linked.note)
    linked.trash()
    operations = sorted(
        (row["domain"], row["operation"])
        for row in linked.intents()
        if row["domain"] in LINKS
        and _subject(row["domain"], json.loads(row["payload_json"])) == linked.note
    )
    assert operations == sorted(
        [(domain, "upsert") for domain in LINKS] * 2
        + [(domain, "tombstone") for domain in LINKS]
    )

    assert await linked.purge() is True
    server = _AcknowledgingServer()
    await linked.sync_service(server).sync_once(**SCOPE, domains=ORG_DOMAINS)

    assert not [entry for entry in server.pushed() if entry[2] == linked.note], (
        server.pushed()
    )
    assert linked.links_of(linked.note) == set()
    assert {
        ("notes.keyword_link", "upsert", linked.keeper),
        ("notes.folder_link", "upsert", linked.keeper),
    } <= server.pushed()
    assert not [row for row in linked.intents() if linked.note in row["payload_json"]]


async def test_an_unlink_learned_from_the_server_still_reaches_it_after_a_purge(
    tmp_path,
) -> None:
    """I-1 with no acknowledged predecessor: device B learned the note's
    keyword and folder links from the server (heads, no intents of its own),
    removed both, then deleted the note for good. Each unlink's base is the
    server's head; both are kept, pushed with that base (the transport
    checks it), and the server's heads become the tombstones."""
    device_a = _open_device(tmp_path / "a", "device-a")
    device_b = _open_device(tmp_path / "b", "device-b")
    transport = _FakeOrganizationTransport()
    scope = TWO_DEVICE_SCOPE
    note_id = TWO_DEVICE_NOTE_ID
    try:
        for device in (device_a, device_b):
            assert device.notes.add_note("Lesson", "private", note_id=note_id) == note_id
        folder = device_a.service.create_folder(
            folder_repository=device_a.folders, name="Projects", parent_id=None, **scope
        )
        device_a.service.sync_subject_keywords(
            subject_type="note", subject_id=note_id, keywords=("agent-lesson",), **scope
        )
        device_a.service.attach_folder_link(
            folder_repository=device_a.folders,
            folder_id=folder.folder_id,
            note_id=note_id,
            **scope,
        )
        transport.push(device_a)
        assert {result["status"] for result in transport.pull(device_b)} == {"applied"}
        folder_sync_id = str(
            device_a.notes.get_connection()
            .execute("SELECT sync_id FROM note_folders WHERE id = ?", (folder.folder_id,))
            .fetchone()[0]
        )
        connection_b = device_b.notes.get_connection()
        assert not connection_b.execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents "
            "WHERE domain IN ('notes.keyword_link', 'notes.folder_link')"
        ).fetchone()[0]

        device_b.service.sync_subject_keywords(
            subject_type="note", subject_id=note_id, keywords=(), **scope
        )
        folder_b = _resource_id(device_b, "note_folders", sync_id=folder_sync_id)
        manual = device_b.folders.get_exact_manual_membership(
            folder_id=folder_b, note_id=note_id
        )
        assert manual is not None
        assert device_b.service.detach_folder_link(
            folder_repository=device_b.folders,
            folder_id=folder_b,
            note_id=note_id,
            expected_version=manual[0].version,
            **scope,
        )
        unlinks = connection_b.execute(
            "SELECT domain, operation, predecessor_intent_id, base_server_cursor "
            "FROM notes_organization_sync_intents "
            "WHERE domain IN ('notes.keyword_link', 'notes.folder_link')"
        ).fetchall()
        assert sorted(
            (row[0], row[1], row[2], row[3] is not None) for row in unlinks
        ) == [
            ("notes.folder_link", "tombstone", None, True),
            ("notes.keyword_link", "tombstone", None, True),
        ]
        version = device_b.notes.get_note_by_id(note_id)["version"]
        assert device_b.notes.soft_delete_note(note_id, version)
        assert device_b.notes.hard_delete_note(note_id, expected_version=version + 1)
        device_b.state.discard_sync_v2_note_history(note_id)

        pushed = transport.push(device_b)

        assert sorted((item.domain, item.operation) for item in pushed) == [
            ("notes.folder_link", "tombstone"),
            ("notes.keyword_link", "tombstone"),
        ], "the unlinks never left device B"
        # The server's heads for both links are now the tombstones (the
        # transport accepted each against the head it was based on).
        assert {
            (item.domain, transport.heads[(item.domain, str(item.object_id))].operation)
            for item in pushed
        } == {
            ("notes.folder_link", "tombstone"),
            ("notes.keyword_link", "tombstone"),
        }
        # Acknowledged on B: nothing for the note is left to send.
        assert not connection_b.execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents "
            "WHERE acknowledged_at IS NULL"
        ).fetchone()[0]
    finally:
        device_a.close()
        device_b.close()


# -- PR #3423 re-review 3 m-3: the guard and its heal read one payload -------


async def test_a_link_held_from_its_clear_payload_is_healed_too(
    tmp_path, monkeypatch
) -> None:
    """m-3: the push guard and its heal read one payload. An outbox row
    whose link sits only in ``payload_clear`` (a row ``SyncV2Envelope`` did
    not write -- it fills ``payload`` from ``payload_clear`` itself) is held
    back AND dropped, not held back on every push for ever."""
    linked = _Linked(tmp_path, monkeypatch)
    linked.drain()
    with linked.state.transaction() as connection:
        changed = connection.execute(
            "UPDATE sync_v2_local_outbox SET envelope = "
            "json_set(envelope, '$.payload', json('{}')) "
            "WHERE domain IN ('notes.folder_link', 'notes.keyword_link') "
            "AND instr(envelope, ?) > 0",
            (linked.note,),
        ).rowcount
    assert changed == 2
    # The purge, without its post-commit outbox discard (it never ran).
    assert linked.db.hard_delete_note(linked.note, expected_version=linked.version)
    assert linked.links_of(linked.note) == {
        ("notes.keyword_link", "upsert", linked.note),
        ("notes.folder_link", "upsert", linked.note),
    }
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
        _subject(envelope["domain"], envelope.get("payload") or {})
        for envelope in server.calls[0][3]
    }
    assert linked.note not in pushed
    assert linked.links_of(linked.note) == set(), (
        "held back but never healed: it is held again on every push"
    )


_ABSENT = object()


@pytest.mark.parametrize(
    "payload",
    [_ABSENT, None, {}, {"note_id": "from-payload"}, "not an object"],
    ids=["absent", "null", "empty", "object", "text"],
)
@pytest.mark.parametrize(
    "payload_clear",
    [_ABSENT, {}, {"note_id": "from-clear"}],
    ids=["clear-absent", "clear-empty", "clear-object"],
)
async def test_the_guard_and_its_heal_read_the_same_payload(payload, payload_clear) -> None:
    """m-3: ``LocalFirstSyncService._guarded_payload`` (the push guard) and
    ``SyncStateRepository._GUARDED_PAYLOAD_PATH_SQL`` (its heal) pick the
    same payload for every envelope shape."""
    import sqlite3

    envelope: dict = {}
    if payload is not _ABSENT:
        envelope["payload"] = payload
    if payload_clear is not _ABSENT:
        envelope["payload_clear"] = payload_clear
    guarded = LocalFirstSyncService._guarded_payload(
        envelope.get("payload"), envelope.get("payload_clear")
    )
    text = json.dumps(envelope)
    connection = sqlite3.connect(":memory:")
    try:
        (path,) = connection.execute(
            f"SELECT {SyncStateRepository._GUARDED_PAYLOAD_PATH_SQL} "
            "FROM (SELECT ? AS envelope)",
            (text,),
        ).fetchone()
        (healed,) = connection.execute(
            "SELECT json_extract(?, ?)", (text, path)
        ).fetchone()
    finally:
        connection.close()
    assert (json.loads(healed) if healed is not None else None) == (
        dict(guarded) if guarded is not None else None
    ), (envelope, path)
