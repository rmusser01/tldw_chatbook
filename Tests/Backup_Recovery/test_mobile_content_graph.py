"""Native captured C1 transformations retain body meaning without owner effects."""

from contextlib import closing, contextmanager
from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


@contextmanager
def content_reads(tmp_path, monkeypatch, seed, *, native_seed=None, file_notes=False):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    original_seed = mobile_consolidation_fixture._seed

    def populate(db, legacy):
        with db.transaction() as cursor:
            if legacy:
                # Positive ID collisions independently satisfy native BINARY names.
                cursor.execute(
                    "UPDATE character_cards SET name='Legacy assistant' WHERE id=1"
                )
            seed(cursor.connection, "legacy" if legacy else "canonical")
        if legacy and db.get_message_by_id("graph-user") is not None:
            db.set_message_attachments(
                "graph-user",
                [
                    {
                        "position": 1,
                        "data": b"private attachment\x00\xff",
                        "mime_type": "application/octet-stream",
                        "display_name": "exact name",
                    }
                ],
            )

        if native_seed is not None:
            native_seed(db, "legacy" if legacy else "canonical")

    def seeded(stores, **kwargs):
        result = original_seed(stores, **kwargs, chat_seed=populate)
        if file_notes and not kwargs["legacy"]:
            from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
            from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
            from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica

            # Native canonical inventory requires this operational owner when
            # historical Note locators exist. Construct it before preview.
            config = read_profile_config(tmp_path / "profile" / "config.toml")
            with closing(FileNotesReplica(user_data_dir(config) / "file_notes.sqlite")):
                pass
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, config, _stores, _, _, _ = values
        with session._discovery_reads(private_sqlite=True):
            inventory = discover((config,))
            preview = _preview_selected_mobile(config, config)
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=archives,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with journal._locked(exclusive=False) as fd:
                events = tuple(journal._records(fd))
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                yield reads
            assert _fingerprints(tmp_path / "profile") == before
            with journal._locked(exclusive=False) as fd:
                assert tuple(journal._records(fd)) == events


def _seed_notes(connection, purpose, *, missing=False):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    bodies = {
        "note-a": "Keep note-b in prose; [visible note-b](note://note-b) and [again](note://note-b); [self](note://note-a). Ω",
        "note-b": "Target body",
        "note-deleted": "[deleted source](note://note-b)",
    }
    if missing:
        bodies["note-a"] = "[unknown](note://not-captured)"
    for identity, content in bodies.items():
        connection.execute(
            "INSERT INTO notes(id,title,content,version,deleted) VALUES (?,?,?,7,?)",
            (
                identity,
                identity,
                content if purpose == "legacy" else "canonical body",
                int(identity == "note-deleted"),
            ),
        )
        if purpose == "legacy":
            CharactersRAGDB.replace_note_links(connection.cursor(), identity, content)


def test_note_tokens_survive_native_write_and_backlink_read(tmp_path, monkeypatch):
    from tldw_chatbook.DB.ChaChaNotes_DB import (
        CharactersRAGDB,
        extract_note_link_targets,
    )

    with content_reads(tmp_path, monkeypatch, _seed_notes) as reads:
        build = getattr(reads, "content_graph_plan", None)
        assert callable(build), (
            "native captured reader must provide the private C1 graph plan"
        )
        plan = build()
        assert not plan.blockers
        mapped = {
            identity: plan.identities.mapped("note", identity)
            for identity in ("note-a", "note-b", "note-deleted")
        }
        rows = {
            dict(row.values)["id"]: dict(row.values)
            for row in plan.rows
            if row.table == "notes"
        }
        body = rows[mapped["note-a"]]["content"]
        assert (
            body
            == f"Keep note-b in prose; [visible note-b](note://{mapped['note-b']}) and [again](note://{mapped['note-b']}); [self](note://{mapped['note-a']}). Ω"
        )
        assert extract_note_link_targets(body, source_note_id=mapped["note-a"]) == [
            mapped["note-b"]
        ]
        assert rows[mapped["note-deleted"]]["deleted"] == 1
        assert rows[mapped["note-a"]]["version"] == 7
        with closing(CharactersRAGDB(":memory:", "content-proof")) as db:
            for row in rows.values():
                db.add_note(row["title"], row["content"], note_id=row["id"])
                with db.transaction() as cursor:
                    cursor.execute(
                        "UPDATE notes SET version=?,deleted=? WHERE id=?",
                        (row["version"], row["deleted"], row["id"]),
                    )
            assert db.get_note_by_id(mapped["note-a"])["content"] == body
            assert [row["id"] for row in db.get_notes_linking_to(mapped["note-b"])] == [
                mapped["note-a"]
            ]
            # A subsequent ordinary native save must regenerate the same mapped edge.
            with db.transaction() as cursor:
                db.replace_note_links(cursor, mapped["note-a"], body)
            assert [row["id"] for row in db.get_notes_linking_to(mapped["note-b"])] == [
                mapped["note-a"]
            ]
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()
        with pytest.raises(FrozenInstanceError):
            plan.rows = ()


def test_unproved_note_target_blocks_exact_source_without_rewriting_body(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_notes(connection, purpose, missing=True)

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_graph_plan()
        assert any(
            b.table == "notes"
            and b.original == ("note-a",)
            and b.dependency_domain == "note"
            and b.dependency_id == "not-captured"
            for b in plan.blockers
        )
        assert not any(
            row.table == "notes" and row.original == ("note-a",) for row in plan.rows
        )
        assert any(
            row.table == "notes" and row.original == ("note-b",) for row in plan.rows
        )
        original = next(
            dict(row.values)
            for row in plan.source_rows
            if row.table == "notes" and row.original == ("note-a",)
        )
        assert original["content"] == "[unknown](note://not-captured)"


def _seed_core(connection, purpose):
    _seed_notes(connection, purpose)
    label = purpose + " name"
    connection.execute("INSERT INTO keywords(id,keyword) VALUES (11,?)", (label,))
    for table in ("keyword_collections", "chat_dictionaries", "world_books"):
        connection.execute("INSERT INTO " + table + "(id,name) VALUES (11,?)", (label,))
    connection.execute(
        "INSERT INTO character_cards(id,name,description,extensions) VALUES (11,?,'exact character body','{\"literal\":\"note-a\"}')",
        (label,),
    )
    connection.execute(
        "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (11,11,'[\"forest\"]','exact lore body')"
    )
    connection.execute(
        "INSERT INTO note_folders(id,name,normalized_name,path,normalized_path,created_at,modified_at) VALUES ('folder',?,?,?,?,'2026-10-25','2026-10-25')",
        (label, label, "/" + label, "/" + label),
    )
    connection.execute(
        "INSERT INTO note_folder_memberships(id,folder_id,note_id,ownership,created_at,modified_at) VALUES ('placement','folder','note-a','manual','2026-10-25','2026-10-25')"
    )
    connection.execute(
        "INSERT INTO note_keywords(note_id,keyword_id) VALUES ('note-a',11)"
    )
    connection.execute(
        "INSERT INTO collection_keywords(collection_id,keyword_id) VALUES (11,11)"
    )
    connection.execute(
        "INSERT INTO conversation_keywords(conversation_id,keyword_id) VALUES ('conversation',11)"
    )
    connection.execute(
        "INSERT INTO conversation_dictionaries(conversation_id,dictionary_id) VALUES ('conversation',11)"
    )
    connection.execute(
        "UPDATE conversations SET metadata='{\"active_dictionaries\":[11,\"11\"]}' WHERE id='conversation'"
    )
    connection.execute(
        "INSERT INTO conversation_world_books(conversation_id,world_book_id) VALUES ('conversation',11)"
    )
    for identity, parent, role, variant in (
        ("graph-user", None, "user", None),
        ("graph-answer", "graph-user", "assistant", None),
        ("graph-variant", "graph-user", "assistant", "graph-answer"),
    ):
        connection.execute(
            "INSERT INTO messages(id,conversation_id,parent_message_id,sender,role,content,version,variant_of,variant_number,is_selected_variant,total_variants,client_id) VALUES (?,'conversation',?,?,?,'exact message body',3,?, ?, ?,2,'fixture')",
            (
                identity,
                parent,
                role,
                role,
                variant,
                int(variant is not None),
                int(variant is not None),
            ),
        )
    if purpose == "legacy":
        connection.execute(
            "INSERT INTO character_expression_images(id,character_id,state_id,image,mime) VALUES (11,11,'happy',?,'image/png')",
            (b"expression bytes",),
        )


def _install_rows(db, rows):
    """Test-only private sink; production plan has no SQL or connection argument."""
    with db.transaction() as cursor:
        for row in rows:
            if (
                row.table == "message_attachments"
                or "C2_native_dictionary_metadata_derivation" in row.required_proofs
            ):
                continue
            values = dict(row.values)
            columns = ",".join('"' + key + '"' for key in values)
            cursor.execute(
                'INSERT INTO "'
                + row.table
                + '" ('
                + columns
                + ") VALUES ("
                + ",".join("?" for _ in values)
                + ")",
                tuple(values.values()),
            )

    attachments = {}
    for row in rows:
        if row.table == "message_attachments":
            value = dict(row.values)
            identity = value.pop("message_id")
            attachments.setdefault(identity, []).append(value)
    for identity, values in attachments.items():
        db.set_message_attachments(identity, values)


def test_core_graph_maps_native_entities_associations_and_partial_branches(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository

    with content_reads(tmp_path, monkeypatch, _seed_core) as reads:
        plan = reads.content_graph_plan()
        assert not plan.blockers
        assert any(row.table == "character_cards" for row in plan.rows), (
            "C1 must transform actual dependent entities"
        )
        ids = plan.identities
        conversation = ids.mapped("conversation", "conversation")
        partial = next(
            row
            for row in plan.rows
            if row.table == "messages" and row.original == ("graph-variant",)
        )
        assert set(partial.pending_columns) == {
            "usage_json",
            "metadata_json",
            "provider_continuation_json",
            "assistant_generation_state",
            "thinking_blocks_json",
        }
        assert dict(partial.values)["variant_of"] == ids.mapped(
            "message", "graph-answer"
        )
        assert dict(partial.values)["parent_message_id"] == ids.mapped(
            "message", "graph-user"
        )
        assert all(row.table != "conversations" for row in plan.rows)
        with closing(CharactersRAGDB(":memory:", "graph-proof")) as db:
            db.add_conversation(
                {"id": conversation, "title": "private C1 consumer fixture"}
            )
            _install_rows(db, plan.rows)
            character = db.get_character_card_by_id(ids.mapped("character", 11))
            assert character["description"] == "exact character body"
            assert character["extensions"] == {"literal": "note-a"}
            assert (
                db.get_keywords_for_note(ids.mapped("note", "note-a"))[0]["keyword"]
                == "legacy name"
            )
            folder = LocalNoteFolderRepository(db).get_folder(
                ids.mapped("note_folder", "folder")
            )
            assert folder.name == "legacy name"
            book = WorldBookManager(db).get_world_books_for_conversation(conversation)[
                0
            ]
            assert book["id"] == ids.mapped("world_book", 11)
            assert (
                WorldBookManager(db).get_world_book_entries(book["id"])[0]["content"]
                == "exact lore body"
            )
            messages = db.get_messages_for_conversation(conversation)
            mapped_variant = next(
                row
                for row in messages
                if row["id"] == ids.mapped("message", "graph-variant")
            )
            assert mapped_variant["content"] == "exact message body"
            assert mapped_variant["version"] == 3
            with db.transaction() as cursor:
                attachment = cursor.execute(
                    "SELECT data,display_name FROM message_attachments WHERE message_id=?",
                    (ids.mapped("message", "graph-user"),),
                ).fetchone()
            assert tuple(attachment) == (b"private attachment\x00\xff", "exact name")
            from tldw_chatbook.Chat.console_context_repository import (
                _persisted_lineage_state,
            )

            with db.transaction() as cursor:
                branch_row = cursor.execute(
                    "SELECT * FROM messages WHERE id=?",
                    (ids.mapped("message", "graph-answer"),),
                ).fetchone()
                lineage = _persisted_lineage_state(cursor, branch_row)
                assert lineage is not None
                assert lineage.fence.selected_variant_id == ids.mapped(
                    "message", "graph-variant"
                )
                assert lineage.content == "exact message body"
        derived = next(
            row
            for row in plan.rows
            if row.table == "conversation_dictionary_attachments"
        )
        assert derived.required_proofs == ("C2_native_dictionary_metadata_derivation",)


def test_native_uniqueness_conflict_is_exact_and_does_not_rename(tmp_path, monkeypatch):
    def seed(connection, purpose):
        _seed_core(connection, purpose)
        connection.execute(
            "UPDATE character_cards SET name='Default Assistant' WHERE id=1"
        )
        connection.execute(
            "UPDATE notes SET file_path_on_disk='/historical/shared.md',deleted=? WHERE id='note-a'",
            (int(purpose == "canonical"),),
        )
        connection.execute(
            "UPDATE keywords SET keyword=? WHERE id=11",
            ("Tag" if purpose == "legacy" else "tAG",),
        )

    with content_reads(tmp_path, monkeypatch, seed, file_notes=True) as reads:
        plan = reads.content_graph_plan()
        assert any(
            b.table == "keywords"
            and b.original == (11,)
            and b.reason == "native_unique_conflict"
            for b in plan.blockers
        )
        assert any(
            b.table == "character_cards"
            and b.original == (1,)
            and b.reason == "native_unique_conflict"
            for b in plan.blockers
        )
        assert (
            next(
                dict(r.values)["name"]
                for r in plan.source_rows
                if r.table == "character_cards" and r.original == (1,)
            )
            == "Default Assistant"
        )
        assert any(
            b.table == "notes"
            and b.original == ("note-a",)
            and b.reason == "native_unique_conflict"
            for b in plan.blockers
        )
        assert (
            next(
                dict(r.values)["file_path_on_disk"]
                for r in plan.source_rows
                if r.table == "notes" and r.original == ("note-a",)
            )
            == "/historical/shared.md"
        )
        assert any(
            row.table == "notes"
            and row.original == ("note-deleted",)
            and dict(row.values)["file_path_on_disk"] is None
            for row in plan.rows
        )
        assert not any(row.table == "keywords" for row in plan.rows)
        assert any(row.table == "world_books" for row in plan.rows)
        assert not any(
            row.table
            in {"note_keywords", "collection_keywords", "conversation_keywords"}
            for row in plan.rows
        )


def test_legacy_note_locators_remain_historical_fields_without_file_io(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_notes(connection, purpose)
        if purpose == "legacy":
            connection.execute(
                "UPDATE notes SET file_path_on_disk='/missing/legacy/note.md',relative_file_path_on_disk='note.md',sync_root_folder='/missing/legacy',last_synced_disk_file_hash='historical-hash',is_externally_synced=1 WHERE id='note-a'"
            )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_graph_plan()
        assert not any(
            b.table == "notes" and b.original == ("note-a",) for b in plan.blockers
        )
        row = next(
            dict(row.values)
            for row in plan.rows
            if row.table == "notes" and row.original == ("note-a",)
        )
        assert row["file_path_on_disk"] == "/missing/legacy/note.md"
        assert row["last_synced_disk_file_hash"] == "historical-hash"
        assert row["is_externally_synced"] == 1
        assert "C2" in plan.pending_handlers
