"""Absolute census of the named indexes on the ChaChaNotes schema (task-19045).

Why this exists: the task-16840 close-out review constructed and disclosed a
MUT-INDEX escape — delete a ``CREATE INDEX`` from a ChaChaNotes migration step
and nothing in Tests/ turns red. The historical-bootstrap parity sweep
(``test_historical_bootstrap.py``) cannot catch it BY DESIGN: both sides of
its oracle run the same migration chain, so a deterministic chain mutation is
the identity on the comparison (16840's MUT-A honesty note). At the census
that filed task-19045, 84 of the 94 named indexes were unreferenced by any
test: a dropped or renamed index ships as a silent performance regression,
and a lost UNIQUE index as a silent integrity regression.

This module is the guard, mirroring the ``VALID_TABLES`` precedent
(``Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema``,
TASK-864): an explicit hand-maintained literal, asserted in BOTH directions
against a live fully-migrated DB. ``EXPECTED_CHACHANOTES_INDEXES`` is
deliberately NOT derived from the schema code it checks — a census that
re-derives its expectation from the migration chain would be the identity on
exactly the defect class it exists to catch. The literal IS the pin; updating
it is a deliberate schema-review act, not test appeasement (each failure
message says which side to fix).

What is pinned per index: name, table, the UNIQUE flag (from
``PRAGMA index_list``), and the key-column tuple in order (from
``PRAGMA index_info``). NOT pinned: partial-index ``WHERE`` clauses — their
SQL text is formatting-sensitive; the flag + column tuple are the durable
core. The census runs against BOTH a fresh bootstrap and a chain-migrated DB
(bootstrap genuinely at v4, reopen, full-chain replay) so a stop/resume
divergence the parity sweep normalizes away is also caught.

UNIQUE-ness decisions (AC #2): the UNIQUE flag is pinned for ALL indexes via
``IndexPin.unique`` — losing UNIQUE silently legalizes duplicate rows that
application code assumes cannot exist, so every one of the twelve is treated
as integrity-bearing:

* ``idx_message_trajectory_conv_seq`` — (conversation_id, seq) is the
  trajectory ledger's ordering identity; duplicate seq rows would corrupt
  replay order. Also pinned by a dedicated test below so a mechanical
  literal update cannot silently ride along with a schema downgrade.
* ``idx_actor_pack_persona_intents_state`` — deterministic startup recovery
  scans over unresolved Actor Pack Persona intents.
* ``idx_messages_conversation_id_id`` — message identity within a
  conversation; backs keyset pagination over (conversation_id, id).
* ``idx_notes_file_path_unique`` — at most one note per on-disk file path
  (the notes sync engine's file<->note mapping invariant; partial).
* ``idx_persona_visual_assets_version_key`` — (pack_version_id, asset_key)
  is an immutable visual-graph version's asset identity; duplicates would
  let two different assets answer for the same manifest key.
* ``idx_persona_visual_bindings_persona_active`` — at most one ACTIVE
  persona-visual binding per persona (partial, WHERE status = 'active').
* ``idx_visual_identity_bindings_actor_active`` — at most one ACTIVE visual
  identity binding per actor (partial, WHERE status = 'active').
* ``rag_citation_traces_import_identity_uq`` /
  ``rag_citation_traces_server_identity_uq`` /
  ``rag_evidence_snapshots_content_dedupe_uq`` /
  ``rag_message_trace_owners_active_message_uq`` — RAG trace/evidence
  identity and dedupe constraints (all partial).
* ``uq_note_folder_memberships_active_owner`` /
  ``uq_note_folders_active_normalized_path`` — folder-tree invariants: one
  active membership per (folder, note, ownership, owner); one active folder
  per normalized path (both partial, WHERE deleted = 0).
"""

from __future__ import annotations

import sqlite3
from typing import NamedTuple

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.historical_bootstrap import (
    MINIMUM_BOOTSTRAP_VERSION,
    chachanotes_db_at_version,
)

_THIS_FILE = "Tests/ChaChaNotesDB/test_index_census.py"


class IndexPin(NamedTuple):
    """The pinned shape of one named index."""

    table: str
    unique: bool
    columns: tuple[str, ...]


#: The full expected set of named (non-autoindex) indexes on a fully-migrated
#: ChaChaNotes DB. HAND-MAINTAINED ON PURPOSE (see module docstring): update
#: it only as part of a deliberate schema change, in the same commit as the
#: migration that adds, drops, renames, or reshapes an index. Sorted by name.
EXPECTED_CHACHANOTES_INDEXES: dict[str, IndexPin] = {
    "idx_actor_pack_persona_intents_state": IndexPin(
        "actor_pack_persona_intents", False, ("state", "created_at", "intent_id")
    ),
    "idx_char_expr_images_char": IndexPin(
        "character_expression_images", False, ("character_id",)
    ),
    "idx_chat_dictionaries_deleted": IndexPin("chat_dictionaries", False, ("deleted",)),
    "idx_chat_dictionaries_enabled": IndexPin("chat_dictionaries", False, ("enabled",)),
    "idx_chat_dictionaries_name": IndexPin("chat_dictionaries", False, ("name",)),
    "idx_collkw_kw": IndexPin("collection_keywords", False, ("keyword_id",)),
    "idx_console_aux_attempts_conversation_started": IndexPin(
        "console_auxiliary_attempts", False, ("conversation_id", "started_at")
    ),
    "idx_console_memories_boundary": IndexPin(
        "console_conversation_memories",
        False,
        ("conversation_id", "boundary_message_id"),
    ),
    "idx_console_memories_conversation_active": IndexPin(
        "console_conversation_memories",
        False,
        ("conversation_id", "active", "created_at"),
    ),
    "idx_conv_char": IndexPin("conversations", False, ("character_id",)),
    "idx_conversation_dictionaries_conv": IndexPin(
        "conversation_dictionaries", False, ("conversation_id",)
    ),
    "idx_conversation_dictionaries_dict": IndexPin(
        "conversation_dictionaries", False, ("dictionary_id",)
    ),
    "idx_conversation_dictionary_attachments_dictionary": IndexPin(
        "conversation_dictionary_attachments", False, ("dictionary_id",)
    ),
    "idx_conversation_local_marks_type": IndexPin(
        "conversation_local_marks",
        False,
        ("mark_type", "updated_at", "conversation_id"),
    ),
    "idx_conversation_world_books_book": IndexPin(
        "conversation_world_books", False, ("world_book_id",)
    ),
    "idx_conversation_world_books_conv": IndexPin(
        "conversation_world_books", False, ("conversation_id",)
    ),
    "idx_conversations_assistant_identity": IndexPin(
        "conversations", False, ("assistant_kind", "assistant_id")
    ),
    "idx_conversations_discovery_entity": IndexPin(
        "conversations", False, ("discovery_entity_id",)
    ),
    "idx_conversations_discovery_owner": IndexPin(
        "conversations", False, ("discovery_owner",)
    ),
    "idx_conversations_parent": IndexPin(
        "conversations", False, ("parent_conversation_id",)
    ),
    "idx_conversations_root": IndexPin("conversations", False, ("root_id",)),
    "idx_conversations_runtime_backend": IndexPin(
        "conversations", False, ("runtime_backend",)
    ),
    "idx_conversations_scope_type": IndexPin("conversations", False, ("scope_type",)),
    "idx_conversations_state": IndexPin("conversations", False, ("state",)),
    "idx_conversations_topic_label": IndexPin("conversations", False, ("topic_label",)),
    "idx_conversations_workspace_id": IndexPin(
        "conversations", False, ("workspace_id",)
    ),
    "idx_convkw_kw": IndexPin("conversation_keywords", False, ("keyword_id",)),
    "idx_flashcard_templates_name": IndexPin("flashcard_templates", False, ("name",)),
    "idx_flashcards_deck_id": IndexPin("flashcards", False, ("deck_id",)),
    "idx_flashcards_next_review": IndexPin("flashcards", False, ("next_review",)),
    "idx_kept_briefings_kept_at": IndexPin("kept_briefings", False, ("kept_at", "id")),
    "idx_kept_scripts_briefing": IndexPin("kept_scripts", False, ("kept_briefing_id",)),
    "idx_message_attachments_message": IndexPin(
        "message_attachments", False, ("message_id",)
    ),
    "idx_message_exchanges_message": IndexPin(
        "message_exchanges", False, ("message_id",)
    ),
    "idx_message_trajectory_conv_seq": IndexPin(
        "message_trajectory_metadata", True, ("conversation_id", "seq")
    ),
    "idx_message_trajectory_msg": IndexPin(
        "message_trajectory_metadata", False, ("message_id",)
    ),
    "idx_message_trajectory_turn": IndexPin(
        "message_trajectory_metadata", False, ("conversation_id", "turn_id", "seq")
    ),
    "idx_messages_conversation_id_id": IndexPin(
        "messages", True, ("conversation_id", "id")
    ),
    "idx_messages_feedback": IndexPin("messages", False, ("feedback",)),
    "idx_messages_role": IndexPin("messages", False, ("role",)),
    "idx_messages_selected_variant": IndexPin(
        "messages", False, ("variant_of", "is_selected_variant")
    ),
    "idx_messages_variant_of": IndexPin("messages", False, ("variant_of",)),
    "idx_messages_variants_by_parent": IndexPin(
        "messages", False, ("parent_message_id", "variant_number")
    ),
    "idx_mindmap_nodes_mindmap_id": IndexPin("mindmap_nodes", False, ("mindmap_id",)),
    "idx_msg_gen_meta_message": IndexPin(
        "message_generation_metadata", False, ("message_id",)
    ),
    "idx_msgs_conv_ts": IndexPin("messages", False, ("conversation_id", "timestamp")),
    "idx_msgs_conversation": IndexPin("messages", False, ("conversation_id",)),
    "idx_msgs_parent": IndexPin("messages", False, ("parent_message_id",)),
    "idx_msgs_ranking": IndexPin("messages", False, ("ranking",)),
    "idx_msgs_timestamp": IndexPin("messages", False, ("timestamp",)),
    "idx_note_folder_memberships_active_folder": IndexPin(
        "note_folder_memberships", False, ("folder_id", "note_id")
    ),
    "idx_note_folder_memberships_active_note": IndexPin(
        "note_folder_memberships", False, ("note_id", "folder_id")
    ),
    "idx_note_folder_memberships_managed_owner": IndexPin(
        "note_folder_memberships",
        False,
        ("owner_id", "deleted", "folder_id", "note_id", "owner_active"),
    ),
    "idx_note_folder_memberships_restore_review": IndexPin(
        "note_folder_memberships", False, ("owner_active", "owner_id")
    ),
    "idx_note_folders_active_parent": IndexPin(
        "note_folders", False, ("parent_id", "normalized_name")
    ),
    "idx_notekw_kw": IndexPin("note_keywords", False, ("keyword_id",)),
    "idx_notes_file_path": IndexPin("notes", False, ("file_path_on_disk",)),
    "idx_notes_file_path_unique": IndexPin("notes", True, ("file_path_on_disk",)),
    "idx_notes_is_synced": IndexPin("notes", False, ("is_externally_synced",)),
    "idx_notes_last_modified": IndexPin("notes", False, ("last_modified",)),
    "idx_notes_sync_excluded": IndexPin("notes", False, ("sync_excluded",)),
    "idx_notes_sync_root": IndexPin("notes", False, ("sync_root_folder",)),
    "idx_persona_visual_assets_version_key": IndexPin(
        "persona_visual_assets", True, ("pack_version_id", "asset_key")
    ),
    "idx_persona_visual_bindings_persona_active": IndexPin(
        "persona_visual_bindings", True, ("persona_id",)
    ),
    "idx_quiz_attempts_quiz_id": IndexPin("quiz_attempts", False, ("quiz_id",)),
    "idx_quiz_attempts_started_at": IndexPin("quiz_attempts", False, ("started_at",)),
    "idx_quiz_questions_order": IndexPin(
        "quiz_questions", False, ("quiz_id", "order_index", "id")
    ),
    "idx_quiz_questions_quiz_id": IndexPin("quiz_questions", False, ("quiz_id",)),
    "idx_quizzes_last_modified": IndexPin("quizzes", False, ("last_modified",)),
    "idx_quizzes_name": IndexPin("quizzes", False, ("name",)),
    "idx_quizzes_workspace_id": IndexPin("quizzes", False, ("workspace_id",)),
    "idx_review_history_flashcard_id": IndexPin(
        "review_history", False, ("flashcard_id",)
    ),
    "idx_study_sessions_entity": IndexPin(
        "study_sessions", False, ("entity_type", "entity_id")
    ),
    "idx_sync_conflicts_note": IndexPin("sync_conflicts", False, ("note_id",)),
    "idx_sync_conflicts_resolution": IndexPin("sync_conflicts", False, ("resolution",)),
    "idx_sync_conflicts_session": IndexPin("sync_conflicts", False, ("session_id",)),
    "idx_sync_log_entity": IndexPin("sync_log", False, ("entity", "entity_id")),
    "idx_sync_log_ts": IndexPin("sync_log", False, ("timestamp",)),
    "idx_sync_sessions_started": IndexPin("sync_sessions", False, ("started_at",)),
    "idx_sync_sessions_status": IndexPin("sync_sessions", False, ("status",)),
    "idx_topics_parent_id": IndexPin("topics", False, ("parent_id",)),
    "idx_topics_path_id": IndexPin("topics", False, ("path_id",)),
    "idx_transcript_annotations_conv_row": IndexPin(
        "transcript_annotations", False, ("conversation_id", "row_key")
    ),
    "idx_visual_identity_assets_pack_expression": IndexPin(
        "visual_identity_assets",
        False,
        ("pack_id", "pack_version_id", "expression_key", "deleted"),
    ),
    "idx_visual_identity_bindings_actor_active": IndexPin(
        "visual_identity_bindings", True, ("owner_user_id", "actor_kind", "actor_id")
    ),
    "idx_visual_identity_packs_owner_status": IndexPin(
        "visual_identity_packs", False, ("owner_user_id", "status")
    ),
    "idx_world_book_entries_book": IndexPin(
        "world_book_entries", False, ("world_book_id",)
    ),
    "idx_world_book_entries_enabled": IndexPin(
        "world_book_entries", False, ("enabled",)
    ),
    "idx_world_book_entries_position": IndexPin(
        "world_book_entries", False, ("position",)
    ),
    "idx_world_books_deleted": IndexPin("world_books", False, ("deleted",)),
    "idx_world_books_enabled": IndexPin("world_books", False, ("enabled",)),
    "idx_world_books_name": IndexPin("world_books", False, ("name",)),
    "rag_citation_traces_import_identity_uq": IndexPin(
        "rag_citation_traces",
        True,
        ("profile_id", "import_package_fingerprint", "external_trace_id"),
    ),
    "rag_citation_traces_server_identity_uq": IndexPin(
        "rag_citation_traces",
        True,
        (
            "connection_authority_id",
            "origin_scope_id",
            "server_trace_id",
            "wire_schema_version",
        ),
    ),
    "rag_evidence_snapshots_content_dedupe_uq": IndexPin(
        "rag_evidence_snapshots",
        True,
        (
            "governance_scope_id",
            "authority_id",
            "confidentiality_policy_id",
            "revocation_scope_id",
            "content_hash",
        ),
    ),
    "rag_message_trace_owners_active_message_uq": IndexPin(
        "rag_message_trace_owners",
        True,
        ("profile_id", "message_id", "message_revision"),
    ),
    "uq_note_folder_memberships_active_owner": IndexPin(
        "note_folder_memberships",
        True,
        ("folder_id", "note_id", "ownership", "owner_id"),
    ),
    "uq_note_folders_active_normalized_path": IndexPin(
        "note_folders", True, ("normalized_path",)
    ),
}


def _census(conn: sqlite3.Connection) -> dict[str, IndexPin]:
    """Read every named (non-autoindex) index from a live connection.

    Returns:
        Mapping of index name to its live ``IndexPin`` (table, unique flag
        from ``PRAGMA index_list``, key-column tuple from
        ``PRAGMA index_info``).
    """
    rows = conn.execute(
        "SELECT name, tbl_name FROM sqlite_master "
        "WHERE type = 'index' AND name NOT LIKE 'sqlite_autoindex_%'"
    ).fetchall()
    unique_by_name: dict[str, bool] = {}
    for table in {row["tbl_name"] for row in rows}:
        for index_row in conn.execute(f'PRAGMA index_list("{table}")'):
            unique_by_name[index_row["name"]] = bool(index_row["unique"])
    census: dict[str, IndexPin] = {}
    for row in rows:
        name = row["name"]
        columns = tuple(
            info_row["name"]
            for info_row in conn.execute(f'PRAGMA index_info("{name}")')
        )
        census[name] = IndexPin(row["tbl_name"], unique_by_name[name], columns)
    return census


@pytest.fixture(scope="module", params=["fresh_bootstrap", "chain_migrated_from_v4"])
def live_index_census(request, tmp_path_factory) -> dict[str, IndexPin]:
    """Census of a live fully-migrated DB, built two independent ways.

    ``fresh_bootstrap`` builds the DB straight through in one process;
    ``chain_migrated_from_v4`` bootstraps a genuinely v4-shaped DB (patched
    ``_CURRENT_SCHEMA_VERSION``, real chain), closes it, and reopens it
    unpatched so the production chain replays v4 -> current as a real upgrade
    would. Asserting the same absolute census on both catches a stop/resume
    divergence that the parity sweep in ``test_historical_bootstrap.py``
    normalizes away (both of ITS sides run the same chain).
    """
    if request.param == "fresh_bootstrap":
        db = CharactersRAGDB(":memory:", client_id="index-census-fresh")
        try:
            return _census(db.get_connection())
        finally:
            db.close_connection()
    db_path = tmp_path_factory.mktemp("index_census") / "chain_migrated.sqlite"
    with chachanotes_db_at_version(db_path, MINIMUM_BOOTSTRAP_VERSION):
        pass  # bootstrap a genuinely-v4 DB, then close it
    db = CharactersRAGDB(str(db_path), client_id="index-census-chain")
    try:
        return _census(db.get_connection())
    finally:
        db.close_connection()


class TestChachanotesIndexCensusMatchesLiveSchema:
    """task-19045: the absolute index census, both directions.

    Mirrors ``TestChachanotesValidTablesMatchesLiveSchema`` (TASK-864): the
    expected set is hand-maintained, so these tests fail the moment the
    literal and the live schema diverge — in either direction — instead of a
    dropped index shipping as a silent performance or integrity regression.
    """

    def test_no_missing_indexes(self, live_index_census):
        """Every pinned index must exist on the live, fully-migrated DB."""
        missing = sorted(set(EXPECTED_CHACHANOTES_INDEXES) - set(live_index_census))
        assert not missing, (
            f"Pinned ChaChaNotes indexes are MISSING from the live schema: "
            f"{missing}. A migration dropped or renamed them, and no other "
            f"test turns red for that (task-19045: a deleted CREATE INDEX in "
            f"a migration step is otherwise silent). If the drop/rename is "
            f"deliberate, update EXPECTED_CHACHANOTES_INDEXES in {_THIS_FILE} "
            f"in the same commit as the schema change; otherwise restore the "
            f"CREATE INDEX in tldw_chatbook/DB/ChaChaNotes_DB.py (or the "
            f"migrations/*.sql file the step executes)."
        )

    def test_no_unexpected_indexes(self, live_index_census):
        """Every live named index must be pinned in the expected literal."""
        unexpected = sorted(set(live_index_census) - set(EXPECTED_CHACHANOTES_INDEXES))
        paste_lines = "\n".join(
            f'    "{name}": IndexPin('
            f'"{live_index_census[name].table}", '
            f"{live_index_census[name].unique}, "
            f"{live_index_census[name].columns!r}),"
            for name in unexpected
        )
        assert not unexpected, (
            f"Live ChaChaNotes schema defines indexes not pinned in "
            f"EXPECTED_CHACHANOTES_INDEXES: {unexpected}. If your migration "
            f"deliberately adds them, pin them in {_THIS_FILE} (sorted by "
            f"name) in the same commit — ready to paste:\n{paste_lines}"
        )

    def test_index_shapes_match(self, live_index_census):
        """Table, UNIQUE flag, and key-column tuple must match per index.

        The UNIQUE flag comparison is the integrity half of this census: a
        ``unique=True`` pin going ``False`` live means duplicate rows the
        application assumes impossible have become legal — treat that as a
        data-integrity regression unless the migration explicitly intends it.
        """
        divergent = []
        for name in sorted(set(EXPECTED_CHACHANOTES_INDEXES) & set(live_index_census)):
            expected = EXPECTED_CHACHANOTES_INDEXES[name]
            live = live_index_census[name]
            if expected != live:
                note = ""
                if expected.unique and not live.unique:
                    note = "  <-- UNIQUE flag LOST (integrity regression)"
                divergent.append(
                    f"{name}: expected {expected!r} != live {live!r}{note}"
                )
        assert not divergent, (
            "Pinned index shapes diverge from the live ChaChaNotes schema "
            "(update EXPECTED_CHACHANOTES_INDEXES in "
            f"{_THIS_FILE} only if the schema change is deliberate):\n"
            + "\n".join(divergent)
        )


class TestLoadBearingUniqueIndexes:
    """AC #2's minimum pin, kept independent of the big literal.

    A developer mechanically updating ``EXPECTED_CHACHANOTES_INDEXES`` on a
    red census could ride a UNIQUE downgrade through the shape test; the
    ledger index below gets its own named guard so that downgrade must be
    confronted explicitly.
    """

    def test_trajectory_ledger_index_is_unique(self, live_index_census):
        """(conversation_id, seq) is the trajectory ledger's ordering identity.

        ``message_trajectory_metadata`` rows are an append-only sidecar
        ledger ordered by ``seq`` within a conversation; without UNIQUE,
        duplicate seq values become legal and replay order is ambiguous.
        """
        name = "idx_message_trajectory_conv_seq"
        assert name in live_index_census, (
            f"{name} is missing from the live schema entirely — the "
            f"trajectory ledger has lost its ordering-identity index."
        )
        live = live_index_census[name]
        assert live.unique, (
            f"{name} exists but is no longer UNIQUE (live pin: {live!r}). "
            f"Duplicate (conversation_id, seq) ledger rows are now legal — "
            f"an integrity regression; restore the UNIQUE in "
            f"tldw_chatbook/DB/migrations/"
            f"chachanotes_v37_to_v38_message_trajectory_metadata.sql."
        )
        assert live.columns == ("conversation_id", "seq"), (
            f"{name} no longer covers (conversation_id, seq): {live!r}"
        )
