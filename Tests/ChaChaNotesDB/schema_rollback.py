"""Shared rollback registry for ChaChaNotes historical-migration fixtures.

Several tests build a "historical" ChaChaNotes DB top-down: bootstrap a fresh
DB (which lands at ``_CURRENT_SCHEMA_VERSION``), remove the artifacts newer
migrations added, stamp ``db_schema_version`` back to the historical version,
and reopen so the production migration chain replays. That approach bakes a
trap: every time a migration ships a non-idempotent artifact (a bare
``CREATE TABLE`` or an unguarded ``ALTER TABLE ... ADD COLUMN``), every
rollback fixture that does not also remove that artifact starts failing with
"already exists" at the new migration's step — while the version stamp claims
the artifact should not exist yet.

That is not a hypothetical: it fired three times in three days.

* ``88f5f535a`` (V33->V34 ``compaction_representation``) broke the rollback
  fixtures; task-15730 repaired them one by one.
* ``9174975b0`` (V35->V36 ``note_folders``, task-15705) broke them again. Its
  author fixed the ONE fixture they knew about
  (``Tests/Character_Chat/test_dictionary_attachment_index.py``) and missed
  the other two, producing task-15765
  (``Tests/ChaChaNotesDB/test_chachanotes_db.py`` V17 fixture) and
  task-16197 (``Tests/Chat/test_conversation_local_marks_service.py`` V16
  fixture), each then repaired separately (task-16201, task-16207).

The registry below is the single place that knowledge now lives. Each key is
a schema version ``v``; its value removes the ``(v-1)->v`` migration's
artifacts whose baked presence would collide on replay (plus a few removed
so the migration under test genuinely re-creates them), in a safe order
(triggers that reference a column before the column itself). Be precise
about what the rolled-back DB IS: a current-version DB with those SPECIFIC
artifacts removed — sufficient for replaying the migrations under test, NOT
a faithful historical vN snapshot. Replay-tolerant migrations' artifacts
survive at the rolled-back stamp (measured at a v17 stamp: 7 post-v17
tables, 9 indexes, 5 columns), and the sync triggers a real vN DB has are
deliberately absent until replay recreates them. After replay, column ORDER
may also diverge from a fresh bootstrap (a dropped column is re-appended at
the end of its table), so compare column membership as a set, never by
position. ``rollback_chachanotes_schema`` walks the registry from the
recorded version down to the requested target, so every fixture — current
and future — shares one drop list instead of hand-maintaining its own. For
a genuinely vN-shaped fixture, bootstrap under a patched
``_CURRENT_SCHEMA_VERSION`` instead (as
``Tests/DB/test_chachanotes_note_folders_migration.py`` does) — the
knowledge-free direction a follow-up will evaluate for these fixtures.

Contract for migration authors: when you bump ``_CURRENT_SCHEMA_VERSION``,
add an entry here for the new version. Declare an empty tuple if (and only
if) your migration tolerates replaying over its own artifacts (``IF NOT
EXISTS`` / guarded ``ALTER``). ``test_schema_rollback.py`` enforces both
halves: a completeness ratchet fails the moment a version has no entry, and
a rollback-replay sweep fails if a declared entry does not actually return
the schema to a replayable state.
"""

from __future__ import annotations

import sqlite3

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME

#: The oldest version a fixture may roll back to. Fixtures that need an even
#: older shape should build bottom-up instead (see legacy_conversation_schema).
MINIMUM_ROLLBACK_VERSION = 16

#: version v -> statements removing what the (v-1)->v migration adds.
#: An empty tuple is an explicit declaration that the migration tolerates
#: replaying over its own baked artifacts.
POST_VERSION_SCHEMA_REMOVALS: dict[int, tuple[str, ...]] = {
    # V16->V17: durable local-only conversation marks.
    17: (
        "DROP INDEX IF EXISTS idx_conversation_local_marks_type",
        "DROP TABLE IF EXISTS conversation_local_marks",
    ),
    # V17->V18: conversations.system_prompt, plus redefined conversations
    # sync triggers that reference it (SQLite refuses to drop a column a
    # trigger references, so the triggers go first; replay recreates them).
    18: (
        "DROP TRIGGER IF EXISTS conversations_sync_create",
        "DROP TRIGGER IF EXISTS conversations_sync_update",
        "DROP TRIGGER IF EXISTS conversations_sync_delete",
        "DROP TRIGGER IF EXISTS conversations_sync_undelete",
        "ALTER TABLE conversations DROP COLUMN system_prompt",
    ),
    # V18->V19 through V25->V26 replay-tolerantly over a baked current-shape
    # DB (guarded ALTERs / IF NOT EXISTS), so no removals are required.
    19: (),
    20: (),
    21: (),
    22: (),
    23: (),
    24: (),
    25: (),
    26: (),
    # V26->V27: citation-provenance tables. That migration deliberately
    # rejects pre-existing/partial tables, so every one must be removed.
    27: (
        "DROP TABLE IF EXISTS rag_artifact_owner_operations",
        "DROP TABLE IF EXISTS rag_artifact_owner_leases",
        "DROP TABLE IF EXISTS rag_source_observations",
        "DROP TABLE IF EXISTS rag_message_trace_owners",
        "DROP TABLE IF EXISTS rag_trace_evidence_refs",
        "DROP TABLE IF EXISTS rag_answer_attempt_payloads",
        "DROP TABLE IF EXISTS rag_evidence_runs",
        "DROP TABLE IF EXISTS rag_citation_traces",
        "DROP TABLE IF EXISTS rag_evidence_snapshots",
        "DROP TABLE IF EXISTS rag_payload_tombstones",
        "DROP TABLE IF EXISTS rag_legacy_migration_journal",
        "DROP TABLE IF EXISTS rag_identity_context",
    ),
    # V27->V28: conversations.assistant_authority_id. The conversations sync
    # triggers (last redefined by V19->V20) do not reference it, so the
    # column drops cleanly on its own — do NOT drop the triggers here:
    # no migration after V19->V20 recreates them, and the sweep proves a
    # trigger drop at this step leaves the replayed DB silently missing all
    # conversations sync triggers for targets in V20..V27.
    28: ("ALTER TABLE conversations DROP COLUMN assistant_authority_id",),
    # V28->V29 (kept briefings) replays tolerantly.
    29: (),
    # V29->V30: local-only messages.usage_json (excluded from sync triggers).
    30: ("ALTER TABLE messages DROP COLUMN usage_json",),
    # V30->V31 through V32->V33 replay tolerantly.
    31: (),
    32: (),
    33: (),
    # V33->V34: unguarded ALTER (the task-15730 incident's collision).
    34: (
        "ALTER TABLE console_conversation_context_policy "
        "DROP COLUMN compaction_representation",
    ),
    # V34->V35: the derived dictionary-attachment index. Tables/indexes are
    # IF NOT EXISTS and triggers are dropped-then-created on replay, but the
    # fixtures remove them anyway so replay genuinely rebuilds (and the
    # dictionary backfill test depends on exactly that).
    35: (
        "DROP TRIGGER IF EXISTS conversation_dictionary_index_ai",
        "DROP TRIGGER IF EXISTS conversation_dictionary_index_au",
        "DROP TRIGGER IF EXISTS conversation_dictionary_index_ad",
        "DROP TABLE IF EXISTS conversation_dictionary_attachments",
        "DROP TABLE IF EXISTS conversation_dictionary_unresolved",
    ),
    # V35->V36: note folders — bare CREATE TABLE, the task-15765/task-16197
    # collision. Memberships first (FK child of note_folders).
    36: (
        "DROP TABLE IF EXISTS note_folder_memberships",
        "DROP TABLE IF EXISTS note_folders",
    ),
    # V36->V37: messages.provider_continuation_json, referenced by the
    # redefined messages sync triggers (replay recreates them). The
    # migration would tolerate a baked column, but removing it makes the
    # fixture genuinely pre-V37 and exercises the real ALTER path.
    37: (
        "DROP TRIGGER IF EXISTS messages_sync_create",
        "DROP TRIGGER IF EXISTS messages_sync_update",
        "DROP TRIGGER IF EXISTS messages_sync_delete",
        "DROP TRIGGER IF EXISTS messages_sync_undelete",
        "ALTER TABLE messages DROP COLUMN provider_continuation_json",
    ),
    # V37->V38: local-only trajectory sidecar (IF NOT EXISTS, but removed so
    # replay genuinely creates it). Its indexes drop with the table.
    38: ("DROP TABLE IF EXISTS message_trajectory_metadata",),
    # V38->V39: local Visual Identity tables. Children must be removed before
    # their referenced pack/version parents.
    39: (
        "DROP TABLE IF EXISTS visual_identity_bindings",
        "DROP TABLE IF EXISTS visual_identity_assets",
        "DROP TABLE IF EXISTS visual_identity_pack_versions",
        "DROP TABLE IF EXISTS visual_identity_packs",
    ),
}


def rollback_chachanotes_schema(conn: sqlite3.Connection, target_version: int) -> None:
    """Rewind a freshly-bootstrapped ChaChaNotes DB to ``target_version``.

    Walks ``POST_VERSION_SCHEMA_REMOVALS`` from the recorded schema version
    down to ``target_version + 1`` and stamps ``db_schema_version`` so that
    reopening the DB replays the production migration chain from a genuinely
    ``target_version``-shaped schema.

    The caller owns the transaction (commit after calling) and the
    connection. Intended for connections whose recorded version is the
    current one (a fresh bootstrap); statements are applied exactly once per
    version on the way down.

    Args:
        conn: An open connection to a freshly-bootstrapped ChaChaNotes DB.
        target_version: The historical schema version to rewind to. Must be
            at least ``MINIMUM_ROLLBACK_VERSION`` and below the recorded
            version.

    Raises:
        AssertionError: If the target is out of range or the registry has no
            entry for a version in the walk (the actionable signal that a new
            migration shipped without declaring its rollback).
    """
    row = conn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None, "ChaChaNotes DB has no recorded schema version"
    recorded = row[0]
    assert MINIMUM_ROLLBACK_VERSION <= target_version < recorded, (
        f"rollback target {target_version} must be in "
        f"[{MINIMUM_ROLLBACK_VERSION}, {recorded - 1}]"
    )
    missing = [
        version
        for version in range(target_version + 1, recorded + 1)
        if version not in POST_VERSION_SCHEMA_REMOVALS
    ]
    assert not missing, (
        f"POST_VERSION_SCHEMA_REMOVALS has no entry for schema version(s) "
        f"{missing}. Every migration must declare how to remove what it adds "
        f"(an empty tuple if replay tolerates the baked artifacts) in "
        f"Tests/ChaChaNotesDB/schema_rollback.py."
    )
    for version in range(recorded, target_version, -1):
        for statement in POST_VERSION_SCHEMA_REMOVALS[version]:
            conn.execute(statement)
    conn.execute(
        "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
        (target_version, SCHEMA_NAME),
    )
