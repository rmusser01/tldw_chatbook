"""The trigger-maintained conversation<->dictionary attachment index (task-15469).

`list_dictionary_conversations` used to answer "which conversations use this
dictionary?" with `metadata LIKE '%active_dictionaries%'` -- a full scan of
`conversations` plus a JSON parse of every match -- on the event loop, on every
dictionary row click. It is now an indexed lookup over two derived tables
maintained by SQLite triggers (ChaChaNotes V34->V35).

The bar for that swap is EQUALITY, not "close enough": these tests pin the new
implementation against a byte-for-byte re-implementation of the old scan over a
corpus of metadata shapes that includes everything the old code tolerated
(malformed JSON, non-list values, non-dict documents, string/float/bool/null
ids, duplicate keys, escaped keys) -- for EVERY dictionary id, including ids no
conversation references.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
    statistics_from_record,
)
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


@pytest.fixture
def dictionary_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat_dictionaries.db", "test-client")
    yield db
    db.close_connection()


# --- the corpus -------------------------------------------------------------
# Every shape the old scan could meet. `title` doubles as the label so a
# failure names the offending shape.
METADATA_CORPUS: list[tuple[str, str | None]] = [
    ("plain-ints", '{"active_dictionaries": [1, 2]}'),
    ("single-int", '{"active_dictionaries": [11]}'),
    ("empty-list", '{"active_dictionaries": []}'),
    ("null-value", '{"active_dictionaries": null}'),
    ("non-list-value", '{"active_dictionaries": 5}'),
    ("string-id", '{"active_dictionaries": ["3"]}'),
    ("padded-string-id", '{"active_dictionaries": [" 3 "]}'),
    ("underscore-string-id", '{"active_dictionaries": ["1_0"]}'),
    ("float-id", '{"active_dictionaries": [3.9]}'),
    ("bool-ids", '{"active_dictionaries": [true, false]}'),
    ("null-element", '{"active_dictionaries": [null]}'),
    ("object-element", '{"active_dictionaries": [{"a": 1}]}'),
    ("array-element", '{"active_dictionaries": [[1]]}'),
    ("mixed-elements", '{"active_dictionaries": [null, {"z": 1}, 4]}'),
    ("duplicate-ids", '{"active_dictionaries": [2, 2, 3]}'),
    ("duplicate-key", '{"active_dictionaries":[1],"active_dictionaries":[2]}'),
    ("escaped-key", '{"\\u0061ctive_dictionaries": [7]}'),
    ("huge-int", '{"active_dictionaries": [99999999999999999999]}'),
    ("negative-id", '{"active_dictionaries": [-3]}'),
    ("nan-literal", '{"active_dictionaries": [1], "x": NaN}'),
    ("marker-in-prose", '{"note": "see active_dictionaries", "x": 1}'),
    ("marker-twice", '{"active_dictionaries": [6], "n": "active_dictionaries"}'),
    ("truncated-json", '{"active_dictionaries": [1]'),
    ("single-quotes", "{'active_dictionaries': [1]}"),
    ("bare-scalar", "5"),
    ("bare-string-marker", '"active_dictionaries"'),
    ("top-level-array", "[1, 2]"),
    ("no-marker", '{"title": "hi"}'),
    ("empty-string", ""),
    ("null-metadata", None),
    ("sibling-keys", '{"rag_scope": {"a": 1}, "active_dictionaries": [1, 9]}'),
]

#: ids to sweep: every id any corpus row could plausibly resolve to, plus ids
#: nothing references (the 1-vs-11 substring trap lives in this range too).
SWEEP_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 99]


def _reference_used_by(db: CharactersRAGDB, dictionary_id: int) -> list[dict[str, Any]]:
    """The pre-task-15469 implementation, verbatim, as the parity oracle."""
    did = int(dictionary_id)
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT id, title, metadata FROM conversations "
        "WHERE deleted = 0 AND metadata LIKE '%active_dictionaries%'"
    ).fetchall()
    conversations: list[dict[str, Any]] = []
    for row in rows:
        try:
            is_member = did in LocalChatDictionaryService._active_dictionaries(
                {"metadata": row["metadata"]}
            )
        except Exception:
            continue
        if is_member:
            conversations.append(
                {
                    "conversation_id": str(row["id"]),
                    "title": str(row["title"] or ""),
                }
            )
    return conversations


def _seed_corpus(db: CharactersRAGDB) -> dict[str, str]:
    """One conversation per corpus row; returns {label: conversation_id}."""
    ids: dict[str, str] = {}
    for label, metadata in METADATA_CORPUS:
        conversation_id = db.add_conversation({"title": label})
        record = db.get_conversation_by_id(conversation_id)
        db.update_conversation(
            conversation_id,
            {"metadata": metadata},
            expected_version=record["version"],
        )
        ids[label] = conversation_id
    return ids


class TestParityWithTheOldScan:
    def test_every_dictionary_id_matches_the_old_scan(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        _seed_corpus(dictionary_db)

        for dictionary_id in SWEEP_IDS:
            indexed = service.list_dictionary_conversations(dictionary_id)
            assert indexed["source"] == "local"
            assert indexed["conversations"] == _reference_used_by(
                dictionary_db, dictionary_id
            ), f"used-by diverged for dictionary id {dictionary_id}"

    def test_corpus_actually_exercises_both_branches(self, dictionary_db):
        """Guard the guard: a corpus that resolves everything proves nothing."""
        _seed_corpus(dictionary_db)
        conn = dictionary_db.get_connection()
        resolved = conn.execute(
            "SELECT count(*) FROM conversation_dictionary_attachments"
        ).fetchone()[0]
        unresolved = conn.execute(
            "SELECT count(*) FROM conversation_dictionary_unresolved"
        ).fetchone()[0]
        assert resolved > 0
        assert unresolved > 0

    def test_python_verdict_beats_the_index_for_a_duplicate_key(self, dictionary_db):
        """A duplicated JSON key resolves last-wins in Python, first-wins in
        SQLite. The row is therefore marked unresolved and Python decides -- so
        the conversation belongs to dictionary 2, NOT to the indexed 1."""
        service = LocalChatDictionaryService(dictionary_db)
        conversation_id = dictionary_db.add_conversation({"title": "dup"})
        record = dictionary_db.get_conversation_by_id(conversation_id)
        dictionary_db.update_conversation(
            conversation_id,
            {"metadata": '{"active_dictionaries":[1],"active_dictionaries":[2]}'},
            expected_version=record["version"],
        )
        conn = dictionary_db.get_connection()
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT dictionary_id FROM conversation_dictionary_attachments "
                "WHERE conversation_id = ?",
                (conversation_id,),
            )
        ] == [(1,)]

        assert service.list_dictionary_conversations(1)["conversations"] == []
        assert [
            row["conversation_id"]
            for row in service.list_dictionary_conversations(2)["conversations"]
        ] == [conversation_id]

    def test_result_order_matches_the_old_scan_order(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        attached = []
        for index in range(6):
            conversation_id = dictionary_db.add_conversation({"title": f"c{index}"})
            service.attach_to_conversation(4, conversation_id)
            attached.append(conversation_id)
        # An unresolved row in the middle must not reorder the rest.
        middle = attached[3]
        record = dictionary_db.get_conversation_by_id(middle)
        dictionary_db.update_conversation(
            middle,
            {"metadata": '{"active_dictionaries": ["4"]}'},
            expected_version=record["version"],
        )
        result = [
            row["conversation_id"]
            for row in service.list_dictionary_conversations(4)["conversations"]
        ]
        assert result == attached
        assert result == [
            row["conversation_id"] for row in _reference_used_by(dictionary_db, 4)
        ]

    def test_soft_deleted_conversations_stay_out(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        kept = dictionary_db.add_conversation({"title": "kept"})
        removed = dictionary_db.add_conversation({"title": "removed"})
        service.attach_to_conversation(1, kept)
        service.attach_to_conversation(1, removed)
        record = dictionary_db.get_conversation_by_id(removed)
        dictionary_db.soft_delete_conversation(
            removed, expected_version=record["version"]
        )

        assert [
            row["conversation_id"]
            for row in service.list_dictionary_conversations(1)["conversations"]
        ] == [kept]
        assert service.list_dictionary_conversations(1)[
            "conversations"
        ] == _reference_used_by(dictionary_db, 1)


class TestIndexMaintenance:
    def test_attach_and_detach_keep_the_index_in_step(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        conn = dictionary_db.get_connection()

        service.attach_to_conversation(3, conversation_id)
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT dictionary_id FROM conversation_dictionary_attachments "
                "WHERE conversation_id = ?",
                (conversation_id,),
            )
        ] == [(3,)]

        service.detach_from_conversation(3, conversation_id)
        assert (
            conn.execute(
                "SELECT count(*) FROM conversation_dictionary_attachments "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            == 0
        )

    def test_a_foreign_metadata_writer_keeps_the_index_correct(self, dictionary_db):
        """The point of maintaining the index with TRIGGERS rather than at
        attach/detach time: `chat_persistence_service`, `rag_scope` and any
        other `update_conversation` caller rewrite the whole metadata blob."""
        service = LocalChatDictionaryService(dictionary_db)
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        service.attach_to_conversation(5, conversation_id)

        record = dictionary_db.get_conversation_by_id(conversation_id)
        metadata = json.loads(record["metadata"])
        metadata["rag_scope"] = {"mode": "collections"}
        metadata["active_dictionaries"] = [5, 8]
        dictionary_db.update_conversation(
            conversation_id,
            {"metadata": json.dumps(metadata)},
            expected_version=record["version"],
        )

        assert [
            row["conversation_id"]
            for row in service.list_dictionary_conversations(8)["conversations"]
        ] == [conversation_id]

        record = dictionary_db.get_conversation_by_id(conversation_id)
        dictionary_db.update_conversation(
            conversation_id,
            {"metadata": json.dumps({"rag_scope": {"mode": "collections"}})},
            expected_version=record["version"],
        )
        assert service.list_dictionary_conversations(5)["conversations"] == []
        assert service.list_dictionary_conversations(8)["conversations"] == []

    @pytest.mark.parametrize("foreign_keys", [True, False])
    def test_conversation_id_change_leaves_no_stale_rows(
        self, dictionary_db, foreign_keys
    ):
        """The FK's ON UPDATE CASCADE runs BEFORE the AFTER UPDATE trigger.

        A conversation id change renames the index rows to NEW.id first, so a
        trigger deleting only `WHERE conversation_id = OLD.id` matched nothing
        and the OLD dictionary ids survived under the NEW id -- this exact
        UPDATE (id AND metadata, [1, 2] -> [3]) used to leave 1, 2 AND 3
        indexed. Parametrized over `PRAGMA foreign_keys` because the two modes
        fail in opposite directions: with cascade the stale rows follow the new
        id, without it they linger under the old one.
        """
        service = LocalChatDictionaryService(dictionary_db)
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        record = dictionary_db.get_conversation_by_id(conversation_id)
        dictionary_db.update_conversation(
            conversation_id,
            {"metadata": json.dumps({"active_dictionaries": [1, 2]})},
            expected_version=record["version"],
        )
        conn = dictionary_db.get_connection()
        if not foreign_keys:
            conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                "UPDATE conversations SET id = ?, metadata = ? WHERE id = ?",
                (
                    "renamed-conversation",
                    json.dumps({"active_dictionaries": [3]}),
                    conversation_id,
                ),
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT conversation_id, dictionary_id "
                "FROM conversation_dictionary_attachments"
            )
        ] == [("renamed-conversation", 3)]
        assert service.list_dictionary_conversations(1)["conversations"] == []
        assert service.list_dictionary_conversations(2)["conversations"] == []
        assert [
            row["conversation_id"]
            for row in service.list_dictionary_conversations(3)["conversations"]
        ] == ["renamed-conversation"]
        # ...and the answer still matches the old scan for every id involved.
        for dictionary_id in (1, 2, 3):
            assert service.list_dictionary_conversations(dictionary_id)[
                "conversations"
            ] == _reference_used_by(dictionary_db, dictionary_id)

    def test_hard_delete_clears_index_rows(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        service.attach_to_conversation(2, conversation_id)
        conn = dictionary_db.get_connection()
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

        assert (
            conn.execute(
                "SELECT count(*) FROM conversation_dictionary_attachments"
            ).fetchone()[0]
            == 0
        )

    def test_malformed_metadata_never_breaks_the_write(self, dictionary_db):
        """The triggers call json_each/json_type, which RAISE on malformed
        JSON; a raising trigger would fail the conversation write itself."""
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        for metadata in (
            '{"active_dictionaries": [1]',
            "active_dictionaries not json",
            '{"active_dictionaries": "  "}',
            "5",
        ):
            record = dictionary_db.get_conversation_by_id(conversation_id)
            dictionary_db.update_conversation(
                conversation_id,
                {"metadata": metadata},
                expected_version=record["version"],
            )
            assert (
                dictionary_db.get_conversation_by_id(conversation_id)["metadata"]
                == metadata
            )


class TestNoFullScan:
    def test_used_by_query_plan_never_scans_conversations(self, dictionary_db):
        """`conversations` must only ever be probed by primary key.

        The alias in the query is `conversation`, so this asserts on that
        prefix -- matching on the literal table name would pass vacuously.
        Both joins are CROSS JOINs precisely because the planner otherwise
        chose `SCAN conversation` as the unresolved branch's outer loop.
        """
        _seed_corpus(dictionary_db)
        conn = dictionary_db.get_connection()
        steps = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN " + LocalChatDictionaryService._USED_BY_SQL, (1,)
            ).fetchall()
        ]
        plan = "\n".join(steps)
        assert steps, "EXPLAIN QUERY PLAN returned nothing"
        assert not any(step.startswith("SCAN conversation") for step in steps), plan
        assert (
            "SEARCH conversation USING INDEX sqlite_autoindex_conversations_1" in plan
        ), plan
        assert "idx_conversation_dictionary_attachments_dictionary" in plan, plan

    def test_selection_issues_at_most_two_statements(self, dictionary_db):
        """The DB work behind one dictionary row click: the record load and the
        attachment lookup. Statistics are derived from the loaded record and
        the versions baseline is seeded from it, so neither re-reads the row."""
        service = LocalChatDictionaryService(dictionary_db)
        created = service.create_dictionary({"name": "Meds"})
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        service.attach_to_conversation(created["id"], conversation_id)

        statements: list[str] = []
        conn = dictionary_db.get_connection()
        conn.set_trace_callback(statements.append)
        try:
            # Exactly what PersonasScreen._select_dictionary does.
            record = service.get_dictionary(created["id"])
            stats = statistics_from_record(record, dictionary_id=created["id"])
            service.list_versions(created["id"], record=record)
            used_by = service.list_dictionary_conversations(created["id"])
        finally:
            conn.set_trace_callback(None)

        # `SELECT 1` is the connection liveness ping (task-261), not query work.
        queries = [
            statement
            for statement in statements
            if statement.strip().upper().startswith("SELECT")
            and statement.strip() != "SELECT 1"
        ]
        # Exactly two, and both identified -- an empty trace would satisfy
        # "at most two" while proving nothing.
        assert len(queries) == 2, queries
        assert "FROM chat_dictionaries" in queries[0]
        assert "conversation_dictionary_attachments" in queries[1]
        assert stats["entry_count"] == 0
        assert [row["conversation_id"] for row in used_by["conversations"]] == [
            conversation_id
        ]


class TestStatisticsParity:
    def test_derived_statistics_equal_the_service_payload(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        created = service.create_dictionary(
            {
                "name": "Meds",
                "entries": [
                    {"pattern": "BP", "replacement": "blood pressure"},
                    {"pattern": "HR", "replacement": "heart rate"},
                ],
            }
        )
        disabled = service.create_dictionary({"name": "Off", "is_active": False})

        for dictionary_id in (created["id"], disabled["id"]):
            record = service.get_dictionary(dictionary_id)
            assert statistics_from_record(
                record, dictionary_id=dictionary_id
            ) == service.get_statistics(dictionary_id)

    def test_seeded_versions_baseline_matches_the_loaded_one(self, dictionary_db):
        """`list_versions(record=...)` skips a reload -- the baseline snapshot
        it writes must be the one the reload would have produced."""
        service = LocalChatDictionaryService(dictionary_db)
        created = service.create_dictionary(
            {"name": "Meds", "entries": [{"pattern": "BP", "replacement": "bp"}]}
        )
        service._history = {}  # drop the create-time history
        seeded = service.list_versions(
            created["id"], record=service.get_dictionary(created["id"])
        )

        other = LocalChatDictionaryService(dictionary_db)
        other._history = {}
        loaded = other.list_versions(created["id"])

        def _without_timestamp(versions):
            return [
                {key: value for key, value in version.items() if key != "created_at"}
                for version in versions
            ]

        assert _without_timestamp(seeded["versions"]) == _without_timestamp(
            loaded["versions"]
        )
        assert (
            service._history_bucket(created["id"])["versions"][0]["snapshot"]
            == other._history_bucket(created["id"])["versions"][0]["snapshot"]
        )

    def test_derived_statistics_accept_the_raw_record_shape(self, dictionary_db):
        """`get_statistics` reads a raw `load_chat_dictionary` record (entries
        are ChatDictionary objects); the screen passes the normalized one."""
        from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl

        service = LocalChatDictionaryService(dictionary_db)
        created = service.create_dictionary(
            {"name": "Meds", "entries": [{"pattern": "BP", "replacement": "bp"}]}
        )
        raw = cdl.load_chat_dictionary(dictionary_db, created["id"])
        normalized = service.get_dictionary(created["id"])
        assert statistics_from_record(raw) == statistics_from_record(normalized)


class TestUsedByCountsForListing:
    def test_include_usage_counts_match_the_old_scan(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        first = service.create_dictionary({"name": "One"})
        second = service.create_dictionary({"name": "Two"})
        _seed_corpus(dictionary_db)
        chat = dictionary_db.add_conversation({"title": "chat"})
        service.attach_to_conversation(first["id"], chat)

        listed = service.list_dictionaries(include_usage=True)
        usage = {
            record["id"]: record["usage"]["conversation_count"]
            for record in listed["dictionaries"]
        }
        assert usage[first["id"]] == len(_reference_used_by(dictionary_db, first["id"]))
        assert usage[second["id"]] == len(
            _reference_used_by(dictionary_db, second["id"])
        )


class _ThreadRecordingBackend:
    """Local-service-shaped double that records the thread it ran on."""

    def __init__(self, *, is_memory_db: bool):
        self.db = type("_DB", (), {"is_memory_db": is_memory_db})()
        self.thread_ident: int | None = None

    def get_dictionary(self, dictionary_id: int) -> dict[str, Any]:
        self.thread_ident = threading.get_ident()
        return {"id": int(dictionary_id), "source": "local"}

    def list_dictionary_conversations(self, dictionary_id: int) -> dict[str, Any]:
        self.thread_ident = threading.get_ident()
        return {"conversations": [], "source": "local"}

    def add_entry(self, dictionary_id: int, request_data: Any) -> dict[str, Any]:
        self.thread_ident = threading.get_ident()
        return {"source": "local"}

    def reorder_entries(self, dictionary_id: int, request_data: Any) -> dict[str, Any]:
        self.thread_ident = threading.get_ident()
        return {"source": "local"}

    def list_versions(self, dictionary_id: int, **kwargs: Any) -> dict[str, Any]:
        self.thread_ident = threading.get_ident()
        return {"versions": [], "source": "local"}


@pytest.mark.asyncio
class TestBackendRunsOffTheEventLoop:
    @pytest.mark.parametrize(
        "call",
        [
            lambda scope: scope.get_dictionary(1, mode="local"),
            lambda scope: scope.list_dictionary_conversations(1, mode="local"),
            lambda scope: scope.add_entry(1, {}, mode="local"),
            lambda scope: scope.reorder_entries(1, {}, mode="local"),
            lambda scope: scope.list_versions(1, mode="local"),
        ],
    )
    async def test_file_backed_local_calls_run_on_a_worker_thread(self, call):
        backend = _ThreadRecordingBackend(is_memory_db=False)
        scope = ChatDictionaryScopeService(
            local_service=backend, server_service=None, policy_enforcer=None
        )
        await call(scope)
        assert backend.thread_ident is not None
        assert backend.thread_ident != threading.get_ident()

    async def test_memory_backed_local_calls_stay_on_the_loop_thread(self):
        """A `:memory:` sqlite DB is visible only to the thread that created
        it, so threading the call would hand a worker an empty database."""
        backend = _ThreadRecordingBackend(is_memory_db=True)
        scope = ChatDictionaryScopeService(
            local_service=backend, server_service=None, policy_enforcer=None
        )
        await scope.get_dictionary(1, mode="local")
        assert backend.thread_ident == threading.get_ident()

    async def test_an_unidentifiable_backend_stays_on_the_loop_thread(self):
        """Positive confirmation: a double with no `.db` is NOT threaded."""

        class _Bare:
            thread_ident: int | None = None

            def get_dictionary(self, dictionary_id: int) -> dict[str, Any]:
                self.thread_ident = threading.get_ident()
                return {"id": dictionary_id}

        backend = _Bare()
        scope = ChatDictionaryScopeService(
            local_service=backend, server_service=None, policy_enforcer=None
        )
        await scope.get_dictionary(1, mode="local")
        assert backend.thread_ident == threading.get_ident()

    async def test_real_service_used_by_runs_off_the_loop(self, dictionary_db):
        service = LocalChatDictionaryService(dictionary_db)
        created = service.create_dictionary({"name": "Meds"})
        conversation_id = dictionary_db.add_conversation({"title": "chat"})
        service.attach_to_conversation(created["id"], conversation_id)
        scope = ChatDictionaryScopeService(
            local_service=service, server_service=None, policy_enforcer=None
        )
        seen: dict[str, int] = {}
        original = service.list_dictionary_conversations

        def _record(dictionary_id: int):
            seen["thread"] = threading.get_ident()
            return original(dictionary_id)

        service.list_dictionary_conversations = _record  # type: ignore[assignment]
        result = await scope.list_dictionary_conversations(created["id"], mode="local")
        assert seen["thread"] != threading.get_ident()
        assert [row["conversation_id"] for row in result["conversations"]] == [
            conversation_id
        ]


class TestHistoryStoreConcurrency:
    """The version-history sidecar is now reachable from worker threads.

    Before task-15469 every call into this service was serialized by the event
    loop; threading the scope service means `_record_history`'s
    mutate-bucket-then-rewrite-`<sidecar>.tmp`-then-replace can run twice at
    once, which can publish a half-written file and lose appends.
    """

    def test_concurrent_history_writes_keep_the_sidecar_whole(
        self, dictionary_db, tmp_path
    ):
        """Mutation-checked: with the lock replaced by a no-op this fails
        3 runs out of 3 with `FileNotFoundError` -- one writer's `replace()`
        moves the shared `.tmp` file out from under another's."""
        store = tmp_path / "history.json"
        service = LocalChatDictionaryService(dictionary_db, history_store_path=store)
        records = [
            service.create_dictionary({"name": f"Dictionary {index}"})
            for index in range(4)
        ]
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(records) * 2)

        def _write(record: dict, revision: int) -> None:
            try:
                barrier.wait(timeout=10)
                service._record_history(
                    record["id"], "update", {**record, "version": revision}
                )
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        def _read(record: dict) -> None:
            try:
                barrier.wait(timeout=10)
                for _ in range(10):
                    service.list_versions(record["id"])
                    service.list_activity(record["id"])
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = []
        for revision, record in enumerate(records, start=2):
            threads.append(threading.Thread(target=_write, args=(record, revision)))
            threads.append(threading.Thread(target=_read, args=(record,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not [thread for thread in threads if thread.is_alive()], "deadlock"
        assert errors == []
        # The file on disk is whole, and no write was lost.
        persisted = json.loads(store.read_text(encoding="utf-8"))["dictionaries"]
        for revision, record in enumerate(records, start=2):
            revisions = {
                int(version["revision"])
                for version in persisted[str(record["id"])]["versions"]
            }
            assert revision in revisions, (record["id"], revisions)
        assert not store.with_suffix(store.suffix + ".tmp").exists()

    def test_nested_history_lock_does_not_deadlock(self, dictionary_db, tmp_path):
        """`list_versions` -> `_ensure_history_baseline` -> `_record_history`
        re-enters the lock; a non-reentrant Lock would hang here forever."""
        service = LocalChatDictionaryService(
            dictionary_db, history_store_path=tmp_path / "history.json"
        )
        created = service.create_dictionary({"name": "Meds"})
        service._history = {}  # force the baseline-seeding path

        done: list[dict] = []
        worker = threading.Thread(
            target=lambda: done.append(service.list_versions(created["id"]))
        )
        worker.start()
        worker.join(timeout=15)

        assert not worker.is_alive(), "nested history lock deadlocked"
        assert done and done[0]["versions"]


class TestMigrationBackfill:
    def test_v34_database_backfills_existing_attachments(self, tmp_path):
        """Build a genuinely V34-shaped DB (the production chain stops there
        under a patched _CURRENT_SCHEMA_VERSION, task-16840), seed attachment
        metadata while the derived tables and triggers truly do not exist yet,
        then reopen it: the real V34->V35 runner must rebuild the index from
        metadata that was written while no trigger existed."""
        db_path = tmp_path / "v34.db"
        with chachanotes_db_at_version(db_path, 34, client_id="test-client") as db:
            # Genuine-shape preconditions: no derived index machinery exists
            # at V34, so every seed below goes through the historical
            # metadata-only write path (the registry-era fixture seeded at
            # the CURRENT version — triggers populated the index and the
            # rollback then had to drop it again).
            conn = db.get_connection()
            objects = {
                (row["type"], row["name"])
                for row in conn.execute(
                    "SELECT type, name FROM sqlite_master "
                    "WHERE name LIKE 'conversation_dictionary_%'"
                ).fetchall()
            }
            assert objects == set()
            assert (
                conn.execute(
                    "SELECT version FROM db_schema_version WHERE schema_name = ?",
                    ("rag_char_chat_schema",),
                ).fetchone()["version"]
                == 34
            )

            service = LocalChatDictionaryService(db)
            created = service.create_dictionary({"name": "Meds"})
            attached = db.add_conversation({"title": "attached"})
            service.attach_to_conversation(created["id"], attached)
            loose = db.add_conversation({"title": "loose"})
            record = db.get_conversation_by_id(loose)
            db.update_conversation(
                loose,
                {"metadata": '{"active_dictionaries": ["7"]}'},
                expected_version=record["version"],
            )

        migrated = CharactersRAGDB(db_path, "test-client")
        try:
            conn = migrated.get_connection()
            assert (
                conn.execute(
                    "SELECT version FROM db_schema_version WHERE schema_name = ?",
                    ("rag_char_chat_schema",),
                ).fetchone()["version"]
                == migrated._CURRENT_SCHEMA_VERSION
            )
            assert [
                tuple(row)
                for row in conn.execute(
                    "SELECT conversation_id, dictionary_id "
                    "FROM conversation_dictionary_attachments"
                )
            ] == [(attached, created["id"])]
            assert [
                tuple(row)
                for row in conn.execute(
                    "SELECT conversation_id FROM conversation_dictionary_unresolved"
                )
            ] == [(loose,)]

            rebuilt = LocalChatDictionaryService(migrated)
            assert [
                row["conversation_id"]
                for row in rebuilt.list_dictionary_conversations(created["id"])[
                    "conversations"
                ]
            ] == [attached]
            assert [
                row["conversation_id"]
                for row in rebuilt.list_dictionary_conversations(7)["conversations"]
            ] == [loose]
        finally:
            migrated.close_connection()
