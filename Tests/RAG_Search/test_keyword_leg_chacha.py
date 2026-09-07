"""The engine's FTS leg must cover notes and conversations, not media only.

TASK-3996. `RAGService._keyword_search` joined `media_fts` and nothing else,
so the keyword half of hybrid search could only ever return media rows: on
the P1 fixture corpus, 29 of 49 documents (notes + conversations) were
structurally unreachable through that leg no matter what the query said.

The fix adds two read-only sub-legs over the ChaChaNotes DB (notes,
conversation messages) alongside the existing media sub-leg, interleaved
rank-fairly. Two properties these tests exist to hold down:

* **Read-only, no ORM.** The engine's search path opens ChaChaNotes through
  `sqlite3.connect("file:...?mode=ro", uri=True)` -- structurally incapable
  of writing and never running `CharactersRAGDB`'s constructor-time schema
  work. Tests may use the ORM to *build* fixtures; the engine may not.
* **Fusion-key vocabulary equality.** `_fusion_doc_key` compares raw
  `source_type` strings, so a keyword row only ever merges with its vector
  twin when both stamp the EXACT ingestion vocabulary (`note` /
  `conversation`, singular) and the same bare `source_id`. A plural or
  variant spelling leaves the rows present but never merging -- silently
  reverting this fix's whole purpose, with every test above still green.
  `test_cross_leg_merge_per_source_type` is the pin for that.
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search.ingestion_indexing import (
    conversation_document,
    note_document,
)
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult


def _make_service(media_db_path=None, chachanotes_db_path=None):
    """A RAGService with the in-memory vector store and mock embeddings.

    Same pattern as `test_keyword_leg_db_resolution._make_service`; only the
    two DB paths matter here.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    if chachanotes_db_path is not None:
        cfg.search.chachanotes_db_path = Path(chachanotes_db_path)
    return RAGService(cfg)


def _seed_media(tmp_path, rows, name="tldw_cli_media_v2.db"):
    """Create a real MediaDatabase (FTS populated by its own triggers)."""
    db_path = tmp_path / name
    db = MediaDatabase(db_path=str(db_path), client_id="test_chacha_subleg")
    try:
        for title, content in rows:
            media_id, _, message = db.add_media_with_keywords(
                title=title,
                content=content,
                media_type="article",
                author="Tester",
                url=f"https://example.com/{title.replace(' ', '-').lower()}",
            )
            assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _chacha_db(tmp_path, name="chacha.db"):
    return CharactersRAGDB(tmp_path / name, "test_chacha_subleg")


def _add_conversation(db, title, messages):
    conv_id = db.add_conversation({"title": title})
    assert conv_id, "conversation seed failed"
    for sender, content in messages:
        assert db.add_message(
            {"conversation_id": conv_id, "sender": sender, "content": content}
        ), "message seed failed"
    return conv_id


def _rows_by_type(results):
    out = {}
    for result in results:
        out.setdefault(result.metadata.get("source_type"), []).append(result)
    return out


@pytest.fixture
def warnings_captured():
    """Collect loguru WARNING+ records (pytest's capsys never sees loguru)."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_keyword_leg_returns_note_and_conversation_rows(tmp_path):
    """All three sources reachable, stamped with the ingestion vocabulary."""
    media_path = _seed_media(
        tmp_path,
        [("Wombat Media Digest", "A media article about wombat burrow shapes.")],
    )
    db = _chacha_db(tmp_path)
    note_id = db.add_note(
        "Wombat Field Note", "Wombat burrows are famously cubic in cross-section."
    )
    conv_id = _add_conversation(
        db,
        "Wombat Q&A",
        [
            ("user", "Why is wombat scat cube shaped?"),
            ("assistant", "Intestinal elasticity shapes the wombat pellets."),
        ],
    )
    db.close_connection()

    service = _make_service(media_db_path=media_path, chachanotes_db_path=tmp_path / "chacha.db")
    results = asyncio.run(service._keyword_search("wombat", top_k=10))

    by_type = _rows_by_type(results)
    assert set(by_type) == {"media", "note", "conversation"}, (
        f"FTS leg missed a source type: {sorted(by_type)}"
    )

    note_row = by_type["note"][0]
    assert note_row.metadata["source_id"] == str(note_id)
    assert note_row.metadata["title"] == "Wombat Field Note"
    assert note_row.id == f"note_{note_id}"
    assert note_row.document, "note row must carry its text"

    conv_row = by_type["conversation"][0]
    assert conv_row.metadata["source_id"] == str(conv_id)
    assert conv_row.metadata["title"] == "Wombat Q&A"
    assert conv_row.id == f"conversation_{conv_id}"
    assert conv_row.document, "conversation row must carry matched message text"


def test_sub_legs_interleave_rank_fairly(tmp_path):
    """One media sub-leg must not crowd notes/conversations out of top_k.

    Five media documents all match the query; naive concatenation (media
    first, then chacha) would fill a top_k=3 result set with media alone and
    the note/conversation rows would never be seen. Round-robin interleaving
    by rank position puts each source's best row in the first three slots.
    """
    media_path = _seed_media(
        tmp_path,
        [
            (f"Pelican Media {i}", f"Media document {i} discussing pelican migration.")
            for i in range(5)
        ],
    )
    db = _chacha_db(tmp_path)
    db.add_note("Pelican Note", "Pelican pouches hold more than their stomachs.")
    _add_conversation(
        db, "Pelican Chat", [("user", "Do pelican flocks migrate at night?")]
    )
    db.close_connection()

    service = _make_service(media_db_path=media_path, chachanotes_db_path=tmp_path / "chacha.db")
    results = asyncio.run(service._keyword_search("pelican", top_k=3))

    assert len(results) == 3
    assert {r.metadata.get("source_type") for r in results} == {
        "media",
        "note",
        "conversation",
    }, (
        "top_k=3 was filled by a single sub-leg -- sub-legs are not "
        f"interleaved rank-fairly: {[r.metadata.get('source_type') for r in results]}"
    )


def test_deleted_notes_and_conversations_are_excluded(tmp_path):
    """Soft-deleted rows must never leak out of the raw sub-legs.

    The ORM's `search_notes` filters `notes.deleted = 0`, and
    `search_conversations_by_content` filters BOTH `messages.deleted = 0`
    and `conversations.deleted = 0`. The raw helpers replicate all three --
    a raw query that forgets one turns a user's deletion into a search
    result.

    Two of those filters would be UNTESTABLE without the FTS rebuild below,
    and the rebuild is not a contrivance -- it is the state a maintenance
    rebuild leaves behind, and this repo already issues exactly that
    statement for two other tables (`character_cards_fts`,
    `flashcards_fts`). The soft-delete triggers normally evict a row from
    the index, so the `deleted = 0` predicates look redundant; an
    external-content `'rebuild'` re-indexes the CONTENT TABLE, deleted rows
    included, and the predicates become the only thing standing between a
    deleted note and a search result. Verified by mutation: dropping
    `notes.deleted = 0` left every assertion here green until the rebuild
    was added.

    `conversations.deleted = 0` is load-bearing without any rebuild, since
    deleting a conversation does not soft-delete its messages -- the
    conversation keeps matching through them.
    """
    db = _chacha_db(tmp_path)
    live_note_id = db.add_note("Live Quokka Note", "Quokka sightings on the island.")
    dead_note_id = db.add_note("Dead Quokka Note", "Quokka notes slated for deletion.")
    assert db.soft_delete_note(dead_note_id, expected_version=1)

    live_conv_id = _add_conversation(
        db, "Live Quokka Chat", [("user", "Where do quokka colonies nest?")]
    )
    dead_conv_id = _add_conversation(
        db, "Dead Quokka Chat", [("user", "Quokka questions, since deleted.")]
    )
    dead_conv = db.get_conversation_by_id(dead_conv_id)
    assert db.soft_delete_conversation(
        dead_conv_id, expected_version=dead_conv["version"]
    )

    # A LIVE conversation whose only mention of the redacted term is in a
    # soft-deleted message: the `messages.deleted = 0` filter is the only
    # thing that keeps it out of the results.
    redacted_conv_id = _add_conversation(
        db, "Redacted Quokka Chat", [("user", "Quokka sightings, later redacted.")]
    )
    redacted_messages = db.get_messages_for_conversation(redacted_conv_id)
    assert db.soft_delete_message(
        redacted_messages[0]["id"], expected_version=redacted_messages[0]["version"]
    )

    # Reproduce a post-maintenance index: both FTS tables rebuilt from their
    # content tables, deleted rows and all.
    with db.transaction() as conn:
        conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    db.close_connection()

    service = _make_service(chachanotes_db_path=tmp_path / "chacha.db")
    results = asyncio.run(service._keyword_search("quokka", top_k=10))

    ids = {r.metadata.get("source_id") for r in results}
    assert str(live_note_id) in ids
    assert str(live_conv_id) in ids
    assert str(dead_note_id) not in ids, "soft-deleted note leaked into the FTS leg"
    assert str(dead_conv_id) not in ids, (
        "soft-deleted conversation leaked into the FTS leg"
    )
    assert str(redacted_conv_id) not in ids, (
        "a conversation matched through a soft-deleted message"
    )


def test_conversation_messages_are_aggregated_chronologically(tmp_path):
    """A conversation row's text must read in the ORM's message order.

    The conversation sub-leg renders the matching messages as
    `sender: content` lines. SQLite's `group_concat` has NO defined order,
    so the concatenation order was whatever the query plan happened to
    produce -- in practice the messages' storage order, which is insertion
    order, not chronological order. Any snippet, preview or span-based
    evidence built on that text then depends on a query plan.

    The ORM orders messages by `timestamp ASC`
    (`get_messages_for_conversation`), and the vector leg indexes them in
    exactly that order (`conversation_document` renders whatever the ORM
    handed it), so the two legs must agree.

    This conversation is seeded with the LATER message inserted first, so
    insertion order and chronological order disagree.
    """
    db = _chacha_db(tmp_path)
    conv_id = db.add_conversation({"title": "Numbat Timeline"})
    assert conv_id
    assert db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "SECOND numbat reply, written later, stored first.",
            "timestamp": "2026-02-02T00:00:00.000Z",
        }
    )
    assert db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "FIRST numbat question, written earlier, stored second.",
            "timestamp": "2026-02-01T00:00:00.000Z",
        }
    )
    db.close_connection()

    service = _make_service(chachanotes_db_path=tmp_path / "chacha.db")
    results = asyncio.run(
        service._keyword_search("numbat", top_k=10, include_citations=False)
    )

    conv_rows = _rows_by_type(results).get("conversation") or []
    assert conv_rows, "the conversation sub-leg returned no rows"
    document = conv_rows[0].document
    assert "FIRST" in document and "SECOND" in document, document
    assert document.index("FIRST") < document.index("SECOND"), (
        "matched messages must be concatenated chronologically "
        f"(timestamp ASC), got: {document!r}"
    )


def test_missing_chacha_db_degrades_to_media_only_with_warning(
    tmp_path, warnings_captured
):
    """A missing ChaChaNotes DB kills its sub-legs, not the whole leg."""
    media_path = _seed_media(
        tmp_path, [("Numbat Media", "Media coverage of numbat foraging habits.")]
    )

    service = _make_service(
        media_db_path=media_path, chachanotes_db_path=tmp_path / "absent-chacha.db"
    )
    results = asyncio.run(service._keyword_search("numbat", top_k=10))

    assert results, "media sub-leg must keep working when chacha is missing"
    assert {r.metadata.get("source_type") for r in results} == {"media"}
    assert not (tmp_path / "absent-chacha.db").exists(), "search must never create a DB"
    assert any(
        "ChaChaNotes database not found" in message for message in warnings_captured
    ), f"missing chacha DB degraded silently; warnings: {warnings_captured}"
    assert all("absent-chacha.db" not in message for message in warnings_captured)


def test_chacha_connection_is_read_only(tmp_path):
    """The engine's chacha connection cannot write, by construction."""
    db = _chacha_db(tmp_path)
    db.add_note("Read Only Probe", "Content used to prove the connection reads.")
    db.close_connection()

    service = _make_service(chachanotes_db_path=tmp_path / "chacha.db")
    conn = service._connect_chacha_readonly(tmp_path / "chacha.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO notes (id, title, content, client_id) "
                "VALUES ('probe', 'x', 'y', 'engine')"
            )
    finally:
        conn.close()


def test_chacha_connection_reads_a_live_wal_database(tmp_path):
    """The read-only leg must work against the DB the app is holding open.

    ChaChaNotes runs in WAL mode, and the realistic production shape is
    "app has it open read-write, engine reads it read-only". A read-only
    SQLite connection cannot create the `-shm` file itself, so this is the
    case where a read-only design could fail in production while every
    closed-database test stayed green.
    """
    db = _chacha_db(tmp_path)
    note_id = db.add_note("Wallaby Note", "Wallaby sightings in the scrub.")
    assert db.execute_query("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert (tmp_path / "chacha.db-wal").exists(), "no WAL sidecar to read across"
    try:
        # deliberately NOT closed: the writer still holds the database
        service = _make_service(chachanotes_db_path=tmp_path / "chacha.db")
        results = asyncio.run(service._keyword_search("wallaby", top_k=5))
        assert [r.metadata["source_id"] for r in results] == [str(note_id)]
    finally:
        db.close_connection()


def test_traversal_shaped_chacha_path_is_rejected_before_any_db_open(
    tmp_path, warnings_captured
):
    """`chachanotes_db_path` gets `media_db_path`'s path_validation treatment.

    The traversal string below resolves, at the OS level, to a REAL chacha
    database with a matching row, so a naive `.exists()` gate would happily
    open it -- the rejection has to come from the shared validator, before
    any connection is attempted.
    """
    real_subdir = tmp_path / "a" / "b"
    real_subdir.mkdir(parents=True)
    db = _chacha_db(tmp_path)
    db.add_note("Traversal Bait", "Numbat notes reachable only via traversal.")
    db.close_connection()

    malicious_path = str(real_subdir / ".." / ".." / "chacha.db")
    assert "../.." in malicious_path, "test setup must produce a traversal string"

    service = _make_service(chachanotes_db_path=malicious_path)
    results = asyncio.run(service._keyword_search("numbat", top_k=5))

    assert results == []
    assert any("chachanotes_db_path" in m for m in warnings_captured), (
        f"rejection was silent; warnings: {warnings_captured}"
    )


@pytest.mark.parametrize("link_kind", ["file", "parent_dir"])
def test_symlinked_chacha_path_yields_empty_and_no_db_read(
    tmp_path, link_kind, warnings_captured
):
    """Neither a symlinked DB file nor a symlinked PARENT may be followed.

    Review finding: a final-component-only `is_symlink()` check is strictly
    weaker than what the media sub-leg gets. `MediaDatabase` ->
    `connect_private_sqlite` walks EVERY path component with `O_NOFOLLOW`
    (`verify_trusted_directory`), so a symlinked parent directory is refused
    there while the hand-rolled check followed it and returned the planted
    row. Both shapes are pinned here; the fix is to route this leg through
    the same seam rather than to grow a second walker.
    """
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    db = CharactersRAGDB(outside_dir / "real_chacha.db", "test_chacha_symlink")
    db.add_note("Symlink Bait", "Bandicoot notes that must stay unreachable.")
    db.close_connection()

    if link_kind == "file":
        link = tmp_path / "chacha_via_symlink.db"
        link.symlink_to(outside_dir / "real_chacha.db")
        configured = link
    else:
        link_dir = tmp_path / "dir_via_symlink"
        link_dir.symlink_to(outside_dir, target_is_directory=True)
        configured = link_dir / "real_chacha.db"

    service = _make_service(chachanotes_db_path=configured)
    results = asyncio.run(service._keyword_search("bandicoot", top_k=5))

    assert results == [], f"a symlinked {link_kind} was followed to real data"
    assert warnings_captured, "a refused chacha path must be disclosed, not silent"


def test_unopenable_chacha_file_degrades_with_a_warning(tmp_path, warnings_captured):
    """A file that exists but is not a database costs only its sub-legs."""
    media_path = _seed_media(
        tmp_path, [("Bilby Media", "Media coverage of bilby burrows.")]
    )
    junk = tmp_path / "chacha.db"
    junk.write_bytes(b"this is not a SQLite database")

    service = _make_service(media_db_path=media_path, chachanotes_db_path=junk)
    results = asyncio.run(service._keyword_search("bilby", top_k=5))

    assert {r.metadata.get("source_type") for r in results} == {"media"}, (
        "an unopenable chacha DB must not take the media sub-leg down with it"
    )
    assert warnings_captured, "an unusable chacha DB must be disclosed"


def _vector_row(document, chunk_index, score):
    """A vector-leg row exactly as an indexed chunk of `document` comes back.

    `index_document` spreads the document's own metadata (`source_id`,
    `title`, `source_type` -- straight from `ingestion_indexing`) into every
    chunk and adds the chunk keys on top, so building this row from the real
    `note_document` / `conversation_document` output is what makes this a
    vocabulary pin rather than a restatement of the implementation.
    """
    return SearchResult(
        id=f"{document['id']}_chunk_{chunk_index}",
        score=score,
        document=document["content"][:200],
        metadata={
            **document["metadata"],
            "doc_id": document["id"],
            "chunk_id": f"{document['id']}_chunk_{chunk_index}",
            "chunk_index": chunk_index,
        },
    )


@pytest.mark.parametrize("source_type", ["note", "conversation"])
def test_cross_leg_merge_per_source_type(tmp_path, source_type):
    """A keyword row and its vector twin must fuse into ONE row.

    THE vocabulary-equality pin (TASK-3994 + TASK-3996): fusion matches on
    `(source_type, source_id)`, so a sub-leg that stamped `notes`/`chat` or
    a prefixed id would produce rows that are present but never merge --
    invisible to every other test in this file.
    """
    db = _chacha_db(tmp_path)
    if source_type == "note":
        item_id = db.add_note(
            "Bilby Burrow Note", "Bilby burrows spiral downward to escape the heat."
        )
        note = db.get_note_by_id(item_id)
        document = note_document(note)
    else:
        item_id = _add_conversation(
            db,
            "Bilby Chat",
            [("user", "How deep does a bilby burrow go?")],
        )
        conversation = db.get_conversation_by_id(item_id)
        messages = db.get_messages_for_conversation(item_id)
        document = conversation_document(conversation, messages)
    db.close_connection()
    assert document is not None, "ingestion must consider the fixture indexable"

    service = _make_service(chachanotes_db_path=tmp_path / "chacha.db")
    keyword_results = asyncio.run(
        service._keyword_search("bilby", top_k=5, include_citations=False)
    )
    assert keyword_results, "keyword sub-leg produced no row to fuse"
    assert len(keyword_results) == 1

    semantic_results = [_vector_row(document, 0, score=0.91)]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword_results,
        semantic_results=semantic_results,
        top_k=10,
        alpha=0.5,
        include_citations=False,
    )

    assert len(fused) == 1, (
        f"{source_type} rows did not merge across legs -- keyword metadata "
        f"{keyword_results[0].metadata} vs vector metadata "
        f"{semantic_results[0].metadata}"
    )
    provenance = fused[0].metadata["hybrid_fusion"]
    assert provenance["fts_score"] is not None
    assert provenance["vector_score"] is not None
    assert str(item_id) in {
        str(fused[0].metadata.get("source_id")),
        str(fused[0].metadata.get("doc_id")),
    }
