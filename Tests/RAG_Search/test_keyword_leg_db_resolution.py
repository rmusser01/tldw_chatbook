"""The keyword leg must use the configured media DB and never write.

Today it guesses paths (media_db.db / chacha_notes.db) that can never match
the real tldw_cli_media_v2.db, opens the ChaChaNotes DB with media-schema
SQL, and on a total miss CREATES a MediaDatabase as a search side effect.
"""
import asyncio
from pathlib import Path
import pytest

from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService


def _make_service(tmp_path, media_db_path=None):
    """Build a RAGService with the in-memory vector store and mock (offline,
    deterministic) embeddings backend -- same pattern as
    Tests/RAG/test_ingestion_indexing.py's `_make_real_service`.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.chunking.chunk_size = 60
    cfg.chunking.chunk_overlap = 10
    cfg.search.enable_cache = False
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    return RAGService(cfg)


def test_missing_media_db_returns_empty_and_creates_nothing(tmp_path):
    service = _make_service(tmp_path, media_db_path=tmp_path / "absent.db")
    results = asyncio.run(service._keyword_search("anything", top_k=5))
    assert results == []
    assert not (tmp_path / "absent.db").exists()          # no create-on-miss
    assert list(tmp_path.glob("*.db")) == []              # no rogue DB anywhere


def test_traversal_shaped_media_db_path_is_rejected_before_any_db_open(
    tmp_path, monkeypatch
):
    """(Qodo PR #1428 finding 1) `config.search.media_db_path` reached a
    filesystem check + DB open without running through path_validation.py.
    Proof this matters (not just a naive existence check): the traversal
    string below resolves, at the OS level, to a REAL db file -- so
    pre-fix code's `.exists()`/`.is_file()` gate would happily pass it
    through to `get_connection_pool`. The fix must reject it earlier, by
    running it through the shared path_validation helpers (mirroring
    config.py's `_get_custom_database_path` treatment), before
    `get_connection_pool` -- and therefore `MediaDatabase` -- is ever
    reached, degrading to [] the same way a missing DB does (never raise,
    never write)."""
    import tldw_chatbook.RAG_Search.simplified.rag_service as rag_service_mod

    real_subdir = tmp_path / "a" / "b"
    real_subdir.mkdir(parents=True)
    real_db_path = tmp_path / "media.db"
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = MediaDatabase(db_path=str(real_db_path), client_id="test_traversal_guard")
    db.close_connection()

    calls = []
    original_get_connection_pool = rag_service_mod.get_connection_pool

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_get_connection_pool(*args, **kwargs)

    monkeypatch.setattr(rag_service_mod, "get_connection_pool", _spy)

    # OS-resolvable traversal: a/b/../../media.db -> tmp_path/media.db,
    # a real, valid MediaDatabase -- unlike an unresolvable nonsense path,
    # this would actually be opened by pre-fix code.
    malicious_path = str(real_subdir / ".." / ".." / "media.db")
    assert "../.." in malicious_path, "test setup must produce a traversal string"

    service = _make_service(tmp_path, media_db_path=malicious_path)
    results = asyncio.run(service._keyword_search("anything", top_k=5))

    assert results == []
    assert calls == [], "get_connection_pool must never be reached for a rejected path"


def test_symlinked_media_db_path_yields_empty_and_no_db_open(tmp_path):
    """A `media_db_path` pointing at a symlink must never actually open a
    database through it. `validate_path_simple` defers symlink authority to
    the private SQLite owner (mirrors config.py's `_get_custom_database_
    path`), so this is enforced by `MediaDatabase` -> `connect_private_
    sqlite`'s no-follow open. Proven end to end: the symlink target is a
    REAL, searchable MediaDatabase with a matching row, so if the no-follow
    guard ever regressed this test would see that row come back instead
    of []."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    real_target = outside_dir / "real_media.db"

    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = MediaDatabase(db_path=str(real_target), client_id="test_symlink_guard")
    media_id, _, message = db.add_media_with_keywords(
        title="Symlink Bait",
        content="This row must never be reachable through a symlinked media_db_path.",
        media_type="article",
        author="Tester",
        url="https://example.com/symlink-bait",
    )
    assert media_id is not None, f"seed failed: {message}"
    db.close_connection()

    symlink_path = tmp_path / "media_via_symlink.db"
    symlink_path.symlink_to(real_target)

    service = _make_service(tmp_path, media_db_path=symlink_path)
    results = asyncio.run(service._keyword_search("Bait", top_k=5))
    assert results == []


def test_keyword_rows_carry_media_source_type(tmp_path):
    db_path = tmp_path / "tldw_cli_media_v2.db"
    # Create a real MediaDatabase and insert one item via its public API
    # (add_media_with_keywords) so media_fts is populated by triggers.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(db_path=str(db_path), client_id="test_keyword_leg")
    media_id, media_uuid, message = db.add_media_with_keywords(
        title="Wombat Field Notes",
        content="Field notes mentioning wombat behavior in the wild.",
        media_type="article",
        author="J. Naturalist",
        url="https://example.com/wombat-field-notes",
    )
    assert media_id is not None, f"seed failed: {message}"
    db.close_connection()

    service = _make_service(tmp_path, media_db_path=db_path)
    results = asyncio.run(service._keyword_search("wombat", top_k=5))
    assert results, "FTS row expected"
    assert results[0].metadata["source_type"] == "media"


def test_default_resolution_uses_get_media_db_path(monkeypatch, tmp_path):
    sentinel = tmp_path / "sentinel_media.db"
    import tldw_chatbook.config as cfg
    monkeypatch.setattr(cfg, "get_media_db_path", lambda **kw: sentinel)
    service = _make_service(tmp_path)  # no explicit path configured
    results = asyncio.run(service._keyword_search("anything", top_k=5))
    assert results == []                                   # sentinel absent
    assert not sentinel.exists()                           # still no writes


def test_keyword_search_orders_strongest_match_first(tmp_path):
    """FTS5's hidden `rank` column is smaller-is-better (more negative =
    stronger match). Review finding on this task's PR: `_perform_fts5_search`
    selected `-rank as rank` and then `ORDER BY rank` -- ordering ASCENDING
    on the NEGATED alias, which is worst-match-first. That bug was invisible
    while the leg always returned [] (pre-fix), but is now live and actively
    selects the worst matches, including when a small `top_k` truncates the
    (already over-fetched, wrongly-ordered) row list -- silently dropping the
    single best match in favor of noise.

    Seeds three real media rows with deliberately lopsided relevance for the
    same query term: a short document saturated with the term (strong), a
    medium document mentioning it once amid modest filler (medium), and a
    long document mentioning it exactly once buried in heavy filler (weak,
    penalized hard by bm25's document-length normalization).
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db_path = tmp_path / "tldw_cli_media_v2.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test_keyword_leg_rank")

    strong_id, _, msg = db.add_media_with_keywords(
        title="platypus platypus platypus field study",
        content=(
            ("platypus " * 40)
            + "Extensive notes on platypus behavior, platypus colonies, "
            "and platypus diet in the wild."
        ),
        media_type="article",
        author="A. Strong",
        url="https://example.com/strong-platypus",
    )
    assert strong_id is not None, f"seed failed: {msg}"

    medium_id, _, msg = db.add_media_with_keywords(
        title="Field Notes",
        content=(
            ("Several unrelated observations were logged today. " * 5)
            + "A single platypus was spotted near the creek bank. "
            + ("Weather conditions remained mild throughout. " * 5)
        ),
        media_type="article",
        author="B. Medium",
        url="https://example.com/medium-platypus",
    )
    assert medium_id is not None, f"seed failed: {msg}"

    weak_id, _, msg = db.add_media_with_keywords(
        title="Annual Wildlife Survey Report",
        content=(
            (
                "This section of the report covers general survey "
                "methodology and background unrelated to any single "
                "species observation. "
                * 60
            )
            + "A platypus was mentioned once in passing near the end of "
            "this very long report."
        ),
        media_type="article",
        author="C. Weak",
        url="https://example.com/weak-platypus",
    )
    assert weak_id is not None, f"seed failed: {msg}"
    db.close_connection()

    service = _make_service(tmp_path, media_db_path=db_path)

    # Full result set: strongest match must be FIRST, not last.
    results = asyncio.run(service._keyword_search("platypus", top_k=5))
    ids = [r.metadata["doc_id"] for r in results]
    assert ids and ids[0] == str(strong_id), (
        f"expected strongest match ({strong_id}) first, got order {ids}"
    )

    # Small top_k: the single best match must survive truncation, not be
    # silently dropped in favor of the weak match.
    top2 = asyncio.run(service._keyword_search("platypus", top_k=2))
    top2_ids = {r.metadata["doc_id"] for r in top2}
    assert str(strong_id) in top2_ids, (
        f"strongest match ({strong_id}) dropped from top_k=2 results: {top2_ids}"
    )
    assert str(weak_id) not in top2_ids, (
        f"weak match ({weak_id}) preferentially kept over a stronger match "
        f"under top_k=2: {top2_ids}"
    )


def test_keyword_rows_render_their_real_title_in_library_evidence(tmp_path):
    """A keyword-leg row must reach the Library evidence list titled.

    Found live (Task 11 walkthrough, Hybrid Full profile): a hybrid search
    whose semantic leg was empty rendered its one FTS-leg row as
    "1. Untitled source | keyword match" while its own citation line read
    "Citations: meeting_notes" -- the title was there, under a key the
    display layer does not read.

    The vector leg's chunk metadata carries `title` (spread in from the
    indexing call's document metadata); this leg builds its metadata from
    scratch and stamped only `doc_title`, which
    `library_local_rag_search_service._semantic_row` -- the mapper every
    engine-backed Library row goes through -- does not consult. The symptom
    was unreachable before this branch because the keyword leg could never
    return a row at all.
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Library.library_local_rag_search_service import _semantic_row
    from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow

    db_path = tmp_path / "tldw_cli_media_v2.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test_keyword_title")
    media_id, _, message = db.add_media_with_keywords(
        title="Quokka Census Notes",
        content="Census notes recording quokka sightings across the island.",
        media_type="article",
        author="R. Surveyor",
        url="https://example.com/quokka-census",
    )
    assert media_id is not None, f"seed failed: {message}"
    db.close_connection()

    service = _make_service(tmp_path, media_db_path=db_path)
    results = asyncio.run(service._keyword_search("quokka", top_k=5))
    assert results, "FTS row expected"

    row = _semantic_row(results[0])
    assert row["title"] == "Quokka Census Notes", (
        "keyword-leg row lost its title on the way to the Library evidence "
        f"list: {row['title']!r} (metadata keys: {sorted(results[0].metadata)})"
    )
    # And through the display-state normalizer the panel actually renders.
    assert (
        LibraryRagResultRow.from_result(row).title == "Quokka Census Notes"
    ), "row renders as 'Untitled source' in the evidence list"
