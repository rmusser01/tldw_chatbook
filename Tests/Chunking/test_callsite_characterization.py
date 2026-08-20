"""Tests/Chunking/test_callsite_characterization.py
Call-site characterization (spec §10.3): every §6.1.1 entry works through
the new engine with a stable output contract. Written against the SHIM
(post-Task-3); Phase B converges the regex-path call sites and these tests
must still pass unchanged.

What each test pins, and against WHICH producer it runs today:

* ``words`` shape + DB round-trip: since Phase B Task 7 these run through
  the Chunk_Lib shim (the engine), no longer chunking_service's own regex
  path. These two tests pin the *contract* — flat chunks with top-level
  text/start_char/end_char/word_count/chunk_index, and non-NULL offset
  columns after ``MediaDatabase.add_media_with_keywords`` — so the
  convergence cannot silently change what call sites and the DB seam see.
* ``ebook_chapters``: Phase B Task 7 removed chunking_service's method
  whitelist, so the method now chunks through the engine (the §7.2
  regression fix; the xfail marker was removed with the whitelist).
* ``XML_Ingestion`` import: the ``chunk_xml`` part of the seam was restored
  by the Task 3 shim; the module has a second, PRE-EXISTING broken import
  unrelated to chunking (see the test for details).
"""

import pytest

from tldw_chatbook.RAG_Search import chunking_service


TEXT = ("The first sentence is here. The second sentence follows. "
        "A third sentence for good measure. And a fourth one. " * 5)


def test_book_ingestion_regex_path_shape():
    # Book_Ingestion_Lib:1793 → RAG_Search.chunking_service.improved_chunking_process.
    # Characterizes today's producer (chunking_service's own regex path);
    # Phase B must keep this output shape byte-for-byte compatible.
    chunks = chunking_service.improved_chunking_process(
        TEXT, {"method": "words", "max_size": 10, "overlap": 2}
    )
    assert chunks
    for c in chunks:
        assert set(c) >= {"text", "start_char", "end_char", "word_count", "chunk_index"}
        assert c["start_char"] <= c["end_char"]


def test_db_roundtrip_offsets_populated(tmp_path):
    # §6 shape seam: the DB reads top-level keys (``_persist_chunks`` does
    # ``ch.get("start_char")`` / ``ch.get("end_char")`` —
    # DB/Client_Media_DB_v2.py); NULLs would mean the flat contract was
    # violated somewhere upstream of _persist_chunks.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = MediaDatabase(str(tmp_path / "media.db"), client_id="test")
    chunks = chunking_service.improved_chunking_process(
        TEXT, {"method": "words", "max_size": 10, "overlap": 2}
    )
    media_id, _, _ = db.add_media_with_keywords(
        title="t", media_type="document", content=TEXT, keywords=None,
        url=None, analysis_content=None, author=None, chunks=chunks,
        chunk_options={"method": "words"},
    )
    assert media_id is not None, "ingest was skipped; nothing to round-trip"
    rows = db.get_connection().execute(
        "SELECT chunk_index, start_char, end_char FROM UnvectorizedMediaChunks "
        "WHERE media_id = ? AND deleted = 0 ORDER BY chunk_index", (media_id,)
    ).fetchall()
    assert rows
    for row in rows:
        assert row["start_char"] is not None and row["end_char"] is not None, \
            "flat contract violated: DB offset columns went NULL"


def test_ebook_chapters_through_rag_service():
    # §7.2 regression: no InvalidChunkingMethodError after whitelist removal.
    # Phase B (task 7) removed chunking_service's five-method whitelist; this
    # used to be xfail'd on that whitelist and now runs for real.
    text = "# Chapter 1\n\nText one.\n\n# Chapter 2\n\nText two.\n"
    chunks = chunking_service.improved_chunking_process(
        text, {"method": "ebook_chapters", "max_size": 400, "overlap": 0}
    )
    assert chunks, "ebook_chapters must chunk, not raise"


@pytest.mark.xfail(
    strict=False,
    reason="pre-existing, NOT a chunking seam: XML_Ingestion also imports "
           "'add_media_to_database', which Client_Media_DB_v2 has never "
           "exported (broken at the branch merge-base; dead module)",
)
def test_xml_ingestion_import():
    # §7.1: the module-level chunk_xml name restored (Task 3 shim).
    #
    # The chunking part of this seam is green: chunk_xml imports cleanly and
    # is pinned directly in test_chunk_lib_shim.py
    # (test_module_level_chunk_xml_restored). Importing the whole
    # XML_Ingestion module still fails on a SECOND import that predates this
    # project (verified against the branch merge-base and the pre-Task-1
    # commit — zero occurrences of the name in Client_Media_DB_v2 at either):
    #
    #   XML_Ingestion.py:13:
    #   from tldw_chatbook.DB.Client_Media_DB_v2 import add_media_to_database
    #
    # No other module in the tree imports XML_Ingestion, which is why this
    # went unnoticed. Fixing it means touching non-test code and is out of
    # scope for this task; strict=False so the test flips to XPASS the
    # moment that import is repaired.
    import tldw_chatbook.Local_Ingestion.XML_Ingestion as mod  # noqa: F401
