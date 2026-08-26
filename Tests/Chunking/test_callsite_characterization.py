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
* ``§7.3 preview/ingest agreement`` (task 10): the chunk preview modal the
  user inspects before ingesting must be produced by the same chunking code
  that stores chunks. Both modal branches are pinned against the ingest seam
  they correspond to, by driving the REAL modal (Textual ``run_test``), not
  a replica of its code — see the two ``test_preview_matches_ingest_*``
  tests for which branch maps to which ingest path and why the brief's
  original cross-method assertion (words vs structure_aware) was replaced.
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


# ---------------------------------------------------------------------------
# §7.3 preview/ingest agreement (task 10)
# ---------------------------------------------------------------------------
#
# The chunk preview modal (Widgets/chunk_preview_modal.py) has TWO branches:
#
# * basic-chunker branch (chunk_preview_modal.py:132-138): builds a
#   ``Chunk_Lib.Chunker`` with ``chunk_size``/``chunk_overlap`` from the
#   config and calls ``chunk_text(text, method=...)``. Used for
#   words/sentences/paragraphs — every method the media-details form's
#   method Select offers except the three structural ones.
# * ECS branch (chunk_preview_modal.py:109-117): calls
#   ``EnhancedChunkingService.chunk_text_with_structure`` (the delegating
#   shell → parent_child_adapter → engine structure_aware). Used for
#   hierarchical/structural/contextual.
#
# The corresponding INGEST seams are:
#
# * basic methods → ``chunking_service.improved_chunking_process`` (what the
#   ingestion paths call via the Chunk_Lib shim → engine);
# * structural methods → ``chunk_with_parent_retrieval`` (what
#   RAG_Search/simplified/enhanced_indexing_helpers.py:75 and
#   enhanced_rag_service.py:117 call to store parent/child chunks).
#
# The brief's original assertion — improved_chunking_process(method="words")
# == chunk_with_parent_retrieval(...)["chunks"] — was checked empirically
# and is NOT a §7.3 invariant: the two calls select different engine
# strategies (words vs structure_aware), which legitimately group the same
# text differently (5 x 20-word chunks vs 1 structure-aware chunk for the
# fixture below). §7.3 demands the same chunking CODE produce the preview
# and the stored chunks, i.e. agreement per branch at equal options — which
# is what these tests pin, by driving the real modal.
PREVIEW_TEXT = "# H\n\nBody text here. More text. " * 10


async def _drive_preview_modal(config):
    """Mount the real ChunkPreviewModal in a host app, return its chunks.

    The modal is a ModalScreen and needs an active app to mount; this drives
    the actual widget code (``_generate_chunks`` runs in ``on_mount``), not a
    replica of its branches.
    """
    from textual.app import App
    from tldw_chatbook.Widgets.chunk_preview_modal import ChunkPreviewModal

    class _Host(App):
        def __init__(self):
            super().__init__()
            self.modal = None

        def on_mount(self) -> None:
            self.modal = ChunkPreviewModal(
                content=PREVIEW_TEXT, config=config, media_title="test"
            )
            self.push_screen(self.modal)

    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        return list(app.modal.chunks)


@pytest.mark.parametrize("method", ["words", "sentences", "paragraphs"])
async def test_preview_matches_ingest_basic_chunker_branch(method):
    # §7.3, basic-chunker branch: the modal's Chunker() preview must produce
    # the same chunks the ingestion path (improved_chunking_process) stores
    # for the same method and options (the media-details form feeds the modal
    # ``chunk_size``/``chunk_overlap``; ingestion passes the same values as
    # ``max_size``/``overlap``).
    modal_chunks = await _drive_preview_modal(
        {"method": method, "chunk_size": 20, "chunk_overlap": 5}
    )
    ingest_chunks = chunking_service.improved_chunking_process(
        PREVIEW_TEXT, {"method": method, "max_size": 20, "overlap": 5}
    )
    assert modal_chunks, "modal produced no chunks"
    assert [c["text"] for c in modal_chunks] == [c["text"] for c in ingest_chunks]
    # §6 shape note: the modal's word-count surface stays non-zero and agrees
    # with the ingestion chunks' word counts (the modal's basic branch
    # computes word_count itself via len(chunk.split())).
    assert [c["word_count"] for c in modal_chunks] == [
        c["word_count"] for c in ingest_chunks
    ]
    for c in modal_chunks:
        assert c["word_count"] == len(c["text"].split()) > 0


@pytest.mark.parametrize("method", ["hierarchical", "structural", "contextual"])
async def test_preview_matches_ingest_structural_branch(method):
    # §7.3, ECS branch: the modal's structural preview (chunk_text_with_structure
    # → parent_child_adapter → engine structure_aware) must produce the same
    # chunk texts the structural ingestion path (chunk_with_parent_retrieval)
    # stores. The three method names are legacy aliases — the adapter maps all
    # of them to the engine's structure_aware strategy — so one comparison per
    # alias keeps every Select entry in the form pinned.
    from tldw_chatbook.RAG_Search.parent_child_adapter import (
        chunk_with_parent_retrieval,
    )

    modal_chunks = await _drive_preview_modal(
        {"method": method, "chunk_size": 20, "chunk_overlap": 5}
    )
    ingest_result = chunk_with_parent_retrieval(PREVIEW_TEXT, max_size=20, overlap=5)
    assert modal_chunks, "modal produced no chunks"
    assert [c["text"] for c in modal_chunks] == [
        c["text"] for c in ingest_result["chunks"]
    ]
    # §6 shape note: the ECS branch surfaces StructuredChunk.word_count
    # (len(text.split())); it must stay non-zero.
    for c in modal_chunks:
        assert c["word_count"] == len(c["text"].split()) > 0
