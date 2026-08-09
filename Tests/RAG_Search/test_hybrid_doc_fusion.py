"""Hybrid RRF fusion must match the two legs on DOCUMENT identity (TASK-3994).

The FTS leg emits document-level rows (`media_15`) and the vector leg emits
chunk-level rows (`media_15_chunk_0`), so matching on `SearchResult.id` could
never fuse the same document across legs: on the P1 corpus hybrid returned
byte-identical results to pure semantic for all 44 golden queries.

Every row here carries the REAL metadata the two producers stamp -- the
keyword leg's `doc_id`/`source_type` (`RAGService._process_keyword_results_basic`)
and the vector leg's `source_id`/`source_type` spread from
`ingestion_indexing.media_document` into every chunk by
`RAGService.index_document`. The shapes are the point of these tests: fusion
now depends on that metadata, and hand-built rows with empty metadata (the
sibling `test_hybrid_fusion_metadata.py` oracle) exercise only the fallback.
"""
import pytest

from tldw_chatbook.RAG_Search.simplified.citations import (
    Citation,
    CitationType,
    SearchResultWithCitations,
)
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

K = 60


def _keyword_row(media_id: int, title: str, content: str, score: float = 0.8):
    """A keyword-leg row exactly as `_process_keyword_results_basic` builds it."""
    return SearchResult(
        id=f"media_{media_id}",
        score=score,
        document=content[:1000],
        metadata={
            "doc_id": str(media_id),
            "doc_title": title,
            "title": title,
            "media_type": "article",
            "url": None,
            "author": None,
            "ingestion_date": None,
            "text_preview": content[:200],
            "source_type": "media",
            "source": "media",
        },
    )


def _vector_row(media_id: int, chunk_index: int, title: str, text: str, score: float):
    """A vector-leg row exactly as an indexed media chunk comes back.

    `index_document` spreads the document metadata (`source_id`, `title`,
    `source_type` from `media_document`) into every chunk and adds the
    chunk keys on top -- including a `doc_id` that is the PREFIXED document
    id, unlike the keyword leg's bare row id.
    """
    return SearchResult(
        id=f"media_{media_id}_chunk_{chunk_index}",
        score=score,
        document=text,
        metadata={
            "source_id": str(media_id),
            "title": title,
            "source_type": "media",
            "media_type": "article",
            "doc_id": f"media_{media_id}",
            "doc_title": title,
            "chunk_id": f"media_{media_id}_chunk_{chunk_index}",
            "chunk_index": chunk_index,
            "chunk_start": chunk_index * 480,
            "chunk_end": (chunk_index + 1) * 480,
            "chunk_size": 480,
            "word_count": 80,
            "text_preview": text[:200],
        },
    )


def test_same_document_across_legs_merges():
    """One document found by both legs is ONE fused row carrying both legs."""
    keyword = [_keyword_row(15, "Beekeeping Basics", "hive frames and supers")]
    semantic = [
        _vector_row(15, 0, "Beekeeping Basics", "hive frames and supers", 0.83)
    ]
    # The naive alternatives both fail on these real rows: the ids live in
    # different spaces, and so does `doc_id` (bare vs prefixed). Only
    # (source_type, source_id-or-doc_id) with source_id winning matches.
    assert keyword[0].id != semantic[0].id
    assert keyword[0].metadata["doc_id"] != semantic[0].metadata["doc_id"]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword,
        semantic_results=semantic,
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )

    assert len(fused) == 1
    fusion = fused[0].metadata["hybrid_fusion"]
    assert fusion["fts_rank"] == 1
    assert fusion["vector_rank"] == 1
    assert fusion["fts_score"] == pytest.approx(0.8)
    assert fusion["vector_score"] == pytest.approx(0.83)
    # Both legs contribute to the fused score (not the vector leg alone).
    assert fused[0].score == pytest.approx(0.3 / (K + 1) + 0.7 / (K + 1))


def test_merged_row_displays_the_vector_item():
    """A merged row shows the matched CHUNK, not the whole-document FTS row."""
    fused = RAGService._fuse_hybrid_results(
        keyword_results=[
            _keyword_row(15, "Beekeeping Basics", "whole document text here")
        ],
        semantic_results=[
            _vector_row(15, 2, "Beekeeping Basics", "the matched chunk text", 0.83)
        ],
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )

    assert len(fused) == 1
    row = fused[0]
    assert row.id == "media_15_chunk_2"
    assert row.document == "the matched chunk text"
    # The chunk row's identity metadata survives the merge -- this is what
    # downstream row mappers read (`_semantic_row`: source_id, chunk_id).
    assert row.metadata["source_id"] == "15"
    assert row.metadata["chunk_id"] == "media_15_chunk_2"


def test_vector_leg_chunks_of_one_doc_collapse_to_best_rank():
    """Several chunks of one document occupy ONE fused slot, at its best rank."""
    semantic = [
        _vector_row(7, 0, "Doc Seven", "chunk zero", 0.91),
        _vector_row(9, 0, "Doc Nine", "unrelated", 0.88),
        _vector_row(7, 3, "Doc Seven", "chunk three", 0.85),
    ]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=[],
        semantic_results=semantic,
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )

    assert len(fused) == 2
    by_source = {r.metadata["source_id"]: r for r in fused}
    assert set(by_source) == {"7", "9"}
    assert by_source["7"].metadata["hybrid_fusion"]["vector_rank"] == 1
    assert by_source["9"].metadata["hybrid_fusion"]["vector_rank"] == 2
    # The best-ranked chunk is the one displayed for that document.
    assert by_source["7"].id == "media_7_chunk_0"


def test_rows_without_metadata_keep_todays_no_merge_behavior():
    """Rows with no ingestion metadata fall back to the row id (no merge)."""
    keyword = [SearchResult(id="a1", score=0.5, document="a", metadata={})]
    semantic = [SearchResult(id="b2", score=0.9, document="b", metadata={})]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword,
        semantic_results=semantic,
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )

    assert [r.id for r in fused] == ["b2", "a1"]
    for row in fused:
        fusion = row.metadata["hybrid_fusion"]
        # Each row came from exactly one leg -- nothing merged.
        assert (fusion["fts_rank"] is None) != (fusion["vector_rank"] is None)


def test_merged_citations_combine_without_duplication_or_crash():
    """The citation-merge branch's first real run: both legs' citations, deduped."""
    shared = Citation(
        document_id="15",
        document_title="Beekeeping Basics",
        chunk_id="media_15_shared",
        text="shared snippet",
        start_char=10,
        end_char=24,
        confidence=0.9,
        match_type=CitationType.EXACT,
    )
    keyword_row = _keyword_row(15, "Beekeeping Basics", "hive frames and supers")
    vector_row = _vector_row(15, 0, "Beekeeping Basics", "hive frames", 0.83)
    keyword = [
        SearchResultWithCitations(
            id=keyword_row.id,
            score=keyword_row.score,
            document=keyword_row.document,
            metadata=keyword_row.metadata,
            citations=[
                # Document-level citation: bare document id, offsets into the
                # WHOLE media item (`_create_keyword_result_with_citations`).
                Citation(
                    document_id="15",
                    document_title="Beekeeping Basics",
                    chunk_id="media_15_kw_120",
                    text="...frames...",
                    start_char=120,
                    end_char=126,
                    confidence=1.0,
                    match_type=CitationType.EXACT,
                ),
                shared,
            ],
        )
    ]
    semantic = [
        SearchResultWithCitations(
            id=vector_row.id,
            score=vector_row.score,
            document=vector_row.document,
            metadata=vector_row.metadata,
            citations=[
                # Chunk-level citation: prefixed doc id + chunk offsets
                # (`ChromaVectorStore._create_citations_from_result`).
                Citation(
                    document_id="media_15",
                    document_title="Beekeeping Basics",
                    chunk_id="media_15_chunk_0",
                    text="hive frames",
                    start_char=0,
                    end_char=480,
                    confidence=0.83,
                    match_type=CitationType.SEMANTIC,
                ),
                shared,
            ],
        )
    ]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword,
        semantic_results=semantic,
        top_k=10,
        alpha=0.7,
        include_citations=True,
    )

    assert len(fused) == 1
    chunk_ids = sorted(c.chunk_id for c in fused[0].citations)
    assert chunk_ids == ["media_15_chunk_0", "media_15_kw_120", "media_15_shared"]


def test_merged_citations_survive_a_leg_without_citations():
    """A citation-less leg must not crash the merge (mixed leg row types)."""
    keyword = [_keyword_row(15, "Beekeeping Basics", "hive frames")]
    semantic = [
        SearchResultWithCitations(
            id="media_15_chunk_0",
            score=0.83,
            document="hive frames",
            metadata=_vector_row(15, 0, "Beekeeping Basics", "hive frames", 0.83).metadata,
            citations=[
                Citation(
                    document_id="media_15",
                    document_title="Beekeeping Basics",
                    chunk_id="media_15_chunk_0",
                    text="hive frames",
                    start_char=0,
                    end_char=480,
                    confidence=0.83,
                    match_type=CitationType.SEMANTIC,
                )
            ],
        )
    ]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword,
        semantic_results=semantic,
        top_k=10,
        alpha=0.7,
        include_citations=True,
    )

    assert len(fused) == 1
    assert [c.chunk_id for c in fused[0].citations] == ["media_15_chunk_0"]


def test_fts_only_docs_enter_when_the_vector_leg_has_fewer_than_top_k_documents():
    """Document-level keys stop one document's chunks from eating every slot.

    What this pins: five chunks of two documents used to be five fusion keys
    and consumed all three top-k slots; collapsed to two document keys they
    consume two, and the free third slot goes to the FTS-only row. That is the
    id-space/dedup fix (TASK-3994) observed through its effect on slot
    occupancy -- nothing more.

    **This is NOT the starvation fix, and must not be cited as evidence for
    one.** The keyword-only row gets in here only because the vector leg
    supplies FEWER DISTINCT DOCUMENTS than top_k (2 < 3), leaving a slot no
    vector row wants. The real defect is that whenever the vector leg returns
    k or more distinct documents -- the normal case -- an FTS-only row scores
    (1-alpha)/(rrf_k+1) = 0.00492 under the shipped defaults and is beaten by
    every vector row ranked better than about 82, so it is structurally
    unreachable. That is TASK-4110, whose description carries the corrected
    "k or more distinct documents" mechanism and whose AC #1 explicitly
    disqualifies this thin-vector-leg case as evidence.
    """
    semantic = [
        _vector_row(7, 0, "Doc Seven", "seven a", 0.91),
        _vector_row(9, 0, "Doc Nine", "nine a", 0.90),
        _vector_row(7, 1, "Doc Seven", "seven b", 0.89),
        _vector_row(9, 1, "Doc Nine", "nine b", 0.88),
        _vector_row(7, 2, "Doc Seven", "seven c", 0.87),
    ]
    keyword = [_keyword_row(42, "Keyword Only", "the exact phrase")]

    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword,
        semantic_results=semantic,
        top_k=3,
        alpha=0.7,
        include_citations=False,
    )

    # The keyword leg has no `source_id` -- its identity is `doc_id`.
    identities = [
        r.metadata.get("source_id") or r.metadata.get("doc_id") for r in fused
    ]
    assert identities == ["7", "9", "42"]
    assert fused[-1].id == "media_42"
    assert fused[-1].metadata["hybrid_fusion"]["vector_rank"] is None
    assert fused[-1].metadata["hybrid_fusion"]["fts_rank"] == 1
