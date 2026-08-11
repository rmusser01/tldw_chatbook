# Tests/RAG_Eval/test_canonicalize.py
"""Always-on tests for doc-level canonicalization of seam rows.

Pure: hand-built row dicts in the *exact* shapes
`LibraryLocalRagSearchService` emits (see `canonicalize.py`'s docstring for
where each shape is built), so this module carries no env gate, imports no
app module, and needs no model.

These rows are not invented. Each one is transcribed from the row builder
that produces it in `tldw_chatbook/Library/library_local_rag_search_service.py`;
the shapes that matter are the three keyword builders, the semantic/vector
row, and the hybrid FTS-leg row whose `source_id` is a *document* id rather
than a source id.
"""
from __future__ import annotations

import pytest

from Tests.RAG_Eval.harness.canonicalize import (
    UNKNOWN_PREFIX,
    canonical_source_type,
    rows_to_doc_ids,
    slug_lookup_from,
)

#: A runtime map in `EvalRuntime.slug_to_source`'s shape (slug -> (type, id)).
SLUG_TO_SOURCE = {
    "note-zephyr": ("note", "1"),
    "media-calyx": ("media", "7"),
    "conv-nimbus": ("conversation", "3"),
}
LOOKUP = slug_lookup_from(SLUG_TO_SOURCE)


def _keyword_note_row(source_id: str) -> dict:
    """`_note_row`: singular provenance, bare source id, no chunk."""
    return {
        "source_id": source_id,
        "chunk_id": "",
        "title": "Zephyr",
        "snippet": "...",
        "score": None,
        "provenance": {"source_type": "note"},
    }


def _keyword_media_row(source_id: str) -> dict:
    return {
        "source_id": source_id,
        "chunk_id": "",
        "title": "Calyx",
        "snippet": "Matched media · document",
        "score": None,
        "provenance": {"source_type": "media"},
    }


def _keyword_conversation_row(source_id: str) -> dict:
    return {
        "source_id": source_id,
        "chunk_id": "",
        "title": "Nimbus",
        "snippet": "Matched conversation · 1 message",
        "score": None,
        "provenance": {"source_type": "conversation"},
    }


def _semantic_row(source_id: str, source_type: str, chunk_id: str) -> dict:
    """`_semantic_row` over an indexed chunk: doc metadata + chunk_id."""
    return {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "title": "Calyx",
        "snippet": "chunk text",
        "score": 0.42,
        "provenance": {
            "source_type": source_type,
            "doc_id": f"{source_type}_{source_id}",
            "chunk_id": chunk_id,
            "chunk_index": 0,
        },
    }


def _hybrid_fts_leg_row(media_id: str) -> dict:
    """The hybrid FTS leg's row: `source_id` is the DOCUMENT id.

    The engine's keyword leg builds its metadata from scratch with `doc_id`
    (a key `_semantic_row` never reads) and no `source_id`, so
    `_semantic_row` falls all the way through to `SearchResult.id`, which is
    `f"media_{id}"`. A canonicalizer that only tried the bare id would map
    every hybrid keyword hit to nothing.
    """
    return {
        "source_id": f"media_{media_id}",
        "chunk_id": "",
        "title": "Calyx",
        "snippet": "matched text",
        "score": 0.016,
        "provenance": {
            "source_type": "media",
            "doc_id": media_id,
            "source": "media",
            "hybrid_fusion": {"fts_rank": 1, "vector_score": None},
        },
    }


def test_each_source_type_maps_to_its_slug():
    rows = [
        _keyword_note_row("1"),
        _keyword_media_row("7"),
        _keyword_conversation_row("3"),
    ]
    assert rows_to_doc_ids(rows, LOOKUP) == [
        "note-zephyr",
        "media-calyx",
        "conv-nimbus",
    ]


def test_semantic_chunk_rows_map_to_the_document_slug():
    rows = [_semantic_row("7", "media", "media_7_chunk_2")]
    assert rows_to_doc_ids(rows, LOOKUP) == ["media-calyx"]


def test_repeated_chunks_of_one_document_dedup_keeping_first_hit_rank():
    """Doc-level metrics: three chunks of one doc are one retrieved doc.

    And the surviving entry keeps the *first* chunk's rank — a later chunk
    must not be able to push its document down the list.
    """
    rows = [
        _semantic_row("7", "media", "media_7_chunk_0"),
        _keyword_note_row("1"),
        _semantic_row("7", "media", "media_7_chunk_1"),
        _semantic_row("7", "media", "media_7_chunk_5"),
    ]
    assert rows_to_doc_ids(rows, LOOKUP) == ["media-calyx", "note-zephyr"]


def test_hybrid_fts_leg_document_id_form_resolves_to_the_slug():
    assert rows_to_doc_ids([_hybrid_fts_leg_row("7")], LOOKUP) == ["media-calyx"]


def test_hybrid_and_vector_rows_for_one_document_dedup_together():
    """The same doc reached by both legs is still one retrieved document."""
    rows = [
        _hybrid_fts_leg_row("7"),
        _semantic_row("7", "media", "media_7_chunk_0"),
    ]
    assert rows_to_doc_ids(rows, LOOKUP) == ["media-calyx"]


def test_unmapped_row_becomes_a_synthetic_id_that_still_costs_precision():
    """Junk retrieval is kept, not dropped.

    Dropping a row the golden set does not know would make precision measure
    "of the documents I recognized, how many were right" — a number that
    improves when retrieval returns more garbage.
    """
    rows = [_keyword_media_row("999"), _keyword_note_row("1")]
    doc_ids = rows_to_doc_ids(rows, LOOKUP)
    assert doc_ids == [f"{UNKNOWN_PREFIX}media:999", "note-zephyr"]
    assert doc_ids[0] not in SLUG_TO_SOURCE


def test_unmapped_document_id_form_keeps_the_raw_id_in_the_synthetic_id():
    """Prefix-stripping is a lookup fallback, never a rewrite of the id."""
    assert rows_to_doc_ids([_hybrid_fts_leg_row("999")], LOOKUP) == [
        f"{UNKNOWN_PREFIX}media:media_999"
    ]


def test_two_distinct_unknown_rows_stay_distinct_and_one_repeat_dedups():
    rows = [
        _keyword_media_row("998"),
        _keyword_media_row("999"),
        _keyword_media_row("998"),
    ]
    assert rows_to_doc_ids(rows, LOOKUP) == [
        f"{UNKNOWN_PREFIX}media:998",
        f"{UNKNOWN_PREFIX}media:999",
    ]


def test_unclaimed_prompt_rows_are_unknown_rather_than_silently_dropped():
    """A prompt row no fixture claims still occupies its top-k slot.

    `LOOKUP` deliberately maps no prompt, which since TASK-15020/B2 is the
    unusual case rather than the only one — see the sibling test below.
    """
    row = {
        "source_id": "4",
        "chunk_id": "",
        "title": "A prompt",
        "snippet": "...",
        "score": None,
        "provenance": {"source_type": "prompt"},
    }
    assert rows_to_doc_ids([row], LOOKUP) == [f"{UNKNOWN_PREFIX}prompt:4"]


def test_a_claimed_prompt_row_resolves_to_its_fixture_slug():
    """TASK-15020/B2: prompt rows are SCORED now, not counted as noise.

    The engine's prompts sub-leg stamps the singular `prompt` and the bare
    prompt id, which is exactly the shape `slug_to_source` records — so a
    prompt hit resolves without the document-id-prefix fallback the media
    rows need. If this ever regressed to `unknown:`, the prompt category
    would read 0.000 with a working sub-leg underneath it.
    """
    lookup = slug_lookup_from({**SLUG_TO_SOURCE, "prompt-shift": ("prompt", "4")})
    row = {
        "source_id": "4",
        "chunk_id": "",
        "title": "A prompt",
        "snippet": "...",
        "score": None,
        "provenance": {"source_type": "prompt"},
    }
    assert rows_to_doc_ids([row], lookup) == ["prompt-shift"]


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("note", "note"),
        ("notes", "note"),
        ("media", "media"),
        ("media_chunk", "media"),
        ("conversation", "conversation"),
        ("conversations", "conversation"),
        ("chat", "conversation"),
        ("prompt", "prompt"),
        ("prompts", "prompt"),
        ("  Note  ", "note"),
        ("", ""),
        (None, ""),
        ("workspace", "workspace"),
    ],
)
def test_source_type_canonicalization(raw, canonical):
    assert canonical_source_type(raw) == canonical


def test_plural_provenance_vocabulary_still_maps():
    """The semantic post-filter's vocabulary, not just the writers'.

    `_SEMANTIC_SOURCE_TYPE_MAP` in the seam accepts plural/aliased forms, so
    an index whose metadata says "notes" or "chat" must not read as unknown.
    """
    rows = [
        dict(_keyword_note_row("1"), provenance={"source_type": "notes"}),
        dict(_keyword_conversation_row("3"), provenance={"source_type": "chat"}),
    ]
    assert rows_to_doc_ids(rows, LOOKUP) == ["note-zephyr", "conv-nimbus"]


def test_row_without_provenance_is_unknown_with_an_empty_type():
    row = {"source_id": "1", "chunk_id": "", "title": "", "snippet": "", "score": None}
    assert rows_to_doc_ids([row], LOOKUP) == [f"{UNKNOWN_PREFIX}:1"]


def test_row_with_no_source_id_is_unknown_and_never_matches_a_slug():
    row = {"source_id": "", "provenance": {"source_type": "note"}}
    assert rows_to_doc_ids([row], LOOKUP) == [f"{UNKNOWN_PREFIX}note:"]


def test_falsy_int_source_id_zero_resolves_via_the_slug_not_unknown():
    """Qodo PR #1458 finding 1: `row.get("source_id") or ""` rewrote a falsy
    but valid id (the int `0`) to `""` before it ever reached `str()`, so a
    row whose real source id is `0` could never match a lookup key and was
    scored as unknown instead. Only an actually-missing id (`None`) should
    fall back to `""`.
    """
    lookup = slug_lookup_from({"media-zero": ("media", 0)})
    row = {
        "source_id": 0,
        "chunk_id": "",
        "title": "Zero",
        "snippet": "...",
        "score": None,
        "provenance": {"source_type": "media"},
    }
    assert rows_to_doc_ids([row], lookup) == ["media-zero"]


def test_no_rows_is_no_documents():
    assert rows_to_doc_ids([], LOOKUP) == []


def test_a_non_mapping_row_raises_rather_than_being_skipped():
    """A shape change at the seam must fail loudly, not lose a result."""
    # Position is 1-indexed and names the offending row, not the first one.
    with pytest.raises(TypeError, match="row 2 is a tuple"):
        rows_to_doc_ids([_keyword_note_row("1"), ("note", "1")], LOOKUP)


def test_slug_lookup_from_inverts_the_runtime_map():
    assert slug_lookup_from(SLUG_TO_SOURCE) == {
        ("note", "1"): "note-zephyr",
        ("media", "7"): "media-calyx",
        ("conversation", "3"): "conv-nimbus",
    }


def test_slug_lookup_from_rejects_two_slugs_for_one_source():
    """Two fixture slugs on one DB row would make retrieval unscoreable."""
    with pytest.raises(ValueError, match="note-a"):
        slug_lookup_from({"note-a": ("note", "1"), "note-b": ("note", "1")})
