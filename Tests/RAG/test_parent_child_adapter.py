# Tests/RAG/test_parent_child_adapter.py
"""Q5 ruling: ECS retired; adapter preserves the parent/child retrieval shape."""
import pytest


TEXT = "# Section A\n\nPara one under A.\n\n## Sub A1\n\nDeep text.\n\n# Section B\n\nPara under B.\n"


def test_parent_child_shape():
    from tldw_chatbook.RAG_Search import parent_child_adapter as pca
    result = pca.chunk_with_parent_retrieval(TEXT, max_size=100, overlap=0)
    assert "chunks" in result and "parent_chunks" in result
    for parent in result["parent_chunks"]:
        assert "text" in parent
        assert isinstance(parent.get("children"), list)
    # every child references exactly one parent
    for chunk in result["chunks"]:
        assert chunk.get("parent_id") is not None


def test_structureaware_engine_underneath():
    # The adapter must call the engine's hierarchical path, not ECS logic.
    from tldw_chatbook.RAG_Search import parent_child_adapter as pca
    from tldw_chatbook.Chunking.engine import Chunker
    calls = []
    real = Chunker.chunk_text_hierarchical_flat
    def spy(self, text, **kwargs):
        calls.append(kwargs)
        return real(self, text, **kwargs)
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(Chunker, "chunk_text_hierarchical_flat", spy)
    try:
        pca.chunk_with_parent_retrieval(TEXT, max_size=100, overlap=0)
        assert calls, "adapter must delegate to the engine's hierarchical path"
        # The engine's structure_aware strategy is the only structure-aware
        # implementation (Q5 ruling) — any other method must fail this test.
        assert calls[0]['method'] == 'structure_aware'
    finally:
        monkeypatch_obj.undo()


# ---------------------------------------------------------------------------
# ECS delegation seam (review I-1): the thin EnhancedChunkingService shell
# must forward to the adapter correctly. Runs everywhere — no embeddings deps.
# ---------------------------------------------------------------------------

ECS_TEXT = (
    "# Section A\n\nPara one under A.\n\n## Sub A1\n\nDeep text.\n\n"
    "# Section B\n\nPara under B.\n"
)

BIG_TEXT = "# Big Section\n\n" + "\n\n".join(
    f"Paragraph {i} with content words." for i in range(10)
)


def _ecs():
    from tldw_chatbook.RAG_Search.enhanced_chunking_service import (
        EnhancedChunkingService,
    )

    return EnhancedChunkingService()


def test_ecs_delegation_legacy_shape():
    service = _ecs()
    result = service.chunk_with_parent_retrieval(
        ECS_TEXT, chunk_size=100, chunk_overlap=0, parent_size_multiplier=3
    )
    # Legacy top-level keys (exactly these three).
    assert set(result.keys()) == {"chunks", "parent_chunks", "metadata"}
    # Legacy metadata keys with legacy arithmetic.
    assert set(result["metadata"].keys()) == {
        "chunk_size",
        "parent_chunk_size",
        "total_chunks",
        "total_parent_chunks",
    }
    meta = result["metadata"]
    assert meta["chunk_size"] == 100
    assert meta["parent_chunk_size"] == 100 * 3
    assert meta["total_chunks"] == len(result["chunks"])
    assert meta["total_parent_chunks"] == len(result["parent_chunks"])
    assert result["chunks"] and result["parent_chunks"]
    # Legacy per-chunk keys the two RAG indexing consumers read.
    for chunk in result["chunks"]:
        assert {"text", "start_char", "end_char", "chunk_index", "chunk_type",
                "level", "parent_index", "children_indices", "word_count",
                "char_count", "metadata"} <= set(chunk.keys())
        assert chunk["metadata"]["parent_chunk_index"] is not None


def test_ecs_delegation_forwards_kwargs():
    # chunk_size must reach the engine as max_size (kwarg aliasing seam).
    from tldw_chatbook.Chunking.engine import Chunker

    calls = []
    real = Chunker.chunk_text_hierarchical_flat

    def spy(self, text, **kwargs):
        calls.append(kwargs)
        return real(self, text, **kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(Chunker, "chunk_text_hierarchical_flat", spy)
    try:
        service = _ecs()
        service.chunk_with_parent_retrieval(
            ECS_TEXT, chunk_size=7, chunk_overlap=3, parent_size_multiplier=2
        )
        assert calls, "ECS must forward to the engine's hierarchical path"
        assert calls[0]["max_size"] == 7
        assert calls[0]["overlap"] == 3
    finally:
        mp.undo()

    # parent_size_multiplier must have an observable effect on parent count:
    # with 10 single-paragraph elements and chunk_size=2, a multiplier of 1
    # yields one parent per element group, a multiplier of 8 one parent.
    service = _ecs()
    tight = service.chunk_with_parent_retrieval(
        BIG_TEXT, chunk_size=2, chunk_overlap=0, parent_size_multiplier=1
    )
    wide = service.chunk_with_parent_retrieval(
        BIG_TEXT, chunk_size=2, chunk_overlap=0, parent_size_multiplier=8
    )
    assert tight["metadata"]["total_parent_chunks"] > 1
    assert wide["metadata"]["total_parent_chunks"] == 1


def test_ecs_delegation_factory():
    from tldw_chatbook.RAG_Search.enhanced_chunking_service import (
        EnhancedChunkingService,
        create_enhanced_chunking_service,
    )

    service = create_enhanced_chunking_service()
    assert isinstance(service, EnhancedChunkingService)
    result = service.chunk_with_parent_retrieval(
        ECS_TEXT, chunk_size=100, chunk_overlap=0
    )
    assert set(result.keys()) == {"chunks", "parent_chunks", "metadata"}
    assert result["metadata"]["total_chunks"] == len(result["chunks"])


def test_ecs_delegation_chunk_text_with_structure():
    # The preview modal path: legacy StructuredChunk attribute reads.
    service = _ecs()
    chunks = service.chunk_text_with_structure(
        ECS_TEXT, chunk_size=100, chunk_overlap=0, method="hierarchical"
    )
    assert chunks
    for chunk in chunks:
        for attr in (
            "text",
            "chunk_index",
            "word_count",
            "char_count",
            "metadata",
        ):
            assert hasattr(chunk, attr)
        assert chunk.chunk_type.value  # modal reads chunk.chunk_type.value
