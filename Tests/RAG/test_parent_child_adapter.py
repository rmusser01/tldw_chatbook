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
    finally:
        monkeypatch_obj.undo()
