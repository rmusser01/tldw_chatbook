import json
from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
)


def _profile(**over):
    rag = RAGConfig(
        embedding=EmbeddingConfig(model="round-trip-model"),
        chunking=ChunkingConfig(chunk_size=333, chunk_overlap=77),
        # Pin vector_store to "memory" so persist_directory stays None. With
        # the "auto" default, VectorStoreConfig.__post_init__ resolves to
        # "chroma" (and a real PosixPath persist_directory) whenever the
        # embeddings_rag optional deps happen to be installed, which breaks
        # json.dumps below for reasons unrelated to what this test checks.
        vector_store=VectorStoreConfig(type="memory"),
    )
    return ProfileConfig(name="RT", description="d", profile_type="custom", rag_config=rag)


def test_round_trip_reconstructs_nested_dataclasses():
    p = _profile()
    restored = ProfileConfig.from_dict(json.loads(json.dumps(p.to_dict())))
    # These attribute accesses raise AttributeError today (sub-configs are dicts).
    assert isinstance(restored.rag_config, RAGConfig)
    assert isinstance(restored.rag_config.embedding, EmbeddingConfig)
    assert isinstance(restored.rag_config.chunking, ChunkingConfig)
    assert restored.rag_config.embedding.model == "round-trip-model"
    assert restored.rag_config.chunking.chunk_size == 333
    assert restored.rag_config.chunking.chunk_overlap == 77
