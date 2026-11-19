"""Exercise the actual enhanced/base search boundary without model downloads."""

import pytest

from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
    EnhancedRAGServiceV2,
)


pytestmark = pytest.mark.bootstrap_profile


def _service():
    config = RAGConfig()
    config.embedding.model = "mock"
    config.embedding.device = "cpu"
    config.vector_store.type = "memory"
    config.vector_store.persist_directory = None
    config.search.enable_cache = False
    return EnhancedRAGServiceV2(config, enable_parallel_processing=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", ["semantic", "hybrid", "keyword"])
async def test_enhanced_search_reaches_the_real_base_method(search_type):
    service = _service()
    try:
        assert await service.search("scope regression", search_type=search_type) == []
    finally:
        service.close()


@pytest.mark.asyncio
async def test_unsupported_precomputed_embedding_is_rejected_explicitly():
    service = _service()
    try:
        with pytest.raises(ValueError, match="query_embedding"):
            await service.search("scope regression", query_embedding=[0.25, 0.75])
    finally:
        service.close()
