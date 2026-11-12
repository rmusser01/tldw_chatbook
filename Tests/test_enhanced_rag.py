"""
Test script for the enhanced RAG implementation.

This script demonstrates the new features:
1. Enhanced chunking with character-level position tracking
2. Hierarchical chunking with structure preservation
3. Parent document retrieval

Note (chunking-engine-parity task 8, Q5 ruling): the bespoke PDF-artifact
cleaning (DocumentStructureParser.clean_text) was DELETED with the retired
home-grown structure parser — the feature was NOT ported anywhere. The
vendored engine performs security-oriented input sanitization only, which
is not a replacement for it.

Note (chunking-template-parity PR 0): the bespoke table serialization
(table_serializer.py) and its ``serialize_tables`` kwarg were deleted as
dead code; tables are chunked as structural blocks by the vendored engine.
"""

import logging
import sys
from contextlib import contextmanager

import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_document() -> str:
    """Create a test document with various structures."""
    return """
# Annual Report 2024

## Executive Summary

This document presents the annual performance metrics and financial results for fiscal year 2024. 
The company has achieved significant growth across all key performance indicators.

### Key Highlights

- Revenue increased by 25% year-over-year
- Customer base expanded to over 10,000 active users
- Launched 3 new product lines successfully
- Improved operational efficiency by 15%

## Financial Performance

The following table summarizes our quarterly revenue:

| Quarter | Revenue ($M) | Growth (%) | New Customers |
|---------|-------------|------------|---------------|
| Q1 2024 | 12.5        | 20%        | 1,200         |
| Q2 2024 | 15.2        | 22%        | 1,500         |
| Q3 2024 | 18.7        | 25%        | 2,100         |
| Q4 2024 | 23.1        | 30%        | 2,800         |

### Regional Breakdown

Our expansion into new markets has been particularly successful:

1. North America: 45% of total revenue
2. Europe: 30% of total revenue
3. Asia-Pacific: 20% of total revenue
4. Other regions: 5% of total revenue

## Product Performance

Each product line showed strong performance metrics:

| Product Line | Units Sold | Revenue ($M) | Market Share (%) |
|--------------|------------|--------------|------------------|
| Product A    | 50,000     | 25.5         | 35%              |
| Product B    | 35,000     | 20.2         | 28%              |
| Product C    | 28,000     | 15.8         | 22%              |
| New Line D   | 12,000     | 8.0          | 15%              |

### Customer Satisfaction

Customer feedback has been overwhelmingly positive/five stars, with an average satisfaction 
score of 4.7/5.0 across all products. Key feedback points include:

- Excellent product quality
- Responsive customer support
- Competitive pricing
- Fast delivery times

## Future Outlook

Looking ahead to 2025, we anticipate continued growth with several strategic initiatives:

1. **Market Expansion**: Enter 5 new geographic markets
2. **Product Innovation**: Launch 2 new product lines
3. **Technology Upgrade**: Implement new CRM system
4. **Sustainability**: Achieve carbon neutrality by Q4 2025

### Investment Plans

We plan to invest $15M in the following areas:

- R&D: $6M (40%)
- Marketing: $4.5M (30%)
- Infrastructure: $3M (20%)
- Training: $1.5M (10%)

## Conclusion

The fiscal year 2024 has been transformative for our company. With strong financial performance,
expanding market presence, and positive customer feedback, we are well-positioned for continued
success in 2025 and beyond.

---

*This report was prepared by the Finance Department in collaboration with all business units.*
"""


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_enhanced_chunking():
    """Test enhanced chunking features."""
    from tldw_chatbook.RAG_Search.enhanced_chunking_service import (
        EnhancedChunkingService,
    )

    logger.info("=== Testing Enhanced Chunking Service ===")

    service = EnhancedChunkingService()
    content = create_test_document()

    # Test 1: Hierarchical chunking with structure preservation
    logger.info("\n1. Testing hierarchical chunking...")
    chunks = service.chunk_text_with_structure(
        content,
        chunk_size=200,  # Smaller chunks to see more structure
        chunk_overlap=50,
        method="hierarchical",
        preserve_structure=True,
        clean_artifacts=True,
    )

    assert chunks
    logger.info(f"Created {len(chunks)} hierarchical chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        logger.info(f"\nChunk {i}:")
        logger.info(f"  Type: {chunk.chunk_type.value}")
        logger.info(f"  Level: {chunk.level}")
        logger.info(f"  Position: {chunk.start_char}-{chunk.end_char}")
        logger.info(f"  Parent: {chunk.parent_index}")
        logger.info(f"  Children: {chunk.children_indices}")
        logger.info(f"  Text preview: {chunk.text[:100]}...")

    # Test 2: Parent document retrieval
    logger.info("\n2. Testing parent document retrieval...")
    parent_result = service.chunk_with_parent_retrieval(
        content, chunk_size=150, chunk_overlap=30, parent_size_multiplier=3
    )

    assert parent_result["chunks"] and parent_result["parent_chunks"]
    assert all(child["parent_id"] is not None for child in parent_result["chunks"])
    logger.info(f"Created {parent_result['metadata']['total_chunks']} retrieval chunks")
    logger.info(
        f"Created {parent_result['metadata']['total_parent_chunks']} parent chunks"
    )

    # Show mapping
    for i, chunk in enumerate(parent_result["chunks"][:3]):
        parent_idx = chunk.get("metadata", {}).get("parent_chunk_index")
        if parent_idx is not None:
            logger.info(f"\nRetrieval chunk {i} -> Parent chunk {parent_idx}")
            logger.info(f"  Retrieval text: {chunk['text'][:80]}...")
            parent_text = parent_result["parent_chunks"][parent_idx]["text"]
            logger.info(f"  Parent text: {parent_text[:150]}...")


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_enhanced_rag_service(tmp_path, monkeypatch):
    """Index child chunks and expand a semantic result to its parent text."""
    from tldw_chatbook.RAG_Search import config_profiles
    from tldw_chatbook.RAG_Search.simplified import create_rag_service
    from tldw_chatbook.RAG_Search.simplified.config import (
        EmbeddingConfig,
        RAGConfig,
        VectorStoreConfig,
    )

    manager = config_profiles.get_profile_manager(profiles_dir=tmp_path / "profiles")
    monkeypatch.setattr(config_profiles, "_GLOBAL_PROFILE_MANAGER", manager)
    config = RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        vector_store=VectorStoreConfig(type="memory"),
    )
    # Bind every possible keyword path to this test even though its search is semantic.
    config.search.media_db_path = tmp_path / "absent-media.db"
    config.search.chachanotes_db_path = tmp_path / "absent-chacha.db"
    config.search.prompts_db_path = tmp_path / "absent-prompts.db"

    service = create_rag_service(profile_name="hybrid_enhanced", config=config)
    try:
        result = await service.index_document_with_parents(
            doc_id="test_doc_001",
            content=create_test_document(),
            title="Annual Report 2024",
            metadata={"type": "annual_report", "year": 2024, "department": "finance"},
            use_structural_chunking=True,
        )
        assert result.success and result.chunks_created > 0
        assert result.metadata["parent_chunks_created"] > 0
        assert set(result.to_dict()) == {
            "doc_id",
            "chunks_created",
            "time_taken",
            "success",
            "error",
        }

        for query in (
            "quarterly revenue growth",
            "customer satisfaction score",
            "investment plans for R&D",
        ):
            basic = await service.search(query=query, top_k=2, search_type="semantic")
            expanded = await service.search_with_context_expansion(
                query=query, top_k=2, search_type="semantic"
            )
            assert basic, query
            assert [item.id for item in expanded] == [item.id for item in basic]
            assert any(
                item.metadata.get("context_expanded")
                and item.document == item.metadata["parent_chunk_text"]
                for item in expanded
            ), query
    finally:
        # The mock embeddings wrapper still checks CUDA on normal close.
        # Its no-Torch branch leaves the service's real cleanup in place.
        with monkeypatch.context() as cleanup:
            cleanup.setitem(sys.modules, "torch", None)
            service.close()


@pytest.fixture
def offline_profile_service(tmp_path, monkeypatch):
    from Tests.RAG_Search.test_reranker_degraded_paths import _install_fake_provider
    from tldw_chatbook.RAG_Search import config_profiles
    from tldw_chatbook.RAG_Search.simplified import RAGConfig, create_rag_service

    manager = config_profiles.get_profile_manager(profiles_dir=tmp_path / "profiles")
    monkeypatch.setattr(config_profiles, "_GLOBAL_PROFILE_MANAGER", manager)
    provider_calls = _install_fake_provider(monkeypatch)

    @contextmanager
    def open_service(profile, prompts_db=None):
        config = RAGConfig()
        config.embedding.model = "mock"
        config.embedding.device = "cpu"
        config.vector_store.type = "memory"
        config.search.media_db_path = tmp_path / "absent-media.db"
        config.search.chachanotes_db_path = tmp_path / "absent-chacha.db"
        config.search.prompts_db_path = prompts_db or tmp_path / "absent-prompts.db"
        service = create_rag_service(profile_name=profile, config=config)
        try:
            assert service.profile.id == profile
            yield service
        finally:
            with monkeypatch.context() as cleanup:
                cleanup.setitem(sys.modules, "torch", None)
                service.close()

    return open_service, provider_calls


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_v2_full_features(tmp_path, offline_profile_service):
    """Test V2 service with all features enabled."""
    from Tests.RAG_Search.test_keyword_leg_prompts import _seed_prompts

    prompts, ids = _seed_prompts(
        tmp_path,
        [("Quarterly review", "Summarize revenue growth and customer satisfaction.")],
    )
    open_service, provider_calls = offline_profile_service
    with open_service("hybrid_full", prompts) as service:
        assert service.enable_parent_retrieval
        assert service.enable_reranking
        assert service.enable_parallel_processing
        indexed = await service.index_document(
            doc_id="test_v2_001",
            content=create_test_document(),
            title="Test Document for V2",
            metadata={"test": "v2_features"},
        )
        assert indexed.success and indexed.chunks_created > 0

        for query in ("revenue growth", "customer satisfaction"):
            calls_before = len(provider_calls)
            results = await service.search(
                query=query, top_k=5, search_type="hybrid", rerank=True
            )
            assert any(row.metadata.get("doc_id") == "test_v2_001" for row in results)
            assert any(row.id == f"prompt_{ids['Quarterly review']}" for row in results)
            assert any(row.metadata.get("rerank_score") == 0.5 for row in results)
            assert len(provider_calls) > calls_before


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_profile_switching(tmp_path, offline_profile_service):
    """Test switching between different profiles."""
    from Tests.RAG_Search.test_keyword_leg_prompts import _seed_prompts

    text = "This is a test document for profile testing."
    prompts, ids = _seed_prompts(tmp_path, [("Profile test", text)])
    open_service, _ = offline_profile_service
    profiles = {
        "bm25_only": "keyword",
        "vector_only": "semantic",
        "hybrid_basic": "hybrid",
        "hybrid_enhanced": "hybrid",
        "hybrid_full": "hybrid",
    }
    for profile, search_type in profiles.items():
        doc_id = f"test_{profile}"
        with open_service(profile, prompts if search_type == "keyword" else None) as service:
            indexed = await service.index_document(
                doc_id=doc_id, content=text, title=f"Test {profile}"
            )
            assert indexed.success and indexed.chunks_created > 0
            results = await service.search(
                query="test document", search_type=search_type, top_k=1
            )
            assert len(results) == 1, profile
            if search_type == "keyword":
                assert results[0].id == f"prompt_{ids['Profile test']}"
            else:
                assert results[0].metadata["doc_id"] == doc_id


# Remove main function and if __name__ block - pytest will find and run the tests automatically
