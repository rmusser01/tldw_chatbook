#!/usr/bin/env python3
"""
Test script for the enhanced RAG implementation.

This script demonstrates the new features:
1. Enhanced chunking with character-level position tracking
2. Hierarchical chunking with structure preservation
3. Parent document retrieval
4. PDF artifact cleaning
5. Table serialization
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


async def test_enhanced_chunking():
    """Test enhanced chunking features."""
    from tldw_chatbook.RAG_Search.enhanced_chunking_service import EnhancedChunkingService
    
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
        serialize_tables=True
    )
    
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
        content,
        chunk_size=150,
        chunk_overlap=30,
        parent_size_multiplier=3
    )
    
    logger.info(f"Created {parent_result['metadata']['total_chunks']} retrieval chunks")
    logger.info(f"Created {parent_result['metadata']['total_parent_chunks']} parent chunks")
    
    # Show mapping
    for i, chunk in enumerate(parent_result['chunks'][:3]):
        parent_idx = chunk.get('metadata', {}).get('parent_chunk_index')
        if parent_idx is not None:
            logger.info(f"\nRetrieval chunk {i} -> Parent chunk {parent_idx}")
            logger.info(f"  Retrieval text: {chunk['text'][:80]}...")
            parent_text = parent_result['parent_chunks'][parent_idx]['text']
            logger.info(f"  Parent text: {parent_text[:150]}...")


async def test_enhanced_rag_service():
    """Test the enhanced RAG service."""
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service import create_enhanced_rag_service
    
    logger.info("\n=== Testing Enhanced RAG Service ===")
    
    # Create service with in-memory vector store for testing
    # Use a small embedding model for testing
    service = create_enhanced_rag_service(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Small, fast model
        vector_store="memory",
        enable_parent_retrieval=True
    )
    
    content = create_test_document()
    
    # Test indexing with parent retrieval
    logger.info("\n1. Indexing document with parent retrieval...")
    result = await service.index_document_with_parents(
        doc_id="test_doc_001",
        content=content,
        title="Annual Report 2024",
        metadata={
            "type": "annual_report",
            "year": 2024,
            "department": "finance"
        },
        use_structural_chunking=True
    )
    
    logger.info(f"Indexing result: {result}")
    
    # Test search with context expansion
    logger.info("\n2. Testing search with context expansion...")
    queries = [
        "quarterly revenue growth",
        "customer satisfaction score",
        "investment plans for R&D"
    ]
    
    for query in queries:
        logger.info(f"\nSearching for: '{query}'")
        
        # Search without expansion
        results_basic = await service.search(
            query=query,
            top_k=2,
            search_type="semantic"
        )
        
        # Search with expansion
        results_expanded = await service.search_with_context_expansion(
            query=query,
            top_k=2,
            search_type="semantic",
            expand_to_parent=True
        )
        
        logger.info(f"Basic results: {len(results_basic)} chunks")
        logger.info(f"Expanded results: {len(results_expanded)} chunks")
        
        # Compare first result
        if results_basic and results_expanded:
            basic_len = len(results_basic[0].document)
            expanded_len = len(results_expanded[0].document)
            logger.info(f"Context expansion: {basic_len} chars -> {expanded_len} chars")
            
            if results_expanded[0].metadata.get('context_expanded'):
                logger.info("✓ Context successfully expanded to parent chunk")


async def test_table_serialization():
    """Test table serialization functionality."""
    from tldw_chatbook.RAG_Search.table_serializer import serialize_table, TableFormat
    
    logger.info("\n=== Testing Table Serialization ===")
    
    # Test markdown table
    table_text = """
| Product | Q1 Sales | Q2 Sales | Growth |
|---------|----------|----------|--------|
| Laptop  | 1000     | 1200     | 20%    |
| Phone   | 2000     | 2500     | 25%    |
| Tablet  | 500      | 600      | 20%    |
"""
    
    result = serialize_table(table_text, format=TableFormat.MARKDOWN, method="hybrid")
    
    logger.info("Table serialization result:")
    logger.info(f"  Rows: {result['metadata']['num_rows']}")
    logger.info(f"  Columns: {result['metadata']['num_columns']}")
    
    logger.info("\nEntity blocks:")
    for block in result['entity_blocks'][:2]:
        logger.info(f"  - {block['information_block']}")
    
    logger.info("\nGenerated sentences:")
    for sentence in result['sentences'][:3]:
        logger.info(f"  - {sentence}")


async def test_pdf_artifact_cleaning():
    """Test PDF artifact cleaning."""
    from tldw_chatbook.RAG_Search.enhanced_chunking_service import DocumentStructureParser
    
    logger.info("\n=== Testing PDF Artifact Cleaning ===")
    
    parser = DocumentStructureParser()
    
    # Test text with common PDF artifacts
    pdf_text = """
This is a test/period document with various/comma PDF artifacts/colon
The number/five is written as text/period Also/comma we have/lparen parentheses/rparen
Special characters like/dollar and/percent are common/period
Sometimes we see glyph<123> or /A.cap patterns/period
"""
    
    cleaned_text, corrections = parser.clean_text(pdf_text)
    
    logger.info(f"Original text length: {len(pdf_text)}")
    logger.info(f"Cleaned text length: {len(cleaned_text)}")
    logger.info(f"Corrections made: {len(corrections)}")
    
    logger.info("\nFirst 5 corrections:")
    for original, replacement in corrections[:5]:
        logger.info(f"  '{original}' -> '{replacement}'")
    
    logger.info(f"\nCleaned text preview:\n{cleaned_text[:200]}...")


async def main():
    """Run all tests."""
    logger.info("Starting Enhanced RAG Implementation Tests\n")
    
    try:
        # Test individual components
        await test_enhanced_chunking()
        await test_table_serialization()
        await test_pdf_artifact_cleaning()
        
        # Test integrated service
        await test_enhanced_rag_service()
        
        logger.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())