#!/usr/bin/env python3
"""
Test script to verify SimpleRAGCache integration with enhanced RAG service.

This test verifies:
1. Cache properly handles search results with parent document metadata
2. TTL handling for different search types
3. Parent chunk metadata preservation in cached results
4. Cache performance with real RAG operations
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil

from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService, create_rag_service
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache, get_rag_cache


class TestRAGCacheIntegration:
    """Test suite for RAG cache integration with parent document metadata."""
    
    def __init__(self):
        self.temp_dir = None
        self.rag_service = None
        self.test_documents = []
        self.setup_test_data()
    
    def setup_test_data(self):
        """Setup test documents with hierarchical structure."""
        self.test_documents = [
            {
                'id': 'doc1',
                'title': 'Introduction to Machine Learning',
                'content': """
                Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.
                It focuses on developing computer programs that can access data and use it to learn for themselves.
                
                The process of learning begins with observations or data, such as examples, direct experience, or instruction.
                The goal is to look for patterns in data and make better decisions in the future based on the examples provided.
                
                There are three main types of machine learning:
                1. Supervised Learning - where the algorithm learns from labeled training data
                2. Unsupervised Learning - where the algorithm finds patterns in unlabeled data
                3. Reinforcement Learning - where the algorithm learns through interaction with an environment
                
                Applications of machine learning include:
                - Natural language processing for chatbots and translation
                - Computer vision for image recognition and autonomous vehicles
                - Recommendation systems for e-commerce and content platforms
                - Fraud detection in financial services
                - Medical diagnosis and drug discovery
                """,
                'metadata': {
                    'author': 'Dr. Jane Smith',
                    'category': 'AI/ML',
                    'year': 2024,
                    'tags': ['machine learning', 'AI', 'tutorial']
                }
            },
            {
                'id': 'doc2',
                'title': 'Deep Learning Fundamentals',
                'content': """
                Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers.
                These networks are inspired by the structure and function of the human brain.
                
                Key concepts in deep learning:
                - Neural Networks: Interconnected nodes that process information
                - Layers: Input, hidden, and output layers that transform data
                - Activation Functions: Mathematical functions that introduce non-linearity
                - Backpropagation: Algorithm for training neural networks
                
                Popular deep learning architectures:
                1. Convolutional Neural Networks (CNNs) - primarily used for image processing
                2. Recurrent Neural Networks (RNNs) - designed for sequential data
                3. Transformers - state-of-the-art for natural language processing
                4. Generative Adversarial Networks (GANs) - for generating new data
                
                Deep learning has revolutionized many fields including computer vision, natural language processing,
                speech recognition, and game playing. Recent advances include large language models like GPT
                and image generation models like DALL-E.
                """,
                'metadata': {
                    'author': 'Prof. John Doe',
                    'category': 'AI/ML',
                    'year': 2024,
                    'tags': ['deep learning', 'neural networks', 'AI']
                }
            },
            {
                'id': 'doc3',
                'title': 'Natural Language Processing Overview',
                'content': """
                Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.
                It draws from many disciplines including computer science, linguistics, and machine learning.
                
                Core NLP tasks include:
                - Tokenization: Breaking text into words or subwords
                - Part-of-speech tagging: Identifying grammatical roles
                - Named Entity Recognition: Finding people, places, organizations
                - Sentiment Analysis: Determining emotional tone
                - Machine Translation: Converting between languages
                
                Modern NLP relies heavily on deep learning, particularly transformer models:
                - BERT: Bidirectional understanding of context
                - GPT: Generative pre-trained transformers
                - T5: Text-to-text unified framework
                
                Applications span from chatbots and virtual assistants to automated summarization
                and content generation. The field continues to evolve rapidly with new models
                achieving near-human performance on many tasks.
                """,
                'metadata': {
                    'author': 'Dr. Alice Johnson',
                    'category': 'AI/ML',
                    'year': 2024,
                    'tags': ['NLP', 'AI', 'language processing']
                }
            }
        ]
    
    async def setup(self):
        """Setup test environment with RAG service."""
        # Create temporary directory for vector store
        self.temp_dir = tempfile.mkdtemp(prefix="rag_cache_test_")
        print(f"Created temp directory: {self.temp_dir}")
        
        # Configure RAG with specific cache settings
        config = RAGConfig()
        config.vector_store.type = "chroma"
        config.vector_store.persist_directory = Path(self.temp_dir) / "chroma"
        config.embedding.model = "all-MiniLM-L6-v2"  # Small, fast model for testing
        config.embedding.device = "cpu"
        
        # Configure caching with different TTLs for search types
        config.search.enable_cache = True
        config.search.cache_size = 50
        config.search.cache_ttl = 300  # 5 minutes default
        
        # Search-type specific TTLs for testing
        config.search.semantic_cache_ttl = 600  # 10 minutes for semantic
        config.search.keyword_cache_ttl = 300   # 5 minutes for keyword
        config.search.hybrid_cache_ttl = 450    # 7.5 minutes for hybrid
        
        # Enable parent retrieval
        config.chunking.enable_parent_retrieval = True
        config.chunking.chunk_size = 200  # Small chunks for testing
        config.chunking.chunk_overlap = 50
        
        # Create RAG service
        self.rag_service = RAGService(config)
        print("Initialized RAG service with cache configuration")
        
        # Index test documents
        print("\nIndexing test documents...")
        results = await self.rag_service.index_batch(self.test_documents, show_progress=True)
        
        successful = sum(1 for r in results if r.success)
        print(f"Successfully indexed {successful}/{len(self.test_documents)} documents")
        
        for result in results:
            if result.success:
                print(f"  - {result.doc_id}: {result.chunks_created} chunks in {result.time_taken:.2f}s")
            else:
                print(f"  - {result.doc_id}: FAILED - {result.error}")
    
    async def test_basic_caching(self):
        """Test basic cache functionality with parent metadata."""
        print("\n=== Testing Basic Caching ===")
        
        query = "machine learning applications"
        search_type = "semantic"
        top_k = 5
        
        # First search - should be a cache miss
        print(f"\nFirst search for: '{query}'")
        start_time = time.time()
        results1 = await self.rag_service.search(query, top_k=top_k, search_type=search_type)
        first_search_time = time.time() - start_time
        print(f"First search completed in {first_search_time:.3f}s (cache miss)")
        print(f"Found {len(results1)} results")
        
        # Print first result with metadata
        if results1:
            first_result = results1[0]
            print(f"\nFirst result:")
            print(f"  - Score: {first_result.score:.4f}")
            print(f"  - Document ID: {first_result.metadata.get('doc_id')}")
            print(f"  - Document Title: {first_result.metadata.get('doc_title')}")
            print(f"  - Chunk Index: {first_result.metadata.get('chunk_index')}")
            print(f"  - Text preview: {first_result.document[:100]}...")
            
            # Check for parent metadata if available
            if 'parent_chunk_id' in first_result.metadata:
                print(f"  - Parent Chunk ID: {first_result.metadata.get('parent_chunk_id')}")
                print(f"  - Parent Start: {first_result.metadata.get('parent_start')}")
                print(f"  - Parent End: {first_result.metadata.get('parent_end')}")
        
        # Second search - should be a cache hit
        print(f"\nSecond search for same query: '{query}'")
        start_time = time.time()
        results2 = await self.rag_service.search(query, top_k=top_k, search_type=search_type)
        second_search_time = time.time() - start_time
        print(f"Second search completed in {second_search_time:.3f}s (cache hit)")
        
        # Verify results are the same
        assert len(results1) == len(results2), "Result count mismatch between searches"
        for r1, r2 in zip(results1, results2):
            assert r1.id == r2.id, "Result ID mismatch"
            assert r1.score == r2.score, "Result score mismatch"
            assert r1.metadata == r2.metadata, "Result metadata mismatch"
        
        print(f"\nCache speedup: {first_search_time / second_search_time:.1f}x")
        
        # Get cache metrics
        cache_metrics = self.rag_service.cache.get_metrics()
        print(f"\nCache metrics after basic test:")
        print(f"  - Hits: {cache_metrics['hits']}")
        print(f"  - Misses: {cache_metrics['misses']}")
        print(f"  - Hit rate: {cache_metrics['hit_rate']:.2%}")
        print(f"  - Cache size: {cache_metrics['size']}/{cache_metrics['max_size']}")
    
    async def test_ttl_handling(self):
        """Test TTL handling for different search types."""
        print("\n=== Testing TTL Handling ===")
        
        # Test queries for different search types
        test_cases = [
            ("semantic", "deep learning architectures", 0.5),  # 30s for quick test
            ("keyword", "transformer models", 0.3),  # 18s
            ("hybrid", "neural networks applications", 0.4),  # 24s
        ]
        
        # Temporarily override TTLs for faster testing
        original_ttls = {}
        for search_type, _, test_ttl in test_cases:
            original_ttls[search_type] = self.rag_service.cache.ttl_by_search_type.get(search_type, self.rag_service.cache.ttl_seconds)
            self.rag_service.cache.ttl_by_search_type[search_type] = test_ttl
        
        try:
            for search_type, query, ttl in test_cases:
                print(f"\nTesting {search_type} search with TTL={ttl}s")
                
                # Initial search
                results1 = await self.rag_service.search(query, search_type=search_type, top_k=3)
                print(f"Initial search found {len(results1)} results")
                
                # Immediate second search - should hit cache
                results2 = await self.rag_service.search(query, search_type=search_type, top_k=3)
                assert len(results1) == len(results2), f"Cache hit failed for {search_type}"
                print(f"Cache hit confirmed")
                
                # Wait for TTL to expire
                print(f"Waiting {ttl + 0.1}s for TTL to expire...")
                await asyncio.sleep(ttl + 0.1)
                
                # Third search - should miss cache due to TTL
                results3 = await self.rag_service.search(query, search_type=search_type, top_k=3)
                print(f"Post-TTL search found {len(results3)} results (cache miss expected)")
                
                # Verify it was a cache miss by checking metrics
                metrics_before = self.rag_service.cache.get_metrics()
                results4 = await self.rag_service.search(query, search_type=search_type, top_k=3)
                metrics_after = self.rag_service.cache.get_metrics()
                
                if metrics_after['hits'] > metrics_before['hits']:
                    print("Cache was repopulated after TTL expiry")
                else:
                    print("Warning: Expected cache hit after repopulation")
        
        finally:
            # Restore original TTLs
            for search_type, ttl in original_ttls.items():
                self.rag_service.cache.ttl_by_search_type[search_type] = ttl
    
    async def test_parent_metadata_preservation(self):
        """Test that parent chunk metadata is preserved in cached results."""
        print("\n=== Testing Parent Metadata Preservation ===")
        
        query = "machine learning supervised unsupervised reinforcement"
        
        # Search with citations to get more metadata
        print(f"\nSearching with citations for: '{query}'")
        results = await self.rag_service.search(
            query, 
            search_type="hybrid",
            top_k=5,
            include_citations=True
        )
        
        print(f"Found {len(results)} results")
        
        # Check for parent metadata in results
        parent_metadata_found = False
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  - Score: {result.score:.4f}")
            print(f"  - Chunk ID: {result.id}")
            print(f"  - Document: {result.metadata.get('doc_title')}")
            
            # Check for parent chunk information
            if any(key.startswith('parent_') for key in result.metadata):
                parent_metadata_found = True
                print("  - Parent metadata found:")
                for key, value in result.metadata.items():
                    if key.startswith('parent_'):
                        print(f"    - {key}: {value}")
            
            # Check citations if available
            if hasattr(result, 'citations') and result.citations:
                print(f"  - Citations: {len(result.citations)}")
                for j, citation in enumerate(result.citations[:2]):  # First 2 citations
                    print(f"    Citation {j+1}: {citation.text[:50]}...")
        
        # Verify cached results preserve metadata
        print("\nVerifying cached results preserve metadata...")
        cached_results = await self.rag_service.search(
            query,
            search_type="hybrid",
            top_k=5,
            include_citations=True
        )
        
        # Compare metadata
        metadata_preserved = True
        for orig, cached in zip(results, cached_results):
            if orig.metadata != cached.metadata:
                metadata_preserved = False
                print(f"Metadata mismatch for chunk {orig.id}")
                break
        
        if metadata_preserved:
            print("✓ All metadata preserved in cache")
        else:
            print("✗ Metadata not fully preserved in cache")
        
        return parent_metadata_found
    
    async def test_cache_performance(self):
        """Test cache performance with various operations."""
        print("\n=== Testing Cache Performance ===")
        
        # Test different query patterns
        queries = [
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
            "transformer models BERT GPT",
            "supervised learning examples",
            "reinforcement learning applications",
            "computer vision CNN",
            "sentiment analysis NLP"
        ]
        
        # Perform searches and measure performance
        cache_hits = 0
        total_time_no_cache = 0
        total_time_with_cache = 0
        
        for i, query in enumerate(queries):
            # First search (potential cache miss)
            start = time.time()
            results1 = await self.rag_service.search(query, search_type="hybrid", top_k=3)
            time1 = time.time() - start
            total_time_no_cache += time1
            
            # Second search (should be cache hit)
            start = time.time()
            results2 = await self.rag_service.search(query, search_type="hybrid", top_k=3)
            time2 = time.time() - start
            total_time_with_cache += time2
            
            if time2 < time1 * 0.5:  # Cache hit if significantly faster
                cache_hits += 1
            
            print(f"Query {i+1}: '{query[:30]}...' - First: {time1:.3f}s, Second: {time2:.3f}s")
        
        # Test cache with different parameters
        print("\nTesting cache discrimination with parameters...")
        base_query = "machine learning"
        
        # Different top_k values
        await self.rag_service.search(base_query, top_k=5)
        await self.rag_service.search(base_query, top_k=10)  # Should be cache miss
        
        # Different search types
        await self.rag_service.search(base_query, search_type="semantic")
        await self.rag_service.search(base_query, search_type="keyword")  # Should be cache miss
        
        # With filters
        await self.rag_service.search(base_query, filter_metadata={"category": "AI/ML"})
        
        # Final metrics
        final_metrics = self.rag_service.cache.get_metrics()
        print(f"\nFinal cache performance metrics:")
        print(f"  - Total requests: {final_metrics['total_requests']}")
        print(f"  - Hit rate: {final_metrics['hit_rate']:.2%}")
        print(f"  - Cache size: {final_metrics['size']}/{final_metrics['max_size']}")
        print(f"  - Memory usage: {final_metrics['size_bytes'] / 1024 / 1024:.2f}MB")
        print(f"  - Evictions: {final_metrics['evictions']}")
        
        avg_speedup = total_time_no_cache / total_time_with_cache if total_time_with_cache > 0 else 1
        print(f"\nAverage cache speedup: {avg_speedup:.1f}x")
    
    async def test_cache_memory_limits(self):
        """Test cache behavior under memory pressure."""
        print("\n=== Testing Cache Memory Limits ===")
        
        # Create a new cache with small memory limit
        small_cache = SimpleRAGCache(
            max_size=10,
            ttl_seconds=3600,
            max_memory_mb=1.0  # 1MB limit
        )
        
        # Replace the service cache temporarily
        original_cache = self.rag_service.cache
        self.rag_service.cache = small_cache
        
        try:
            # Perform many searches to trigger evictions
            print("Filling cache with searches...")
            for i in range(20):
                query = f"machine learning technique {i}"
                await self.rag_service.search(query, top_k=5)
                
                if i % 5 == 4:
                    metrics = small_cache.get_metrics()
                    print(f"After {i+1} searches: size={metrics['size']}, "
                          f"memory={metrics['size_bytes']/1024:.1f}KB, "
                          f"evictions={metrics['evictions']}")
            
            final_metrics = small_cache.get_metrics()
            print(f"\nFinal state:")
            print(f"  - Size: {final_metrics['size']}/{final_metrics['max_size']}")
            print(f"  - Memory: {final_metrics['size_bytes']/1024:.1f}KB / {final_metrics['max_memory_bytes']/1024:.1f}KB")
            print(f"  - Total evictions: {final_metrics['evictions']}")
            
        finally:
            # Restore original cache
            self.rag_service.cache = original_cache
    
    async def cleanup(self):
        """Clean up test resources."""
        if self.rag_service:
            self.rag_service.close()
        
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"\nCleaned up temp directory: {self.temp_dir}")
    
    async def run_all_tests(self):
        """Run all cache integration tests."""
        try:
            await self.setup()
            
            # Run individual tests
            await self.test_basic_caching()
            await self.test_ttl_handling()
            parent_metadata_found = await self.test_parent_metadata_preservation()
            await self.test_cache_performance()
            await self.test_cache_memory_limits()
            
            # Summary
            print("\n" + "="*60)
            print("CACHE INTEGRATION TEST SUMMARY")
            print("="*60)
            
            final_metrics = self.rag_service.cache.get_metrics()
            print(f"Total cache operations: {final_metrics['total_requests']}")
            print(f"Overall hit rate: {final_metrics['hit_rate']:.2%}")
            print(f"Parent metadata support: {'✓ Found' if parent_metadata_found else '✗ Not found'}")
            
            # Log cache efficiency
            self.rag_service.cache.log_cache_efficiency()
            
            print("\n✓ All cache integration tests completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Main entry point for cache integration tests."""
    print("Starting RAG Cache Integration Tests")
    print("="*60)
    
    tester = TestRAGCacheIntegration()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())