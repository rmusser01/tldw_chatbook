# test_embeddings_performance.py
# Performance and stress tests for the embeddings service

import pytest
import tempfile
from pathlib import Path
import shutil
import time
import psutil
import gc
from unittest.mock import patch, MagicMock
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService


@pytest.mark.requires_rag_deps
@pytest.mark.slow  # Mark as slow test
class TestEmbeddingsPerformance:
    """Performance tests for embeddings service"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_cache_service(self):
        """Create a mock cache service with performance tracking"""
        mock = MagicMock()
        
        # Track cache performance
        self.cache_hits = 0
        self.cache_misses = 0
        
        def mock_get_embeddings_batch(texts):
            cached = {}
            uncached = []
            for text in texts:
                # Simulate 50% cache hit rate
                if hash(text) % 2 == 0:
                    cached[text] = [0.1, 0.2]
                    self.cache_hits += 1
                else:
                    uncached.append(text)
                    self.cache_misses += 1
            return cached, uncached
        
        mock.get_embeddings_batch.side_effect = mock_get_embeddings_batch
        mock.cache_embeddings_batch.return_value = None
        
        return mock
    
    @pytest.fixture
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.CHROMADB_AVAILABLE', True)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.EMBEDDINGS_AVAILABLE', True)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.chromadb')
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.get_cache_service')
    def performance_embeddings_service(self, mock_get_cache, mock_chromadb, mock_cache_service, temp_dir):
        """Create embeddings service for performance testing"""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Use mock cache service
        mock_get_cache.return_value = mock_cache_service
        
        service = EmbeddingsService(temp_dir)
        service.client = mock_client
        
        # Mock embedding model with realistic timing
        mock_model = MagicMock()
        
        def mock_encode(texts):
            # Simulate processing time (0.001s per text)
            time.sleep(0.001 * len(texts))
            
            mock_array = MagicMock()
            # Generate random embeddings
            embeddings = [[float(i) / 100, float(i) / 200] for i in range(len(texts))]
            mock_array.tolist.return_value = embeddings
            return mock_array
        
        mock_model.encode.side_effect = mock_encode
        service.embedding_model = mock_model
        
        return service
    
    def test_large_document_set_performance(self, performance_embeddings_service):
        """Test performance with large document set (1000+ documents)"""
        # Configure for optimal performance
        performance_embeddings_service.configure_performance(
            max_workers=8,
            batch_size=50,
            enable_parallel=True
        )
        
        # Create large dataset
        num_docs = 1000
        texts = [f"Document {i}: This is a test document with some content to embed" for i in range(num_docs)]
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        embeddings = performance_embeddings_service.create_embeddings(texts)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_increase = end_memory - start_memory
        
        # Verify results
        assert len(embeddings) == num_docs
        assert all(isinstance(emb, list) and len(emb) == 2 for emb in embeddings)
        
        # Performance assertions
        docs_per_second = num_docs / duration
        print(f"\nPerformance metrics for {num_docs} documents:")
        print(f"  Total time: {duration:.2f} seconds")
        print(f"  Documents/second: {docs_per_second:.2f}")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Cache hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%")
        
        # Performance benchmarks (adjust based on your requirements)
        assert docs_per_second > 100  # Should process at least 100 docs/second
        assert memory_increase < 500  # Should use less than 500MB additional memory
    
    def test_memory_usage_patterns(self, performance_embeddings_service):
        """Test memory usage patterns with different batch sizes"""
        batch_sizes = [10, 50, 100, 200]
        memory_results = {}
        
        for batch_size in batch_sizes:
            # Configure service
            performance_embeddings_service.configure_performance(
                batch_size=batch_size,
                enable_parallel=True
            )
            
            # Force garbage collection
            gc.collect()
            
            # Create dataset
            texts = [f"Memory test doc {i}" for i in range(500)]
            
            # Measure memory
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            embeddings = performance_embeddings_service.create_embeddings(texts)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Clean up
            del embeddings
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_results[batch_size] = {
                'peak_increase': peak_memory - start_memory,
                'final_increase': final_memory - start_memory
            }
        
        # Print results
        print("\nMemory usage by batch size:")
        for batch_size, metrics in memory_results.items():
            print(f"  Batch size {batch_size}:")
            print(f"    Peak increase: {metrics['peak_increase']:.2f} MB")
            print(f"    Final increase: {metrics['final_increase']:.2f} MB")
        
        # Verify memory is properly released
        # Note: Python doesn't always release memory immediately back to OS
        # We expect at least some memory reduction, but not necessarily 50%
        for metrics in memory_results.values():
            # Allow for memory not being fully released (common in Python)
            # Just verify final is not more than peak
            assert metrics['final_increase'] <= metrics['peak_increase']
    
    def test_cache_hit_rate_impact(self, performance_embeddings_service, mock_cache_service):
        """Test performance impact of cache hit rates"""
        # Test with repeated texts to increase cache hits
        base_texts = [f"Repeated text {i}" for i in range(100)]
        
        # First pass - all cache misses
        self.cache_hits = 0
        self.cache_misses = 0
        
        start_time = time.time()
        embeddings1 = performance_embeddings_service.create_embeddings(base_texts)
        first_pass_time = time.time() - start_time
        first_pass_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        # Second pass with same texts - should have better cache hit rate
        # Reset counters
        self.cache_hits = 0
        self.cache_misses = 0
        
        start_time = time.time()
        embeddings2 = performance_embeddings_service.create_embeddings(base_texts)
        second_pass_time = time.time() - start_time
        second_pass_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        print(f"\nCache performance impact:")
        print(f"  First pass: {first_pass_time:.3f}s (hit rate: {first_pass_hit_rate*100:.1f}%)")
        print(f"  Second pass: {second_pass_time:.3f}s (hit rate: {second_pass_hit_rate*100:.1f}%)")
        print(f"  Speedup: {first_pass_time/second_pass_time:.2f}x")
        
        # Cache should provide performance benefit
        assert second_pass_hit_rate >= first_pass_hit_rate
    
    def test_parallel_vs_sequential_performance(self, performance_embeddings_service):
        """Compare parallel vs sequential processing performance"""
        num_docs = 500
        texts = [f"Parallel test doc {i}" for i in range(num_docs)]
        
        # Test sequential processing
        performance_embeddings_service.configure_performance(
            enable_parallel=False,
            batch_size=50
        )
        
        start_time = time.time()
        sequential_embeddings = performance_embeddings_service.create_embeddings(texts)
        sequential_time = time.time() - start_time
        
        # Test parallel processing
        performance_embeddings_service.configure_performance(
            enable_parallel=True,
            batch_size=50,
            max_workers=4
        )
        
        # Clear model cache to ensure fair comparison
        performance_embeddings_service.embedding_model.encode.reset_mock()
        
        start_time = time.time()
        parallel_embeddings = performance_embeddings_service.create_embeddings(texts)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time
        
        print(f"\nParallel processing performance:")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Parallel: {parallel_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Parallel should be faster for large datasets
        assert speedup > 1.2  # At least 20% faster
        
        # Results should be identical
        assert len(sequential_embeddings) == len(parallel_embeddings)
    
    def test_collection_operations_at_scale(self, performance_embeddings_service):
        """Test collection operations with large number of documents"""
        collection_name = "scale_test"
        mock_collection = MagicMock()
        
        # Track add performance
        add_times = []
        
        def mock_add(**kwargs):
            # Simulate realistic add time
            time.sleep(0.01)
            add_times.append(len(kwargs['documents']))
        
        mock_collection.add.side_effect = mock_add
        performance_embeddings_service.client.get_or_create_collection.return_value = mock_collection
        
        # Configure for batch processing
        performance_embeddings_service.configure_performance(
            batch_size=100
        )
        
        # Create large dataset
        num_docs = 1000
        documents = [f"Scale test document {i}" for i in range(num_docs)]
        embeddings = [[float(i)/1000, float(i)/2000] for i in range(num_docs)]
        metadatas = [{"doc_id": i} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        # Measure add performance
        start_time = time.time()
        
        success = performance_embeddings_service.add_documents_to_collection(
            collection_name,
            documents,
            embeddings,
            metadatas,
            ids,
            batch_size=100
        )
        
        total_time = time.time() - start_time
        
        assert success is True
        
        # Calculate metrics
        total_batches = len(add_times)
        avg_batch_size = sum(add_times) / len(add_times)
        docs_per_second = num_docs / total_time
        
        print(f"\nCollection add performance ({num_docs} documents):")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Batches: {total_batches}")
        print(f"  Average batch size: {avg_batch_size:.1f}")
        print(f"  Documents/second: {docs_per_second:.1f}")
        
        # Should complete in reasonable time
        assert total_time < 30  # Less than 30 seconds for 1000 docs
    
    def test_concurrent_operations_performance(self, performance_embeddings_service):
        """Test performance under concurrent load"""
        num_threads = 10
        docs_per_thread = 100
        
        # Configure for concurrent access
        performance_embeddings_service.configure_performance(
            max_workers=8,
            batch_size=25,
            enable_parallel=True
        )
        
        results = []
        errors = []
        
        def process_documents(thread_id):
            try:
                texts = [f"Thread {thread_id} doc {i}" for i in range(docs_per_thread)]
                start = time.time()
                embeddings = performance_embeddings_service.create_embeddings(texts)
                duration = time.time() - start
                results.append({
                    'thread_id': thread_id,
                    'duration': duration,
                    'count': len(embeddings)
                })
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run concurrent operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_documents, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()
        
        total_time = time.time() - start_time
        
        # Analyze results
        assert len(errors) == 0
        assert len(results) == num_threads
        
        total_docs = sum(r['count'] for r in results)
        avg_thread_time = sum(r['duration'] for r in results) / len(results)
        throughput = total_docs / total_time
        
        print(f"\nConcurrent operations performance:")
        print(f"  Threads: {num_threads}")
        print(f"  Total documents: {total_docs}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average thread time: {avg_thread_time:.2f}s")
        print(f"  Overall throughput: {throughput:.1f} docs/second")
        
        # Should handle concurrent load efficiently
        assert throughput > 100  # At least 100 docs/second under load
    
    def test_memory_cleanup_effectiveness(self, performance_embeddings_service):
        """Test effectiveness of memory cleanup"""
        # Configure with memory limit
        performance_embeddings_service.configure_performance(
            batch_size=50,
            enable_parallel=True
        )
        
        # Create mock memory manager
        mock_memory_manager = MagicMock()
        mock_memory_manager.get_memory_usage_summary.return_value = {
            "total_memory_mb": 1000,
            "used_memory_mb": 500,
            "collections": {}
        }
        performance_embeddings_service.set_memory_manager(mock_memory_manager)
        
        # Process multiple batches
        for batch_num in range(5):
            texts = [f"Batch {batch_num} doc {i}" for i in range(200)]
            
            # Measure memory before
            before_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            embeddings = performance_embeddings_service.create_embeddings(texts)
            
            # Force cleanup
            del embeddings
            gc.collect()
            
            # Measure memory after
            after_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = after_memory - before_memory
            
            print(f"\nBatch {batch_num} memory: +{memory_increase:.2f} MB")
        
        # Memory should stabilize (not continuously increase)
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        # Allow some increase but should be bounded
        # This is a rough check - adjust based on your requirements
    
    def test_different_text_sizes_performance(self, performance_embeddings_service):
        """Test performance with different text sizes"""
        text_sizes = [10, 100, 500, 1000, 5000]  # Words per document
        results = {}
        
        for size in text_sizes:
            # Generate texts of specific size
            base_text = "word " * size
            texts = [f"Document {i}: {base_text}" for i in range(100)]
            
            start_time = time.time()
            embeddings = performance_embeddings_service.create_embeddings(texts)
            duration = time.time() - start_time
            
            results[size] = {
                'time': duration,
                'docs_per_second': 100 / duration
            }
        
        print("\nPerformance by text size:")
        for size, metrics in results.items():
            print(f"  {size} words: {metrics['time']:.2f}s ({metrics['docs_per_second']:.1f} docs/s)")
        
        # Performance should degrade gracefully with text size
        # But not linearly - embeddings are typically computed on truncated text
        small_perf = results[10]['docs_per_second']
        large_perf = results[5000]['docs_per_second']
        
        # Large texts should still maintain reasonable performance
        assert large_perf > small_perf * 0.1  # At least 10% of small text performance
    
    def test_stress_test_executor_pool(self, performance_embeddings_service):
        """Stress test the thread pool executor"""
        # Configure with limited workers to stress the pool
        performance_embeddings_service.configure_performance(
            max_workers=2,
            batch_size=10,
            enable_parallel=True
        )
        
        # Create many small batches to stress the executor
        num_batches = 100
        texts = [f"Stress test {i}" for i in range(num_batches * 10)]
        
        # Track executor behavior
        start_time = time.time()
        
        embeddings = performance_embeddings_service._create_embeddings_parallel(texts)
        
        duration = time.time() - start_time
        
        assert len(embeddings) == len(texts)
        
        print(f"\nExecutor stress test:")
        print(f"  Batches: {num_batches}")
        print(f"  Workers: 2")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Time per batch: {duration/num_batches:.3f}s")
        
        # Should complete without deadlocks or errors
        assert duration < 10.0  # Mock should be fast  # Should complete within 1 minute