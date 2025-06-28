# test_embeddings_performance.py
# Performance benchmarking tests for embeddings service

import pytest
import time
import statistics
import threading
import psutil
import gc
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import shutil

from tldw_chatbook.RAG_Search.Services.embeddings_service import (
    EmbeddingsService,
    SentenceTransformerProvider,
    InMemoryStore,
    ChromaDBStore
)

# Import test utilities from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import (
    MockEmbeddingProvider,
    requires_embeddings,
    requires_chromadb,
    large_text_batch,
    performance_monitor
)


@pytest.mark.performance
class TestEmbeddingPerformance:
    """Performance tests for embedding generation"""
    
    def test_single_vs_batch_performance(self, performance_monitor):
        """Compare performance of single vs batch embedding generation"""
        provider = MockEmbeddingProvider(delay=0.001)  # Small delay to simulate work
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        texts = [f"Test document {i}" for i in range(100)]
        
        # Single embedding generation (one at a time)
        performance_monitor.start()
        single_embeddings = []
        for text in texts:
            emb = service.create_embeddings([text])
            single_embeddings.extend(emb)
        single_time = performance_monitor.stop()
        
        # Batch embedding generation
        performance_monitor.start()
        batch_embeddings = service.create_embeddings(texts)
        batch_time = performance_monitor.stop()
        
        # Batch should be significantly faster
        assert batch_time < single_time * 0.5  # At least 2x faster
        assert len(single_embeddings) == len(batch_embeddings)
        
        # Record metrics
        performance_monitor.record_metric("single_time", single_time)
        performance_monitor.record_metric("batch_time", batch_time)
        performance_monitor.record_metric("speedup", single_time / batch_time)
    
    def test_parallel_processing_performance(self, performance_monitor):
        """Test performance improvement with parallel processing"""
        provider = MockEmbeddingProvider(delay=0.001)
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        # Large batch to trigger parallel processing
        texts = [f"Document {i}" for i in range(500)]
        
        # Sequential processing
        service.configure_performance(enable_parallel=False)
        performance_monitor.start()
        seq_embeddings = service.create_embeddings(texts)
        seq_time = performance_monitor.stop()
        
        # Parallel processing
        service.configure_performance(
            enable_parallel=True,
            max_workers=4,
            batch_size=50
        )
        performance_monitor.start()
        par_embeddings = service.create_embeddings(texts)
        par_time = performance_monitor.stop()
        
        # Parallel should be faster
        assert par_time < seq_time
        assert len(seq_embeddings) == len(par_embeddings)
        
        # Record speedup
        speedup = seq_time / par_time
        performance_monitor.record_metric("parallel_speedup", speedup)
        assert speedup > 1.5  # At least 1.5x speedup
    
    def test_cache_performance_impact(self, performance_monitor):
        """Measure performance impact of caching"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        # Prepare texts with some duplicates
        unique_texts = [f"Unique document {i}" for i in range(50)]
        repeated_texts = unique_texts * 3  # 150 texts, 100 duplicates
        
        # First run - no cache hits
        performance_monitor.start()
        embeddings1 = service.create_embeddings(repeated_texts)
        first_run_time = performance_monitor.stop()
        
        # Second run - should have cache hits
        performance_monitor.start()
        embeddings2 = service.create_embeddings(repeated_texts)
        second_run_time = performance_monitor.stop()
        
        # With cache, second run should be faster
        assert second_run_time < first_run_time * 0.7  # At least 30% faster
        
        # Record cache impact
        cache_speedup = first_run_time / second_run_time
        performance_monitor.record_metric("cache_speedup", cache_speedup)
    
    @requires_embeddings
    def test_real_model_performance(self, performance_monitor, sample_texts):
        """Benchmark performance with real embedding model"""
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Warmup
        _ = service.create_embeddings(["warmup"])
        
        # Benchmark different batch sizes
        batch_sizes = [1, 10, 50, 100]
        results = {}
        
        for batch_size in batch_sizes:
            texts = sample_texts[:batch_size]
            
            # Run multiple times for average
            times = []
            for _ in range(3):
                performance_monitor.start()
                embeddings = service.create_embeddings(texts)
                elapsed = performance_monitor.stop()
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time  # texts per second
            
            results[batch_size] = {
                "avg_time": avg_time,
                "throughput": throughput
            }
            
            performance_monitor.record_metric(f"batch_{batch_size}_throughput", throughput)
        
        # Larger batches should have better throughput
        assert results[100]["throughput"] > results[1]["throughput"]


@pytest.mark.performance
class TestVectorStorePerformance:
    """Performance tests for vector store operations"""
    
    def test_document_insertion_performance(self, performance_monitor):
        """Test performance of document insertion at scale"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        # Prepare documents
        num_docs = 1000
        texts = [f"Document {i}" for i in range(num_docs)]
        embeddings = service.create_embeddings(texts)
        metadatas = [{"id": i} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        # Measure insertion time
        performance_monitor.start()
        success = service.add_documents_to_collection(
            "perf_collection",
            texts,
            embeddings,
            metadatas,
            ids
        )
        insertion_time = performance_monitor.stop()
        
        assert success
        
        # Calculate insertion rate
        insertion_rate = num_docs / insertion_time
        performance_monitor.record_metric("insertion_rate", insertion_rate)
        
        # Should handle at least 100 docs/second with mock provider
        assert insertion_rate > 100
    
    def test_search_performance_scaling(self, performance_monitor):
        """Test how search performance scales with collection size"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        collection_sizes = [100, 500, 1000, 5000]
        search_times = {}
        
        for size in collection_sizes:
            # Create collection of given size
            collection_name = f"collection_{size}"
            texts = [f"Document {i}" for i in range(size)]
            embeddings = service.create_embeddings(texts)
            
            service.add_documents_to_collection(
                collection_name,
                texts,
                embeddings,
                [{"id": i} for i in range(size)],
                [f"doc_{i}" for i in range(size)]
            )
            
            # Measure search time
            query_embeddings = service.create_embeddings(["search query"])
            
            times = []
            for _ in range(5):  # Multiple runs for average
                performance_monitor.start()
                results = service.search_collection(
                    collection_name,
                    query_embeddings,
                    n_results=10
                )
                elapsed = performance_monitor.stop()
                times.append(elapsed)
            
            avg_search_time = statistics.mean(times)
            search_times[size] = avg_search_time
            performance_monitor.record_metric(f"search_time_{size}", avg_search_time)
        
        # Search time should not increase linearly with collection size
        # For in-memory store, it might increase somewhat
        time_ratio = search_times[5000] / search_times[100]
        size_ratio = 5000 / 100
        
        # Time should increase slower than size
        assert time_ratio < size_ratio * 0.5
    
    @requires_chromadb
    def test_chromadb_vs_memory_performance(self, temp_dir, performance_monitor):
        """Compare ChromaDB vs in-memory store performance"""
        provider = MockEmbeddingProvider()
        
        # Prepare test data
        texts = [f"Document {i}" for i in range(500)]
        
        # Test in-memory store
        memory_service = EmbeddingsService(vector_store=InMemoryStore())
        memory_service.add_provider("test", provider)
        embeddings = memory_service.create_embeddings(texts)
        
        performance_monitor.start()
        memory_service.add_documents_to_collection(
            "test_collection",
            texts,
            embeddings,
            [{"id": i} for i in range(len(texts))],
            [f"doc_{i}" for i in range(len(texts))]
        )
        memory_time = performance_monitor.stop()
        
        # Test ChromaDB store
        chroma_service = EmbeddingsService(persist_directory=temp_dir)
        chroma_service.add_provider("test", provider)
        
        performance_monitor.start()
        chroma_service.add_documents_to_collection(
            "test_collection",
            texts,
            embeddings,
            [{"id": i} for i in range(len(texts))],
            [f"doc_{i}" for i in range(len(texts))]
        )
        chroma_time = performance_monitor.stop()
        
        # Record comparison
        performance_monitor.record_metric("memory_store_time", memory_time)
        performance_monitor.record_metric("chromadb_store_time", chroma_time)
        
        # Both should complete in reasonable time
        assert memory_time < 5.0
        assert chroma_time < 10.0


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Performance tests for concurrent operations"""
    
    def test_concurrent_embedding_throughput(self, performance_monitor):
        """Test throughput under concurrent load"""
        provider = MockEmbeddingProvider(delay=0.001)
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        service.configure_performance(max_workers=8)
        
        num_threads = 10
        texts_per_thread = 50
        
        def create_embeddings_task(thread_id):
            texts = [f"Thread {thread_id} doc {i}" for i in range(texts_per_thread)]
            return service.create_embeddings(texts)
        
        # Sequential baseline
        performance_monitor.start()
        for i in range(num_threads):
            create_embeddings_task(i)
        sequential_time = performance_monitor.stop()
        
        # Concurrent execution
        threads = []
        performance_monitor.start()
        
        for i in range(num_threads):
            thread = threading.Thread(target=create_embeddings_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = performance_monitor.stop()
        
        # Calculate throughput
        total_texts = num_threads * texts_per_thread
        sequential_throughput = total_texts / sequential_time
        concurrent_throughput = total_texts / concurrent_time
        
        performance_monitor.record_metric("sequential_throughput", sequential_throughput)
        performance_monitor.record_metric("concurrent_throughput", concurrent_throughput)
        
        # Concurrent should be faster
        assert concurrent_throughput > sequential_throughput * 1.5
    
    def test_provider_switching_overhead(self, performance_monitor):
        """Test overhead of switching between providers"""
        service = EmbeddingsService(vector_store=InMemoryStore())
        
        # Add multiple providers
        for i in range(5):
            provider = MockEmbeddingProvider(dimension=384 + i * 128)
            service.add_provider(f"provider_{i}", provider)
        
        texts = ["Test document"]
        
        # Measure overhead of provider switching
        switch_times = []
        
        for _ in range(100):
            provider_id = f"provider_{_ % 5}"
            
            performance_monitor.start()
            service.set_provider(provider_id)
            embeddings = service.create_embeddings(texts)
            elapsed = performance_monitor.stop()
            
            switch_times.append(elapsed)
        
        avg_switch_time = statistics.mean(switch_times)
        performance_monitor.record_metric("avg_provider_switch_time", avg_switch_time)
        
        # Switching should be fast
        assert avg_switch_time < 0.01  # Less than 10ms


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and efficiency"""
    
    def test_memory_usage_scaling(self):
        """Test memory usage scales appropriately with data size"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add documents in batches and measure memory growth
        batch_size = 1000
        num_batches = 5
        memory_usage = [baseline_memory]
        
        for batch in range(num_batches):
            texts = [f"Batch {batch} doc {i}" for i in range(batch_size)]
            embeddings = service.create_embeddings(texts)
            
            service.add_documents_to_collection(
                "memory_test",
                texts,
                embeddings,
                [{"batch": batch} for _ in texts],
                [f"batch_{batch}_doc_{i}" for i in range(batch_size)]
            )
            
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
        
        # Calculate memory growth
        memory_growth = memory_usage[-1] - memory_usage[0]
        docs_added = batch_size * num_batches
        memory_per_doc = memory_growth / docs_added * 1000  # KB per doc
        
        # Memory usage should be reasonable
        assert memory_per_doc < 10  # Less than 10KB per document
    
    def test_memory_cleanup(self):
        """Test memory is properly released after cleanup"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        # Get baseline
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Create large amount of data
        texts = [f"Document {i}" * 100 for i in range(1000)]  # Long documents
        embeddings = service.create_embeddings(texts)
        
        service.add_documents_to_collection(
            "temp_collection",
            texts,
            embeddings,
            [{"id": i} for i in range(len(texts))],
            [f"doc_{i}" for i in range(len(texts))]
        )
        
        # Measure peak memory
        gc.collect()
        peak_memory = process.memory_info().rss / 1024 / 1024
        
        # Clean up
        service.delete_collection("temp_collection")
        service._cleanup_providers()
        del texts
        del embeddings
        
        gc.collect()
        time.sleep(0.1)  # Give time for cleanup
        
        # Measure after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Memory should be released (within 20% of baseline)
        memory_released = (peak_memory - final_memory) / (peak_memory - baseline_memory)
        assert memory_released > 0.8  # At least 80% of memory released


@pytest.mark.performance
class TestPerformanceOptimizations:
    """Test performance optimization features"""
    
    def test_batch_size_optimization(self, performance_monitor):
        """Find optimal batch size for performance"""
        provider = MockEmbeddingProvider(delay=0.001)
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        texts = [f"Document {i}" for i in range(1000)]
        batch_sizes = [10, 25, 50, 100, 200]
        results = {}
        
        for batch_size in batch_sizes:
            service.configure_performance(
                batch_size=batch_size,
                enable_parallel=True,
                max_workers=4
            )
            
            performance_monitor.start()
            embeddings = service.create_embeddings(texts)
            elapsed = performance_monitor.stop()
            
            throughput = len(texts) / elapsed
            results[batch_size] = throughput
            performance_monitor.record_metric(f"throughput_batch_{batch_size}", throughput)
        
        # Find optimal batch size
        optimal_batch = max(results.items(), key=lambda x: x[1])[0]
        
        # Optimal should not be the extremes
        assert optimal_batch not in [min(batch_sizes), max(batch_sizes)]
    
    def test_worker_count_optimization(self, performance_monitor):
        """Find optimal worker count for parallel processing"""
        provider = MockEmbeddingProvider(delay=0.001)
        service = EmbeddingsService(vector_store=InMemoryStore())
        service.add_provider("test", provider)
        
        texts = [f"Document {i}" for i in range(500)]
        worker_counts = [1, 2, 4, 8, 16]
        results = {}
        
        for workers in worker_counts:
            service.configure_performance(
                max_workers=workers,
                batch_size=50,
                enable_parallel=True
            )
            
            # Close existing executor to apply new settings
            service._close_executor()
            
            performance_monitor.start()
            embeddings = service.create_embeddings(texts)
            elapsed = performance_monitor.stop()
            
            throughput = len(texts) / elapsed
            results[workers] = throughput
            performance_monitor.record_metric(f"throughput_workers_{workers}", throughput)
        
        # More workers should improve performance up to a point
        assert results[4] > results[1]
        assert results[8] >= results[4]  # Might plateau


def test_performance_summary(performance_monitor):
    """Generate performance summary report"""
    # This would run after all performance tests
    # and generate a summary of all metrics collected
    
    if hasattr(performance_monitor, 'metrics') and performance_monitor.metrics:
        print("\n=== Performance Test Summary ===")
        for metric_name, values in performance_monitor.metrics.items():
            avg_value = statistics.mean(values)
            print(f"{metric_name}: {avg_value:.3f}")
        print("==============================\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])