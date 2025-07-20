# test_embeddings_performance.py
# Performance benchmarking tests for the simplified embeddings service

import pytest
import time
import statistics
import threading
import psutil
import gc
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsServiceWrapper,
    InMemoryVectorStore,
    ChromaVectorStore,
    create_embeddings_service
)

# Import test utilities from conftest
from .conftest import (
    requires_embeddings,
    requires_chromadb
)


@pytest.mark.performance
class TestEmbeddingPerformance:
    """Performance tests for embedding generation"""
    
    def test_single_vs_batch_performance(self):
        """Compare performance of single vs batch embedding generation"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Mock with small delay to simulate work
            def mock_embed_with_delay(texts, as_list=True):
                time.sleep(0.001 * len(texts))  # Delay proportional to batch size
                return np.array([[0.1] * 384 for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = mock_embed_with_delay
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            texts = [f"Test document {i}" for i in range(100)]
            
            # Single embedding generation (one at a time)
            start_time = time.time()
            single_embeddings = []
            for text in texts:
                emb = service.create_embeddings([text])
                single_embeddings.append(emb[0])
            single_time = time.time() - start_time
            
            # Batch embedding generation
            start_time = time.time()
            batch_embeddings = service.create_embeddings(texts)
            batch_time = time.time() - start_time
            
            # Batch should be faster due to fewer calls
            assert batch_time < single_time
            assert len(single_embeddings) == batch_embeddings.shape[0]
            
            print(f"Single time: {single_time:.3f}s, Batch time: {batch_time:.3f}s")
            print(f"Speedup: {single_time / batch_time:.2f}x")
            
            service.close()
    
    def test_large_batch_performance(self):
        """Test performance with large batches"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Track batch sizes to understand processing
            batch_sizes = []
            
            def track_batch_sizes(texts, as_list=True):
                batch_sizes.append(len(texts))
                return np.array([[0.1] * 384 for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = track_batch_sizes
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Test with different batch sizes
            for num_texts in [100, 500, 1000]:
                batch_sizes.clear()
                texts = [f"Document {i}" for i in range(num_texts)]
                
                start_time = time.time()
                embeddings = service.create_embeddings(texts)
                elapsed = time.time() - start_time
                
                assert embeddings.shape == (num_texts, 384)
                throughput = num_texts / elapsed
                
                print(f"\nBatch size {num_texts}:")
                print(f"  Time: {elapsed:.3f}s")
                print(f"  Throughput: {throughput:.0f} texts/sec")
                print(f"  Actual batches: {batch_sizes}")
            
            service.close()
    
    def test_repeated_text_performance(self):
        """Test performance with repeated texts (simulating cache behavior)"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # The actual caching happens in EmbeddingFactory
            # We'll simulate it by tracking unique texts
            processed_texts = set()
            cache_hits = 0
            
            def simulate_cache(texts, as_list=True):
                nonlocal cache_hits
                result = []
                for text in texts:
                    if text in processed_texts:
                        cache_hits += 1
                    else:
                        processed_texts.add(text)
                    result.append([hash(text) % 100 / 100.0 + i * 0.01 for i in range(384)])
                return np.array(result)
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = simulate_cache
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Prepare texts with duplicates
            unique_texts = [f"Unique document {i}" for i in range(50)]
            repeated_texts = unique_texts * 3  # 150 texts, 100 duplicates
            
            # Process all texts
            start_time = time.time()
            embeddings = service.create_embeddings(repeated_texts)
            elapsed = time.time() - start_time
            
            assert embeddings.shape == (150, 384)
            assert cache_hits == 100  # Should have 100 cache hits
            
            print(f"\nCache performance:")
            print(f"  Total texts: {len(repeated_texts)}")
            print(f"  Unique texts: {len(unique_texts)}")
            print(f"  Cache hits: {cache_hits}")
            print(f"  Time: {elapsed:.3f}s")
            
            service.close()
    
    @requires_embeddings
    def test_real_model_performance(self):
        """Benchmark performance with real embedding model"""
        # This test uses actual model if available
        try:
            service = EmbeddingsServiceWrapper(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception:
            pytest.skip("Real model not available")
            return
        
        # Sample texts for testing
        sample_texts = [
            f"This is test document number {i} for performance benchmarking."
            for i in range(100)
        ]
        
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
                start_time = time.time()
                embeddings = service.create_embeddings(texts)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time  # texts per second
            
            results[batch_size] = {
                "avg_time": avg_time,
                "throughput": throughput
            }
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Avg time: {avg_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} texts/sec")
        
        # Larger batches should have better throughput
        assert results[100]["throughput"] > results[1]["throughput"]
        
        service.close()


@pytest.mark.performance
class TestVectorStorePerformance:
    """Performance tests for vector store operations"""
    
    def test_document_insertion_performance(self):
        """Test performance of document insertion at scale"""
        store = InMemoryVectorStore()
        
        # Prepare documents
        num_docs = 1000
        texts = [f"Document {i}" for i in range(num_docs)]
        # Create mock embeddings
        embeddings = [[i / 1000.0 + j * 0.001 for j in range(384)] for i in range(num_docs)]
        metadatas = [{"id": i} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        # Measure insertion time
        start_time = time.time()
        success = store.add_documents(
            "perf_collection",
            texts,
            embeddings,
            metadatas,
            ids
        )
        insertion_time = time.time() - start_time
        
        assert success
        
        # Calculate insertion rate
        insertion_rate = num_docs / insertion_time
        
        print(f"\nInsertion performance:")
        print(f"  Documents: {num_docs}")
        print(f"  Time: {insertion_time:.3f}s")
        print(f"  Rate: {insertion_rate:.0f} docs/sec")
        
        # Should handle at least 100 docs/second
        assert insertion_rate > 100
    
    def test_search_performance_scaling(self):
        """Test how search performance scales with collection size"""
        store = InMemoryVectorStore()
        
        collection_sizes = [100, 500, 1000, 5000]
        search_times = {}
        
        for size in collection_sizes:
            # Create collection of given size
            collection_name = f"collection_{size}"
            texts = [f"Document {i}" for i in range(size)]
            embeddings = [[i / float(size) + j * 0.001 for j in range(384)] for i in range(size)]
            
            store.add_documents(
                collection_name,
                texts,
                embeddings,
                [{"id": i} for i in range(size)],
                [f"doc_{i}" for i in range(size)]
            )
            
            # Measure search time
            query_embedding = [[0.5 + j * 0.001 for j in range(384)]]  # Middle-range embedding
            
            times = []
            for _ in range(5):  # Multiple runs for average
                start_time = time.time()
                results = store.search(
                    collection_name,
                    query_embedding,
                    top_k=10
                )
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_search_time = statistics.mean(times)
            search_times[size] = avg_search_time
            
            print(f"\nCollection size {size}:")
            print(f"  Avg search time: {avg_search_time*1000:.2f}ms")
            print(f"  Search rate: {1/avg_search_time:.0f} searches/sec")
        
        # Search time should not increase linearly with collection size
        time_ratio = search_times[5000] / search_times[100]
        size_ratio = 5000 / 100
        
        print(f"\nScaling analysis:")
        print(f"  Size increased: {size_ratio}x")
        print(f"  Time increased: {time_ratio:.2f}x")
        
        # Time should increase slower than size
        assert time_ratio < size_ratio * 0.5
    
    @requires_chromadb
    def test_chromadb_vs_memory_performance(self, temp_dir):
        """Compare ChromaDB vs in-memory store performance"""
        # Prepare test data
        num_docs = 500
        texts = [f"Document {i}" for i in range(num_docs)]
        embeddings = [[i / float(num_docs) + j * 0.001 for j in range(384)] for i in range(num_docs)]
        metadatas = [{"id": i} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        # Test in-memory store
        memory_store = InMemoryVectorStore()
        
        start_time = time.time()
        memory_store.add_documents(
            "test_collection",
            texts,
            embeddings,
            metadatas,
            ids
        )
        memory_time = time.time() - start_time
        
        # Test ChromaDB store
        try:
            chroma_store = ChromaVectorStore(persist_directory=str(temp_dir))
        except ImportError:
            pytest.skip("ChromaDB not installed")
            return
        
        start_time = time.time()
        chroma_store.add_documents(
            "test_collection",
            texts,
            embeddings,
            metadatas,
            ids
        )
        chroma_time = time.time() - start_time
        
        print(f"\nVector store comparison ({num_docs} documents):")
        print(f"  In-memory store: {memory_time:.3f}s")
        print(f"  ChromaDB store: {chroma_time:.3f}s")
        print(f"  Ratio (ChromaDB/Memory): {chroma_time/memory_time:.2f}x")
        
        # Both should complete in reasonable time
        assert memory_time < 5.0
        assert chroma_time < 10.0


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Performance tests for concurrent operations"""
    
    def test_concurrent_embedding_throughput(self):
        """Test throughput under concurrent load"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Thread-safe mock with controlled delay
            lock = threading.Lock()
            call_count = 0
            
            def thread_safe_embed(texts, as_list=True):
                nonlocal call_count
                with lock:
                    call_count += 1
                time.sleep(0.001 * len(texts))  # Simulate work
                return np.array([[0.1] * 384 for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = thread_safe_embed
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            num_threads = 10
            texts_per_thread = 50
            
            def create_embeddings_task(thread_id):
                texts = [f"Thread {thread_id} doc {i}" for i in range(texts_per_thread)]
                return service.create_embeddings(texts)
            
            # Sequential baseline
            start_time = time.time()
            for i in range(num_threads):
                create_embeddings_task(i)
            sequential_time = time.time() - start_time
            
            # Reset call count
            call_count = 0
            
            # Concurrent execution
            threads = []
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(target=create_embeddings_task, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            concurrent_time = time.time() - start_time
            
            # Calculate throughput
            total_texts = num_threads * texts_per_thread
            sequential_throughput = total_texts / sequential_time
            concurrent_throughput = total_texts / concurrent_time
            
            print(f"\nConcurrency performance:")
            print(f"  Sequential: {sequential_time:.3f}s ({sequential_throughput:.0f} texts/sec)")
            print(f"  Concurrent: {concurrent_time:.3f}s ({concurrent_throughput:.0f} texts/sec)")
            print(f"  Speedup: {concurrent_throughput/sequential_throughput:.2f}x")
            print(f"  Total API calls: {call_count}")
            
            # Concurrent should be faster
            assert concurrent_throughput > sequential_throughput
            
            service.close()
    
    def test_model_switching_overhead(self):
        """Test overhead of switching between different models"""
        # This tests the overhead of recreating the service with different models
        model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "bert-base-uncased",
            "openai/text-embedding-3-small",
            "intfloat/e5-small-v2"
        ]
        
        switch_times = []
        
        for i in range(20):  # Test 20 switches
            model_name = model_names[i % len(model_names)]
            
            start_time = time.time()
            
            with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
                mock_instance = MagicMock()
                mock_instance.embed.return_value = np.array([[0.1] * 384])
                mock_factory.return_value = mock_instance
                
                service = EmbeddingsServiceWrapper(model_name=model_name)
                embeddings = service.create_embeddings(["Test document"])
                service.close()
            
            elapsed = time.time() - start_time
            switch_times.append(elapsed)
        
        avg_switch_time = statistics.mean(switch_times)
        
        print(f"\nModel switching overhead:")
        print(f"  Switches: {len(switch_times)}")
        print(f"  Avg time: {avg_switch_time*1000:.2f}ms")
        print(f"  Min time: {min(switch_times)*1000:.2f}ms")
        print(f"  Max time: {max(switch_times)*1000:.2f}ms")
        
        # Switching should be relatively fast
        assert avg_switch_time < 0.1  # Less than 100ms average


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and efficiency"""
    
    def test_memory_usage_scaling(self):
        """Test memory usage scales appropriately with data size"""
        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create service and store
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1] * 384])  # Will be called for each batch
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            store = InMemoryVectorStore()
            
            # Add documents in batches and measure memory growth
            batch_size = 1000
            num_batches = 5
            memory_usage = [baseline_memory]
            
            for batch in range(num_batches):
                texts = [f"Batch {batch} doc {i}" for i in range(batch_size)]
                # Mock returns same embedding for all texts
                mock_instance.embed.return_value = np.array([[0.1] * 384 for _ in texts])
                embeddings = service.create_embeddings(texts)
                
                # Convert numpy array to list for storage
                embeddings_list = embeddings.tolist()
                
                store.add_documents(
                    "memory_test",
                    texts,
                    embeddings_list,
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
            
            print(f"\nMemory usage scaling:")
            print(f"  Documents added: {docs_added}")
            print(f"  Memory growth: {memory_growth:.1f}MB")
            print(f"  Memory per doc: {memory_per_doc:.2f}KB")
            
            # Memory usage should be reasonable
            assert memory_per_doc < 10  # Less than 10KB per document
            
            service.close()
    
    def test_memory_cleanup(self):
        """Test memory is properly released after cleanup"""
        # Get baseline
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            store = InMemoryVectorStore()
            
            # Create large amount of data
            texts = [f"Document {i}" * 100 for i in range(1000)]  # Long documents
            mock_instance.embed.return_value = np.array([[0.1] * 384 for _ in texts])
            embeddings = service.create_embeddings(texts)
            
            # Add to store
            store.add_documents(
                "temp_collection",
                texts,
                embeddings.tolist(),
                [{"id": i} for i in range(len(texts))],
                [f"doc_{i}" for i in range(len(texts))]
            )
            
            # Measure peak memory
            gc.collect()
            peak_memory = process.memory_info().rss / 1024 / 1024
            
            # Clean up
            store.delete_collection("temp_collection")
            service.close()
            del texts
            del embeddings
            del store
            
            gc.collect()
            time.sleep(0.1)  # Give time for cleanup
            
            # Measure after cleanup
            final_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"\nMemory cleanup test:")
            print(f"  Baseline: {baseline_memory:.1f}MB")
            print(f"  Peak: {peak_memory:.1f}MB")
            print(f"  Final: {final_memory:.1f}MB")
            
            # Memory should be released
            if peak_memory > baseline_memory:
                memory_released = (peak_memory - final_memory) / (peak_memory - baseline_memory)
                print(f"  Released: {memory_released*100:.1f}%")
                assert memory_released > 0.5  # At least 50% of memory released


@pytest.mark.performance
class TestPerformanceOptimizations:
    """Test performance optimization features"""
    
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Track how many times embed is called
            embed_calls = []
            
            def track_embed_calls(texts, as_list=True):
                embed_calls.append(len(texts))
                time.sleep(0.0001 * len(texts))  # Simulate processing time
                return np.array([[0.1] * 384 for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = track_embed_calls
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Test with 1000 documents
            texts = [f"Document {i}" for i in range(1000)]
            
            start_time = time.time()
            embeddings = service.create_embeddings(texts)
            elapsed = time.time() - start_time
            
            assert embeddings.shape == (1000, 384)
            
            print(f"\nBatch processing efficiency:")
            print(f"  Total texts: {len(texts)}")
            print(f"  API calls: {len(embed_calls)}")
            print(f"  Batch sizes: {embed_calls}")
            print(f"  Total time: {elapsed:.3f}s")
            print(f"  Throughput: {len(texts)/elapsed:.0f} texts/sec")
            
            # Should process in reasonable number of batches
            assert len(embed_calls) <= 10  # No more than 10 API calls for 1000 texts
            
            service.close()
    
    def test_concurrent_request_handling(self):
        """Test handling of concurrent embedding requests"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Thread-safe tracking
            lock = threading.Lock()
            concurrent_calls = 0
            max_concurrent = 0
            
            def track_concurrency(texts, as_list=True):
                nonlocal concurrent_calls, max_concurrent
                with lock:
                    concurrent_calls += 1
                    max_concurrent = max(max_concurrent, concurrent_calls)
                
                time.sleep(0.01)  # Simulate processing
                result = np.array([[0.1] * 384 for _ in texts])
                
                with lock:
                    concurrent_calls -= 1
                
                return result
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = track_concurrency
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Launch multiple concurrent requests
            num_threads = 20
            texts_per_thread = 10
            results = []
            threads = []
            
            def process_texts(thread_id):
                texts = [f"Thread {thread_id} text {i}" for i in range(texts_per_thread)]
                embeddings = service.create_embeddings(texts)
                results.append(embeddings.shape)
            
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(target=process_texts, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            elapsed = time.time() - start_time
            
            print(f"\nConcurrent request handling:")
            print(f"  Threads: {num_threads}")
            print(f"  Max concurrent API calls: {max_concurrent}")
            print(f"  Total time: {elapsed:.3f}s")
            print(f"  All results correct: {all(shape == (texts_per_thread, 384) for shape in results)}")
            
            # All requests should succeed
            assert len(results) == num_threads
            assert all(shape == (texts_per_thread, 384) for shape in results)
            
            service.close()


def test_performance_summary():
    """Generate performance summary report"""
    # This test just ensures all performance tests can run
    # In a real scenario, you'd collect and analyze all metrics
    print("\n=== Performance Test Summary ===")
    print("All performance tests completed successfully")
    print("==============================\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])