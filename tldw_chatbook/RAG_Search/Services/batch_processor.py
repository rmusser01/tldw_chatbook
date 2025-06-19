# batch_processor.py
# Description: Batch processing utilities for RAG performance optimization
#
# Imports
from typing import List, Dict, Any, Callable, Optional, TypeVar, Generic
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import time
from loguru import logger

logger = logger.bind(module="batch_processor")

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class BatchResult(Generic[R]):
    """Container for batch processing results"""
    successful: List[R]
    failed: List[Dict[str, Any]]
    total_time: float
    items_per_second: float

class BatchProcessor:
    """
    Utility class for efficient batch processing of RAG operations
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the batch processor
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = None  # Created on demand
    
    async def process_batch_async(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult[R]:
        """
        Process items in batches asynchronously
        
        Args:
            items: List of items to process
            process_func: Async function to process each item
            batch_size: Number of items to process concurrently
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult containing successful results and failures
        """
        start_time = time.time()
        successful_results = []
        failed_items = []
        total_items = len(items)
        
        # Process in batches
        for i in range(0, total_items, batch_size):
            batch = items[i:i + batch_size]
            batch_start = time.time()
            
            # Create tasks for batch
            tasks = [asyncio.create_task(self._process_item_async(item, process_func, idx + i)) 
                    for idx, item in enumerate(batch)]
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successes and failures
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_items.append({
                        'item': batch[idx],
                        'error': str(result),
                        'index': i + idx
                    })
                else:
                    successful_results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(min(i + batch_size, total_items), total_items)
            
            # Log batch completion
            batch_time = time.time() - batch_start
            logger.debug(f"Processed batch {i//batch_size + 1}: "
                        f"{len(batch)} items in {batch_time:.2f}s")
        
        # Calculate statistics
        total_time = time.time() - start_time
        items_per_second = total_items / total_time if total_time > 0 else 0
        
        logger.info(f"Batch processing complete: {len(successful_results)} successful, "
                   f"{len(failed_items)} failed, {items_per_second:.2f} items/sec")
        
        return BatchResult(
            successful=successful_results,
            failed=failed_items,
            total_time=total_time,
            items_per_second=items_per_second
        )
    
    async def _process_item_async(self, item: T, process_func: Callable, index: int) -> R:
        """Process a single item with error handling"""
        try:
            if asyncio.iscoroutinefunction(process_func):
                return await process_func(item)
            else:
                # Run sync function in thread executor
                return await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor, process_func, item
                )
        except Exception as e:
            logger.error(f"Error processing item {index}: {e}")
            raise
    
    def process_batch_threaded(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult[R]:
        """
        Process items using thread pool
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Number of items to process concurrently
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult containing successful results and failures
        """
        start_time = time.time()
        successful_results = []
        failed_items = []
        total_items = len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process in batches
            for i in range(0, total_items, batch_size):
                batch = items[i:i + batch_size]
                
                # Submit batch to executor
                futures = {executor.submit(process_func, item): (idx + i, item) 
                          for idx, item in enumerate(batch)}
                
                # Collect results
                for future in futures:
                    idx, item = futures[future]
                    try:
                        result = future.result()
                        successful_results.append(result)
                    except Exception as e:
                        failed_items.append({
                            'item': item,
                            'error': str(e),
                            'index': idx
                        })
                
                # Progress callback
                if progress_callback:
                    progress_callback(min(i + batch_size, total_items), total_items)
        
        # Calculate statistics
        total_time = time.time() - start_time
        items_per_second = total_items / total_time if total_time > 0 else 0
        
        return BatchResult(
            successful=successful_results,
            failed=failed_items,
            total_time=total_time,
            items_per_second=items_per_second
        )
    
    def chunk_list(self, items: List[T], chunk_size: int) -> List[List[T]]:
        """Split a list into chunks of specified size"""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    async def map_reduce_async(
        self,
        items: List[T],
        map_func: Callable[[T], Any],
        reduce_func: Callable[[List[Any]], R],
        chunk_size: int = 100
    ) -> R:
        """
        Map-reduce pattern for batch processing
        
        Args:
            items: Items to process
            map_func: Function to apply to each item
            reduce_func: Function to combine results
            chunk_size: Size of processing chunks
            
        Returns:
            Reduced result
        """
        # Map phase
        chunks = self.chunk_list(items, chunk_size)
        chunk_results = []
        
        for chunk in chunks:
            # Process chunk
            tasks = [asyncio.create_task(self._process_item_async(item, map_func, i)) 
                    for i, item in enumerate(chunk)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if not isinstance(r, Exception)]
            chunk_results.extend(valid_results)
        
        # Reduce phase
        return reduce_func(chunk_results)
    
    def shutdown(self):
        """Shutdown executors"""
        self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)

# Utility functions for common batch operations

async def batch_chunk_documents(
    documents: List[Dict[str, Any]],
    chunking_service,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    batch_size: int = 50
) -> List[List[Dict[str, Any]]]:
    """
    Batch process document chunking
    
    Args:
        documents: List of documents to chunk
        chunking_service: ChunkingService instance
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        batch_size: Number of documents to process concurrently
        
    Returns:
        List of chunk lists for each document
    """
    processor = BatchProcessor()
    
    def chunk_document(doc):
        return chunking_service.chunk_document(doc, chunk_size, chunk_overlap)
    
    result = await processor.process_batch_async(
        documents,
        chunk_document,
        batch_size=batch_size
    )
    
    processor.shutdown()
    return result.successful

async def batch_create_embeddings(
    texts: List[str],
    embeddings_service,
    batch_size: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Batch create embeddings for texts
    
    Args:
        texts: List of texts to embed
        embeddings_service: EmbeddingsService instance
        batch_size: Number of texts to process at once
        progress_callback: Optional progress callback
        
    Returns:
        List of embeddings
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embeddings_service.create_embeddings(batch)
        
        if embeddings:
            all_embeddings.extend(embeddings)
        else:
            # Fill with None for failed embeddings
            all_embeddings.extend([None] * len(batch))
        
        if progress_callback:
            progress_callback(min(i + batch_size, len(texts)), len(texts))
    
    return all_embeddings