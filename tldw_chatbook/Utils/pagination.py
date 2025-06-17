"""
Pagination utilities for handling large datasets efficiently.
"""
from typing import List, Dict, Any, Callable, TypeVar, Generic, AsyncIterator
import asyncio
from loguru import logger

T = TypeVar('T')


class PaginatedResult(Generic[T]):
    """Container for paginated results."""
    
    def __init__(self, items: List[T], total_count: int, page: int, page_size: int):
        self.items = items
        self.total_count = total_count
        self.page = page
        self.page_size = page_size
        self.total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
        self.has_next = page < self.total_pages
        self.has_previous = page > 1


async def paginated_fetch(
    fetch_func: Callable[[int, int], List[T]], 
    page_size: int = 100,
    max_items: int = 1000,
    status_callback: Callable[[str], Any] = None
) -> List[T]:
    """
    Fetch data in paginated chunks to avoid loading too much at once.
    
    Args:
        fetch_func: Function that takes (limit, offset) and returns items
        page_size: Number of items per page
        max_items: Maximum total items to fetch
        status_callback: Optional callback for status updates
        
    Returns:
        List of all fetched items
    """
    all_items = []
    offset = 0
    page = 1
    
    while len(all_items) < max_items:
        current_limit = min(page_size, max_items - len(all_items))
        
        if status_callback:
            result = status_callback(f"⏳ Loading page {page} ({len(all_items)} items loaded so far)...")
            if asyncio.iscoroutine(result):
                await result
        
        try:
            # Fetch current page
            items = await asyncio.to_thread(fetch_func, current_limit, offset)
            
            if not items:
                # No more items available
                break
                
            all_items.extend(items)
            offset += len(items)
            page += 1
            
            # If we got fewer items than requested, we've reached the end
            if len(items) < current_limit:
                break
                
        except Exception as e:
            logger.error(f"Error during paginated fetch at page {page}: {e}")
            break
    
    if status_callback:
        result = status_callback(f"✅ Loaded {len(all_items)} items total")
        if asyncio.iscoroutine(result):
            await result
    
    return all_items


async def paginated_fetch_async(
    fetch_func: Callable[[int, int], Any],
    page_size: int = 100,
    max_items: int = 1000,
    status_callback: Callable[[str], None] = None
) -> List[Any]:
    """
    Async version of paginated fetch for functions that are already async.
    
    Args:
        fetch_func: Async function that takes (limit, offset) and returns items
        page_size: Number of items per page
        max_items: Maximum total items to fetch
        status_callback: Optional callback for status updates
        
    Returns:
        List of all fetched items
    """
    all_items = []
    offset = 0
    page = 1
    
    while len(all_items) < max_items:
        current_limit = min(page_size, max_items - len(all_items))
        
        if status_callback:
            result = status_callback(f"⏳ Loading page {page} ({len(all_items)} items loaded so far)...")
            if asyncio.iscoroutine(result):
                await result
        
        try:
            # Fetch current page
            items = await fetch_func(current_limit, offset)
            
            if not items:
                # No more items available
                break
                
            all_items.extend(items)
            offset += len(items)
            page += 1
            
            # If we got fewer items than requested, we've reached the end
            if len(items) < current_limit:
                break
                
        except Exception as e:
            logger.error(f"Error during async paginated fetch at page {page}: {e}")
            break
    
    if status_callback:
        result = status_callback(f"✅ Loaded {len(all_items)} items total")
        if asyncio.iscoroutine(result):
            await result
    
    return all_items


class LazyPaginator:
    """
    Lazy paginator that fetches data only when needed.
    Useful for UI components that show items progressively.
    """
    
    def __init__(self, fetch_func: Callable[[int, int], List[Any]], page_size: int = 100):
        self.fetch_func = fetch_func
        self.page_size = page_size
        self.cached_pages: Dict[int, List[Any]] = {}
        self.total_count: int = None
        self.current_page = 1
        
    async def get_page(self, page: int) -> PaginatedResult:
        """Get a specific page of results."""
        if page in self.cached_pages:
            items = self.cached_pages[page]
        else:
            offset = (page - 1) * self.page_size
            items = await asyncio.to_thread(self.fetch_func, self.page_size, offset)
            self.cached_pages[page] = items
        
        # Estimate total count if not known
        if self.total_count is None:
            if len(items) < self.page_size:
                # This page is not full, so we can estimate total
                self.total_count = (page - 1) * self.page_size + len(items)
            else:
                # We don't know the total yet, estimate conservatively
                self.total_count = page * self.page_size + 1
        
        return PaginatedResult(items, self.total_count, page, self.page_size)
    
    async def get_next_page(self) -> PaginatedResult:
        """Get the next page of results."""
        result = await self.get_page(self.current_page)
        if result.has_next:
            self.current_page += 1
        return result
    
    async def get_all_items(self, max_items: int = 1000, 
                           status_callback: Callable[[str], None] = None) -> List[Any]:
        """
        Get all items up to max_items using lazy pagination.
        """
        all_items = []
        page = 1
        
        while len(all_items) < max_items:
            if status_callback:
                status_callback(f"⏳ Loading page {page} ({len(all_items)} items so far)...")
            
            result = await self.get_page(page)
            
            if not result.items:
                break
                
            # Add items up to our limit
            remaining_capacity = max_items - len(all_items)
            items_to_add = result.items[:remaining_capacity]
            all_items.extend(items_to_add)
            
            if len(result.items) < self.page_size or len(all_items) >= max_items:
                break
                
            page += 1
        
        if status_callback:
            status_callback(f"✅ Loaded {len(all_items)} items total")
            
        return all_items