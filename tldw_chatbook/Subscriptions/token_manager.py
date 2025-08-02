# token_manager.py
# Description: Token counting and budget management for LLM processing
#
# This module provides accurate token counting and intelligent budget
# allocation across multiple content items.
#
# Imports
import re
from typing import List, Dict, Any, Optional, Tuple
import heapq
#
# Third-Party Imports
from loguru import logger
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None
#
# Local Imports
from ..Metrics.metrics_logger import log_counter
#
########################################################################################################################
#
# Token Counting
#
########################################################################################################################

class TokenCounter:
    """
    Accurate token counting for various LLM models.
    
    Uses tiktoken when available, falls back to approximation.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for token counting
        """
        self.model = model
        self.encoder = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for the model
                self.encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base for newer models
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                except:
                    logger.warning(f"Could not load tiktoken encoding for {model}, using approximation")
        else:
            logger.info("tiktoken not available, using approximation for token counting")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.encoder:
            # Use tiktoken for accurate counting
            try:
                tokens = self.encoder.encode(text)
                return len(tokens)
            except Exception as e:
                logger.error(f"Error encoding text: {str(e)}")
                # Fall through to approximation
        
        # Approximation: ~4 characters per token for English
        # This is a rough estimate that works reasonably well
        return self._approximate_tokens(text)
    
    def _approximate_tokens(self, text: str) -> int:
        """Approximate token count when tiktoken not available."""
        # Basic approximation rules:
        # - Average ~4 characters per token
        # - Adjust for whitespace and punctuation
        
        # Clean text
        text = text.strip()
        if not text:
            return 0
        
        # Count words (rough approximation)
        words = len(text.split())
        
        # Count special characters that often become separate tokens
        special_chars = len(re.findall(r'[^\w\s]', text))
        
        # Estimate tokens
        # Most words = 1 token, but longer words might be 2-3 tokens
        char_estimate = len(text) / 4
        word_estimate = words * 1.3  # Account for some multi-token words
        
        # Take average of estimates
        estimate = int((char_estimate + word_estimate) / 2)
        
        # Add special characters
        estimate += special_chars // 3
        
        return max(estimate, 1)
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: List of texts
            
        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Binary search for the right truncation point
        left, right = 0, len(text)
        result = ""
        
        while left < right:
            mid = (left + right + 1) // 2
            truncated = text[:mid]
            tokens = self.count_tokens(truncated)
            
            if tokens <= max_tokens:
                result = truncated
                left = mid
            else:
                right = mid - 1
        
        # Try to truncate at sentence boundary
        if result:
            last_period = result.rfind('.')
            last_newline = result.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > len(result) * 0.8:  # If we're not losing too much
                result = result[:boundary + 1]
            else:
                result = result.rstrip() + '...'
        
        return result


########################################################################################################################
#
# Token Budget Management
#
########################################################################################################################

class TokenBudgetManager:
    """
    Manage token allocation across multiple items.
    
    Implements sophisticated algorithms for fair and optimal token distribution
    based on priorities, content length, and constraints.
    """
    
    def __init__(self, total_budget: int):
        """
        Initialize budget manager.
        
        Args:
            total_budget: Total tokens available
        """
        self.total_budget = total_budget
        self.allocations = {}
        self.remaining_budget = total_budget
    
    def allocate_by_priority(self, items: List[Dict[str, Any]], 
                           min_per_item: int = 50,
                           max_per_item: int = 5000) -> Dict[Any, int]:
        """
        Allocate tokens based on item priorities.
        
        Args:
            items: List of dicts with 'id', 'priority', and 'current_tokens'
            min_per_item: Minimum tokens per item
            max_per_item: Maximum tokens per item
            
        Returns:
            Dict mapping item IDs to allocated tokens
        """
        if not items:
            return {}
        
        # Ensure we have enough budget for minimums
        total_min_needed = len(items) * min_per_item
        if total_min_needed > self.total_budget:
            # Distribute evenly if we can't meet minimums
            equal_share = self.total_budget // len(items)
            return {item['id']: equal_share for item in items}
        
        # Start with minimum allocation
        allocations = {item['id']: min_per_item for item in items}
        remaining_budget = self.total_budget - total_min_needed
        
        # Calculate priority weights
        total_priority = sum(item.get('priority', 3) for item in items)
        if total_priority == 0:
            total_priority = len(items) * 3  # Default priority
        
        # Allocate remaining budget based on priority
        for item in items:
            item_id = item['id']
            priority = item.get('priority', 3)
            current_tokens = item.get('current_tokens', 0)
            
            # Calculate this item's share of remaining budget
            priority_share = int(remaining_budget * (priority / total_priority))
            
            # Total allocation for this item
            total_allocation = allocations[item_id] + priority_share
            
            # Apply constraints
            total_allocation = min(total_allocation, max_per_item)
            total_allocation = min(total_allocation, current_tokens)  # Don't allocate more than needed
            
            allocations[item_id] = total_allocation
        
        # Redistribute any unused budget
        self._redistribute_unused_budget(allocations, items, max_per_item)
        
        log_counter("token_allocation_performed", labels={
            "item_count": str(len(items)),
            "total_budget": str(self.total_budget)
        })
        
        return allocations
    
    def allocate_to_items(self, items: List[Dict[str, Any]],
                         total_budget: int,
                         min_per_item: int = 50,
                         max_per_item: int = 5000) -> Dict[Any, int]:
        """
        Allocate specific budget to items.
        
        Args:
            items: List of dicts with item data
            total_budget: Budget for these items
            min_per_item: Minimum tokens per item
            max_per_item: Maximum tokens per item
            
        Returns:
            Dict mapping items to allocated tokens
        """
        # Create temporary manager for this allocation
        temp_manager = TokenBudgetManager(total_budget)
        
        # Convert items to expected format
        formatted_items = []
        for item in items:
            if isinstance(item, dict) and 'id' in item:
                formatted_items.append(item)
            else:
                # Assume item is the ID itself
                formatted_items.append({
                    'id': item,
                    'priority': getattr(item, 'priority', 3) if hasattr(item, 'priority') else 3,
                    'current_tokens': getattr(item, 'token_count', 1000) if hasattr(item, 'token_count') else 1000
                })
        
        return temp_manager.allocate_by_priority(formatted_items, min_per_item, max_per_item)
    
    def allocate_proportional(self, items: List[Dict[str, Any]],
                            weights: Optional[Dict[Any, float]] = None) -> Dict[Any, int]:
        """
        Allocate tokens proportionally based on weights.
        
        Args:
            items: List of items with IDs
            weights: Optional dict of item weights
            
        Returns:
            Dict mapping item IDs to allocated tokens
        """
        if not items:
            return {}
        
        # Use equal weights if not provided
        if not weights:
            weights = {item['id']: 1.0 for item in items}
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = len(items)
        
        allocations = {}
        for item in items:
            item_id = item['id']
            weight = weights.get(item_id, 1.0)
            allocation = int(self.total_budget * (weight / total_weight))
            allocations[item_id] = allocation
        
        return allocations
    
    def _redistribute_unused_budget(self, allocations: Dict[Any, int],
                                  items: List[Dict[str, Any]],
                                  max_per_item: int):
        """Redistribute any budget not used due to constraints."""
        # Calculate unused budget
        total_allocated = sum(allocations.values())
        unused = self.total_budget - total_allocated
        
        if unused <= 0:
            return
        
        # Create priority queue of items that could use more tokens
        candidates = []
        for item in items:
            item_id = item['id']
            current_allocation = allocations[item_id]
            current_tokens = item.get('current_tokens', max_per_item)
            
            # Can this item use more tokens?
            if current_allocation < min(current_tokens, max_per_item):
                # Priority queue: negative priority for max heap
                heapq.heappush(candidates, (-item.get('priority', 3), item_id, item))
        
        # Redistribute to highest priority items first
        while unused > 0 and candidates:
            _, item_id, item = heapq.heappop(candidates)
            
            current_allocation = allocations[item_id]
            current_tokens = item.get('current_tokens', max_per_item)
            max_additional = min(current_tokens, max_per_item) - current_allocation
            
            if max_additional > 0:
                additional = min(unused, max_additional)
                allocations[item_id] += additional
                unused -= additional
    
    def get_allocation_stats(self, allocations: Dict[Any, int]) -> Dict[str, Any]:
        """Get statistics about the allocation."""
        if not allocations:
            return {
                'total_allocated': 0,
                'num_items': 0,
                'avg_allocation': 0,
                'min_allocation': 0,
                'max_allocation': 0,
                'utilization': 0.0
            }
        
        values = list(allocations.values())
        total = sum(values)
        
        return {
            'total_allocated': total,
            'num_items': len(allocations),
            'avg_allocation': total / len(allocations),
            'min_allocation': min(values),
            'max_allocation': max(values),
            'utilization': total / self.total_budget if self.total_budget > 0 else 0.0
        }


class TokenBudgetTracker:
    """Track token usage across operations."""
    
    def __init__(self, total_budget: int):
        """Initialize tracker."""
        self.total_budget = total_budget
        self.used_tokens = 0
        self.operations = []
    
    def use_tokens(self, amount: int, operation: str = "unknown"):
        """Record token usage."""
        self.used_tokens += amount
        self.operations.append({
            'operation': operation,
            'tokens': amount,
            'remaining': self.remaining_tokens
        })
        
        if self.used_tokens > self.total_budget:
            logger.warning(f"Token budget exceeded: {self.used_tokens}/{self.total_budget}")
    
    @property
    def remaining_tokens(self) -> int:
        """Get remaining tokens."""
        return max(0, self.total_budget - self.used_tokens)
    
    @property
    def usage_percentage(self) -> float:
        """Get usage percentage."""
        if self.total_budget == 0:
            return 0.0
        return (self.used_tokens / self.total_budget) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        operation_totals = {}
        for op in self.operations:
            operation_totals[op['operation']] = operation_totals.get(op['operation'], 0) + op['tokens']
        
        return {
            'total_budget': self.total_budget,
            'used_tokens': self.used_tokens,
            'remaining_tokens': self.remaining_tokens,
            'usage_percentage': self.usage_percentage,
            'operation_count': len(self.operations),
            'operation_totals': operation_totals
        }


# End of token_manager.py