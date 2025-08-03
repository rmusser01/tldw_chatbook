# aggregation_engine.py
# Description: Content aggregation engine with intelligent token management
#
# This module provides sophisticated content aggregation from multiple sources,
# managing token budgets for LLM processing, and organizing content by priority
# and relevance.
#
# Imports
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
import heapq
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Metrics.metrics_logger import log_histogram, log_counter
from .token_manager import TokenBudgetManager, TokenCounter
from .recursive_summarizer import RecursiveSummarizer, SummarizationConfig
#
########################################################################################################################
#
# Data Classes
#
########################################################################################################################

@dataclass
class AggregatedItem:
    """Represents an item ready for aggregation."""
    source_id: int  # subscription_id
    source_name: str
    url: str
    title: str
    content: str
    summary: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    categories: List[str] = field(default_factory=list)
    priority: int = 3  # 1-5, inherited from subscription
    token_count: int = 0
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority, then relevance, then date."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        if self.relevance_score != other.relevance_score:
            return self.relevance_score > other.relevance_score
        if self.published_date and other.published_date:
            return self.published_date > other.published_date
        return False


@dataclass
class AggregationConfig:
    """Configuration for content aggregation."""
    total_token_budget: int = 50000
    min_tokens_per_item: int = 100
    max_tokens_per_item: int = 5000
    summary_token_ratio: float = 0.2  # Summary should be 20% of original
    
    # Section allocations (percentages of total budget)
    executive_summary_allocation: float = 0.1
    section_headers_allocation: float = 0.05
    transitions_allocation: float = 0.05
    content_allocation: float = 0.8
    
    # Grouping options
    group_by: str = 'source'  # 'source', 'category', 'date', 'priority'
    sort_within_groups: str = 'relevance'  # 'relevance', 'date', 'priority'
    
    # Content filters
    min_relevance_score: float = 0.0
    date_range_days: Optional[int] = None
    include_categories: Optional[List[str]] = None
    exclude_categories: Optional[List[str]] = None
    
    # Output options
    include_metadata: bool = True
    include_sources: bool = True
    include_timestamps: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AggregationConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class AggregatedContent:
    """Result of content aggregation."""
    sections: List['ContentSection']
    executive_summary: Optional[str] = None
    total_items: int = 0
    total_sources: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'executive_summary': self.executive_summary,
            'sections': [s.to_dict() for s in self.sections],
            'total_items': self.total_items,
            'total_sources': self.total_sources,
            'total_tokens': self.total_tokens,
            'metadata': self.metadata,
            'generated_at': self.generated_at.isoformat()
        }


@dataclass
class ContentSection:
    """A section within aggregated content."""
    title: str
    items: List[AggregatedItem]
    summary: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'summary': self.summary,
            'item_count': len(self.items),
            'token_count': self.token_count,
            'items': [
                {
                    'title': item.title,
                    'url': item.url,
                    'summary': item.summary or item.content[:200] + '...',
                    'source': item.source_name,
                    'author': item.author,
                    'published': item.published_date.isoformat() if item.published_date else None,
                    'categories': item.categories
                }
                for item in self.items
            ]
        }


########################################################################################################################
#
# Aggregation Engine
#
########################################################################################################################

class AggregationEngine:
    """
    Intelligent content aggregation with token budget management.
    
    This engine:
    - Groups content by configurable criteria
    - Allocates tokens based on priority and relevance
    - Manages recursive summarization when needed
    - Generates structured output for briefings
    """
    
    def __init__(self, db: SubscriptionsDB, config: Optional[AggregationConfig] = None):
        """
        Initialize aggregation engine.
        
        Args:
            db: Subscriptions database
            config: Aggregation configuration
        """
        self.db = db
        self.config = config or AggregationConfig()
        self.token_manager = TokenBudgetManager(self.config.total_token_budget)
        self.token_counter = TokenCounter()
        
        # Initialize recursive summarizer with configuration
        summarization_config = SummarizationConfig(
            summary_ratio=self.config.summary_token_ratio,
            preserve_structure=True,
            style='balanced',
            format='prose',
            use_fallback=True
        )
        self.recursive_summarizer = RecursiveSummarizer(summarization_config)
    
    async def aggregate_items(self, items: List[Dict[str, Any]], 
                            subscriptions: Optional[Dict[int, Dict[str, Any]]] = None) -> AggregatedContent:
        """
        Aggregate items into structured content.
        
        Args:
            items: List of subscription items to aggregate
            subscriptions: Optional subscription metadata dict
            
        Returns:
            Aggregated content with sections and summaries
        """
        start_time = datetime.now(timezone.utc)
        
        # Convert to AggregatedItem objects
        aggregated_items = self._prepare_items(items, subscriptions)
        
        # Apply filters
        filtered_items = self._apply_filters(aggregated_items)
        
        # Calculate relevance scores if needed
        if self.config.sort_within_groups == 'relevance':
            filtered_items = await self._calculate_relevance_scores(filtered_items)
        
        # Group items
        grouped_items = self._group_items(filtered_items)
        
        # Allocate token budgets
        section_budgets = self._allocate_section_budgets(grouped_items)
        
        # Create sections with token management
        sections = []
        for group_key, items in grouped_items.items():
            section = await self._create_section(
                group_key, 
                items, 
                section_budgets.get(group_key, 1000)
            )
            sections.append(section)
        
        # Sort sections by priority
        sections.sort(key=lambda s: -sum(i.priority for i in s.items) / max(len(s.items), 1))
        
        # Generate executive summary if configured
        executive_summary = None
        if self.config.executive_summary_allocation > 0:
            executive_summary = await self._generate_executive_summary(sections)
        
        # Calculate totals
        total_items = sum(len(s.items) for s in sections)
        total_sources = len(set(item.source_id for s in sections for item in s.items))
        total_tokens = sum(s.token_count for s in sections)
        if executive_summary:
            total_tokens += self.token_counter.count_tokens(executive_summary)
        
        # Create result
        result = AggregatedContent(
            sections=sections,
            executive_summary=executive_summary,
            total_items=total_items,
            total_sources=total_sources,
            total_tokens=total_tokens,
            metadata={
                'config': self.config.__dict__,
                'processing_time': (datetime.now(timezone.utc) - start_time).total_seconds()
            }
        )
        
        # Log metrics
        log_histogram("aggregation_duration", result.metadata['processing_time'], labels={
            "item_count": str(total_items),
            "section_count": str(len(sections))
        })
        log_counter("aggregation_performed", labels={
            "total_items": str(total_items),
            "total_tokens": str(total_tokens),
            "has_summary": str(bool(executive_summary))
        })
        
        return result
    
    def _prepare_items(self, items: List[Dict[str, Any]], 
                      subscriptions: Optional[Dict[int, Dict[str, Any]]]) -> List[AggregatedItem]:
        """Convert raw items to AggregatedItem objects."""
        aggregated_items = []
        
        for item in items:
            # Get subscription metadata
            sub_id = item.get('subscription_id')
            sub_data = subscriptions.get(sub_id, {}) if subscriptions else {}
            
            # Count tokens in content
            content = item.get('content', '')
            token_count = self.token_counter.count_tokens(content)
            
            # Create aggregated item
            agg_item = AggregatedItem(
                source_id=sub_id,
                source_name=sub_data.get('name', f'Source {sub_id}'),
                url=item.get('url', ''),
                title=item.get('title', 'Untitled'),
                content=content,
                summary=item.get('summary'),
                author=item.get('author'),
                published_date=self._parse_date(item.get('published_date')),
                categories=self._parse_categories(item.get('categories')),
                priority=sub_data.get('priority', 3),
                token_count=token_count,
                metadata=item.get('metadata', {})
            )
            
            aggregated_items.append(agg_item)
        
        return aggregated_items
    
    def _apply_filters(self, items: List[AggregatedItem]) -> List[AggregatedItem]:
        """Apply configured filters to items."""
        filtered = items
        
        # Filter by relevance score
        if self.config.min_relevance_score > 0:
            filtered = [i for i in filtered if i.relevance_score >= self.config.min_relevance_score]
        
        # Filter by date range
        if self.config.date_range_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.date_range_days)
            filtered = [i for i in filtered if i.published_date and i.published_date >= cutoff]
        
        # Filter by categories
        if self.config.include_categories:
            included = set(self.config.include_categories)
            filtered = [i for i in filtered if any(c in included for c in i.categories)]
        
        if self.config.exclude_categories:
            excluded = set(self.config.exclude_categories)
            filtered = [i for i in filtered if not any(c in excluded for c in i.categories)]
        
        return filtered
    
    async def _calculate_relevance_scores(self, items: List[AggregatedItem]) -> List[AggregatedItem]:
        """Calculate relevance scores for items."""
        # For now, use simple heuristics
        # In future, could use LLM or more sophisticated scoring
        
        for item in items:
            score = 1.0
            
            # Boost for recent items
            if item.published_date:
                age_hours = (datetime.now(timezone.utc) - item.published_date).total_seconds() / 3600
                if age_hours < 24:
                    score *= 1.5
                elif age_hours < 72:
                    score *= 1.2
                elif age_hours > 168:  # Older than a week
                    score *= 0.8
            
            # Boost for high-priority sources
            score *= (1 + (item.priority - 3) * 0.2)
            
            # Boost for items with summaries
            if item.summary:
                score *= 1.1
            
            # Adjust for content length (prefer substantial content)
            if item.token_count > 1000:
                score *= 1.2
            elif item.token_count < 100:
                score *= 0.7
            
            item.relevance_score = min(score, 2.0)  # Cap at 2.0
        
        return items
    
    def _group_items(self, items: List[AggregatedItem]) -> Dict[str, List[AggregatedItem]]:
        """Group items by configured criteria."""
        groups = defaultdict(list)
        
        for item in items:
            if self.config.group_by == 'source':
                key = item.source_name
            elif self.config.group_by == 'category':
                # Use first category or 'Uncategorized'
                key = item.categories[0] if item.categories else 'Uncategorized'
            elif self.config.group_by == 'date':
                # Group by day
                if item.published_date:
                    key = item.published_date.strftime('%Y-%m-%d')
                else:
                    key = 'Undated'
            elif self.config.group_by == 'priority':
                key = f'Priority {item.priority}'
            else:
                key = 'All Items'
            
            groups[key].append(item)
        
        # Sort items within each group
        for key, group_items in groups.items():
            if self.config.sort_within_groups == 'relevance':
                group_items.sort(key=lambda x: (-x.relevance_score, -x.priority))
            elif self.config.sort_within_groups == 'date':
                group_items.sort(key=lambda x: x.published_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            elif self.config.sort_within_groups == 'priority':
                group_items.sort(key=lambda x: (-x.priority, -x.relevance_score))
        
        return dict(groups)
    
    def _allocate_section_budgets(self, grouped_items: Dict[str, List[AggregatedItem]]) -> Dict[str, int]:
        """Allocate token budgets to each section."""
        # Calculate available content budget
        content_budget = int(self.config.total_token_budget * self.config.content_allocation)
        
        # Calculate section weights based on priority and item count
        section_weights = {}
        total_weight = 0
        
        for key, items in grouped_items.items():
            # Weight = average priority * sqrt(item count)
            avg_priority = sum(i.priority for i in items) / len(items)
            weight = avg_priority * (len(items) ** 0.5)
            section_weights[key] = weight
            total_weight += weight
        
        # Allocate budgets proportionally
        section_budgets = {}
        for key, weight in section_weights.items():
            budget = int(content_budget * (weight / total_weight))
            # Ensure minimum budget
            budget = max(budget, self.config.min_tokens_per_item * len(grouped_items[key]))
            section_budgets[key] = budget
        
        return section_budgets
    
    async def _create_section(self, title: str, items: List[AggregatedItem], 
                            token_budget: int) -> ContentSection:
        """Create a content section with token management."""
        # Allocate tokens to items
        allocations = self.token_manager.allocate_to_items(
            items=[{'id': i, 'priority': i.priority, 'current_tokens': i.token_count} for i in items],
            total_budget=token_budget,
            min_per_item=self.config.min_tokens_per_item,
            max_per_item=self.config.max_tokens_per_item
        )
        
        # Process items based on allocation
        processed_items = []
        total_tokens = 0
        
        for item, allocation in allocations.items():
            if allocation < item.token_count:
                # Need to summarize
                item.summary = await self._summarize_content(
                    item.content, 
                    target_tokens=allocation
                )
                item.token_count = self.token_counter.count_tokens(item.summary or '')
            
            processed_items.append(item)
            total_tokens += item.token_count
        
        # Create section
        section = ContentSection(
            title=title,
            items=processed_items,
            token_count=total_tokens,
            metadata={
                'original_item_count': len(items),
                'processed_item_count': len(processed_items),
                'budget': token_budget,
                'actual_tokens': total_tokens
            }
        )
        
        return section
    
    async def _summarize_content(self, content: str, target_tokens: int) -> str:
        """Summarize content to fit token budget using recursive summarization."""
        try:
            # Use the recursive summarizer for high-quality summarization
            result = await self.recursive_summarizer.summarize_content(
                content=content,
                target_tokens=target_tokens,
                context="Content from aggregated items for briefing"
            )
            
            # Log summarization metrics
            logger.debug(f"Summarized content: {result.original_tokens} -> {result.final_tokens} tokens "
                        f"(compression ratio: {result.compression_ratio:.2f}, method: {result.method_used})")
            
            return result.final_summary
            
        except Exception as e:
            logger.error(f"Recursive summarization failed: {str(e)}. Falling back to truncation.")
            
            # Fallback to simple truncation if recursive summarization fails
            current_tokens = self.token_counter.count_tokens(content)
            if current_tokens <= target_tokens:
                return content
            
            # Use the token counter's truncate method for proper truncation
            return self.token_counter.truncate_to_tokens(content, target_tokens)
    
    async def _generate_executive_summary(self, sections: List[ContentSection]) -> Optional[str]:
        """Generate executive summary of all sections."""
        if not sections:
            return None
        
        # For now, create a simple summary
        # Will be enhanced with LLM in the summarization module
        summary_parts = []
        
        # Overview
        total_items = sum(len(s.items) for s in sections)
        summary_parts.append(f"This briefing contains {total_items} items from {len(sections)} sections.")
        
        # Key highlights from each section
        summary_parts.append("\nKey Highlights:")
        for section in sections[:5]:  # Top 5 sections
            if section.items:
                top_item = section.items[0]
                summary_parts.append(f"- {section.title}: {top_item.title}")
        
        # Token budget for summary
        budget = int(self.config.total_token_budget * self.config.executive_summary_allocation)
        summary = '\n'.join(summary_parts)
        
        # Ensure within budget
        if self.token_counter.count_tokens(summary) > budget:
            summary = await self._summarize_content(summary, budget)
        
        return summary
    
    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats."""
        if isinstance(date_value, datetime):
            return date_value if date_value.tzinfo else date_value.replace(tzinfo=timezone.utc)
        
        if isinstance(date_value, str):
            try:
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except:
                pass
        
        return None
    
    def _parse_categories(self, categories: Any) -> List[str]:
        """Parse categories from various formats."""
        if isinstance(categories, list):
            return categories
        if isinstance(categories, str):
            return [c.strip() for c in categories.split(',') if c.strip()]
        return []


# End of aggregation_engine.py