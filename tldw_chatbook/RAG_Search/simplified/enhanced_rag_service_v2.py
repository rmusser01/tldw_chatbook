"""
Enhanced RAG Service v2 with Phase 2 features integrated.

This extends the Phase 1 enhanced RAG service with:
- LLM-based reranking
- Parallel processing optimization
- Configuration profiles
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from pathlib import Path
from loguru import logger
import uuid

from .enhanced_rag_service import EnhancedRAGService
from .config import RAGConfig
from .vector_store import SearchResult, SearchResultWithCitations
from ..reranker import create_reranker, BaseReranker, RerankingConfig
from ..parallel_processor import (
    create_embedding_processor, create_chunking_processor,
    ProcessingConfig, EmbeddingBatchProcessor, ChunkingBatchProcessor
)
from ..config_profiles import (
    get_profile_manager, ProfileConfig, ExperimentConfig,
    ProfileType, ConfigProfileManager
)
from .data_models import IndexingResult
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, timeit


class EnhancedRAGServiceV2(EnhancedRAGService):
    """
    Enhanced RAG Service with Phase 2 features.
    
    Adds:
    - Intelligent reranking with LLMs
    - Parallel processing for batch operations
    - Configuration profiles and A/B testing
    """
    
    def __init__(self, 
                 config: Optional[Union[RAGConfig, ProfileConfig, str]] = None,
                 enable_parent_retrieval: bool = True,
                 enable_reranking: bool = False,
                 enable_parallel_processing: bool = True,
                 profile_manager: Optional[ConfigProfileManager] = None):
        """
        Initialize enhanced RAG service v2.
        
        Args:
            config: RAG config, profile config, or profile name
            enable_parent_retrieval: Whether to enable parent document retrieval
            enable_reranking: Whether to enable LLM-based reranking
            enable_parallel_processing: Whether to use parallel processing
            profile_manager: Configuration profile manager
        """
        # Handle different config types
        if isinstance(config, str):
            # Load profile by name
            self.profile_manager = profile_manager or get_profile_manager()
            profile = self.profile_manager.get_profile(config)
            if not profile:
                raise ValueError(f"Profile '{config}' not found")
            self.profile = profile
            config = profile.rag_config
            self.reranking_config = profile.reranking_config
            self.processing_config = profile.processing_config
        elif isinstance(config, ProfileConfig):
            # Use provided profile
            self.profile = config
            config = config.rag_config
            self.reranking_config = config.reranking_config
            self.processing_config = config.processing_config
            self.profile_manager = profile_manager or get_profile_manager()
        else:
            # Direct RAG config
            self.profile = None
            self.reranking_config = None
            self.processing_config = None
            self.profile_manager = profile_manager or get_profile_manager()
        
        # Initialize base service
        super().__init__(config, enable_parent_retrieval)
        
        # Phase 2 features
        self.enable_reranking = enable_reranking
        self.enable_parallel_processing = enable_parallel_processing
        
        # Initialize reranker if enabled
        self.reranker = None
        if self.enable_reranking and self.reranking_config:
            self.reranker = create_reranker(
                strategy=self.reranking_config.strategy,
                **self.reranking_config.__dict__
            )
            logger.info(f"Initialized {self.reranking_config.strategy} reranker")
        
        # Initialize parallel processors if enabled
        self.embedding_processor = None
        self.chunking_processor = None
        if self.enable_parallel_processing:
            proc_config = self.processing_config or ProcessingConfig()
            self.embedding_processor = create_embedding_processor(**proc_config.__dict__)
            self.chunking_processor = create_chunking_processor(**proc_config.__dict__)
            logger.info(f"Initialized parallel processors with {proc_config.num_workers or 'auto'} workers")
        
        # Experiment tracking
        self._current_experiment = None
        self._experiment_profile = None
        
        log_counter("rag_service_v2_initialized", labels={
            "profile": self.profile.name if self.profile else "custom",
            "reranking": str(self.enable_reranking),
            "parallel": str(self.enable_parallel_processing)
        })
    
    @classmethod
    def from_profile(cls, 
                    profile_name: ProfileType,
                    **kwargs) -> "EnhancedRAGServiceV2":
        """
        Create service from a predefined profile.
        
        Args:
            profile_name: Name of the profile to use
            **kwargs: Additional arguments for initialization
            
        Returns:
            Configured RAG service
        """
        manager = get_profile_manager()
        profile = manager.get_profile(profile_name)
        
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        # Auto-enable features based on profile
        if "enable_reranking" not in kwargs:
            kwargs["enable_reranking"] = profile.reranking_config is not None
        
        if "enable_parallel_processing" not in kwargs:
            kwargs["enable_parallel_processing"] = profile.processing_config is not None
        
        return cls(config=profile, profile_manager=manager, **kwargs)
    
    @timeit("rag_search_v2")
    async def search(self,
                    query: str,
                    top_k: Optional[int] = None,
                    search_type: Literal["semantic", "hybrid", "keyword"] = "semantic",
                    filter_metadata: Optional[Dict[str, Any]] = None,
                    include_citations: Optional[bool] = None,
                    score_threshold: Optional[float] = None,
                    rerank: Optional[bool] = None,
                    user_id: Optional[str] = None) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Enhanced search with optional reranking and experiment tracking.
        
        Args:
            query: Search query
            top_k: Number of results
            search_type: Type of search
            filter_metadata: Metadata filters
            include_citations: Include citations
            score_threshold: Minimum score
            rerank: Override reranking setting
            user_id: User ID for experiment tracking
            
        Returns:
            Search results, optionally reranked
        """
        start_time = time.time()
        
        # Handle experiment profile selection
        experiment_profile = None
        if self._current_experiment and user_id:
            profile_name, experiment_profile = self.profile_manager.select_profile_for_experiment(user_id)
            # Temporarily use experiment profile settings
            if experiment_profile and experiment_profile.rag_config:
                # Override search parameters from experiment profile
                top_k = top_k or experiment_profile.rag_config.search.top_k
                include_citations = include_citations if include_citations is not None else experiment_profile.rag_config.search.include_citations
                score_threshold = score_threshold if score_threshold is not None else experiment_profile.rag_config.search.score_threshold
        
        # Perform base search
        results = await super().search(
            query=query,
            top_k=top_k,
            search_type=search_type,
            filter_metadata=filter_metadata,
            include_citations=include_citations,
            score_threshold=score_threshold
        )
        
        # Apply reranking if enabled
        should_rerank = rerank if rerank is not None else self.enable_reranking
        if should_rerank and self.reranker and len(results) > 1:
            rerank_start = time.time()
            
            # Use experiment profile's reranking config if available
            if experiment_profile and experiment_profile.reranking_config:
                # Create temporary reranker with experiment config
                temp_reranker = create_reranker(
                    strategy=experiment_profile.reranking_config.strategy,
                    **experiment_profile.reranking_config.__dict__
                )
                results = await temp_reranker.rerank(query, results)
            else:
                results = await self.reranker.rerank(query, results)
            
            rerank_time = time.time() - rerank_start
            log_histogram("rag_reranking_time", rerank_time)
            logger.debug(f"Reranking completed in {rerank_time:.3f}s")
        
        # Record experiment metrics if active
        if self._current_experiment and user_id:
            search_time = time.time() - start_time
            metrics = {
                "search_latency": search_time * 1000,  # Convert to ms
                "results_returned": len(results),
                "search_type": search_type,
                "reranked": should_rerank
            }
            
            # Add result quality metrics
            if results:
                scores = [r.score for r in results]
                metrics["avg_score"] = sum(scores) / len(scores)
                metrics["top_score"] = scores[0] if scores else 0
            
            self.profile_manager.record_experiment_result(
                profile_name if experiment_profile else "default",
                query,
                metrics
            )
        
        return results
    
    async def index_batch_optimized(self,
                                   documents: List[Dict[str, Any]],
                                   show_progress: Optional[bool] = None,
                                   batch_size: Optional[int] = None) -> List[IndexingResult]:
        """
        Optimized batch indexing.

        NOTE (task-247): the previous "parallel" branch here was broken in
        both directions -- it imported ``generate_embeddings_batch`` /
        ``store_documents_batch`` from ``enhanced_indexing_helpers`` (which
        only defines the ``*_with_parents`` variants, so the imports raised
        ImportError at first use), and the chunking processor's
        ``doc_chunk_info`` (a dict keyed by doc_id) does not match the shape
        the storage helper expects (a list with 'doc_idx'/'chunks'). Any
        profile with a ``processing_config`` would therefore crash on its
        first indexing call. Until a parallel pipeline is actually
        implemented, always delegate to the base optimized batch path, which
        works for every profile.
        """
        if self.enable_parallel_processing and self.chunking_processor:
            logger.debug(
                "Parallel batch-indexing pipeline is not implemented; using base optimized path"
            )
        if show_progress is None:
            show_progress = self.processing_config.show_progress if self.processing_config else True
        return await super().index_batch_optimized(
            documents,
            show_progress=show_progress,
            batch_size=batch_size or 32,
        )
    
    def start_experiment(self, experiment_config: ExperimentConfig):
        """Start an A/B testing experiment."""
        self._current_experiment = experiment_config
        self.profile_manager.start_experiment(experiment_config)
        logger.info(f"Started RAG experiment: {experiment_config.name}")
    
    def end_experiment(self) -> Dict[str, Any]:
        """End current experiment and get results."""
        if not self._current_experiment:
            return {}
        
        results = self.profile_manager.end_experiment()
        self._current_experiment = None
        logger.info("Ended RAG experiment")
        
        return results
    
    def switch_profile(self, profile_name: str):
        """
        Switch to a different configuration profile.
        
        Args:
            profile_name: Name of the profile to switch to
        """
        profile = self.profile_manager.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        # Update configurations
        self.profile = profile
        self.config = profile.rag_config
        self.reranking_config = profile.reranking_config
        self.processing_config = profile.processing_config
        
        # Reinitialize components if needed
        if self.enable_reranking and self.reranking_config:
            self.reranker = create_reranker(
                strategy=self.reranking_config.strategy,
                **self.reranking_config.__dict__
            )
        
        logger.info(f"Switched to profile: {profile_name}")
        log_counter("rag_profile_switch", labels={"profile": profile_name})
    
    def get_current_profile(self) -> Optional[ProfileConfig]:
        """Get the current configuration profile."""
        return self.profile
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any warnings."""
        if self.profile:
            return self.profile_manager.validate_profile(self.profile)
        return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including v2 features."""
        metrics = super().get_metrics()
        
        # Add v2-specific metrics
        v2_metrics = {
            "profile": self.profile.name if self.profile else "custom",
            "reranking_enabled": self.enable_reranking,
            "parallel_processing_enabled": self.enable_parallel_processing,
            "active_experiment": self._current_experiment.name if self._current_experiment else None
        }
        
        # Add reranker metrics if available
        if self.reranker and hasattr(self.reranker, 'get_metrics'):
            v2_metrics["reranker_metrics"] = self.reranker.get_metrics()
        
        metrics["v2_features"] = v2_metrics
        
        return metrics


# Convenience functions for quick setup

async def create_rag_from_profile(
    profile_name: ProfileType,
    documents: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Tuple[EnhancedRAGServiceV2, List[IndexingResult]]:
    """
    Create and optionally index documents using a profile.
    
    Args:
        profile_name: Profile to use
        documents: Optional documents to index
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (service, indexing_results)
    """
    service = EnhancedRAGServiceV2.from_profile(profile_name, **kwargs)
    
    results = []
    if documents:
        results = await service.index_batch_optimized(documents)
    
    return service, results


def quick_search(
    query: str,
    documents: List[Dict[str, Any]],
    profile: ProfileType = "balanced",
    rerank: bool = True
) -> List[Union[SearchResult, SearchResultWithCitations]]:
    """
    Quick one-shot search with automatic setup.
    
    Args:
        query: Search query
        documents: Documents to search
        profile: Profile to use
        rerank: Whether to rerank results
        
    Returns:
        Search results
    """
    async def _search():
        service, _ = await create_rag_from_profile(profile, documents)
        return await service.search(query, rerank=rerank)
    
    return asyncio.run(_search())