# RAG v2: Functional Pipeline Integration

## Executive Summary

This document outlines the plan to refactor the current RAG (Retrieval-Augmented Generation) system from a monolithic, object-oriented design with TOML configuration overlay into a clean, functional, composable pipeline system where TOML defines the actual function composition.

## Current State Analysis

### What Currently Exists

1. **Monolithic Search Functions** (`chat_rag_events.py`)
   - `perform_plain_rag_search()` - 400+ lines handling FTS5/BM25 search
   - `perform_full_rag_pipeline()` - 300+ lines handling semantic search
   - `perform_hybrid_rag_search()` - Complex orchestration of both approaches
   - Each function handles retrieval, processing, formatting internally

2. **TOML Configuration Overlay** 
   - `rag_pipelines.toml` defines pipelines but only maps to existing functions
   - `pipeline_loader.py` loads TOML but doesn't change execution
   - `pipeline_integration.py` provides optional fallback to TOML configs
   - Middleware defined but not implemented

3. **Configuration Profiles** (`config_profiles.py`)
   - Predefined configurations for different use cases
   - A/B testing framework
   - Not integrated with TOML pipeline system

4. **RAG Service** (`simplified/rag_service.py`)
   - Coordinates embeddings, vector stores, chunking
   - Has its own configuration system
   - Not aware of pipeline concept

### Problems with Current Approach

1. **Dual Systems**: TOML pipelines exist alongside hardcoded functions
2. **Optional Integration**: Pipeline system treated as optional with fallbacks everywhere
3. **Monolithic Functions**: Can't compose or reuse parts of search functions
4. **Configuration Confusion**: Multiple configuration systems not unified
5. **Incomplete Implementation**: Middleware, strategies, composition not implemented

## Proposed Solution: Functional Pipelines

### Core Design Principles

1. **Pure Functions**: Each pipeline component is a pure function with clear inputs/outputs
2. **Function Composition**: Pipelines are composed using simple combinators
3. **Data-First**: All configuration and state flows through data structures
4. **TOML as DSL**: TOML becomes a simple DSL for function composition
5. **No Fallbacks**: New system is the primary implementation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    TOML Pipeline Definition              │
│  Defines: steps, functions, parameters, flow control    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Pipeline Builder                      │
│  Reads TOML, composes functions, returns pipeline       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Pipeline Functions                      │
│  retrieve_fts5, retrieve_semantic, rerank, format, etc  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Core Services                          │
│  MediaDB, VectorStore, Embeddings, ChunkingService      │
└─────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Core Pipeline Functions

Create `tldw_chatbook/RAG_Search/pipeline_functions.py`:

```python
"""
Pure pipeline functions for RAG system.
Each function has clear inputs/outputs and no side effects.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from loguru import logger

# Type aliases for clarity
Query = str
Sources = Dict[str, bool]
Config = Dict[str, Any]
Results = List[Dict[str, Any]]
Context = str

# ==============================================================================
# Retrieval Functions
# ==============================================================================

async def retrieve_fts5(
    query: Query, 
    sources: Sources, 
    config: Config,
    context: Dict[str, Any]
) -> Results:
    """
    Pure FTS5/BM25 retrieval function.
    
    Args:
        query: Search query
        sources: Which sources to search (media, conversations, notes)
        config: Configuration including top_k, filters, etc.
        context: Shared context including app instance
        
    Returns:
        List of search results with scores
    """
    app = context.get('app')
    if not app:
        raise ValueError("App instance required in context")
    
    results = []
    top_k = config.get('top_k', 10)
    keyword_filter = config.get('keyword_filter', [])
    
    # Search each enabled source
    if sources.get('media') and hasattr(app, 'client_media_db'):
        media_results = await search_media_fts5(
            app.client_media_db, query, top_k, keyword_filter
        )
        results.extend(media_results)
    
    if sources.get('conversations') and hasattr(app, 'chachanotes_db'):
        conv_results = await search_conversations_fts5(
            app.chachanotes_db, query, top_k, keyword_filter
        )
        results.extend(conv_results)
        
    if sources.get('notes') and hasattr(app, 'notes_service'):
        notes_results = await search_notes_fts5(
            app.notes_service, query, top_k, keyword_filter
        )
        results.extend(notes_results)
    
    return results

async def retrieve_semantic(
    query: Query,
    sources: Sources,
    config: Config,
    context: Dict[str, Any]
) -> Results:
    """
    Pure semantic/vector retrieval function.
    
    Uses embeddings and vector similarity search.
    """
    app = context.get('app')
    rag_service = context.get('rag_service')
    
    if not rag_service:
        # Initialize RAG service if needed
        from .simplified import RAGService, create_config_for_collection
        rag_config = create_config_for_collection("media")
        rag_service = RAGService(rag_config)
        context['rag_service'] = rag_service
    
    # Perform vector search
    results = await rag_service.search(
        query=query,
        search_type="semantic",
        top_k=config.get('top_k', 10),
        score_threshold=config.get('score_threshold', 0.0),
        include_citations=config.get('include_citations', True)
    )
    
    # Convert to standard format
    return [result.to_dict() for result in results]

# ==============================================================================
# Processing Functions
# ==============================================================================

def rerank_results(
    results: Results,
    query: Query,
    config: Config
) -> Results:
    """
    Rerank results using specified model.
    
    Pure function - no side effects.
    """
    model = config.get('model', 'flashrank')
    top_k = config.get('top_k', len(results))
    
    if model == 'flashrank':
        return rerank_with_flashrank(results, query, top_k)
    elif model == 'cohere':
        return rerank_with_cohere(results, query, top_k, config)
    else:
        # No reranking, just limit results
        return results[:top_k]

def deduplicate_results(
    results: Results,
    config: Config
) -> Results:
    """
    Remove duplicate results based on content similarity.
    """
    strategy = config.get('strategy', 'content_hash')
    threshold = config.get('threshold', 0.9)
    
    if strategy == 'content_hash':
        seen = set()
        deduped = []
        for result in results:
            content_hash = hash(result.get('content', '')[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                deduped.append(result)
        return deduped
    else:
        return results

def filter_by_score(
    results: Results,
    config: Config
) -> Results:
    """Filter results by minimum score threshold."""
    min_score = config.get('min_score', 0.0)
    return [r for r in results if r.get('score', 0) >= min_score]

# ==============================================================================
# Formatting Functions
# ==============================================================================

def format_as_context(
    results: Results,
    config: Config
) -> Context:
    """
    Format results as LLM context string.
    """
    max_length = config.get('max_length', 10000)
    include_citations = config.get('include_citations', True)
    
    context_parts = []
    total_chars = 0
    
    for i, result in enumerate(results):
        # Format each result
        source = result.get('source', 'unknown').upper()
        title = result.get('title', 'Untitled')
        content = result.get('content', '')
        
        if include_citations:
            result_text = f"[{source} - {title}]\n{content}"
        else:
            result_text = content
            
        # Check length limit
        remaining = max_length - total_chars - len(result_text)
        if remaining <= 0:
            break
            
        context_parts.append(result_text)
        total_chars += len(result_text)
        
        if i < len(results) - 1:
            separator = "\n\n---\n\n"
            if total_chars + len(separator) < max_length:
                context_parts.append(separator)
                total_chars += len(separator)
    
    return "".join(context_parts)

def format_as_json(
    results: Results,
    config: Config
) -> str:
    """Format results as JSON string."""
    import json
    max_results = config.get('max_results', len(results))
    return json.dumps(results[:max_results], indent=2)

# ==============================================================================
# Combinator Functions
# ==============================================================================

async def parallel(*funcs):
    """
    Run functions in parallel and collect results.
    
    Returns a function that executes all provided functions in parallel.
    """
    async def executor(query, sources, config, context):
        tasks = [func(query, sources, config, context) for func in funcs]
        results = await asyncio.gather(*tasks)
        # Flatten results
        all_results = []
        for result_list in results:
            all_results.extend(result_list)
        return all_results
    return executor

async def sequential(*funcs):
    """
    Run functions in sequence, passing output to input.
    
    First function gets standard inputs, subsequent functions
    get the previous function's output as first argument.
    """
    async def executor(query, sources, config, context):
        result = None
        for i, func in enumerate(funcs):
            if i == 0:
                result = await func(query, sources, config, context)
            else:
                # For processing functions that take results as first arg
                if asyncio.iscoroutinefunction(func):
                    result = await func(result, query, config)
                else:
                    result = func(result, query, config)
        return result
    return executor

async def weighted_merge(
    results_lists: List[Results],
    weights: List[float]
) -> Results:
    """
    Merge multiple result lists with weighted scores.
    """
    if len(results_lists) != len(weights):
        raise ValueError("Number of result lists must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Create merged results with weighted scores
    merged = {}
    
    for results, weight in zip(results_lists, weights):
        for result in results:
            key = result.get('content', '')[:200]  # Use content prefix as key
            if key in merged:
                # Update score with weighted average
                merged[key]['score'] = (
                    merged[key]['score'] + result.get('score', 0) * weight
                )
            else:
                # New result
                result_copy = result.copy()
                result_copy['score'] = result.get('score', 0) * weight
                merged[key] = result_copy
    
    # Sort by final score
    final_results = list(merged.values())
    final_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return final_results

# ==============================================================================
# Helper Functions
# ==============================================================================

async def search_media_fts5(db, query, top_k, keyword_filter):
    """Extract media search logic from existing code."""
    # Implementation extracted from perform_plain_rag_search
    pass

async def search_conversations_fts5(db, query, top_k, keyword_filter):
    """Extract conversation search logic from existing code."""
    # Implementation extracted from perform_plain_rag_search
    pass

async def search_notes_fts5(service, query, top_k, keyword_filter):
    """Extract notes search logic from existing code."""
    # Implementation extracted from perform_plain_rag_search
    pass

def rerank_with_flashrank(results, query, top_k):
    """Rerank using FlashRank model."""
    # Implementation extracted from existing code
    pass

def rerank_with_cohere(results, query, top_k, config):
    """Rerank using Cohere API."""
    # Implementation extracted from existing code
    pass

# ==============================================================================
# Function Registry
# ==============================================================================

PIPELINE_FUNCTIONS = {
    # Retrievers
    'retrieve_fts5': retrieve_fts5,
    'retrieve_semantic': retrieve_semantic,
    
    # Processors
    'rerank_results': rerank_results,
    'deduplicate_results': deduplicate_results,
    'filter_by_score': filter_by_score,
    
    # Formatters
    'format_as_context': format_as_context,
    'format_as_json': format_as_json,
    
    # Combinators
    'parallel': parallel,
    'sequential': sequential,
    'weighted_merge': weighted_merge,
}

def get_function(name: str):
    """Get a pipeline function by name."""
    if name not in PIPELINE_FUNCTIONS:
        raise ValueError(f"Unknown pipeline function: {name}")
    return PIPELINE_FUNCTIONS[name]
```

### 2. Pipeline Builder

Create `tldw_chatbook/RAG_Search/pipeline_builder.py`:

```python
"""
Pipeline builder that creates executable pipelines from TOML configuration.
"""

from typing import Dict, Any, List, Tuple, Callable, Awaitable
import asyncio
from loguru import logger

from .pipeline_functions import get_function, Results, Context

# Pipeline type is a function that returns results and formatted context
Pipeline = Callable[
    [str, Dict[str, bool], Dict[str, Any], Dict[str, Any]], 
    Awaitable[Tuple[List[Dict], str]]
]

def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Build an executable pipeline from configuration.
    
    Args:
        config: Pipeline configuration from TOML
        
    Returns:
        Async function that executes the pipeline
    """
    steps = config.get('steps', [])
    pipeline_name = config.get('name', 'Unnamed Pipeline')
    
    async def execute_pipeline(
        query: str,
        sources: Dict[str, bool],
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[List[Dict], str]:
        """Execute the configured pipeline."""
        logger.info(f"Executing pipeline: {pipeline_name}")
        
        # Initialize pipeline context
        pipeline_context = {
            **context,
            'query': query,
            'sources': sources,
            'params': params,
            'results': [],
            'formatted': ''
        }
        
        # Execute each step
        for i, step in enumerate(steps):
            step_type = step.get('type')
            logger.debug(f"Executing step {i+1}/{len(steps)}: {step_type}")
            
            try:
                if step_type == 'retrieve':
                    results = await execute_retrieve_step(step, pipeline_context)
                    pipeline_context['results'] = results
                    
                elif step_type == 'parallel':
                    results = await execute_parallel_step(step, pipeline_context)
                    pipeline_context['results'] = results
                    
                elif step_type == 'process':
                    results = execute_process_step(step, pipeline_context)
                    pipeline_context['results'] = results
                    
                elif step_type == 'merge':
                    results = await execute_merge_step(step, pipeline_context)
                    pipeline_context['results'] = results
                    
                elif step_type == 'format':
                    formatted = execute_format_step(step, pipeline_context)
                    pipeline_context['formatted'] = formatted
                    
                else:
                    logger.warning(f"Unknown step type: {step_type}")
                    
            except Exception as e:
                logger.error(f"Error in pipeline step {i+1}: {e}")
                raise
        
        return pipeline_context['results'], pipeline_context['formatted']
    
    return execute_pipeline

async def execute_retrieve_step(
    step: Dict[str, Any],
    context: Dict[str, Any]
) -> Results:
    """Execute a retrieval step."""
    func_name = step.get('function')
    config = step.get('config', {})
    
    func = get_function(func_name)
    return await func(
        context['query'],
        context['sources'],
        {**context['params'], **config},
        context
    )

async def execute_parallel_step(
    step: Dict[str, Any],
    context: Dict[str, Any]
) -> Results:
    """Execute parallel retrieval functions."""
    functions = step.get('functions', [])
    
    tasks = []
    for func_config in functions:
        func_name = func_config.get('function')
        config = func_config.get('config', {})
        func = get_function(func_name)
        
        task = func(
            context['query'],
            context['sources'],
            {**context['params'], **config},
            context
        )
        tasks.append(task)
    
    results_lists = await asyncio.gather(*tasks)
    
    # Store individual results for merge step
    context['parallel_results'] = results_lists
    
    # Flatten results by default
    all_results = []
    for results in results_lists:
        all_results.extend(results)
    
    return all_results

def execute_process_step(
    step: Dict[str, Any],
    context: Dict[str, Any]
) -> Results:
    """Execute a processing step."""
    func_name = step.get('function')
    config = step.get('config', {})
    
    func = get_function(func_name)
    results = context.get('results', [])
    
    # Processing functions are synchronous and take results as first arg
    return func(results, context['query'], config)

async def execute_merge_step(
    step: Dict[str, Any],
    context: Dict[str, Any]
) -> Results:
    """Execute a merge step for parallel results."""
    func_name = step.get('function', 'weighted_merge')
    config = step.get('config', {})
    
    func = get_function(func_name)
    
    # Get parallel results from context
    parallel_results = context.get('parallel_results', [])
    if not parallel_results:
        logger.warning("No parallel results to merge")
        return context.get('results', [])
    
    weights = config.get('weights', [1.0] * len(parallel_results))
    
    return await func(parallel_results, weights)

def execute_format_step(
    step: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """Execute a formatting step."""
    func_name = step.get('function')
    config = step.get('config', {})
    
    func = get_function(func_name)
    results = context.get('results', [])
    
    return func(results, config)

def validate_pipeline_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate pipeline configuration.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    if 'name' not in config:
        errors.append("Pipeline must have a name")
    
    if 'steps' not in config:
        errors.append("Pipeline must have steps")
    elif not isinstance(config['steps'], list):
        errors.append("Pipeline steps must be a list")
    else:
        for i, step in enumerate(config['steps']):
            if 'type' not in step:
                errors.append(f"Step {i+1} must have a type")
            
            step_type = step.get('type')
            if step_type in ['retrieve', 'process', 'format']:
                if 'function' not in step:
                    errors.append(f"Step {i+1} must specify a function")
            elif step_type == 'parallel':
                if 'functions' not in step:
                    errors.append(f"Step {i+1} must specify functions")
    
    return errors
```

### 3. Updated TOML Pipeline Definitions

Update `rag_pipelines.toml`:

```toml
# RAG Pipeline Definitions - Functional Version
# Each pipeline is composed of pure functions executed in steps

# ==============================================================================
# Built-in Pipelines
# ==============================================================================

[pipelines.plain_v2]
name = "Plain Search (Functional)"
description = "Fast keyword search using FTS5"
type = "functional"
enabled = true
tags = ["fast", "keyword", "fts5"]

[[pipelines.plain_v2.steps]]
type = "retrieve"
function = "retrieve_fts5"
config = { top_k = 20, min_score = 0.5 }

[[pipelines.plain_v2.steps]]
type = "process"
function = "deduplicate_results"
config = { strategy = "content_hash" }

[[pipelines.plain_v2.steps]]
type = "process" 
function = "filter_by_score"
config = { min_score = 0.5 }

[[pipelines.plain_v2.steps]]
type = "format"
function = "format_as_context"
config = { max_length = 10000, include_citations = true }

# ------------------------------------------------------------------------------

[pipelines.semantic_v2]
name = "Semantic Search (Functional)"
description = "AI-powered semantic search using embeddings"
type = "functional"
enabled = true
tags = ["semantic", "embeddings", "ai"]

[[pipelines.semantic_v2.steps]]
type = "retrieve"
function = "retrieve_semantic"
config = { top_k = 20, score_threshold = 0.0 }

[[pipelines.semantic_v2.steps]]
type = "process"
function = "rerank_results"
config = { model = "flashrank", top_k = 10 }

[[pipelines.semantic_v2.steps]]
type = "format"
function = "format_as_context"
config = { max_length = 10000, include_citations = true }

# ------------------------------------------------------------------------------

[pipelines.hybrid_v2]
name = "Hybrid Search (Functional)"
description = "Combined keyword and semantic search"
type = "functional"
enabled = true
tags = ["hybrid", "balanced", "comprehensive"]

# Step 1: Parallel retrieval
[[pipelines.hybrid_v2.steps]]
type = "parallel"
functions = [
    { function = "retrieve_fts5", config = { top_k = 15 } },
    { function = "retrieve_semantic", config = { top_k = 15 } }
]

# Step 2: Merge results with weights
[[pipelines.hybrid_v2.steps]]
type = "merge"
function = "weighted_merge"
config = { weights = [0.5, 0.5] }

# Step 3: Deduplicate
[[pipelines.hybrid_v2.steps]]
type = "process"
function = "deduplicate_results"
config = { strategy = "content_hash" }

# Step 4: Rerank
[[pipelines.hybrid_v2.steps]]
type = "process"
function = "rerank_results"
config = { model = "flashrank", top_k = 10 }

# Step 5: Format
[[pipelines.hybrid_v2.steps]]
type = "format"
function = "format_as_context"
config = { max_length = 10000, include_citations = true }

# ==============================================================================
# Custom Pipeline Examples
# ==============================================================================

[pipelines.technical_docs_v2]
name = "Technical Documentation Search"
description = "Optimized for code and technical content"
type = "functional"
enabled = true
tags = ["technical", "code", "documentation"]

# Parallel search with different strategies
[[pipelines.technical_docs_v2.steps]]
type = "parallel"
functions = [
    { function = "retrieve_fts5", config = { top_k = 20 } },
    { function = "retrieve_semantic", config = { top_k = 10 } }
]

# Weighted merge favoring exact matches
[[pipelines.technical_docs_v2.steps]]
type = "merge"
function = "weighted_merge"
config = { weights = [0.7, 0.3] }  # Favor FTS5 for technical terms

[[pipelines.technical_docs_v2.steps]]
type = "process"
function = "deduplicate_results"
config = { strategy = "content_hash" }

[[pipelines.technical_docs_v2.steps]]
type = "format"
function = "format_as_context"
config = { 
    max_length = 15000,  # Larger context for code
    include_citations = true,
    preserve_formatting = true
}

# ------------------------------------------------------------------------------

[pipelines.quick_support_v2]
name = "Quick Support Search"
description = "Fast search for customer support"
type = "functional"
enabled = true
tags = ["support", "fast", "customer-service"]

# Single FTS5 search for speed
[[pipelines.quick_support_v2.steps]]
type = "retrieve"
function = "retrieve_fts5"
config = { top_k = 5 }

# No reranking for speed
[[pipelines.quick_support_v2.steps]]
type = "process"
function = "filter_by_score"
config = { min_score = 0.7 }  # Higher threshold for relevance

[[pipelines.quick_support_v2.steps]]
type = "format"
function = "format_as_context"
config = { 
    max_length = 2000,  # Shorter for quick responses
    include_citations = false  # Cleaner for support
}
```

### 4. Integration Layer

Update `pipeline_loader.py` to use the new functional system:

```python
def _get_builtin_function(self, pipeline: PipelineConfig) -> Optional[Callable]:
    """Get a built-in pipeline function."""
    if pipeline.type == "functional":
        # Use new pipeline builder
        from .pipeline_builder import build_pipeline
        return build_pipeline(pipeline.__dict__)
    else:
        # Legacy support
        if pipeline.function not in self._builtin_functions:
            logger.error(f"Built-in function '{pipeline.function}' not found")
            return None
        
        base_func = self._builtin_functions[pipeline.function]
        # ... rest of legacy code
```

## Migration Strategy

### Phase 1: Foundation (Week 1)
1. Create `pipeline_functions.py` with function signatures
2. Create `pipeline_builder.py` with basic implementation
3. Extract one function (e.g., FTS5 search) as proof of concept
4. Test with a simple pipeline

### Phase 2: Function Extraction (Week 2)
1. Extract search logic from `perform_plain_rag_search`
2. Extract search logic from `perform_full_rag_pipeline`
3. Extract reranking and formatting logic
4. Create comprehensive tests for each function

### Phase 3: Integration (Week 3)
1. Update `pipeline_loader.py` to support functional pipelines
2. Create functional versions of all built-in pipelines
3. Update UI components to work with new pipelines
4. Add backward compatibility layer

### Phase 4: Migration (Week 4)
1. Switch default pipelines to functional versions
2. Deprecate old monolithic functions
3. Update documentation
4. Performance testing and optimization

### Phase 5: Cleanup (Week 5)
1. Remove fallback logic
2. Remove deprecated functions
3. Unify configuration systems
4. Final testing and validation

## Benefits of This Approach

### 1. Simplicity
- Pure functions are easy to understand
- No hidden state or side effects
- Clear data flow through pipeline

### 2. Composability
- Build complex pipelines from simple functions
- Reuse functions across pipelines
- Easy to add new functions

### 3. Testability
- Each function can be tested in isolation
- Predictable behavior
- Easy to mock dependencies

### 4. Performance
- Functions can be optimized independently
- Parallel execution where possible
- Efficient caching strategies

### 5. Configuration
- TOML becomes actual pipeline definition
- No hidden behavior in code
- Easy to share and version pipelines

## Example Usage

### From Code
```python
from tldw_chatbook.RAG_Search.pipeline_loader import get_pipeline_loader

# Load pipeline
loader = get_pipeline_loader()
pipeline = loader.get_pipeline_function("hybrid_v2")

# Execute
results, context = await pipeline(
    query="What is RAG?",
    sources={"media": True, "conversations": False, "notes": True},
    params={"user_id": "123"},
    context={"app": app_instance}
)
```

### From TOML
```toml
# Custom pipeline for specific use case
[pipelines.my_custom_pipeline]
name = "My Custom Pipeline"
type = "functional"

[[pipelines.my_custom_pipeline.steps]]
type = "retrieve"
function = "retrieve_fts5"
config = { top_k = 50 }

[[pipelines.my_custom_pipeline.steps]]
type = "process"
function = "deduplicate_results"

[[pipelines.my_custom_pipeline.steps]]
type = "process"
function = "rerank_results"
config = { model = "cohere", top_k = 10 }

[[pipelines.my_custom_pipeline.steps]]
type = "format"
function = "format_as_json"
config = { max_results = 10 }
```

## Conclusion

This functional approach provides a clean, composable, and maintainable RAG system where:
- Pipelines are built from pure functions
- TOML defines actual pipeline behavior
- No complex class hierarchies
- Clear data flow and transformation
- Easy to extend and customize

The migration can be done incrementally while maintaining backward compatibility, ultimately resulting in a much simpler and more powerful system.