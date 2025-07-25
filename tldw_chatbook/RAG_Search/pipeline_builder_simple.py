"""
Simple pipeline builder for executing search pipelines from TOML configuration.

No complex error handling or effects - just execute steps in order.
"""

import asyncio
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import os
import sys
if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
from pathlib import Path

from .pipeline_types import SearchResult, StepType, PipelineContext
from .pipeline_functions_simple import (
    RETRIEVAL_FUNCTIONS, PROCESSING_FUNCTIONS, FORMATTING_FUNCTIONS
)


async def execute_pipeline(
    config: Dict[str, Any],
    app: Any,
    query: str,
    sources: Dict[str, bool],
    **params
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Execute a pipeline based on configuration.
    
    Args:
        config: Pipeline configuration from TOML
        app: TldwCli app instance
        query: Search query
        sources: Which sources to search
        **params: Additional parameters
        
    Returns:
        Tuple of (results list, formatted context)
    """
    pipeline_name = config.get('name', 'Unknown Pipeline')
    steps = config.get('steps', [])
    
    logger.info(f"Executing pipeline: {pipeline_name}")
    
    # Initialize context
    context: PipelineContext = {
        'app': app,
        'query': query,
        'sources': sources,
        'params': {**config.get('parameters', {}), **params},
        'results': []
    }
    
    # Execute each step
    for i, step_config in enumerate(steps):
        step_type = step_config.get('type')
        logger.debug(f"Step {i+1}/{len(steps)}: {step_type}")
        
        try:
            if step_type == 'retrieve':
                results = await _execute_retrieve_step(step_config, context)
                context['results'] = results
                
            elif step_type == 'parallel':
                results = await _execute_parallel_step(step_config, context)
                context['results'] = results
                
            elif step_type == 'process':
                results = _execute_process_step(step_config, context)
                context['results'] = results
                
            elif step_type == 'format':
                formatted = _execute_format_step(step_config, context)
                # Return results and formatted context
                return _results_to_dicts(context['results']), formatted
                
        except Exception as e:
            logger.error(f"Pipeline step {i+1} failed: {e}")
            raise
    
    # If no format step, return results with default formatting
    max_length = context['params'].get('max_context_length', 10000)
    formatted = FORMATTING_FUNCTIONS['format_as_context'](
        context['results'], 
        max_length=max_length
    )
    return _results_to_dicts(context['results']), formatted


async def _execute_retrieve_step(
    step_config: Dict[str, Any],
    context: PipelineContext
) -> List[SearchResult]:
    """Execute a retrieval step."""
    func_name = step_config.get('function')
    if not func_name or func_name not in RETRIEVAL_FUNCTIONS:
        raise ValueError(f"Unknown retrieval function: {func_name}")
    
    func = RETRIEVAL_FUNCTIONS[func_name]
    config = {**context['params'], **step_config.get('config', {})}
    
    # Special handling for different retrieval functions
    if func_name == 'parallel_search':
        return await func(
            context['app'],
            context['query'],
            context['sources'],
            step_config.get('functions', [])
        )
    elif func_name.endswith('_fts5'):
        # FTS5 functions don't take sources parameter
        return await func(
            context['app'],
            context['query'],
            config.get('top_k', 10),
            config.get('keyword_filter')
        )
    else:
        # Semantic search takes sources
        return await func(
            context['app'],
            context['query'],
            context['sources'],
            **config
        )


async def _execute_parallel_step(
    step_config: Dict[str, Any],
    context: PipelineContext
) -> List[SearchResult]:
    """Execute parallel retrieval functions."""
    functions = step_config.get('functions', [])
    if not functions:
        return []
    
    tasks = []
    for func_config in functions:
        func_name = func_config.get('function')
        if func_name and func_name in RETRIEVAL_FUNCTIONS:
            func = RETRIEVAL_FUNCTIONS[func_name]
            config = {**context['params'], **func_config.get('config', {})}
            
            # Create appropriate task based on function
            if func_name.endswith('_fts5'):
                # Only search_media_fts5 accepts keyword_filter
                if func_name == 'search_media_fts5':
                    task = func(
                        context['app'],
                        context['query'],
                        config.get('top_k', 10),
                        config.get('keyword_filter')
                    )
                else:
                    # search_conversations_fts5 and search_notes_fts5
                    task = func(
                        context['app'],
                        context['query'],
                        config.get('top_k', 10)
                    )
            else:
                task = func(
                    context['app'],
                    context['query'],
                    context['sources'],
                    **config
                )
            tasks.append(task)
    
    # Execute in parallel
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    all_results = []
    for results in results_lists:
        if isinstance(results, Exception):
            logger.error(f"Parallel search failed: {results}")
            continue
        all_results.extend(results)
    
    # Apply merge if specified
    merge_func = step_config.get('merge')
    if merge_func == 'weighted_merge' and 'weights' in step_config.get('config', {}):
        # Split results back into lists for weighted merge
        weights = step_config['config']['weights']
        if len(results_lists) == len(weights):
            valid_results = [r for r in results_lists if not isinstance(r, Exception)]
            return PROCESSING_FUNCTIONS['weighted_merge'](valid_results, weights)
    
    return all_results


def _execute_process_step(
    step_config: Dict[str, Any],
    context: PipelineContext
) -> List[SearchResult]:
    """Execute a processing step."""
    func_name = step_config.get('function')
    if not func_name or func_name not in PROCESSING_FUNCTIONS:
        raise ValueError(f"Unknown processing function: {func_name}")
    
    func = PROCESSING_FUNCTIONS[func_name]
    config = {**context['params'], **step_config.get('config', {})}
    
    # Special handling for different processing functions
    if func_name == 'rerank_results':
        return func(
            context['results'],
            context['query'],
            config.get('model', 'flashrank'),
            config.get('top_k', 10)
        )
    elif func_name == 'filter_by_score':
        return func(
            context['results'],
            config.get('min_score', 0.0)
        )
    else:
        # Generic processing
        return func(context['results'])


def _execute_format_step(
    step_config: Dict[str, Any],
    context: PipelineContext
) -> str:
    """Execute a formatting step."""
    func_name = step_config.get('function', 'format_as_context')
    if func_name not in FORMATTING_FUNCTIONS:
        raise ValueError(f"Unknown formatting function: {func_name}")
    
    func = FORMATTING_FUNCTIONS[func_name]
    config = step_config.get('config', {})
    
    # Only pass parameters that format_as_context accepts
    valid_params = ['max_length', 'include_citations', 'separator']
    format_params = {k: v for k, v in config.items() if k in valid_params}
    
    # Map max_context_length to max_length if present
    if 'max_context_length' in context['params']:
        format_params['max_length'] = context['params']['max_context_length']
    
    return func(context['results'], **format_params)


def _results_to_dicts(results: List[SearchResult]) -> List[Dict[str, Any]]:
    """Convert SearchResult objects to dictionaries, preserving citations if present."""
    result_dicts = []
    for r in results:
        result_dict = r.to_dict()
        
        # Check if citations are stored in metadata
        if result_dict.get('metadata', {}).get('_has_citations'):
            citations = result_dict['metadata'].pop('_citations', [])
            result_dict['metadata'].pop('_has_citations', None)
            result_dict['citations'] = citations
        
        result_dicts.append(result_dict)
    return result_dicts


# Pipeline configurations for backward compatibility
BUILTIN_PIPELINES = {
    'plain': {
        'name': 'Plain RAG Search',
        'steps': [
            {
                'type': 'parallel',
                'functions': [
                    {'function': 'search_media_fts5', 'config': {'top_k': 10}},
                    {'function': 'search_conversations_fts5', 'config': {'top_k': 10}},
                    {'function': 'search_notes_fts5', 'config': {'top_k': 10}}
                ]
            },
            {'type': 'process', 'function': 'deduplicate_results'},
            {'type': 'process', 'function': 'rerank_results'},
            {'type': 'format', 'function': 'format_as_context'}
        ]
    },
    'semantic': {
        'name': 'Semantic RAG Search',
        'steps': [
            {'type': 'retrieve', 'function': 'search_semantic'},
            {'type': 'process', 'function': 'rerank_results'},
            {'type': 'format', 'function': 'format_as_context'}
        ]
    },
    'hybrid': {
        'name': 'Hybrid RAG Search',
        'steps': [
            {
                'type': 'parallel',
                'functions': [
                    {'function': 'search_media_fts5', 'config': {'top_k': 20}},
                    {'function': 'search_conversations_fts5', 'config': {'top_k': 20}},
                    {'function': 'search_notes_fts5', 'config': {'top_k': 20}},
                    {'function': 'search_semantic', 'config': {'top_k': 20}}
                ],
                'merge': 'weighted_merge',
                'config': {'weights': [0.25, 0.25, 0.25, 0.25]}  # Equal weights
            },
            {'type': 'process', 'function': 'deduplicate_results'},
            {'type': 'process', 'function': 'rerank_results'},
            {'type': 'format', 'function': 'format_as_context'}
        ]
    }
}


# Load pipelines from TOML
_TOML_PIPELINES = None

def load_pipelines_from_toml():
    """Load pipeline configurations from TOML file."""
    global _TOML_PIPELINES
    if _TOML_PIPELINES is not None:
        return _TOML_PIPELINES
    
    _TOML_PIPELINES = {}
    
    # Check user config directory first
    user_config_dir = Path.home() / ".config" / "tldw_cli"
    user_config_file = user_config_dir / "rag_pipelines.toml"
    
    # Fall back to default location
    default_config_file = Path(__file__).parent.parent / "Config_Files" / "rag_pipelines.toml"
    
    config_file = None
    if user_config_file.exists():
        config_file = user_config_file
        logger.info(f"Loading pipeline config from user directory: {config_file}")
    elif default_config_file.exists():
        config_file = default_config_file
        logger.info(f"Loading pipeline config from default location: {config_file}")
        
        # Copy default file to user directory if it doesn't exist
        try:
            user_config_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(default_config_file, user_config_file)
            logger.info(f"Copied default pipeline config to user directory: {user_config_file}")
        except Exception as e:
            logger.warning(f"Could not copy default pipeline config to user directory: {e}")
    
    if not config_file:
        logger.warning("Pipeline config file not found in user or default locations")
        return _TOML_PIPELINES
    
    try:
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)
        
        pipelines = config_data.get("pipelines", {})
        
        # Convert TOML pipeline format to our simplified format
        for pipeline_id, pipeline_config in pipelines.items():
            if pipeline_config.get("type") == "functional" and pipeline_config.get("enabled", True):
                # Parse the functional pipeline format
                steps = []
                for step in pipeline_config.get("steps", []):
                    if step["type"] == "parallel":
                        # Handle parallel steps
                        functions = []
                        for func in step.get("functions", []):
                            # Map TOML function names to our function names
                            func_name = func['function']
                            if func_name == 'retrieve_fts5':
                                # This needs to be split into the three FTS5 functions
                                functions.extend([
                                    {'function': 'search_media_fts5', 'config': func.get('config', {})},
                                    {'function': 'search_conversations_fts5', 'config': func.get('config', {})},
                                    {'function': 'search_notes_fts5', 'config': func.get('config', {})}
                                ])
                            elif func_name == 'retrieve_semantic':
                                functions.append({
                                    'function': 'search_semantic',
                                    'config': func.get('config', {})
                                })
                            else:
                                functions.append({
                                    'function': func_name,
                                    'config': func.get('config', {})
                                })
                        steps.append({
                            'type': 'parallel',
                            'functions': functions,
                            'merge': step.get('merge', 'concat'),
                            'config': step.get('config', {})
                        })
                    else:
                        # Regular step - map function names
                        func_name = step.get('function')
                        if func_name == 'retrieve_fts5':
                            # For non-parallel retrieve_fts5, we need to handle this differently
                            # Convert to parallel step with all three FTS5 searches
                            steps.append({
                                'type': 'parallel',
                                'functions': [
                                    {'function': 'search_media_fts5', 'config': step.get('config', {})},
                                    {'function': 'search_conversations_fts5', 'config': step.get('config', {})},
                                    {'function': 'search_notes_fts5', 'config': step.get('config', {})}
                                ]
                            })
                        elif func_name == 'retrieve_semantic':
                            steps.append({
                                'type': step['type'],
                                'function': 'search_semantic',
                                'config': step.get('config', {})
                            })
                        else:
                            steps.append({
                                'type': step['type'],
                                'function': func_name,
                                'config': step.get('config', {})
                            })
                
                _TOML_PIPELINES[pipeline_id] = {
                    'name': pipeline_config['name'],
                    'description': pipeline_config.get('description', ''),
                    'steps': steps,
                    'parameters': pipeline_config.get('parameters', {})
                }
        
        logger.info(f"Loaded {len(_TOML_PIPELINES)} pipelines from TOML")
        
    except Exception as e:
        logger.error(f"Failed to load TOML pipelines: {e}")
    
    return _TOML_PIPELINES


def get_pipeline(pipeline_id: str) -> Optional[Dict[str, Any]]:
    """Get a pipeline configuration by ID."""
    # Check built-ins first
    if pipeline_id in BUILTIN_PIPELINES:
        return BUILTIN_PIPELINES[pipeline_id]
    
    # Load and check TOML pipelines
    toml_pipelines = load_pipelines_from_toml()
    if pipeline_id in toml_pipelines:
        return toml_pipelines[pipeline_id]
    
    # Also check if it's a non-functional pipeline that maps to a builtin
    # (e.g., "fast_search" -> "plain", "high_accuracy" -> "semantic")
    if pipeline_id == "fast_search":
        config = BUILTIN_PIPELINES['plain'].copy()
        config['parameters'] = {'top_k': 3, 'max_context_length': 5000, 'enable_rerank': False}
        return config
    elif pipeline_id == "high_accuracy":
        config = BUILTIN_PIPELINES['semantic'].copy()
        config['parameters'] = {
            'top_k': 20, 'max_context_length': 15000, 
            'chunk_size': 512, 'chunk_overlap': 128,
            'enable_rerank': True, 'reranker_model': 'cohere'
        }
        return config
    
    return None