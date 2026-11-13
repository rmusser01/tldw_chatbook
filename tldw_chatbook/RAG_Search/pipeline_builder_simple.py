"""
Simple pipeline builder for executing search pipelines from TOML configuration.

No complex error handling or effects - just execute steps in order.
"""

import asyncio
import copy
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import sys

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
from ..Backup_Recovery.storage_admission import acquire_storage
from pathlib import Path

from ..Chat.rag_scope import EffectiveScope
from .pipeline_types import SearchResult, PipelineContext
from .pipeline_functions_simple import (
    RETRIEVAL_FUNCTIONS,
    PROCESSING_FUNCTIONS,
    FORMATTING_FUNCTIONS,
)
from .fusion import (
    reciprocal_rank_fusion,
    interleave_rankings,
    resolve_hybrid_alpha,
    resolve_rrf_k,
)
from .semantic_availability import (
    record_semantic_unavailable,
    SEMANTIC_REASON_SEARCH_ERROR,
)

# Retrieval functions whose results form the vector (semantic) leg of hybrid
# fusion; every other retrieval function is treated as an FTS/keyword leg.
VECTOR_RETRIEVAL_FUNCTIONS = {"search_semantic"}


async def execute_pipeline(
    config: Dict[str, Any],
    app: Any,
    query: str,
    sources: Dict[str, bool],
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
    **params,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Execute a pipeline based on configuration.

    Args:
        config: Pipeline configuration from TOML
        app: TldwCli app instance
        query: Search query
        sources: Which sources to search
        diagnostics: Optional dict that rides through the PipelineContext and
            collects per-leg availability states (e.g. the semantic leg's
            unavailable/empty-index reasons, task-250). Callers that pass a
            dict can inspect it after the call; the (results, context) return
            shape is unchanged.
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-4). Seeded into ``PipelineContext["scope"]`` so every
            retrieval leg -- across builtin, parallel, and custom-TOML
            pipeline shapes -- self-enforces it identically, regardless of
            which step type invokes it. ``None`` performs today's
            unrestricted retrieval (real callers resolving and passing a
            live scope land in a follow-up task).
        **params: Additional parameters

    Returns:
        Tuple of (results list, formatted context)
    """
    pipeline_name = config.get("name", "Unknown Pipeline")
    steps = config.get("steps", [])

    logger.info(f"Executing pipeline: {pipeline_name}")

    # Initialize context
    context: PipelineContext = {
        "app": app,
        "query": query,
        "sources": sources,
        "params": {**config.get("parameters", {}), **params},
        "results": [],
        "diagnostics": diagnostics if diagnostics is not None else {},
        "scope": scope,
    }

    # Execute each step
    for i, step_config in enumerate(steps):
        step_type = step_config.get("type")
        logger.debug(f"Step {i + 1}/{len(steps)}: {step_type}")

        try:
            if step_type == "retrieve":
                results = await _execute_retrieve_step(step_config, context)
                context["results"] = results

            elif step_type == "parallel":
                results = await _execute_parallel_step(step_config, context)
                context["results"] = results

            elif step_type == "process":
                results = await _execute_process_step(step_config, context)
                context["results"] = results

            elif step_type == "format":
                formatted = _execute_format_step(step_config, context)
                # Return results and formatted context
                return _results_to_dicts(context["results"]), formatted

        except Exception as e:
            logger.error(f"Pipeline step {i + 1} failed: {e}")
            raise

    # If no format step, return results with default formatting
    max_length = context["params"].get("max_context_length", 10000)
    formatted = FORMATTING_FUNCTIONS["format_as_context"](
        context["results"], max_length=max_length
    )
    return _results_to_dicts(context["results"]), formatted


async def _execute_retrieve_step(
    step_config: Dict[str, Any], context: PipelineContext
) -> List[SearchResult]:
    """Execute a retrieval step."""
    func_name = step_config.get("function")
    if not func_name or func_name not in RETRIEVAL_FUNCTIONS:
        raise ValueError(f"Unknown retrieval function: {func_name}")

    func = RETRIEVAL_FUNCTIONS[func_name]
    config = {**context["params"], **step_config.get("config", {})}
    # Only forwarded when an active scope exists: keeps the unscoped call
    # shape to every leg byte-identical to before this parameter existed
    # (tests that monkeypatch RETRIEVAL_FUNCTIONS with narrow-signature
    # fakes -- e.g. Tests/RAG/test_fusion.py -- never pass scope, so they
    # never see the extra kwarg).
    scope = context.get("scope")
    scope_kwargs: Dict[str, Any] = {"scope": scope} if scope is not None else {}

    # Special handling for different retrieval functions
    if func_name == "parallel_search":
        return await func(
            context["app"],
            context["query"],
            context["sources"],
            step_config.get("functions", []),
            diagnostics=context.setdefault("diagnostics", {}),
            **scope_kwargs,
        )
    elif func_name.endswith("_fts5"):
        # FTS5 functions don't take sources parameter. All three FTS5 legs
        # now accept diagnostics (media/notes gained it alongside the
        # EMPTY-scope fail-closed guard, PR #734 review) -- forwarded only
        # when an active scope exists, matching scope_kwargs' own
        # zero-drift-for-unscoped-calls convention.
        fts5_kwargs: Dict[str, Any] = dict(scope_kwargs)
        if scope is not None:
            fts5_kwargs["diagnostics"] = context.setdefault("diagnostics", {})
        return await func(
            context["app"],
            context["query"],
            config.get("top_k", 10),
            config.get("keyword_filter"),
            **fts5_kwargs,
        )
    elif func_name == "search_semantic":
        # Vector leg: forward only kwargs the RAG service accepts. This is
        # the retrieve-step cousin of the task-256 parallel-step fix below:
        # splatting the full pipeline params (top_k alongside
        # search_semantic's own limit->top_k, chunk_*, include_citations,
        # ...) raised TypeError inside the RAG service call, so the pure
        # 'semantic' pipeline could never return vector results (task-250).
        semantic_kwargs = {
            key: config[key]
            for key in ("score_threshold", "filter_metadata")
            if key in config
        }
        return await func(
            context["app"],
            context["query"],
            context["sources"],
            limit=config.get("top_k", 10),
            diagnostics=context.setdefault("diagnostics", {}),
            **scope_kwargs,
            **semantic_kwargs,
        )
    else:
        # Generic retrieval function takes sources
        return await func(
            context["app"], context["query"], context["sources"], **config
        )


async def _execute_parallel_step(
    step_config: Dict[str, Any], context: PipelineContext
) -> List[SearchResult]:
    """Execute parallel retrieval functions."""
    functions = step_config.get("functions", [])
    if not functions:
        return []

    # Only forwarded when an active scope exists: keeps the unscoped call
    # shape to every leg byte-identical to before this parameter existed
    # (tests that monkeypatch RETRIEVAL_FUNCTIONS with narrow-signature
    # fakes -- e.g. Tests/RAG/test_fusion.py -- never pass scope, so they
    # never see the extra kwarg).
    scope = context.get("scope")
    scope_kwargs: Dict[str, Any] = {"scope": scope} if scope is not None else {}
    tasks = []
    task_func_names = []
    for func_config in functions:
        func_name = func_config.get("function")
        if func_name and func_name in RETRIEVAL_FUNCTIONS:
            func = RETRIEVAL_FUNCTIONS[func_name]
            config = {**context["params"], **func_config.get("config", {})}

            # Create appropriate task based on function
            if func_name.endswith("_fts5"):
                # All three FTS5 legs now accept diagnostics (media/notes
                # gained it alongside the EMPTY-scope fail-closed guard,
                # PR #734 review) -- forwarded only when an active scope
                # exists, matching scope_kwargs' own
                # zero-drift-for-unscoped-calls convention.
                fts5_kwargs = dict(scope_kwargs)
                if scope is not None:
                    fts5_kwargs["diagnostics"] = context.setdefault("diagnostics", {})
                # Only search_media_fts5 accepts keyword_filter
                if func_name == "search_media_fts5":
                    task = func(
                        context["app"],
                        context["query"],
                        config.get("top_k", 10),
                        config.get("keyword_filter"),
                        **fts5_kwargs,
                    )
                else:
                    # search_conversations_fts5 / search_notes_fts5
                    task = func(
                        context["app"],
                        context["query"],
                        config.get("top_k", 10),
                        **fts5_kwargs,
                    )
            elif func_name == "search_semantic":
                # Vector leg: forward only kwargs the RAG service accepts.
                # Splatting the full pipeline params (top_k alongside
                # search_semantic's own limit->top_k, chunk_*, ...) raises
                # TypeError inside search_semantic, which gather() swallows
                # and the vector leg comes back silently empty.
                semantic_kwargs = {
                    key: config[key]
                    for key in ("score_threshold", "filter_metadata")
                    if key in config
                }
                task = func(
                    context["app"],
                    context["query"],
                    context["sources"],
                    limit=config.get("top_k", 10),
                    diagnostics=context.setdefault("diagnostics", {}),
                    **scope_kwargs,
                    **semantic_kwargs,
                )
            else:
                task = func(
                    context["app"], context["query"], context["sources"], **config
                )
            tasks.append(task)
            task_func_names.append(func_name)

    # Execute in parallel
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    all_results = []
    for func_name, results in zip(task_func_names, results_lists):
        if isinstance(results, Exception):
            logger.error(f"Parallel search failed ({func_name}): {results}")
            if func_name in VECTOR_RETRIEVAL_FUNCTIONS:
                # The vector leg died mid-search; record it so callers can
                # honestly say the fused results are FTS-only (task-250).
                record_semantic_unavailable(
                    context.setdefault("diagnostics", {}),
                    SEMANTIC_REASON_SEARCH_ERROR,
                )
            continue
        all_results.extend(results)

    # Apply merge if specified
    merge_func = step_config.get("merge")
    if merge_func == "rrf_merge":
        return _rrf_merge_parallel_results(
            task_func_names,
            results_lists,
            step_config.get("config", {}) or {},
            context,
        )
    if merge_func == "weighted_merge" and "weights" in step_config.get("config", {}):
        # Split results back into lists for weighted merge
        weights = step_config["config"]["weights"]
        if len(results_lists) == len(weights):
            valid_results = [r for r in results_lists if not isinstance(r, Exception)]
            return PROCESSING_FUNCTIONS["weighted_merge"](valid_results, weights)

    return all_results


def _rrf_merge_parallel_results(
    func_names: List[str],
    results_lists: List[Any],
    merge_config: Dict[str, Any],
    context: PipelineContext,
) -> List[SearchResult]:
    """Fuse parallel retrieval legs via RRF + alpha (tldw_server parity).

    The FTS5 source lists (media, conversations, notes) are interleaved
    rank-fairly into a single FTS leg; ``search_semantic`` results form the
    vector leg. The two legs are fused with Reciprocal Rank Fusion and an
    alpha-weighted blend where alpha weights the vector leg (0 = FTS only,
    1 = vector only).

    Alpha precedence: step ``config.alpha`` -> ``hybrid_alpha`` runtime
    param -> ``[AppRAGSearchConfig.rag.retriever] hybrid_alpha`` -> 0.7.

    k precedence (TASK-4110 review, mirrors alpha's): step
    ``config.rrf_k`` -> the active profile's ``search.rrf_k`` -> shipped
    default (``config.DEFAULT_HYBRID_RRF_K``, 5)
    (``resolve_rrf_k``'s own fallback chain). Before this, ``rrf_k`` had NO
    profile fallback while ``alpha`` did, so a default change would move
    this (Chat RAG) hybrid path without moving
    ``RAGService._fuse_hybrid_results`` (Library hybrid), or vice versa --
    the two live fusion call sites disagreeing on a value measured on one.

    DISCLOSURE (TASK-4110 Task 5): the shipped ``SearchConfig.rrf_k``
    default is now 5, measured on the Library hybrid path, and this path
    picks it up through that profile fallback -- so in practice **this merge
    now fuses at k=5 too**, without ever having been measured here. That is
    deliberate: TASK-3501 consciously retained separate materializers while
    sharing the fusion primitive, resolved alpha/k, and provenance fields
    because the pipeline and engine have caller-specific identity,
    display-item, and citation policies.

    Args:
        func_names: Retrieval function name per results list, same order.
        results_lists: Per-function results (exceptions already logged).
        merge_config: The parallel step's merge config.
        context: Pipeline execution context.

    Returns:
        Fused results sorted by fused score descending.
    """
    fts_lists: List[List[SearchResult]] = []
    vector_lists: List[List[SearchResult]] = []
    for func_name, results in zip(func_names, results_lists):
        if isinstance(results, Exception):
            continue
        if func_name in VECTOR_RETRIEVAL_FUNCTIONS:
            vector_lists.append(results)
        else:
            fts_lists.append(results)

    def result_key(result: SearchResult):
        return (result.source, result.id)

    # EXEMPT from TASK-15700's form tiering, verified rather than assumed.
    # `RAGService._keyword_search` had to tier this same round robin because
    # its sub-legs can return rows from two different MATCH forms in one
    # query (a fallback construction widens a zero-row sub-leg), and a
    # fallback row taking leg rank 1 from an untouched primary row cost the
    # vector-blind fixture its hybrid rescue. This path's FTS legs are
    # `search_media_fts5` / `search_conversations_fts5` /
    # `search_notes_fts5`, which call the DB-level searches
    # (`media_db.search_media_db`, `db.search_conversations_by_content`,
    # `db.search_notes`). None of them reads
    # `SearchConfig.fts_match_construction`, none runs
    # `_fts5_match_expressions` / `_fts_rows_with_fallback`, and none stamps
    # `fts_match` -- there is no second MATCH form to partition. TASK-3501
    # intentionally retained this pipeline materializer; do not refactor it
    # speculatively.
    fts_leg = interleave_rankings(fts_lists, key=result_key)
    vector_leg = interleave_rankings(vector_lists, key=result_key)

    alpha = resolve_hybrid_alpha(
        merge_config.get("alpha", context["params"].get("hybrid_alpha"))
    )
    rrf_k = resolve_rrf_k(merge_config.get("rrf_k"))

    fused_entries = reciprocal_rank_fusion(
        fts_leg, vector_leg, key=result_key, alpha=alpha, rrf_k=rrf_k
    )

    fused_results = []
    for entry in fused_entries:
        result = entry.item
        fts_score = entry.fts_item.score if entry.fts_item is not None else None
        vector_score = (
            entry.vector_item.score if entry.vector_item is not None else None
        )
        result.score = entry.score
        merged_metadata = dict(result.metadata or {})
        # When a doc surfaced in both legs the FTS item is primary; carry the
        # other leg's citation metadata (search_semantic stashes citations as
        # _has_citations/_citations) so citations survive fusion and reach
        # _results_to_dicts.
        if entry.fts_item is not None and entry.vector_item is not None:
            other = entry.vector_item if result is entry.fts_item else entry.fts_item
            other_metadata = other.metadata or {}
            if other_metadata.get("_has_citations"):
                combined_citations = list(
                    merged_metadata.get("_citations") or []
                ) + list(other_metadata.get("_citations") or [])
                merged_metadata["_citations"] = combined_citations
                merged_metadata["_has_citations"] = True
        result.metadata = {
            **merged_metadata,
            "hybrid_fusion": {
                **entry.provenance(),
                "fts_score": fts_score,
                "vector_score": vector_score,
                "alpha": alpha,
                "rrf_k": rrf_k,
            },
        }
        fused_results.append(result)

    logger.debug(
        f"RRF fusion merged {len(fused_results)} results "
        f"(fts_leg={len(fts_leg)}, vector_leg={len(vector_leg)}, alpha={alpha}, k={rrf_k})"
    )
    return fused_results


async def _execute_process_step(
    step_config: Dict[str, Any], context: PipelineContext
) -> List[SearchResult]:
    """Execute a processing step.

    Async so CPU-heavy process bodies can run off the event loop:
    ``rerank_results`` (FlashRank model load + inference, the
    default-enabled step on every RAG chat send) is dispatched via
    ``asyncio.to_thread``. Called through the ``PROCESSING_FUNCTIONS``
    registry entry so monkeypatched fakes keep working.
    """
    func_name = step_config.get("function")
    if not func_name or func_name not in PROCESSING_FUNCTIONS:
        raise ValueError(f"Unknown processing function: {func_name}")

    func = PROCESSING_FUNCTIONS[func_name]
    config = {**context["params"], **step_config.get("config", {})}

    # Special handling for different processing functions
    if func_name == "rerank_results":
        return await asyncio.to_thread(
            func,
            context["results"],
            context["query"],
            config.get("model", "flashrank"),
            config.get("top_k", 10),
        )
    elif func_name == "filter_by_score":
        return func(context["results"], config.get("min_score", 0.0))
    else:
        # Generic processing
        return func(context["results"])


def _execute_format_step(step_config: Dict[str, Any], context: PipelineContext) -> str:
    """Execute a formatting step."""
    func_name = step_config.get("function", "format_as_context")
    if func_name not in FORMATTING_FUNCTIONS:
        raise ValueError(f"Unknown formatting function: {func_name}")

    func = FORMATTING_FUNCTIONS[func_name]
    config = step_config.get("config", {})

    # Only pass parameters that format_as_context accepts
    valid_params = ["max_length", "include_citations", "separator"]
    format_params = {k: v for k, v in config.items() if k in valid_params}

    # Map max_context_length to max_length if present
    if "max_context_length" in context["params"]:
        format_params["max_length"] = context["params"]["max_context_length"]

    return func(context["results"], **format_params)


def _results_to_dicts(results: List[SearchResult]) -> List[Dict[str, Any]]:
    """Convert SearchResult objects to dictionaries, preserving citations if present."""
    result_dicts = []
    for r in results:
        result_dict = r.to_dict()

        # Check if citations are stored in metadata
        if result_dict.get("metadata", {}).get("_has_citations"):
            citations = result_dict["metadata"].pop("_citations", [])
            result_dict["metadata"].pop("_has_citations", None)
            result_dict["citations"] = citations

        result_dicts.append(result_dict)
    return result_dicts


# Pipeline configurations for backward compatibility
BUILTIN_PIPELINES = {
    "plain": {
        "name": "Plain RAG Search",
        "steps": [
            {
                "type": "parallel",
                "functions": [
                    {"function": "search_media_fts5", "config": {"top_k": 10}},
                    {"function": "search_conversations_fts5", "config": {"top_k": 10}},
                    {"function": "search_notes_fts5", "config": {"top_k": 10}},
                ],
            },
            {"type": "process", "function": "deduplicate_results"},
            {"type": "process", "function": "rerank_results"},
            {"type": "format", "function": "format_as_context"},
        ],
    },
    "semantic": {
        "name": "Semantic RAG Search",
        "steps": [
            {"type": "retrieve", "function": "search_semantic"},
            {"type": "process", "function": "rerank_results"},
            {"type": "format", "function": "format_as_context"},
        ],
    },
    "hybrid": {
        "name": "Hybrid RAG Search",
        "steps": [
            {
                "type": "parallel",
                "functions": [
                    {"function": "search_media_fts5", "config": {"top_k": 20}},
                    {"function": "search_conversations_fts5", "config": {"top_k": 20}},
                    {"function": "search_notes_fts5", "config": {"top_k": 20}},
                    {"function": "search_semantic", "config": {"top_k": 20}},
                ],
                # Server-parity fusion: RRF (k=60) + alpha blend, alpha
                # weights the vector leg (0 = FTS only, 1 = vector only).
                # Alpha resolves from [AppRAGSearchConfig.rag.retriever]
                # hybrid_alpha (default 0.7) unless overridden here.
                "merge": "rrf_merge",
                "config": {},
            },
            {"type": "process", "function": "deduplicate_results"},
            {"type": "process", "function": "rerank_results"},
            {"type": "format", "function": "format_as_context"},
        ],
    },
}


# Load pipelines from TOML
_TOML_PIPELINES = None


def load_pipelines_from_toml():
    """Load pipeline configurations from TOML file."""
    global _TOML_PIPELINES
    if _TOML_PIPELINES is not None:
        return _TOML_PIPELINES

    from tldw_chatbook.config import _get_effective_config_path

    # Check user config directory first
    user_config_dir = _get_effective_config_path().parent
    user_config_file = user_config_dir / "rag_pipelines.toml"

    # Fall back to default location
    default_config_file = (
        Path(__file__).parent.parent / "Config_Files" / "rag_pipelines.toml"
    )

    config_file = None
    if user_config_file.exists():
        config_file = user_config_file
        logger.info(f"Loading pipeline config from user directory: {config_file}")
    elif default_config_file.exists():
        config_file = default_config_file
        logger.info(f"Loading pipeline config from default location: {config_file}")

        # Copy default file to user directory if it doesn't exist
        with acquire_storage(user_config_file):
            try:
                user_config_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(default_config_file, user_config_file)
                logger.info(
                    f"Copied default pipeline config to user directory: {user_config_file}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not copy default pipeline config to user directory: {e}"
                )

    _TOML_PIPELINES = {}

    if not config_file:
        logger.warning("Pipeline config file not found in user or default locations")
        return _TOML_PIPELINES

    try:
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)

        pipelines = config_data.get("pipelines", {})

        # Convert TOML pipeline format to our simplified format
        for pipeline_id, pipeline_config in pipelines.items():
            if pipeline_config.get("type") == "functional" and pipeline_config.get(
                "enabled", True
            ):
                # Parse the functional pipeline format
                steps = []
                for step in pipeline_config.get("steps", []):
                    if step["type"] == "parallel":
                        # Handle parallel steps
                        functions = []
                        for func in step.get("functions", []):
                            # Map TOML function names to our function names
                            func_name = func["function"]
                            if func_name == "retrieve_fts5":
                                # This needs to be split into the three FTS5 functions
                                functions.extend(
                                    [
                                        {
                                            "function": "search_media_fts5",
                                            "config": func.get("config", {}),
                                        },
                                        {
                                            "function": "search_conversations_fts5",
                                            "config": func.get("config", {}),
                                        },
                                        {
                                            "function": "search_notes_fts5",
                                            "config": func.get("config", {}),
                                        },
                                    ]
                                )
                            elif func_name == "retrieve_semantic":
                                functions.append(
                                    {
                                        "function": "search_semantic",
                                        "config": func.get("config", {}),
                                    }
                                )
                            else:
                                functions.append(
                                    {
                                        "function": func_name,
                                        "config": func.get("config", {}),
                                    }
                                )
                        steps.append(
                            {
                                "type": "parallel",
                                "functions": functions,
                                "merge": step.get("merge", "concat"),
                                "config": step.get("config", {}),
                            }
                        )
                    elif step["type"] == "merge":
                        # Fold a standalone merge step into the preceding
                        # parallel step: execute_pipeline has no 'merge' step
                        # type, so these were silently skipped before.
                        if steps and steps[-1].get("type") == "parallel":
                            steps[-1]["merge"] = step.get("function", "concat")
                            steps[-1]["config"] = step.get("config", {})
                        else:
                            logger.warning(
                                f"Pipeline '{pipeline_id}': merge step without a "
                                f"preceding parallel step is ignored"
                            )
                    else:
                        # Regular step - map function names
                        func_name = step.get("function")
                        if func_name == "retrieve_fts5":
                            # For non-parallel retrieve_fts5, we need to handle this differently
                            # Convert to parallel step with all three FTS5 searches
                            steps.append(
                                {
                                    "type": "parallel",
                                    "functions": [
                                        {
                                            "function": "search_media_fts5",
                                            "config": step.get("config", {}),
                                        },
                                        {
                                            "function": "search_conversations_fts5",
                                            "config": step.get("config", {}),
                                        },
                                        {
                                            "function": "search_notes_fts5",
                                            "config": step.get("config", {}),
                                        },
                                    ],
                                }
                            )
                        elif func_name == "retrieve_semantic":
                            steps.append(
                                {
                                    "type": step["type"],
                                    "function": "search_semantic",
                                    "config": step.get("config", {}),
                                }
                            )
                        else:
                            steps.append(
                                {
                                    "type": step["type"],
                                    "function": func_name,
                                    "config": step.get("config", {}),
                                }
                            )

                _TOML_PIPELINES[pipeline_id] = {
                    "name": pipeline_config["name"],
                    "description": pipeline_config.get("description", ""),
                    "steps": steps,
                    "parameters": pipeline_config.get("parameters", {}),
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
        config = BUILTIN_PIPELINES["plain"].copy()
        config["parameters"] = {
            "top_k": 3,
            "max_context_length": 5000,
            "enable_rerank": False,
        }
        return config
    elif pipeline_id == "high_accuracy":
        config = BUILTIN_PIPELINES["semantic"].copy()
        config["parameters"] = {
            "top_k": 20,
            "max_context_length": 15000,
            "chunk_size": 512,
            "chunk_overlap": 128,
            "enable_rerank": True,
            "reranker_model": "cohere",
        }
        return config

    return None

# Search entry points over the built-in and TOML pipelines. They lived in
# Event_Handlers/Chat_Events/chat_rag_events.py until TASK-36947.3; the
# pipeline loader maps its built-in function names onto them.


async def perform_plain_rag_search(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a plain RAG search using the pipeline system.

    Args:
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it; ``None`` performs today's unrestricted search.
    """
    logger.info("RAG pipeline starting; mode=plain")

    # Build pipeline configuration (deep copy: steps are nested dicts and the
    # builtin definition must not be mutated across calls)
    config = copy.deepcopy(BUILTIN_PIPELINES["plain"])
    config["parameters"] = {
        "top_k": top_k,
        "max_context_length": max_context_length,
        "keyword_filter": keyword_filter_list,
    }

    # Adjust reranking step if needed
    if not enable_rerank:
        # Remove rerank step
        config["steps"] = [
            s for s in config["steps"] if s.get("function") != "rerank_results"
        ]
    elif reranker_model != "flashrank":
        # Update reranker model
        for step in config["steps"]:
            if step.get("function") == "rerank_results":
                step.setdefault("config", {})["model"] = reranker_model

    # Execute pipeline
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, top_k=top_k
    )


async def perform_full_rag_pipeline(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    include_metadata: bool = True,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a full semantic RAG pipeline using the pipeline system.

    Args:
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it; ``None`` performs today's unrestricted search.
    """
    logger.info("RAG pipeline starting; mode=semantic")

    # Build pipeline configuration (deep copy: steps are nested dicts and the
    # builtin definition must not be mutated across calls)
    config = copy.deepcopy(BUILTIN_PIPELINES["semantic"])
    config["parameters"] = {
        "top_k": top_k,
        "max_context_length": max_context_length,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_type": chunk_type,
        "include_metadata": include_metadata,
        "include_citations": include_metadata,
    }

    # Adjust reranking step if needed
    if not enable_rerank:
        config["steps"] = [
            s for s in config["steps"] if s.get("function") != "rerank_results"
        ]
    elif reranker_model != "flashrank":
        for step in config["steps"]:
            if step.get("function") == "rerank_results":
                step.setdefault("config", {})["model"] = reranker_model

    # Execute pipeline
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, top_k=top_k
    )


async def perform_hybrid_rag_search(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    hybrid_alpha: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    keyword_filter_list: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform a hybrid RAG search using the pipeline system.

    The FTS5 and semantic legs are fused via Reciprocal Rank Fusion plus an
    alpha-weighted blend, matching the tldw_server design. Alpha weights the
    vector leg: 0 = FTS only, 1 = vector only. The RRF constant k resolves
    through ``fusion.resolve_rrf_k`` (step config -> active profile ->
    shipped default (``config.DEFAULT_HYBRID_RRF_K``)); that shipped
    default is 5 (TASK-4110).

    Alpha precedence: ``hybrid_alpha`` argument -> legacy ``bm25_weight`` /
    ``vector_weight`` (mapped to ``vector / (bm25 + vector)``) ->
    ``[AppRAGSearchConfig.rag.retriever] hybrid_alpha`` config knob (0.7).

    Args:
        app: The pipeline context ``execute_pipeline`` hands to each step.
        query: Search query text.
        sources: Which sources to search (e.g. media/conversations/notes).
        top_k: Maximum number of fused results to return.
        max_context_length: Character budget for the formatted context.
        enable_rerank: Whether to run the reranking step after fusion.
        reranker_model: Reranker model name when reranking is enabled.
        chunk_size: Chunk size forwarded to the pipeline parameters.
        chunk_overlap: Chunk overlap forwarded to the pipeline parameters.
        chunk_type: Chunking method forwarded to the pipeline parameters.
        hybrid_alpha: Explicit fusion alpha (0 = FTS only, 1 = vector only);
            overrides the legacy weights and the config knob.
        bm25_weight: Legacy FTS-leg weight; mapped onto alpha together with
            vector_weight when hybrid_alpha is not given.
        vector_weight: Legacy vector-leg weight; see bm25_weight.
        keyword_filter_list: Optional keywords the media FTS leg must match.
        diagnostics: Optional dict receiving the semantic-leg availability
            state (task-250) so callers can say when results are FTS-only.
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it; ``None`` performs today's unrestricted search.

    Returns:
        Tuple of (result dicts sorted by fused score, formatted context
        string for the LLM).
    """
    logger.info("RAG pipeline starting; mode=hybrid")

    # Map legacy weight parameters onto alpha for backwards compatibility
    if hybrid_alpha is None and (bm25_weight is not None or vector_weight is not None):
        bm25 = bm25_weight if bm25_weight is not None else 0.5
        vector = vector_weight if vector_weight is not None else 0.5
        total_weight = bm25 + vector
        if total_weight > 0:
            hybrid_alpha = vector / total_weight
    alpha = resolve_hybrid_alpha(hybrid_alpha)

    # Build pipeline configuration (deep copy: steps are nested dicts and the
    # builtin definition must not be mutated across calls)
    config = copy.deepcopy(BUILTIN_PIPELINES["hybrid"])

    # Update config
    config["parameters"] = {
        "top_k": top_k,
        "max_context_length": max_context_length,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_type": chunk_type,
        "keyword_filter": keyword_filter_list,
    }

    # Pin the resolved alpha on the fusion step
    for step in config["steps"]:
        if step.get("type") == "parallel" and step.get("merge") == "rrf_merge":
            step["config"] = {**step.get("config", {}), "alpha": alpha}

    # Adjust reranking step if needed
    if not enable_rerank:
        config["steps"] = [
            s for s in config["steps"] if s.get("function") != "rerank_results"
        ]
    elif reranker_model != "flashrank":
        for step in config["steps"]:
            if step.get("function") == "rerank_results":
                step.setdefault("config", {})["model"] = reranker_model

    # Execute pipeline
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, top_k=top_k
    )


async def perform_search_with_pipeline(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    pipeline_id: str,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a search using a specific pipeline ID.

    This allows using custom pipelines defined in TOML files.

    Args:
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it, including custom TOML pipeline shapes;
            ``None`` performs today's unrestricted search.
    """
    logger.info("RAG pipeline starting; mode=custom")

    # Get pipeline configuration (from built-ins or TOML)
    config = get_pipeline(pipeline_id)

    if not config:
        logger.error(
            "RAG pipeline unavailable; reason=pipeline_not_found; "
            f"pipeline_id={pipeline_id}; "
            f"selected_source_count={sum(bool(value) for value in sources.values())}"
        )
        return [], f"Pipeline '{pipeline_id}' not found"

    # Make a copy to avoid modifying the original
    config = config.copy()

    # Merge any pipeline-specific parameters with runtime parameters
    pipeline_params = config.get("parameters", {})
    merged_params = {**pipeline_params, **kwargs}

    # Execute pipeline with merged parameters
    return await execute_pipeline(
        config,
        app,
        query,
        sources,
        diagnostics=diagnostics,
        scope=scope,
        **merged_params,
    )


def format_results_for_llm(
    results: List[Dict[str, Any]], max_chars: int = 10000
) -> str:
    """Format search results as context for LLM."""
    from .pipeline_functions_simple import format_as_context

    # Convert dicts back to SearchResult objects
    search_results = []
    for r in results:
        search_results.append(
            SearchResult(
                source=r["source"],
                id=r["id"],
                title=r["title"],
                content=r["content"],
                score=r.get("score", 1.0),
                metadata=r.get("metadata", {}),
            )
        )

    return format_as_context(search_results, max_chars)
