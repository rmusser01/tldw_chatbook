"""
WebSearch_APIs.py
=================

Web Search API Integration Module

This module provides a unified interface for performing web searches across
multiple search engine APIs and processing the results with LLM-powered
analysis and summarization.

Supported Search Engines:
------------------------
- Google Custom Search
- Bing Search
- DuckDuckGo
- Brave Search
- Kagi
- Tavily
- SearX
- Baidu (partial support)
- Yandex (partial support)

Key Features:
------------
- Sub-query generation for comprehensive searches
- Result relevance analysis with LLMs
- Automatic summarization and aggregation
- Standardized result format across engines
- Rate limiting and error handling

Main Functions:
--------------
- generate_and_search(): Generate sub-queries and perform searches
- analyze_and_aggregate(): Analyze relevance and create final answer
- perform_websearch(): Execute search on specified engine
- search_result_relevance(): Evaluate result relevance using LLM
"""

# Imports
import asyncio
import base64
import concurrent.futures
import json
import random
import re
import threading
import time
from functools import wraps
from html import unescape
from typing import Any, Callable, Dict, List, NotRequired, Optional, TypedDict, Union
from urllib.parse import unquote, urlencode, urlparse

#
# 3rd-Party Imports
import requests
from requests import RequestException
from requests.adapters import HTTPAdapter
from urllib3 import Retry

# Handle optional lxml dependency
try:
    from lxml.etree import _Element
    from lxml.html import document_fromstring

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    _Element = None
    document_fromstring = None

#
# Local Imports
# `analyze` (LLM_Calls.Summarization_General_Lib) pulls in the summarization
# stack (nltk/scipy/sklearn/pandas via Chunking/Chunk_Lib). It is imported
# lazily inside search_result_relevance(), only when actually summarizing a
# relevant result, so a plain `import tldw_chatbook.app` doesn't eagerly
# load it (this module sits on the app.py -> Tools -> WebSearch_APIs boot
# path via the tool-executor registry).
from loguru import logger

from tldw_chatbook.Chat.Chat_Functions import chat_api_call, chat_reply_text
from tldw_chatbook.config import load_settings
from tldw_chatbook.Internal_Prompts import render_internal_prompt
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Utils.egress import is_public_http_url
from tldw_chatbook.Web_Scraping import deep_search_citations
from tldw_chatbook.Web_Scraping.Article_Extractor_Lib import scrape_article

# Handle optional defusedxml (Yandex XML parsing)
try:
    import defusedxml.ElementTree as _yandex_ET
except ImportError:
    import xml.etree.ElementTree as _yandex_ET

    logger.warning(
        "defusedxml not available, using standard xml.etree for Yandex result parsing. "
        "Install defusedxml for better security."
    )


# Common error handling and retry mechanisms
def handle_search_error(error, search_engine_name):
    """
    Common error handling function for search engine errors.

    Args:
        error: The exception that was raised
        search_engine_name: Name of the search engine (e.g., 'Bing', 'Google')

    Returns:
        Appropriate exception with detailed error message

    This function categorizes errors and provides consistent error handling
    across all search engine implementations.
    """
    logger.error(f"{search_engine_name} search error: {error}")

    # Handle timeout errors
    if isinstance(error, requests.exceptions.Timeout):
        return TimeoutError(
            f"{search_engine_name} search request timed out. Please try again later."
        )

    # Handle connection errors
    if isinstance(error, requests.exceptions.ConnectionError):
        return ConnectionError(
            f"Network error while connecting to {search_engine_name}: {error}"
        )

    # Handle HTTP errors
    if isinstance(error, requests.exceptions.HTTPError):
        status_code = (
            error.response.status_code
            if hasattr(error, "response") and error.response
            else "unknown"
        )

        if status_code == 401:
            return ValueError(
                f"Invalid {search_engine_name} API key. Please check your configuration."
            )
        elif status_code == 403:
            return ValueError(
                f"Access denied. Your {search_engine_name} API key may not have permission for this operation."
            )
        elif status_code == 429:
            return ValueError(
                f"{search_engine_name} API rate limit exceeded. Please try again later."
            )
        else:
            return RequestException(
                f"HTTP error during {search_engine_name} search: {error}"
            )

    # Handle value errors
    if isinstance(error, ValueError):
        return ValueError(f"Invalid parameter for {search_engine_name} search: {error}")

    # Handle JSON decode errors
    if isinstance(error, json.JSONDecodeError):
        return ValueError(
            f"Invalid response from {search_engine_name} (not valid JSON): {error}"
        )

    # Handle any other errors
    return Exception(f"Error performing {search_engine_name} search: {error}")


# Retry decorator for transient errors
def retry_on_transient_error(max_tries=3, backoff_factor=1.5):
    """
    Decorator to retry functions on transient errors.

    Args:
        max_tries: Maximum number of retry attempts
        backoff_factor: Factor to increase delay between retries

    Returns:
        Decorated function that will retry on transient errors

    This decorator will retry the function when it raises specific exceptions
    that are likely to be transient (e.g., connection errors, timeouts).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Define which exceptions should trigger a retry
            retry_exceptions = (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            )

            # Define a function to determine if we should retry
            def should_retry(exception):
                if isinstance(exception, requests.exceptions.HTTPError):
                    # Only retry on 5xx errors (server errors) and 429 (rate limit)
                    if hasattr(exception, "response") and exception.response:
                        status_code = exception.response.status_code
                        return status_code >= 500 or status_code == 429
                    return False
                return isinstance(exception, retry_exceptions)

            # Initialize retry count
            retry_count = 0

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1

                    # Check if we should retry
                    if retry_count >= max_tries or not should_retry(e):
                        # We've reached max retries or this is not a retryable error
                        raise

                    # Calculate delay with exponential backoff
                    delay = backoff_factor ** (retry_count - 1)
                    logger.warning(
                        f"Retrying {func.__name__} after error: {e} (attempt {retry_count}/{max_tries}, delay: {delay:.2f}s)"
                    )

                    # Wait before retrying
                    time.sleep(delay)

        return wrapper

    return decorator


#
#######################################################################################################################
#
# Functions:
# 1. analyze_question
#
#######################################################################################################################
#
# Functions:


# Initialize configuration data
def initialize_config():
    """
    Initialize the configuration data from config.py.

    Returns:
        Dict: A dictionary containing the configuration data.
    """
    config_data = load_settings()

    # Create a search_engines section that matches the expected structure
    search_engines = {}

    # Copy settings from SearchEngines section
    if "SearchEngines" in config_data:
        for key, value in config_data["SearchEngines"].items():
            search_engines[key] = value

    # Copy settings from search_engine_specific_settings section
    if "search_engine_specific_settings" in config_data:
        for key, value in config_data["search_engine_specific_settings"].items():
            search_engines[key] = value

    # Copy settings from search_engines_keys section
    if "search_engines_keys" in config_data:
        for key, value in config_data["search_engines_keys"].items():
            search_engines[key] = value

    # Create a new config dictionary with the search_engines section
    result = {"search_engines": search_engines}

    return result


# Load configuration data
loaded_config_data = initialize_config()
######################### Main Orchestration Workflow #########################
#
# FIXME - Add Logging


def initialize_web_search_results_dict(search_params: Dict) -> Dict:
    """
    Initializes and returns a dictionary for storing web search results and metadata.

    Args:
        search_params (Dict): A dictionary containing search parameters.

    Returns:
        Dict: A dictionary initialized with search metadata.
    """
    return {
        "search_engine": search_params.get("engine", "google"),
        "search_query": "",
        "content_country": search_params.get("content_country", "US"),
        "search_lang": search_params.get("search_lang", "en"),
        "output_lang": search_params.get("output_lang", "en"),
        "result_count": 0,
        "date_range": search_params.get("date_range"),
        "safesearch": search_params.get("safesearch", "active"),
        "site_blacklist": search_params.get("site_blacklist", []),
        "exactTerms": search_params.get("exactTerms"),
        "excludeTerms": search_params.get("excludeTerms"),
        "filter": search_params.get("filter"),
        "geolocation": search_params.get("geolocation"),
        "search_result_language": search_params.get("search_result_language"),
        "sort_results_by": search_params.get("sort_results_by"),
        "results": [],
        "total_results_found": 0,
        "search_time": 0.0,
        "error": None,
        "processing_error": None,
    }


def _sanitize_sub_questions(raw_values: Any) -> List[str]:
    """Normalize model-generated sub-questions into a deduplicated list of strings.

    Ported from tldw_server2's WebSearch_APIs._sanitize_sub_questions (~:255-286),
    extended to also accept a dict carrying "sub_questions"/"search_queries" at the
    top level -- folding in the unwrapping the server does at its analyze_question
    call site (~:751-754) so callers can hand this function a raw parsed LLM
    response directly, list or dict, without pre-extracting the list themselves.

    Args:
        raw_values: A list/tuple/set of str or dict items, a single string, a dict
            carrying "sub_questions" or "search_queries", or None/anything else.

    Returns:
        A list of stripped, non-empty strings with case-insensitive duplicates
        dropped (first occurrence wins).
    """
    if isinstance(raw_values, dict):
        raw_values = raw_values.get("sub_questions", raw_values.get("search_queries", []))

    if isinstance(raw_values, str):
        candidates: List[Any] = [raw_values]
    elif isinstance(raw_values, (list, tuple, set)):
        candidates = list(raw_values)
    else:
        return []

    sanitized: List[str] = []
    seen: set = set()
    for item in candidates:
        text = ""
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            query_value = item.get("sub_question")
            if not isinstance(query_value, str):
                query_value = item.get("query")
            if not isinstance(query_value, str):
                query_value = item.get("text")
            if isinstance(query_value, str):
                text = query_value.strip()
        else:
            continue

        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(text)
    return sanitized


#: search_params keys generate_and_search cannot run without. Exported
#: because callers that ASSEMBLE these params need to be able to check their
#: own assembly before spending a run on it -- LocalResearchEngine's
#: pre-flight (task-17371) does exactly that, after a window-launched run
#: reaching this validation was the only signal that the window passed no
#: params at all.
#: Failure markers the summarizers use when they report by RETURNING a string
#: instead of raising (task-17382). Matched case-insensitively against the
#: START of the value only -- these functions put the marker in the first
#: clause ("Llama: Error occurred while ...", "Ollama: JSON parse error ..."),
#: so a legitimate summary that merely discusses errors downstream cannot be
#: mistaken for one. The old guard tested only the "Error:" prefix, which no
#: provider-prefixed message matches.
_SUMMARY_FAILURE_MARKERS = (
    "error",
    "unexpected error",
    "api request failed",
    "json parse error",
    "no choices in response",
    "failed to",
)

#: How far into the value a marker still counts as the leading clause.
_SUMMARY_FAILURE_PREFIX_CHARS = 60


def _is_summary_failure(value: Any) -> bool:
    """Whether a summarizer's return value is one of its error strings.

    Args:
        value: The summarizer's return value (any type; only str can fail).

    Returns:
        True when the value is a provider failure report rather than a summary.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    head = text[:_SUMMARY_FAILURE_PREFIX_CHARS].casefold()
    # Either the message leads with the marker, or a provider prefix does:
    # "<provider>: <marker> ...". Split once so prose containing a colon
    # later on cannot smuggle a marker in.
    candidates = [head]
    if ":" in head:
        candidates.append(head.split(":", 1)[1].lstrip())
    return any(
        candidate.startswith(marker)
        for candidate in candidates
        for marker in _SUMMARY_FAILURE_MARKERS
    )


GENERATE_AND_SEARCH_REQUIRED_PARAMS = (
    "engine",
    "content_country",
    "search_lang",
    "output_lang",
    "result_count",
)


def generate_and_search(question: str, search_params: Dict) -> Dict:
    """
    Generate sub-queries and perform web searches.

    This function orchestrates the search process by:
    1. Optionally generating sub-queries from the main question
    2. Performing searches on all queries
    3. Accumulating results in a standardized format

    Args:
        question (str): The main search query/question
        search_params (Dict): Search configuration containing:
            - engine: Search engine name (google, bing, etc.)
            - content_country: Country code for results
            - search_lang: Language for search
            - output_lang: Language for output
            - result_count: Number of results per query
            - subquery_generation: Enable sub-query generation
            - subquery_generation_llm: LLM for generating sub-queries
            - Additional engine-specific parameters

    Returns:
        Dict: Contains:
            - web_search_results_dict: All search results
            - sub_query_dict: Generated sub-queries and metadata

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> params = {
        ...     "engine": "google",
        ...     "content_country": "US",
        ...     "search_lang": "en",
        ...     "result_count": 10,
        ...     "subquery_generation": True
        ... }
        >>> results = generate_and_search("quantum computing", params)

    Generates sub-queries (if enabled) and performs web searches for each query.

    Args:
        question (str): The user's original question or query.
        search_params (Dict): A dictionary containing parameters for performing web searches
                              and specifying LLM endpoints.

    Returns:
        Dict: A dictionary containing all search results and related metadata.

    Raises:
        ValueError: If the input parameters are invalid.
    """
    time.time()
    logger.info(f"Starting generate_and_search with query: {question}")

    # Log search attempt
    engine = search_params.get("engine", "unknown")
    log_counter(
        "websearch_generate_and_search_attempt",
        labels={
            "engine": engine,
            "subquery_generation": str(search_params.get("subquery_generation", False)),
        },
    )

    # Validate input parameters
    if not question or not isinstance(question, str):
        raise ValueError("Invalid question parameter")
    if not search_params or not isinstance(search_params, dict):
        raise ValueError("Invalid search_params parameter")

    # Check for required keys in search_params
    for key in GENERATE_AND_SEARCH_REQUIRED_PARAMS:
        if key not in search_params:
            raise ValueError(f"Missing required key in search_params: {key}")

    # 1. Generate sub-queries if requested
    logger.info(f"Generating sub-queries for the query: {question}")
    sub_query_dict = {
        "main_goal": question,
        "sub_questions": [],
        "search_queries": [],
        "analysis_prompt": None,
    }

    # Set only when subquery_generation is on AND analyze_question exhausted
    # every attempt without producing a single sub-question (task-3221): up
    # to _SUBQUERY_GENERATION_MAX_ATTEMPTS paid LLM calls were made and none
    # of them yielded anything, which is otherwise indistinguishable from
    # the feature being off. Appended to warnings below, once the dict
    # exists.
    subquery_generation_failure_warning: Optional[str] = None
    if search_params.get("subquery_generation", False):
        logger.info("Sub-query generation enabled")
        api_endpoint = search_params.get("subquery_generation_llm", "openai")
        sub_query_dict = analyze_question(question, api_endpoint)
        if not sub_query_dict.get("sub_questions"):
            subquery_generation_failure_warning = (
                f"sub-query generation failed after "
                f"{_SUBQUERY_GENERATION_MAX_ATTEMPTS} attempts; searched only "
                "the original query"
            )
            logger.warning(subquery_generation_failure_warning)

    # Merge original question with sub-queries, dropping any sub-query that's
    # just the original question again (case-insensitive) before fan-out
    # (port of server generate_and_search ~:522-528).
    sub_queries = _sanitize_sub_questions(sub_query_dict.get("sub_questions", []))
    question_key = question.strip().casefold()
    sub_queries = [
        sub_query for sub_query in sub_queries if sub_query.strip().casefold() != question_key
    ]

    # Cap total fan-out at search_default_max_queries (final review,
    # Important 2): the resolved cap was never applied here, so an LLM
    # returning e.g. 12 sub-questions fanned out to 13 total searches --
    # far past the tool description's "~25 LLM calls at defaults" budget
    # when subquery generation is enabled. The original question always
    # counts as one query, so sub-queries are truncated to cap - 1. Read
    # from search_params (the same pydantic-safe route the timeouts use --
    # WebSearchRequest drops unknown fields, so this value travels INSIDE
    # search_params), coerced defensively like the timeouts above (float(...
    # or default) below).
    max_queries_cap = int(search_params.get("search_default_max_queries", 5) or 5)
    sub_queries = sub_queries[: max(0, max_queries_cap - 1)]

    sub_query_dict["sub_questions"] = sub_queries
    sub_query_dict["search_queries"] = sub_queries
    logger.info(f"Sub-queries generated: {sub_queries}")
    all_queries = [question] + sub_queries

    # 2. Initialize a single web_search_results_dict
    web_search_results_dict = initialize_web_search_results_dict(search_params)
    web_search_results_dict["search_query"] = question
    web_search_results_dict["warnings"] = []

    # 3. Perform searches and accumulate all raw results
    #
    # Cheap between-queries deadline bound (final review, Important 3a): NO
    # search backend except serper/exa/yandex/bing sets a request timeout, so
    # a hung phase-1 socket could previously push past the caller's overall
    # deep-search deadline unbounded -- the runtime would then abandon the
    # whole worker with a bare timeout instead of the tool's own honest
    # partial-results path. This bounds everything EXCEPT one in-flight
    # request: the caller places its remaining phase-1 budget into
    # search_params (e.g. "phase1_time_budget_s"); checked BEFORE each
    # per-query call, coerced defensively like the timeouts elsewhere in
    # this module. Absent (None) -> no bound, unchanged behavior for any
    # caller that doesn't opt in.
    phase1_start = time.monotonic()
    phase1_budget_raw = search_params.get("phase1_time_budget_s")
    try:
        phase1_budget = float(phase1_budget_raw) if phase1_budget_raw is not None else None
    except (TypeError, ValueError):
        phase1_budget = None

    for n, q in enumerate(all_queries):
        if phase1_budget is not None and (time.monotonic() - phase1_start) >= phase1_budget:
            warning = (
                f"deadline reached during search fan-out; searched {n} of {len(all_queries)} queries"
            )
            logger.warning(warning)
            web_search_results_dict["warnings"].append(warning)
            break
        random.uniform(1, 1.5)  # Add a random delay to avoid rate limiting
        logger.info(f"Performing web search for query: {q}")
        raw_results = perform_websearch(
            search_engine=search_params.get("engine"),
            search_query=q,
            content_country=search_params.get("content_country", "US"),
            search_lang=search_params.get("search_lang", "en"),
            output_lang=search_params.get("output_lang", "en"),
            result_count=search_params.get("result_count", 10),
            date_range=search_params.get("date_range"),
            safesearch=search_params.get("safesearch", "active"),
            site_blacklist=search_params.get("site_blacklist", []),
            exactTerms=search_params.get("exactTerms"),
            excludeTerms=search_params.get("excludeTerms"),
            filter=search_params.get("filter"),
            geolocation=search_params.get("geolocation"),
            search_result_language=search_params.get("search_result_language"),
            sort_results_by=search_params.get("sort_results_by"),
        )

        # Debug: Inspect raw results
        logger.debug(f"Raw results for query '{q}': {raw_results}")

        # Check for errors or invalid data
        if not isinstance(raw_results, dict) or raw_results.get("processing_error"):
            logger.error(
                f"Error or invalid data returned for query '{q}': {raw_results}"
            )
            if isinstance(raw_results, dict):
                error_text = str(raw_results.get("processing_error") or "").strip()
                if error_text:
                    web_search_results_dict["warnings"].append(f"{q!r}: {error_text}")
            continue

        logger.info(
            f"Search results found for query '{q}': {len(raw_results.get('results', []))}"
        )

        # Append results to the single web_search_results_dict
        web_search_results_dict["results"].extend(raw_results["results"])
        web_search_results_dict["total_results_found"] += raw_results.get(
            "total_results_found", 0
        )
        web_search_results_dict["search_time"] += raw_results.get("search_time", 0.0)
        logger.info(
            f"Total results found so far: {len(web_search_results_dict['results'])}"
        )

    # Promote a provider error to the top-level `error` field BEFORE the
    # sub-query-generation notice (below) ever joins `warnings` (fix-wave
    # Important 2, 2026-08-07 review). The old code appended that notice at
    # warnings[0] -- BEFORE the fan-out loop even ran -- which this
    # promotion check then treated as the first-provider-error slot. Two
    # regressions followed: (1) a REAL provider error landing at warnings[1]
    # got silently demoted/misattributed as the sub-query notice instead,
    # and (2) an empty-without-error run (every provider legitimately
    # returned zero results, no processing_error at all) got a FALSE
    # top-level `error` -- the notice itself isn't a provider error and must
    # never be promoted to one, including when it's the only entry.
    # Evaluating this check against only the loop's own provider warnings --
    # before the notice is appended just below -- keeps `warnings[0]`
    # meaning exactly what the port comment says (the first provider
    # error) and leaves `error` at its None default whenever no provider
    # actually errored, sub-query notice or not.
    #
    # If every query came back empty and at least one provider errored,
    # surface the first such error as the top-level error (port of server
    # generate_and_search ~:624-627).
    if not web_search_results_dict["results"] and web_search_results_dict["warnings"]:
        web_search_results_dict["error"] = web_search_results_dict["warnings"][0]

    # Append the sub-query-generation-exhausted notice LAST -- after the
    # promotion check above has already run -- so it rides at the end of
    # `warnings` for the caller's warning COUNT/footer display, exactly
    # like any other non-error warning, without ever being eligible for
    # promotion to `error` itself.
    if subquery_generation_failure_warning:
        web_search_results_dict["warnings"].append(subquery_generation_failure_warning)

    return {
        "web_search_results_dict": web_search_results_dict,
        "sub_query_dict": sub_query_dict,
    }


async def analyze_and_aggregate(
    web_search_results_dict: Dict,
    sub_query_dict: Dict,
    search_params: Dict,
    *,
    cancel_event: Optional[asyncio.Event] = None,
) -> Dict:
    """
    Analyze search results for relevance and create a final aggregated answer.

    This function:
    1. Scores/filters results for relevance
    2. Scrapes full content from relevant URLs
    3. Summarizes relevant content
    4. Aggregates into a comprehensive answer

    Args:
        web_search_results_dict (Dict): Raw search results from generate_and_search
        sub_query_dict (Dict): Sub-queries and metadata
        search_params (Dict): Search configuration with:
            - relevance_analysis_llm: LLM for relevance scoring
            - final_answer_llm: LLM for final aggregation
            - user_review: Enable manual result selection
            - relevance_llm_timeout_s / relevance_scrape_timeout_s: per-call
              timeouts passed through to search_result_relevance (default 30/30)
            - respect_robots_txt: bool, default False when absent -- passed
              through to search_result_relevance's pre-scrape robots.txt
              consult (task-3260). Absence (the dead-wired research-service
              caller) keeps today's no-robots-check behavior; web_deep_search
              (the tool) always sets this from the real [webfetch] setting.
        cancel_event: Optional cooperative-cancellation flag passed through to
            search_result_relevance (port of server WebSearch_APIs.py :638;
            task-1356).

    Returns:
        Dict: Contains:
            - final_answer: Aggregated answer with citations (FinalAnswerDict)
            - relevant_results: Filtered relevant results
            - web_search_results_dict: Original search results

    Example:
        >>> # In an async function:
        >>> # final_results = await analyze_and_aggregate(
        >>> #     phase1_results["web_search_results_dict"],
        >>> #     phase1_results["sub_query_dict"],
        >>> #     search_params
        >>> # )
    """
    start_time = time.time()
    logger.info("Starting analyze_and_aggregate")

    # 4. Score/filter results
    logger.info("Scoring and filtering search results")
    sub_questions = sub_query_dict.get("sub_questions", [])
    relevance_llm_timeout_s = float(search_params.get("relevance_llm_timeout_s", 30.0) or 30.0)
    relevance_scrape_timeout_s = float(search_params.get("relevance_scrape_timeout_s", 30.0) or 30.0)
    # task-3260: default False when the key is absent -- the dead-wired
    # research-service caller (never sets this) keeps today's no-robots-
    # check behavior; web_deep_search (the tool) always places the real
    # [webfetch] respect_robots_txt setting here.
    raw_respect_robots = search_params.get("respect_robots_txt", False)
    # Strict parse (Qodo PR #1451 — the fourth bool("false") catch of this
    # arc): a stringly caller's "false" must not ENABLE enforcement. Bools
    # pass through; only "true"/"1" strings enable; anything else is False.
    if isinstance(raw_respect_robots, bool):
        respect_robots_txt = raw_respect_robots
    elif isinstance(raw_respect_robots, str):
        respect_robots_txt = raw_respect_robots.strip().lower() in ("true", "1")
    else:
        respect_robots_txt = False
    relevant_results = await search_result_relevance(
        web_search_results_dict["results"],
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get("relevance_analysis_llm"),
        cancel_event=cancel_event,
        llm_timeout_s=relevance_llm_timeout_s,
        scrape_timeout_s=relevance_scrape_timeout_s,
        respect_robots_txt=respect_robots_txt,
    )
    # FIXME
    logger.debug("Relevant results returned by search_result_relevance:")
    logger.debug(json.dumps(relevant_results, indent=2))

    # 5. Allow user to review and select relevant results (if enabled)
    logger.info("Reviewing and selecting relevant results")
    if search_params.get("user_review", False):
        logger.info("User review enabled")
        relevant_results = review_and_select_results(relevant_results)

    # 6. Summarize/aggregate final answer
    # Offloaded to a worker thread (task-1356 review fix): aggregate_results
    # is SYNCHRONOUS (chat_api_call/_analyze do blocking HTTP) -- calling it
    # directly here would block this coroutine's event loop thread for the
    # whole call, starving any outer asyncio.wait_for's timeout callback of
    # a chance to ever run (a callback scheduled via call_later cannot fire
    # while the loop is stuck inside a synchronous call), so a caller's
    # deadline+grace backstop around this whole function could never
    # actually cut in -- it would just return late. Mirrors
    # search_result_relevance's own asyncio.to_thread(chat_api_call, ...)
    # calls above, which offload the same kind of blocking call for the
    # same reason.
    final_answer = await asyncio.to_thread(
        aggregate_results,
        relevant_results,
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get("final_answer_llm"),
    )

    # 7. Return the final data
    logger.info("Returning final websearch results")

    # Define engine and all_queries for metrics logging
    engine = search_params.get("engine", "unknown")
    main_goal = sub_query_dict.get("main_goal", "")
    sub_questions = sub_query_dict.get("sub_questions", [])
    all_queries = [main_goal] + sub_questions if main_goal else sub_questions

    # Log success metrics
    duration = time.time() - start_time
    result_count = len(web_search_results_dict.get("results", []))
    log_histogram(
        "websearch_generate_and_search_duration",
        duration,
        labels={
            "engine": engine,
            "subquery_generation": str(search_params.get("subquery_generation", False)),
            "result_count": str(result_count),
        },
    )
    log_counter(
        "websearch_generate_and_search_success",
        labels={
            "engine": engine,
            "total_queries": str(len(all_queries)),
            "result_count": str(result_count),
        },
    )

    if isinstance(final_answer, dict):
        final_answer["gate"] = {
            "relevant": len(relevant_results),
            "raw": len(web_search_results_dict.get("results") or []),
            "fallback": any(
                entry.get("gate_unverified") for entry in relevant_results.values()
            ),
        }
    return {
        "final_answer": final_answer,
        "relevant_results": relevant_results,
        "web_search_results_dict": web_search_results_dict,
    }


######################### Question Analysis #########################
#
_SUBQUERY_GENERATION_MAX_ATTEMPTS = 3

# One shared per-request HTTP timeout for every search backend this repo
# owns (task-1355 added the literal to serper/exa/yandex; task-3060 to the
# six older engines; Qodo PR #1451 named it). Bing predates both with its
# own 10s and deliberately keeps it.
SEARCH_BACKEND_TIMEOUT_S = 30
"""Number of paid LLM attempts `analyze_question` makes at generating
sub-questions before giving up. Shared with `generate_and_search`'s
total-failure warning (task-3221) so the "N attempts" the user is told
about can never drift from the loop bound that actually produced it."""


def analyze_question(question: str, api_endpoint) -> Dict:
    """
    Analyze a question and generate relevant sub-queries.

    Uses an LLM to break down complex questions into multiple
    specific sub-queries for more comprehensive search coverage.

    Args:
        question (str): The original question to analyze
        api_endpoint (str): LLM API endpoint name (e.g., 'openai')

    Returns:
        Dict: Contains:
            - main_goal: Original question
            - sub_questions: List of generated sub-queries
            - search_queries: Same as sub_questions
            - analysis_prompt: The prompt used

    Example:
        >>> result = analyze_question(
        ...     "What are the environmental impacts of electric vehicles?",
        ...     "openai"
        ... )
        >>> print(result["sub_questions"])
        ["carbon footprint of EV manufacturing",
         "battery disposal environmental impact",
         "electricity source for EV charging", ...]
    """
    logger.debug(f"Analyzing question: {question} with API endpoint: {api_endpoint}")
    """
    Analyzes the input question and generates sub-questions

    Returns:
        Dict containing:
        - main_goal: str
        - sub_questions: List[str]
        - search_queries: List[str]
        - analysis_prompt: str
    """
    original_query = question
    sub_question_generation_prompt = render_internal_prompt(
        "websearch.sub_question_generation", original_query=original_query
    )

    input_data = "Follow the above instructions."

    sub_questions: List[str] = []
    for attempt in range(_SUBQUERY_GENERATION_MAX_ATTEMPTS):
        try:
            logger.info(f"Generating sub-questions (attempt {attempt + 1})")

            messages_payload = [
                {
                    "role": "user",
                    "content": input_data + "\n\n" + sub_question_generation_prompt,
                }
            ]
            response = chat_reply_text(
                chat_api_call(
                    api_endpoint=api_endpoint,
                    messages_payload=messages_payload,
                    api_key=None,
                    temp=0.7,
                    system_message=None,
                    streaming=False,
                    minp=None,
                    maxp=None,
                    model=None,
                )
            )
            if response:
                try:
                    # Try to parse as JSON first
                    parsed_response = json.loads(response)
                    sub_questions = _sanitize_sub_questions(parsed_response)
                    if sub_questions:
                        logger.info("Successfully generated sub-questions from JSON")
                        break
                except json.JSONDecodeError:
                    # If JSON parsing fails, attempt a regex-based fallback
                    logger.warning(
                        "Failed to parse as JSON. Attempting regex extraction."
                    )
                    matches = re.findall(r'"([^"]*)"', response)
                    sub_questions = _sanitize_sub_questions(matches)
                    if sub_questions:
                        logger.info("Successfully extracted sub-questions using regex")
                        break

        except Exception as e:
            logger.error(f"Error generating sub-questions: {str(e)}")

    if not sub_questions:
        logger.error(
            "Failed to extract sub-questions from API response after all attempts."
        )
        sub_questions = []  # No fallback to the original query (task-1356; port of
        # server analyze_question, which never re-seeds sub_questions with the
        # original query on total failure -- generate_and_search already always
        # searches the original question, so [original_query] here just meant a
        # duplicate, wasted search on total LLM failure).

    # Construct and return the result dictionary
    logger.info("Sub-questions generated successfully")
    return {
        "main_goal": original_query,
        "sub_questions": sub_questions,
        "search_queries": sub_questions,
        "analysis_prompt": sub_question_generation_prompt,
    }


def _build_result_fallback_content(result: Dict[str, Any]) -> str:
    """Construct a safe fallback text blob (title/snippet/url) from a
    normalized search-result dict, used both as the relevance-eval prompt's
    content and as the summarization source when scraping fails or a result
    carries no `content` field (port of server WebSearch_APIs.py :306-327;
    task-1356)."""
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    parts: List[str] = []
    seen: set = set()

    def _append(label: str, value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        parts.append(f"{label}: {text}")

    _append("Title", result.get("title"))
    _append("Snippet", result.get("content"))
    _append("Snippet", metadata_dict.get("snippet"))
    _append("URL", result.get("url"))
    return "\n".join(parts).strip()


######################### Relevance Analysis #########################
#
_DNS_GUARD_EXECUTOR_MAX_WORKERS = 4

_DNS_GUARD_EXECUTOR_LOCK = threading.Lock()
"""Guards creation of the module-level DNS-guard executor below."""

_DNS_GUARD_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
"""Dedicated, lazily-created, bounded executor for TWO guard-class,
synchronous-network-I/O offloads in `search_result_relevance`, both wrapped
in `asyncio.wait_for(..., timeout=scrape_timeout_s)`: the pre-scrape SSRF
DNS guard (`is_public_http_url`, task-3220) and the pre-scrape robots.txt
check (`robots_allows_for_scrape`, task-3260). Both do synchronous network
I/O (`socket.getaddrinfo` / a blocking `httpx.Client` request respectively)
that cannot be cancelled once started -- when the caller's `wait_for` times
out, the abandoned thread keeps occupying its executor slot until the
underlying call itself gives up, not until the caller stopped waiting.
Routing either through `asyncio.to_thread` would put it on the DEFAULT
executor, the same one this loop's other offloads (the relevance/
summarization `chat_api_call` calls, `aggregate_results`) share -- so a
result set full of slow-DNS/slow-robots hosts could queue paid LLM calls
behind dead resolvers/fetches. A small, separate pool isolates the
abandoned threads so they can never crowd out those offloads.

The two consumers fail in OPPOSITE directions once `wait_for` gives up on
them (task-3260 design doc ruling 5, deliberate): the SSRF guard fails
CLOSED (treated as non-public -> the scrape is refused) while the robots
check fails OPEN (treated as allowed -> the scrape proceeds, matching
`_fetch_robots_parser`'s existing fail-open for web_fetch/web_crawl).
Sharing one pool between opposite-failing consumers has a real interaction
under saturation: a hung robots check can hold its slot far longer than
`scrape_timeout_s` -- `robots_allows_for_scrape`'s own client can chase up
to `FETCH_MAX_REDIRECTS + 1` (6) hops, each independently bounded by
`FETCH_TIMEOUT_SECONDS` (30s), so a host that hangs at every hop's timeout
boundary can occupy a slot for ~180s even though the caller's own
`wait_for` gave up after 30s. With only `_DNS_GUARD_EXECUTOR_MAX_WORKERS`
(4) slots, roughly 4 such hung hosts saturate the whole pool; every guard
call submitted after that -- SSRF AND robots alike, since they share this
one pool -- queues UNSTARTED behind them. A queued robots check that never
gets to run before its OWN `wait_for` elapses times out and fails OPEN
(robots enforcement goes silently off for it), while a queued SSRF check
queued the same way times out and fails CLOSED (still refuses) -- so a
burst of slow/hung hosts degrades robots enforcement specifically, without
weakening the SSRF guard. Spec-sanctioned, recorded here rather than fixed
(the alternative -- separate pools per consumer, or a circuit breaker -- is
out of scope for task-3260)."""


def _get_dns_guard_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared DNS-guard executor, creating it on first use."""
    global _DNS_GUARD_EXECUTOR
    if _DNS_GUARD_EXECUTOR is None:
        with _DNS_GUARD_EXECUTOR_LOCK:
            if _DNS_GUARD_EXECUTOR is None:
                _DNS_GUARD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_DNS_GUARD_EXECUTOR_MAX_WORKERS,
                    thread_name_prefix="deep-search-dns-guard",
                )
    return _DNS_GUARD_EXECUTOR


def _reset_dns_guard_executor_for_tests() -> None:
    """TEST-ONLY: force-recreate the module-level DNS-guard executor.

    `_DNS_GUARD_EXECUTOR` is a process-wide singleton by design (see its
    docstring above) -- production code never wants to discard in-flight
    guard calls. But a test that saturates the pool with blocked callables
    to prove non-starvation (fix-wave Important 3, task-3220) leaves that
    saturation sitting in the singleton for every test that runs afterward
    in the same process, since nothing else ever resets it. Shuts the old
    executor down with `cancel_futures=True` (drops anything still queued
    but not yet started) and clears the singleton so the next
    `_get_dns_guard_executor()` call lazily rebuilds a clean pool. No
    production code path calls this.
    """
    global _DNS_GUARD_EXECUTOR
    with _DNS_GUARD_EXECUTOR_LOCK:
        executor = _DNS_GUARD_EXECUTOR
        _DNS_GUARD_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


# FIXME - Ensure edge cases are handled properly / Structured outputs?
async def search_result_relevance(
    search_results: List[Dict],
    original_question: str,
    sub_questions: List[str],
    api_endpoint: str,
    *,
    cancel_event: Optional[asyncio.Event] = None,
    llm_timeout_s: float = 30.0,
    scrape_timeout_s: float = 30.0,
    respect_robots_txt: bool = False,
) -> Dict[str, Dict]:
    """
    Evaluate search results for relevance and extract key content.

    This function:
    1. Uses LLM to score each result's relevance
    2. Scrapes full content from relevant URLs
    3. Summarizes the content focused on the question

    Port of server WebSearch_APIs.py :789-985 (timeouts, cancel_event, scrape
    fallback), with :306-327's `_build_result_fallback_content` supplying the
    relevance-eval prompt's content instead of the raw (often-empty) `content`
    field. Binding adaptations (task-1356): (1) no circuit breaker -- a
    timed-out/erroring provider just costs this one result via
    `asyncio.wait_for` around `asyncio.to_thread(chat_api_call, ...)` (the
    chatbook transport is sync); (2) no outbound-policy calls; (3) each
    relevant entry now also carries `url`/`title` captured from the source
    result (not present on the server), for downstream citation display;
    (6) jitter stays chatbook's `random.uniform(0.2, 0.6)` (no config knob).

    robots.txt parity (task-3260): when ``respect_robots_txt`` is True, a
    relevant result's URL is also checked against its host's robots.txt --
    same guard-class offload discipline as the SSRF check just above it
    (dedicated DNS-guard executor, bounded by ``scrape_timeout_s``) -- right
    before the ``scrape_article`` call. Disallowed takes the SAME path as an
    SSRF refusal: the scrape is skipped and the result is kept with its
    existing snippet/title/url fallback content, never discarded. A robots-
    check error/timeout fails OPEN (proceeds to scrape) -- deliberately the
    opposite of the SSRF guard's own timeout/refusal, matching
    ``_fetch_robots_parser``'s existing fail-open for web_fetch/web_crawl.

    Args:
        search_results (List[Dict]): List of search results to evaluate.
        original_question (str): The original question posed by the user.
        sub_questions (List[str]): List of sub-questions generated from the original question.
        api_endpoint (str): The LLM or API endpoint to use for relevance analysis.
        cancel_event: Optional cooperative-cancellation flag; checked at the
            top of each iteration so a caller-side deadline watchdog can stop
            the loop between results.
        llm_timeout_s: Wall-clock timeout for each relevance/summarization LLM call.
        scrape_timeout_s: Wall-clock timeout for the per-result scrape.
        respect_robots_txt: When True, consult the target host's robots.txt
            before scraping a relevant result (task-3260); default False
            (no robots.txt fetch at all) preserves the pre-task-3260
            behavior for any caller that doesn't pass this explicitly.

    Returns:
        Dict[str, Dict]: A dictionary of relevant results, keyed by a unique ID or index.
    """
    relevant_results: Dict[str, Dict] = {}
    # task-16333: results the gate EVALUATED and rejected (verdict False).
    # Only these are eligible for the zero-relevant fallback -- results
    # skipped via timeout/cancel/no-content were never judged, and promoting
    # them would launder unevaluated evidence.
    gate_rejected: List[tuple] = []

    for idx, result in enumerate(search_results):
        if cancel_event and cancel_event.is_set():
            logger.info("search_result_relevance: cancel_event set, stopping loop")
            break

        content = _build_result_fallback_content(result)
        if not content:
            logger.error("No Content found in search results array!")
            continue

        # First, evaluate relevance
        eval_prompt = render_internal_prompt(
            "websearch.result_relevance_eval",
            original_question=original_question,
            sub_questions=sub_questions,
            content=content,
        )
        # task-17066: calibrate for the kind of evidence under judgment.
        # The strict usefulness prompt is calibrated for papers; repository
        # records (datasets/software/figures) and scholarly metadata records
        # fail the "answers the question" bar even when topically on-point
        # (measured: repositories gate_pass 0.29 vs 0.72 for papers).
        input_data = "Evaluate the relevance of the search result."
        try:
            from tldw_chatbook.Research_Interop.research_source_catalog import (
                source_kind_for_provider,
            )

            _provider = None
            _metadata = result.get("metadata")
            if isinstance(_metadata, dict):
                _provider = _metadata.get("provider")
            _source_kind = source_kind_for_provider(_provider)
            _source_notes = {
                "repository": (
                    "Note: this result is a repository record (dataset, "
                    "software, or figure) rather than a paper. It is relevant "
                    "if it is topically related to the question and could "
                    "serve as supporting evidence (data, methods, or "
                    "artifacts); it does NOT need to directly answer the "
                    "question."
                ),
                "metadata": (
                    "Note: this result is a scholarly metadata record rather "
                    "than full text. It is relevant if it describes a work "
                    "topically related to the question; it does NOT need to "
                    "directly answer the question."
                ),
            }
            _source_note = _source_notes.get(_source_kind)
            if _source_note:
                input_data = f"{input_data} {_source_note}"
        except Exception:  # noqa: BLE001 - note computation fails OPEN to the strict paper prompt
            logger.debug("source-kind note skipped", exc_info=True)

        try:
            # Add delay to avoid rate limiting
            sleep_time = random.uniform(0.2, 0.6)
            await asyncio.sleep(sleep_time)

            # Evaluate relevance (chat_api_call is sync; run off-thread so the
            # wait_for timeout can actually bound it).
            messages_payload = [
                {"role": "user", "content": input_data + "\n\n" + eval_prompt}
            ]

            async def _eval_call(_mp=messages_payload):
                return await asyncio.to_thread(
                    lambda: chat_api_call(
                        api_endpoint=api_endpoint,
                        messages_payload=_mp,
                        api_key=None,
                        # Classification, not generation (task-16333): the
                        # binary relevant/not-relevant verdict must be stable
                        # across identical runs; 0.7 flipped verdicts.
                        temp=_RELEVANCE_JUDGMENT_TEMP,
                        system_message=None,
                        streaming=False,
                        minp=None,
                        maxp=None,
                        model=None,
                        topk=None,
                        topp=None,
                    )
                )

            relevancy_result = chat_reply_text(
                await asyncio.wait_for(_eval_call(), timeout=llm_timeout_s)
            )

            # FIXME
            logger.debug(
                f"[DEBUG] Relevancy LLM response for index {idx}:\n{relevancy_result}\n---"
            )

            if relevancy_result:
                # Extract the selected answer and reasoning via regex
                logger.debug(f"LLM Relevancy Response for item: {relevancy_result}")
                selected_answer_match = re.search(
                    r"Selected Answer:\s*(True|False)", relevancy_result, re.IGNORECASE
                )
                reasoning_match = re.search(
                    r"Reasoning:\s*(.+)", relevancy_result, re.IGNORECASE
                )

                if selected_answer_match and reasoning_match:
                    is_relevant = (
                        selected_answer_match.group(1).strip().lower() == "true"
                    )
                    reasoning = reasoning_match.group(1).strip()

                    if is_relevant:
                        logger.debug("Relevant result found.")
                        # Use the 'id' from the result if available, otherwise use idx
                        result_id = result.get("id", str(idx))
                        source_content = content

                        try:
                            # Pre-scrape SSRF guard (task-1356): scrape_article's
                            # own validation (input_validation.validate_url) is
                            # well-formedness only -- no DNS resolution, no
                            # private-range blocking. This phase browses arbitrary
                            # search-result URLs with Playwright, so a result
                            # pointing at e.g. http://169.254.169.254/ must be
                            # refused BEFORE navigation, not after. A refusal is
                            # counted exactly like a scrape failure -- never an
                            # exception -- and falls through to the existing
                            # snippet/title/url fallback below.
                            #
                            # is_public_http_url does synchronous DNS resolution
                            # (socket.getaddrinfo) -- its own docstring chain
                            # (evaluate_url_policy) warns never to call that kind
                            # of check directly from an event loop. Offload it,
                            # bounded by the same timeout used for the scrape
                            # itself; a guard timeout raises asyncio.TimeoutError,
                            # which is an Exception (unlike CancelledError) so it
                            # falls straight into the `except Exception` below and
                            # is handled exactly like a scrape failure -- no
                            # separate code path.
                            #
                            # Routed through the dedicated `_get_dns_guard_executor`
                            # pool (task-3220), NOT `asyncio.to_thread`'s shared
                            # default executor: on a wait_for timeout the abandoned
                            # getaddrinfo thread keeps occupying its slot until the
                            # OS resolver gives up, and the default executor is
                            # also where this loop's own chat_api_call/summarize
                            # offloads run -- a run of slow-DNS hosts must not be
                            # able to queue those paid LLM calls behind dead
                            # resolvers. This bounds the BLAST RADIUS, not the
                            # symptom for THIS call: under full saturation of the
                            # dedicated pool (all `_DNS_GUARD_EXECUTOR_MAX_WORKERS`
                            # slots already occupied by abandoned threads), a new
                            # guard call submitted here queues unstarted behind
                            # them, and `wait_for` below still fires once
                            # `scrape_timeout_s` elapses -- it has no way to tell
                            # "queued" from "running slow" apart. That still
                            # correctly skips the scrape fail-safe (falls through
                            # to the snippet/title/url fallback exactly like a
                            # real guard timeout), it just does so without this
                            # particular call ever having gotten a real DNS
                            # answer -- the isolation guarantee above is about
                            # protecting OTHER offloads, not about this call
                            # getting to run promptly.
                            is_public = await asyncio.wait_for(
                                asyncio.get_running_loop().run_in_executor(
                                    _get_dns_guard_executor(), is_public_http_url, result["url"]
                                ),
                                timeout=scrape_timeout_s,
                            )
                            if not is_public:
                                logger.warning(
                                    f"Refusing to scrape non-public URL for result "
                                    f"{result_id}: {result.get('url')!r}; falling "
                                    "back to search snippet/title/url"
                                )
                            else:
                                # robots.txt parity (task-3260): checked here,
                                # between the SSRF guard above and the scrape
                                # below -- same guard-class offload discipline
                                # (dedicated DNS-guard executor, bounded by
                                # scrape_timeout_s) as the SSRF check, since a
                                # robots.txt fetch does its own network I/O.
                                # Function-local import: no module-level cycle
                                # (web_tool_impls already imports WebSearch_APIs
                                # function-locally in the OTHER direction).
                                robots_ok = True
                                if respect_robots_txt:
                                    from tldw_chatbook.Tools.web_tool_impls import (
                                        robots_allows_for_scrape,
                                    )

                                    try:
                                        robots_ok = await asyncio.wait_for(
                                            asyncio.get_running_loop().run_in_executor(
                                                _get_dns_guard_executor(),
                                                robots_allows_for_scrape,
                                                result["url"],
                                            ),
                                            timeout=scrape_timeout_s,
                                        )
                                    except asyncio.CancelledError:
                                        raise
                                    except Exception as robots_error:
                                        # Fail-OPEN on a robots-check error or
                                        # timeout (deliberately the opposite of
                                        # the SSRF guard just above, whose own
                                        # timeout/refusal still refuses) --
                                        # matches _fetch_robots_parser's
                                        # existing fail-open for web_fetch/
                                        # web_crawl.
                                        logger.debug(
                                            f"robots.txt check failed for result "
                                            f"{result_id}, failing open (will "
                                            f"scrape): {robots_error}"
                                        )
                                        robots_ok = True

                                if not robots_ok:
                                    # Disallowed -> same path as an SSRF
                                    # refusal: skip the scrape, keep the
                                    # result via its existing fallback
                                    # content (never discard). Log names the
                                    # HOST only, never the query.
                                    host = urlparse(result.get("url") or "").hostname or "unknown"
                                    logger.debug(
                                        f"Skipping scrape for result {result_id}: "
                                        f"robots.txt disallows {host!r}; falling "
                                        "back to search snippet/title/url"
                                    )
                                else:
                                    scraped_content = await asyncio.wait_for(
                                        scrape_article(result["url"]), timeout=scrape_timeout_s
                                    )
                                    scraped_text = ""
                                    if isinstance(scraped_content, dict):
                                        scraped_text = str(scraped_content.get("content") or "").strip()
                                    elif isinstance(scraped_content, str):
                                        scraped_text = scraped_content.strip()
                                    if scraped_text:
                                        source_content = scraped_text
                        except asyncio.CancelledError:
                            raise
                        except Exception as scrape_error:
                            logger.warning(
                                f"Scrape failed for relevant result {result_id}; "
                                f"falling back to search snippet/title/url: {scrape_error}"
                            )

                        # Create Summarization prompt
                        logger.debug(
                            f"Creating Summarization Prompt for result idx={idx}"
                        )
                        summary_prompt = render_internal_prompt(
                            "websearch.result_summarization",
                            question=original_question,
                            content=source_content,
                        )

                        # Add delay before summarization
                        await asyncio.sleep(sleep_time)

                        # `analyze` (LLM_Calls.Summarization_General_Lib) is imported
                        # lazily here (chatbook precedent, see module docstring)
                        # so a plain import of this module doesn't eagerly pull in
                        # the summarization stack.
                        from tldw_chatbook.LLM_Calls.Summarization_General_Lib import (
                            analyze,
                        )

                        logger.info(f"Summarizing relevant result: ID={result_id}")

                        async def _summ_call(_sc=source_content, _sp=summary_prompt):
                            return await asyncio.to_thread(
                                lambda: analyze(
                                    input_data=_sc,
                                    custom_prompt_arg=_sp,
                                    api_name=api_endpoint,
                                    api_key=None,
                                    temp=0.7,
                                    system_message=None,
                                    streaming=False,
                                )
                            )

                        summary = None
                        try:
                            summary = await asyncio.wait_for(_summ_call(), timeout=llm_timeout_s)
                        except asyncio.CancelledError:
                            raise
                        except Exception as summ_error:
                            logger.error(f"Summary generation failed: {summ_error}")

                        # `analyze()` reports failure by RETURNING one of its
                        # providers' error strings rather than raising -- treat
                        # that the same way the port source treats a raised
                        # exception: fall back to the (scraped or fallback)
                        # source content itself. Detection cannot be a bare
                        # "Error:" prefix test: a llama.cpp failure returns
                        # "Llama: Error occurred while ..." and used to be
                        # stored here AS the evidence (task-17382).
                        if _is_summary_failure(summary) or not summary:
                            summary = source_content[:2000] or "Summary generation failed"

                        relevant_results[result_id] = {
                            "content": summary,  # Store the summary instead of full content
                            "original_content": source_content,  # Keep original content if needed
                            "reasoning": reasoning,
                            "url": result.get("url"),
                            "title": result.get("title"),
                        }
                        logger.info(
                            f"Relevant result found and summarized: ID={result_id}; Reasoning={reasoning}"
                        )
                    else:
                        logger.info(f"Irrelevant result: {reasoning}")
                        gate_rejected.append((idx, result, content))

                else:
                    logger.warning(
                        "Failed to parse the API response for relevance analysis."
                    )
        except asyncio.CancelledError:
            logger.warning("Relevance evaluation cancelled")
            raise
        except asyncio.TimeoutError:
            logger.error(f"Timeout during LLM/scrape for result idx={idx}")
        except Exception as e:
            logger.error(
                f"Error during relevance evaluation/summarization for result idx={idx}: {e}"
            )

    # task-16333 zero-relevant fallback: the live baseline showed a strict
    # gate silently producing NO report at all. When every EVALUATED result
    # was rejected but raw results exist (and the run was not cancelled --
    # a deadline hit must keep reporting the honest cutoff), keep the
    # top-ranked rejected results as snippet-level evidence flagged
    # gate_unverified: a flagged report beats no report. No scrape or
    # summarization spend on fallback entries.
    if (
        not relevant_results
        and gate_rejected
        and not (cancel_event and cancel_event.is_set())
    ):
        for fb_idx, fb_result, fb_content in gate_rejected[:_GATE_FALLBACK_MAX_RESULTS]:
            fb_result_id = str(fb_result.get("id", fb_idx))
            relevant_results[fb_result_id] = {
                "content": fb_content,
                "original_content": fb_content,
                "reasoning": "gate fallback: evidence not relevance-verified",
                "url": fb_result.get("url"),
                "title": fb_result.get("title"),
                "gate_unverified": True,
            }
        logger.warning(
            f"Relevance gate rejected all {len(gate_rejected)} evaluated result(s); "
            f"proceeding with top {len(relevant_results)} flagged gate-unverified"
        )

    return relevant_results


def review_and_select_results(
    web_search_results_dict: Dict, selector: Optional[Callable[[Dict], bool]] = None
) -> Dict:
    """
    Select relevant results from a search-results payload -- pure, no
    blocking `input()` (port of server WebSearch_APIs.py :988-1029;
    task-1356). Two calling shapes are supported and the return shape mirrors
    whichever one was given:
      - ``{"results": [result, ...]}`` -> ``{"results": [selected, ...]}``
      - ``{result_id: result, ...}`` (a flat mapping, e.g. relevant_results
        from search_result_relevance) -> ``{result_id: selected, ...}``

    Args:
        web_search_results_dict (Dict): The dictionary containing all search results.
        selector: Optional predicate `(result) -> bool`; when omitted every
            candidate passes (there is no interactive prompt anymore -- the
            caller decides selection, e.g. via config or an explicit filter).

    Returns:
        Dict: Only the selected results, in the same shape as the input.
    """
    if not isinstance(web_search_results_dict, dict):
        return {}

    results_list = web_search_results_dict.get("results")
    is_results_shape = "results" in web_search_results_dict and isinstance(results_list, list)

    if is_results_shape:
        candidates = [
            (str(result.get("id", idx)) if isinstance(result, dict) else str(idx), result)
            for idx, result in enumerate(results_list)
            if isinstance(result, dict)
        ]
    else:
        candidates = [
            (str(result_id), result)
            for result_id, result in web_search_results_dict.items()
            if isinstance(result, dict)
        ]

    if selector is None:
        selected = candidates
    else:
        selected = []
        for result_id, result in candidates:
            try:
                if selector(result):
                    selected.append((result_id, result))
            except Exception:
                # If selector throws, skip selection for this item
                continue

    if is_results_shape:
        return {"results": [result for _, result in selected]}
    return {result_id: result for result_id, result in selected}


######################### Result Aggregation & Combination #########################
#
# task-16333: binary verdicts need determinism, not creativity.
_RELEVANCE_JUDGMENT_TEMP = 0.1
# task-16333: bounded zero-relevant fallback (search-rank order).
_GATE_FALLBACK_MAX_RESULTS = 3


class FinalAnswerDict(TypedDict):
    """Structured payload returned by the aggregation phase (port of server
    WebSearch_APIs.py :1034-1039; task-1356). `evidence` entries are dicts
    shaped ``{id: int, url, title, content, original_content, reasoning,
    chunk_index}``."""

    text: str
    evidence: List[Dict[str, Any]]
    confidence: float
    chunks: List[Dict[str, Any]]
    # Present ONLY on the LLM-success branch (task-16331): marker resolution
    # and quote-check counts from deep_search_citations.verify_citations.
    # Failure/empty branches omit it rather than fabricating a clean verdict.
    citation_verification: NotRequired[Dict[str, Any]]
    # Present whenever relevance outcomes are known (task-16333):
    # {"relevant": int, "raw": int, "fallback": bool} -- fallback marks a
    # report built from gate-unverified evidence.
    gate: NotRequired[Dict[str, Any]]


def _build_chunk_infos(items: List[str], max_chars: int = 6000) -> List[Dict[str, Any]]:
    """Greedily pack pre-formatted text entries into <= max_chars chunks for
    map-reduce summarization (port of server WebSearch_APIs.py :1075-1117,
    adapted to operate on plain formatted strings rather than (id, result)
    tuples -- task-1356). A single entry at/over max_chars becomes its own
    truncated chunk instead of being combined with neighbors.

    Returns a list of dicts: ``{index: int (1-based), item_indices: list[int]
    (1-based positions into `items` packed into this chunk), text: str,
    truncated: bool}``.
    """
    chunk_infos: List[Dict[str, Any]] = []
    current_entries: List[tuple] = []
    current_length = 0

    def flush_entries() -> None:
        nonlocal current_entries, current_length
        if not current_entries:
            return
        text = "\n\n".join(entry for _, entry in current_entries)
        chunk_infos.append(
            {
                "index": len(chunk_infos) + 1,
                "item_indices": [item_idx for item_idx, _ in current_entries],
                "text": text,
                "truncated": False,
            }
        )
        current_entries = []
        current_length = 0

    for item_idx, entry in enumerate(items, start=1):
        entry_length = len(entry)
        if entry_length >= max_chars:
            flush_entries()
            chunk_infos.append(
                {
                    "index": len(chunk_infos) + 1,
                    "item_indices": [item_idx],
                    "text": entry[:max_chars],
                    "truncated": True,
                }
            )
            continue

        if current_length + entry_length > max_chars and current_entries:
            flush_entries()

        current_entries.append((item_idx, entry))
        current_length += entry_length

    flush_entries()
    return chunk_infos


def _estimate_confidence(
    relevant_count: int, chunk_count: int, failed_chunks: int, has_llm: bool
) -> float:
    """Confidence heuristic, port of server WebSearch_APIs.py :1119-1133
    VERBATIM (task-1356; the plan doc's "Global Constraints" transcription
    dropped two server nuances -- see task-2 fix-report -- so this now
    matches the live server source exactly, not the plan doc):
    ``coverage = min(relevant_count, 10) / 10``;
    ``chunk_success = 1.0 if chunk_count == 0 else (chunk_count - failed_chunks) / chunk_count``
    (zero chunks means nothing failed, not a 0.4x penalty);
    ``llm_bonus = 0.1 if has_llm and failed_chunks == 0 else (0.05 if has_llm else 0.0)``
    (a fully-clean LLM run earns the full bonus, not a flat 0.05);
    ``confidence = (0.35 + 0.45 * coverage) * (0.6 + 0.4 * chunk_success) + llm_bonus``,
    clamped to [0.1, 0.99]; 0.0 only when relevant_count == 0.
    """
    if relevant_count <= 0:
        return 0.0
    coverage = min(relevant_count, 10) / 10.0
    chunk_success = 1.0 if chunk_count == 0 else (chunk_count - failed_chunks) / chunk_count
    base = 0.35 + 0.45 * coverage
    modifier = 0.6 + 0.4 * chunk_success
    llm_bonus = 0.1 if has_llm and failed_chunks == 0 else (0.05 if has_llm else 0.0)
    confidence = base * modifier + llm_bonus
    return max(0.1, min(0.99, round(confidence, 3)))


def aggregate_results(
    relevant_results: Dict[str, Dict],
    question: str,
    sub_questions: List[str],
    api_endpoint: Optional[str],
) -> FinalAnswerDict:
    """
    Combines and summarizes relevant results into a final answer via a
    chunked map-reduce: relevant results are renumbered 1..N (stable dict
    order) and packed into <= 6000-char chunks (`_build_chunk_infos`). Every
    "[n]" citation the eventual synthesis LLM emits resolves to a real
    evidence id (the citation-integrity fix; task-1356 binding adaptation 4,
    replacing the server's un-renumbered "ID: {rid}" payload):
      - **1 chunk** (context already bounded by construction): the MAP
        summarization call is skipped entirely -- it would cost a provider
        round-trip whose output fed nothing -- and the synthesis prompt (REDUCE)
        consumes the raw numbered evidence directly.
      - **>1 chunks**: each chunk is summarized (MAP; chatbook's lazy
        `Summarization_General_Lib.analyze` import, binding adaptation 5, not
        the server's `summarize`), instructed to preserve "[n]" markers
        verbatim, and the synthesis prompt (REDUCE) consumes those chunk
        summaries -- restoring the server's actual context-bounding design
        (port of server WebSearch_APIs.py :1042-1298) instead of feeding the
        full originals regardless of size.

    Args:
        relevant_results (Dict[str, Dict]): Dictionary of relevant articles/content.
        question (str): Original question.
        sub_questions (List[str]): List of sub-questions.
        api_endpoint (str): LLM or API endpoint for summarization; falsy ->
            no-LLM fallback branch (server :1154-1173).

    Returns:
        FinalAnswerDict: `{text, evidence, confidence, chunks}` on every branch.
    """
    logger.info("Aggregating and summarizing relevant results")
    if not relevant_results:
        empty_answer: FinalAnswerDict = {
            "text": "No relevant results found. Unable to provide an answer.",
            "evidence": [],
            "confidence": 0.0,
            "chunks": [],
        }
        return empty_answer

    logger.info("Summarizing relevant results")

    # Renumber 1..N in the dict's stable iteration order and build the
    # numbered "[n] title/content/reasoning" payload used both for chunk
    # packing and for the synthesis prompt.
    numbered_items = list(enumerate(relevant_results.items(), start=1))
    entry_texts: List[str] = []
    for n, (_rid, res) in numbered_items:
        title = res.get("title") or res.get("url") or f"Result {n}"
        entry_texts.append(
            f"[{n}] {title}\n{res.get('content', '')}\nReasoning: {res.get('reasoning', '')}"
        )

    chunk_infos = _build_chunk_infos(entry_texts, max_chars=6000)
    chunk_index_by_n: Dict[int, int] = {}
    for info in chunk_infos:
        for item_n in info["item_indices"]:
            chunk_index_by_n[item_n] = info["index"]

    evidence_payload: List[Dict[str, Any]] = []
    for n, (_rid, res) in numbered_items:
        evidence_entry = {
            "id": n,
            "url": res.get("url"),
            "title": res.get("title"),
            "content": res.get("content"),
            "original_content": res.get("original_content"),
            "reasoning": res.get("reasoning"),
            "chunk_index": chunk_index_by_n.get(n),
        }
        if res.get("gate_unverified"):
            evidence_entry["gate_unverified"] = True
        evidence_payload.append(evidence_entry)

    concatenated_texts = "\n\n".join(entry_texts)

    if not api_endpoint:
        logger.warning("No final answer LLM configured; returning evidence summaries only.")
        chunk_metadata = [
            {
                "chunk_index": info["index"],
                "evidence_ids": info["item_indices"],
                "summary": info["text"][:1500],
                "generated": False,
                "source_characters": len(info["text"]),
                "truncated_source": info["truncated"],
            }
            for info in chunk_infos
        ]
        combined_text = "\n\n".join(
            str(res.get("content") or "") for _, res in relevant_results.items()
        )
        fallback_answer: FinalAnswerDict = {
            "text": combined_text or "Unable to generate a final answer without an LLM.",
            "evidence": evidence_payload,
            "confidence": _estimate_confidence(
                len(evidence_payload), len(chunk_infos), 0, has_llm=False
            ),
            "chunks": chunk_metadata,
        }
        return fallback_answer

    chunk_metadata: List[Dict[str, Any]] = []
    failed_chunks = 0

    if len(chunk_infos) <= 1:
        # A single chunk already bounds context by construction -- the MAP
        # summarization call would cost a provider round-trip for zero
        # effect (its output would only feed the audit trail), so skip it
        # and synthesize directly from the raw numbered evidence.
        for info in chunk_infos:
            chunk_metadata.append(
                {
                    "chunk_index": info["index"],
                    "evidence_ids": info["item_indices"],
                    "summary": info["text"][:1500],
                    "generated": False,
                    "source_characters": len(info["text"]),
                    "truncated_source": info["truncated"],
                }
            )
        synthesis_source = concatenated_texts
    else:
        # MAP: multiple chunks -- summarize each one so the synthesis prompt
        # below stays context-bounded (the point of chunking); the summarizer
        # is explicitly told to preserve "[n]" citation markers verbatim so
        # the citation-integrity fix (adaptation 4) survives this reduce step.
        # `analyze` is imported lazily (chatbook precedent, see module docstring).
        from tldw_chatbook.LLM_Calls.Summarization_General_Lib import (
            analyze as _analyze,
        )

        summarized_chunks: List[str] = []
        for info in chunk_infos:
            chunk_prompt = (
                "Summarize the following set of relevant search snippets into a "
                f'concise digest that preserves high-signal facts for answering the question: "{question}".\n\n'
                "Requirements:\n"
                "1. Keep the summary under 1500 characters.\n"
                "2. Focus on verifiable facts and key statistics.\n"
                "3. Mention the reasoning notes when helpful.\n"
                "4. Preserve all [n] citation markers exactly as they appear; "
                "never renumber, merge, or drop them.\n\n"
                f"<chunk index=\"{info['index']}\">\n{info['text']}\n</chunk>"
            )
            chunk_used_fallback = False
            try:
                chunk_summary = _analyze(
                    input_data=info["text"],
                    custom_prompt_arg=chunk_prompt,
                    api_name=api_endpoint,
                    api_key=None,
                    temp=0.3,
                    system_message=None,
                    streaming=False,
                )
                if _is_summary_failure(chunk_summary) or not chunk_summary:
                    raise RuntimeError(chunk_summary or "empty chunk summary")
                generated = True
            except Exception as chunk_error:
                failed_chunks += 1
                logger.warning(
                    f"Chunk summarization failed for chunk {info['index']}: {chunk_error}"
                )
                chunk_summary = info["text"][:1500]
                generated = False
                chunk_used_fallback = True

            chunk_meta_entry = {
                "chunk_index": info["index"],
                "evidence_ids": info["item_indices"],
                "summary": chunk_summary,
                # "generated" means only "an LLM produced this summary" -- it
                # is NOT a failure signal on its own (the single-chunk skip
                # path above also sets it False with nothing having failed).
                # "fallback" (below, set ONLY here) is the ONE field that
                # means "summarization actually failed and truncated raw
                # text was substituted" -- final review, Important 1: the
                # footer used to read "generated" as a failure signal and
                # falsely called the healthiest possible run (single-chunk,
                # nothing failed) a fallback.
                "generated": generated,
                "source_characters": len(info["text"]),
                "truncated_source": info["truncated"],
            }
            if chunk_used_fallback:
                chunk_meta_entry["fallback"] = True
            chunk_metadata.append(chunk_meta_entry)
            summarized_chunks.append(f"Chunk {info['index']} Summary:\n{chunk_summary}")

        synthesis_source = "\n\n".join(summarized_chunks)

    current_date = time.strftime("%Y-%m-%d")

    # REDUCE: single-chunk synthesizes from the raw numbered evidence above;
    # multi-chunk synthesizes from the MAP-phase chunk summaries (restoring
    # the server's context-bounding design) which still carry the "[n]"
    # markers per the preservation instruction above.
    analyze_search_results_prompt_2 = render_internal_prompt(
        "websearch.answer_synthesis",
        concatenated_texts=synthesis_source,
        current_date=current_date,
        question=question,
    )

    input_data = "Follow the above instructions."

    try:
        logger.info("Generating the report")
        messages_payload = [
            {
                "role": "user",
                "content": input_data + "\n\n" + analyze_search_results_prompt_2,
            }
        ]
        returned_response = chat_reply_text(
            chat_api_call(
                api_endpoint=api_endpoint,
                messages_payload=messages_payload,
                api_key=None,
                temp=0.7,
                system_message=None,
                streaming=False,
                minp=None,
                maxp=None,
                model=None,
                topk=None,
                topp=None,
            )
        )
        logger.debug(f"Returned response from LLM: {returned_response}")
        if returned_response:
            # Citation verification (task-16331): resolve the "[n]" markers
            # against the numbered evidence ids and quote-check quoted spans
            # against the scraped originals -- pure string work, no network.
            # Unknown ids are flagged inline ("[n?]") and counted, never
            # deleted; failure/empty branches below carry no verdict rather
            # than a fabricated clean one.
            cv = deep_search_citations.verify_citations(
                returned_response, evidence_payload
            )
            success_answer: FinalAnswerDict = {
                "text": cv["annotated_text"],
                "evidence": evidence_payload,
                "confidence": _estimate_confidence(
                    len(evidence_payload), len(chunk_infos), failed_chunks, has_llm=True
                ),
                "chunks": chunk_metadata,
                "citation_verification": {
                    key: cv[key]
                    for key in (
                        "markers_total",
                        "markers_resolved",
                        "unknown_marker_ids",
                        "quotes_checked",
                        "quotes_verified",
                        "quotes_misquoted",
                        "uncited_sentences",
                        "claims",
                    )
                },
            }
            return success_answer
    except Exception as e:
        logger.error(f"Error aggregating results: {e}")

    logger.error("Could not create the report due to an error.")
    failure_answer: FinalAnswerDict = {
        "text": "Could not create the report due to an error.",
        "evidence": evidence_payload,
        "confidence": _estimate_confidence(
            len(evidence_payload), len(chunk_infos), len(chunk_infos), has_llm=False
        ),
        "chunks": chunk_metadata,
    }
    return failure_answer


#
# End of Orchestration functions
#######################################################################################################################


#######################################################################################################################
#
# Search Engine Functions


# FIXME
def perform_websearch(
    search_engine,
    search_query,
    content_country,
    search_lang,
    output_lang,
    result_count,
    date_range=None,
    safesearch=None,
    site_blacklist=None,
    exactTerms=None,
    excludeTerms=None,
    filter=None,
    geolocation=None,
    search_result_language=None,
    sort_results_by=None,
):
    """
    Execute a web search using the specified search engine.

    This is the main dispatcher function that routes searches to
    the appropriate engine-specific implementation.

    Args:
        search_engine (str): Engine name (google, bing, duckduckgo, etc.)
        search_query (str): The search query
        content_country (str): Country code for localized results
        search_lang (str): Language code for search
        output_lang (str): Language code for output
        result_count (int): Number of results to return
        date_range (str, optional): Time filter (e.g., 'y', 'w', 'm')
        safesearch (str, optional): Safe search level
        site_blacklist (list, optional): Sites to exclude
        exactTerms (str, optional): Exact phrase to match
        excludeTerms (str, optional): Terms to exclude
        filter (str, optional): Additional filters
        geolocation (str, optional): Geographic location
        search_result_language (str, optional): Language filter
        sort_results_by (str, optional): Sort order

    Returns:
        Dict: Standardized search results or error dict

    Supported Engines:
        - google: Google Custom Search API
        - bing: Bing Search API
        - brave: Brave Search API
        - duckduckgo: DuckDuckGo (HTML scraping)
        - kagi: Kagi Search API
        - exa: Exa Search API
        - serper: Serper (Google Search proxy) API
        - tavily: Tavily Search API
        - searx: SearX instance
        - yandex: Yandex Cloud Search API v2
    """
    start_time = time.time()

    # Log search attempt
    log_counter(
        "websearch_perform_search_attempt",
        labels={
            "engine": search_engine.lower(),
            "country": content_country,
            "lang": search_lang,
        },
    )

    try:
        if search_engine.lower() == "baidu":
            web_search_results = search_web_baidu(search_query, None, None)

        elif search_engine.lower() == "bing":
            # Prepare the arguments for search_web_bing
            bing_args = {
                "search_query": search_query,
                "bing_lang": search_lang,
                "bing_country": content_country,
                "result_count": result_count,
                "bing_api_key": loaded_config_data["search_engines"].get(
                    "bing_api_key"
                ),  # Fetch Bing API key from config
                "date_range": date_range,
            }

            # Call the search_web_bing function with the prepared arguments
            web_search_results = search_web_bing(**bing_args)

        elif search_engine.lower() == "brave":
            web_search_results = search_web_brave(
                search_query,
                content_country,
                search_lang,
                output_lang,
                result_count,
                safesearch,
                site_blacklist,
                date_range,
            )

        elif search_engine.lower() == "duckduckgo":
            # Prepare the arguments for search_web_duckduckgo
            ddg_args = {
                "keywords": search_query,
                "region": f"{content_country.lower()}-{search_lang.lower()}",  # Format: "us-en"
                "timelimit": date_range[0]
                if date_range
                else None,  # Use first character of date_range (e.g., "y" -> "y")
                "max_results": result_count,
            }

            # Call the search_web_duckduckgo function with the prepared arguments
            ddg_results = search_web_duckduckgo(**ddg_args)

            # Wrap the results in a dictionary to match the expected format
            web_search_results = {"results": ddg_results}

        elif search_engine.lower() == "google":
            # Convert site_blacklist list to a comma-separated string
            if site_blacklist and isinstance(site_blacklist, list):
                site_blacklist = ",".join(site_blacklist)

            # Prepare the arguments for search_web_google
            google_args = {
                "search_query": search_query,
                "google_search_api_key": loaded_config_data["search_engines"][
                    "google_search_api_key"
                ],
                "google_search_engine_id": loaded_config_data["search_engines"][
                    "google_search_engine_id"
                ],
                "result_count": result_count,
                "c2coff": "1",  # Default value
                "results_origin_country": content_country,
                "ui_language": output_lang,
                "search_result_language": search_result_language
                or "lang_en",  # Default value
                "geolocation": geolocation or "us",  # Default value
                "safesearch": safesearch or "off",  # Default value,
            }

            # If site_blacklist has multiple domains, do not use siteSearch
            if site_blacklist and len(site_blacklist) == 1:
                google_args["siteSearch"] = site_blacklist[0]
                google_args["siteSearchFilter"] = "e"
            else:
                # Do not use siteSearch for multiple domains
                # Either skip it entirely or see Option 2 below
                google_args.pop("siteSearch", None)
                google_args.pop("siteSearchFilter", None)

            # Add optional parameters only if they are provided
            if date_range:
                google_args["date_range"] = date_range
            if exactTerms:
                google_args["exactTerms"] = exactTerms
            if excludeTerms:
                google_args["excludeTerms"] = excludeTerms
            if filter:
                google_args["filter"] = filter
            if site_blacklist:
                google_args["site_blacklist"] = site_blacklist
            if sort_results_by:
                google_args["sort_results_by"] = sort_results_by

            # Call the search_web_google function with the prepared arguments
            web_search_results = search_web_google(**google_args)  # raw JSON
            web_search_results_dict = process_web_search_results(
                web_search_results, "google"
            )

            # Log success metrics for google
            duration = time.time() - start_time
            result_count = len(web_search_results_dict.get("results", []))
            log_histogram(
                "websearch_perform_search_duration",
                duration,
                labels={
                    "engine": "google",
                    "country": content_country,
                    "result_count": str(result_count),
                },
            )
            log_counter(
                "websearch_perform_search_success",
                labels={"engine": "google", "result_count": str(result_count)},
            )

            return web_search_results_dict

        elif search_engine.lower() == "kagi":
            web_search_results = search_web_kagi(search_query, content_country)

        elif search_engine.lower() == "exa":
            web_search_results = search_web_exa(search_query, result_count)

        elif search_engine.lower() == "serper":
            web_search_results = search_web_serper(
                search_query, content_country, search_lang, result_count
            )

        elif search_engine.lower() == "tavily":
            web_search_results = search_web_tavily(
                search_query, result_count, site_blacklist
            )

        elif search_engine.lower() == "searx":
            web_search_results = search_web_searx(
                search_query,
                language="auto",
                time_range="",
                safesearch=0,
                pageno=1,
                categories="general",
            )

        elif search_engine.lower() == "yandex":
            web_search_results = search_web_yandex(search_query, result_count)

        else:
            return f"Error: Invalid Search Engine Name {search_engine}"

        # Process the raw search results
        web_search_results_dict = process_web_search_results(
            web_search_results, search_engine
        )
        # FIXME
        # logger.debug("After process_web_search_results:")
        # logger.debug(json.dumps(web_search_results_dict, indent=2))

        # Log success metrics
        duration = time.time() - start_time
        result_count = len(web_search_results_dict.get("results", []))
        log_histogram(
            "websearch_perform_search_duration",
            duration,
            labels={
                "engine": search_engine.lower(),
                "country": content_country,
                "result_count": str(result_count),
            },
        )
        log_counter(
            "websearch_perform_search_success",
            labels={"engine": search_engine.lower(), "result_count": str(result_count)},
        )

        return web_search_results_dict

    except Exception as e:
        # Log error metrics
        duration = time.time() - start_time
        log_histogram(
            "websearch_perform_search_duration",
            duration,
            labels={
                "engine": search_engine.lower(),
                "country": content_country,
                "result_count": "0",
            },
        )
        log_counter(
            "websearch_perform_search_error",
            labels={"engine": search_engine.lower(), "error_type": type(e).__name__},
        )

        return {"processing_error": f"Error performing web search: {str(e)}"}


def test_perform_websearch_google():
    # Google Searches
    try:
        test_1 = perform_websearch(
            "google", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 1: {test_1}")
        # FIXME - Fails. Need to fix arg formatting
        test_2 = perform_websearch(
            "google",
            "What is the capital of France?",
            "US",
            "en",
            "en",
            10,
            date_range="y",
            safesearch="active",
            site_blacklist=["spam-site.com"],
        )
        print(f"Test 2: {test_2}")
        test_3 = perform_websearch(
            "google", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 3: {test_3}")
    except Exception as e:
        print(f"Error performing google searches: {str(e)}")
    pass


def test_perform_websearch_bing():
    # Bing Searches
    try:
        test_4 = perform_websearch(
            "bing", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 4: {test_4}")
        test_5 = perform_websearch(
            "bing",
            "What is the capital of France?",
            "US",
            "en",
            "en",
            10,
            date_range="y",
        )
        print(f"Test 5: {test_5}")
    except Exception as e:
        print(f"Error performing bing searches: {str(e)}")


def test_perform_websearch_brave():
    # Brave Searches
    try:
        test_7 = perform_websearch(
            "brave", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 7: {test_7}")
    except Exception as e:
        print(f"Error performing brave searches: {str(e)}")


def test_perform_websearch_ddg():
    # DuckDuckGo Searches
    try:
        test_6 = perform_websearch(
            "duckduckgo", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 6: {test_6}")
        test_7 = perform_websearch(
            "duckduckgo",
            "What is the capital of France?",
            "US",
            "en",
            "en",
            10,
            date_range="y",
        )
        print(f"Test 7: {test_7}")
    except Exception as e:
        print(f"Error performing duckduckgo searches: {str(e)}")


# FIXME
def test_perform_websearch_kagi():
    # Kagi Searches
    try:
        test_8 = perform_websearch(
            "kagi", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 8: {test_8}")
    except Exception as e:
        print(f"Error performing kagi searches: {str(e)}")


# FIXME
def test_perform_websearch_serper():
    # Serper Searches
    try:
        test_9 = perform_websearch(
            "serper", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 9: {test_9}")
    except Exception as e:
        print(f"Error performing serper searches: {str(e)}")


# FIXME
def test_perform_websearch_tavily():
    # Tavily Searches
    try:
        test_10 = perform_websearch(
            "tavily", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 10: {test_10}")
    except Exception as e:
        print(f"Error performing tavily searches: {str(e)}")


# FIXME
def test_perform_websearch_searx():
    # Searx Searches
    try:
        test_11 = perform_websearch(
            "searx", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 11: {test_11}")
    except Exception as e:
        print(f"Error performing searx searches: {str(e)}")


# FIXME
def test_perform_websearch_yandex():
    # Yandex Searches
    try:
        test_12 = perform_websearch(
            "yandex", "What is the capital of France?", "US", "en", "en", 10
        )
        print(f"Test 12: {test_12}")
    except Exception as e:
        print(f"Error performing yandex searches: {str(e)}")
    pass


#
######################### Search Result Parsing ##################################################################
#


def process_web_search_results(search_results: Union[Dict, str], search_engine: str) -> Dict:
    """
    Process raw search results into standardized format.

    Converts engine-specific result formats into a common structure
    for consistent handling across different search providers.

    Args:
        search_results (Union[Dict, str]): Raw results from search engine; dict of raw results; for tavily/searx a raw string payload is also accepted
        search_engine (str): Name of the search engine

    Returns:
        Dict: Standardized results containing:
            - search_engine: Engine used
            - results: List of result items with:
                - title: Result title
                - url: Result URL
                - content: Snippet/description
                - metadata: Additional info (date, author, etc.)
            - total_results_found: Total results available
            - search_time: Search duration
            - error: Any error messages

    Standard Result Structure:
        {
            "title": str,
            "url": str,
            "content": str,
            "metadata": {
                "date_published": Optional[str],
                "author": Optional[str],
                "source": Optional[str],
                "language": Optional[str],
                "relevance_score": Optional[float],
                "snippet": Optional[str]
            }
        }
    Processes search results from a search engine and formats them into a standardized dictionary structure.

    Args:
        search_results (Dict): The raw search results from the search engine.
        search_engine (str): The name of the search engine (e.g., "Google", "Bing").

    Returns:
        Dict: A dictionary containing the processed search results in the specified structure.

    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range", None),
        "safesearch": search_results.get("safesearch", None),
        "site_blacklist": search_results.get("site_blacklist", None),
        "exactTerms": search_results.get("exactTerms", None),
        "excludeTerms": search_results.get("excludeTerms", None),
        "filter": search_results.get("filter", None),
        "geolocation": search_results.get("geolocation", None),
        "search_result_language": search_results.get("search_result_language", None),
        "sort_results_by": search_results.get("sort_results_by", None),
        "results": [
            {
                "title": str,
                "url": str,
                "content": str,
                "metadata": {
                    "date_published": Optional[str],
                    "author": Optional[str],
                    "source": Optional[str],
                    "language": Optional[str],
                    "relevance_score": Optional[float],
                    "snippet": Optional[str]
                }
            },
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error", None),
        "processing_error": None
        ]
    """
    # Validate input parameters. Every backend but tavily/searx always
    # returns a dict; a string payload for those two is a valid input too
    # (tavily: a request-error message; searx: its ENTIRE payload, success
    # or failure, is JSON-encoded as a string), deferred to the
    # engine-specific parser below, which raises ValueError to surface it as
    # processing_error (task-2990). The str allowance is deliberately scoped
    # to just these two engines: widening it for every engine would let a
    # stray string reach e.g. parse_brave_results, whose `"query" in
    # raw_results`-style membership checks against a str run silently and
    # can produce zero results with no error at all -- the exact defect
    # class this task exists to close, just relocated. Reject anything else
    # outright.
    if not isinstance(search_results, dict) and not (
        isinstance(search_results, str) and search_engine.lower() in ("tavily", "searx")
    ):
        raise TypeError("search_results must be a dictionary (or a string for tavily/searx)")

    # Only a dict carries this request-echo metadata; a string payload (see
    # above) has none of it, so every field below falls back to its default.
    _meta = search_results if isinstance(search_results, dict) else {}

    # Initialize the output dictionary with default values
    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": _meta.get("search_query", ""),
        "content_country": _meta.get("content_country", ""),
        "search_lang": _meta.get("search_lang", ""),
        "output_lang": _meta.get("output_lang", ""),
        "result_count": _meta.get("result_count", 0),
        "date_range": _meta.get("date_range", None),
        "safesearch": _meta.get("safesearch", None),
        "site_blacklist": _meta.get("site_blacklist", None),
        "exactTerms": _meta.get("exactTerms", None),
        "excludeTerms": _meta.get("excludeTerms", None),
        "filter": _meta.get("filter", None),
        "geolocation": _meta.get("geolocation", None),
        "search_result_language": _meta.get("search_result_language", None),
        "sort_results_by": _meta.get("sort_results_by", None),
        "results": [],
        "total_results_found": _meta.get("total_results_found", 0),
        "search_time": _meta.get("search_time", 0.0),
        "error": _meta.get("error", None),
        "processing_error": None,
    }
    try:
        # Parse results based on the search engine
        if search_engine.lower() == "baidu":
            pass  # Placeholder for Baidu-specific parsing
        elif search_engine.lower() == "bing":
            parse_bing_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "brave":
            parse_brave_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "duckduckgo":
            parse_duckduckgo_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "exa":
            parse_exa_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "google":
            parse_google_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "kagi":
            parse_kagi_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "serper":
            parse_serper_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "tavily":
            parse_tavily_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "searx":
            parse_searx_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "yandex":
            parse_yandex_results(search_results, web_search_results_dict)
        else:
            raise ValueError(f"Error: Invalid Search Engine Name {search_engine}")

    except Exception as e:
        web_search_results_dict["processing_error"] = (
            f"Error processing search results: {str(e)}"
        )
        logger.error(f"Error in process_web_search_results: {str(e)}")

    return web_search_results_dict


def parse_html_search_results_generic(soup):
    results = []
    for result in soup.find_all("div", class_="result"):
        title = result.find("h3").text if result.find("h3") else ""
        url = (
            result.find("a", class_="url")["href"]
            if result.find("a", class_="url")
            else ""
        )
        content = (
            result.find("p", class_="content").text
            if result.find("p", class_="content")
            else ""
        )
        published_date = (
            result.find("span", class_="published_date").text
            if result.find("span", class_="published_date")
            else ""
        )

        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "publishedDate": published_date,
            }
        )
    return results


######################### Baidu Search #########################
#
# https://cloud.baidu.com/doc/APIGUIDE/s/Xk1myz05f
# https://oxylabs.io/blog/how-to-scrape-baidu-search-results
def search_web_baidu(arg1, arg2, arg3):
    pass


def test_baidu_search(arg1, arg2, arg3):
    result = search_web_baidu(arg1, arg2, arg3)
    return result


def search_parse_baidu_results():
    pass


######################### Bing Search #########################
#
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview0
# https://learn.microsoft.com/en-us/bing/search-apis/bing-news-search/overview
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
# Country/Language code: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes
# https://github.com/Azure-Samples/cognitive-services-REST-api-samples/tree/master/python/Search
@retry_on_transient_error(max_tries=3, backoff_factor=1.5)
def search_web_bing(
    search_query,
    bing_lang=None,
    bing_country=None,
    result_count=None,
    bing_api_key=None,
    date_range=None,
):
    """
    Perform a search using Bing Search API.

    Args:
        search_query (str): The search query
        bing_lang (str, optional): Language code (e.g., 'en', 'fr', 'de')
        bing_country (str, optional): Country code (e.g., 'US', 'GB', 'FR')
        result_count (int, optional): Number of results to return
        bing_api_key (str, optional): Bing Search API key
        date_range (str, optional): Date range for results ('day', 'week', 'month', or 'YYYY-MM-DD..YYYY-MM-DD')

    Returns:
        dict: Raw Bing search results

    Raises:
        ValueError: If API key is missing or invalid
        RequestException: For HTTP errors
        ConnectionError: For network issues
        TimeoutError: If the request times out

    Note:
        This function uses the retry_on_transient_error decorator to automatically
        retry on transient errors like network issues or server errors.
    """
    # Load Search API URL from config file
    search_url = loaded_config_data["search_engines"]["bing_search_api_url"]

    if not bing_api_key:
        # load key from config file
        bing_api_key = loaded_config_data["search_engines"]["bing_search_api_key"]
        if not bing_api_key:
            raise ValueError("Please Configure a valid Bing Search API key")

    # Get default result count from config if not provided
    if not result_count:
        result_count = loaded_config_data["search_engines"].get("search_result_max", 10)

    # Get default language from config if not provided
    if not bing_lang:
        bing_lang = loaded_config_data["search_engines"].get("bing_language_code", "en")

    # Get default country from config if not provided
    if not bing_country:
        bing_country = loaded_config_data["search_engines"].get(
            "bing_country_code", "US"
        )

    # Construct market code (language-COUNTRY format)
    mkt = f"{bing_lang}-{bing_country}"

    # Construct request parameters
    params = {
        "q": search_query,
        "mkt": mkt,
        "textDecorations": True,
        "textFormat": "HTML",
        "count": result_count,
        "safeSearch": "Moderate",
    }

    # Add optional parameters if provided
    if date_range:
        params["freshness"] = date_range

    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}

    # Call the API with better error handling
    try:
        logger.debug(f"Sending Bing search request: URL={search_url}, params={params}")

        # Create a session with retry capability
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504, 429],
            allowed_methods=["GET"],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        # Send the request with the session
        response = session.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        logger.debug("Bing search response headers:")
        logger.debug(response.headers)

        try:
            bing_search_results = response.json()
            logger.debug("Bing search response received successfully")

            # Log metrics for successful search
            log_counter("search.bing.success", 1)
            log_histogram(
                "search.bing.result_count",
                len(bing_search_results.get("webPages", {}).get("value", [])),
            )

            return bing_search_results
        except json.JSONDecodeError as jde:
            # Handle invalid JSON response
            logger.error(f"Invalid JSON response from Bing: {jde}")
            raise ValueError(f"Invalid response from Bing (not valid JSON): {jde}")

    except Exception as ex:
        # Use common error handling function
        error = handle_search_error(ex, "Bing")

        # Log metrics for failed search
        log_counter("search.bing.error", 1)
        log_counter(f"search.bing.error.{error.__class__.__name__}", 1)

        raise error


def test_search_web_bing():
    """
    Test function for Bing search with different scenarios.
    This function tests the search_web_bing function with various parameters
    and validates the results.
    """
    try:
        logger.info("Testing Bing search with default parameters...")
        search_query = "How can I get started learning machine learning?"

        # Test with default parameters
        result = search_web_bing(search_query)

        # Validate the result structure
        if not isinstance(result, dict):
            logger.error(f"Expected dict result, got {type(result)}")
            return

        # Check if we got any results
        if "webPages" not in result:
            logger.warning("No web pages found in results")
        else:
            web_pages = result["webPages"]
            logger.info(f"Found {len(web_pages.get('value', []))} web page results")

        # Parse the results
        output_dict = {"results": []}
        parse_bing_results(result, output_dict)
        logger.info(f"Parsed {len(output_dict['results'])} results")

        # Test with different language and country
        logger.info("Testing Bing search with different language and country...")
        try:
            result_fr = search_web_bing(
                search_query, bing_lang="fr", bing_country="FR", result_count=5
            )

            # Parse the French results
            output_dict_fr = {"results": []}
            parse_bing_results(result_fr, output_dict_fr)
            logger.info(f"Parsed {len(output_dict_fr['results'])} French results")

        except Exception as e:
            logger.error(f"Error testing French search: {e}")

        # Test with date range
        logger.info("Testing Bing search with date range...")
        try:
            result_recent = search_web_bing(search_query, date_range="month")

            # Parse the recent results
            output_dict_recent = {"results": []}
            parse_bing_results(result_recent, output_dict_recent)
            logger.info(f"Parsed {len(output_dict_recent['results'])} recent results")

        except Exception as e:
            logger.error(f"Error testing date range search: {e}")

        # Print the original results for reference
        logger.info("Original Bing search results:")
        logger.info(json.dumps(result, indent=2))

        # Print the parsed results
        logger.info("Parsed Bing search results:")
        logger.info(json.dumps(output_dict, indent=2))

        return output_dict

    except ValueError as ve:
        logger.error(f"Value error in Bing search test: {ve}")
        print(f"Value error: {ve}")

    except ConnectionError as ce:
        logger.error(f"Connection error in Bing search test: {ce}")
        print(f"Connection error: {ce}")

    except TimeoutError as te:
        logger.error(f"Timeout error in Bing search test: {te}")
        print(f"Timeout error: {te}")

    except Exception as e:
        logger.error(f"Unexpected error in Bing search test: {e}")
        print(f"Error: {e}")

    return None


def parse_bing_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Bing search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Bing API response
        output_dict (Dict): Dictionary to store processed results
    """
    logger.info(f"Raw Bing results received: {json.dumps(raw_results, indent=2)}")
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract web pages results
        if "webPages" in raw_results:
            web_pages = raw_results["webPages"]
            output_dict["total_results_found"] = web_pages.get(
                "totalEstimatedMatches", 0
            )

            for result in web_pages.get("value", []):
                processed_result = {
                    "title": result.get("name", ""),
                    "url": result.get("url", ""),
                    "content": result.get("snippet", ""),
                    "metadata": {
                        "date_published": None,  # Bing doesn't typically provide this
                        "author": None,  # Bing doesn't typically provide this
                        "source": result.get("displayUrl", None),
                        "language": None,  # Could be extracted from result.get("language") if available
                        "relevance_score": None,  # Could be calculated from result.get("rank") if available
                        "snippet": result.get("snippet", None),
                    },
                }
                output_dict["results"].append(processed_result)

        # Optionally process other result types
        if "news" in raw_results:
            for news_item in raw_results["news"].get("value", []):
                processed_result = {
                    "title": news_item.get("name", ""),
                    "url": news_item.get("url", ""),
                    "content": news_item.get("description", ""),
                    "metadata": {
                        "date_published": news_item.get("datePublished", None),
                        "author": news_item.get("provider", [{}])[0].get("name", None),
                        "source": news_item.get("provider", [{}])[0].get("name", None),
                        "language": None,
                        "relevance_score": None,
                        "snippet": news_item.get("description", None),
                    },
                }
                output_dict["results"].append(processed_result)

        # Add spell suggestions if available
        if "spellSuggestion" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spellSuggestion"]

        # Add related searches if available
        if "relatedSearches" in raw_results:
            output_dict["related_searches"] = [
                item.get("text", "")
                for item in raw_results["relatedSearches"].get("value", [])
            ]

    except Exception as e:
        logger.error(f"Error processing Bing results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Bing results: {str(e)}"


######################### Brave Search #########################
#
# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_web_brave(
    search_term,
    country,
    search_lang,
    ui_lang,
    result_count,
    safesearch="moderate",
    brave_api_key=None,
    result_filter=None,
    search_type="ai",
    date_range=None,
):
    search_url = "https://api.search.brave.com/res/v1/web/search"
    if not brave_api_key and search_type == "web":
        # load key from config file
        brave_api_key = loaded_config_data["search_engines"]["brave_search_api_key"]
        if not brave_api_key:
            raise ValueError("Please provide a valid Brave Search API subscription key")
    if not country:
        loaded_config_data["search_engines"]["search_engine_country_code_brave"]
    else:
        country = "US"
    if not search_lang:
        search_lang = "en"
    if not ui_lang:
        ui_lang = "en"
    if not result_count:
        result_count = 10
    # if not date_range:
    #     date_range = "month"
    if not result_filter:
        result_filter = "webpages"
    if search_type == "ai":
        brave_api_key = loaded_config_data["search_engines"]["brave_search_ai_api_key"]
    else:
        raise ValueError("Invalid search type. Please choose 'ai' or 'web'.")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_api_key,
    }

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {
        "q": search_term,
        "textDecorations": True,
        "textFormat": "HTML",
        "count": result_count,
        "freshness": date_range,
        "promote": "webpages",
        "safeSearch": "Moderate",
    }

    # task-3060: bound worst-case latency -- an unresponsive Brave endpoint
    # must not hang perform_websearch (and the deep-search pipeline) indefinitely.
    response = requests.get(search_url, headers=headers, params=params, timeout=SEARCH_BACKEND_TIMEOUT_S)
    response.raise_for_status()
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results


def test_search_brave():
    search_term = "How can I bake a cherry cake"
    country = "US"
    search_lang = "en"
    ui_lang = "en"
    result_count = 10
    safesearch = "moderate"
    date_range = None
    result_filter = None
    result = search_web_brave(
        search_term,
        country,
        search_lang,
        ui_lang,
        result_count,
        safesearch,
        date_range,
        result_filter,
    )
    print("Brave Search Results:")
    print(result)

    output_dict = {"results": []}
    parse_brave_results(result, output_dict)
    print("Parsed Brave Results:")
    print(json.dumps(output_dict, indent=2))


def parse_brave_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Brave search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Brave API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract query information
        if "query" in raw_results:
            query_info = raw_results["query"]
            output_dict.update(
                {
                    "search_query": query_info.get("original", ""),
                    "content_country": query_info.get("country", ""),
                    "city": query_info.get("city", ""),
                    "state": query_info.get("state", ""),
                    "more_results_available": query_info.get(
                        "more_results_available", False
                    ),
                }
            )

        # Process web results
        if "web" in raw_results and "results" in raw_results["web"]:
            for result in raw_results["web"]["results"]:
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("description", ""),
                    "metadata": {
                        "date_published": result.get("page_age", None),
                        "author": None,
                        "source": result.get("profile", {}).get("name", None),
                        "language": result.get("language", None),
                        "relevance_score": None,
                        "snippet": result.get("description", None),
                        "family_friendly": result.get("family_friendly", None),
                        "type": result.get("type", None),
                        "subtype": result.get("subtype", None),
                        "thumbnail": result.get("thumbnail", {}).get("src", None),
                    },
                }
                output_dict["results"].append(processed_result)

        # Update total results count
        if "mixed" in raw_results:
            output_dict["total_results_found"] = len(
                raw_results["mixed"].get("main", [])
            )

        # Set family friendly status
        if "mixed" in raw_results:
            output_dict["family_friendly"] = raw_results.get("family_friendly", True)

    except Exception as e:
        logger.error(f"Error processing Brave results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Brave results: {str(e)}"


def test_parse_brave_results():
    pass


######################### DuckDuckGo Search #########################
#
# https://github.com/deedy5/duckduckgo_search
# Copied request format/structure from https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def search_web_duckduckgo(
    keywords: str,
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
    max_results: Optional[int] = None,
) -> list[dict[str, str]]:
    assert keywords, "keywords is mandatory"

    if not LXML_AVAILABLE:
        logger.error(
            "lxml not available for DuckDuckGo search. Install with: pip install tldw_chatbook[websearch]"
        )
        return []

    payload = {
        "q": keywords,
        "s": "0",
        "o": "json",
        "api": "d.js",
        "vqd": "",
        "kl": region,
        "bing_market": region,
    }

    def _normalize_url(url: str) -> str:
        """Unquote URL and replace spaces with '+'."""
        return unquote(url).replace(" ", "+") if url else ""

    def _normalize(raw_html: str) -> str:
        """Strip HTML tags from the raw_html string."""
        REGEX_STRIP_TAGS = re.compile("<.*?>")
        return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""

    if timelimit:
        payload["df"] = timelimit

    cache = set()
    results: list[dict[str, str]] = []

    for _ in range(5):
        # task-3060: bound worst-case latency per bootstrap/pagination call
        # (this loop can issue up to 5 requests.post calls, all this one site).
        response = requests.post("https://html.duckduckgo.com/html", data=payload, timeout=SEARCH_BACKEND_TIMEOUT_S)
        resp_content = response.content
        if b"No  results." in resp_content:
            return results

        tree = document_fromstring(resp_content)
        elements = tree.xpath("//div[h2]")
        if not isinstance(elements, list):
            return results

        for e in elements:
            if isinstance(e, _Element):
                hrefxpath = e.xpath("./a/@href")
                href = (
                    str(hrefxpath[0])
                    if hrefxpath and isinstance(hrefxpath, list)
                    else None
                )
                if (
                    href
                    and href not in cache
                    and not href.startswith(
                        (
                            "http://www.google.com/search?q=",
                            "https://duckduckgo.com/y.js?ad_domain",
                        )
                    )
                ):
                    cache.add(href)
                    titlexpath = e.xpath("./h2/a/text()")
                    title = (
                        str(titlexpath[0])
                        if titlexpath and isinstance(titlexpath, list)
                        else ""
                    )
                    bodyxpath = e.xpath("./a//text()")
                    body = (
                        "".join(str(x) for x in bodyxpath)
                        if bodyxpath and isinstance(bodyxpath, list)
                        else ""
                    )
                    results.append(
                        {
                            "title": _normalize(title),
                            "href": _normalize_url(href),
                            "body": _normalize(body),
                        }
                    )
                    if max_results and len(results) >= max_results:
                        return results

        npx = tree.xpath('.//div[@class="nav-link"]')
        if not npx or not max_results:
            return results
        next_page = npx[-1] if isinstance(npx, list) else None
        if isinstance(next_page, _Element):
            names = next_page.xpath('.//input[@type="hidden"]/@name')
            values = next_page.xpath('.//input[@type="hidden"]/@value')
            if isinstance(names, list) and isinstance(values, list):
                payload = {str(n): str(v) for n, v in zip(names, values)}

    return results


def test_search_duckduckgo():
    try:
        results = search_web_duckduckgo(
            keywords="How can I bake a cherry cake?",
            region="us-en",
            timelimit="w",
            max_results=10,
        )
        print(f"Number of results: {len(results)}")
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Snippet: {result['body']}")
            print("---")

        # Parse the results
        output_dict = {"results": []}
        parse_duckduckgo_results({"results": results}, output_dict)
        print("Parsed DuckDuckGo Results:")
        print(json.dumps(output_dict, indent=2))

    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except requests.RequestException as e:
        print(f"Request error: {str(e)}")


def parse_duckduckgo_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse DuckDuckGo search results and update the output dictionary

    Args:
        raw_results (Dict): Raw DuckDuckGo response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # DuckDuckGo results are in a list of dictionaries
        results = raw_results.get("results", [])

        for result in results:
            # Extract information directly from the dictionary
            title = result.get("title", "")
            url = result.get("href", "")
            snippet = result.get("body", "")

            # Log warnings for missing data
            if not title:
                logger.warning("Missing title in result")
            if not url:
                logger.warning("Missing URL in result")
            if not snippet:
                logger.warning("Missing snippet in result")

            # Add the processed result to the output dictionary
            processed_result = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": None,  # DuckDuckGo doesn't typically provide this
                    "author": None,  # DuckDuckGo doesn't typically provide this
                    "source": extract_domain(url) if url else None,
                    "language": None,  # DuckDuckGo doesn't typically provide this
                    "relevance_score": None,  # DuckDuckGo doesn't typically provide this
                    "snippet": snippet,
                },
            }

            output_dict["results"].append(processed_result)

        # Update total results count
        output_dict["total_results_found"] = len(output_dict["results"])

    except Exception as e:
        logger.error(f"Error processing DuckDuckGo results: {str(e)}")
        output_dict["processing_error"] = (
            f"Error processing DuckDuckGo results: {str(e)}"
        )


def extract_domain(url: str) -> str:
    """
    Extract domain name from URL

    Args:
        url (str): Full URL

    Returns:
        str: Domain name
    """
    try:
        from urllib.parse import urlparse

        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        return domain.replace("www.", "")
    except (ImportError, ValueError, AttributeError) as e:
        # ImportError if urllib.parse not available (very unlikely)
        # ValueError if URL is malformed
        # AttributeError if parsed_uri doesn't have expected attributes
        logger.debug(f"Failed to parse domain from URL '{url}': {e}")
        return url


def test_parse_duckduckgo_results():
    pass


######################### Google Search #########################
#
# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
def search_web_google(
    search_query: str,
    google_search_api_key: Optional[str] = None,
    google_search_engine_id: Optional[str] = None,
    result_count: Optional[int] = None,
    c2coff: Optional[str] = None,
    results_origin_country: Optional[str] = None,
    date_range: Optional[str] = None,
    exactTerms: Optional[str] = None,
    excludeTerms: Optional[str] = None,
    filter: Optional[str] = None,
    geolocation: Optional[str] = None,
    ui_language: Optional[str] = None,
    search_result_language: Optional[str] = None,
    safesearch: Optional[str] = None,
    site_blacklist: Optional[str] = None,
    sort_results_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a Google web search with the given parameters.

    :param search_query: The search query string
    :param google_search_api_key: Google Search API key
    :param google_search_engine_id: Google Search Engine ID
    :param result_count: Number of results to return
    :param c2coff: Enable/disable traditional Chinese search
    :param results_origin_country: Limit results to a specific country
    :param date_range: Limit results to a specific date range
    :param exactTerms: Exact terms that must appear in results
    :param excludeTerms: Terms that must not appear in results
    :param filter: Control duplicate content filter
    :param geolocation: Geolocation of the user
    :param ui_language: Language of the user interface
    :param search_result_language: Language of search results
    :param safesearch: Safe search setting
    :param site_blacklist: Single Site to exclude from search
    :param sort_results_by: Sorting criteria for results
    :return: JSON response from Google Search API
    """
    try:
        # Load Search API URL from config file
        search_url = loaded_config_data["search_engines"]["google_search_api_url"]
        logger.info(f"Using search URL: {search_url}")

        # Initialize params dictionary
        params: Dict[str, Any] = {"q": search_query}

        # Handle c2coff
        if c2coff is None:
            c2coff = loaded_config_data["search_engines"]["google_simp_trad_chinese"]
        if c2coff is not None:
            params["c2coff"] = c2coff

        # Handle results_origin_country
        if results_origin_country is None:
            limit_country_search = loaded_config_data["search_engines"][
                "limit_google_search_to_country"
            ]
            if limit_country_search:
                results_origin_country = loaded_config_data["search_engines"][
                    "google_search_country"
                ]
        if results_origin_country:
            params["cr"] = results_origin_country

        # Handle google_search_engine_id
        if google_search_engine_id is None:
            google_search_engine_id = loaded_config_data["search_engines"][
                "google_search_engine_id"
            ]
        if not google_search_engine_id:
            raise ValueError(
                "Please set a valid Google Search Engine ID in the config file"
            )
        params["cx"] = google_search_engine_id

        # Handle google_search_api_key
        if google_search_api_key is None:
            google_search_api_key = loaded_config_data["search_engines"][
                "google_search_api_key"
            ]
        if not google_search_api_key:
            raise ValueError(
                "Please provide a valid Google Search API subscription key"
            )
        params["key"] = google_search_api_key

        # Handle other parameters
        if result_count:
            params["num"] = result_count
        if date_range:
            params["dateRestrict"] = date_range
        if exactTerms:
            params["exactTerms"] = exactTerms
        if excludeTerms:
            params["excludeTerms"] = excludeTerms
        if filter:
            params["filter"] = filter
        if geolocation:
            params["gl"] = geolocation
        if ui_language:
            params["hl"] = ui_language
        if search_result_language:
            params["lr"] = search_result_language
        if safesearch is None:
            safesearch = loaded_config_data["search_engines"]["google_safe_search"]
        if safesearch:
            params["safe"] = safesearch
        if sort_results_by:
            params["sort"] = sort_results_by

        logger.info(f"Prepared parameters for Google Search: {params}")

        # Make the API call
        # task-3060: bound worst-case latency -- an unresponsive Google CSE
        # endpoint must not hang perform_websearch indefinitely.
        response = requests.get(search_url, params=params, timeout=SEARCH_BACKEND_TIMEOUT_S)
        response.raise_for_status()
        google_search_results = response.json()

        logger.info(
            f"Successfully retrieved search results. Items found: {len(google_search_results.get('items', []))}"
        )

        return google_search_results

    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise

    except RequestException as re:
        logger.error(f"Error during API request: {str(re)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise


def test_search_google():
    search_query = "How can I bake a cherry cake?"
    google_search_api_key = loaded_config_data["search_engines"][
        "google_search_api_key"
    ]
    google_search_engine_id = loaded_config_data["search_engines"][
        "google_search_engine_id"
    ]
    result_count = 10
    c2coff = "1"
    results_origin_country = "countryUS"
    date_range = None
    exactTerms = None
    excludeTerms = None
    filter = None
    geolocation = "us"
    ui_language = "en"
    search_result_language = "lang_en"
    safesearch = "off"
    site_blacklist = None
    sort_results_by = None
    result = search_web_google(
        search_query,
        google_search_api_key,
        google_search_engine_id,
        result_count,
        c2coff,
        results_origin_country,
        date_range,
        exactTerms,
        excludeTerms,
        filter,
        geolocation,
        ui_language,
        search_result_language,
        safesearch,
        site_blacklist,
        sort_results_by,
    )
    print(result)
    return result


def parse_google_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Google Custom Search API results and update the output dictionary.

    Args:
        raw_results (Dict): Raw Google API response.
        output_dict (Dict): Dictionary to store processed results.
    """
    logger.info(f"Raw results received: {json.dumps(raw_results, indent=2)}")
    # For debugging only FIXME
    logger.debug("Raw web_search_results from Google:")
    logger.debug(json.dumps(raw_results, indent=2))
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract search information
        if "searchInformation" in raw_results:
            search_info = raw_results["searchInformation"]
            output_dict["total_results_found"] = int(
                search_info.get("totalResults", "0")
            )
            output_dict["search_time"] = float(search_info.get("searchTime", 0.0))

        # Extract spelling suggestions
        if "spelling" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spelling"].get(
                "correctedQuery"
            )

        # Extract search parameters from queries
        if "queries" in raw_results and "request" in raw_results["queries"]:
            request = raw_results["queries"]["request"][0]
            output_dict.update(
                {
                    "search_query": request.get("searchTerms", ""),
                    "search_lang": request.get("language", ""),
                    "result_count": request.get("count", 0),
                    "safesearch": request.get("safe", None),
                    "exactTerms": request.get("exactTerms", None),
                    "excludeTerms": request.get("excludeTerms", None),
                    "filter": request.get("filter", None),
                    "geolocation": request.get("gl", None),
                    "search_result_language": request.get("hl", None),
                    "sort_results_by": request.get("sort", None),
                }
            )

        # Process search results
        if "items" in raw_results:
            for item in raw_results["items"]:
                processed_result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    # IMPORTANT: 'snippet' is used as 'content'
                    "content": item.get("snippet", ""),
                    "metadata": {
                        "date_published": item.get("pagemap", {})
                        .get("metatags", [{}])[0]
                        .get("article:published_time"),
                        "author": item.get("pagemap", {})
                        .get("metatags", [{}])[0]
                        .get("article:author"),
                        "source": item.get("displayLink", None),
                        "language": item.get("language", None),
                        "relevance_score": None,  # Google doesn't provide this directly
                        "snippet": item.get("snippet", None),
                        "file_format": item.get("fileFormat", None),
                        "mime_type": item.get("mime", None),
                        "cache_url": item.get("cacheId", None),
                    },
                }

                # Extract additional metadata if available
                if "pagemap" in item:
                    pagemap = item["pagemap"]
                    if "metatags" in pagemap and pagemap["metatags"]:
                        metatags = pagemap["metatags"][0]
                        processed_result["metadata"].update(
                            {
                                "description": metatags.get(
                                    "og:description", metatags.get("description")
                                ),
                                "keywords": metatags.get("keywords"),
                                "site_name": metatags.get("og:site_name"),
                            }
                        )

                output_dict["results"].append(processed_result)

        # Add pagination information
        output_dict["pagination"] = {
            "has_next": "nextPage" in raw_results.get("queries", {}),
            "has_previous": "previousPage" in raw_results.get("queries", {}),
            "current_page": raw_results.get("queries", {})
            .get("request", [{}])[0]
            .get("startIndex", 1),
        }

    except Exception as e:
        logger.error(f"Error processing Google results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Google results: {str(e)}"


def test_parse_google_results():
    parsed_results = {}
    raw_results = {}
    raw_results = test_search_google()
    parse_google_results(raw_results, parsed_results)
    print(f"Parsed search results: {parsed_results}")
    pass


######################### Kagi Search #########################
#
# https://help.kagi.com/kagi/api/search.html
def search_web_kagi(query: str, limit: int = 10) -> Dict:
    search_url = "https://kagi.com/api/v0/search"

    # load key from config file
    kagi_api_key = loaded_config_data["search_engines"]["kagi_search_api_key"]
    if not kagi_api_key:
        raise ValueError("Please provide a valid Kagi Search API subscription key")

    """
    Queries the Kagi Search API with the given query and limit.
    """
    if kagi_api_key is None:
        raise ValueError("API key is required.")

    headers = {"Authorization": f"Bot {kagi_api_key}"}
    endpoint = f"{search_url}/search"
    params = {"q": query, "limit": limit}

    # task-3060: bound worst-case latency -- an unresponsive Kagi endpoint
    # must not hang perform_websearch indefinitely.
    response = requests.get(endpoint, headers=headers, params=params, timeout=SEARCH_BACKEND_TIMEOUT_S)
    response.raise_for_status()
    logger.debug(response.json())
    return response.json()


def test_search_kagi():
    search_term = "How can I bake a cherry cake"
    result_count = 10
    result = search_web_kagi(search_term, result_count)
    print(result)


def parse_kagi_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Kagi search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Kagi API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract metadata
        if "meta" in raw_results:
            meta = raw_results["meta"]
            output_dict["search_time"] = (
                meta.get("ms", 0) / 1000.0
            )  # Convert to seconds
            output_dict["api_balance"] = meta.get("api_balance")
            output_dict["search_id"] = meta.get("id")
            output_dict["node"] = meta.get("node")

        # Process search results
        if "data" in raw_results:
            for item in raw_results["data"]:
                # Skip related searches (type 1)
                if item.get("t") == 1:
                    output_dict["related_searches"] = item.get("list", [])
                    continue

                # Process regular search results (type 0)
                if item.get("t") == 0:
                    processed_result = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("snippet", ""),
                        "metadata": {
                            "date_published": item.get("published"),
                            "author": None,  # Kagi doesn't typically provide this
                            "source": None,  # Could be extracted from URL if needed
                            "language": None,  # Kagi doesn't typically provide this
                            "relevance_score": None,
                            "snippet": item.get("snippet"),
                            "thumbnail": item.get("thumbnail", {}).get("url")
                            if "thumbnail" in item
                            else None,
                        },
                    }
                    output_dict["results"].append(processed_result)

            # Update total results count
            output_dict["total_results_found"] = len(
                [item for item in raw_results["data"] if item.get("t") == 0]
            )

    except Exception as e:
        output_dict["processing_error"] = f"Error processing Kagi results: {str(e)}"


def test_parse_kagi_results():
    pass


######################### SearX Search #########################
#
# https://searx.space
# https://searx.github.io/searx/dev/search_api.html
def searx_create_session() -> requests.Session:
    """
    Create a requests session with retry logic.
    """
    session = requests.Session()
    retries = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"],  # Only retry on GET requests
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def search_web_searx(
    search_query,
    language="auto",
    time_range="",
    safesearch=0,
    pageno=1,
    categories="general",
    searx_url=None,
):
    """
    Perform a search using a Searx instance.

    Args:
        search_query (str): The search query.
        language (str): Language for the search results.
        time_range (str): Time range for the search results.
        safesearch (int): Safe search level (0=off, 1=moderate, 2=strict).
        pageno (int): Page number of the results.
        categories (str): Categories to search in (e.g., 'general', 'news').
        searx_url (str): Custom Searx instance URL (optional).

    Returns:
        str: JSON string containing the search results or an error message.
    """
    # Use the provided Searx URL or fall back to the configured one
    if not searx_url:
        searx_url = loaded_config_data["search_engines"]["searx_search_api_url"]
    if not searx_url:
        return json.dumps(
            {
                "error": "SearX Search is disabled and no content was found. This functionality is disabled because the user has not set it up yet."
            }
        )

    # Validate and construct URL
    try:
        parsed_url = urlparse(searx_url)
        params = {
            "q": search_query,
            "language": language,
            "time_range": time_range,
            "safesearch": safesearch,
            "pageno": pageno,
            "categories": categories,
        }
        search_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(params)}"
        logger.info(f"Search URL: {search_url}")
    except Exception as e:
        return json.dumps({"error": f"Invalid URL configuration: {str(e)}"})

    # Perform the search request
    try:
        # Mimic browser headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Add a random delay to mimic human behavior
        delay = random.uniform(2, 5)  # Random delay between 2 and 5 seconds
        time.sleep(delay)

        session = searx_create_session()
        # task-3060: bound worst-case latency -- Session.get() does not
        # inherit a timeout from the Session itself, so it must be passed
        # per-request like every other engine here.
        response = session.get(search_url, headers=headers, timeout=SEARCH_BACKEND_TIMEOUT_S)
        response.raise_for_status()

        # Check if the response is JSON
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            search_data = response.json()
        else:
            # If not JSON, assume it's HTML and parse it
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            search_data = parse_html_search_results_generic(soup)

        # Process results
        data = []
        for result in search_data:
            data.append(
                {
                    "title": result.get("title"),
                    "link": result.get("url"),
                    "snippet": result.get("content"),
                    "publishedDate": result.get("publishedDate"),
                }
            )

        if not data:
            return json.dumps(
                {"error": "No information was found online for the search query."}
            )

        return json.dumps(data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for content: {str(e)}")
        return json.dumps(
            {"error": f"There was an error searching for content. {str(e)}"}
        )


def test_search_searx():
    # Use a different Searx instance to avoid rate limiting
    searx_url = "https://searx.be"  # Example of a different Searx instance
    result = search_web_searx(
        "What goes into making a cherry cake?", searx_url=searx_url
    )
    print(result)


def parse_searx_results(searx_search_results: "list | dict | str", web_search_results_dict: dict) -> None:
    """Parse SearX/SearXNG results into the standardized shape.

    Unlike every other backend in this file, the local `search_web_searx`
    always returns a JSON-encoded STRING: `json.dumps(hits)` on success
    (a list of `{title, link, snippet, publishedDate}` dicts), or
    `json.dumps({"error": ...})` when nothing was found or the request
    failed. A string is decoded first; an already-parsed list is also
    accepted defensively for direct/test callers. Only a decoded list is
    tolerated as real results -- a decoded dict never is: `{"error": ...}`
    re-raises with that message, and any other dict (or any non-list
    scalar) raises a generic shape error, both surfacing via the
    `process_web_search_results` seam as `processing_error` instead of
    silently producing zero results (task-2990).

    `link`/`url` and `snippet`/`content` are each read with the OR-fallback
    pair (search_web_searx's own hits use link/snippet; a raw SearXNG API
    hit -- as a caller might hand this parser directly -- uses url/content),
    matching the port reference this parser was adapted from.

    Args:
        searx_search_results: Raw Searx response, as returned by
            `search_web_searx` -- a JSON string encoding either a list of
            hits or an error dict, or an already-decoded list.
        web_search_results_dict: Output dict; mutated in place, appending
            standardized result entries to its "results" list.

    Raises:
        ValueError: when a string payload cannot be parsed as JSON, the
            decoded payload is an error dict, the decoded payload is
            anything other than a list, or a list element is not a dict.
    """
    if isinstance(searx_search_results, str):
        try:
            searx_search_results = json.loads(searx_search_results)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid Searx response: {e}") from e

    if isinstance(searx_search_results, dict) and "error" in searx_search_results:
        raise ValueError(searx_search_results["error"])

    if not isinstance(searx_search_results, list):
        raise ValueError("Unexpected Searx payload shape: expected a list of results")

    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []

    for i, item in enumerate(searx_search_results):
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected Searx result item at index {i}: expected an object")
        url = item.get("link") or item.get("url") or ""
        snippet = item.get("snippet") or item.get("content") or ""
        web_search_results_dict["results"].append({
            "title": item.get("title", ""),
            "url": url,
            "content": snippet,
            "metadata": {
                "date_published": item.get("publishedDate", None),
                "author": None,
                "source": None,
                "language": None,
                "relevance_score": None,
                "snippet": snippet or None,
            },
        })


######################### Serper.dev Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_serper(
    search_query: str,
    content_country: Optional[str] = None,
    search_lang: Optional[str] = None,
    result_count: Optional[int] = None,
) -> dict:
    """Query the Serper google-search API and return its raw JSON.

    Args:
        search_query: The query string.
        content_country: 2-letter country code for `gl` (lowercased; default "us").
        search_lang: Interface language for `hl` (default "en").
        result_count: Number of organic results (default 10).

    Returns:
        dict: Raw Serper response JSON (organic results under "organic").

    Raises:
        ValueError: when no Serper API key is configured.
        requests.exceptions.HTTPError: on non-2xx responses.
    """
    serper_api_key = loaded_config_data["search_engines"].get("serper_search_api_key", "")
    if not serper_api_key:
        raise ValueError("Please provide a valid Serper API key ([SearchEngines] serper_search_api_key)")
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    payload = {
        "q": search_query,
        "gl": (content_country or "us").lower(),
        "hl": search_lang or "en",
        "num": int(result_count) if result_count else 10,
    }
    response = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=SEARCH_BACKEND_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def parse_serper_results(serper_search_results: dict, web_search_results_dict: dict) -> None:
    """Parse Serper organic results into the standardized shape.

    answerBox/knowledgeGraph blocks are deliberately ignored — organic web
    results only, like every sibling parser (spec 2026-08-06 §2). `position`
    is stored as-is under metadata.position; relevance_score stays None
    (mapping rank into a "relevance" field would invert its meaning).

    Args:
        serper_search_results: Raw Serper response JSON, as returned by
            `search_web_serper` (organic results under "organic").
        web_search_results_dict: Output dict; mutated in place, appending
            standardized result entries to its "results" list.
    """
    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []
    for result in (serper_search_results or {}).get("organic", []):
        web_search_results_dict["results"].append({
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "content": result.get("snippet", ""),
            "metadata": {
                "date_published": result.get("date", None),
                "author": None,
                "source": None,
                "language": None,
                "relevance_score": None,
                "position": result.get("position", None),
                "snippet": result.get("snippet", None),
            },
        })


######################### Exa Search #########################
#
# https://exa.ai/docs/reference/search
def search_web_exa(search_query: str, result_count: Optional[int] = None) -> dict:
    """Query the Exa search API and return its raw JSON.

    Requests `contents.highlights` — billed as contents retrieval on top of
    the search call; a deliberate paid trade for snippet text (spec
    2026-08-06 §2), since a result without a snippet is nearly useless to
    the model.

    Args:
        search_query: The query string.
        result_count: numResults (default 10).

    Returns:
        dict: Raw Exa response JSON (results under "results").

    Raises:
        ValueError: when no Exa API key is configured.
        requests.exceptions.HTTPError: on non-2xx responses.
    """
    exa_api_key = loaded_config_data["search_engines"].get("exa_search_api_key", "")
    if not exa_api_key:
        raise ValueError("Please provide a valid Exa API key ([SearchEngines] exa_search_api_key)")
    headers = {"x-api-key": exa_api_key, "Content-Type": "application/json"}
    payload = {
        "query": search_query,
        "numResults": int(result_count) if result_count else 10,
        "type": "auto",
        "contents": {"highlights": True},
    }
    response = requests.post("https://api.exa.ai/search", headers=headers, json=payload, timeout=SEARCH_BACKEND_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def parse_exa_results(exa_search_results: dict, web_search_results_dict: dict) -> None:
    """Parse Exa results into the standardized shape (first highlight = snippet).

    Args:
        exa_search_results: Raw Exa response JSON, as returned by
            `search_web_exa` (results under "results").
        web_search_results_dict: Output dict; mutated in place, appending
            standardized result entries to its "results" list.
    """
    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []
    for result in (exa_search_results or {}).get("results", []):
        highlights = result.get("highlights") or []
        snippet = highlights[0] if highlights else ""
        web_search_results_dict["results"].append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": snippet,
            "metadata": {
                "date_published": result.get("publishedDate", None),
                "author": result.get("author", None),
                "source": None,
                "language": None,
                "relevance_score": None,
                "snippet": snippet or None,
            },
        })


######################### Tavily Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_tavily(
    search_query, result_count=10, site_whitelist=None, site_blacklist=None
):
    # Check if API URL is configured
    tavily_api_url = "https://api.tavily.com/search"

    tavily_api_key = loaded_config_data["search_engines"]["tavily_search_api_key"]

    # Prepare the request payload
    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "max_results": result_count,
    }

    # Add optional parameters if provided
    if site_whitelist:
        payload["include_domains"] = site_whitelist
    if site_blacklist:
        payload["exclude_domains"] = site_blacklist

    # Perform the search request
    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        }

        # task-3060: bound worst-case latency -- an unresponsive Tavily
        # endpoint must not hang perform_websearch indefinitely.
        response = requests.post(
            tavily_api_url, headers=headers, data=json.dumps(payload), timeout=SEARCH_BACKEND_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"There was an error searching for content. {str(e)}"


def test_search_tavily():
    result = search_web_tavily("How can I bake a cherry cake?")
    print(result)


def parse_tavily_results(tavily_search_results: "dict | str", web_search_results_dict: dict) -> None:
    """Parse Tavily results into the standardized shape.

    The local `search_web_tavily` backend returns `response.json()` (a
    dict with hits under "results") on success, or a plain error STRING
    on request failure (e.g. "There was an error searching for content.
    ...") -- unlike every other backend in this file, which always
    returns a dict. A string input is re-raised as ValueError so its text
    survives the `process_web_search_results` seam as `processing_error`,
    instead of silently producing zero results (task-2990).

    Tavily's `score` is a real 0-1 relevance score (unlike serper's
    `position`, which is a SERP rank and would invert the meaning of
    "relevance" if stored there), so it maps directly to
    `metadata.relevance_score`.

    Args:
        tavily_search_results: Raw Tavily response, as returned by
            `search_web_tavily` -- a dict with hits under "results", or an
            error string.
        web_search_results_dict: Output dict; mutated in place, appending
            standardized result entries to its "results" list.

    Raises:
        ValueError: when `tavily_search_results` is an error string, or when
            a list element in results is not a dict.
    """
    if isinstance(tavily_search_results, str):
        raise ValueError(tavily_search_results)

    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []

    for i, result in enumerate((tavily_search_results or {}).get("results", [])):
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected Tavily result item at index {i}: expected an object")
        content = result.get("content", "")
        web_search_results_dict["results"].append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": content,
            "metadata": {
                "date_published": result.get("published_date") or result.get("publishedDate") or None,
                "author": None,
                "source": None,
                "language": None,
                "relevance_score": result.get("score", None),
                "snippet": content or None,
            },
        })


######################### Yandex Search #########################
#
# https://yandex.cloud/en/docs/search-api/operations/web-search
# https://yandex.cloud/en/docs/search-api/quickstart/
# https://yandex.cloud/en/docs/search-api/concepts/response
# https://github.com/yandex-cloud/cloudapi/blob/master/yandex/cloud/searchapi/v2/search_query.proto
def search_web_yandex(search_query: str, result_count: Optional[int] = None) -> dict:
    """Query Yandex Cloud Search API v2 (synchronous REST) and return raw JSON.

    The response wraps a base64-encoded XML document in "rawData"
    (proto: yandex/cloud/searchapi/v2/search_service.proto — WebSearchService.Search,
    POST /v2/web/search). Decoding and parsing live in parse_yandex_results so
    process_web_search_results' try/except is the single error seam.

    Args:
        search_query: The query string.
        result_count: Unused by the request (Yandex returns its default page,
            ~10 groups; the agent layer trims client-side) — accepted for
            dispatch-signature parity.

    Returns:
        dict: Raw response JSON ({"rawData": "<base64 XML>"}).

    Raises:
        ValueError: when the API key or folder id is not configured.
        requests.exceptions.HTTPError: on non-2xx responses.
    """
    yandex_api_key = loaded_config_data["search_engines"].get("yandex_search_api_key", "")
    if not yandex_api_key:
        raise ValueError("Please provide a valid Yandex Search API key ([SearchEngines] yandex_search_api_key)")
    folder_id = loaded_config_data["search_engines"].get("yandex_search_folder_id", "")
    if not folder_id:
        raise ValueError("Please provide the Yandex Cloud folder id ([SearchEngines] yandex_search_folder_id)")
    headers = {"Authorization": f"Api-Key {yandex_api_key}", "Content-Type": "application/json"}
    payload = {
        "query": {"searchType": "SEARCH_TYPE_COM", "queryText": search_query},
        "folderId": folder_id,
        "responseFormat": "FORMAT_XML",
    }
    response = requests.post(
        "https://searchapi.api.cloud.yandex.net/v2/web/search", headers=headers, json=payload, timeout=SEARCH_BACKEND_TIMEOUT_S
    )
    response.raise_for_status()
    return response.json()


def parse_yandex_results(yandex_search_results: dict, web_search_results_dict: dict) -> None:
    """Decode rawData base64 XML and parse docs into the standardized shape.

    Raises on an in-XML <error> element (quota/auth/malformed-query arrive
    inside HTTP 200): a quota error must never render as "No results found"
    for a query that was never searched (spec 2026-08-06 §2). The raise is
    caught by process_web_search_results and lands in processing_error.

    Args:
        yandex_search_results: Raw Yandex response JSON, as returned by
            `search_web_yandex` ({"rawData": "<base64 XML>"}).
        web_search_results_dict: Output dict; mutated in place, appending
            standardized result entries to its "results" list.

    Raises:
        ValueError: when rawData is missing, or the decoded XML contains
            an <error> element (quota/auth/malformed-query).
    """
    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []
    raw_b64 = (yandex_search_results or {}).get("rawData", "")
    if not raw_b64:
        raise ValueError("Yandex response had no rawData field")
    xml_bytes = base64.b64decode(raw_b64)
    root = _yandex_ET.fromstring(xml_bytes)
    error_el = root.find(".//error")
    if error_el is not None:
        code = error_el.get("code", "?")
        text = "".join(error_el.itertext()).strip()
        raise ValueError(f"Yandex API error (code {code}): {text}")
    for doc in root.findall(".//group/doc"):
        url_el = doc.find("url")
        title_el = doc.find("title")
        passages = [
            " ".join("".join(p.itertext()).split())
            for p in doc.findall(".//passage")
        ]
        content = " ".join(passages).strip()
        web_search_results_dict["results"].append({
            "title": "".join(title_el.itertext()).strip() if title_el is not None else "",
            "url": url_el.text.strip() if url_el is not None and url_el.text else "",
            "content": content,
            "metadata": {
                "date_published": None,
                "author": None,
                "source": None,
                "language": None,
                "relevance_score": None,
                "snippet": content or None,
            },
        })


#
# End of WebSearch_APIs.py
#######################################################################################################################
