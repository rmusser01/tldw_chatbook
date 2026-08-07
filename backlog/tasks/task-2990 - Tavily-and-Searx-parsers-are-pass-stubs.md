---
id: TASK-2990
title: Tavily and Searx parsers are pass stubs
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 05:19'
updated_date: '2026-08-07 12:56'
labels:
  - web-tools
  - tech-debt
dependencies:
  - TASK-1355
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
parse_tavily_results and parse_searx_results in Web_Scraping/WebSearch_APIs.py are empty `pass` stubs. Both engines are advertised as working (SEARCH_ENGINES in Tools/web_tool_impls.py), so a real API response from either silently parses to zero results and is rendered to the user as "No results found" for a query that was actually answered -- an honesty gap, not just a missing feature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 parse_tavily_results implements the standardized result shape (title/url/content/metadata) used by every other parser in the file, from Tavily's real API response shape
- [x] #2 parse_searx_results implements the same standardized result shape from SearX/SearXNG's real JSON API response shape
- [x] #3 Tests/Web_Scraping/test_search_backends.py's _KNOWN_BROKEN_PARSERS allowlist is emptied and test_agent_enum_engines_all_dispatchable passes both engines through the normal (>=1 result) assertion branch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Port tldw_server2's complete parsers (tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:2816/3078) with chatbook adaptations: tavily = dict-with-results, score->relevance_score, error-string input raises; searx = chatbook's real return shape is a JSON STRING of a [{title,link,snippet,publishedDate}] list (or an error-dict string) -> parser loads the string, raises on the error dict (honest seam), standardizes the list. Empty _KNOWN_BROKEN_PARSERS; correct the parity sample payloads to the real shapes; tests with realistic fixtures incl. both error paths.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported tldw_server2's tavily/searx parsers (app/core/Web_Scraping/WebSearch_APIs.py:2816 searx, :3078 tavily) with chatbook-specific adaptations, since the local backends' return shapes differ from the server's.

parse_tavily_results: dict input (search_web_tavily's response.json()) -> standardized items (title/url/content, metadata.snippet<-content, metadata.relevance_score<-score [a real 0-1 relevance score, correctly directioned unlike serper's rank], metadata.date_published<-published_date/publishedDate). String input (search_web_tavily's own request-failure return value, e.g. "There was an error searching for content. ...") re-raises as ValueError so the text survives as processing_error instead of silently producing zero results.

parse_searx_results: chatbook's search_web_searx ALWAYS returns a JSON-encoded STRING (json.dumps of a [{title,link,snippet,publishedDate}] list, or json.dumps({"error": ...})) -- unlike every other backend in the file, which returns a dict directly. The parser json.loads a string first (also tolerates an already-parsed list/dict defensively), raises ValueError on a decoded error dict or unparseable JSON, and standardizes a decoded list (url<-link, content<-snippet, metadata.date_published<-publishedDate, relevance_score None -- Searx doesn't expose one).

Necessary but unplanned fix: process_web_search_results()'s top-level `if not isinstance(search_results, dict): raise TypeError(...)` guard ran BEFORE the try/except that turns parser exceptions into processing_error, and before the dict.get() calls that build the request-echo metadata. Since tavily's error path and searx's entire return value are strings, that guard would have raised an uncaught TypeError ahead of ever reaching the new parsers -- silently defeating both honest-ValueError seams in the real perform_websearch pipeline (not just in direct/test calls). Widened the guard to accept (dict, str), and made the metadata-echo block use a `_meta = search_results if isinstance(search_results, dict) else {}` fallback so a string payload doesn't AttributeError on `.get()`. No other engine's behavior changes (they only ever pass dicts).

Tests (Tests/Web_Scraping/test_search_backends.py): new Tavily/Searx sections mirroring the existing Serper/Exa/Yandex pattern -- realistic-fixture parse-shape tests (incl. relevance_score for tavily), one combined "raises ValueError directly + surfaces as processing_error through process_web_search_results" seam test per engine, searx JSON-string-loading + already-parsed-list + empty-list + unparseable-JSON tests, and a tavily end-to-end HTTP-mocked round trip (skipped the equivalent for searx: search_web_searx's real HTTP path calls time.sleep(random 2-5s) and builds its own requests.Session, making a fast unit-level HTTP round trip impractical and out of scope for this task). _KNOWN_BROKEN_PARSERS emptied to set() (comment updated); the now-always-false allowlist branch in test_agent_enum_engines_all_dispatchable removed rather than left inert, since it was trivially removable; _ENGINE_SAMPLE_PAYLOADS' tavily/searx entries now reuse the new dedicated fixture constants (_TAVILY_PAYLOAD: a dict, matching search_web_tavily's real success shape; _SEARX_PAYLOAD: a json.dumps'd list, matching search_web_searx's real return shape) instead of the old generic/wrong-shaped placeholders.

RED/GREEN: confirmed by temporarily swapping in the pre-change WebSearch_APIs.py (via file copy, not git) -- 10 of the new/strengthened tests failed (AssertionError / zero results) against the `pass` stubs, then passed once the real file was restored. Full suite: Tests/Web_Scraping/test_search_backends.py 33 passed, 3 skipped (live-gated); Tests/Tools/test_web_search_tool.py + Tests/Agents/test_local_tool_provider.py 93 passed (no regressions in the callers). Tests/ --collect-only: 31882 collected, no import errors.

Modified files: tldw_chatbook/Web_Scraping/WebSearch_APIs.py, Tests/Web_Scraping/test_search_backends.py.
<!-- SECTION:NOTES:END -->
