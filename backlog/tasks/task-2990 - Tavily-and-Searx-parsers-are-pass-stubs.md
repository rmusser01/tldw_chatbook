---
id: TASK-2990
title: Tavily and Searx parsers are pass stubs
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 05:19'
updated_date: '2026-08-07 13:09'
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

parse_searx_results: chatbook's search_web_searx ALWAYS returns a JSON-encoded STRING -- for its success case too, not just errors -- (json.dumps of a [{title,link,snippet,publishedDate}] list, or json.dumps({"error": ...})), unlike every other backend in the file, which returns a dict directly. The parser json.loads a string first, raises ValueError on a decoded error dict or on any decoded non-list shape (a bare dict without "error", or a JSON scalar), and standardizes a decoded list (url<-link or url, content<-snippet or content [restored the port reference's OR-fallback pair so a pre-parsed raw-SearXNG-shaped item still populates url/content], metadata.date_published<-publishedDate, relevance_score None -- Searx doesn't expose one).

Necessary but unplanned fix, corrected justification (review round 2 -- the original wording here was wrong): process_web_search_results()'s top-level `if not isinstance(search_results, dict): raise TypeError(...)` guard ran BEFORE the try/except that turns parser exceptions into processing_error. It does NOT need widening to avoid "silence" via perform_websearch -- perform_websearch already wraps its whole dispatch (including the process_web_search_results call) in its own try/except that turns ANY exception into `{"processing_error": f"Error performing web search: {e}"}`, so an uncaught TypeError there was never actually silent. The real, stronger reason the guard had to change: search_web_searx's SUCCESS payload is ALSO a plain string (json.dumps of the hit list) -- not just its error payload. Without widening, the type guard would reject EVERY searx call unconditionally, success or failure, before ever reaching parse_searx_results -- so even a fully working SearX response, with real hits, would be reported as "search_results must be a dictionary" instead of those hits. That is a categorically different defect from message fidelity: the parser this task exists to fix could never run for searx at all. For tavily specifically, only the error path is a string (its success path was already a dict, and was already reaching the old pass-stub parser fine), so the practical effect there of NOT widening was narrower: perform_websearch's outer catch would still surface an error, just the generic wrapped text instead of the parser's specific request-error text. Widened the guard to `isinstance(dict) OR (isinstance(str) AND engine in ("tavily", "searx"))` -- scoped to just these two engines (review round 2, Important 1): the local backends are the only ones whose real return value can be a non-dict, so this doesn't reopen the class of bug this task exists to close for any other engine (e.g. a stray string reaching parse_brave_results, whose `"query" in raw_results`-style membership checks against a str run silently and could produce zero results with no error). Made the metadata-echo block use a `_meta = search_results if isinstance(search_results, dict) else {}` fallback so a string payload doesn't AttributeError on `.get()`. No other engine's behavior changes (they only ever pass dicts, and the guard is unchanged for them).

Tests (Tests/Web_Scraping/test_search_backends.py): new Tavily/Searx sections mirroring the existing Serper/Exa/Yandex pattern -- realistic-fixture parse-shape tests (incl. relevance_score for tavily), one combined "raises ValueError directly + surfaces as processing_error through process_web_search_results" seam test per engine, searx JSON-string-loading + already-parsed-list + raw-SearXNG-shape-key-fallback + empty-list + unparseable-JSON + non-error-dict-raises + JSON-scalar-raises tests, and a tavily end-to-end HTTP-mocked round trip (skipped the equivalent for searx: search_web_searx's real HTTP path calls time.sleep(random 2-5s) and builds its own requests.Session, making a fast unit-level HTTP round trip impractical and out of scope for this task). Two new type-guard-scoping tests (review round 2): process_web_search_results("some error text", "brave") still raises TypeError (not silently reachable), and process_web_search_results(None, "tavily") still raises TypeError (the str allowance is exact-type, not "anything non-dict"). _KNOWN_BROKEN_PARSERS emptied to set() (comment updated); the now-always-false allowlist branch in test_agent_enum_engines_all_dispatchable removed rather than left inert, since it was trivially removable; _ENGINE_SAMPLE_PAYLOADS' tavily/searx entries now reuse the new dedicated fixture constants (_TAVILY_PAYLOAD: a dict, matching search_web_tavily's real success shape; _SEARX_PAYLOAD: a json.dumps'd list, matching search_web_searx's real return shape) instead of the old generic/wrong-shaped placeholders.

RED/GREEN: confirmed by temporarily swapping in the pre-change WebSearch_APIs.py (via file copy, not git) -- 10 of the new/strengthened tests failed (AssertionError / zero results) against the `pass` stubs, then passed once the real file was restored. Full suite after review round 2: Tests/Web_Scraping/test_search_backends.py 38 passed, 3 skipped (live-gated); Tests/Tools/test_web_search_tool.py + Tests/Agents/test_local_tool_provider.py 93 passed (no regressions in the callers). Tests/ --collect-only (pre-round-2): 31882 collected, no import errors.

Known residual, recorded not fixed (review round 2, no code change): search_web_searx's JSON-content-type branch (`if "application/json" in content_type: search_data = response.json()`, then `for result in search_data: ... result.get("title")`) would AttributeError if a Searx instance ever actually returned JSON, since search_data would then be a dict and the loop would iterate its string KEYS, not hit dicts. Latent, not live: search_web_searx's own request never asks for JSON (no format=json param), so real instances return HTML and take the BeautifulSoup branch instead. Out of scope for this task, which is about the PARSER (parse_searx_results), not the fetch function.

Modified files: tldw_chatbook/Web_Scraping/WebSearch_APIs.py, Tests/Web_Scraping/test_search_backends.py.
<!-- SECTION:NOTES:END -->
