---
id: TASK-2990
title: Tavily and Searx parsers are pass stubs
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 05:19'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Port tldw_server2's complete parsers (tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:2816/3078) with chatbook adaptations: tavily = dict-with-results, score->relevance_score, error-string input raises; searx = chatbook's real return shape is a JSON STRING of a [{title,link,snippet,publishedDate}] list (or an error-dict string) -> parser loads the string, raises on the error dict (honest seam), standardizes the list. Empty _KNOWN_BROKEN_PARSERS; correct the parity sample payloads to the real shapes; tests with realistic fixtures incl. both error paths.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 parse_tavily_results implements the standardized result shape (title/url/content/metadata) used by every other parser in the file, from Tavily's real API response shape
- [ ] #2 parse_searx_results implements the same standardized result shape from SearX/SearXNG's real JSON API response shape
- [ ] #3 Tests/Web_Scraping/test_search_backends.py's _KNOWN_BROKEN_PARSERS allowlist is emptied and test_agent_enum_engines_all_dispatchable passes both engines through the normal (>=1 result) assertion branch
<!-- AC:END -->
