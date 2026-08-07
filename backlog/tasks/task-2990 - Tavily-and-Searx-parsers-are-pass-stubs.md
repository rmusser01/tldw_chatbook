---
id: TASK-2990
title: Tavily and Searx parsers are pass stubs
status: To Do
assignee: []
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

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 parse_tavily_results implements the standardized result shape (title/url/content/metadata) used by every other parser in the file, from Tavily's real API response shape
- [ ] #2 parse_searx_results implements the same standardized result shape from SearX/SearXNG's real JSON API response shape
- [ ] #3 Tests/Web_Scraping/test_search_backends.py's _KNOWN_BROKEN_PARSERS allowlist is emptied and test_agent_enum_engines_all_dispatchable passes both engines through the normal (>=1 result) assertion branch
<!-- AC:END -->
