---
id: TASK-3260
title: Deep-search scrape path ignores robots.txt
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 21:15'
updated_date: '2026-08-09 02:18'
labels:
  - web-tools
dependencies:
  - TASK-2833
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-2833 added robots.txt enforcement for web_fetch and web_crawl (per-domain fetch+cache, Disallow honored, `[webfetch] respect_robots_txt` toggle), but its design doc recorded a non-goal: web_deep_search's scrape path (`Web_Scraping.WebSearch_APIs`'s per-result relevance/summarization scraping, invoked from `analyze_and_aggregate`) was left out of scope.

This leaves an inconsistency: a path that web_fetch/web_crawl now refuse to fetch (per that host's robots.txt) can still be scraped by web_deep_search, because its scrape path never consults robots.txt at all. The codebase already has precedent for closing exactly this kind of gap — `is_public_http_url`'s SSRF check was bolted onto `scrape_article` after the same shape of inconsistency was noticed between the SSRF-guarded tools and the deep-search scrape path — so the fix shape here is the same: thread a robots.txt consult into the scrape path rather than carving out a permanent architectural exemption.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 web_deep_search's scrape path (the per-result fetch inside `analyze_and_aggregate`/its scraping helper) is refused for a URL disallowed by that host's robots.txt, under the same `[webfetch] respect_robots_txt` toggle task-2833 introduced (default on, fail-open on an unreachable/unparsable robots.txt)
- [x] #2 A robots-disallowed result is skipped (not fatal to the overall deep-search run), mirroring web_crawl's skip-and-count behavior rather than web_fetch's hard refusal
- [x] #3 Fixture-based tests (MockTransport, no live network) cover: a disallowed scrape URL is skipped, an allowed one proceeds, and the toggle-off path makes no robots.txt fetch at all
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Threaded a robots.txt consult into web_deep_search's scrape path (search_result_relevance,
called from analyze_and_aggregate), reusing task-2833's existing robots infrastructure in
Tools/web_tool_impls.py rather than building a parallel mechanism.

New public helper `robots_allows_for_scrape(url) -> bool` in web_tool_impls.py: builds a
short-lived httpx.Client on the module `_transport` seam with trust_env=False (mirrors
_new_web_fetch_client -- an honored HTTP(S)_PROXY would otherwise do its own DNS/connect,
defeating validate_outbound_url's SSRF check on the robots.txt URL) and a bounded timeout,
then delegates to `_robots_allows` with a new, honest UA constant `_DEEP_SEARCH_ROBOTS_UA =
"tldw-chatbook-deep-search/1.0"` (distinct from scrape_article's Chrome-masquerading UA,
pre-existing FIXME, out of scope) -- this shares the module's robots cache (30-min TTL,
negative caching, redirect-following, IPv6 handling) with web_fetch/web_crawl for free.

Toggle plumbing (ruling 1): web_deep_search places `_webfetch_settings()["respect_robots_txt"]`
into `search_params` (the same pydantic-safe channel the timeouts already use).
analyze_and_aggregate reads it (`bool(search_params.get("respect_robots_txt", False))`,
default False when absent) and passes it to search_result_relevance as a new
`respect_robots_txt` kwarg (default False) -- the dead-wired research-service caller that
never sets this key keeps today's no-robots-check behavior; web_deep_search (the tool) is
the only caller that passes the real, configured setting.

Check site (ruling 2): in search_result_relevance, between the existing SSRF guard
(is_public_http_url) and the scrape_article call, gated on `respect_robots_txt`. Same
guard-class offload discipline as the SSRF check: routed through
asyncio.wait_for(loop.run_in_executor(_get_dns_guard_executor(), robots_allows_for_scrape,
url), timeout=scrape_timeout_s). `robots_allows_for_scrape` is imported FUNCTION-LOCALLY
inside the gated block (no module-level import cycle -- web_tool_impls already imports
WebSearch_APIs function-locally in the other direction).

Disallowed -> fallback, not discard (ruling 4): a disallowed URL takes the exact same path
as an SSRF refusal -- skip scrape_article, keep the result via its existing
_build_result_fallback_content-derived snippet/title/url content. A logger.debug names the
host only (via urlparse(...).hostname), never the query.

Fail-open (ruling 5): a robots-check error or timeout (caught via `except Exception`, with
CancelledError re-raised first) sets robots_ok back to True and proceeds to scrape --
deliberately the OPPOSITE of the SSRF guard immediately above, whose own timeout/refusal
still refuses. Matches _fetch_robots_parser's existing fail-open for web_fetch/web_crawl.

Tests: Tests/Web_Scraping/test_deep_search_pipeline.py gained three pipeline-level tests --
disallowed-skips/allowed-proceeds (with all THREE required fakes: is_public_http_url ->
True, web_tool_impls._transport MockTransport, and socket.getaddrinfo -> a public IP, since
_fetch_robots_parser's own _validate_hop does a separate DNS check that fails open and would
otherwise silently bypass the MockTransport), toggle-absent-makes-zero-robots-fetches
(parity pin for the research-service caller), and robots-unreachable-fails-open.
Tests/Tools/test_web_deep_search.py gained two tool-level tests proving web_deep_search
places the real (and non-hardcoded) respect_robots_txt value into search_params.
Tests/Tools/test_web_tool_impls.py gained five direct unit tests for
robots_allows_for_scrape itself (disallowed/allowed/own-UA-group/unreachable-fail-open/
cache-sharing-with-web_fetch), reusing the existing fetch_env MockTransport+DNS+clock
fixture.

Mutation checks: (1) bypassed the robots consult in search_result_relevance (forced the
gated block to `if False:`) -- confirmed the disallowed-skip test goes red, then restored
via Edit. (2) dropped the respect_robots_txt entry from web_deep_search's search_params dict
-- confirmed both tool-level plumbing tests go red, then restored via Edit.

Files: tldw_chatbook/Tools/web_tool_impls.py, tldw_chatbook/Web_Scraping/WebSearch_APIs.py,
Tests/Web_Scraping/test_deep_search_pipeline.py, Tests/Tools/test_web_deep_search.py,
Tests/Tools/test_web_tool_impls.py. No deviations from the design doc's five rulings.
<!-- SECTION:NOTES:END -->
