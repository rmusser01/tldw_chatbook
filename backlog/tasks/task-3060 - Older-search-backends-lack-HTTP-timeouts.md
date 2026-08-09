---
id: TASK-3060
title: Older search backends lack HTTP timeouts
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 14:10'
updated_date: '2026-08-09 02:18'
labels:
  - web-tools
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six older web-search backends (google, brave, duckduckgo, kagi, tavily, searx) plus the baidu stub issue requests.post/requests.get calls with no timeout parameter (bing already carries its own timeout=10; correction 2026-08-07 — the original filing wrongly listed bing among the unbounded ones). An unresponsive provider API hangs perform_websearch (and by extension generate_and_search / the deep-search pipeline) indefinitely, with no way for a caller to bound worst-case latency. Task-1356's phase-1 hardening added timeout=30 to the newer serper/exa/yandex backends (task-1355) but deliberately left these seven+baidu out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every backend HTTP call (google, brave, duckduckgo, kagi, tavily, searx) carries an explicit timeout (amended 2026-08-08: bing already carries timeout=10 — earlier correction; baidu is a bare `pass` stub with NO HTTP call, dropped — spec review caught the false premise)
- [x] #2 A simulated hang/unresponsive provider surfaces as a bounded-time error instead of blocking indefinitely
- [x] #3 Existing request-shape tests for each backend assert the timeout value
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added timeout=30 to every requests.get/post call site in the six engine functions
(search_web_google, search_web_brave, search_web_duckduckgo, search_web_kagi,
search_web_tavily, search_web_searx) in Web_Scraping/WebSearch_APIs.py -- each function
has exactly one requests. call site (duckduckgo's is inside a 5-iteration bootstrap/
pagination loop, still a single call site). bing (already timeout=10) and baidu (bare
`pass` stub, no HTTP call -- AC #1 amended to drop it) were left untouched, per scope.

searx breaks the standard `requests.get/post` idiom (Important 5): it goes through
searx_create_session() -> requests.Session() -> session.get(...), so `timeout=30` was
added to the session.get() call, not a bare requests call.

Tests: extended Tests/Web_Scraping/test_search_backends.py with one request-shape test
per engine asserting `timeout == 30` on every captured call (google, brave, duckduckgo,
kagi, tavily, searx). duckduckgo's fake returns "No  results." to short-circuit past lxml
parsing while still exercising the real requests.post call. searx's test patches
searx_create_session directly (its own _FakeSearxSession stub) rather than extending the
shared _FakeRequests fixture, and mocks random.uniform (Minor 8) so the real
time.sleep(random.uniform(2, 5)) pacing delay doesn't cost 2-5s per test run (precedent:
test_deep_search_pipeline.py:667). Added AC #2's bounded-time-error test on google: a
faked requests.Timeout is asserted to surface via google's EXISTING error contract
(catches RequestException, logs, re-raises unchanged) -- not a new/invented contract.

Mutation check: removed timeout=30 from search_web_google, confirmed
test_google_request_carries_timeout goes red (KeyError: 'timeout'), then restored via Edit.

Files: tldw_chatbook/Web_Scraping/WebSearch_APIs.py, Tests/Web_Scraping/test_search_backends.py.
No deviations from the design doc beyond the AC #1 baidu amendment already recorded in the
task file before this task started.
<!-- SECTION:NOTES:END -->
