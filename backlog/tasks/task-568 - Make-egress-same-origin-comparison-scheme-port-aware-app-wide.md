---
id: TASK-568
title: Make egress same-origin comparison scheme/port-aware app-wide
status: Done
assignee: []
created_date: '2026-07-25 06:00'
updated_date: '2026-07-25 13:58'
labels:
  - security
  - egress
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Utils/egress.py`'s credential-stripping decision on redirect hops (`guarded_fetch_httpx` → `_hop_headers`) compares origins by HOSTNAME only (`host_of`), ignoring scheme and port. A same-host HTTPS→HTTP downgrade redirect, or a redirect to a different port on the same host, therefore keeps `Authorization`/`Cookie` headers. The Image_Generation redirect loops (PR #862, task-498) deliberately reuse the same primitive to stay policy-consistent, so they inherit the same weakness — flagged by Qodo on #862 and declined there precisely because the right fix is central, not a local fork. Note the app is internally inconsistent: SwarmUI's same-origin image-URL gate already compares scheme+host+port.

Upgrading `host_of`-based same-origin decisions to scheme+host+port must be done centrally so every consumer (egress helpers, Web_Scraping, Subscriptions, Image_Generation) changes together, with the local-backend trust flows (trusted_origins) re-verified — `origin_set` already produces scheme://host:port origins, so the comparison upgrade must not break trusted-origin matching for operator-configured base_urls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Same-origin decisions that gate credential forwarding on redirect hops compare scheme + host + port (with default-port normalization: 443 for https, 80 for http) in `Utils/egress.py` and every consumer that reuses the primitive (including `Image_Generation/http_client.py` and `adapters/image_format_utils.py`).
- [x] #2 A same-host HTTPS→HTTP downgrade redirect and a same-host different-port redirect both strip credentials; a genuinely same-origin redirect (same scheme, host, effective port) still carries them — each pinned by test.
- [x] #3 Operator-configured local backends (trusted_origins flows) keep working, including through their own same-origin redirects — regression-tested.
- [x] #4 All existing egress/SSRF suites pass (`Tests/Utils/test_egress.py`, `Tests/Image_Generation/`, subscriptions egress wiring).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Utils/egress.py in full: understand host_of/origin_set (host-only, feeds trusted_origins SSRF policy) vs the same_origin/host_of equality used to gate credential stripping in the 4 guarded_fetch_* redirect loops.
2. Sweep the whole repo for host_of/_hop_headers/same_origin/origin_set usages and classify each: blocking-policy (trusted_origins/log-label, stays host-based) vs credential-forwarding decision (upgrade).
3. Add origin_of(url) -> (scheme, host, effective_port)|None and same_origin(url_a, url_b) -> bool to Utils/egress.py, with default-port normalization (443/80) and conservative (False) behavior on any parse ambiguity. Do not touch host_of/origin_set.
4. Swap the same-origin computation in guarded_fetch_httpx, guarded_fetch_httpx_async, guarded_fetch_requests, guarded_fetch_aiohttp from host_of equality to same_origin().
5. Update Image_Generation/http_client.py (fetch_json) and adapters/image_format_utils.py (fetch_image_bytes) to import and call egress.same_origin instead of reimplementing host_of equality.
6. Refactor SwarmUIAdapter's local _same_origin/_origin_port (already scheme+host+port) to delegate to the new central egress.same_origin, removing the duplicate implementation.
7. TDD: add egress-level tests for origin_of/same_origin semantics and the scheme-downgrade/port-change/default-port-equivalence matrix on guarded_fetch_httpx (+1 async, +requests, +aiohttp); add matching tests at the Image_Generation layer for fetch_json/fetch_image_bytes; add SwarmUIAdapter._resolve_image_url regression tests. Verify red via git-stash revert + a standalone repro before restoring the fix.
8. Run full Utils/test_egress.py, Image_Generation/, Subscriptions/, Web_Scraping/, Local_Ingestion suites plus ruff + app import; confirm zero existing tests needed modification (trusted_origins flows untouched).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Utils/egress.py::origin_of(url) -> (scheme, host, effective_port)|None and same_origin(url_a, url_b) -> bool (default-port normalization 443/80; conservative False on any parse ambiguity). Swapped the credential-forwarding same-origin computation in all 4 guarded_fetch_* redirect loops (httpx sync/async, requests, aiohttp) from host_of equality to same_origin(). Updated Image_Generation/http_client.py::fetch_json and adapters/image_format_utils.py::fetch_image_bytes to call egress.same_origin instead of reimplementing host_of equality. Refactored SwarmUIAdapter's local _same_origin/_origin_port (already scheme+host+port) to delegate to the new central primitive, removing a duplicate implementation.

Swept the repo for every host_of/_hop_headers/same_origin/origin_set hit (19 non-definition call sites) and classified each: 7 call sites across 4 files were credential-forwarding decisions and got upgraded; the remaining 15 (all in Subscriptions/Web_Scraping/Local_Ingestion/Media plus 2 monitoring_engine.py log-label uses) are the SSRF trusted_origins blocking-policy, which is intentionally hostname-only and was left untouched -- origin_set/host_of are byte-for-byte unchanged.

TDD: verified red by git-stash-reverting the 4 source files and confirming both an ImportError on origin_of/same_origin and, via a standalone repro script against guarded_fetch_httpx, that a same-host HTTPS->HTTP downgrade hop leaked Authorization/Cookie pre-fix. Restored the fix and added 26 new tests: origin_of/same_origin unit tests (12), scheme-downgrade/port-change/default-port-equivalence matrices on guarded_fetch_httpx (3) + async (1) + requests (2) + aiohttp (1) in Tests/Utils/test_egress.py; matching fetch_json/fetch_image_bytes tests (2+2) in Tests/Image_Generation/; 3 direct SwarmUIAdapter._resolve_image_url regression tests. Zero pre-existing tests needed modification -- trusted_origins/local-backend-through-its-own-redirect flows all pass unmodified, proving compatibility.

Full suite green: Tests/Utils/test_egress.py (67), Tests/Image_Generation/ (83 passed, 6 skipped), Tests/Subscriptions/ (all), Tests/Web_Scraping/ (79), Tests/Local_Ingestion/test_web_article_ingestion.py -- all unmodified pre-existing tests pass. ruff clean on all 8 touched files (the 9 pre-existing E402/F821 findings in egress.py/test_egress.py confirmed identical before/after via git-stash diff). python -c "import tldw_chatbook.app" clean.

Full audit table, semantics writeup, and self-review in .superpowers/sdd/task-568-report.md.

Files: tldw_chatbook/Utils/egress.py, tldw_chatbook/Image_Generation/http_client.py, tldw_chatbook/Image_Generation/adapters/image_format_utils.py, tldw_chatbook/Image_Generation/adapters/swarmui_adapter.py, Tests/Utils/test_egress.py, Tests/Image_Generation/test_http_client.py, Tests/Image_Generation/test_image_format_utils.py, Tests/Image_Generation/test_swarmui_adapter.py.
<!-- SECTION:NOTES:END -->
