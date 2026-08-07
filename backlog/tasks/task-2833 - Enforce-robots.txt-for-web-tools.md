---
id: TASK-2833
title: Enforce robots.txt for web tools
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-05 06:05'
updated_date: '2026-08-07 20:44'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
v1 documents but does not enforce robots.txt. Add per-domain robots fetch+cache and disallow-rule enforcement for tool-initiated fetches/crawls, behind a config toggle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 robots.txt fetched+cached per domain,Disallow honored for web_fetch/web_crawl,[webfetch] respect_robots_txt toggle; fixture-based tests
<!-- AC:END -->

## Implementation Notes

Implemented per `Docs/superpowers/specs/2026-08-07-robots-txt-enforcement-design.md` (rulings 1-6, injection sites, error contract, and test list are binding and were followed as written).

**Approach.** New module-level machinery in `tldw_chatbook/Tools/web_tool_impls.py`, beside the existing `_fetch_cache` idiom:
- `ROBOTS_MAX_BYTES` (64 KiB), `ROBOTS_CACHE_TTL_SECONDS` (1800.0), `ROBOTS_CACHE_MAX_ENTRIES` (128), `_robots_cache` (keyed by `scheme://netloc`, value `(expires_at, RobotFileParser | None)`, earliest-expiry eviction via `_robots_cache_put`, cleared by `_reset_state_for_tests`).
- `_webfetch_settings()` resolves `[webfetch] respect_robots_txt` via `get_cli_setting`, mirroring `_deep_search_settings()`'s `_bool` true-set coercion (a raw bool passes through; a string coerces via `"true"`/`"1"` membership) — read once per tool invocation, not per hop.
- `_robots_allows(client, url, user_agent)`: cache lookup by host; on miss, rate-limits via the existing `_enforce_rate_limit`, fetches `{scheme}://{netloc}/robots.txt` through `_fetch_once` under the dedicated byte cap, and parses with stdlib `RobotFileParser`. One broad `except Exception` makes any fetch/parse failure — including a body TRUNCATED at the byte cap — fail OPEN (cached `None` = no restrictions), matching ruling 2's "compat" semantics.
- `_robots_disallowed_message(url)` builds the exact `[robots-disallowed] {url} — {host}/robots.txt disallows this path for this tool's user agent; set [webfetch] respect_robots_txt = false to override` string.

**Injection sites** (exactly as named in the design doc): `web_fetch`'s cache-hit re-check and its redirect-loop hop, both right beside `_validate_hop`; `_crawl_fetch_page`'s hop loop (shared by web_crawl's BFS pages AND every sitemap fetch — root + children — for free), always checked against `_CRAWL_USER_AGENT`. `_crawl_fetch_page` and `_seed_from_sitemap` gained a `respect_robots: bool` parameter threaded from `web_crawl`'s single settings read.

**Behavior.** `web_fetch` raises the structured refusal per disallowed hop (cache-hit path re-checks the same way it re-checks SSRF policy). `web_crawl` gained a third exception-dispatch branch (`str(exc).startswith("[robots-disallowed]")` → `robots_disallowed` counter, alongside the existing `[ssrf]`→blocked / else→failed split) and `_format_crawl_result` gained an additive `robots_disallowed: int = 0` parameter rendered as `"; N robots-disallowed"` in the footer's parenthetical — additive-only (omitted when zero) so every pre-existing exact-string footer assertion (e.g. `test_format_blocks_footer_and_marker`'s `endswith`) stays byte-identical. A disallowed BFS seed or root sitemap flows through the existing unconditional wrap unchanged, producing the double-wrapped `[crawl-failed] start URL could not be fetched: [robots-disallowed] …` (or `sitemap could not be fetched:`); a disallowed child sitemap is caught by the existing broad `except (LocalToolError, _CrawlDeadline)` and counted in `children_skipped`, no new counter needed there.

**Existing-suite compatibility (design doc Critical 1).** With the shipped default ON, a robots pre-fetch would add transport calls and an extra `_enforce_rate_limit` hit that break several pre-existing exact-list/count/sleep assertions. Per the RULING, `fetch_env` (test_web_tool_impls.py) and `crawl_env` (test_web_crawl.py) now monkeypatch `_webfetch_settings` to `{"respect_robots_txt": False}` as part of their existing isolation duties — a TEST-FIXTURE default only; the shipped config default remains `true`. New robots tests opt back in via a local `_enable_robots(monkeypatch)` helper added to each file.

**Config + descriptions.** Added a commented `[webfetch]` block to `CONFIG_TOML_CONTENT` in `config.py` (documents scope, default, and fail-open semantics). Added a one-clause "Honors robots.txt (configurable)" mention to both `web_fetch` and `web_crawl`'s tool descriptions in `Agents/local_tool_provider.py`.

**Deviations from the design doc:** none — the mechanism, injection sites, error strings, config key, and test-fixture ruling were all implemented as specified.

**Tests.** 14 new tests in `Tests/Tools/test_web_tool_impls.py` (disallow/allow, specific-UA-vs-wildcard precedence with the first-match-wins fixture caveat, fail-open on missing/500/garbage/truncated robots.txt, cache hit/TTL-expiry/negative-cache, redirect-into-disallowed mid-chain, cache-hit re-check refusal, toggle-off, robots.txt's own rate-limiting) and 6 in `Tests/Tools/test_web_crawl.py` (disallowed page skipped+counted, disallowed child sitemap skipped, disallowed BFS seed and disallowed root sitemap both structured-refuse double-wrapped, crawl UA correctness, toggle-off). Full suite: `Tests/Tools/test_web_tool_impls.py Tests/Tools/test_web_crawl.py` 127 passed (107 pre-existing + 20 new); `Tests/Tools/ Tests/Web_Scraping/` 543 passed, 3 skipped (pre-existing live-search skips, `TLDW_LIVE_SEARCH_TESTS` gate, unrelated); full-repo `pytest --collect-only` shows 32223 tests collected, no new collection errors. Both spec-mandated mutation checks performed by hand (Edit-based, restored immediately after): (1) `_robots_allows` forced to always `return True` past the cache lookup → 8 disallow-path tests failed (`DID NOT RAISE`) as expected, then restored and reconfirmed green; (2) the `robots_disallowed` footer clause dropped from `_format_crawl_result` → `test_crawl_disallowed_page_skipped_and_counted` failed (footer silently omitted the count) as expected, then restored and reconfirmed green.

**Follow-up filed:** task-3260 (deep-search's scrape path does not honor robots.txt — recorded non-goal in the design doc's rulings; the same inconsistency `is_public_http_url`/`scrape_article` closed for SSRF).

**Modified files:** `tldw_chatbook/Tools/web_tool_impls.py`, `tldw_chatbook/config.py`, `tldw_chatbook/Agents/local_tool_provider.py`, `Tests/Tools/test_web_tool_impls.py`, `Tests/Tools/test_web_crawl.py`. **Added:** `backlog/tasks/task-3260 - Deep-search-scrape-path-ignores-robots-txt.md`.
