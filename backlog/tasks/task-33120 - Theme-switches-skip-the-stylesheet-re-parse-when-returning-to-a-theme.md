---
id: TASK-33120
title: Theme switches skip the stylesheet re-parse when returning to a theme
status: Done
assignee: []
created_date: '2026-09-28 00:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-33075 profiling (2026-09-27, 211x44, TLDW_TEST_CSS_CACHE=0): a theme switch costs ~0.9-1.2 s, ~430 ms of it Textual re-parsing the ~830 KB stylesheet bundle, identical across screens. A parse cache keyed on the theme variables on the app's TieAwareStylesheet would make returning to a theme (Try then Revert, cycling rows) skip that cost; a first use would still pay it. Pytest's Tests/UI/css_cache.py hides this cost — probes need TLDW_TEST_CSS_CACHE=0.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Returning to a previously used theme does not re-parse the stylesheet
- [x] #2 The first use of a theme still renders correctly
- [x] #3 A measurement shows the saved time
<!-- AC:END -->

## Implementation Plan

1. `TieAwareStylesheet` keeps a small LRU (8) of per-variables parse caches: each is an upstream-shaped `LRUCache(64)` keyed on (css content, location, is_default, tie_breaker, scope) — Textual's own key — and selected by a fingerprint of the variables dict. `set_variables` swaps to the matching cache instead of clearing it; `reparse` hands that cache to its fresh upstream `Stylesheet` (upstream builds a cache-less one every time). CSS source identity = content, so an edited/rebuilt bundle can never hit a stale parse.
2. Tests (Tests/UI/test_tie_aware_stylesheet_parse_cache.py), with pytest's global parse cache bypassed: return-to-theme does not parse; changed variable parses; changed CSS source parses; LRU eviction; resolved styles after a cached switch equal a fresh parse in a real app.
3. Measure first vs return switch at 211x44 on Settings ▸ Theme with TLDW_TEST_CSS_CACHE=0, before/after; measure per-entry memory.

## Implementation Notes

**What:** `TieAwareStylesheet` (the app's stylesheet) keeps an LRU of per-variables parse caches (`THEME_PARSE_CACHE_SIZE = 4`). `set_variables` detaches the outgoing cache before upstream clears it, then selects/creates the cache for the new variables; `reparse` mirrors textual 8.2.8's body with one added line handing that cache to the fresh upstream `Stylesheet` it parses into. No Textual source patched, no class-level monkeypatch; the upstream reparse body is pinned by a source hash so a Textual upgrade fails loudly.

**Cache key / staleness:** inner key is Textual's own `(css content, location, is_default, tie_breaker, scope)` — sources are identified by *content*, so a regenerated bundle, an edited file or a lowered tie-breaker is a miss, never a stale hit; outer key is the full sorted variables dict (the only other input to `parse`), so an edited theme under the same name misses. RuleSets are reused exactly as upstream already reuses them across `parse()` calls on one instance. Known gap unchanged: dev-mode CSS hot-reload swaps in a plain `Stylesheet` (no cache, correct).

**Measured** (Settings ▸ Theme, `run_test(size=(211, 44))`, private profile, `TLDW_TEST_CSS_CACHE=0`, interleaved A/B in one process, medians of 4 themes per pass, host load avg 18-34 so absolutes run high):

| | first use: reparse / refresh_css / wall | return: reparse / refresh_css / wall |
|---|---|---|
| before (upstream) | 364-659 / 571-1083 / 708-1288 ms | 313-336 / 561-588 / 808-918 ms |
| after | 407-496 / 601-820 / 751-1102 ms | **1 / 224-370 / 469-613 ms** |

Return-switch reparse ~330 ms -> ~1 ms; wall ~0.85 s -> ~0.55 s. First use unchanged (still pays the parse). Memory: one theme entry retains ~15.7 MB (tracemalloc; 4,734 rules, 27 sources), hence cap 4 (~63 MB) rather than 8 (~125 MB).

**Tests:** `Tests/Performance/test_tie_aware_stylesheet_parse_cache.py` (return hits / changed variable misses / changed CSS source misses / LRU eviction / upstream-reparse pin / real-bundle app: resolved styles of Static, Button, Input after a cached return equal a plain-Stylesheet fresh parse, 0 parses vs >0). Placed in Tests/Performance because Tests/UI's autouse fixture imports the app and trips the ADR-126 gate in a clean worktree. The global test cache is stepped around via a new `__wrapped__` on `Tests/UI/css_cache.py`'s wrapper.

Verified 2026-09-27 on fix/theme-tail-lane-b (from dev 89dd84943a): 210 passed across the parse-cache, css-fastpath, boot-css-budget, tie-aware counter, consolidated-css, theme picker/screen and theme_catalog suites; preflight rc=0.

Files: `tldw_chatbook/css/tie_aware_stylesheet.py`, `Tests/Performance/test_tie_aware_stylesheet_parse_cache.py`, `Tests/UI/css_cache.py`, `Docs/User_Guide/settings.md`, `backlog/docs/lessons-testing-evidence.md`.
