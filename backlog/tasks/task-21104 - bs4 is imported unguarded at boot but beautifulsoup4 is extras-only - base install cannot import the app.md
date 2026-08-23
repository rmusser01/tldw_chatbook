---
id: TASK-21104
title: >-
  bs4 is imported unguarded at boot but beautifulsoup4 is extras-only - base install cannot import the app
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
labels:
  - bug
  - packaging
  - startup
  - subscriptions
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21104).

`Subscriptions/monitoring_engine.py:37` does `from bs4 import BeautifulSoup` with no guard
(directly after a carefully guarded defusedxml import), and the module is eager at boot via
scheduler handlers <- app.py:429. `beautifulsoup4` exists only in
`[project.optional-dependencies]` (verified: four extras, none core). A base `pip install .`
therefore cannot `import tldw_chatbook.app` at all, and every configured install pays the
import at boot.

## Acceptance Criteria

- [x] Decision made and implemented: either beautifulsoup4 moves to core dependencies, or the import is guarded with graceful degradation of the affected monitors (consistent with `optional_deps.py`)
- [x] A test covers the import-closure of extras-gated packages so the next unguarded optional import is caught at PR time
- [x] A base (no-extras) install can import tldw_chatbook.app

## Implementation Plan

1. Census every `from bs4`/`import bs4` site in `tldw_chatbook/` and classify:
   guarded (try/except), lazy (function-local), or unguarded-top-level; identify
   which are reachable from `import tldw_chatbook.app`.
2. Baseline with teed evidence (`test-logs/`): (a) subprocess with bs4 blocked
   via a meta-path finder -> `import tldw_chatbook.app` fails today; (b) plain
   import -> bs4/soupsieve ARE in the boot closure today; (c) census which
   optional-registry modules are already unavoidably in the closure, to scope
   the guard list honestly.
3. Decision: guard-and-degrade (owner prefers durable; promoting bs4 to core
   would not fix the boot-cost half anyway). At the defect site
   (`Subscriptions/monitoring_engine.py:37`) move the import into
   `ContentExtractor.extract_text_from_html` (its only use) behind a
   try/except that raises an actionable ImportError naming the
   `[subscriptions]` extra — the per-check error handling in FeedMonitor/
   URLMonitor already records exceptions per subscription, so HTML monitors
   degrade at use time instead of crashing boot. Also make the module-level
   `_SELECTOR_PARSE_ERRORS = selector_parse_errors()` call lazy (it imports
   soupsieve at boot; the function is already lru_cached and its other caller
   already calls it at the except site).
4. Guard the three remaining unguarded top-level sites found by the census
   (`baseline_manager.py`, `scrapers/generic_scraper.py`,
   `scrapers/custom_scraper.py`) with the same lazy/actionable-error pattern so
   a base install can import any Subscriptions module.
5. Red-first tests in `Tests/Packaging/test_extras_import_closure.py`
   (subprocess-isolated, following `Tests/Performance/test_app_import_weight.py`):
   (a) app imports with bs4 blocked; (b) explicit-list import-closure guard for
   extras-gated packages (scoped by the step-2 census, exclusions documented);
   plus a use-time degrade test asserting the actionable error message.
6. Run: new tests, existing Subscriptions/monitoring tests, full
   `pytest Tests --collect-only -q` sweep. Tick ACs, notes, Done, commit.

## Implementation Notes

Took the guard-and-degrade branch (owner prefers durable over clever, and
promotion-to-core would not have fixed the boot-cost half — the lazy import
was needed either way).

**Census of `from bs4`/`import bs4` in `tldw_chatbook/` (21 sites).** Exactly
four were unguarded module-level imports, all in `Subscriptions/`:
`monitoring_engine.py:37` (the defect — the only one on the boot path, via
app.py -> Scheduling/scheduler/handlers -> watchlist_check_handler),
`baseline_manager.py:52` (NOT-WIRED module, zero importers),
`scrapers/generic_scraper.py:15` and `scrapers/custom_scraper.py:20` (the
scrapers package has no external importers). Every other site was already
guarded (try/except with a `BS4_AVAILABLE` flag, the Web_Scraping house
pattern) or function-local lazy.

**Fix.** `monitoring_engine` now resolves BeautifulSoup lazily inside
`ContentExtractor.extract_text_from_html` (its only use) via
`_require_beautifulsoup()`, which raises an ImportError naming the package
and `pip install tldw_chatbook[subscriptions]`; the monitors' existing
per-check exception handling records that against the subscription, so HTML
monitors degrade at use time with an actionable message instead of crashing
boot. The module-level `_SELECTOR_PARSE_ERRORS = selector_parse_errors()`
constant was also retired in favour of calling the (lru_cached) function at
the except site — it probed soupsieve at import, which would have kept
soupsieve on the boot path. The other three files got the same guarded
pattern (module-level try/except keeps the `BeautifulSoup` name defined for
annotations; every runtime constructor call goes through the raising helper;
`custom_scraper.validate_rules` reports missing bs4 as itself rather than as
"invalid CSS selector"). Deliberately did NOT use
`optional_deps.require_dependency("bs4", "subscriptions")`: its
`check_dependency` side effect writes `DEPENDENCIES_AVAILABLE["subscriptions"]`
from bs4 alone, which would collide with `check_subscriptions_deps()`'s
five-package computation.

**Evidence** (teed to `test-logs/task-21104-*`). Baseline: subprocess with
bs4 blocked failed `import tldw_chatbook.app` with ImportError at
monitoring_engine.py:37; closure census showed bs4+soupsieve resident
(1852 modules). Post-fix: bs4-blocked import succeeds; bs4/soupsieve out of
the closure (1831 modules); the only extras-tracked residents left
(defusedxml, PIL, rich_pixels, textual_image) are all CORE deps in
pyproject. New tests: `Tests/Packaging/test_extras_import_closure.py`
(4 passed) — bs4-absent app-import guard, explicit-list extras closure guard
(detector mutation-tested against a known-resident module), use-time
actionable-error test, bs4-present happy path. Existing:
Tests/Subscriptions 808 passed / 1 failed pre-existing (the finding-21106
`_build_test_app()` Actor_Packs crash — A/B'd against base `0f9638cef`,
fails identically there); Tests/Scheduling+Tests/Watchlists 765 passed /
285 failed, ALL with that same pre-existing 21106 signature (285/285 grep
match, two of them A/B'd on base). Collect sweep: 55,012 collected,
33 errors — identical count on base (missing-optional-dep envs: playwright,
numpy, audio stacks).

**Worth knowing.** The finding's "a base install cannot import the app" is
in practice masked on healthy installs: the CORE dep `markdownify` pins
`beautifulsoup4<5,>=4.9` transitively, so bs4 is usually present even
without extras. Nothing in the project's own declarations guarantees it
(and `--no-deps`/partial envs break), so the guard is still the right fix —
but absence tests must BLOCK the module rather than trust pip state, which
is what the new subprocess guard does.

**Files.** `tldw_chatbook/Subscriptions/monitoring_engine.py`,
`baseline_manager.py`, `scrapers/generic_scraper.py`,
`scrapers/custom_scraper.py`, new `Tests/Packaging/test_extras_import_closure.py`.
