---
id: TASK-1363
title: Offer appeared/disappeared semantics for content alerts
status: Done
assignee: []
created_date: '2026-07-29 23:55'
labels:
  - watchlists
  - enhancement
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Content alerts and filters match against the **full page text**, which TASK-1343 deliberately
preserved: a rule that has matched a phrase for months must not silently stop firing because the
phrase happens to sit in an unchanged part of the page, and a narrowed *exclude* filter would admit
items the user told the app to drop.

But "tell me when this phrase **appears**" — matching only newly-added text — is a genuinely useful
thing to want from a site watcher, and the diff now makes it cheap to compute. The same applies to
"tell me when it **disappears**".

This should be a per-rule opt-in with its own affordance, not a change to the default. Filed because
the capability arrived as a side effect of TASK-1343's diff and would otherwise be forgotten.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A content alert rule can be set to match on newly-added text, on removed text, or anywhere on the page, with anywhere remaining the default
- [x] #2 Exclude filters continue to match the whole page regardless of the setting, so a narrowed scope can never admit an excluded item
- [x] #3 Tests cover each scope, including that an existing rule with no explicit scope keeps its current page-wide behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Backend/service capability (no create-UI exists for pattern notify rules today; the scope lives in
the rule's `conditions` JSON and the service honors it — a future rule editor sets it).
1. `watchlist_rule_matching.py`: add `RULE_MATCH_ADDED_TEXT_KEY`/`RULE_MATCH_REMOVED_TEXT_KEY`; give
   `build_rule_haystack(item, scope="anywhere")` a scope param. Default "anywhere" = current behavior
   → `WatchlistFilterService` (exclude filters) never passes scope, so it stays whole-page (AC#2, free).
2. `monitoring_engine.check_url`: compute added/removed text from previous/current (reuse
   `_segment_for_diff` + SequenceMatcher for consistency with the shown diff); attach under the two
   new keys on the url_change item (matching-only, non-persisted, like RULE_MATCH_TEXT_KEY).
3. `WatchlistContentAlertService.evaluate`: read `conditions.get("scope")` (default/unknown →
   "anywhere"), pass to the haystack. Feed/API items (no added/removed keys): "appeared" → whole
   item (it all just appeared, fall back to anywhere); "disappeared" → empty (never matches).
4. Tests (AC#3): each scope; absent-scope == page-wide (regression); exclude filter stays whole-page.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Followed the plan exactly; no UI (none exists for these rules today — see plan preamble).

- `watchlist_rule_matching.py`: `RULE_MATCH_ADDED_TEXT_KEY` / `RULE_MATCH_REMOVED_TEXT_KEY` added,
  documented as matching-only/non-persisted like `RULE_MATCH_TEXT_KEY`. `build_rule_haystack` gained
  a `scope="anywhere"` param, with the original title/summary/body/author logic pulled into
  `_page_wide_haystack` so `"anywhere"` and the "appeared" fallback for a wholly-new item share one
  implementation instead of two copies that could drift. `"appeared"` narrows to the added text (plus
  title/summary/author) when `RULE_MATCH_ADDED_TEXT_KEY` is present, else falls back to page-wide (a
  feed/API item was never diffed — the whole item just appeared). `"disappeared"` uses ONLY the
  removed text when `RULE_MATCH_REMOVED_TEXT_KEY` is present (not title/summary/author — those
  describe the item as it stands now, not what left), else empty (never matches — nothing is known to
  have disappeared). Any unrecognized scope value falls back to `"anywhere"`.
- `monitoring_engine.py`: new `added_and_removed_text(previous_text, current_text)` helper, next to
  `build_change_diff`, reuses `_segment_for_diff` + `SequenceMatcher(...).get_opcodes()` so
  added/removed segments line up with what the reader's diff pane already shows. `check_url` calls it
  alongside the existing `build_change_diff` call and attaches both keys on `change_info` — a second
  segmentation pass rather than reusing `build_change_diff`'s internal one, per the plan, since the two
  helpers serve different consumers (rendering vs. matching) and diverging later must not require
  threading state between them. Confirmed `persist_subscription_item` reads a fixed key set via
  `.get(...)` for named columns only, so both new keys are silently ignored at persistence — same
  non-persistence guarantee `RULE_MATCH_TEXT_KEY` already relies on.
- `WatchlistContentAlertService.evaluate`: haystack is now built per-rule (was once per-item) reading
  `conditions.get("scope")`, defaulting to `"anywhere"`. `WatchlistFilterService` is untouched — it
  still calls `build_rule_haystack(item)` with no `scope` argument, so it always gets the `"anywhere"`
  default regardless of what a filter's `conditions` happen to contain, which is what makes AC#2 true
  without any code in the filter path referencing scope at all.
- Tests: `Tests/Subscriptions/test_watchlist_content_alert_service.py` (appeared/disappeared/anywhere/
  absent-scope/unrecognized-scope through the real service, plus a direct `build_rule_haystack`
  no-scope-arg pin), `Tests/Subscriptions/test_watchlist_filter_service.py` (AC#2: an exclude filter
  whose conditions carry a `scope` key still matches whole-page), and
  `Tests/Subscriptions/test_watchlist_content_kind_producer.py` (the producer half: `added_and_removed_text`
  unit tests including pure-addition/pure-removal shapes, an end-to-end `check_url` test asserting the
  real before/after split, and a feed/API-item test proving the no-key fallback behaviour). 59/59 green
  in the targeted run; `--collect-only Tests/Subscriptions` unchanged at 633 tests, no collection
  errors. Two mutations applied and reverted (Edit-revert, `git status --short` clean after): dropping
  the scope passthrough in the service reddened 2 tests; swapping added/removed in the producer helper
  reddened 2 tests.
- Left status `In Progress` per dispatch instructions (no UI exists to close AC#1 end-to-end for a
  human user — the capability is proven at the rule-data/service layer the tests exercise directly, as
  the plan's preamble anticipated).

## Review refinement (2026-08-03)

Whole-branch review (CLEAN — "anywhere" default proven byte-identical by md5, both mutations
reproduced, non-persistence confirmed) flagged one non-blocking note: under "appeared" on a SITE
change, the haystack originally included the synthetic change title ("Change detected: <source
name>") + summary + author alongside the added text, so an "appeared" pattern that sat in the source
name would fire on EVERY change — page-wide noise, the opposite of what "appeared" is for. Refined so
"appeared" matches ONLY the added text when the delta key is present (symmetric with "disappeared");
the feed/API whole-item fallback (delta key absent) is unchanged. Pinned by
`test_appeared_scope_ignores_the_synthetic_change_title_and_metadata` (a "Test source" pattern misses
under "appeared" but hits under "anywhere"), mutation-verified.
