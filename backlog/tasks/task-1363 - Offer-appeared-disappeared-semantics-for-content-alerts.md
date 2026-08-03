---
id: TASK-1363
title: Offer appeared/disappeared semantics for content alerts
status: In Progress
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
- [ ] #1 A content alert rule can be set to match on newly-added text, on removed text, or anywhere on the page, with anywhere remaining the default
- [ ] #2 Exclude filters continue to match the whole page regardless of the setting, so a narrowed scope can never admit an excluded item
- [ ] #3 Tests cover each scope, including that an existing rule with no explicit scope keeps its current page-wide behaviour
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
