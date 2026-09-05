---
id: TASK-31680
title: Settle Watchlists startup snapshots before direct fixture navigation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:20'
updated_date: '2026-09-05 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent direct database seeding in the scoped item fixture from racing an older startup tree snapshot that legitimately reconciles missing selections.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The scoped item fixture waits for initial startup work and publishes its newly seeded tree before direct scope navigation.
- [x] #2 A controlled delayed startup snapshot demonstrates the original failure and the corrected fixture preserves exact member-only assertions.
- [x] #3 The complete affected Watchlists collection test file passes without changing production reconciliation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve deterministic RED probe: freeze startup tree snapshot before direct DBseeds, publish after fixture selects newwatchlist; runtime correctly reconciles missing IDtoAll and original exactmember assertion fails. 2. In Tests/Watchlists/test_watchlists_collections_screen.py::test_items_reload_scopes_to_watchlist wait for startup workers before seeding, then await the real _load_tree_data worker after seeding before direct scope commit. Keep exact loaded/member-only assertions. 3. Repeat same in-memory delayed-snapshot probe against corrected fixture, then full affected4file225case gate alongside31670. 4. Scoped staticchecks,parent review,evidence notes,done,scoped commit independentof31670. ADR required:no. ADR path:N/A. Reason:test-only faithful startup and directseed preconditions, production reconciliation unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scoped item fixture now drains real startup workers before direct DBseeding and awaits a fresh real tree snapshot afterward, before selecting its newlycreatedwatchlist. Runtime reconciliation and exact member-only assertions are unchanged. Controlled in-memory gate froze startupwatchlistIDs[], released the snapshot after fixtureselectedwatchlist1, and deterministically changed scope toAll beforefix (member-onlyassertionfailed). Sameprobe afterfix preserveswatchlist1 andpasses10.39s. Full exact4filegate with collection, bundle, OPML and localservice tests:225passed106.09s; report /private/tmp/tldw-31670-31680-watchlists-green.xml. Scoped Ruff lint, changedrangeformat, diffcheck and parentindependentreview passed. No new ADR: faithful testpreconditions only. Existing startup-readiness lessons cover this mechanism.
<!-- SECTION:NOTES:END -->
