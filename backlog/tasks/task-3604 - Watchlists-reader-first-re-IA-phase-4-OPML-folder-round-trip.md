---
id: TASK-3604
title: 'Watchlists reader-first re-IA, phase 4: OPML folder round-trip'
status: Done
assignee: []
created_date: '2026-08-08 22:40'
updated_date: '2026-08-09 00:03'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement phase 4 of the reader-first design (Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md, ADR-042/043): map OPML folders to watchlists on import (innermost folder wins, case-insensitive reuse, top-level feeds stay Unassigned, additive only) and nest watchlists as folders on export, so the structure round-trips losslessly. The spec's polish tasks 2308/2310/2312/2313 are already Done. Plan: Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-4-opml-round-trip.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OPML parse preserves folder structure (innermost folder wins, feed-with-children stays a feed),Import creates/reuses watchlists by folder name case-insensitively, assigns member sources, leaves top-level feeds Unassigned, and returns an honest summary,Export nests one folder per watchlist with member feeds, deterministic order, hostile names escaped,A fresh-DB import of an exported document reproduces the exact watchlist structure (round-trip pin),Folderless OPML behaves exactly as before,Tests/Subscriptions and Tests/Watchlists green, ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-4-opml-round-trip.md

ADR required: yes (created)
ADR path: backlog/decisions/043-opml-watchlist-folder-mapping.md
Reason: the folder-to-watchlist mapping is an interchange/conflict policy (naming, nesting, reuse, additive-only) that ADR-042 does not cover.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 4 executed TDD per the plan (Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-4-opml-round-trip.md), one commit per plan task, with the mapping policy recorded in ADR-043 before implementation.

**Per task:** c9d4210e1 docs (plan + ADR-043 + this task); 0c0018780 folder-preserving parse; 16e8fdd26 import membership + URL dedupe; d51605bfc nested export; plus the closing round-trip/toast/notes commit.

**Key decisions (all in ADR-043):** innermost-folder-wins for nesting; a feed outline with children stays a feed (children inherit its context); case-insensitive watchlist reuse by stripped name; top-level feeds stay Unassigned; deterministic name-ordered export with multi-watchlist sources faithfully repeated under each folder; import is additive only.

**Found during implementation, not in the plan:**
- `add_subscription` is a plain INSERT with NO uniqueness constraint -- the plan assumed URL dedupe existed. Task 3 therefore added `SubscriptionsDB.get_subscription_id_by_source` and resolve-before-create in `import_opml`; without it the additive-only round-trip was impossible.
- The bundle service owns watchlist membership SQL directly (no SubscriptionsDB methods exist for it); the local service reaches it through thin delegates (`resolve_or_create_watchlist`, `add_source_to_watchlist`, the three list delegates) rather than growing a second SQL copy. The two bundle row queries gained `url` (additive) for the export path.
- The server backend kept its pre-ADR-043 flat export (its source model has no local membership seam) rather than losing a working path.
- The pre-round-trip toast ("Imported N source(s)") read identically for a structured import and a no-op re-import; `_opml_import_summary_text` now names new/existing sources, watchlists created/reused, and the Unassigned remainder.
- Branch-hygiene note: p4 was initially cut from a stale local origin/dev (phase 3 unfetched); caught by a missing phase-3 method and fixed by rebase before any phase-4 commit landed.

**PR #1448 review follow-up:** repeated URLs in one document now reuse an
in-import memo, so a shared feed exported under several watchlists counts once
and never reads as pre-existing on a fresh import. The summary carries an
explicit unique-source Unassigned count instead of subtracting membership edges.
Case-insensitive watchlist reuse is a direct database lookup with no 10,000-row
correctness cap and a deterministic Python-normalization SQL function preserves
Unicode case matching that SQLite `LOWER()` cannot provide; the two modified
export reads use the shared transaction seam;
and the public parser docstring records its arguments, return shape, and parse
error. Backlog Guard also exposed a pre-existing `TASK-3603` collision on `dev`;
the later Watchlists phase-3 task and all of its references were renumbered to
`TASK-3791` after a full remote/worktree sweep.

**Files:** `Subscriptions/watchlist_opml_service.py` (recursive folder-walk parse; structured export), `Subscriptions/watchlist_scope_service.py` (import dedupe + membership + summary; export assembly), `Subscriptions/local_watchlists_service.py` (+5 delegates), `Subscriptions/watchlist_bundle_service.py` (row queries gain url), `DB/Subscriptions_DB.py` (get_subscription_id_by_source), `UI/Screens/watchlists_collections_screen.py` (summary toast), `Tests/Subscriptions/test_watchlist_opml_service.py`, `Tests/Watchlists/test_watchlist_scope_service.py`, `Tests/Watchlists/test_watchlists_collections_screen.py`, `backlog/decisions/043-opml-watchlist-folder-mapping.md`.
<!-- SECTION:NOTES:END -->
