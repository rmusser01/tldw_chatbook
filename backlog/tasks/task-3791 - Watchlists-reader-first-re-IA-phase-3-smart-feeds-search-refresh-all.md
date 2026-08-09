---
id: TASK-3791
title: 'Watchlists reader-first re-IA, phase 3: smart feeds, search, refresh-all'
status: Done
assignee: []
created_date: '2026-08-08 15:44'
updated_date: '2026-08-08 21:05'
labels:
  - watchlists
  - ux
dependencies: []
---

## Task Identity Note

This task was renumbered from `TASK-3603` to `TASK-3791` during PR #1448.
Backlog Guard found that the earlier QwenCloud provider task already owned
`TASK-3603`; the Watchlists task was filed later and therefore moves while the
earlier task keeps its identity. The replacement was checked across every
fetched remote ref and local worktree before this rename.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement phase 3 of the reader-first design (Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md, ADR-042): All Unread + Today rail nodes beside Starred, / corpus-wide search via subscription_items_fts with LIKE fallback, r refresh-all with guardrails + aggregated notification, and an N-new-items pill. Plan: Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-3-smart-feeds-search-refresh-all.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rail shows All Unread and Today smart feeds with correct badges that refresh through the existing counts path,/ focuses the items search and results span the whole corpus via FTS5, with LIKE fallback when FTS is unavailable and no FTS-syntax errors on hostile input,r checks every active non-paused source exactly once per press, toasts one aggregated summary with the unread delta, and shows an N-new-items pill that never yanks the list mid-triage,Help text advertises / and r (decision 031),Tests/Watchlists and Tests/Subscriptions green, ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-3-smart-feeds-search-refresh-all.md

ADR required: no (already exists)
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: ADR-042 covers the re-IA; phase 3 is a direct implementation of it, same ruling as phase 2.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 3 executed TDD per the plan (Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-3-smart-feeds-search-refresh-all.md), one commit per plan task, on the tree phase 2 left (PR #1430).

**Per task:** 62bf0aed8 docs (plan + this task); 75b19fef4 search/since predicates; 820e33561 `/` corpus-wide search; 40ea49b8e All Unread + Today rail nodes; a1911a649 `r` refresh-all + pill; plus the closing help/pins/notes commit.

**Key decisions:**
- `subscription_items_fts` (external-content FTS5 over title/content/author, with triggers and backfill) existed since an earlier migration but had NO query consumer; phase 3 adds the first. Every whitespace term is a double-quoted literal AND-joined (the Library `_quote_fts_term` rule, without the RAG plural widening), so FTS5 operator injection is structurally impossible; an `OperationalError` (pre-migration DB, fts5 compiled out) degrades to AND-of-terms OR-across-columns LIKE with escaped wildcards. Pinned against a real dropped-table DB and a hostile-query battery.
- The pane's instant client-side filter stays as the in-flight pre-filter and now reads content/author too — the same columns FTS indexes, so a content-matched corpus result is not filtered out of the page it just arrived on. Server reloads debounce at 0.3s (the tree-counts timer shape), one query per typing pause.
- All Unread's badge reuses the All-sources unread count (one fact, two angles — no new query); Today's rides a new TODAY_BUCKET from `get_unread_items_count_since(local-midnight-as-UTC-ISO)`, inserted on every tree-data load so both badges refresh through the existing debounced counts path. The floor is computed in local time and converted back to UTC so the COALESCE string compare stays same-shape.
- The All Unread node is the stronger statement: under it, `_load_items` drops the filter's `statuses` kwarg (the DB raises on status+statuses together, and widening an All Unread list would make the node lie).
- Refresh-all guardrails: eligibility reads the normalized `active` (is_active AND NOT paused — an auto-paused source is skipped, not poked); sequential launches through the new controller `check_all` (the local executor serializes runs already); one-batch-at-a-time flag; per-source failures soft-fail into the aggregate; ONE toast at the end ("Checked N sources — M new items (K failed)").
- "N new items" is the ALL-sources unread DELTA across the batch — the same fact the rail counts, per the legend — not run-table archaeology. The pill is a Static (a notice you can act on, not one of the strip's verb Buttons), shown in place via a plain reactive; its click dismisses it and posts the Refresh button's own RefreshItemsRequested, so it never yanks the list mid-triage.

**Test-found mechanics, pinned:** a server reload recomposes the pane and rebuilds the search Input, so test code (and any future caller) must re-query the input after a reload — even a freshly-queried handle can be the about-to-be-destroyed widget while the recompose is in flight (the clearing test settles, re-queries, and proves propagation through the screen's mirror before waiting on the reload). The pill's click is dispatched as a fabricated Click in the bare harness (no project CSS there; the 1fr input eats the strip and a pointer click is OutOfBounds — a harness layout artifact, not the behavior).

**Disclosure:** the 5 schedules/mcp parity failures reproduce on pristine origin/dev (verified on a detached checkout) and are unrelated to this branch; the phase-2 PR's CI flake notes (macOS runner Errno 5 at dependency install) are environmental.

**Files:** `DB/Subscriptions_DB.py` (search/since predicates, `_search_items_rows` FTS+LIKE, `get_unread_items_count_since`), `Subscriptions/{local_watchlists_service,watchlist_scope_service,watchlist_bundle_service}.py` (forwarding/delegation), `UI/Watchlists_Modules/{article_list,watchlist_tree,watchlists_backend_controller}.py`, `UI/Screens/watchlists_collections_screen.py` (bindings, debounce reload, scope mapping, merge rule, batch worker), `css/features/_watchlists.tcss` + regenerated bundle, and the Tests/{DB,Watchlists,UI}/ suites listed in the per-task commits.
<!-- SECTION:NOTES:END -->
