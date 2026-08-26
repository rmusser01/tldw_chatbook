---
id: TASK-22450
title: Stabilize Watchlists Feed Items snapshots and atomic scope commits
status: Done
assignee: []
created_date: '2026-08-26 04:28'
updated_date: '2026-08-26 10:35'
labels:
  - watchlists
  - reader
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md
  - backlog/decisions/042-watchlists-reader-first-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement ADR-042’s stable Feed Items snapshot and pending/committed scope foundation so navigation, pagination, refresh, and background arrivals remain honest before aggregate feed selection is added.

Renumbering provenance: the Backlog CLI initially allocated TASK-22301, which already existed on `origin/fix/citation-boundary-burndown`. This task was renumbered before implementation to TASK-22450 after sweeping remote branches, worktrees, and local task files; TASK-22451 is its dependent aggregate-feed follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scope navigation keeps the committed highlight, rows, and Reader active until the replacement snapshot succeeds, then commits scope, rows, and Reader clearing atomically.
- [x] #2 Feed Items pagination uses the canonical effective-date and item-id keyset with an initial item-id high-water mark and a seen-id guard instead of offset paging.
- [x] #3 New arrivals remain behind the new-items affordance and do not reorder or relabel the mounted snapshot before explicit refresh.
- [x] #4 Failed and superseded loads preserve the committed view and cannot publish stale results.
- [x] #5 Search, status filters, page navigation, and open-item pinning operate within the stable snapshot contract.
- [x] #6 Affected Watchlists tests, modified-file Ruff, and branch diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: ADR-042 already governs the stable Reader snapshot, pending/committed scope, keyset, high-water, seen-id, and arrival behavior. This task implements that decision without changing schema, storage ownership, or application-wide interfaces.

Detailed plan: `Docs/superpowers/plans/2026-08-25-watchlists-stable-feed-items-snapshots.md`

1. Define typed Reader cursor/page values and a pure screen-side cached snapshot with a seen-id guard.
2. Add the real-SQLite Reader keyset, matching high-water/count, and exact arrival-count queries while preserving the established agent cursor contract.
3. Route typed Reader pages and arrival counts through the local service, scope service, and Watchlists controller without changing legacy list callers.
4. Replace offset paging with cached keyset pages, cached Previous navigation, and transactional query-context replacement.
5. Split Read scope request from commit so the candidate scope publishes only with its mounted first page; preserve immediate management-tab commits without hidden Reader I/O.
6. Make snapshot counts and new-arrival notices screen-owned and clear them only after successful explicit refresh.
7. Run only the affected Watchlists/Subscriptions tests, modified-file Ruff, and diff checks; self-review, document evidence, and close TASK-22450.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-042 stable Reader foundation: immutable Reader page/cursor values, a screen-owned cached snapshot with high-water and seen-id guards, DESC/DESC SQLite keyset pages, exact arrival counts, typed local service routing, atomic pending/committed scope publication, and honest snapshot-count/new-arrival presentation. Legacy list callers and the agent's DESC/ASC cursor contract remain unchanged; TASK-22451 aggregate feed children remain documentation-only follow-up scope.

Modified production areas: `Subscriptions_DB.py`; Watchlists Reader page, normalizer, local/scope service, backend controller, collection screen, cached snapshot, and article-list modules. Added or updated the corresponding DB, Subscriptions, and Watchlists focused tests plus this task and execution plan. No schema change or new ADR was required; existing `backlog/decisions/042-watchlists-reader-first-ia.md` governs the behavior.

Verification at final implementation HEAD: the exact Task 7 changed-functionality selection passed **276 tests** with **161 deselected** and **2 warnings** (Requests dependency-version mismatch and Python 3.13 `audioop` deprecation from pydub) in 140.06 seconds. The exact modified-file Ruff command exited 0 with `All checks passed!`; `git diff --check` exited 0; the worktree was clean before closeout edits; and the `origin/dev...HEAD` scope contained 25 expected TASK-22450 implementation/test/docs files plus only the already-approved TASK-22451 spec/task bookkeeping. Requirements-oriented review found no Reader `OFFSET`, preserved Reader DESC/DESC versus agent DESC/ASC ordering, initial-watermark continuations, duplicate suppression, atomic failure/supersession behavior, immediate management invalidation without hidden Reader I/O, exact arrival predicates, and refresh-failure retention of the committed notice.
<!-- SECTION:NOTES:END -->
