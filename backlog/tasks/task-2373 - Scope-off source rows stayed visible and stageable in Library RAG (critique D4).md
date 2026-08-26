---
id: TASK-2373
title: Scope-off source rows stayed visible and stageable in Library RAG (critique D4)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04 20:07'
labels:
  - library
  - rag
  - honesty
dependencies: []
priority: medium
---

## Description

The 2026-08-04 RAG re-score critique (D4) found that toggling a source off in the Library RAG scope had no effect on results already on screen: that source's rows stayed displayed, selectable, and stageable in the same snapshot — the scope toggle only affected future fetches, not what a user could act on right now.

## Acceptance Criteria

- [x] Toggling a source off in scope hides and disables its rows in the current snapshot immediately, not only on the next fetch
- [x] Row select/open handlers address the correct row after filtering (not a raw pre-filter index)
- [x] A count-drift regression is covered (e.g. a source's row count changing via an unrelated refresh path while scope filtering is active)

## Implementation Notes

Fixed in PR-T1 Task 5, commits `c01645a8c` (initial) and `50bc76966` (review round 2 fix).

The initial fix deviated from a pure display filter in two respects, both upheld by review as necessary: (1) the index handlers for select/open were also fixed, because a display-only filter would have left card indices misaligned with the underlying (unfiltered) data — causing select/open to act on the WRONG row, which is worse than the original defect; and (2) a new singular/plural canonicalization map was added, distinct from the pre-existing `_OPEN_SOURCE_TYPE_MAP`.

Review (sonnet) approved the index-handler fix but flagged the filter's basis (raw `selected_source_types` rather than a count-intersected scope) as reachable to a real count-drift bug: `_refresh_local_source_snapshot`'s ~30 call sites mean a source's count can change elsewhere (e.g. deleting the last note) while scope filtering is active, showing a stale "○ Notes (0)" while a stale row remained stageable — D4's symptom via a different trigger. Fix round 1 shipped a **hybrid** basis: filtering skips (uses the raw arg) only when the caller passes `None`; when a caller passes a real scope, filtering uses the count-intersected version. This is keyed off the raw argument's None-ness, which is structurally forced since the scope argument itself is never `None`. A map-parity test now guards the three hand-synced canonicalization dicts against drift, and a mutation-reasoning check confirmed the count-drift scenario now fails the new test as intended.

Deferred (documented, not fixed): parity tests don't catch a key added to only one of the three maps; unrecognized provenance defaults permissively, mirroring an existing precedent elsewhere in the same module.
