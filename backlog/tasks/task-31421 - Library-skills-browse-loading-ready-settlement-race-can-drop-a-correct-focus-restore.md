---
id: TASK-31421
title: >-
  Library skills browse: loading/ready settlement race can drop a correct focus restore
status: To Do
assignee: []
created_date: '2026-09-04 00:00'
updated_date: '2026-09-04 00:00'
labels:
  - skills
  - library
  - bug
dependencies: []
references:
  - .superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/task-2-report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`LibrarySkillsBrowseController`'s committed-mutation refresh settles in two
rounds against the shared canvas-sync dispatcher: `dispatch()`'s synchronous
"loading" call, then the async worker's own "ready" call via `apply()`. Each
round schedules its own `_sync_library_canvas(..., then=restore_focus, ...)`
callback, but `queue_after_recompose` holds only one pending callback per
host. Against a skills-scope service fast enough for both rounds to land
within the same event-loop turn, the ready round's own resync overwrites the
loading round's still-pending, CORRECT focus-restore callback before it ever
fires — so a focus identity the loading round correctly derived (e.g. the
skills filter input) is silently dropped, and the ready round's own re-derived
`focus_identity` (typically `None`, since nothing is focused yet at that
point) wins instead.

This is a pre-existing race in `library_skills_browse_controller.py`,
unrelated to and unaffected by the wave-4 skills controller/state move — it
was found while building the covering test for that move's own CRITICAL
review finding (an unrelated unbound `focused` property), and reproduces
identically whether that property is present or absent. Confirmed only
against a bounded-delay fake service; the practical risk is intermittent
focus-restore misses whenever the real skills-scope service resolves fast
enough for both settlement rounds to coincide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Given a skills-browse scope service that resolves fast enough for the loading and ready settlement rounds to land in the same event-loop turn, the loading round's own correct focus-restore callback is not silently discarded by the ready round's resync.
- [ ] #2 A test reproduces the race with a near-instant fake skills-scope service, fails against the current `library_skills_browse_controller.py`, and passes after the fix.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: This is a scheduling-order fix inside an existing controller's
already-established two-round settlement flow; it does not add or remove a
durable contract, a trust boundary, or a public API.

1. Reproduce the race directly (instrument or spy on
   `_sync_library_skills_browse_result` / `queue_after_recompose` to show the
   loading round's callback getting silently replaced by the ready round's
   own resync within the same turn).
2. Decide the fix shape: either merge/coalesce the two rounds' own
   `then=restore_focus` callbacks so the later one only overwrites the
   earlier one with an equal-or-better focus identity, or make
   `queue_after_recompose` per-host queueing preserve a still-pending
   correct restore rather than dropping it outright.
3. Add the reproducing test (a near-instant/synchronous fake
   `SkillsScopeService`) alongside the existing bounded-delay fake this
   finding's own covering test already uses
   (`Tests/UI/test_library_skills_canvas.py`).
4. Run the full skills wiring/characterization/battery per the recipe
   (`backlog/docs/library-decomposition-recipe.md`) to confirm no other
   canvas-sync consumer regresses.
