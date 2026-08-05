---
id: TASK-2303
title: Create-source and assign-source stop sharing a vocabulary
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: three near-synonym labels coexist for two DIFFERENT operations — the
rail's "Add source" ASSIGNS an existing source to the selected watchlist,
while the header's "Create source" and the pane's "New Source" CREATE one.
Users will click the wrong one confidently. Assignment is also only
discoverable through that ambiguous rail button: the selected source's
Inspector has no assign/move action, and the assignment modal is a bare list
with no instruction line.

UAT findings F1 (high), F18.

## Acceptance Criteria (the what)

- [ ] One verb consistently means "create a new source" and a clearly
      different verb means "put an existing source into a watchlist", across
      rail, header, pane, guidance copy and Inspector.
- [ ] A selected source's Inspector offers the assign/move action.
- [ ] The assignment modal explains what clicking an entry does.
- [ ] First-run guidance references labels that actually exist on screen.

## Implementation Plan (the how)

1. Fix the vocabulary at one point per surface. **New** = bring a source
   into existence; **Add** = put a source that already exists into a
   watchlist. No surface may use the other family's verb.
   * pane toolbar `New Source` -> `New source`
   * centre empty-state `Create source` -> `New source`
   * rail `Add source` -> `Add existing…`, tooltip naming `New source` as
     the other operation
   * Overview first-run copy and the Inspector first-run hint -> `New source`
2. Give assignment a second, discoverable entry point: an
   `Add to watchlist…` action on a selected SOURCE's Inspector (new
   `AssignSourceToWatchlistRequested` message) and an `Add existing…` action
   on a selected WATCHLIST's Inspector (reusing the tree's existing
   `AddSourceToWatchlistRequested`).
3. Add the reverse picker dialog (`WatchlistPickerDialog`: pick a WATCHLIST
   for a source) beside the existing `WatchlistSourcePickerDialog`, and give
   BOTH an instruction line stating what clicking an entry does and that
   nothing is created.
4. Screen handler for the new message: candidate watchlists = those the
   source is not already in; write via `WatchlistBundleService.add_source`;
   confirm with a toast naming both ends; reload the tree.
5. Update the two suites that assert the old literals
   (`test_destination_visual_parity_correction.py`) and add a vocabulary
   suite that fails if the two verb families ever overlap again.
