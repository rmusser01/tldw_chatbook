---
id: TASK-22221
title: 'Avatar geometry: PIL resample off the loop; trim allocation-reconcile legs'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 05:25'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22221).

New with PR #2034. (a) `UI/Console_Modules/left_rail.py:1037-1094` +
`Chat/console_image_view.py:103-107`: each distinct rail viewport size triggers
`image.copy()` + LANCZOS `thumbnail` synchronously on the UI thread (drag-resize burst
cost; the memo prevents steady-state cost). (b) `_run_allocation_reconcile` gained three
per-pass legs since the pin (`left_rail.py:944-1035`): avatar geometry reconcile,
`set_allocation(None)`+height reset across all 7 sections before every measurement pass,
and `_measure_outer_content_height` iterating every outer child — plus
`_refresh_workspace_tree_after_reflow` (`:1030`) clears the tree hover row on EVERY pass
(~5 Hz during runs: hover flicker + 2 repaints + tooltip per tick).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The avatar resample runs off the loop with the existing memo and race fences intact
- [x] #2 The allocation reconcile does not clear tree hover when the hover row is unaffected
- [x] #3 Per-pass query/measure counts recorded before/after; reconcile passes converge in the same number of frames
- [x] #4 Resize-drag cost measured before/after with a high-resolution character card
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red probes: (a) count+time PIL copy/LANCZOS resamples executed on the event loop inside the geometry-fit path during a scripted viewport-size burst with a 1024px card (nonzero today); (b) count hover clears per allocation-reconcile pass when the tree did not move (nonzero today).
2. (a) Replace the discarded-pixels resample in fit_character_avatar_cell_box with an exact arithmetic replication of PIL thumbnail target-size rounding (pure function in console_image_view), property-tested for equivalence against real Image.thumbnail across the dimension space. All memo/epoch/followup/suppression fences stay untouched. (Written before measuring the second leg; the deviation is in the notes. Removing the discarded-pixels resample turned out to be only 28 ms of a 215 ms drag -- the visible mosaic render is the other 187 ms and is genuine work, so it went to a worker thread after all, race class and all. AC 1 stands unchanged and is met.)
3. (b) Make _refresh_workspace_tree_after_reflow signature-guarded: defer one coalesced post-refresh check that compares (tree.region, tree.scroll_offset) against the last settled reading and clears hover + recomputes tooltip only when the tree actually moved on screen.
4. Measure after: same burst -> zero resamples on the loop; steady-state reconcile passes -> zero hover clears; per-pass query counts and convergence pass-count unchanged (gate-file counters).
5. Targeted tests + 22203 gate file + collect-only sweep, tee everything; mutation tests: restore unconditional hover clear -> probe reds; break the arithmetic rounding -> property test reds; drop the existing replace-generation fence -> serialization test reds.
6. preflight, task hygiene, commit, push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two independent legs of the Console rail's per-viewport work, both measured
before and after with a 1024x1024 card (the render cache's decode cap, so the
production worst case).

**(a) Avatar geometry.** The finding's first leg was pure waste:
`fit_character_avatar_cell_box` called `scale_image_for_cell_box` (a `copy()`
plus a LANCZOS `thumbnail`) and then read only `.width`/`.height` off the
result, discarding every pixel. Replaced with
`scale_image_pixel_size_for_cell_box`, an exact arithmetic replication of
Pillow 12.3.0's `thumbnail` size rule (`preserve_aspect_ratio` +
`round_aspect`, including the no-enlargement early return). Verified against
the real resample over 2,952 size/box combinations, zero mismatches, and
pinned by a parametrized equivalence test plus a 120-step drag sweep that
forbids `resize`/`thumbnail` outright.

The second leg is the mosaic render the user actually sees, and it is genuine
work, so it moved off the loop rather than away: a loop-side factory
(`build_character_avatar_prerender_job`) snapshots the live spec and colour
mode, and the rail runs the returned job through `asyncio.to_thread` inside
the existing replacement worker. The result is an opaque
`CharacterAvatarPrerender` token; the widget builder uses it only when its
recorded image identity, resolved box, and colour mode still match, and
rebuilds inline otherwise. The renderer itself moved into
`character_avatar_layout.render_character_avatar_mosaic` so the off-loop and
inline paths are one code path, not two that can drift.

The memo (`_character_avatar_fit_signature`), the geometry epoch, and the
`_character_avatar_followup_pending`/`_suppressed_epoch` fixed-point guards
are untouched.

**The race fence, and a fence I deleted.** Moving the render off the loop
introduces the async-completion race, so I added a generation check after the
thread hop -- and mutation testing killed my own addition: removing it changed
nothing, because `replace_character_avatar_widget` already re-checks
`is_current()` as the FIRST statement inside the mount lock, before it
unmounts anything. Both properties (a stale-size render never paints; a
superseded pass never blanks the live portrait) are therefore already owned by
the pre-existing fence. I deleted my redundant check rather than ship a line no
test can kill, left a comment naming the fence that does own the race, and
verified both properties red under a mutation of THAT fence.

**(b) Allocation reconcile.** `_refresh_workspace_tree_after_reflow` cleared
the tree's hover row and recomputed its tooltip on every pass (~5 Hz while
streaming). It now defers one coalesced post-refresh check and compares a
cheap geometry signature -- `(tree.region, tree.scroll_offset)` -- against the
last settled reading, clearing only when the tree really moved. Deferred
rather than immediate because the leg runs at the END of a pass, before that
pass's own style writes have been laid out: reading geometry inline would
compare the previous layout against itself and miss the move it just caused.
An unreadable signature is treated as "assume it moved", so the leg fails
toward clearing. Content changes stay out of scope: `sync_projection` already
re-checks hover identity and `watch_scroll_y` already handles local scrolling.

**Measurements** (mounted Console, 32-step scripted rail-width drag, both arms
in one process):

| | on-loop avatar work | fit calls | mosaic on loop | off-loop |
|---|---|---|---|---|
| before | 152.7 ms | 54 / 44.9 ms | 24 / 107.8 ms | 0 |
| after | **0.9 ms** | 56 / 0.9 ms | **0** | 50 / 210.8 ms |

Hover, over 8 no-op reconcile passes: **8/8 clears -> 0/8**, tooltip
recomputes **24 -> 0**. Per-pass counts and convergence are unchanged, which
was the point: `query_one` 44 -> 44 per pass, outer measures 1 -> 1, passes to
converge 1 -> 1.

**Mutations run:** unconditional hover clear restored -> 2 reds; never-clear
(the opposite over-correction) -> 2 reds; naive `floor` rounding in the
thumbnail arithmetic -> 6 reds; the real avatar generation fence removed -> 2
reds. My own post-hop fence -> 0 reds, which is why it is gone.

**Test doubles updated, not worked around.** Six stand-ins for the avatar
builder seam had the pre-change signature; one produced exactly the
`WorkerFailed` this widening risks. All six now accept the seam's real
contract. `test_mosaic_fallback_contains_rather_than_crops` asserted on the
SOURCE TEXT of one function for a `fit="contain"` literal; it now asserts the
fit the shared renderer is actually called with, which survives the move.

**Files:** `Chat/console_image_view.py`,
`UI/Console_Modules/character_avatar_layout.py`,
`UI/Console_Modules/left_rail.py`, `UI/Screens/chat_screen.py`;
new gates `Tests/UI/test_console_avatar_geometry_offloop.py`,
`Tests/UI/test_console_rail_reflow_hover_budget.py`; updated
`Tests/Chat/test_console_image_view.py`,
`Tests/UI/test_console_character_avatar.py`,
`Tests/UI/test_console_rail_reconciliation.py`,
`Tests/UI/test_console_reaction_picker.py`;
`Docs/security/production-diagnostic-inventory.json` (two constant-string
debug statements, reviewed per preflight's procedure -- no interpolation).
<!-- SECTION:NOTES:END -->
