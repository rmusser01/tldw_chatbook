---
id: TASK-21117
title: >-
  Inspector right-rail scroll forces refresh(layout=True) per scroll frame -
  split the pure-scroll path
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 16:49'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21117).

`UI/Console_Modules/right_rail.py` (249 -> 1,092 lines since the pin): `_InspectorOuterBody.
watch_scroll_y` (:116-120) routes every scroll frame into the outer geometry reconcile, which
unconditionally runs two DOM queries plus `self.refresh(layout=True)` on the whole rail (:313)
and a second call_after_refresh hop with focus-recovery + fold measurement - even for a pure
scroll where no geometry changed. Coalescing trims a wheel gesture to ~2-3 full layout passes,
on the app's hottest screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pure scroll updates only what scroll can change (hint text / clamp) without refresh(layout=True) or the refold chain
- [x] #2 The full reconcile still runs for resize / section-demand / virtual-size changes (the `_size_updated` override already distinguishes them)
- [x] #3 A counter probe while wheel-scrolling the rail shows zero whole-rail layout refreshes on pure scrolls; rail behavior (fold, focus recovery) unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-locate the defect on base 30c7e1fe9 and enumerate every caller that invalidates the Inspector outer fold (mount, rail Resize, body Resize, body _size_updated, section on_reconcile owner demand, focus-recovery scheduling, hint-toggle continuation, scroll).
2. Baseline: tee the right-rail/bounded-section/rail-cluster/CSS-harness test files at base; record pass/fail counts.
3. Red-first: add a counter probe test that wheel-scrolls the mounted production Inspector body N frames and asserts ZERO whole-rail refresh(layout=True); confirm it fails on base with ~2-3 per gesture. Add a companion test proving a real geometry change (content growth / section collapse) still takes the full path and still flips the fold + hint.
4. Implement the split: give _InspectorOuterBody a separate on_scrolled callback used only by watch_scroll_y; keep on_geometry_changed for on_resize/_size_updated. The scroll path clamps scroll_y and repaints only the outer hint copy (no refresh(layout=True), no refold chain, no focus recovery), and defers entirely to any already-scheduled geometry generation.
5. Avoid the 21115 stale-cache class of bug: add no cached size/virtual-size state - derive the hint copy from the live widgets and skip the repaint only when the painted copy is already identical (read back from the widget, not from a shadow field).
6. Re-run the full baseline set plus a --collect-only sweep; A/B every red against base.
7. Tick ACs, write Implementation Notes with the trigger -> path table and probe numbers, commit locally with the standard trailer.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Split the Inspector rail's scroll notice off its geometry notice. `_InspectorOuterBody`
now raises two distinct owner callbacks: `on_scrolled` (offset moved, nothing else) and
`on_geometry_changed` (committed size / virtual-size change). Only the second still
schedules the full outer reconcile.

Trigger -> path (also recorded in the `ConsoleInspectorRail` docstring, so the routing is
enumerated where it is maintained):

| Source | Path |
| --- | --- |
| `on_mount` | full (owner demand) |
| rail `Resize` | full (owner demand) |
| section-widget state sync - the staged-context tray, changed-files section, run inspector and settings summary each call the `on_reconcile` callback this rail hands them at compose time when a new display state lands | full (owner demand) |
| the screen's live-work card swap (`chat_screen.py:4963`, the one external `request_outer_reconcile` caller) | full (owner demand) |
| focus-recovery scheduling / retry | full (owner demand) |
| body `Resize` | full (geometry only) |
| body `_size_updated` - any committed size or virtual-size change. **This is the route a `ConsoleBoundedSection` collapse/expand or content growth actually takes** - geometry only, not owner demand (`ConsoleBoundedSection` has no `on_reconcile` of its own) | full (geometry only) |
| hint display-toggle continuation | full (geometry only) |
| body `scroll_y` change - wheel, keys, `scroll_to`, reveal | **pure scroll** |

The pure-scroll path (`_handle_outer_scrolled`) re-clamps the offset and repaints the fold
copy; no `refresh(layout=True)` on the rail, no refold chain, no focus-recovery pass. It is
sound because scroll position is not an input to `outer_hint_required` - that predicate reads
content demand against the viewport, and a scroll moves neither. When a geometry generation is
already scheduled the scroll path stands down, so it can never shadow a pending reconcile.
This is the shape the LEFT rail has always had (`_ContextOuterBody.watch_scroll_y` ->
`_update_outer_hint`, left_rail.py:124); the Inspector copy had drifted.

No cached geometry state was added, deliberately: AC 3's stale-cache class of bug needs an
invalidation hook at every mutation source, and there is nothing to invalidate here. The
no-op-repaint check reads the live widget's own `Static.content` (what was last painted)
rather than a shadow string on the rail, which the fold path would otherwise have to remember
to invalidate when it clears the copy before a display flip.

Measured (production Console shell at 160x45, 4-notch wheel gesture down + back up = 8 frames;
`Tests/UI/test_console_rail_reconciliation.py`):

| Metric | Pre-split | After |
| --- | --- | --- |
| whole-rail `refresh(layout=True)`, 8 sequential frames | 8 | **0** |
| whole-rail `refresh(layout=True)`, one coalesced 4-notch burst | 2 | **0** |
| `Screen._refresh_layout` passes over the whole gesture | 19 | **9** |

The last row is why the fold copy is now painted with `Static.update(..., layout=False)`:
`update()` defaults to `layout=True`, so a two-character repaint of a slot whose height is
pinned at compose time still scheduled a view layout (11 -> 9 of the drop above).

Tests: three added to `Tests/UI/test_console_rail_reconciliation.py` -
`test_pure_inspector_scroll_never_relayouts_the_rail` (red on base: "8 pure wheel frames
forced 8 whole-rail layout refreshes"), `test_inspector_section_collapse_still_runs_the_full_
reconcile` (green on base and after - collapsing and re-expanding a real `ConsoleBoundedSection`
must still take the layout+refold path and satisfy the fold predicate), and
`test_scroll_cost_probe_still_detects_pre_split_routing` (mutation arm: restores the old
routing and asserts the probe goes red again, so the zero-refresh assertion cannot silently
stop measuring its subject).

Modified: `tldw_chatbook/UI/Console_Modules/right_rail.py`,
`Tests/UI/test_console_rail_reconciliation.py`, `backlog/docs/lessons-textual.md`.

### Review fix round

Whole-branch review probed every fold input across a full gesture and returned two fix-first
items; both are addressed here.

**MAJOR (test coverage, not runtime) - the `layout=False` repaint and the unchanged-copy skip
were completely unguarded.** The reviewer replaced the whole `_paint_outer_hint` body with
base's `hint.update(text)` and all three new tests still passed: the `hint.region ==
hint_region` assertion pins the PRECONDITION (the slot's height is pinned) and not the
optimization, so the measured 11 -> 9 layout-pass win could regress silently. The first test now
wraps `hint.update` for the whole 12-frame gesture and asserts (a) every scroll-path write
passes `layout=False`, (b) at most 4 writes occur. Both halves are proven to bite: dropping
`layout=False` reds (a) with `[('', True), ('▼ more sections — scroll', True), ('', True)]`;
dropping the skip reds (b) with `12 scroll frames wrote the fold copy 12 times`. Shipped code
writes 3 times, all `layout=False`.

**MINOR (doc accuracy, and it was my own safety argument) - the trigger table conflated two
distinct full-path routes.** The row credited "section local reconcile ... collapse/expand or
content growth" to owner demand via `on_reconcile`. `on_reconcile` does exist -- but as a
constructor kwarg on four *section widgets* (tray, changed-files, run inspector, settings
summary; wired at right_rail.py:1083/1121/1139/1144), fired on a display-state SYNC.
`ConsoleBoundedSection` has no such callback, and the reviewer's mutation proved a section
collapse reaches the fold through the body's `_size_updated` -- geometry only. The docstring
and the table above now separate the two routes and name the live-work card swap as the one
external owner-demand caller. Also softened the `clamp=True` docstring: `validate_scroll_y`
clamps every assignment, so that clamp is defensive symmetry, not load-bearing.

Inventory gate (`scripts/check_persistent_diagnostic_inventory.py`): red, exit 1, but NOT from
this change and deliberately not regenerated. It names `DB/ChaChaNotes_DB.py`,
`DB/chachanotes_fts_backfill.py` and `app.py`; `git diff --name-only 30c7e1fe9 HEAD` shows this
branch touches none of them, and `right_rail.py` contains zero logging statements, so it cannot
contribute a row. This base (30c7e1fe9) predates dev's repair of that drift, so running
`--write` here would pin an inventory rebuilt from a stale tree and could clobber the repair on
merge.

Also added `test-logs/` to `.gitignore` (9.4 MB of teed evidence logs that were untracked and
not ignored -- one stray `git add -A` from committing). No prior convention existed: nothing
under that name has ever been tracked and no other worktree carries one.

Review-round re-run: `test_console_rail_reconciliation.py` + `test_consolidated_css_harness.py`
= 47 passed (21115's geometry contract holds).

One red appeared in the first review-round re-run and was chased down rather than re-run away:
`test_overflow_focus_order_and_recovery_stay_within_context_section` (a left-rail `_RailHarness`
focus-order test that this change cannot reach -- that harness mounts no Inspector). It then
failed once more in isolation while base passed, which looked like a regression; repeated
sampling showed it is a PRE-EXISTING FLAKE, and not one this branch worsens:

    isolated runs, this branch:   10 pass / 0 fail
    isolated runs, base 30c7e1fe9: 7 pass / 3 fail

Worth filing separately -- it went green in all three large runs (baseline, after-run,
full-file) and only misbehaves in isolation, so the ordinary suite hides it.
<!-- SECTION:NOTES:END -->
