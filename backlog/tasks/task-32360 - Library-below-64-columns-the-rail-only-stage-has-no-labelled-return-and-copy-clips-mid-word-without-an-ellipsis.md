---
id: TASK-32360
title: >-
  Library below 64 columns: the rail-only stage has no labelled return and copy
  clips mid-word without an ellipsis
status: In Progress
assignee: []
created_date: '2026-09-11 06:18'
updated_date: '2026-09-11 09:05'
labels:
  - library
  - layout
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 60x24 on the rail-only stage Escape is the only return and the footer never says so; strings clip without an ellipsis; reader-shell grips overpaint list rows (B D8 caps 56-59; PROVEN library_adaptive_reader_shell.py:151-156). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The footer names the return on every single-stage surface
- [x] #2 Clipped copy ends in an ellipsis
- [x] #3 Grips never overpaint content
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Footer: return a SHORT context on the narrow stage so the return chip survives the tier.
2. text-overflow: ellipsis on the canvas classes that clip.
3. Measure grip regions at 235/100/60 before touching geometry.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All three ACs closed: AC#1 and AC#3 on the layout branch (below), AC#2 on
`fix/library-crit10-notes-details` (after the horizontal rule).

AC#1 (footer names the return) was ALREADY TRUE and is now pinned as a regression. The plan expected the chip to be lost to AppFooterStatus's width ladder and prescribed trimming the narrow-stage context to two chips; driving the real footer widget at width 60 with one, two, three and five chips renders 'esc back to Library | F1 · F6 · Ctrl+P · Ctrl+Q' in EVERY case, and the live app at 60x24 on a Library browse list paints exactly that. task-32225's 'FIRST, not appended' already carries it, so the prescribed trim would have dropped chips the footer was not painting anyway -- a no-op dressed as a fix, and it was reverted. Whatever B captured at 60x24 was a surface where this context is not active at all (the predicate stands down when an earlier Escape action owns the key, and requires a CLOSED Library pane).

AC#3 (grips never overpaint) was ALSO already true, by two independent measurements: `grip.region.right <= neighbour.region.x` holds at 235x52, 100x30 and 60x24, and a painted-frame scan finds no '<---'/'--->' run on any column outside a grip's own columns. B's '17-media-list.txt' runs at char columns 42, 42 and 183 ARE the grips' own columns (x=41 and x=182) -- the arrows looked stray because the handle was unlabelled, which task-32355 now fixes. Both assertions are pinned.

AC#2 (clipped copy ends in an ellipsis) NOT DONE, reproduced and narrowed. Live at 60x24 on Notes: the status line paints 'Library notes · Ready · Next: / Create a note or add from' and loses 'files.' (a HEIGHT clip -- the Static resolved two lines for three lines of content), and the toolbar paints 'New  Sort: Newest  Sel' (a WIDTH crop of the Button by its container). `text-overflow: ellipsis` on `.library-canvas-action` and `#library-notes-status` was tried and measured live: NO EFFECT on either, because neither loss is the horizontal text-overflow the property governs. The existing `#library-shell-grid.library-notes-compact` rules already carry the right fix for both (`width: auto` on the actions, `height: 1` + `text-wrap: nowrap` + `text-overflow: ellipsis` on the status) -- they are simply not in effect at 60x24 on the Notes route, so the real defect is the compact gate in `library_notes_controller.py` (`legacy_compact`), a file this branch does not own. Needs its own task against the Notes canvas.

Files: Tests/UI/test_library_crit10_layout.py.

### Fix round 1 (review P1)

AC#1's tick was WRONG after the round-0 cross-branch commit and is now earned
again by a PAINTED pin in both focus states.

What the review measured and I confirmed: at width 60 the real
`AppFooterStatus` paints exactly ONE context chip, about 24 rendered
characters of it. `esc back to Library` (19) survives; `esc typing · back to
Library` (28), `esc typing · leaves field` (25) and Task 1's wide-footer form
all collapse the WHOLE context to `…`. So the field-state marker and the
return cannot both paint, and round 0's ordering commit silently dropped the
return whenever a field had focus.

The coordinator's ruling was to paint `esc leaves field` in that state. That
chip would be a DEAD KEY here, which the same ruling forbids: the
`library_narrow_stage_return` binding is declared ABOVE
`library_blur_text_field`, and on this stage with an Input focused
`check_action("library_blur_text_field")` is measurably **False** while
`check_action("library_narrow_stage_return")` is **True** -- Escape performs
the return, not a blur. Both answers are now asserted inside the pin, so the
chip can never drift away from the key. The single chip therefore stays
`esc back to Library` in both states (the review's own option (a)), and the
"typing" signal yields at this one width; nothing the caret would swallow is
advertised either, because the typing block above has already dropped every
printable-key chip.

The 32346 grammar item (field state first) holds at every width where more
than one chip paints; the eliding is `AppFooterStatus`'s, so that is where
the wide/narrow grammar belongs -- flagged to the coordinator, not worked
around here.

Pins added: `test_the_narrow_stage_return_is_painted_in_both_focus_states`
(painted, parametrized over field-focused true/false, asserts no `…`) and
`test_only_one_context_chip_paints_at_sixty_columns` (the budget itself).
The round-0 registered-tuple pin is gone -- it was blind to this.
### Fix round 1 (review nit 5)

The AC#3 geometry half measured only the two grips against the reader. The
list pane and the rail -- the neighbours B actually reported overpainted --
are now in the same sorted-span assertion, with a floor on how many spans were
measured so the check cannot silently degrade to two. The whole-frame scan for
`<---`/`--->` runs outside grip columns is unchanged; it is what carries the
claim.

---

**AC#2 only.** Handed to the `notes-details` branch by the coordinator because the two
measured losses are both composed in `library_notes_canvas.py`. AC#1 and AC#3
belong to the layout branch and are untouched here.

**The handed-over diagnosis did not survive measurement.** The compact gate
(`legacy_compact` in `library_notes_controller.py`) is NOT the defect: at 60x24
on the Notes route `_notes_state.compact` is `True`, `.library-notes-route`
matches, and `#library-shell-grid` carries `library-notes-compact` --
the compact rules are in effect. What they do at that width is the problem,
and one input never arrives:

1. **The status line** (`#library-notes-authority`) is capped at two rows by
   the compact sheet, and the full line needs three in the 32-cell pane a
   60-column terminal resolves, so "files." was cut off the bottom. Compact
   now drops the `Library notes · ` prefix, which the source strip directly
   above already carries; what is left fits both rows whole.
2. **The browse toolbar** cropped "Select" to "Sel" because three actions
   need 33 cells and the pane has 32, under `overflow-x: hidden`. The action
   that does not fit moves to a row of its own -- decided from the labels
   about to be rendered (a disabled action grows a "○ " marker two cells
   wide, which a width constant cannot see).
3. **Root cause of (2):** the canvas's `pane_width` -- task-32127's
   width-aware toolbar input, resolved from the reader layout -- arrived as
   `0` ("not measured yet") on this route and no later sync ever carried the
   real one, so the toolbar was permanently on its widest shape. The canvas
   now falls back to its own rendered width (`on_resize`, the
   `LibraryRailRowButton` precedent) whenever the screen's number is absent,
   and never writes `pane_width` itself -- that attribute stays the screen's
   contract, which its own pin asserts against the resolved layout.

Nothing ends in an ellipsis because nothing is clipped any more: `text-overflow`
was measured against both losses by the layout branch and has no effect on
either (one is a height clip, the other a container crop).

**Still open, and NOT this branch's file:** at 60 columns the Notes list pane
resolves to 32 cells because `_sync_library_notes_reader_layout_from_shell`
(`library_screen.py:6248-6258`) passes `priority="items"` whenever the list
owns the workflow, and the resolver answers that with a 32/18 split -- 18
cells spent on a work pane holding only "Select a note to edit it here.".
Measured: `resolve_adaptive_reader_layout(60, …, priority=None)` closes the
Items pane instead, and `priority="items"` opens it at 32. Whether the list
should simply take the width at that size is a layout decision.

Pinned by `Tests/UI/test_library_crit10_notes_details.py::
test_notes_copy_is_not_clipped_mid_word_below_64_columns` (red first against
the unpatched canvas), and verified live at 60x24 on a 7-note profile
(`crit10/wave/notes-details/caps/cap-13-power-notes-60x24-no-midword-clip.txt`).
<!-- SECTION:NOTES:END -->
