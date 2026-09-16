---
id: TASK-32642
title: >-
  Library Notes: Info tab's blank rows become a two-column properties layout
status: Done
assignee: []
created_date: '2026-09-15 10:35'
labels:
  - library
  - notes
  - critique-4
  - idea
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 8's unlanded half, ACCEPTED in task-32627. task-32143 already
delivered the editor chrome strip (word count, cursor line, save state, the
main actions without tabbing). What it did not do is Info: roughly 25 blank
rows below a short single-column list.

A two-column properties layout fills that, and keywords moving inline under
the title is the other half of the same idea — both are density, not new
information.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Info uses its height: no run of blank rows below the last property at 235x52. **NOT MET, and not reachable as written — see Implementation Notes.** Measured on the real screen: 21 blank rows below the last content row before, 18 after (each property now claims a row instead of sharing one). The remaining run is the pane having more rows than the note has facts; closing it needs invented content or a runtime spacer, both rejected.
- [x] #2 The layout collapses to one column at 100x30 rather than truncating values.
- [x] #3 Keywords are reachable and editable from the editor without opening Info.
- [x] #4 No property is removed to make the layout fit — the fix is arrangement, not loss.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure Info on the REAL LibraryScreen at 235x52, 100x30 and 60x20 before touching anything.
2. Carry the Created/Modified/Version/Words facts as (label, value) pairs from the one function that already builds the joined line.
3. Render one labelled row per property -- aligned second column when wide, single column when compact -- and let the compact sheet give the Static `height: auto`.
4. Move the existing, permanently undisplayed `#library-note-keywords` twin into the editor region under the title.
5. RED-first pins on the real screen; re-measure at all three sizes; guide + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What the measurement found, which is not quite what the task said.** Info at
235x52 does waste its height -- 21 blank rows under Delete on the real screen
-- but the properties were not merely "a short single-column list": they were
ONE Static carrying one " · "-joined sentence. At 100x30 that sentence is 84
characters in a 46-column pane which the compact sheet pins to `height: 1`, so
it painted exactly `Created 2026-06-30 20:00 · 10w ago · Modified` and stopped.
Three of Info's four properties were unreadable at the critique's own compact
size. That is the defect this task actually closes.

**One construction, two shapes.** `build_library_note_editor_state` already
builds the joined line from a `parts` list; it now appends to a `properties`
list at the same four sites and returns both. The controller appends the word
count to both in the same statement it already used. Nothing derives one shape
from the other's text, so they cannot disagree -- the failure mode the
"one meta line" work (task-32143) spent a round on.

`library_note_property_block` is the whole renderer: aligned `label  value`
rows when wide, unpadded `label value` rows when compact, and the joined line
only when a caller supplies no pairs at all (harness states that predate the
field).

**The Tab hole this opened, and how it was caught.** Moving
`#library-note-keywords` out of `#library-note-wide-utilities` was not enough:
`apply_session_state` still carried `wide_keywords.disabled = state.compact or
show_context or locked`, correct while the field was invisible and a silent
focus hole the moment it was not. At 100x30 and 60x20 the field painted with
`display` and `visible` both True and `focusable` False, so Tab went from
Title straight to Body. Only the `pilot.press("tab")` walk sees that; a
`widget.focus()` pin would have passed. The rule is now the same one the title
and body carry (`not show_editor or locked`).

**AC#1 is not met and I do not think it is reachable.** At 235x52 the Info
region is 36 rows and the note has, at most, four properties, a keywords
field, a links heading, six actions and two section headings -- about 18 rows
with real timestamps. The blank run went 21 -> 18. Getting it to zero needs
either content Info does not have, or a runtime-computed spacer that
bottom-anchors Reuse & Export and Danger. I rejected the spacer: it is
hand-written layout on a pane that has none, it moves the blank run rather
than removing it, and `height: 1fr` on the links list -- the CSS version of
the same idea -- clips backlinks 21..50 of a 50-row cap at that size. Recorded
here rather than quietly re-scoped.

**Trade-off.** The editor's Keywords row costs one row of body at every size;
at 60x20 the body goes 4 rows -> 3. Info keeps its own Keywords field (they
are never visible at once -- the editor and context regions are mutually
exclusive), so nothing was removed to pay for it.

**Files.** `Library/library_notes_state.py`,
`UI/Library_Modules/library_notes_controller.py`,
`Widgets/Library/library_notes_canvas.py`,
`UI/Screens/library_screen.py` (screen CSS),
`css/components/_agentic_terminal.tcss` (+ the three generated sheets),
`Tests/UI/test_library_notes_w5_ideas.py` (new),
`Tests/UI/test_library_notes_wave_chrome_strip.py` and
`Tests/UI/test_library_notes_wave_editor_keys.py` (three pins re-aimed at the
rule they were written for, not the spelling they encoded),
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
