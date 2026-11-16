---
id: TASK-41304
title: "Collections 80x24: after a note saves, the focused Save sits off the 24-row screen"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 23:54'
created_date: 2026-11-16 23:32
labels:
- follow-up
- library
- focus
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
- tldw_chatbook/Widgets/Library/library_collections_capture_reader.py
- Tests/UI/test_library_collection_reader_journeys.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 15 of the 2026-10-23 burn-down (coordinator authorization, libcoll fix round 1, 2026-11-16). Found by fu-r15-libtests (TASK-41127).

**Why.** At 80x24 in both themes, `test_annotation_drafts_survive_disclosures_modes_and_capture_return` fails after "Save capture note". The Save button keeps focus but sits at y=26 on a 24-row screen and paints nothing, so the focused control is invisible. The Collections User Guide says saving a capture note "keeps its Save action focused and visible, scrolling Work when needed".

**Evidence.** Reproduced at dev cc6006fcf5 with the test run under the Library harness's `bootstrap_profile`: `[textual-dark-size1]` and `[textual-light-size1]` fail with "Expected readable #library-collections-freeform-note-save; focus=Button(id='library-collections-freeform-note-save'...); region=Region(x=11, y=26, width=21, height=1); paint=''". Run plainly, the test fails earlier with RecoveryRequired (raw_source_selection_changed), so it needs the marker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a capture note saves at 80x24, the control that keeps focus is painted on screen (Work is scrolled to it), in both themes
- [x] #2 The annotation-draft journey runs plainly (bootstrap_profile marker) and is red before the fix, green after
- [x] #3 The journey is checked for a PR lane: appended at the census tail with its measured serial times, or the reason it is not
<!-- AC:END -->

## Implementation Plan

1. Reproduce at base and at the branch head under the marker; find which repaint loses the scroll.
2. Fix it in Collections so the refocused Save is scrolled into view after the save's repaint.
3. Mark the test `bootstrap_profile`; show it red at base and green at head; time it serially with the UI lane's flags and append it to the census tail if it fits.

## Implementation Notes

- **Reproduced and fixed by TASK-41126's in-place repaint (28c4c9fffa); no further source change.** With the `bootstrap_profile` marker, the journey's two 80x24 cases fail with base sources (dev cc6006fcf5) in each of two runs: Save focused at `Region(x=11, y=26, ...)`, painted ''. With head sources they pass 14 of 14 runs (load 88-100).
- **Mechanism.** On dev the note save ended in a whole-screen recompose. It built a new Work pane at scroll offset 0 and restored focus with `scroll_visible=False`, and the controller's `reveal_current_focus` ran two `call_after_refresh` hops later. Those hops are not a layout pass (lessons-textual), so Work was never scrolled to Save. Now only the pane's children are rebuilt and the Work pane keeps its scroll offset, so Save stays where the user had it, in view. Negative control: with the refocus in `recompose_capture_panes` changed to `focus(scroll_visible=False)`, both cases still pass 8 of 8. The retained offset is what keeps Save visible, not the refocus scroll.
- **Test** (AC#2): `test_annotation_drafts_survive_disclosures_modes_and_capture_return` is marked `bootstrap_profile` (the Library harness's profile, as its neighbours are). Unmarked it fails locally with RecoveryRequired (raw_source_selection_changed). Red at base and green at head as above; the whole node passes 4 of 4 plainly at head.
- **Lane** (AC#3): `[textual-dark-size1]` and `[textual-light-size1]` are appended at the census tail, `MINIMUM_FILES` 2146 -> 2148. Serial with the lane's flags at load 96-100: 48.5 s and 50.8 s of call. The 170x48 pair (60.8 s and 67.1 s) is left off: it does not pin this bug, and the lane has no margin.
- Files: `Tests/UI/test_library_collection_reader_journeys.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
