---
id: TASK-32549
title: >-
  Library Notes: "○ Sort", "○ Export selected" and "○ Resolution history" carry
  their disabled reason only in a tooltip that never renders
status: Done
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 17:06'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Sam. D13. Task-32257 fixed Import selected items / Check selection through `_disabled_action_label`; task-32362 fixed Export bundle / Find; task-32261 AC#3 keeps the ○ glyph by design (Library-wide marker, task-32235). These three controls remain reason-less.

**What happened.** "○ Sort: Newest" while a filter shows (B 36, 37), "○ Export selected" with 0 selected (B 43), "○ Resolution history" (B 39) — glyph only; the reason lives in a tooltip that does not render in the TUI. Only "○ Server notes" carries its reason on screen. Captures: B 36, 37, 39, 43.

**Cause.** PROVEN pattern (reason on tooltip only), sites INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the three controls states its disabled reason as on-screen text at the control, through shared sentence builders in library_shell_state.py (on a line beside the control rather than inside its label: measured live, the label spelling clipped at 100x30, and all three rows are width-starved -- task-32261's select strip, the merge-threshold budget, and a max-height-3 pinned bar)
- [x] #2 A test pins the three reasons in their disabled states
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce '○ Sort: Newest' while filtered and '○ Export selected' with none selected at 235x52.
2. Measure whether each reason fits in its control's label at every pane width the control is reachable at.
3. State each reason on screen through ONE shared seam.
4. Pin the three disabled states.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All three reasons are on screen now, and where each one goes was MEASURED rather than argued.

The label spelling (`library_disabled_action_label(..., reason=...)`) was implemented first and walked live. At 100x30 the Notes list pane is 42 cells; "○ Sort unavailable — clear the filter" costs 41 with its button chrome, and beside "New" it painted "○ Sort unavailable — clear the" against the grip (`wave4-caps/layout/layout-21-100x30-sort-reason`) — re-creating, inside this very wave, the defect task-32544 and task-32557 fix. It would also have invalidated `_TOOLBAR_MERGE_MIN_WIDTH`, a measured constant derived from the widest browse composition. So the reasons live on lines beside their controls, which is the treatment "○ Server notes" has had in this codebase since task-32257:

- **Sort** — its own `.library-disabled-reason` line under the toolbar, and only while the Sort control is on screen: select mode replaces the whole toolbar, and a reason for a control nobody can see is the same dishonesty in the other direction (spotted live at 100x30, `layout-23`).
- **Export selected** — on the count line that was already under the strip: "0 selected — Export selected unavailable", reverting to the plain count on the first check. A dedicated line was tried first and cost the tree a row at 60x20 — `test_library_note_60x20_navigator_state_allocation[selection]` is green on dev and went red on the branch, which is exactly what a full-file comparison against a detached dev baseline is for. Compose and `_apply_library_row_toggle` build that string through one function, and the action's compact/full spelling comes from the base the canvas already stashes on the button, so the two cannot drift (task-32272's rule).
- **Resolution history** — a `.library-disabled-reason` line above the sync canvas's pinned action bar, which is a single `max-height: 3` row. Its two hand-spelled "○ Resolution history" literals now go through the shared marker helper.

Seams added to `library_shell_state.py`: `library_disabled_reason_line(label, reason)` (the sentence), `library_selection_count_line(count, action_label)` (the count-line spelling of it), and a `reason=` keyword on `library_disabled_action_label` for any future control whose row does have the cells. **AC#1 was updated before the change landed** to name what was actually built.

Live at 235x52 and 100x30: both list reasons paint whole, the Sort one disappears in select mode, the Export one on the first check (`layout-22`, `layout-23`, `layout-24`, `layout-25`, `layout-14`).

Files: Library/library_shell_state.py, Widgets/Library/library_notes_canvas.py, Widgets/Library/library_notes_add_from_files_canvas.py, UI/Library_Modules/canvas_sync.py, Tests/Library/test_library_shell_state.py, Tests/UI/test_library_notes_w4_layout.py, Docs/User_Guide/library/notes.md.

NOT done, deliberately: the brief's relocation of `_disabled_action_label` out of `library_note_import_canvas.py`. The sibling `sync-roots` branch is already written against it and does not switch, so relocating would break an unlanded branch for no behaviour change; the reason grammar is identical in both.
<!-- SECTION:NOTES:END -->
