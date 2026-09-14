---
id: TASK-32549
title: >-
  Library Notes: "○ Sort", "○ Export selected" and "○ Resolution history" carry
  their disabled reason only in a tooltip that never renders
status: In Progress
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 16:58'
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
- [x] #1 Each of the three controls states its disabled reason as on-screen text at the control, through one shared reason-sentence seam in library_shell_state.py (the `.library-disabled-reason` line rather than the label: measured live, the label spelling clipped at 100x30, and two of the three rows are already width-starved by task-32261 and a max-height-3 pinned bar)
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
All three reasons are on screen now, through ONE seam -- but the seam is the shared `.library-disabled-reason` line, not the button label, and that choice was MEASURED rather than argued.

The label spelling was implemented first and walked live. At 100x30 the Notes list pane is 42 cells; '○ Sort unavailable — clear the filter' costs 41 with its button chrome, and beside 'New' it painted '○ Sort unavailable — clear the' against the grip (wave4-caps/layout/layout-21-100x30-sort-reason) -- re-creating, in this very wave, the defect task-32544 and task-32557 fix. Export selected is worse: task-32261 already had to hide the select strip's own counter to keep it on a 42-column pane, so the compact row has at most three spare characters. The sync canvas's opener sits in a `max-height: 3` pinned action bar. So all three take the line that '○ Server notes' has carried in this codebase since task-32257, and AC#1's 'through the shared _disabled_action_label seam' is honoured as one shared sentence builder rather than one shared label: `library_disabled_reason_line(label, reason)` in `library_shell_state.py`, with `library_disabled_action_label(..., reason=...)` composing the same sentence into a label for any future control whose row has the cells. **AC#1 was updated to say so before the change landed.**

The three lines: 'Sort unavailable — clear the filter' (under the toolbar, only while the Sort control is on screen -- select mode replaces the toolbar, and a reason for an invisible control is the same dishonesty in the other direction), 'Export selected unavailable — nothing selected' (under the select strip, flipped in place by `_apply_library_row_toggle` so it clears the moment a row is checked), 'Resolution history unavailable — it starts after this root is activated' (above the sync canvas's pinned bar). Two hand-spelled '○ Resolution history' literals now go through the shared marker helper.

Live at 235x52 and 100x30: both list reasons paint whole, the Sort one disappears in select mode, and the Export one disappears on the first check (layout-22, layout-23, layout-24, layout-25, layout-14).

Files: Library/library_shell_state.py, Widgets/Library/library_notes_canvas.py, Widgets/Library/library_notes_add_from_files_canvas.py, UI/Library_Modules/canvas_sync.py, Tests/Library/test_library_shell_state.py, Tests/UI/test_library_notes_w4_layout.py.

NOT done, deliberately: the brief's relocation of `_disabled_action_label` out of `library_note_import_canvas.py`. The sibling `sync-roots` branch is already written against it and does not switch, so relocating would break an unlanded branch for no behaviour change; the reason grammar is identical in both.
<!-- SECTION:NOTES:END -->
