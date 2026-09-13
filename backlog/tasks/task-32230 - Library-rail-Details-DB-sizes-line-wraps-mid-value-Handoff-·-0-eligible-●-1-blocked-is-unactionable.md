---
id: TASK-32230
title: >-
  Library rail Details: DB-sizes line wraps mid-value; 'Handoff · 0 eligible, ●
  1 blocked' is unactionable
status: Done
assignee: []
created_date: '2026-09-10 14:56'
updated_date: '2026-09-10 19:05'
labels:
  - library
  - rail
  - copy
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DB sizes line wraps across three rail lines mid-value; the Handoff line names a count and a colour dot but not what is blocked or how to unblock it. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 29.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DB sizes render one source per line
- [x] #2 The Handoff line names the blocked item and its remedy, or links to it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep every #library-details-db-sizes pin before changing the shape.
2. Failing test at 100 columns: one painted row per source, region.height == 1.
3. One row per source through a single shared row builder used by all three consumers.
4. Unit-test the Handoff label from the eligibility reason code; keep the 0-blocked case bare.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: the three sizes shared one Static joined by "·" and wrapped mid-value at the rail's 22-cell Details column ("Chats/Notes" on one line, its size on the next). Each size takes a row now. MEASURED, the label cannot ride along on the first of them either -- "DB sizes · Prompts 180.0KB" is 26 cells and wraps in exactly the same place -- so the label takes its own row and the values hang under it with a two-cell indent (11 cells, the align-under-the-value width the plan suggested, re-wraps the very value the split was meant to keep whole).

`library_db_size_rows` in library_rail.py is the single source of the id/label rule for the three consumers that must agree: rail compose, the rail's in-place patch, and the screen's Details-open refresh. `#library-details-db-sizes` stays the id of the first VALUE row -- the id every existing pin queries -- so the polling loops in the honesty and shell tests stay green untouched.

AC#2: "Handoff · 0 eligible, ● 1 blocked" named neither what was blocked nor how to unblock it. The row now reads "0 eligible · 1 blocked · not in this workspace · Link it from the conversation's header", built from the `reason_code` the workspace state already carries (`linkable_ineligibility_label`); a block linking cannot resolve falls back to `LIBRARY_GENERIC_WORKSPACE_BLOCK` plus the rule's own recovery sentence. Mixed item types degrade to "the item's header", plural to "them". The unblocked case is unchanged and grows no dot.

Two exact-string pins moved with the copy, both kept exact and neither loosened: the Details-open refresh pin now reads the block across its rows, and `test_post_release_workspaces_library_depth`'s handoff pin carries the new grammar (that file has 2 failures on dev unrelated to this change -- its seeded sources do not load -- so the new string was computed against the real function rather than observed in that test).

Known trade-off: a genuinely blocked Handoff row is prose and word-wraps over 3-4 rail lines at 34 cells. It only appears when something is blocked, and the fold cue from task-32219 covers the reachability cost.

Live at 235x52: label + "Prompts 156.0KB" / "Chats/Notes 5.1MB" / "Media 880.0KB", one line each; "Handoff · 0 eligible · 1 blocked · not in this workspace · Link it from the note's header" on the fresh profile.

Files: tldw_chatbook/Widgets/Library/library_rail.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_crit9_rail.py, Tests/UI/test_library_honesty_accessibility.py, Tests/UI/test_library_shell.py (helper), Tests/UI/test_post_release_workspaces_library_depth.py
<!-- SECTION:NOTES:END -->
