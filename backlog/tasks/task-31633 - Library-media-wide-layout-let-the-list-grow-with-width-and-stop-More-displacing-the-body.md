---
id: TASK-31633
title: >-
  Library media wide layout - let the list grow with width and stop More
  displacing the body
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 06:18'
updated_date: '2026-09-06 19:03'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P1: at 235x52 the Items list is 38 cells and truncates a 98-character title while at 100x30 it is about 47 cells and fits; two 5-cell gutters flank it; the Reader lays 83 characters into a 145-cell frame; each item costs three rows; opening More pushes the tab row and the body down about 19 rows. The Items-pane floor was set for the collapse case and never told to grow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 235x52 the list column is at least as wide as at 100x30 and a 98-character title fits or truncates later
- [x] #2 No 5-cell dead gutter remains between rail, list and Reader
- [x] #3 More renders without displacing the Reader body
- [x] #4 Painted tests pin the widths at both sizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 1: opt-in `list_grows` on the shared adaptive-reader profile (Media only); the resolver moves half the Reader's surplus above its comfort width to the Items column, capped at the profile max; all four profiles pinned byte-identical at 100 and 235 before the change, only Media at 235 may differ; painted pins.
2. Task 2: close the 5-cell gutters between rail, list and Reader; drop the per-item spacer row (two rows per item); re-pin PR D/F/G row positions honestly.
3. Task 3: the Reader's More actions render as one row under the toolbar (`More ▴` while open) instead of a Vertical that pushed the body ~19 rows; focus stays on More.
4. SDD per task (review + carried minors), final whole-branch review + fix round, PR H.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Critique #5 P1 (measured): at 235x52 the Items list was 38 cells and truncated a 98-character title to 30 characters while at 100x30 it was ~47 cells and fit; two 5-cell dead gutters; a Reader laying 83 characters in a 145-cell frame; three rows per item; More displaced the body by ~19 rows. Task 1: opt-in `list_grows` on the shared profile (Media only) moves half the Reader's surplus above `max(work_min_width, READER_COMFORT_WIDTH)` to the Items column, capped at the profile max — Media at 235: Items 40→56 (cap = comfort width, per review), title 31→46 chars, Reader keeps >92-char measure; growth skipped under custom widths; the three sibling profiles byte-identical at 100/235/library-open edges (pinned before the change; resolver tests 60→326). Task 2: the 5-cell gutters WERE the pane grips (resolver reserves what they paint; no CSS margin involved) → per-profile `grip_width` (Media 1, default 5, glyph ‹/› at one cell); the third row per item was `margin: 0 0 1 0` on `.library-media-row` (no spacer widget) → 15/15 items visible at 235x52 (was 11), stride 2, 1-cell gutters; the 8 freed cells go to the Reader at 235 (135→143) and to the list at 100x30 (44→52, Reader 46 unchanged) — the ONLY 100x30 movement. Consequence pinned, not hidden: the open-all-three threshold moves 120→112 (band 112–119: rail beside a 40–43-cell list instead of no rail + 56) — ruled acceptable as the resolver's ordinary threshold policy, edge pinned at 111/112. Task 3: More = one `ItemGrid` row `#library-media-reader-more-actions` (a Horizontal clipped 'Move to trash' at 100x30): +1 row at 235x52 / +2 at 100x30 (was +19 / +9); `More ▴` while open; focus target via `viewer.queue_after_recompose` (a `call_after_refresh` focused the about-to-be-detached button → orphan focus that swallowed every key; lesson).
Trade-offs: the growth rule is opt-in per profile so Conversations/Skills/Collections keep today's geometry (pinned); the 36-cell Items-pane floor is unchanged (PR D/F/G pins re-pinned to the new honest positions, never deleted); the 100x30 composition — the good one — moves only by the 8 grip cells; sibling grips stay 5 cells (follow-up filed).
Verification: resolver tuples for all four profiles at 100/235 before/after; painted widths and gutters at 235x52 and 100x30; rows-per-item with 15 seeded items; More displacement at both sizes; sibling-surface UI test files; live tmux with a 98-character title at both sizes.
Files: tldw_chatbook/Utils/adaptive_reader_state.py, the Media reader profile construction, tldw_chatbook/Widgets/Library/library_media_canvas.py, tldw_chatbook/Widgets/Library/library_media_viewer.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ generated sheet/bundle), tests under Tests/Utils and Tests/UI (adaptive_reader_shell, render_fixes, reader_shell, toolbar_adapt, multiselect), Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
