---
id: TASK-2236
title: 'Library: rail subtitles fit a real width budget (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 20:37'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F-013 plain-language subtitles truncate into noise ('imported…', 'saved…') at real rail widths (~24-31 cells). Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Subtitles rewrite to fit (~16 cells) or drop below a width threshold instead of mid-word cutting,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (gloss copy + fitting rule; count-protection machinery unchanged). Analysis: at the rail's realistic content widths (17 cells at 100x30, ~25 at 170x50) the F-013 glosses (17-29 cells) could only ever render as word-cut noise. Fix has two parts: (a) rewrite all five glosses to fit content width 25 with title+count present: Media 'your files' (10), Prompts 'AI asks' (7), Skills 'AI add-ons' (10), Collections 'item sets' (9), Search / RAG 'find all' (8); (b) the F-015 fitting becomes full-gloss-or-drop -- a partial gloss is noise (the review's exact complaint), so when the full gloss doesn't fit, it drops entirely; title-still-truncates-before-count protection stays. Steps: 1. RED: shell-state subtitle pins; rendered test at 170x48 asserts the FULL gloss renders dim; 100x30 test asserts glosses drop with no partial '—' noise and counts survive. 2. library_shell_state.py gloss strings; library_rail.py _row_label drops the word-cut branch. 3. Guide line update. 4. Run shell-state/shell/rail tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two parts: (a) all five F-013 glosses rewritten to fit content width 25 with title+count present -- Media 'your files', Prompts 'AI asks', Skills 'AI add-ons', Collections 'item sets', Search / RAG 'find all'; (b) the F-015 fitting became full-gloss-or-drop: a partial gloss is exactly the noise the round-2 review flagged, so when the full gloss doesn't fit it drops (title truncation for count protection unchanged). Files: library_shell_state.py (gloss strings), library_rail.py (_row_label drops the word-cut branch), Tests/Library/test_library_shell_state.py (subtitle pins), Tests/UI/test_library_shell.py (F-013 pin now asserts the FULL dim gloss renders at 170x48; new 100x30 test pins no-partial-fragment + count survival), Docs/User_Guide/library.md. Verified: 8 targeted tests RED->GREEN; full shell+rail+state suite 362 passed. Ruff clean (1 pre-existing F401 in test_library_shell.py untouched). ADR: not required (gloss copy + presentation rule). Commit 0c20b8ec7.
<!-- SECTION:NOTES:END -->
