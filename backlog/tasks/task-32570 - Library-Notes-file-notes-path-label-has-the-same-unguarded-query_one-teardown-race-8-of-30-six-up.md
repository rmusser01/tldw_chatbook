---
id: TASK-32570
title: >-
  Library Notes: #file-notes-path-label has the same unguarded query_one
  teardown race, 8 of 30 six-up
status: To Do
assignee: []
created_date: '2026-09-14 22:44'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evidenced twice in wave 4. Group 9 wrapped three unguarded query_one calls in library_file_notes_workspace.py (:2231/:2234/:2240) after proving the wrap narrowed its own failure mode from 6/30 occurrences and 2/30 failures to 0/30 across 30 six-up runs. The COMPOSITE failure ratio stayed flat at 8/30 because all eight residual failures come from a DIFFERENT site — #file-notes-path-label — whose query_one is still unguarded and raises NoMatches during teardown. It is the same signature as the four standing test_library_footer_focus reds on dev, and it fails on both trees, so it is pre-existing and not a branch effect. The repair is the pattern the file already uses at _render_session_git_label: acquire-or-return inside one try/except NoMatches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The #file-notes-path-label query_one is acquire-or-return, matching _render_session_git_label
- [ ] #2 Measured before and after by running the affected tests six-up in parallel over at least 30 runs, with both ratios reported
- [ ] #3 The four standing test_library_footer_focus reds on dev are re-measured against the same fix and either cleared or shown to have a different cause
<!-- AC:END -->
