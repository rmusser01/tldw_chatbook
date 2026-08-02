---
id: TASK-1980
title: 'Change review: live end-to-end verification (real app, real agent run)'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - change-review
  - verification
dependencies:
  - TASK-1972
  - TASK-1973
  - TASK-1974
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The programme's standing lesson: green tests are not a usable feature. Drive the REAL app in tmux with an isolated TLDW_CONFIG_PATH profile: register a scratch root, run a real agent turn that creates+edits+deletes files (including one via a script side effect), read the summary row, open the Review screen, read each diff, revert one file, Undo-all a turn — at 80×24 and 212×64. File defects found as tasks before closing.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The full journey above completes on the live app at both sizes, evidenced by captured panes
- [ ] #2 The script-side-effect file appears in the review (with its TASK-1978 badge if merged)
- [ ] #3 Reverted files verified on DISK, not just in the UI
- [ ] #4 Any defect found is filed as a backlog task and linked here
<!-- AC:END -->
