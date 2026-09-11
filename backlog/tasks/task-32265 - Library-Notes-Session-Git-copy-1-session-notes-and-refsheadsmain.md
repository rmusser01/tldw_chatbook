---
id: TASK-32265
title: >-
  Library Notes Session Git copy: 1 session notes, and refs/heads/main
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 12:00'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two copy defects on the flow that is otherwise the best-designed on the screen: "1 session notes will be committed" and "Committed 1 session notes as bd746be6..." do not pluralise, and the branch is displayed as `refs/heads/main` rather than `main`.

Both are on the pre-commit disclosure screen, which the design assessor called the most honest copy in the product; these are the only two places on it that read as machine output.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The session-note count pluralises correctly, including the one-note case
- [x] #2 The branch renders as its short name
- [x] #3 Covered by a test on the one-note case
<!-- AC:END -->


## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Find every place the session-note count and the branch are rendered (panel, commit form, commit review, push review, progress, commit receipt).
2. RED tests on the one-note case and on `refs/heads/...`.
3. Two named helpers, `_session_note_count` and `_branch_for_display`, and the receipt's own sentence fixed at its source in `file_notes_git_service`.
4. Live GREEN on the real commit flow.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Two named helpers instead of six ad-hoc expressions: `_branch_for_display`
(the `refs/heads/` strip the push panel already did inline, named) and
`_session_note_count`. Applied to the commit form meta, the commit review
branch and promise, the push review lead/counts/local-branch, the execution
progress line, and the repository `HEAD` label.

The receipt -- "Committed 1 session notes as bd746be6…" -- is built in
`file_notes_git_service._publish_*`, not in the panel, so it is fixed at its
source; the workspace's own progress line got the same treatment.

Live GREEN: `Branch: main · 1 session note staged` on the commit form,
"1 session note will be committed" on the review, "Committed 1 session note"
on the receipt.

Files: `Widgets/Library/library_file_notes_git_panel.py`,
`Notes/file_notes_git_service.py`,
`Widgets/Library/library_file_notes_workspace.py`,
`Tests/UI/test_library_file_notes_git.py`,
`Docs/User_Guide/library/file-notes.md`.

Residual (rider task-32475): the entry-focus rule can still take focus from
a chosen in-panel control when the anchor is hidden off the repair path.
<!-- SECTION:NOTES:END -->
