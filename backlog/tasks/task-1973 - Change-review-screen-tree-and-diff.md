---
id: TASK-1973
title: 'Change review: Review screen — changed-file tree, windowed diff viewer, turn history'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-1971
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dedicated screen (push_screen, Esc returns): header with turn selector (previous turns from change_snapshots), workspace/root labels, totals, honesty banners; left tree of changed files grouped Added/Modified/Deleted/Renamed (existing Tree widgets — no new deps) with per-file +a/−d and badges; right unified diff with syntax highlighting, rename detection (-M), binary rows as `Binary (2.1 KB → 3.4 KB)`, per-file line cap with explicit 'diff truncated — N more lines'. Windowed rendering: only the focused file's hunks are mounted, so a 50k-line generated file cannot freeze the screen. Keyboard-first: j/k files, Enter to diff, Esc back.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every group renders correctly for a turn containing an add, an edit, a delete, a rename, and a binary change (real-git fixture)
- [ ] #2 The diff pane renders markup=False/escaped — a file containing Rich markup or [brackets] displays verbatim
- [ ] #3 A file over diff_display_max_lines shows the truncation row with an accurate count
- [ ] #4 Only the focused file's diff is mounted (asserted via widget census on a many-file turn)
- [ ] #5 Turn selector navigates to a previous turn's diff
- [ ] #6 UI tests load the shipped stylesheet bundle and wait on conditions, not pause counts
- [ ] #7 All states legible in monochrome
<!-- AC:END -->
