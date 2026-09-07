---
id: TASK-1973
title: 'Change review: Review screen — changed-file tree, windowed diff viewer, turn history'
status: Done
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
- [x] #1 Every group renders correctly for a turn containing an add, an edit, a delete, a rename, and a binary change (real-git fixture)
- [x] #2 The diff pane renders markup=False/escaped — a file containing Rich markup or [brackets] displays verbatim
- [x] #3 A file over diff_display_max_lines shows the truncation row with an accurate count
- [x] #4 Only the focused file's diff is mounted (asserted via widget census on a many-file turn)
- [x] #5 Turn selector navigates to a previous turn's diff
- [x] #6 UI tests load the shipped stylesheet bundle and wait on conditions, not pause counts
- [x] #7 All states legible in monochrome
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD in `Tests/UI/test_change_review_screen.py` with the shipped-bundle CSS harness and condition waits: group rendering for A/M/D/R/binary from a REAL git fixture (rows produced by the real ChangeTurnTracker), markup-safety (a file containing Rich markup renders verbatim), truncation row with accurate count, widget census proving only the focused file's diff is mounted, turn-selector navigation, monochrome-legible states.
2. `UI/Screens/change_review_screen.py`: a plain pushed `Screen` (NOT BaseAppScreen -- no tab chrome; spec chose push_screen + Esc). Header (turn Select, workspace/root labels, totals), honesty banners (tracking_error rows), left Tree grouped Added/Modified/Deleted/Renamed/Other (verbatim-status bucket from 1970), right diff pane mounting ONLY the focused file -- built as a Rich Text with per-line diff coloring, never markup-parsed. Keys: j/k file nav, Enter -> diff pane, Esc -> dismiss.
3. Data through a small provider over (AgentRunsDB, ShadowRepoService, conversation_id) -- concrete implementation used in tests too (real git + real db, no fakes).
4. `css/components/_change_review.tcss` + bundle rebuild; heights explicit (the fr-inside-flex trap).
5. Truncation cap read from flat `[change_review] diff_display_max_lines` (Settings surface itself is TASK-1979).
6. Sabotage anything that passes first try. Per-language token highlighting inside hunks is deliberately NOT v1 -- diff-line coloring is the spec's v1 highlight; noted at closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`UI/Screens/change_review_screen.py` (plain pushed Screen + `AgentRunsChangeReviewProvider` + `ReviewTurn`), `css/components/_change_review.tcss` registered in build_css's ORDERED list (components are not auto-discovered), bundle rebuilt.

The provider is used by production AND the tests — fixtures produce turns through the real `ChangeTurnTracker` against real git and a real AgentRunsDB, per the fixture-invented-shapes rule. Windowing: exactly one `.change-review-diff-body` exists; selection swaps its content. Diff coloring is per-line `Text.append` with explicit styles — nothing is ever markup-parsed.

**New instance of the markup-as-data trap, caught by this task's own test:** Textual TREE LABELS parse markup when given as strings — the `[binary]` tag silently vanished, and a filename containing brackets would corrupt identically. Labels are now `Text` instances (rendered verbatim); the binary tag became `(binary)` and the group test pins a bracket-bearing label shape.

Per-language token highlighting inside hunks is deliberately not v1 — diff-line coloring is the spec's v1 highlight.

Three sabotages, each failing exactly its test: silent truncation, selector pinned to latest, all-diffs-mounted. 187 passed across the screen, turn-tracking and Workspaces suites.
<!-- SECTION:NOTES:END -->
