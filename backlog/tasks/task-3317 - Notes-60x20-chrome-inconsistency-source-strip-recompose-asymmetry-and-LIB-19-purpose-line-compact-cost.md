---
id: TASK-3317
title: >-
  Notes 60x20 chrome inconsistency: source strip renders only after a full
  recompose, and the LIB-19 purpose line eats 4 of 10 compact list rows
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 09:45'
updated_date: '2026-08-11 19:26'
labels:
  - library
  - notes
  - ux
dependencies: []
priority: medium
---

## Implementation Plan

ADR required: no
ADR path: N/A (conforms to ADR-011 and ADR-015)
Reason: This is a compact presentation and route-chrome consistency correction within the existing Library Notes ownership and responsive boundary; it introduces no new storage, service, or long-lived application boundary.

1. Pin the current 60x20 list and Create chrome in failing mounted geometry tests.
2. Keep the Database | Files source-authority strip mounted for every Database Notes route, including Create.
3. Hide the supportive Database-purpose sentence in compact mode while preserving the full copy at wide widths and across live breakpoint transitions.
4. Collapse the task-3315 per-route source-strip test fork to one deterministic compact contract and restore the Notes list row budget.
5. Run focused Library shell and responsive Notes tests, static checks, a mutation check of each guard, and self-review the scoped diff.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while repairing `Tests/UI/test_library_shell.py` against the merged surface (task-3315). Two product observations at 60x20 that deserve an owner ruling rather than a silent test pin:

1. **Per-route chrome asymmetry (nondeterministic-looking UI).** The one-row Database|Files source strip (`#library-notes-source-strip`, file-notes workspace `b83852eda`) is composed by the screen only when a FULL screen recompose runs while the notes canvas is selected. Entering Notes by pressing the rail row goes through `_replace_library_browse_canvas` (PR #1439's fast path, `03cd682df`), which swaps the canvas in place — so the plain list view shows NO strip (shell 15 rows), while entering the editor/sync/loading views forces a full recompose and the strip appears (shell 14 rows). The same logical "Notes selected" state renders different screen chrome depending on which internal update path ran last. Task-3315 pinned this per-route truth in `_assert_task8_compact_chrome` (test file names this task); the product should either always render the strip for notes routes or never render it at compact.

2. **LIB-19 purpose line's compact cost.** `#library-notes-database-purpose` (task-2858, `a3591b503`) is styled as a "muted one-line treatment", but at width 60 it wraps to 3 rows + 1 margin — 4 of the compact Navigator's 10 list rows. The notes-adaptive 60x20 program (PR #1439) and the LIB-19 copy (PR #1420) merged a day apart and were never reconciled; a compact-mode treatment (e.g. hide or truncate the sentence when `_library_notes_compact`) would restore the reviewed 60x20 budgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The source strip's presence for notes routes is deterministic across entry paths (always or never per terminal class), with the chosen direction recorded
- [x] #2 The database-purpose sentence has an explicit compact-mode treatment (kept, truncated, or hidden) chosen by the owner, not by layout accident
- [x] #3 `Tests/UI/test_library_shell.py`'s task8 pins are updated to the single ruled truth (removing the per-route `source_strip` fork task-3315 had to pin)
<!-- AC:END -->

## Update — AC#1 closed dev-side (2026-08-09)

Observation 1 was fixed independently on dev by `d1df7d0a7` (TASK-13213,
"restore file notes source access"), which lands the fix in the direction this
task asked for: `_replace_library_browse_canvas` now refuses its targeted swap
whenever the mounted contextual chrome disagrees with the destination
(`notes_source_strip_mounted != (shell.canvas_kind == "notes")`), so entering
Notes by rail press goes through the full recompose and shows the strip exactly
like the editor/sync/loading routes. Every Database-notes route now settles at
3 + 1 + 1 + 14 + 1 at 60x20.

Confirmed empirically while rebasing the media-ingest follow-up branch onto dev
`f6911b37b`: the 60x20 `normal` case and the navigator `compact_surplus` case
both fail their old strip-less pins against a `git archive` extraction of dev's
product tree, i.e. the change is dev's, not the arc's. The pins were re-measured
in task-3315's round-2 addendum.

AC#2 is untouched (the LIB-19 sentence still costs 4 of the compact list's rows;
at 60x20 the navigator list is now 5). AC#3 is only partly satisfied: the
per-route `source_strip` fork survives for the create-note canvas
(`canvas_kind "notes-create"`), which still never composes the strip.

## Implementation Notes

- Chose the compact treatment explicitly: the Database-purpose sentence is hidden below the existing 120-column Notes breakpoint and restored on the same mounted canvas at wide widths. This returns four 60x20 rows to the primary list without discarding the explanatory copy.
- Extended the route-owned source-strip contract to both `notes` and `notes-create`, so List, Editor, Sync, loading, and Create all retain the visible `Database | Files` authority control and the same 14-row compact canvas allocation.
- Simplified the task-8 geometry helpers to one source-strip truth, restored the 60x20 navigator list from 5 to 9 rows (selection from 6 to 10), and added a compact/wide/compact identity-preserving regression.
- Verification: 24 responsive Library shell tests passed; 2 `LibraryNotesCanvas` widget tests passed; both new guards were mutation-checked and failed when individually removed, then the two focused regressions passed again after restoration. Scoped Ruff checks passed for production and test changes (the test-file run excluded its pre-existing unrelated F401 import).
- Qodo review remediation: introduced canonical `LIBRARY_CANVAS_KIND_NOTES_CREATE` ownership in `library_shell_state.py` and replaced every behavior-critical `notes-create` literal in the shell builder and Library screen. The 30-test shell-state suite, both focused mounted-UI regressions, and scoped Ruff checks passed after the change.
- ADR: no new ADR; the change conforms to ADR-011 and ADR-015. Documentation is complete in this task record; no reusable incident warranted a new lessons entry.
- Modified files: `tldw_chatbook/Library/library_shell_state.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `Tests/Library/test_library_shell_state.py`, `Tests/UI/test_library_shell.py`, and this task record.
