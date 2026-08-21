---
id: TASK-19639
title: Keep Console workspace geometry inside the viewport at exactly 100 columns
status: In Progress
assignee: []
created_date: '2026-08-20 07:07'
labels:
  - console
  - ux
dependencies:
  - TASK-18912
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the reproduced exact-100-column Console rail state from expanding the grid far beyond the viewport so every supported persona retains a visible transcript and usable rail controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 At exactly 100x30, all four stored Context/Inspector open-state combinations keep every displayed Console workspace child within the viewport; the default Context-only state gives `#console-main-column` at least 55 allocated columns and every state gives `#console-native-transcript` at least 40 readable content columns excluding borders and padding.
- [x] #2 Effective rail priority, compact overrides, responsive focus handoff, selected-message and transcript-anchor state, stored preferences, and the 70/74-column usable-transcript floors remain consistent with ADR-043; cold start and responsive resize make no preference-save call.
- [x] #3 Existing geometry and access behavior at 80x24, 120x30, 160x45, and 235x52 do not regress.
- [x] #4 Production-CSS Textual compositor regressions prove the exact-100 cold-start failure and the 101→100 live-resize failure, cover both directions across 99/100/101, and are mutation-checked independently against the inclusive override and exact-100 resize-boundary corrections.
- [x] #5 The change is terminal-specific and does not alter TASK-18911's phone, pointer, hover, or served-browser ownership.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-20-task-18913-console-exact-100-column-containment-design.md`.
<!-- SECTION:DESIGN:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing pure-state tests for the 99/100/101 minimum-waiver and resize-band boundaries.
2. Implement the inclusive exact-100 waiver and distinct semantic resize band in `console_rail_state.py`.
3. Add production-CSS four-state cold geometry coverage with separate 55-column allocation and 40-column readable-content assertions.
4. Add all four adjacent live-resize directions with focus, selection, anchor, and zero-preference-save assertions.
5. Correct stale width arithmetic and compact-override comments without changing other rail behavior.
6. Mutation-check both corrections independently, run focused/full verification, and record evidence before closing the task.

Detailed plan: `Docs/superpowers/plans/2026-08-20-task-18913-console-exact-100-column-containment.md`.

ADR required: no

ADR path: `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`

Reason: direct defect correction within ADR-043's existing minimum-width-waiver and responsive priority mechanism; no persistence or application-structure decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Corrected the Context compact boundary so the preferred default remains effectively open at exactly 100 columns and receives layout-minimum-waiver authority, while widths below 100 still collapse the default without rewriting preference. Added a distinct exact-100 resize band and cleared stale Context waiver authority when Inspector wins rail priority. The persisted-preference, explicit-toggle, 70/74 usable-transcript-floor, and ADR-043 priority contracts are unchanged.
- Production-CSS compositor coverage separates the two geometry metrics required by the design: the default Context-only state allocates at least 55 columns to `#console-main-column`, while the native transcript retains at least 40 readable content columns in every state. The cold matrix covers Context-only, both closed, Inspector-only, and the both-open Inspector-priority conflict at first paint and settled state. Live coverage exercises all four adjacent transitions (101→100, 100→101, 99→100, and 100→99) and preserves responsive focus handoff, selected message, detached/tail-follow state, raw anchor flags, reading offset, stored rail intent, and zero preference-save calls.
- Modified functionality files: `tldw_chatbook/Chat/console_rail_state.py`, `Tests/Chat/test_console_rail_state.py`, and `Tests/UI/test_console_shell_regions.py`. `tldw_chatbook/UI/Screens/chat_screen.py` received geometry/ADR comment corrections only. No CSS was changed by the committed task implementation; the CSS-integrity check's generated timestamp rewrite was restored before final review.
- Mutation evidence from committed work: removing the inclusive waiver made the cold matrix 1 failed/3 passed before green 4; the compose-only first-paint mutation exposed the old oracle's false-green 4, then the strengthened oracle produced red 3/1 and green 4; removing the exact-width band made 101→100 fail before green 4; sabotaging transcript population made 101→100 fail with `max_scroll_y == 0 < 2` before green 4.
- Fresh scoped verification: import provenance `1 passed, 1 warning in 0.37s` and resolved `tldw_chatbook/__init__.py` to this worktree; focused rail/shell/resize/compact-access/width-budget behavior `136 passed, 2 warnings in 149.12s`; production CSS build integrity `10 passed, 1 warning in 1.53s`; `git diff --check` passed. AST comparison from `52f03ef5c` to `HEAD` was identical for all four scoped Python files, proving the subsequent correction commits were comment-only.
- Exact four-file static evidence is not green: `ruff check` exits 1 for the pre-existing unused `inspect` import at `chat_screen.py:8` (`F401`, one fixable error); `ruff format --check` exits 1 because it would reformat `console_rail_state.py` and `chat_screen.py` (two test files already formatted). These baselines were recorded without suppression or out-of-scope edits.
- ADR required: no. Reused ADR-043 (`backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`) because the correction stays inside its responsive-priority and persistence boundary. No new reusable incident was produced beyond existing testing/backlog lessons, so no lessons entry was added.
- Verification was intentionally limited to changed functionality at user direction; repository-wide governance, architecture, and full-suite gates are not part of this evidence. Status remains **In Progress** despite all acceptance criteria being supported because the repository Definition of Done requires green linter/formatter gates, and the exact scoped Ruff commands still reproduce the baselines above.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-18913, colliding with the older
"Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns" task that arrived on dev first.
Per the owner rule decided 2026-08-21 in TASK-19601 (**older id keeps it;
the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-19639. Citations to TASK-18913
in already-merged commit messages, ADRs, or code comments written before
2026-08-21 refer to THIS task; the other TASK-18913 holder is the
older arrival and keeps the id.
