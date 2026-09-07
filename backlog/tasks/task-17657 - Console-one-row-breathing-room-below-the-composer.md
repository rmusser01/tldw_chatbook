---
id: TASK-17657
title: 'Console: one row of breathing room below the composer'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies:
  - task-17651
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner request 2026-08-17, after viewing the merged one-row composer (TASK-17651) live: give the composer one row of empty space below it so it visually stands out against the footer (and, in below-placement mode, the status row). Unlike the zero-information phantom row TASK-17650 deleted, this is a deliberate, pinned separation — Gestalt proximity: the composer groups with the conversation above it, not the status/footer cluster below. One row now; the owner may later ask for two.

Compact mode (terminal < 35 rows) drops the gap, following the established compact grammar that strips decorative rows where height is scarcest (same precedent as the tab strip's dropped margin).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 One blank row renders between the composer and the widget below it (footer in above-placement mode; status row in below mode), pinned by a painted assertion at 150x44
- [x] #2 Compact mode (`-console-compact`) drops the gap, verified on the running screen
- [x] #3 Existing bottom-stack contract tests stay green (adjacency pins updated to the deliberate gap); bundle rebuilt from source
- [x] #4 User Guide Console page stamp refreshed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: update the single-separator contract pin — footer sits one row below the composer, and the row between them paints blank; run and watch fail.
2. `#console-native-composer` margin 0 -> `0 0 1 0` in `_agentic_terminal.tcss` (comment marking it deliberate, vs. the TASK-17650 phantom); `-console-compact` override restores margin 0.
3. Rebuild bundle; GREEN; targeted bottom-stack suites; live probe both placements + compact.
4. Docs stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One CSS declaration: `#console-native-composer` margin `0` -> `0 0 1 0` in `_agentic_terminal.tcss` (bundle rebuilt), with a `-console-compact` override restoring `margin: 0` — the same compact grammar as the tab strip's dropped margin. The comment at the rule marks the row as DELIBERATE (owner call, Gestalt grouping with the conversation) to distinguish it from the TASK-17650 phantom this programme deleted.

TDD: the single-separator contract pin was updated first and watched RED (footer adjacent), then GREEN with the painted blank-row assertion (`render_strips` row must be whitespace). Live probes: above mode composer y41 / BLANK y42 / footer y43; below mode composer y40 / BLANK y41 / chips y42; compact 150x30 flush (no gap). Transcript region yields exactly one row (33 -> 32 at 150x44). Bottom-stack sweep green.

Files: `css/components/_agentic_terminal.tcss` (+ bundle), `Tests/UI/test_console_composer_collapse.py`, `Docs/User_Guide/console.md` (stamp).
<!-- SECTION:NOTES:END -->
