---
id: TASK-32815
title: Guard Console footer focus queries during empty-stack shutdown
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 18:40'
updated_date: '2026-09-18 19:12'
labels:
  - ui
  - console
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32813 consumer qualification observed a Console archive case complete its interaction assertions and then fail during shutdown: a deferred footer refresh called _console_rail_focus_active after the final screen had been popped. The same exact case passed on isolated rerun. The implicated ChatScreen helper is unchanged from the saved head; this is a sibling path to TASK-32297, which guards the environment poll.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deferred footer focus checks treat an empty screen stack as inactive and do not raise during shutdown.
- [x] #2 A deterministic regression reproduces the observed footer path without depending on a timing-sensitive full-app failure.
- [x] #3 Targeted footer and archive checks pass, with the original incident and remaining lifecycle limits recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain TASK-32813 original archive teardown failure and trace setup refresh through footer registration to App.focused. 2. Reproduce deterministically with a mounted Console, a real temporarily empty Textual screen stack and preserved live-focus controls. Cover both footer focus queries and the observed setup-block callback. 3. Treat only ScreenStackError as inactive focus; preserve ordinary rail/composer hint behavior. 4. Run serial private-profile footer/archive/environment checks, obtain bounded independent review, document evidence and save to draft PR2707. ADR required: no. ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md and existing TASK-32297 lifecycle precedent. Reason: Routine bug fix at existing focus-query boundary; no new architecture, styling, keybinding or persistence change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both Console footer focus helpers now catch only Textual ScreenStackError from App.focused and report inactive focus after the last screen is popped. No style, shortcut, persistence or normal live-focus behavior changes. Three real mounted regressions fail before the fix and pass afterward, including the exact late _apply_console_setup_block(False) path, real stack restoration and rail/composer hint continuity.

Docs/superpowers/qa/2026-09-18-console-footer-shutdown/README.md records 12 passing serial private-profile cases and independent review with no actionable findings. Two additional archive restore/send cases fail on both saved source 2cc702b85c and fixed code because CapturingGateway lacks cached_context_window; TASK-32817 retains these as separate fixture debt, not passing coverage. Original TASK-32813 failure is linked. No full suite or fresh native capture is claimed for this nonvisual exception guard. New test has no Ruff findings, existing ChatScreen diagnostic count is unchanged, range/new-file format checks pass and authored diff passes whitespace checks.

ADR required: no; ADR-031 and TASK-32297 govern the existing footer/lifecycle behavior. Modified ChatScreen, added the focused regression file, and updated QA/component ledgers. No new generalized lesson beyond the existing empty-stack lifecycle precedent. PR2707 remains draft and requires its own visual review and merge approval.
<!-- SECTION:NOTES:END -->
