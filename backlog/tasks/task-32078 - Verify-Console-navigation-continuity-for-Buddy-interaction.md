---
id: TASK-32078
title: Verify Console navigation continuity for Buddy interaction
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:27'
updated_date: '2026-09-08 21:03'
labels:
  - console
  - buddy
dependencies: []
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify actual reusable Console navigation before adding cross-screen Buddy interaction; preserve accepted work and explicit cancellation boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mounted navigation preserves streaming, queues and pending decisions; explicit shutdown still cancels.
- [x] #2 Document current reuse semantics and correct obsolete user-facing cancellation claims using targeted tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md; backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: implement approved answerable-time and exact-owner boundaries while preserving TASK-31520 reusable Console navigation.
1. Read reusable-route implementation and relevant lifetime tests.
2. Exercise actual streaming, queued continuation and all three approval bridges through mounted navigation and covering modals; cover all five human decision kinds at the shared clock boundary.
3. Fix hidden attachment-as-visibility notices and finite answerable-time clocks; preserve explicit Stop, final unmount and shutdown.
4. Resolve Canvas selection for an accepted turn from its exact live runtime session and branch, separately from browser active-view authorization; reject mismatched and closed owners. This narrow correction follows the mounted queue failure and the approved ADR-121/139 execution-owner boundary.
5. Correct obsolete user guide copy and run targeted mounted, interrupt, Canvas and runtime lifetime checks; record evidence and limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved TASK-31520 retained Console navigation. Suspend/resume now informs the interrupt host of actual visibility (including covering modals), so hidden requests surface one app-wide notice and finite budgets count only time when their owning session and FIFO head are answerable. All five interrupt kinds share the clock; explicit Stop, final-unmount and shutdown fences remain intact.

The mounted queued-turn test exposed Canvas run admission depending on the browser active-view guard after another conversation was selected. Added an exact live runtime session/conversation/branch resolver used only by run selection capture; browser interaction guards remain unchanged and wrong/closed owners fail closed. Implements existing ADR-094 answerable-time and ADR-121/139 exact execution ownership, with the current screen reuse ruling recorded in the foundations plan.

Changed controller visibility/notification methods, interrupt host, runtime Canvas resolver, native Canvas authority and screen suspend/resume hooks. Added mounted navigation/decision and exact-owner Canvas tests, plus deterministic decision-clock coverage; updated existing headless tests and the stale risk-floor fixture to the current call-id/profile API. Corrected Console, index, tools/runs and Watchlists guide copy and recorded the accepted-work verification lesson.

Validation used the shared main .venv Python with cwd and PYTHONPATH explicitly set to .worktrees/buddy-console-management and isolated Tests fixtures:
- Tests/UI/test_console_navigation_decisions.py: 6 passed in 87.63s, covering actual stalled gateway work, queued continuation without retargeting, explicit Stop/shutdown, and all three approval bridges through Home and a modal. The added already-armed-before-navigation scenarios were then verified with -k hidden_finite: 3 passed, 3 deselected in 39.55s.
- Tests/Chat/test_console_decision_clock.py, test_console_interrupt_rounds.py, test_console_interrupt_attention.py; Tests/Canvas/test_hidden_run_scope.py, test_native_authority.py, test_canvas_kill_switch.py, test_startup_deferral.py: 110 passed in 18.31s.
- Tests/UI/test_console_headless_approval.py, Tests/Chat/test_console_runtime_lifetime.py, Tests/UI/test_console_screen_reuse.py: 44 passed in 75.33s.
- Final small clock/visibility changes: decision_clock, interrupt_rounds, interrupt_attention and hidden_run_scope files: 59 passed in 5.16s.
- New test files pass Ruff lint/format; changed definitions pass Ruff range-format checks; comparison to HEAD finds zero introduced lint diagnostics; git diff --check passes. Full-file lint on the large existing files retains unrelated legacy findings.

Self-reviewed exact-owner guards, cancellation ordering, lazy Canvas construction and shared-file edits. No schema, dependency or license changes in this task. Evidence is mounted Textual with a deterministic fake provider plus real controller/store/SQLite paths; no real-provider or audible-terminal run, full suite, restart-resumption, or fresh-screen redesign is claimed.

Review follow-up: legacy run_round(session_id=None) deliberately mounts without FIFO parking. Added a requires_head flag so its visible card still consumes the configured budget; explicit session rounds retain FIFO eligibility. All five legacy round kinds reproduced the regression before the fix. Targeted clock/authority gate: 59 passed in 5.23s (/private/tmp/console-legacy-clock-green.log). Buddy interaction extends the same host with per-kind Buddy visibility claims; unsupported worktree review links do not consume answerable time.
<!-- SECTION:NOTES:END -->
