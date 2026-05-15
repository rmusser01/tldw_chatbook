# Phase 4.6 Schedules And Workflows Run Control

Date: 2026-05-15
Branch: `codex/phase4-6-schedules-workflows-control`
Backlog task: TASK-11.6
Screens: Schedules, Workflows

## Scope

Make Schedules and Workflows honest run-control surfaces without pretending the underlying run-control services exist:

- Schedules failed-run recovery derives list detail inspector and action copy from the same active-work item.
- Workflows pending-approval recovery derives detail inspector approval and next-action copy from the same active-work item.
- No-active-run states explain why Console follow or launch is blocked and what to select or start next.
- Retry pause and approval-review actions are visible but disabled with explicit service-not-wired tooltips.
- Console follow or launch continues through the existing Home active-work and Console handoff seams.

## Automated Evidence

- Command: `python -m pytest -q Tests/UI/test_destination_shells.py::test_schedules_empty_state_reads_as_live_queue_with_recovery_path Tests/UI/test_destination_shells.py::test_workflows_empty_state_reads_as_live_queue_with_recovery_path --tb=short`
- Result: `2 passed, 1 warning in 6.68s`
- Command: `python -m pytest -q Tests/UI/test_destination_shells.py::test_schedules_failed_run_exposes_consistent_retry_control_state Tests/UI/test_destination_shells.py::test_workflows_approval_pending_run_exposes_review_before_console_state Tests/UI/test_destination_shells.py::test_schedules_empty_state_reads_as_live_queue_with_recovery_path Tests/UI/test_destination_shells.py::test_workflows_empty_state_reads_as_live_queue_with_recovery_path 'Tests/UI/test_destination_shells.py::test_automation_destination_wrappers_explain_ownership[schedules-expected_sections1]' 'Tests/UI/test_destination_shells.py::test_automation_destination_wrappers_explain_ownership[workflows-expected_sections2]' Tests/Home/test_active_work_adapter.py --tb=short`
- Result: `26 passed, 8 warnings in 14.92s`
- Command: `git diff --check`
- Result: passed

## Screenshot Evidence

- Capture method: textual-web on localhost using Playwright-controlled Chromium, following the project CDP/browser QA workflow.
- Viewport: `2050x1240` browser viewport, device scale factor `1`.
- Schedules screenshot: `Docs/superpowers/qa/product-maturity/phase-4/schedules-run-control-2026-05-15-polish.png`
- Workflows screenshot: `Docs/superpowers/qa/product-maturity/phase-4/workflows-run-control-2026-05-15-polish.png`
- User approval: approved
- Notes:
  - The approved Schedules capture shows top navigation, filter row, three clearly separated columns, no-active-run recovery copy, blocked Console state, and disabled retry pause approval controls.
  - The approved Workflows capture shows top navigation, modes row, three clearly separated columns, no-active-run recovery copy, blocked Console state, and disabled retry pause approval controls.

## QA Walkthrough Notes

- Schedules failed-run recovery shows `Status: failed`, `State: failed`, retry/backoff copy, `Run control: retry available`, and `Next action: retry or open in Console` while preserving the enabled Console follow seam.
- Workflows pending-approval recovery shows `Status: pending_approval`, `Approvals: pending`, `Run control: approval required`, and `Next action: review approval before Console follow` while preserving the enabled Console launch seam.
- Empty Schedules and Workflows states now read as live queues with count rows, selected-run absence, recovery instructions, and disabled controls that explain the missing active run.

## Residual Risks

- This slice does not implement real retry pause resume or approval-review services. The controls remain disabled until those services exist.
- Console follow and launch are verified through existing handoff seams, not through live Schedules or Workflows backend execution.
- Future resizable panes should build on the explicit column widths and visible separators added here; no drag behavior is implemented in this slice.
