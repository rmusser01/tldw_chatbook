# Post-Release UX/HCI Walkthrough Evidence: Workflows

## Metadata

- Task: `TASK-60.2`
- Screen or workflow: Workflows
- Date: 2026-05-20
- Branch: `codex/post-release-screen-functionality-audit`
- App command: textual-web on `127.0.0.1:8871` with isolated HOME/XDG profile
- Evidence method: textual-web/CDP with Playwright browser automation
- Actual screenshot path: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-20-screen-workflows.png`
- Screenshot approval: pending
- Reviewer: pending user approval

## User Goal

Open Workflows, inspect run availability, and understand how a workflow run can stage Console context.

## Steps Attempted

1. Opened Workflows in the isolated profile.
2. Reviewed empty workflow/run state and inspector copy.

## What Worked

- Three columns are clearly delineated.
- Empty state explains no workflow run is selected.
- Console handoff is honestly blocked until adapter context exists.

## What Broke Or Slowed The Workflow

- No populated workflow run was available in the clean profile, so launch/handoff remains unverified.

## Nielsen Norman Heuristic Findings

- Visibility of system status: good empty-state visibility.
- Match between system and real world: workflow/run language is clear.
- User control and freedom: no dead destructive controls exposed.
- Consistency and standards: follows the destination column shell.
- Error prevention: Console handoff is blocked until a run exists.
- Recognition rather than recall: missing run prerequisite is visible.
- Flexibility and efficiency of use: needs populated workflow validation.
- Aesthetic and minimalist design: readable.
- Error recognition, diagnosis, and recovery: blocked state explains why.
- Help and documentation: enough for empty state.

## Keyboard And Focus Findings

- Not deeply validated in this pass.

## Empty Error Setup State Findings

- Empty state is visible and recoverable.

## Cross-Screen Handoff Findings

- Actual workflow-run-to-Console path remains for `TASK-60.3`.

## Power-User Repetition Findings

- Shortcuts: global shortcuts remain visible.
- Batch actions: not available in empty state.
- State persistence: not validated.
- Recovery paths: clear selected-run prerequisite.
- Repeated-use friction: cannot judge until populated.

## Severity Decisions

| Finding | Severity | Follow-Up Task | Decision |
| --- | --- | --- | --- |
| Populated run handoff unverified | P1 | `TASK-60.3` | Validate in cross-screen workflow pass. |

## Acceptance Decision

- Accepted: no
- Reason: screenshot approval is pending and populated handoff is unvalidated.
- Required follow-up before acceptance: user approval plus `TASK-60.3` run-handoff evidence.
