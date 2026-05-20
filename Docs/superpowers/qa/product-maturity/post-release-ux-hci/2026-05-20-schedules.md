# Post-Release UX/HCI Walkthrough Evidence: Schedules

## Metadata

- Task: `TASK-60.2`
- Screen or workflow: Schedules
- Date: 2026-05-20
- Branch: `codex/post-release-screen-functionality-audit`
- App command: textual-web on `127.0.0.1:8871` with isolated HOME/XDG profile
- Evidence method: textual-web/CDP with Playwright browser automation
- Actual screenshot path: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-20-screen-schedules.png`
- Screenshot approval: pending
- Reviewer: pending user approval

## User Goal

Open Schedules, understand scheduled job availability, and identify how Console handoff works when a run exists.

## Steps Attempted

1. Opened Schedules in the isolated profile.
2. Reviewed empty schedule/run state and inspector recovery copy.

## What Worked

- Columns are clearly restored and visually separated.
- Empty state explains that no active scheduled run is selected.
- Inspector communicates Console context availability boundaries.

## What Broke Or Slowed The Workflow

- No populated schedule was available in the clean profile, so actual run launch/handoff remains unverified.

## Nielsen Norman Heuristic Findings

- Visibility of system status: good empty-state visibility.
- Match between system and real world: schedule/job language is clear.
- User control and freedom: no destructive actions exposed.
- Consistency and standards: follows destination column pattern.
- Error prevention: unavailable actions are blocked.
- Recognition rather than recall: recovery copy names the missing selected run.
- Flexibility and efficiency of use: needs populated workflow validation.
- Aesthetic and minimalist design: legible and calm.
- Error recognition, diagnosis, and recovery: empty recovery is clear.
- Help and documentation: enough for clean state.

## Keyboard And Focus Findings

- Not deeply validated in this pass.

## Empty Error Setup State Findings

- Empty state is visible and recoverable.

## Cross-Screen Handoff Findings

- Actual scheduled-run-to-Console path remains for `TASK-60.3`.

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
