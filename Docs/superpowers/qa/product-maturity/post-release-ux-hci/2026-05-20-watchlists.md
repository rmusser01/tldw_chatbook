# Post-Release UX/HCI Walkthrough Evidence: Watchlists

## Metadata

- Task: `TASK-60.2`
- Screen or workflow: Watchlists
- Date: 2026-05-20
- Branch: `codex/post-release-screen-functionality-audit`
- App command: textual-web on `127.0.0.1:8871` with isolated HOME/XDG profile
- Evidence method: textual-web/CDP with Playwright browser automation
- Actual screenshot path: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-20-screen-watchlists-long-wait.png`
- Screenshot approval: pending
- Reviewer: pending user approval

## User Goal

Open Watchlists, inspect local runs, and open a run or Console handoff if available.

## Steps Attempted

1. Opened Watchlists in the isolated profile.
2. Waited for the local Watchlists snapshot to resolve.
3. Captured the long-wait state after the screen remained in loading.

## What Worked

- Destination chrome renders and clearly separates list, run detail, and run inspector columns.
- Inspector states that no active run is selected.

## What Broke Or Slowed The Workflow

- The main content remained stuck on `Loading local Watchlists snapshot...` after an extended wait.
- The screen does not resolve into empty state or error state in a clean profile.
- Primary run review and Console handoff workflows cannot be attempted.

## Nielsen Norman Heuristic Findings

- Visibility of system status: poor; loading does not resolve.
- Match between system and real world: Watchlists/run language is clear once visible.
- User control and freedom: no retry or fallback appears.
- Consistency and standards: layout matches the shell, but behavior does not.
- Error prevention: no invalid run action is performed.
- Recognition rather than recall: no explanation for missing data appears.
- Flexibility and efficiency of use: blocked.
- Aesthetic and minimalist design: not enough usable content while stuck.
- Error recognition, diagnosis, and recovery: missing.
- Help and documentation: missing for loading failure.

## Keyboard And Focus Findings

- Not meaningfully testable while content is stuck loading.

## Empty Error Setup State Findings

- Expected empty local run state did not appear.

## Cross-Screen Handoff Findings

- Watchlist run-to-Console handoff is blocked by unresolved local snapshot loading.

## Power-User Repetition Findings

- Shortcuts: not enough usable content to validate.
- Batch actions: unavailable.
- State persistence: loading did not resolve.
- Recovery paths: missing.
- Repeated-use friction: blocking.

## Severity Decisions

| Finding | Severity | Follow-Up Task | Decision |
| --- | --- | --- | --- |
| Watchlists remains indefinitely loading and blocks run review workflow | P1 | `TASK-60.6` | Must be fixed or explicitly accepted before Watchlists can be accepted. |

## Acceptance Decision

- Accepted: no
- Reason: P1 loading blocker and pending screenshot approval.
- Required follow-up before acceptance: `TASK-60.6` plus user approval of a fixed rendered screenshot.
