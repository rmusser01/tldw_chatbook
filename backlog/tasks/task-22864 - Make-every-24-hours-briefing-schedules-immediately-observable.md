---
id: TASK-22864
title: Make every-24-hours briefing schedules immediately observable
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-28 01:21'
labels:
  - watchlists
  - scheduling
  - console
  - briefings
dependencies:
  - TASK-22863
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-console-watchlists-commands-and-operations.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
  - backlog/decisions/019-watchlist-scheduler-migration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Store interval-based briefing cadence, wake the running scheduler, and return an honest receipt that distinguishes reload request from acknowledgement and global disablement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console-only schedule command supports `every_12_hours`, `every_24_hours`, `every_7_days`, `off`, and advanced 3,600–2,678,400-second intervals; “Every 24 hours” stores exactly 86,400 seconds.
- [ ] #2 A never-attempted enabled schedule is immediately eligible; later eligibility is the latest attempt plus interval, including after failure, and overdue work runs when the app/scheduler resumes.
- [ ] #3 `off` clears only cadence and preserves prior briefings, preset, and selection mode; optional supplied preset/selection changes validate and commit with cadence atomically, while omitted values reuse the collection's stored briefing preset and existing app/provider defaults rather than the current Console conversation model.
- [ ] #4 A successful write requests immediate scheduler reload and returns stored cadence, gate/running state, next eligibility, last attempt/success, and separate reload-requested versus reload-acknowledged values.
- [ ] #5 A new bounded acknowledgement token completes only after the running loop reloads its queue; stopped, timed-out, disabled, or failed callbacks never report acknowledgement.
- [ ] #6 Artifacts and Settings show stored-but-inactive schedules honestly and use “Every 24 hours,” UTC storage/comparison, local display time, and app-open limitation copy.
- [ ] #7 Projection, loop, command, persistence-failure, timezone, gate, restart, and UI receipt tests pin the interval contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED real-SQLite projection and persistence tests for exact interval vocabulary, UTC eligibility, failure anchoring, off preservation, atomic optional preset/selection changes, and default-resolution isolation from the Console model.
2. Add RED SchedulerLoop tests and implement a thread-safe monotonic reload token that wakes the loop and acknowledges only after a successful queue load, with bounded waits and honest stopped/disabled/failure outcomes.
3. Add the Console-only `watchlists_set_briefing_schedule` command and descriptor with exact canonical validation, atomic persistence, stored receipt fields, gate/loop state, bounded reload acknowledgement, and fixed scrubbed recovery copy.
4. Share automation receipt semantics with Artifacts and Settings, including “Every 24 hours” copy, UTC storage/local display, app-open limitation, and stored-but-inactive states; keep Settings as global gate owner and Artifacts as collection cadence owner.
5. Run task-targeted scheduler/projection/DB/command/provider/Textual tests, Ruff, CSS integrity, diff checks, self-review, and independent review.

ADR required: yes
ADR path: backlog/decisions/019-watchlist-scheduler-migration.md and backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-019 already establishes `SchedulerLoop` as the sole Watchlists scheduling authority, while ADR-032 already fixes the Console-only mutation/exposure boundary. TASK-22864 extends those accepted contracts with observable interval reload acknowledgement; no new architectural decision is required.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added the Console-only schedule mutation over the existing Subscriptions DB,
  briefing projection, app-owned command facade, and SchedulerLoop ownership
  seams. Writes return committed schedule state before requesting a bounded,
  monotonic queue-reload acknowledgement.
- Reused one UTC eligibility projection for command and Artifacts receipts.
  Failed attempts anchor the next interval; never-attempted schedules are due
  immediately; Artifacts renders timestamps locally and keeps the app-open,
  stopped, and globally disabled states explicit.
- Kept ADR-019 and ADR-032 as the governing decisions. No schema, dependency,
  external MCP publication, or additional runtime state owner was introduced.
- Added real-SQLite, scheduler race/failure/coalescing, command/provider,
  controller wiring, and production-shaped Textual coverage. The task remains
  In Progress with acceptance criteria unchecked for independent review.
