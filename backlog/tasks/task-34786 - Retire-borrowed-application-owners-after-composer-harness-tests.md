---
id: TASK-34786
title: Retire borrowed application owners after composer harness tests
status: Done
assignee:
  - '@codex'
created_date: '2026-10-10 22:43'
updated_date: '2026-10-10 23:04'
labels:
  - testing
  - console
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/2882'
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The session-tab composer harness constructs a real application but shuts down only its Console host. Retire the borrowed durable owners while the private test profile remains selected so repeated cases do not retain native SQLite descriptors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Composer harness teardown drains its borrowed application runtime before retiring actual durable owners through their existing public lifecycle APIs.
- [x] #2 A mounted resource regression fails on the original teardown; all existing session-tab composer cases preserve their behavior with bounded descriptor growth.
- [x] #3 Targeted checks, static analysis, derived-artifact preflight and independent review qualify the scoped fixture fix without changing generic fixtures or production ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: repair a local test harness lifetime using established ownership APIs; no storage, runtime boundary or production behavior change.

1. Preserve the 90-case FD warning and constructor/mounted controls identifying native owners retained after ConsoleHarness exits.
2. Add one actual-owner resource regression that fails with the existing local _running teardown.
3. Extend only the session-tab composer module teardown: release held gates, finish Textual shutdown, drain borrowed application owners, then call their existing public close/aclose routes before fixture profile cleanup.
4. Append the resource regression node to the required UI census and raise its literal floor to179, preserving existing entries and shard placements.
5. Verify the new regression and all17 composer cases, CI census contracts, static analysis and preflight; independently review the fixture and CI wiring before publication.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The local session-tab composer _running context releases held gates and finishes ConsoleHarness shutdown, then drains the borrowed application runtime and retires EvaluationOrchestrator plus workspace, collections and subscriptions owners through their existing public APIs while the private profile is selected. Generic fixtures and production lifecycles are unchanged.

The real mounted native-handle regression failed before repair (expected ProgrammingError did not raise;1failure/0errors). Source34146da27eca37bdfebe7dcbccd874eab648df43, carried byte-identically toe1d8b9ae2d07a3580500f55ffdfff5b048c9b369, passes all18 composer cases in64.25s with no failures/errors/skips/warnings. FD census14to21 (+7) retains only visible instance locks/event-loop handles, no SQLite owners or native registry entries; bounded growth is not zero retention.

One resource node is appended to required UI census (179entries/floor179, prior178 prefix/shards unchanged). All218 CI contracts pass4.78s. Full derived-artifact preflight passed on9b9a5223; changed final census/task guards pass at e1d8. Scoped formatter/fatal Ruff and whitespace pass; full-file Ruff retains two exact baseline return-None diagnostics, zero introduced. Independent review3029 clears fixture ownership/exception lifetime, real native-close test and CI wiring. Existing ADR126 applies; no new ADR. The testing-evidence lesson records constructor/mounted controls and the203FD incident.

The separate hosted Notes executor_failed failure remains unreproduced after20 fresh-process repetitions,12module cases and88neighbors; no Notes change or claim of its resolution. Fresh required hosted CI remains the publication/merge gate.
<!-- SECTION:NOTES:END -->
