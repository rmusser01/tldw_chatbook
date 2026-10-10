---
id: TASK-34786
title: Retire borrowed application owners after composer harness tests
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-10 22:43'
updated_date: '2026-10-10 22:49'
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
- [ ] #1 Composer harness teardown drains its borrowed application runtime before retiring actual durable owners through their existing public lifecycle APIs.
- [ ] #2 A mounted resource regression fails on the original teardown; all existing session-tab composer cases preserve their behavior with bounded descriptor growth.
- [ ] #3 Targeted checks, static analysis, derived-artifact preflight and independent review qualify the scoped fixture fix without changing generic fixtures or production ownership.
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
