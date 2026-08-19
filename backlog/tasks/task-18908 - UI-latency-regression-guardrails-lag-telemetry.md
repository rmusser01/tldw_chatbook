---
id: TASK-18908
title: UI latency regression guardrails + lag telemetry
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-19 15:55'
updated_date: '2026-08-19 16:45'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the Aug-2026 Windows 3s-lag incident (fixed via PR #1824): add merge-blocking latency guardrails in a small dedicated CI workflow (the main Tests workflow is red on all branches per TASK-18608), and persist in-app UIResponsivenessMonitor stall data to the diagnostics sink so user lag reports arrive with evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deterministic guards: CSS source count stays under the Textual LRU-64 parse cliff after a destination tour (44 today, soft limit 56),Screen-switch latency budgets for hot destinations (Console/Library/Settings) with generous CI-safe margins,Perf guard workflow runs green independently of the broken Tests workflow,Stall events persisted to the diagnostics sink with lag/timer context,Efficiency spike findings filed as measured backlog tasks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Spike-measure current dev screen-switch latencies + CSS source count (Pilot tour, scratch config)
2. Write guardrail tests: destination-tour switch budgets (class-name arrival asserts) + CSS LRU-64 cliff guard
3. Persist stall telemetry: edge-triggered persist_event on heartbeat breach, re-armed on recovery, best-effort contract
4. Dedicated perf-guard.yml workflow independent of the red Tests workflow (TASK-18608)
5. File measured follow-ups (TASK-18909 Console switch cost, TASK-18910 boot CSS rebuild)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on branch perf/ui-latency-guardrails, PR #1827. Spike baselines at dev f6ae7d23e (M-series Mac, scratch config): Home 0.76s / Console 1.55s / Library 1.39s / Settings 0.89s warm; CSS sources after full tour 44 (< 56 soft limit < 64 cliff). First probe's 31s 'Console lag' was a probe bug (label never matched ChatScreen; own deadline expired) - the arrival asserts in the guardrail use screen CLASS names specifically so this cannot recur. 1 pre-existing test failure (test_console_sync_records_worker_lifecycle, MagicMock await) verified identical on clean dev - TASK-18608 family, not this change.
<!-- SECTION:NOTES:END -->
