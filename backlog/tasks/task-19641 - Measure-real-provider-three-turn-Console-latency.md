---
id: TASK-19641
title: Measure real-provider three-turn Console latency
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-21 21:03'
labels:
  - console
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure the reported three-turn Console failure path against a real local LLM so the Change Review fixes have reproducible median and p95 performance evidence rather than only ordering tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Thirty balanced three-turn samples per arm exercise the mounted Console through the real local OpenAI-compatible provider at `127.0.0.1:9099`
- [ ] #2 The pinned `origin/dev` control, branch-disabled arm, and branch-enabled mutating arm run in isolated profiles, databases, workspaces, and shadow repositories without touching user conversations or retained Change Review history
- [ ] #3 Raw samples and median/p95 summaries report the one/three/one provider-round sequence, third-send-to-worker, terminal turn release, common terminal-provider-completion trigger, descriptive third-send/E relationships, arm-specific Change Review events, per-sample event-loop lag, failures, and prompt loss
- [ ] #4 Every arm records the exact revision, model identity, fixed request parameters, rotated execution order, warmup policy, and host/runtime metadata needed to reproduce the comparison
- [ ] #5 All ninety conversations complete three turns with exactly one `load_tools`, one confined `fs_write`, one terminal tool-result follow-up, and no lost third prompt; paired one-sided confidence bounds show branch third-send-to-worker and per-sample event-loop-lag p95 do not regress more than ten percent against the control
- [ ] #6 The report distinguishes provider latency from application-owned latency and makes no performance claim when noise, failures, or sample completeness invalidate the comparison
- [ ] #7 Focused tests, static checks, privacy scans, and an independent evidence review pass before the task is closed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Specify a production-shaped, revision-pinned three-arm benchmark that drives the mounted Console through the real local llama.cpp provider while isolating every mutable profile and workspace path.
2. Add RED tests for percentile/confidence calculation, revision/import-path validation, balanced arm rotation, arm-specific completeness/privacy validation, exact tool-call ownership, watchdog behavior, and failure-preserving raw output.
3. Implement the smallest standalone runner and target adapter needed to execute the same benchmark contract against the pinned control and candidate checkouts.
4. Run one untimed warmup per arm, then thirty balanced measured iterations per arm with fixed bounded generation parameters, continuously flushed boundary evidence, and heartbeat samples buffered off the Textual I/O path.
5. Independently recompute the report, run a live three-turn smoke through the product send path, document conclusions without conflating model latency with application latency, and close only if every acceptance criterion is supported.

ADR required: no
ADR path: `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md` (existing governing ADR)
Reason: this task adds opt-in performance instrumentation and retained evidence only; it does not change runtime ownership, provider contracts, storage, or user-visible behavior. ADR-077 governs the consent and finalization behavior being measured.
<!-- SECTION:PLAN:END -->
