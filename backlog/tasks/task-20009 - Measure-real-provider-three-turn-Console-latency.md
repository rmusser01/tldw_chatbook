---
id: TASK-20009
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
- [x] #1 Thirty balanced three-turn samples per arm exercise the mounted Console through the real local OpenAI-compatible provider at `127.0.0.1:9099`
- [x] #2 The pinned `origin/dev` control, branch-disabled arm, and branch-enabled mutating arm run in isolated profiles, databases, workspaces, and shadow repositories without touching user conversations or retained Change Review history
- [x] #3 Raw samples and median/p95 summaries report the one/three/one provider-round sequence, third-send-to-worker, terminal turn release, common terminal-provider-completion trigger, descriptive third-send/E relationships, arm-specific Change Review events, per-sample event-loop lag, failures, and prompt loss
- [x] #4 Every arm records the exact revision, model identity, fixed request parameters, rotated execution order, warmup policy, and host/runtime metadata needed to reproduce the comparison
- [ ] #5 All ninety conversations complete three turns with exactly one `load_tools`, one confined `fs_write`, one terminal tool-result follow-up, and no lost third prompt; paired one-sided confidence bounds show branch third-send-to-worker and per-sample event-loop-lag p95 do not regress more than ten percent against the control
- [x] #6 The report distinguishes provider latency from application-owned latency and makes no performance claim when noise, failures, or sample completeness invalidate the comparison
- [x] #7 Focused tests, static checks, privacy scans, and an independent evidence review pass before the task is closed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Specify a production-shaped, revision-pinned three-arm benchmark that drives the mounted Console through the real local llama.cpp provider while isolating every mutable profile and workspace path.
2. Add RED tests for percentile/confidence calculation, revision/import-path validation, balanced arm rotation, arm-specific completeness/privacy validation, exact tool-call ownership, watchdog behavior, and failure-preserving raw output.
3. Implement the smallest standalone runner and target adapter needed to execute the same benchmark contract against the pinned control and candidate checkouts.
4. Run one untimed warmup per arm, then thirty balanced measured iterations per arm with fixed bounded generation parameters, continuously flushed boundary evidence, and heartbeat samples buffered off the Textual I/O path.
5. Independently recompute the report, run a live three-turn smoke through the product send path, document conclusions without conflating model latency with application latency, and close only if every acceptance criterion is supported.

ADR required: no
ADR path: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md` (existing governing ADR)
Reason: this task adds opt-in performance instrumentation and retained evidence only; it does not change runtime ownership, provider contracts, storage, or user-visible behavior. ADR-084 governs the consent and finalization behavior being measured.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added a revision-pinned parent/child benchmark that drives the mounted Console
  through the real composer, prompt queue, local tool catalog, confined
  `fs_write`, llama.cpp provider, and arm-specific Change Review lifecycle while
  isolating every profile, database, workspace, shadow repository, and child
  environment.
- Retained content-free raw JSONL, manifest, machine summary, reproduction guide,
  and human interpretation under
  `Docs/superpowers/qa/console-three-turn-real-provider/`. The manifest pins
  control `5f720a40417eaa78f33619d5cbc82effc470104b`, candidate
  `eb8225a32f88ea43c337aff99804d360384e7668`, the exact model/request/corpus/tool
  fixtures, runtime/server metadata, host load, and listener resource samples.
- The completed run contains three warmups plus 90 measured conversations (30
  per arm), 450 measured provider calls with coherent token accounting, 93
  successful mutations, zero prompt loss, and zero terminal ownership failures.
  A separate standard-library recomputation exactly matched every median, p95,
  paired bootstrap bound, improvement claim, and the final verdict.
- The pre-registered verdict is `inconclusive`. Disabled third-send-to-worker
  passed its 10% non-regression gate, but disabled event-loop lag and both enabled
  gates had confidence bounds crossing the ceiling. AC #5 is therefore only
  partially supported and remains unchecked; the task intentionally remains In
  Progress rather than moving the threshold after seeing the data.
- Hardened long-run evidence ownership after a retained failed profile showed
  Textual 8.2.8 can propagate owned child-loop cancellation from
  `_message_loop_exit` after all product assertions are durable. The harness now
  preserves cancellation failures, suppresses only completed-contract child
  teardown cancellation when the caller itself is not cancelled, and still
  requires final thread/provider/SQLite/shadow/source ownership to be clean.
- Verification: 86 benchmark tests and 582 changed-surface Console/agent/review
  tests passed; Ruff, `py_compile`, JSON parsing, `git diff --check`, recursive
  privacy scans, and the independent evidence audit passed.
- ADR required: no new ADR. Existing ADR-084 continues to govern Change Review
  consent/finalization; this work changes benchmark instrumentation and retained
  evidence only.
