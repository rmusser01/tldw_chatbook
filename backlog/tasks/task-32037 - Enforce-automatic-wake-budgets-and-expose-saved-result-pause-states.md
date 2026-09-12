---
id: TASK-32037
title: Enforce automatic wake budgets and expose saved-result pause states
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:22'
updated_date: '2026-09-12 03:40'
labels:
  - agents
  - console
dependencies:
  - TASK-32019
  - TASK-32020
  - TASK-32021
  - TASK-32035
  - TASK-32036
references:
  - backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
  - backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A finite automatic allowance must be enforced on real wake dispatch and provider calls, with a visible pause that keeps completed results available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The defaults of three accepted wakes, six automatic child launches, 32 shared model calls, 500000 budget tokens, 8192 output tokens per call, and 900 elapsed seconds are enforced through both agent and plain-provider wake paths.
- [x] #2 One wake batches one causal chain; other chains remain pending, and exhausted or ambiguous chains are not retried automatically.
- [x] #3 Manual draft and queue claims win admission; automatic primary work leaves one existing primary slot available for manual work without changing approval authority.
- [x] #4 Budget exhaustion preserves pending results and unseen state and exposes an actionable pause reason through the existing fleet surface; manual continuation is explicit.
- [x] #5 Multi-generation, concurrent-reservation, childless/plain-path, cancellation, configuration-change, and crash/restart tests exercise the actual runtime boundary.
- [x] #6 Individual terminal survivors become eligible after a fixed coalescing window without waiting for siblings; final-drain usage reconciliation remains unchanged.
- [x] #7 Up to two conversations may wake concurrently subject to primary and manual reserves, with round-robin eligible conversation selection and unchanged approval resolution.
- [x] #8 A required durable acceptance fence precedes all automatic model/tool work, including preparation helpers; failed bookkeeping never reruns an accepted attempt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md and backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
Reason: Direct implementation of the accepted finite allowance, dispatch, delivery, recovery, and fleet projection contracts.
1. Add failing real controller/SQLite tests for generation exhaustion, required acceptance before helpers, claim recovery, and completion-write failure without replay. Integrate typed attempt dispatch and chain pauses first.
2. Trace and fence all automatic provider/helper calls and child launches; reserve prepared input plus bounded output, settle actual/unknown usage, and enforce the shared elapsed deadline.
3. Add individual terminal-survivor fanout with fixed 250 ms coalescing; preserve final-drain usage reconciliation and add fair per-conversation wake ownership with two slots and one manual primary reserve.
4. Reconstruct pending results independently of unseen marks through explicit native runtime startup recovery; project metadata-only pauses and explicit manual continuation through the extracted fleet module.
5. Verify the ADR-135 crash matrix, multi-generation/plain/agent/helper paths, concurrency, configuration reductions, approvals, cancellation, UI behavior, targeted regression tests, lint/format and diagnostics; update docs and issue ledger, independently review, then complete AC/notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented finite automatic-work enforcement and conservative recovery across controller preparation, real provider calls, helpers, children, and late tool approvals. Defaults are 3 accepted wakes, 6 automatic child launches, 32 shared model calls, 500000 budget tokens, 8192 output tokens per call, and 900 elapsed seconds. Automatic work shares its causal chain; explicit manual work clears inherited automatic context and creates a distinct allowance.

Individual terminal survivors enter a fixed 250 ms grouping window; two conversations can wake concurrently while retaining a manual primary reserve and round-robin selection. Exact durable acceptance precedes helpers or dispatch, and completion stamps only the claimed batch. Startup discovers existing history independently of badges and performs one recovery audit. Interrupted/ambiguous work remains saved without replay.

Review-driven fixes: schema v18 adds a FULL-synchronized runtime-owner fence for completed-parent survivors, physical provider workers recheck after executor/client waits, preparation cancellation releases Console occupancy, and late skill/MCP approvals recheck before side effects. An empty-startup recovery failure remains visible in later resumed sessions. The fleet inspector uses an optional wrapped notice because ordinary rows clipped the pause explanation; viewing clears attention without clearing the pause. Run history and explicit manual continuation remain available.

ADR required: no new ADR. Implemented backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md and backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md. Schema v18 is a necessary implementation of ADR-135 runtime revocation; agent_runs_v17_to_v18_runtime_owner.sql preserves existing ledger history. The optional inspector notice and scope-preserving fixture migrations are documented deviations from the initial file-level plan, within its acceptance criteria.

Verification: final backend run 1344 passed; its two excluded loopback-server cases passed separately with local socket permission. The 7 affected legacy UI modules passed 34 tests; launch 11; startup 14; final pause/widget run 32; the final history-error copy correction passed 2 focused cases. After the final recovery/view changes, 46 recovery/coordinator tests passed. Independent runtime re-review passed 65 focused tests, and UI/recovery re-review passed 18, clearing all five findings. These runs overlap and are not additive. Real-gateway tests cover plain and agent multi-generation dispatch, automatic child/follow-up shared usage, live reductions, blocked-stream deadlines, and a second wake during real approval. Real SQLite tests cover rollback, owner replacement, queued physical dispatch, and schema upgrades.

New modules/tests pass Ruff and formatting; modified legacy production files add no lint findings. Scoped whitespace checks pass. Five reviewed diagnostic owners were updated with metadata-only calls. Existing unrelated TTS diagnostic-inventory drift and the TASK-3070 Console screen-size failure remain; this task neither refreshes that TTS entry nor raises the screen budget. No full-suite, external-provider, killed-process, or power-loss certification is claimed. Existing remote side effects and Python threads retain cooperative cancellation limits.

Updated the user guide, implementation plans, ADR status/implementation notes, review ledger, private SQLite owner inventory, and testing-evidence lesson. Changes remain uncommitted in the shared working tree.

Final PR #2631 integration preserves latest dev lifecycle custody and controller boundaries. Review corrected live stuck-child intake and durable selection to match eligible done/error/cancelled results, releasing exact wake ownership without pausing the chain or blocking later done siblings. Both real-badge regression variants pass; focused dispatch/scheduling/recovery selections passed 59 and 38 overlapping cases. Permanent bridge-close fencing, readiness-gated startup recovery and plural wake projection were also verified. Existing ADR-135 applies; automatic authority was not broadened.
<!-- SECTION:NOTES:END -->
