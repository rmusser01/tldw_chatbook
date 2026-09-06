---
id: TASK-31742
title: Address Qodo review findings for Canvas V1
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 21:37'
updated_date: '2026-09-06 00:40'
labels:
  - canvas
  - review
dependencies:
  - TASK-31741
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve every finding posted on PR 2432 with verified corrections or evidence-backed explanations while preserving the approved V1 security and lifecycle contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All eight initial Qodo findings have a documented technical disposition and a reply in their original review thread.
- [x] #2 Queued Canvas card actions cannot resolve an old card against a different active conversation; current-card actions still work.
- [x] #3 Valid review corrections preserve path authorization, transaction ownership, strict bounded wire validation, effective configuration precedence, compatibility and source-private diagnostics.
- [ ] #4 Targeted regression tests, independent review and required current-head CI support merge readiness; no security or performance gate is weakened.
- [ ] #5 A completed tool turn followed by a saved user prompt admits Capture On atomically, preserving original call reconstruction, exact owner/response/range validation and failure rollback; calculator and Canvas production-factory controls pass.
- [ ] #6 Pre-dispatch Retry proves and reuses its exact owned reservation without admitting unrelated calls, reviving terminal calls or duplicating dispatch; real controller/gateway recovery tests cover repeated failure and stale authority.
- [ ] #7 Commit reconciliation distinguishes committed, rolled-back and unknown outcomes; post-commit failures cannot duplicate surface writes or provider entry, or incorrectly mark a dispatched call not dispatched.
- [ ] #8 Both agent-mode and ordinary fresh next-message sends support the verified transition after a completed tool turn; other routes gain no implicit permission.
- [ ] #9 Short-lived trace worker operations release only their owned database handles on completion, failure and cancellation; repeated real agent and settlement operations do not accumulate exited-thread handles, and caller-owned or same-file observer connections remain usable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for direct review corrections. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md; ADR-097 governs startup costs. Reason: preserve existing security and ownership boundaries; stop for design approval if a suggestion requires new authority or architecture. 1. Read every Qodo review body and inline comment; record stable comment IDs and evaluate against actual call paths and approved contracts. 2. Reproduce verified behavioral defects before changes, starting with stale card session routing; use one bounded correction at a time and retain first-use/strict-zero-egress coverage. 3. For path, transactions, bridge validation and configuration findings, use existing shared mechanisms only when semantics remain exact; document justified disagreement instead of inventing containment roots or loosening validation. 4. Correct public helper documentation, compatibility wrapper naming and bounded operational log context; audit diagnostic inventory before regeneration. 5. Run targeted checks and independent review, reply to every original thread with evidence, update the PR and wait for current-head protected CI and Qodo completion. Root exclusively executes isolated pytest/browser checks; no full sweep, OS resource changes or V2 work before merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all eight initial Qodo corrections under ADR-121 (Canvas ownership, zero-egress and privacy) and the existing startup-budget ADR. Captured transcript-row session ownership survives queued actions and delayed construction; archive sources use shared lexical validation without inventing a workspace root; reads use owned deferred transactions while preserving caller rollback; bridge shape validation uses a lazy strict shared Pydantic envelope with unchanged domain limits; strict environment preferences preserve fail-closed recovery and the process latch. Public limit contracts, lazy compatibility aliases and bounded diagnostic attempt context are corrected. Targeted evidence: card50, archive56, reads140 plus16, wire176, config71 and final cleanup66 passing tests; these are overlapping focused runs, not a full suite. All scoped independent reviews pass; one final prose-only depth clarification was applied. Reviewed diagnostic inventory changes are two moved archive log statements and two fixed Canvas error statements with host-owned attempt integers; existing path candidates remain legacy/unreviewed. Current-head CI, original-thread replies and the final dev rebase remain pending before Done/merge. See Docs/Canvas/V1_VERIFICATION.md for commands, warnings and root-owned execution evidence.

Published reviewed corrections at b87f7ac31 after a 127-commit rebase onto dev8e9d1128d; only append-only lessons context conflicted and both sides were preserved. Replied in all eight original Qodo threads (reply IDs3942331184,3942331194,3942331230,3942331284,3942331319,3942331344,3942331384,3942331424). Post-rebase derived preflight passes; Chromium native/served/zero-egress89passed2optionalbrowser skips; trace/provider/Canvas/startup/mount538passed, census967/972. A test-only production trace plus Canvas composition control is still under verification; it initially omitted required progressive tool discovery, which the product correctly rejected. Current-head protected CI and final integration qualification remain pending.

Final integration is blocked, not Done: after a genuinely completed progressive Canvas turn, the next saved AGENT_FIRST request fails before transport with unsupported_surface_change. Runtime diagnostic proves prefix1/suffix0, six active tool artifacts versus two incoming saved revisions (assistant plus user): 1failed1warning1.80s. An ordinary successful calculator turn reproduces the next-turn failure; plain-history positive and changed-history negative still pass (1failed2passed1warning2.06s). These probes run on the feature tree, not untouched dev. No existing compound admission path composes bounded replacement plus append; implementing one changes the shared trace-admission/persistence contract governed by backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Per this task's plan, pause for design approval before that expansion. Retain the failing uncommitted diagnostic tests, all recovery refs and evidence; no capture bypass, weakened guard, merge or V2 work. All eight original Qodo replies are posted, but AC4 remains unchecked.
<!-- SECTION:NOTES:END -->

### Approved integration expansion, 2026-09-05

The user approved the focused shared trace repair after the production-factory
calculator and Canvas failures were diagnosed. Written contract:
[atomic tool-turn transition](../../Docs/superpowers/specs/2026-09-05-console-tool-turn-surface-transition-design.md).
ADR required: yes, amend the existing ledger contract.
ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`.
Reason: explicitly compose one verified replacement and one user append at the
existing atomic dispatch boundary, without changing schema or granting authority
from response links. Written-spec review precedes detailed implementation planning
and product changes; no capture bypass or V2 work is authorized by this expansion.

Requested design-review corrections are now incorporated in the linked contract:
reuse the exact owned pre-dispatch reservation through the controller/gateway
recovery handoff; reconcile write outcomes by exact call and surface/header
identity before Retry/Cancel/adapter entry; include ordinary FRESH next sends as
well as AGENT_FIRST. Add real recovery, post-commit error and route-transition
tests to the subsequent implementation plan. These are design requirements,
not implemented or passing behavior; AC4 through AC8 remain unchecked.

The user approved continuation after those corrections. Execute
[the repair implementation plan](../../Docs/superpowers/plans/2026-09-05-console-tool-turn-surface-transition.md)
serially: (1) verified compound shape/persistence and reconstruction, (2) exact
owned Retry and three-way commit reconciliation, (3) growth/integration checks,
review, latest-base protected PR completion. The existing trace-ledger ADR-097
now records the approved amendment before product changes. Root alone runs
isolated targeted tests and Git operations; independent task review is required.
