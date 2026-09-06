---
id: TASK-31742
title: Address Qodo review findings for Canvas V1
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 21:37'
updated_date: '2026-09-06 06:12'
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
- [x] #4 Targeted regression tests, independent review and required current-head CI support merge readiness; no security or performance gate is weakened.
- [x] #5 A completed tool turn followed by a saved user prompt admits Capture On atomically, preserving original call reconstruction, exact owner/response/range validation and failure rollback; calculator and Canvas production-factory controls pass.
- [x] #6 Pre-dispatch Retry proves and reuses its exact owned reservation without admitting unrelated calls, reviving terminal calls or duplicating dispatch; real controller/gateway recovery tests cover repeated failure and stale authority.
- [x] #7 Commit reconciliation distinguishes committed, rolled-back and unknown outcomes; post-commit failures cannot duplicate surface writes or provider entry, or incorrectly mark a dispatched call not dispatched.
- [x] #8 Both agent-mode and ordinary fresh next-message sends support the verified transition after a completed tool turn; other routes gain no implicit permission.
- [x] #9 Short-lived trace worker operations release only their owned database handles on completion, failure and cancellation; repeated real agent and settlement operations do not accumulate exited-thread handles, and caller-owned or same-file observer connections remain usable.
- [x] #10 After a proven pre-dispatch trace failure, explicit Send without capture and Cancel work for ordinary and agent sends without reviving terminal trace calls, weakening uncertain-delivery guards, or dispatching automatically.
- [x] #11 The explicitly owner-approved M5 Max reference fixture is versioned and documented in ADR-097; workload and numerical limits are unchanged, and a retained benchmark artifact proves environment matching and threshold enforcement.
- [x] #12 Latest-dev integration preserves canonical character-search and Canvas migration histories, verifies genuine upgrades and rollback, refuses incompatible pre-release predecessor databases without mutation, and resolves task-ID collisions with provenance-aware reference updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for direct review corrections. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md; ADR-097 governs startup costs. Reason: preserve existing security and ownership boundaries; stop for design approval if a suggestion requires new authority or architecture. 1. Read every Qodo review body and inline comment; record stable comment IDs and evaluate against actual call paths and approved contracts. 2. Reproduce verified behavioral defects before changes, starting with stale card session routing; use one bounded correction at a time and retain first-use/strict-zero-egress coverage. 3. For path, transactions, bridge validation and configuration findings, use existing shared mechanisms only when semantics remain exact; document justified disagreement instead of inventing containment roots or loosening validation. 4. Correct public helper documentation, compatibility wrapper naming and bounded operational log context; audit diagnostic inventory before regeneration. 5. Run targeted checks and independent review, reply to every original thread with evidence, update the PR and wait for current-head protected CI and Qodo completion. Root exclusively executes isolated pytest/browser checks; no full sweep, OS resource changes or V2 work before merge.
Approved reference-platform amendment: ADR required: yes, amend
backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md.
Reason: the owner explicitly designated the directly verified local M5 Max
18-core/128-GiB host as the benchmark reference. Reproduce the existing identity
failure, bump fixture identity/version only, then run the entire latency file
without non-reference opt-in and retain the raw result outside pytest retention.
Preserve all workload, software, SQLite and threshold values. Record the changed
platform honestly; this cannot establish performance on the former M4 Pro.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all eight initial Qodo corrections under ADR-121 (Canvas ownership, zero-egress and privacy) and the existing startup-budget ADR. Captured transcript-row session ownership survives queued actions and delayed construction; archive sources use shared lexical validation without inventing a workspace root; reads use owned deferred transactions while preserving caller rollback; bridge shape validation uses a lazy strict shared Pydantic envelope with unchanged domain limits; strict environment preferences preserve fail-closed recovery and the process latch. Public limit contracts, lazy compatibility aliases and bounded diagnostic attempt context are corrected. Targeted evidence: card50, archive56, reads140 plus16, wire176, config71 and final cleanup66 passing tests; these are overlapping focused runs, not a full suite. All scoped independent reviews pass; one final prose-only depth clarification was applied. Reviewed diagnostic inventory changes are two moved archive log statements and two fixed Canvas error statements with host-owned attempt integers; existing path candidates remain legacy/unreviewed. Current-head CI, original-thread replies and the final dev rebase remain pending before Done/merge. See Docs/Canvas/V1_VERIFICATION.md for commands, warnings and root-owned execution evidence.

Published reviewed corrections at b87f7ac31 after a 127-commit rebase onto dev8e9d1128d; only append-only lessons context conflicted and both sides were preserved. Replied in all eight original Qodo threads (reply IDs3942331184,3942331194,3942331230,3942331284,3942331319,3942331344,3942331384,3942331424). Post-rebase derived preflight passes; Chromium native/served/zero-egress89passed2optionalbrowser skips; trace/provider/Canvas/startup/mount538passed, census967/972. A test-only production trace plus Canvas composition control is still under verification; it initially omitted required progressive tool discovery, which the product correctly rejected. Current-head protected CI and final integration qualification remain pending.

Final integration is blocked, not Done: after a genuinely completed progressive Canvas turn, the next saved AGENT_FIRST request fails before transport with unsupported_surface_change. Runtime diagnostic proves prefix1/suffix0, six active tool artifacts versus two incoming saved revisions (assistant plus user): 1failed1warning1.80s. An ordinary successful calculator turn reproduces the next-turn failure; plain-history positive and changed-history negative still pass (1failed2passed1warning2.06s). These probes run on the feature tree, not untouched dev. No existing compound admission path composes bounded replacement plus append; implementing one changes the shared trace-admission/persistence contract governed by backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Per this task's plan, pause for design approval before that expansion. Retain the failing uncommitted diagnostic tests, all recovery refs and evidence; no capture bypass, weakened guard, merge or V2 work. All eight original Qodo replies are posted, but AC4 remains unchecked.

Completed the approved Task 2 recovery repair under backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md in c388c2bd7 and ed86d6a90: exact accepted-owner reservation reuse, verifier retirement with recoverable preparation faults, atomic bind outcome read-back, cancellation and uncertain-delivery handling, cold/new-invocation replay refusal, and operation-owned worker connection cleanup. Independent task review and scoped I1 re-review are clean. Root verification: 983 affected passes (one known custom-API exclusion, Requests and descriptor warnings), 207 focused passes with zero regular-file descriptor growth, then 169 fix-covering passes with only Requests warning. Same affected baseline has 885 passes and descriptor growth345 versus current204; this is not global resource-cleanliness evidence. Ordinary Ruff remains baseline-qualified with no new fix findings. Growth/latency/startup/browser release gates, current-dev rebase, Qodo, protected CI and merge remain pending; task stays In Progress.

Task3 checkpoint ffb934ce9: compound-growth file7passed; corrected observed test-harness counter publication and completion ordering with deterministic RED before atomic publication. Final three Chromium files90passed2optionalbrowser skips1Requests warning184.52s; startup/Canvas110passed4warnings40.72s after an unchanged isolated/full rerun; census967/972 and import635/660 remain within budgets. All six derived preflight categories pass after source-private fixed-warning inventory review. Three changed test/fixture files pass Ruff and formatter checks. Numeric latency gate has18passes1failure solely at reference-hardware identity: actual M5 Max18CPU128GiB differs from pinned M4Pro14CPU48GiB; thresholds not applied and no waiver. Independent checkpoint review, actual-dev migration-number reconciliation/rebase, new Qodo/current-headCI and merge remain pending. No merge or V2; AC4 stays unchecked.

Broad repair review found fresh AGENT_FIRST actor/chain identities could replay the same unresolved saved turn. Two real gateway regressions confirmed this before product correction. Real agent-enabled Retry anyway positive established its existing Capture Off behavior and unchanged old trace ledger, so the fix reuses the current owner/turn unresolved-call query for AGENT_FIRST as well as FRESH without a new recovery authority. Post-format preservation selection172passed1Requests warning73.73s; all six preflight guards pass. Scoped final rereview pending. Reference-machine latency and latest-dev migration/task collision/rebase plus fresh Qodo/CI remain open; no merge or V2.

Final scoped rereview of f55ab2cbe..ee27c7193 approves runtimeI1 and minorM2 with no new Critical/Important fix breakage. I2 reference-hardware numerical latency and I3 actual-dev migration/task reconciliation plus current-head PR/Qodo/CI remain unaddressed merge gates. Current code has172focused passes,90Chromium passes(two optional skips),1compound-growth pass and all6preflight guards. No rebase, push, merge or V2 in this checkpoint; keep In Progress/AC4 unchecked. Reference M4Pro14CPU48GiB availability requested from user; actual hostM5Max cannot satisfy the unchanged identity gate.

Owner-approved reference update: amended backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md and the repair plan before changing fixture hardware/version only (v5 M5 Max18CPU128GiB). Fresh v4 gate fails on the four hardware fields:1failed1Requests warning32.25s. Full v5 latency file passes19tests1Requests warning43.11s, environment_match=true and threshold_gate_applied=true with no opt-in. Reservation/dispatch p95 2.811ms/max4.467ms and settlement p95 1.701ms satisfy unchanged10/50/25ms limits; all five per-database checks pass. Both raw sample artifacts retained in the local SDD evidence workspace with SHA-256 digests and exact commands in Docs/Canvas/V1_VERIFICATION.md. No runtime/test-logic changes or claim of performance on the former M4 Pro. I2 closed for this checkpoint; I3 latest-dev migration/task-ID reconciliation, rebase, Qodo/current-headCI and merge remain open. AC4 stays unchecked and task remains In Progress.

I3 integrated and independently approved: canonical character-search65→66 retained, Canvas66→67→68, real predecessor/refusal/rollback/state preservation and exact index census verified. Later task collisions renumbered31812/31813 with all tracked pointers and provenance. Final root gates:179growth/runtime/recovery passes;19approved-reference latency passes;90Chromium passes/2optional skips;110startup/controller passes;all6preflight guards. Warnings remain qualified in Docs/Canvas/V1_VERIFICATION.md. Latest devcc22deb0a adds only two CI workflow fixes; final143commit rebase is exactly equivalent for every replay commit, and resulting tree differs only in those workflows. Independent I3 follow-up approves pointer correction with no blocking findings. Retained recovery ref codex/canvas-v1-before-ci-dev-20260905. AC4/current-head Qodo/CI remain pending; no merge or V2 yet.

Final protected integration completed: PR #2432 merged normally into dev at 2026-09-06T06:11:26Z, merge commit f32a16839d1810618107f7e2ffaed8ac2e3e634f, verified head 18828038d080f59405828d8634e03b56b504c9ee. Fresh fetch and ancestor check confirm dev contains the reviewed head. Qodo summary 5554930741 references that exact head with zero bugs/rule violations/skill insights; all original threads were addressed. Current-head checks passed, including Fast Lane 9m1s and Derived Artifacts 6m23s. Windows GGUF import required one unchanged failed-job rerun after a missing-node assertion; rerun passed, initial failure retained. Latest rebase onto 723c81460 was conflict-free with all 144 replay commits patch-equivalent. Directly affected checks: 66 passed, one Requests warning, 37.57s; six preflight categories passed. Two old suspend-stub failures reproduce exactly on untouched dev and remain baseline-qualified, not repaired or claimed passing. No full-suite claim, protection bypass, pause of other work, or cleanup. Recovery refs, feature worktree and raw evidence preserved. ADR-121 and trace ADR-097 remain governing contracts. V2 may now enter brainstorming; no V2 implementation has begun.
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

Historical design-review state (before implementation): requested corrections
were incorporated in the linked contract:
reuse the exact owned pre-dispatch reservation through the controller/gateway
recovery handoff; reconcile write outcomes by exact call and surface/header
identity before Retry/Cancel/adapter entry; include ordinary FRESH next sends as
well as AGENT_FIRST. Add real recovery, post-commit error and route-transition
tests to the subsequent implementation plan. At that checkpoint these were design
requirements, not implemented or passing behavior; AC4 through AC8 were unchecked.
The current checklist and appended implementation notes above supersede that state.

The user approved continuation after those corrections. Execute
[the repair implementation plan](../../Docs/superpowers/plans/2026-09-05-console-tool-turn-surface-transition.md)
serially: (1) verified compound shape/persistence and reconstruction, (2) exact
owned Retry and three-way commit reconciliation, (3) growth/integration checks,
review, latest-base protected PR completion. The existing trace-ledger ADR-097
now records the approved amendment before product changes. Root alone runs
isolated targeted tests and Git operations; independent task review is required.
