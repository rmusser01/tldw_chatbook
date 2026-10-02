---
id: TASK-33664
title: Keep Console Resend out of empty startup imports
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 22:23'
labels:
  - agents
  - console
  - integration
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-09-29-agent-orchestration-burndown.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required Resend merge-base integration loads its new module during empty Console startup and exceeds the unchanged UI-ready module budget by one. Resend should load when its actual action or transcript projection is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original UI-ready module census passes its unchanged 1033-module ceiling with the empty Console behavior and expected mount members preserved.
- [x] #2 Real Resend click and keyboard, duplicate-worker, custody polling and selected-row action checks pass after deferring the imports.
- [ ] #3 App import, storage, CSS and source artifact guards remain unchanged and pass; focused independent review approves.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; existing ADR199 unchanged.
Reason: preserve incoming PR2962 implementation of the same accepted lazy-import repair; no new owner, authority, dependency, schema or ceiling.
1. Preserve qualified2cf source and all passing/non-green evidence. Read exact incoming ef8/e45 diff, task33661 post-merge notes and testing lesson before preserving rebase.
2. Retain incoming three deferred imports, legitimate module mock seam and complete new lesson, composing our shared hook/readiness/queue/publication repairs without broad source replacement. Compare all source bytes and exact changed methods.
3. Qualify affected actual Resend/slow-preflight/readiness/media consumers and unchanged original budgets proportionately; obtain focused immutable review. Current budget subprocesses already force PYTHONPATH to their exact REPO_ROOT and are unchanged.
4. Record exact source and evidence; recheck AC3 and close through CLI, then publish once observed003 lease with fresh-head Qodo/four jobs and normal protected merge.

Required PERF-07 integration qualification, after all four published-head checks passed:
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md
Reason: preserve landed ADR-126 D2 path memo and existing config/recovery owners; no new caching, schema, permission or runtime decision.
1. Preserve completed 84b8 CI and exact-head Qodo evidence; live strict protection requires the update to dev ecc0a531c855bc9e80906bff90180fd2045f7159 (PR2924/TASK33266).
2. Read incoming task, D2 corollary and config/sensitive-path/bundle callers. Incoming eight files have no direct patch or reviewed-source overlap. Capture exact source manifests, then rebase while preserving both sides.
3. Prove every reviewed Python/CSS file unchanged and incoming config/profile/sensitive-path/bundle sources byte-exact. Qualify incoming real private-profile memo invalidation/refusal, existing sensitive-path consumers and bundle fail-closed behavior.
4. Run focused real replay readiness, saved-close, wake/refund/nonreplay/config consumers; obtain independent immutable config/recovery and runtime review. Leave incoming TASK33266 status as shipped and separate open work untouched.
5. After reviewers/runtime checks finish, run the original five budget cases unchanged, plus fatal/added-line/new-file static, CSS and affected artifact/task guards. Retain non-green evidence and original limits.
6. Close this requalification through CLI only after evidence; documentation-only closure preserves approved bytes. Publish once with exact observed84b8 lease, then require fresh exact-head Qodo/all four CI jobs and normal protected merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred the three existing eager Resend imports to actual message resend, refused-echo dispatch and transcript action projection consumers; updated the existing legitimate test mock to the owning module. No new owner, dependency, UI behavior or budget. ADR097 directly governs the repair; no new ADR required.
Original UI-ready RED1034/1033 also reproduces on exact incoming Resend source. Final original five-case tested/untested storage/import/UI/CSS selection passes85.975s with681/686imports and1033/1033UI-ready, unchanged performance-source bytes and original drift warnings/no UI headroom. Real click/keyboard/duplicate worker, held-preflight publication and task33663 authority controls pass; final independent UI/runtime reviews approve2cfbb01c76. Static94patchPython/tennewRuff+format/whitespace and diagnostic/worker/index/UI/timestamp/CSS artifacts pass. Evidence and inherited limits are retained in the final review; no raw-suite/full-suite/live-provider/Windows result is claimed.

Preserved incoming PR2962 implementation of the same three deferred Resend imports, owning-module test mock and full boot-source lesson. Only the upstream module alias method/comments and local test alias differ from approved2cf;92other reviewed Python/CSS files remain exact. Both independent preservation reviews approve immutablea7d9dcbad73ac12ecbc7bd3f515e48a139e0eae9. Whole13-case Resend UI selection passes54.462s and final original five-case tested/untested storage/import/UI/CSS budgets pass75.893s, exit0. Imports remain681/686,UI-ready1033/1033 with unchanged sources, limits and drift warnings. Current subprocess guards explicitly pin PYTHONPATH to exactREPO_ROOT; no main-checkout measurement is claimed. Final93-file fatal/added-line,tennewRuff/format/whitespace and CSS reproduction pass. Exact source/AST and full lesson preservation proofs are retained; inherited upstream task33661 EOF whitespace is outside diff against actualef8. No new ADR; existing ADR097/199 apply. Detailed positive/non-green evidence and separate limits remain in the final review. Fresh published-head Qodo/four checks and protected merge remain.
<!-- SECTION:NOTES:END -->
