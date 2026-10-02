---
id: TASK-33664
title: Keep Console Resend out of empty startup imports
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 23:28'
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

## Required release integration — October 2

ADR required: no new ADR
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md; backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md; backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md
Reason: preserve the landed release metadata, first-paint guard and missing-field API parity without new ownership, runtime, storage, permission, packaging or performance decisions.
1. Published e1377 has all four required CI jobs PASS, current-head Qodo clear and no actionable threads. Actual dev f3aeb32 is CONFLICTING/DIRTY after PR2961; strict protection requires a preserving update after completed checks. Reopen existing TASK33664 AC3 before implementation.
2. Capture exact 96 reviewed Python/CSS and prior eight PERF-07 sources plus 17 incoming files. Read TASK33803/TASK33645, existing ADRs and actual overlap. Rebase preserving both complete testing-lesson tails and exact incoming version0.2.3, evaluation field, source-digest list and first-paint absence assertion. No other source repair is planned.
3. Prove approved runtime/schema/recovery/permission/physical-custody/CSS bytes unchanged. Run only incoming evaluation schema/client and release metadata/app-only/source-digest controls; obtain a focused independent immutable source/lesson/guard review. Leave incoming tasks and separate release publication outside this PR qualification.
4. After functional checks/reviewer settle, run the original five storage/import/UI-ready/boot-CSS cases with unchanged pins/ceilings/counts/warmup/work, now including incoming Resend absence guard. Verify fatal/added-line/new-file static, CSS and affected governance/task artifacts. Retain all non-green and earlier limits.
5. Recheck/close existing task via CLI after evidence, record qualification in plan/review/follow-up and concise PR body; doc-only closure preserves all approved bytes. Publish once exact observed e1377 lease, then fresh exact-head Qodo/all four jobs and protected head-pinned merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred the three existing eager Resend imports to actual message resend, refused-echo dispatch and transcript action projection consumers; updated the existing legitimate test mock to the owning module. No new owner, dependency, UI behavior or budget. ADR097 directly governs the repair; no new ADR required.
Original UI-ready RED1034/1033 also reproduces on exact incoming Resend source. Final original five-case tested/untested storage/import/UI/CSS selection passes85.975s with681/686imports and1033/1033UI-ready, unchanged performance-source bytes and original drift warnings/no UI headroom. Real click/keyboard/duplicate worker, held-preflight publication and task33663 authority controls pass; final independent UI/runtime reviews approve2cfbb01c76. Static94patchPython/tennewRuff+format/whitespace and diagnostic/worker/index/UI/timestamp/CSS artifacts pass. Evidence and inherited limits are retained in the final review; no raw-suite/full-suite/live-provider/Windows result is claimed.

Preserved incoming PR2962 implementation of the same three deferred Resend imports, owning-module test mock and full boot-source lesson. Only the upstream module alias method/comments and local test alias differ from approved2cf;92other reviewed Python/CSS files remain exact. Both independent preservation reviews approve immutablea7d9dcbad73ac12ecbc7bd3f515e48a139e0eae9. Whole13-case Resend UI selection passes54.462s and final original five-case tested/untested storage/import/UI/CSS budgets pass75.893s, exit0. Imports remain681/686,UI-ready1033/1033 with unchanged sources, limits and drift warnings. Current subprocess guards explicitly pin PYTHONPATH to exactREPO_ROOT; no main-checkout measurement is claimed. Final93-file fatal/added-line,tennewRuff/format/whitespace and CSS reproduction pass. Exact source/AST and full lesson preservation proofs are retained; inherited upstream task33661 EOF whitespace is outside diff against actualef8. No new ADR; existing ADR097/199 apply. Detailed positive/non-green evidence and separate limits remain in the final review. Fresh published-head Qodo/four checks and protected merge remain.

Preserving PERF-07 integration is independently approved at immutable source 13e261a2582d661d8b41fa3111937659d589241a on dev ecc0a531c855bc9e80906bff90180fd2045f7159. The rebase is clean: all 96 previously approved Python/CSS files and all eight incoming files are exact, with no direct overlap. Incoming config memo remains behind admission and re-verifies path posture; sensitive contexts resolve their memoized raw inputs afresh. Existing ADR-126 D2 applies; no new ADR, source repair, owner, permission, schema, dependency or ceiling.
Independent config/recovery selection passes 11 cases (8.363s); independent replay passes 13 (51.881s), custody/refund/nonreplay/saved-close passes 12 (65.239s). Root incoming memo/bundle/sensitive-path/mounted consumers have 24 passes and one inherited config-retarget fixture failure (45.702s). The same raw_source_selection_changed failure reproduces on exact incoming ecc with cwd/PYTHONPATH pinned (1.511s); both non-green logs/XML are retained, with no gate/test masking. All twelve incoming memo tests pass. Final original five budgets pass 83.786s, exit 0, at 681/686 imports and 1033/1033 UI-ready with unchanged sources, workload and warnings.
Final 93-file fatal/added-line, ten new Ruff/format, whitespace, CSS, diagnostic, worker and 4789 task guards pass. Evidence is /private/tmp/pr2918-perf07-*. All schema76/both gates/frozen AgentRuns and runtime/replay/queue/hook/refund/physical custody bytes remain exact. Incoming TASK33266 retains shipped To Do status; separate open tasks and all earlier limits remain. Fresh published-head Qodo, four CI jobs and normal protected merge remain delivery gates.
<!-- SECTION:NOTES:END -->
