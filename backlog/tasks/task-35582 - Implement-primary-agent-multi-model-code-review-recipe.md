---
id: TASK-35582
title: Implement primary-agent multi-model code review recipe
status: In Progress
created_date: 2026-10-18 19:51
labels:
- agents
- code-review
references:
- backlog/decisions/147-agent-provider-routing.md
- backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md
- backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
- backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
documentation:
- Docs/superpowers/specs/2026-10-18-multi-model-code-review-design.md
- backlog/decisions/281-primary-agent-multi-model-code-review.md
- Docs/superpowers/plans/2026-10-18-multi-model-code-review-implementation.md
updated_date: 2026-10-28 21:48
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users request independent code reviews from their chosen providers and models using a reusable default team. Present each reviewer's report first, then offer a fresh verifier and a unified report that accounts for every finding with evidence. Preserve user control over reviewer selection, optional specialist focuses, and when verification runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A natural-language multi-model review request loads the primary recipe and uses existing fleet execution without an installed skill or nested delegation; the optional trusted inline skill follows the same recipe.
- [x] #2 Canonical Agents settings persist one ordered default reviewer team, optional focuses and a verifier by stable preset ID; per-run overrides preserve saved defaults and use existing provider/model routing authority.
- [x] #3 All reviewers receive the same general brief and captured target; focuses supplement it, and task composition excludes peer reports and primary-agent suspected findings.
- [x] #4 PR, commit-range, patch and working-tree inputs use exact revisions or captured bytes/hashes with consistent read-only source access; source drift, unavailable history, exclusions, binaries and truncation have explicit coverage or refusal.
- [x] #5 Setup preserves workspace/path exclusions, instruction delivery, provider-destination consent, routing allowlists, budgets and permission floors; missing/disabled presets and duplicate destinations remain visible without silent substitution.
- [x] #6 The prepare/scoped-spawn/read/submit/release contracts bind caller roles, source and preset snapshots, target digest, owning chat/source path, stage grants and actual run receipts; generic spawn overrides cannot mix with a review-scoped launch, and accepted launch retries do not dispatch twice.
- [x] #7 Version-1 finding, coverage, host-envelope, disposition and check-receipt records are validated; host original IDs are stable, records are at most 8 KiB and chunks at most eight records, and invalid versions/references/local keys or stale/conflicting submissions preserve accepted data.
- [x] #8 Reviewer findings are structured submissions with severity, trigger, impact, source/evidence and uncertainty; reports seal only against actual terminal receipts and complete coverage, while empty findings are valid and malformed/failed/cancelled/exhausted roles remain incomplete.
- [x] #9 Individual reports appear first with host-derived requested/actual destinations and fallback switches; raw role output and structured originals are separately readable under session retention, without mistaking a bounded run summary for a full report.
- [x] #10 The primary coordinates from the complete host finding index and capture receipts instead of concatenating original report pages; fleet/capture/storage gaps stay explicit and no prose parser or another model reconstructs a supposedly complete index.
- [x] #11 Verification requires an existing explicit user stage grant or acceptance and launches one fresh configured verifier; model booleans cannot mint consent, and already-authorized stages do not require redundant confirmation.
- [x] #12 The verifier view uses unrelated random opaque unit IDs with a private reversible original-ID map; original IDs, models, severity votes, report order, receipt identities, group proposals and exact-repeat multiplicity are absent, and only byte-identical substantive claims are coalesced.
- [x] #13 Nonidentical claim triggers/evidence remain distinct verification units with individual dispositions; duplicate proposals cannot transfer verdicts between them, minority findings remain eligible, and non-reproduction without counter-evidence stays unresolved.
- [x] #14 The unified report preserves originals, validates every unit and expands every original finding exactly once through the private map, with valid canonical references, actual target/participants/checks, and explicit unresolved or incomplete coverage.
- [x] #15 Executable checks use authorized process paths and disposable quota-bounded locations; proposed checks are distinct from host execution receipts, results return to the live verifier before a final decision, and post-return checks require a new explicit verification attempt.
- [x] #16 Admission preserves the default two total children per primary turn and separately checks concurrency: two reviewers plus pre-authorized verification use a later verifier turn, and three reviewers use admitted continuation slots without raising global defaults or forgetting earlier spawns.
- [x] #17 Review continuations use host-sealed exact-next-stage/slot receipts through ADR-134/135 acceptance and the original finite causal chain, with no fabricated survivors, fresh allowances or duplicate dispatch; kill switches, exhaustion, manual priority, uncertain acceptance and stop/close/context fences preserve progress without automatic replay.
- [x] #18 Verification batches contain at most eight units and 4,096 estimated claim tokens, further bounded by half the actual remaining admissible input; evidence pages are at most 1,024 estimated tokens and full request admission includes history/tools/guidance/output reserve; unknown capacity and oversized claims are explicit.
- [x] #19 Batch submissions atomically checkpoint dispositions, evidence and cursors outside prompt history; compaction, repeated cursors, stale revisions and interrupted runs cannot lose or double-count accepted decisions, and unfinished units finalize unresolved without minting extra model/child budgets.
- [x] #20 Storage admission enforces 64 MiB source/verification content, 16 MiB report/raw/manifest/index/ledger content, 1 MiB per participant output, 80 MiB total per review, 2,048 total regular files per review, 160 MiB retained/reserved per tab and four retained contexts, counting staging and physical copies.
- [x] #21 Concurrent reservations cannot overspend storage; overflow/disk errors preserve accepted partial data, full tabs require explicit keep/export/release or scope choice, no silent eviction occurs, and tombstoned cleanup releases quota only after actual native leases and deletion settle.
- [x] #22 Reviewer capability gates prohibit mutation, execution, peer-report and unrelated history access; verifier reads expose only its captured source/blinded batches, and stale, guessed or cross-chat references and expired bundles cannot widen authority or switch to newer code.
- [x] #23 Focused tests exercise actual concurrent provider dispatch, schema/role/reference validation, the default-two-child continuation matrix and acceptance races, identity projection/duplicate expansion, small-context batching across compaction, exact quota boundaries/concurrent reservations, cleanup failures, cancellation and actual native Capture On round trips.
- [ ] #24 A bounded real Console/provider qualification uses two reviewer destinations and a fresh verifier, including a pre-authorized request under the default two-child ceiling, seeded defect/false positive/duplicate, failed reviewer, truncation, source drift, small-context and storage-overflow controls; live-model evidence is distinguished from scripted controls.
- [x] #25 Architecture, user and example-skill documentation reflects ADR-281 and its ADR-134/135 continuation amendment; any changed existing DB owner uses normal versioned migration, and no general workflow engine, automatic fixes/publication or durable review-job recovery is added.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-10-18-multi-model-code-review-implementation.md.
The approved implementation and scoped repairs are complete; preserve their
permission, accounting, role and default-two-child contracts. For delivery:
1. Integrate on current dev in a separate worktree; retain required fixtures and concise evidence, preserving raw history separately.
2. Adopt a fail-closed qualification result gate and prepare a narrow real Haiku native tool-result/accounting preflight.
3. Obtain one explicit consolidated request/child allowance from actual receipts, with fixed retry boundaries.
4. After preflight passes, run one final bounded Console journey. Require presented reviewer reports, a fresh [2,1] verifier, and full original mapping/unified report.
5. Finish only affected targeted checks and reviews, then update AC24/DoD honestly and prepare the PR. A new failure gets a narrow causal reproduction, not a broad restart.
ADR required: yes (existing decision, renumbered for integration).
ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md.
Reason: existing review ownership/runtime contracts; dev already uses the old ADR240 number for another decision. No new architecture is introduced.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 After moving this task to In Progress, add an implementation plan with ADR required: yes; ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md; Reason: review entry point, source/report ownership, role boundaries, and cross-run verification contracts. Do this before implementation code changes.
- [ ] #2 Keep acceptance criteria unchecked until supported by recorded evidence; add Implementation Notes, updated docs, targeted verification and self-review before marking Done.
<!-- DOD:END -->


## Implementation Notes

Implemented a primary-agent review recipe with saved provider/model teams,
read-only captured source, isolated reviewer capabilities, structured original
reports, blinded checkpointed verification, and host-expanded unified reports.
Existing fleet boundaries admit the fresh verifier on the same causal chain
without raising the two-child default. Native session retention, storage quotas,
privacy and executable-check restrictions remain enforced. AgentRuns uses the
normal version24→25 additive boundary-link migration.

Delivery preserves current dev bridge Stop/stall tracking, native-thread callback
handling and workspace/index authority. Feature ADR240 becomes ADR281 to resolve
a real numbering collision; contracts are unchanged. The original worktree,
71-commit history and raw QA archives remain preserved in a verified Git bundle.
Required fixtures and concise independent reviews remain in the PR.

AC23 retains its recorded targeted evidence. AC24 remains open after the single
approved final qualification on47093a3175: the real Haiku preflight passed, then
the final run used54 requests/3 children and exited2 (`live_report_incomplete_R2`).
R2's new-ID correction left its earlier invalid-record rejection unresolved; V1
accepted both correct decisions but reread a cached revision-0 batch after a
revision-1 checkpoint and never completed its protocol. Both originals map once
and the unified report is readable; incomplete participant seals still fail AC24.
The two boundaries were reproduced offline without changing product behavior.

The approved scope consumed59 new requests/3 children; cumulative314/21. Seven
financial requests remain unused, but retry/final-invocation/child scope is spent;
no additional live execution or budget request follows. Eight module-size CI
failures reproduce; a composer-state CI failure passes the exact local test and
is not claimed fixed. No broad refactor, raised budget or full rerun was started.
See Docs/superpowers/qa/2026-10-18-multi-model-code-review/bounded-final/report.md
and its independent review, result hashes and durable archive references.

PR https://github.com/rmusser01/tldw_chatbook/pull/3286 remains draft against dev.
The original worktree and history are preserved. ADR281 remains the governing
decision; no architecture change was introduced during bounded closeout.

No merge is authorized. Keep the PR draft and this task In Progress until the
remaining live qualification and Definition of Done are actually met.
