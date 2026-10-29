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
updated_date: 2026-10-29 02:47
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
Offline follow-up authorized 2026-10-28: repair only the demonstrated R2/V1 completion boundaries and eight module-size CI failures using narrow offline tests. Preserve native permissions/accounting/two-child limits and immutable replay. Keep PR3286 draft and AC24 unchecked; no live qualification or budget request. Existing ADR281 applies. Follow the bounded offline continuation in the detailed plan and record results separately from live evidence.
Offline continuation follow-up authorized 2026-10-29: reproduce the project-context deferral through the real agent loop, Console request preparation and strict custom-hosted payload builder. If reproduced, emit native deferral rows without the fenced-only name field while preserving native IDs/order, fenced names, instruction delivery, permission review and strict validation. Run only affected grammar/runtime/provider tests and static guards, get a scoped review, then update the existing draft PR. No live qualification or budget request. ADR required: no new ADR; existing ADR-069 and ADR-281 apply because this restores their existing protocol-safe continuation contract without new boundaries.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 After moving this task to In Progress, add an implementation plan with ADR required: yes; ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md; Reason: review entry point, source/report ownership, role boundaries, and cross-run verification contracts. Do this before implementation code changes.
- [ ] #2 Keep acceptance criteria unchecked until supported by recorded evidence; add Implementation Notes, updated docs, targeted verification and self-review before marking Done.
<!-- DOD:END -->


## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a primary-agent review recipe with saved provider/model teams,
read-only captured source, isolated reviewer capabilities, structured original
reports, blinded checkpointed verification, and host-expanded unified reports.
Existing fleet boundaries admit the fresh verifier on the same causal chain
without raising the two-child default. Native session retention, storage quotas,
privacy and executable-check restrictions remain enforced. AgentRuns uses the
normal version24→25 additive boundary-link migration. ADR281 governs the feature;
it was renumbered during integration because dev already owned ADR240.

AC24 remains open after the single approved final qualification on47093a3175:
the Haiku preflight passed, but the final run exited2 with incomplete R2/V1 seals.
Both originals mapped once and unified pages were readable; those facts did not
establish completion. The immutable failed outcome and raw preservation references
remain in Docs/superpowers/qa/2026-10-18-multi-model-code-review/bounded-final/.

The authorized offline follow-up fixes R2 same-ID correction/completion feedback
and V1 current-checkpoint recovery after history reduction without altering cached
page replay or weakening validation/sealing (3e5054c5ad). Connected request,
permission, metadata and continuation helpers move out of eight oversized modules
while retaining explicit callers and lazy imports (32f40af781). Seven size limits
are lowered; none are raised. No workflow redesign or unrelated baseline repair.

Verification:307 affected completion tests passed; the later recovery run covers9
cases including multiple batches, and17 post-format checks passed (overlapping
runs). Extraction passes102 agent and247 Chat/provider/config cases plus20
routing/authority and10 workspace-binding cases. All17 exact size/no-slack/helper
checks and3 startup-import guards pass. Fifteen extracted bodies are mechanically
equivalent and original owner lookups remain present. No new lint findings;
existing baseline lint remains. Twenty-five failing/error test nodes reproduce
identically on the original pre-turn candidate and are not claimed fixed.
See Docs/superpowers/qa/2026-10-18-multi-model-code-review/offline-repairs/report.md
for exact counts, baseline classifications, reviews and limits. No full-suite run.

This offline follow-up consumed zero provider requests and zero product children;
cumulative actual spend remains314 requests/21 children. The prior allowance's
unused financial capacity does not authorize another live invocation. No live
qualification or budget request was made. AC24 and DoD stay unchecked; the task
remains In Progress and PR https://github.com/rmusser01/tldw_chatbook/pull/3286
remains draft against dev. The original worktree/history and evidence are preserved.
No merge is authorized.
Current-dev follow-up: rebased onto live dev 3efe8476cebf70eceafa605d1abed76a4aad4e0e, preserving upstream first-request schema/compact-budget/disclosure behavior and existing review gates. The actual overlapping runtime changes pass 54 upstream, 34 review/continuation and five affected size controls. All 74 shared size ceilings retain the stricter parent limit; diagnostic inventory verifies after nine reviewed owner updates with privacy/sink sections unchanged. R2/V1 completion source/tests are unchanged. No provider requests, product children or new qualification; AC24 remains open and PR3286 remains draft. Exact current-dev receipts: /private/tmp/task35582-dev-sync/runtime-conflict-report.md; concise public notes: Docs/superpowers/qa/2026-10-18-multi-model-code-review/README.md.
Normal PR CI on ef56ce83f6 passed all34 test jobs, including all eight size guards and28 UI shards, plus Perf Guard/Tool Recovery. The final derived job failed only the worker/DOM guard in five new review UI methods. Reproduced five mounted removal races with NoMatches and fixed the exact UI lifetime boundary;29 targeted tests and the exact static guard pass without a census change. Completed configuration saves survive panel removal; only retired UI publication is skipped. No new lint findings; compile/whitespace pass. Detailed evidence: /private/tmp/task35582-dev-sync/worker-dom-repair/report.md. No new architecture or ADR; existing ADR281 applies. No live requests/children/budget question; AC24 open, draft retained. The new source head requires its own normal PR CI result.
Final single authorized run on0ab1e549f0 after successful required normal PR CI failed live_report_incomplete_V1. R1/R2 both sealed and presented; accepted fresh primary continuation failed locally after a code_review_read model response and before another physical request/V1 launch. All24 physical sends recordHTTP200; the UI400 classification is not proof of server400. Exact malformed field remains unproved. Native ledger paused usage_unknown after the local refusal; resource shutdown was clean. New24requests/2children, cumulative338/23;36requests/1child unspent but single invocation consumed. AC24 remains unchecked, task InProgress, PR3286 draft. Separate optional Windows GGUF timeout disclosed. No automatic retry, broad repair, additional budget ask or merge. Existing ADR281; concise evidence: Docs/superpowers/qa/2026-10-18-multi-model-code-review/final60/report.md.
Verified narrow offline continuation repair: native instruction-deferral results omit the fenced-only name, and prepared provider copies omit the internal delivery row key while canonical rows/receipts remain intact. Real loop/Console preparation/dispatcher/strict-hosted-builder regression reproduced both rejections separately and now completes deferred/reissued read. Final targeted selection: 191 passed; three pre-existing profile-selection failures reproduced with original HEAD functions; formatter/whitespace pass, zero new lint findings, scoped review clear. Evidence: Docs/superpowers/qa/2026-10-18-multi-model-code-review/continuation-offline/report.md. Existing ADR-069/281; no new architecture. No live requests/children/budget request, cumulative338/23 unchanged. AC24 open, task In Progress, PR3286 draft; prior live exact payload attribution remains unproved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
