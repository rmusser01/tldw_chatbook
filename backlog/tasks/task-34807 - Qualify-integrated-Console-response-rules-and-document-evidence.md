---
id: TASK-34807
title: Qualify integrated Console response rules and document evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:56'
updated_date: '2026-10-12 23:14'
labels: []
dependencies:
  - TASK-34806
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 9. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Integrated route tests cover learning, checking, cancellation, shared repair and inert reopen.
- [ ] #2 Provider behavior and native UI evidence are qualified separately from scripted tests.
- [ ] #3 Targeted regression and static checks pass and independent whole-branch review findings are addressed or recorded.
- [ ] #4 A bounded named critical selection covering helper-preparation custody, installed schema/recovery/voice and native/hook/currentness boundaries runs in actual required PR lanes with nonvacuous executable membership checks, preserved existing targets/order/limits and exact final-head hosted results.
- [ ] #5 Concurrent trace GC between current durable semantic admission and provider-call reservation preserves ordinary and native-repair provenance; revoked/orphaned revisions remain reclaimable and stale graph epochs never sweep.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: integrated response-rule qualification and private lifecycle boundaries. Test actual composer learning/repair/reopen/exclusion and preserve original actions, run targeted feature and touched-owner/static gates, attempt separately-labelled provider/native-terminal qualification, perform one fresh whole-branch review and document evidence and limitations before task closure.

Latest-dev PR integration (2026-10-04): retain ADR-228; merge origin/dev 1b12df2757, preserve incoming Console model/settings and quit behavior, move unpublished response-rule schema to v77 after the installed v76 notes FTS migration, qualify exact v75/v76 recovery plus the targeted feature/owner/static gates, record remaining qualification limits, then push and open the requested PR against dev. ADR required: no new ADR; ADR path: backlog/decisions/228-console-learned-response-rules.md; reason: integration preserves the approved ownership and runtime contracts.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

Scoped dispatch-plan preflight correction: Ruling: bind critical response-rule safety regressions into required CI and qualification-task acceptance — existing green lanes cannot prove unenrolled new contracts — costs bounded recurring cases and executable membership controls within original limits. Follow the revised current integration Task1 before product edits; preserve source-approved scopes and every historical receipt/acceptance record. No new ADR/ID or product boundary.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/228-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.

2026-10-12 current Goal ruling (ADR227/228): assess only the exact coordinator-issued, accepted Goal iteration handoff pinned through native parent identity and message version. A genuine live GoalIterationAuthorization is required; arbitrary SYSTEM rows, copied metadata and shape-compatible objects confer no rule authority. Recheck Goal/binding/source/Stop currentness. Completed Goal runs expose no existing same-native-run repair seam: preserve the assessment and explicitly refuse automatic repair rather than admitting an ordinary queued turn. No second native run, Goal attempt, machine receipt, allowance or repeated tool work; keep record_outcome identity guard and inherited absolute limits unchanged. Phase12 exposed the genuine source_changed assessment defect; qualify passing, violating and revoked/fake input cases. This enforces inherited ownership and leaves settled-Goal automatic repair unsupported, documented for independent review. ADR required: yes (existing amendment), backlog/decisions/228-console-learned-response-rules.md.

2026-10-12 connected trace lifetime ruling: phase24 reproduced startup GC removing newly admitted user/assistant semantic revisions before provider call reservation, leaving an accepted repair retained for recovery. Apply existing ADR097 decision16 and ADR228: preserve the live provenance owned by current durable dispatch checkpoints through call reservation, without new storage, a retention owner, timing suppression or weaker GC authorization. Map: durable turn acceptance -> canonical user/assistant revisions -> captured request provenance -> ConsoleTraceBoundaryFactory reservation; TraceGarbageCollector mark/sweep must respect the current accepted owner during the gap. Prove actual ordinary-send and native-repair survival with real collector/admission/reservation, and orphan/revoked/current-version/epoch-change reclamation or refusal controls. Costs: a wrong missing root deletes live provenance; an overbroad root leaks orphan retention. Keep existing graph epochs, native handles, deletion grants, privacy/fork/purge and caps. ADR required: yes, existing ADR097/228 amendment; no new ADR ID.

2026-10-12 automatic GC epoch qualification: phase29 real worker/collector reproduced a completed epoch4 result indefinitely reused at epoch6 after checkpoint settlement because physical compaction was threshold-deferred. Root approved invalidating pending_gc_result only when its marked_epoch differs from current_graph_epoch, returning through existing collection. Preserve unchanged-epoch reuse, actual compactor eligibility, unknown/incomplete result failure handling and all scheduling/thresholds. Goal deliberately freezes Capture Off under its genuine SYSTEM handoff; qualify real collection before actual coordinator-owned provider entry and rule assessment without introducing USER checkpoints or new retained roots. Cost if wrong: repeated collection during unchanged idle state or stale logical collection hiding orphan reclamation. Existing ADR097/228 apply; no new persistent declarations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the integrated Console response-rule journeys and resolved all seven Important whole-branch review findings in one native TDD pass. Initial implementation evidence: 233 feature cases and 62 touched-owner cases passed. Latest-dev publication integrates origin/dev 1b12df2757, retains incoming model picker/Chat Settings/quit behavior and installs response rules as v77 after dev's v76 notes FTS migration; exact v75/v76 restricted recovery is covered. ADR-228 remains authoritative.

Fresh integrated evidence: the 17-file feature run passed 232 cases with three initial-send UI timeouts; the selected owner run passed 222 cases with two UI timeouts and two documented exclusions. Correcting test predicate polling and settling one frame before the next click preserved all assertions and 30-second domain deadlines; the final six-case gate passed all four learn/repair/reopen/exclude journeys and both literal-send cases in 501.32s. Ruff on 70 changed Python files, Black on 34 authored files plus changed legacy ranges, mypy on 12 domain files, final test formatting/lint, diff and generated-bundle checks passed. Governance passed 29 cases with two ratchets identifying unchanged dev sources. The sparse-context-policy Save assertion reproduces with exact current-dev ConsoleChatStore after real profile bootstrap; symlink privilege is unavailable. Failed combined attempts and exclusions remain in Docs/superpowers/reviews/2026-10-03-console-response-rules.md; no whole-suite pass is claimed. Updated source/UI/tests, migration/recovery owners, ADR-228, user guide, implementation evidence and concrete scratch-ACL/UI-synchronization lessons. Repairs retain completed work and original answers, prompt recall excludes machine feedback, promotion preserves body-free calibration and capacity refuses before publication. One independent review, no re-review, no deferred minors. Actual configured provider probe returned ConnectError, visible native-terminal tooling is unavailable and mixed native/external-hook transport qualification remains open. Acceptance criterion 2 stays unchecked and this task remains In Progress. The subsequent user request authorizes push and a draft PR against dev; retain the managed worktree for review feedback.

Independent bounded plan preflight preserves original scope but found missing recurring required-CI acceptance/verification argument. Current plan and this acceptance now make final executable critical coverage explicit. Applicable original measurement/prerequisite/consumer obligations are clarified without relaxing limits or claiming implementation/UAT.

2026-10-12 current integration (historical source evidence above remains unchanged): Integrated all91 source paths by current owner with ADR228 and current nine-task identities. Required ConsoleP0 received13 exact bounded targets across its two existing processes: measured50 cases40.13s plus19 cases45.46s; observed prior13m3/20min leaves headroom (combined projection14m29, doubled addition15m55). Existing targets/order/processes/flags/timeouts/aggregate and six lanes are guarded by executable removal/comment/filter/cap/aggregate negatives. Phase36 CI/manager212pass2fail and phase38 refusal/workspace/workflow79pass4test-readinessfail remain retained; phase42 allfour corrected mounted journeys pass. Real local provider controls prove literal repair+genuine argv hook, Stop, source_edited and physical settlement, but arithmetic/older-question quality and actual retained-tool use failed. Original wire/history/task-root bindings are preserved. Baseline ChatScreen has759 methods against unchanged756 cap; exact START/current method identities prove no added method. No all-green/full-suite/native-paint claim. Existing ADR228 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Source TASK-34362 at codex/wave2-source-pr-3010 (73823fde3a4e) was created on 2026-10-04. The older current-dev task keeps 34362; this preserved younger response-rules task is TASK-34807. Original implementation history remains above; references and dependency identities follow this remap.
