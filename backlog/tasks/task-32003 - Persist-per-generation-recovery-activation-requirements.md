---
id: TASK-32003
title: Persist per-generation recovery activation requirements
status: In Progress
assignee: []
created_date: 2026-09-07 23:59
labels:
- backup-recovery
dependencies:
- task-31978
- task-31988
- task-31991
- task-31992
- task-32001
updated_date: 2026-09-10 23:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Restored capabilities remain inactive across every supported launch until their own owner review completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restored capabilities remain inactive across every supported launch until their own owner review completes.
- [ ] #2 Missing/corrupt activation records and imported approvals cannot grant execution authority.
- [ ] #3 Safe local inspection works and one owner approval never activates unrelated automation or queued work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task20 and ADR126: first persist private per-generation/per-owner requirements independently of journal/report/catalog, with idempotent require, owner-specific local approval and missing/corrupt/mismatched refusal. Then bind restored generations before publication fences clear; wire every supported startup/composition and owner-specific existing review path so local inspection remains available without automatic execution, reconnection or replay. Verify fresh-process/relaunch/headless/corruption and one-owner isolation with targeted tests; review, lint, Bandit and scoped commits. This initial independent slice implements durable store only; Task18 publication integration and actual consumer gates remain required before task completion.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-20)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial original Task20 durable-store slice implemented in activation.py + test_activation.py under ADR126. Original one-owner approval regression failed behaviorally then passed; final22focused cases pass2.03s (/private/tmp/chatbook-activation-store-final.log). Covers private strict requirements/owner approval records, restart/realchildexit, missing/corrupt/linked/public/different-generation denial, no read-time creation, unchanged idempotent requirements and failed durable retries. Review P2 retry-barrier gap fixed by pinned existing-record identity/validation plus file fsync/F_FULLFSYNC and directory/ancestor flush before returning success; independent finalreview approved /private/tmp/chatbook-activation-store-review.md. FullRuff/formatclean, productionBandit0. No global locks/new journal; independent immutable per-owner files avoid lost updates. Startup association/consumer gating/owner reconciliation remain required before taskDone; no ACchecked or wholefeatureclaim. Integration source map /private/tmp/chatbook-task20-activation-source-map.md.
Next original Task20 slice: bind the current restored generation and owner requirements in fixed bootstrap records independently of convenience reports/catalog and removable pending operation. Bind only from matching local pending operation/selector, create durable ActivationStore requirements under its external control root, and publish checked private association atomically with native barriers while fence remains. Add read-only owner permission lookup that follows fixed selector association across config edits and denies missing/corrupt/mismatched required state while ordinary native startup/local inspection remains separate. No imported association authority; no consumer activation coverage claim until actual execution/review seams are wired. Tests cover first/second generation, idempotent retry, mismatched pending association, missing/corrupt state, config edit, report removal and safe local startup.
Automatic approval review rejected proposed production association patch because activation_permission would allow ordinary operation when fixed association is absent, which could bypass restrictions after association loss. Patch was NOT applied. Only initial unused API stubs and red test_activation_binding.py currently exist as WIP; no startup/bootstrap changes. Read-only design review now seeks a genuine fail-closed distinction using existing local records, preserving untouched behavior and safe inspection without circumventing rejection. Continue unaffected credential/installed-validation/Chroma work. Do not imply association integration is implemented or approved.
Safer redesign after automatic approval rejection, source-backed proposal /private/tmp/chatbook-activation-association-proposal.md: separate fixed activation association plus independent restore-generation witness in existing strict _Profile record. Either surviving record identifies known restoration; missing/corrupt/mismatched counterpart denies owner execution. Historical witness checked before config fingerprint fallback. Existing Admission registry holds SH across MaintenanceSession, so no registry EX update; add narrow profile update seam under exact active session and matching durable pending selector/namespaces, using existing pinned temporary/fsync/replace pattern. Ordinary bind_profile stays exclusive-create and cannot erase witness. No read repair, new registry, or imported approvals. This materially addresses rejected single-record-loss bypass; submit redesigned code through normal automatic review, do not reapply denied branch indirectly. Total loss of all local restore authority remains documented compatibility boundary; no claim it can be distinguished from never-restored installation. Task20 consumer/service integration remains required.
Automatic approval rejected the materially safer paired-witness bootstrap production patch too; rejected patch NOT applied. Exact stated reason: substantial bootstrap change alters durable activation-witness parsing/authorization semantics and requires explicit user authorization for security/startup-gating risk. No workaround/retry attempted. Prepared28 native-session testcases remain red against unusedstubs; report /private/tmp/chatbook-activation-binding-blocked-report.md. User asyncapproval question submitted explicitly describing original section9 restriction, twoindependent records, singlelossdenial and untouchedprofilecompatibility. Await actual response before any production retry. Continue independentoriginaltasks; no activation completion claim.
Activation association remains approval-blocked; no rejected production patch was applied. Root has now presented a single explicit approval question covering this original-plan activation record change and the separately rejected process-local RAG save-continuation boundary. The automatic reviewer classified both as security-sensitive authorization changes requiring specific user authorization. No new authorization response has arrived as of this note. Paired profile-witness/association design remains /private/tmp/chatbook-activation-association-proposal.md; do not infer approval from broad original-plan continuation.
User explicitly approved both previously blocked original-plan changes on this turn: durable bootstrap/profile activation association keeping restored automation inactive until owner review, and process-local accepted RAG-save continuations during closed new intake. Resume the concrete paired-witness and queued-continuation proposals through normal tools/review. This approval resolves the prior automatic-review authorization blocks; no scope expansion, push, merge or real-profile restore is authorized.
Explicitly approved paired activation association implemented and independently APPROVED. bind_activation requires exact live native MaintenanceSession/matching fixed pending operation; preserves existing profile mapping and captures record identity before durable requirements/pair publication. Separate strict profile witness + fixed association deny missing/corrupt/mismatched activation while valid ordinary mappings preserve safe inspection. Temporary update intent remains a publication fence. Final 124 focused tests passed (37 binding), Bandit0→0, Ruff7→3 inherited only; no new findings. Consumers must supply actual admitted namespaces; owner/executor/UI/fence clearing remain original Task20 work. Evidence: /private/tmp/chatbook-activation-pair-approved-report.md and /private/tmp/chatbook-activation-pair-independent-review.md. Only new census producers are exact native control record replace and update-intent retirement.
Root integration gate: exact index snapshot synchronized to all1847 tracked production Python files; unchanged11-test ownership census passed, staged diff check clean. Paired record commit includes only4 code/test files,2 source inventory rows and this task record. Inherited TTS/traversal/ADR history remains unstaged.
Next original Task20 integration unit: use committed paired authority (ddf569164) at actual app startup effects and shared scheduler execution before effects. Scope includes admitted-namespace accessor/helper if necessary, app scheduler/catalog/speech/cleanup/automatic FTS backfill gates, SchedulerLoop run/tick/manual dispatch and SchedulingService run/sync entry. Preserve local content/history readers and ordinary untouched profiles; no imported grants, no owner-approval UI in this unit, no full scope promotion. Test actual fresh-process restored/corrupt/ordinary launch effect sentinels and shared direct scheduler calls. Design must retain real ordinary storage/startup hold while checking actual namespaces; no caller-supplied UI namespace authority.
Startup/scheduler consumer unit frozen and root reviewed: real native expanded StorageLease scope retained through actual effects; restored shared sources outside held scope refuse, ordinary registered-only sources remain usable. Gates scheduler run/load/tick/dispatch/manual/sync, remote CRUD with existing local edit fallback, app catalog/speech/backfill/cleanup callbacks. Final20 fresh-process network/process sentry cases passed18.76s; named scheduling90passed/1explicitly deselected known raw_source_selection_changed app-import fixture; maintenance/speech40passed. Periodic reload test now waits for second load under2s bound instead of fixed10ms; production interval unchanged. Ruff429→429/Bandit42→42; native SQLite and diagnostic census delta0. Source/report /private/tmp/chatbook-activation-startup-report.md; patch excludes inherited storage traversal changes. Root reviewed scoped seven-file diff; Task20 remains In Progress: exact custom TTS/model source consumers, Notes/sync/MCP/skills/agents, owner review UI and full launch composition remain original work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->