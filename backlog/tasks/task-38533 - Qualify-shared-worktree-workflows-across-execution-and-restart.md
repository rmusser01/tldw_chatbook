---
id: TASK-38533
title: Qualify shared-worktree workflows across execution and restart
status: Done
created_date: 2026-11-01 17:31
dependencies:
- TASK-38531
- TASK-38532
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-10
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
updated_date: 2026-11-05 06:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 10 of the accepted native shared-worktree workflow design so qualify shared-worktree workflows across execution and restart. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A1–A12 have retained evidence from real files/Git/SQLite, native admission, held-worker races and fresh-process recovery controls.
- [x] #2 A real provider performs an authorized correction between failed and passing targeted checks with exact manifests, process results and phase approvals.
- [x] #3 Living architecture/user/Settings/recovery docs describe only qualified behavior; unresolved bootstrap, platform or uncertainty limitations remain explicit.
<!-- AC:END -->

## Implementation Plan

ADR required: no new ADR
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Qualify and document the existing native phased workflow contract without introducing a new execution, permission or persistence owner.

1. Read the accepted current native/Console implementation and retained Task 1–9 source-qualified receipts; build the A1–A12 evidence and remaining-gap matrix before running new cases.
2. Add and run the deterministic native journey with exact failed/corrected/passing verifier evidence, dirty-input and Stop preservation; verify only uncovered actual boundary races and admission controls.
3. Exercise genuine fresh-process crash/restart and complete profile backup/restore/rebackup, preserving conservative unknown recovery and inert imported history.
4. Resolve a real configured provider privately and qualify the complete mounted application controls, real file correction, final review, detach and restart Resume without changing the user's profile or exposing secrets.
5. Run targeted affected checks and scoped static/schema/link checks; update living architecture/user/recovery guidance and actual incident lessons. Preserve all unrelated work and keep workflow opt-in disabled by default.
6. Self-review, retain the exact A1–A12 receipts and source/dirty manifest, commit only owned paths, and report to the root for independent spec/quality review before marking Done. No full test sweep, push or merge.

## Task 10 qualification amendment — exclusive native shell, 2026-11-04

The mounted Console publishes an armed shell descriptor to the read-only planner. A public phase accepted that descriptor but failed before its first child because execution preparation had omitted the matching provider. At the existing finite task entry, an approved exclusive task explicitly naming `shell_exec` may compose the original Console raw-shell provider with the current session, configuration and captured local binding. The matching raw-shell review hook and local review hook both run. The published original app/runtime shell source, current permission, arm state, exact provider and session must still match. The verifier uses the existing native process, current workspace authority, original executable bytes and actual exit/manifest records. Planning, scoped tasks and explicit private producer inputs keep their existing paths. A plan, approval or restored record grants no shell authority. Qualification includes a revoked arm, kill switch and changed source before launch. This amends the existing ADR-285 implementation; the feature stays disabled by default.

## Task 10 qualification amendment — failed verifier and passive restart, 2026-11-04

A native child reached Done after its required verifier actually exited 7. Its immutable check correctly said failed, but the workflow task inherited the model's successful completion. The coordinator then tried to settle success and retained Recovery instead of releasing a physically drained phase for explicit repair. When the original required evidence is current, complete, cleaned up and has an integer exit different from the approved expected exit, record a submitted completed workflow task as failed. Preserve its original native Done status, submitted digest, exact verifier, both manifests and process exit. Other evidence states retain their own meanings. Existing physical retirement and FULL settlement must finish before repair.

A separate fresh-process crash revealed that passive accounting imported Console and RAG startup inside a history DB read even when every unknown native attempt was already marked review_required by the original startup audit. For that all-review-required case, return conservative unknown before the Console import. Empty, mixed and accepted-row handling, charges and live witness checks keep their existing rules. This change introduces no new startup audit or recovery capability.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Qualified native correction/dependency/review, true process crashes and complete profile capture/restore/reopen/rebackup with exact source/PID receipts. Whole-mounted TldwCli used the user's local Qwen endpoint for actual failed check, authorized edit, fresh passing verifier and independent review. A later completed-history run qualified the original scheduled retention callback and final Creator return separately.

Five narrowly reviewed corrections reuse existing owners under ADR285 and ADR126. Execution preparation composes the original explicitly allowed exclusive shell provider and review hook. Complete fresh native nonzero evidence derives workflow task failure for explicit repair. All-review-required passive accounting avoids a live Console import. Original hook-source validators serialize and reread current policy/source after waiting. Scheduled sync-log retention uses the existing finite-worker cache helper.

Added native/process/race regression coverage, retained adverse and passing cohorts, updated five Architecture pages plus Console/Settings guidance, and recorded the actual65-second timer incident in testing lessons. Source-qualified A1–A12 evidence, bounded commands and historical limitations are under Docs/superpowers/qa/2026-11-01-native-phased-agent-workflows. Fourteen changed Python/test files pass full Ruff checks. Two narrowly changed files retain exactly76 and1 preexisting lint findings with zero added diagnostics and passing modified-range formatting. Verification stayed targeted. Independent Task10 spec/quality review is approved with the source limits below; final whole-execution review and root archive/census closure remain pending.

Independent task review `310931472fb9e279a490058f4b15aa8b2cc40c65216ab35f91d910c6d9865965` approved spec compliance and quality with zero Critical/Important findings. Native Qwen functional proof (restart9), later same-profile timer/Creator proof (inspect3) and final evidence-label proof retain separate exact source versions. Root independently verified all489 committed paths,8325 current source/config hashes, original staged61 diff and protected279 files. Existing ADR-285 and ADR-126 apply.

Known evidence caveat M1: the functional live run also emitted `RuntimeWarning: coroutine 'ChatScreen._sync_native_console_chat_ui' was never awaited` at qualification/live/restart9/process-tail.txt:13. Its allocation cause and lifetime consequence remain unexplained. This is distinct from the attributed MediaDB cache refusal and later cleanup fix. It is carried to final whole-execution triage. Raw logs, old failed cohorts,12 unavailable early driver bodies, dirty-source dependencies, Darwin-only physical coverage, inherited lint/format debt and raw-data whitespace remain explicitly qualified; no full-suite, clean-checkout or single all-final-source end-to-end claim.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

Implemented and independently reviewed native shared-worktree workflows through final correction8fc0274486. The final scoped spec/quality gate89efdf9c approves joint capacity, same-chat admission and exact-target Console handling with zero blocking findings.149 targeted checks pass on the correction source. Actual local-provider correction/review, crash/restart and complete profile backup/restore/rebackup retain their separate exact-source limits. Root archival byte audit is complete and all acceptance criteria are checked. The feature remains disabled by default; M1 and historical adverse/platform/source/static limits remain disclosed. Complete chronology and governing reports resolve through the durable qualification archive. No full suite or single all-final clean provider-run claim.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->


## Task 10 qualification amendment — concurrent hook-source validation, 2026-11-04

Actual whole-mounted local read5 recorded two original workflow source checks overlapping on the same native legacy hook issuer. One check's original capture held admissions1 with events0 and no published target. A different workflow thread treated that source-only admission as unfinished hook work and cancelled the child before its read. The original capture then returned normally with admissions0. The native cancelled-row guard correctly refused the subsequent tool call. The separate original-App RED2 regression reproduced that refusal while a real no-target ApprovalRequested admission remained correctly blocked.

Serialize only workflow source validators with an issuer-local reentrant lock spanning the existing cleanup checks and the same original capture's actual return. After waiting, obtain a fresh original hook configuration snapshot and recheck exact current runtime engine and issuer identity, attachment, cleanup and published targets. Preserve configured effectful-definition refusal, v2 checks, original capture and physical hook retirement. Actual event admissions retain their existing conservative refusal. No source/permission owner, runtime engine, cleanup bypass or ordinary hook route is introduced. Original source loss and cancelled native rows continue to refuse.

| Lock | Acquisition and return |
| --- | --- |
| Existing config snapshot lock | Initial policy read releases before the validation lock. A second original policy read occurs after acquiring the validation lock and releases before capture. |
| Issuer workflow validation RLock | Only workflow source validators acquire it. It spans current identity/policy/cleanup checks and original capture return. Event producers never acquire it. |
| Existing issuer Condition | Original capture and cleanup readers acquire briefly under the validation lock. The original capture releases it before source/recovery reads and reacquires only for its actual finally decrement. |
| Runtime hooks/v2 map lock | This correction adds no acquisition and preserves existing checkpoint ordering. |

The validation lock has no reverse acquisition from event production, config snapshots, source reads or retirement. It may wait for an actual finite validator return. It does not reinterpret an unresolved admission as returned. Verify overlapping original checks, held real event capture, changed original source, fresh effectful configuration after waiting, original callback/creator boundaries and bounded whole-mounted real read before returning to the full correction journey. Existing ADR285 and root independent review govern this narrow ordering repair. Preserve the original adverse receipts, including read5's separate MediaDB Creator refusal.


## Task 10 qualification amendment — scheduled retention cache ownership, 2026-11-04

The actual mounted local-provider workflow completed its failed-check, authorized correction, fresh passing-check and independent review journey. Its later Creator retirement refused a MediaDB cache after original App shutdown and executor join. A separate completed-history reopen waited for the original65-second sync-log retention timer. The original retention callback returned, but its exact new MediaDB connection survived on a dead executor thread with a live lease and zero active operations. The shorter8.50-second history case ended before that timer and therefore covered a different lifecycle interval.

Apply the existing ADR126 finite-worker helper at this exact original callback boundary. Submit `run_finite_local_worker`, the same `db.delete_sync_log_entries_older_than` bound method and the same cutoff to the existing `asyncio.to_thread`. The helper uses original owner APIs on the producing worker after the real callback returns. Its existing exclusions preserve borrowed/preexisting caches, foreign threads, custom/in-memory owners, active operations, transactions and uncertain closes. The helper itself, retention semantics, timer schedule and existing App shutdown ownership retain their current implementations. This is an application of existing finite-worker custody, with no new global drain or storage owner.

Qualification retains the original two-store cache regression before correction, current preservation/cancellation controls and the same whole-mounted completed-history/original-timer observation after correction. Main-thread handles may legitimately remain until their original owner closes them. Creator return requires their final actual retirement. Keep restart9's functional proof and its separate failed Creator on its exact earlier source, without replaying the full provider flow for this cleanup-only change.


## Task 10 qualification addendum — available failed-check evidence, 2026-11-04

The Console check detail displays each stored check reason. A complete fresh native nonzero exit previously retained the correct failed status and exit but inherited native_evidence_unavailable. For newly published checks, derive native_evidence from the same strict verified_exit predicate already used for conclusive native evidence. This includes both expected and unexpected integer exits with complete output, proven cleanup and current exact manifests/executable/shell. Other evidence states retain native_evidence_unavailable. Preserve native and workflow status handling, full proof, original submitted_digest, exact replay and FULL transaction. Existing historical records retain their old reason label. This narrowly supersedes the earlier keep-check-fields ruling only for future reason publication, with no history rewrite or additional execution authority.


## Final whole-execution correction plan

Reopened after the single final whole-execution review found two Important omissions against the approved contract; the individual Task10 qualification evidence remains source-qualified. ADR required: no new ADR. ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: enforce existing bounded admission and one-nonterminal-workflow-per-chat requirements through their current native transaction and Console owners.

1. Correct joint decision-plus-result-reservation capacity admission inside the existing FULL approval transaction; prove exact-fit, over-limit and rollback behavior for initial/remaining approvals and both storage ceilings, preserving inert duplicates.
2. Enforce saved same-chat nonterminal-workflow admission atomically at native creation, retain same-launch lookup semantics and expose the existing workflow through the public/Console refusal boundary. Preserve imported/history rows and accounting.
3. Correct obsolete integration-stage sentences in the living agent-runtime architecture reference. Keep the unexplained restart9 UI-coroutine warning explicitly disclosed.
4. Run only covering real SQLite/public/UI tests and scoped static checks; preserve original staged61/protected279 bytes. Commit exact owned paths, then obtain the one scoped final rereview and adjudicate any residual findings. No provider replay or full suite.
5. Record the exact correction/review and source limits, complete root archival/disclosure closure, then mark Done. Initial whole-review FAIL is retained separately from the correction verdict.


## Final whole-execution correction notes

Correction `8fc02744869122f961cb92af0b069fccb7b7d6f2` accounts for the decision and all new payload reservations together in the original FULL approval transaction, and atomically refuses a second saved nonterminal workflow in a chat after inert same-launch equality. The body-free refusal carries the exact existing workflow ID through the existing safe Console dismissal/detail path. Historical/imported rows and accounting remain intact. The living runtime architecture now describes the delivered owners. Existing ADR-285 applies; no migration, dependency, scheduler or permission owner was added.

The single scoped final review `89efdf9c44c505ad309a0c9577b0e8b110efe42d654b72526cb2d546aaf8bf29` is SPEC PASS/QUALITY APPROVED: I1/I2/M2 addressed, zero new findings, one retained nonblocking M1. Covering verification passed149 targeted checks with exact committed Python bytes; independent root audits verified all11 correction paths,8326 current source hashes, original staged61/protected279 preservation and four child-copy hashes. Whole-file inherited Ruff/format debt remains qualified; modified range and other checked files pass. Native failure/source/cleanup limits are retained. Full reports and source hashes resolve through [the durable lookup](../../Docs/superpowers/qa/2026-11-01-native-phased-agent-workflows/rulings/index.md).

M1 also recurred in the mounted no-provider covering child at line1176. Neither allocation origin, introduction nor harmlessness is established. No provider replay or full suite. Original whole-review failure and later scoped approval remain separate records. Root archival/disclosure byte validation precedes the final Done status.


Root mechanical closeout: final707-record chronology and all233 archived source identities passed independent byte/hash/coverage/navigation checks; old631 bodies/raw metadata and both deterministic evidence bundles remain exact. [Complete rulings and costs](../../Docs/superpowers/qa/2026-11-01-native-phased-agent-workflows/rulings/complete-rulings.md) and [root archive audit](../../Docs/superpowers/qa/2026-11-01-native-phased-agent-workflows/rulings/root-final-disclosure-audit.json) are durable. Status set Done through Backlog MCP after acceptance; source code remains at the reviewed8fc027 correction. The original all-named-checks-green Task8step7 condition stays explicitly qualified as an unchecked historical execution deviation in the plan.
