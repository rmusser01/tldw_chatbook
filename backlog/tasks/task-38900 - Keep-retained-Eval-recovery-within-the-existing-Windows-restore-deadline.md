---
id: TASK-38900
title: Keep retained Eval recovery within the existing Windows restore deadline
status: Done
created_date: 2026-11-04 22:48
assignee:
- '@codex'
labels:
- bug
- backup-recovery
- windows
- ci
priority: high
documentation:
- Docs/Architecture/backup-recovery.md
- Docs/Architecture/evals.md
updated_date: 2026-11-08 19:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3345's current-head Windows Eval recovery lane times out in the complete retained-definitions restore. Restore work advances through staging and installed validation but exceeds the existing 45-second child deadline. Make the affected recovery complete reliably without weakening the deadline, recovery proof, data retention or file authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original complete-rebackup retained Eval case and affected Eval recovery checks pass on hosted Windows with the restore subprocess deadline still 45 seconds.
- [x] #2 A measured cause and meaningful regression demonstrate the correction; no timeout increases, skips, retries-to-green, fixture data pruning or validation/durability/ownership bypasses are introduced.
- [x] #3 Source and candidate removal, inactive retained definitions, rebackup, rejection controls and physical resource retirement retain their original assertions and behavior.
- [x] #4 Relevant architecture, incident evidence and implementation notes are updated; an independent exact-head review accepts the final change before conditional auto-merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for passive test diagnostics or a mechanical correction preserving all existing recovery boundaries; reassess before an authority, storage or durability contract change.
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md and backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: retain current admission, validation, namespace, source, durability and physical-retirement requirements.
1. Preserve the exact current PR/head/base, failed Windows logs, all-job metadata and original diagnostic artifact. Reproduce the unchanged case locally with the original 45-second restore limit; retain the result as a Mac control only.
2. Measure restore phases and actual call counts. Local profile383 is not Windows proof: Windows uses direct artifact preparation rather than the Mac helper-launch path. Do not infer a Windows cause from local totals.
3. Publish one bounded test-only phase diagnostic through the existing on_staged callback after durable staging/source retirement. Emit only bounded basename/function/line/count/time metadata to the existing captured child log. Keep the original script assertions, restore call, graph and 45-second parent limit. Preserve primary exceptions and observer cleanup. No instrumented success qualifies the final fix.
4. Use actual Windows phase evidence to identify the smallest shared correction. Add a meaningful failing regression and obtain independent review before changing production code. Do not pool helpers, cache validation verdicts, remove barriers, prune data or extend timeouts without an independently justified existing contract.
5. Remove temporary profiling after diagnosis while retaining its source and evidence. Run the original affected cases and connected negative controls uninstrumented; publish a normally committed exact-head reviewed final change while auto-merge is off. Require actual new-head hosted Windows, Perf and required CI.
6. Once genuine gates and DoD pass, complete the finite authorized task reconciliation and independently rebind that metadata head, require its own hosted checks, then enable auto-merge under repository queue rules.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Actual diagnostic head fc259bed Windows run41930900157/job126927201861 failed before restore: native capture rediscovery refused scope_changed after primary ChaChaNotes became unavailable; seven changed identities included two transient sidecar declarations. No restore phase profile was emitted, so no Windows cost conclusion or production fix is selected. Retained original child log and artifact13370879221. Next bounded diagnostic reuses existing tested observe_capture_review around original public child, preserving every native result/cleanup/assertion and emitting only its existing bounded metadata to captured restart-output.log. This identifies exact snapshot refusal/member fields without workflow changes or an unchanged-head rerun. Original restore45 and all source graph/validation/ownership/durability remain unchanged; instrumented success is not qualification.
Source review identified a concrete queued constructor read: app_service_wiring requests Console attention; its retained task runs AttentionMarksRead via run_preparation_read, opens a native DB transaction, then closes/checkpoints its worker connection. Both shared _PUBLIC and _REOPEN fixtures preview synchronously before their first event-loop yield, allowing this accepted initialization read to overlap capture rediscovery. Before fixing fixture ordering, add an actual-stock-App lifecycle regression requiring successful retained task/native tuple return and no preparation reads/offthread native participant leases before unchanged preview/capture. Observe RED, then await the actual retained task before each shared preview (no sleeps/disabling/cancel/retry); retain native concurrent-snapshot refusal controls. This corrects a proven fixture scheduling hazard, not a claim of exact Windows thread attribution or the unmeasured original restore45 cause. Existing ADR125/126; no new production boundary.
Final fixture regression evidence: corrected native stock read observation uses only off-thread/successful-tuple booleans, preserving all three actual fixture marks. The initial empty-projection/empty-return assumptions were wrong and their failures remain retained (epochs387–390); they are not production regressions or counted passes. Final exact regression391 failed on original pending-task ordering in3.56s; exact reviewed wait restored, epoch392 passed all10 original retained/rejection/native-lifetime/startup controls in74.60s (original retained journey47.22s; regression16.37s). Both runs physically reaped with source_changes[]. Epoch385 also passed95 original diagnostic/observer controls. Ruff check/format and diff checks pass on allthree changed Python files. Independent review accepted fixture/diagnostic correctness after both cleanup findings were fixed; Windows restore45 performance and final head review/CI remain open.
Actual Windows b07 run41935498768/job126942809784 reproduced original restore45 timeout (1failed82passed1existingplatformskip559.72s). Retained staged profile wall30.5904: SchemaPolicy immutable4,948,151calls .955self/2.609cumulative; related generator4,968,877calls1.151self. File-write/I/O costs also dominate and remain disclosed. Select only independently assessed mechanical primitive-leaf inlining in SchemaPolicy.__post_init__, preserving exact type(item) in(str,int), exacttuple roots/nesting, validation order/short-circuit and all physical/ownership/durability/fees. No added materialization/caching/authority or buffer change. Not all cumulative2.609s is recoverable; actual uninstrumented Windows45 remains decisive. Before product edit: add frozen-old-predicate bounded property/rejection/short-circuit controls and a deterministic real-installed largest core policy profile-call regression requiring work to follow tuple containers rather than every primitive. Add this regression file to existing Eval workflow selection (no limit/deadline/skip changes). Then meaningful RED→GREEN, real SQLite/forged-policy controls, remove temporary capture/profile diagnostics before final qualification, targeted UAT/static and independent exact-head review. Larger buffering is not selected: shared rollback callers, no prepaid heap seam and pending-buffer disk pricing require separate design. Existing ADR125/126; no new boundary.
Mechanical correction is implemented and independently source-reviewed. Epoch394 meaningfully failed only the real-installed work guard (27,592 calls for125 tuple containers;25semantic controls passed). Epoch395 then passed178 policy/core-owner/real-SQLite/forged-policy/migration/cancellation controls63.07s. After restoring the original retained test byte-exact37dSHA448edf11f4bd14905a78b48bc087439e96e820d971cf9eb927c5047ad5b97425 (removing the two temporary profiling/observer wrappers), exact affected Eval selection396 passed110 uninstrumented cases174.13s, original whole retained journey42.38s. All runs physically reaped/source_changes[]. No Mac timing is Windows proof. Static proof records zero introduced Ruff findings and preserves two inherited model warnings/public error contract; changed tests and all formatting pass. Only models.py differs among3488 production blobs/3050 Python files versus the prior exact37d whole-feature review. Architecture/incident lesson/datedQA describe scope, substantial remaining I/O costs, original limits and pending current-head hosted Windows/UAT. Final committed-head independent review, fresh whole15 mobile/browser and mounted software UAT, current-head Windows45/required/Perf checks and final DoD reconciliation remain open.
Renumbered from this branch's former TASK-38702 before dev integration because the older upstream Console-census task owns that numeric ID. The installed Backlog renumber command preserved this recovery task's body, creation metadata, ACs and status; its prior source/history and hosted45-second evidence remain retained. Replacement ID was checked against local/remote refs and worktrees with a lead buffer. New-head dev integration and CI/UAT qualification remain pending.

### Final software qualification on the accepted source (2026-11-08 UTC)

Source d6f494767bd38fe7462a4200f93ec39658035d81 is independently accepted with zero findings and has passed its OWN genuine software gates. Automatic Eval run42502880149/Windows job128799223321 passed139 unique cases:138passes plus the one unchanged unavailable-FIFO skip. The original complete-rebackup journey passed unskipped119.594s WHOLE NODE, not restore-child elapsed time. Its restore call has no timeout override; the actual shared helper still forwards the original45-second default to subprocess.run. Separate Windows seed/reopen90-second limits are inherited. All26 policy cases and both native-pool error/refusal controls passed unskipped. ZIPsha256 f49781b12e8bedda6e36437011f0a20ae5478e591a41aa90eacf20371642d513 matches the hosted digest; raw log/JUnit/source blobs/command order and independent bounded Windows evidence review reconcile. Passed child logs were not uploaded, so no error-free child-log claim follows.

Own Derived run42502880088 required aggregate128829310062 and all43 software jobs passed, including all32UI shards and every admission/P0/architecture/Dreams lane. Own Perf, actual all-platform Eval/managed/GGUF, Voice, lint and tool-recovery gates passed; no actionable failed check or unresolved thread. The sole pending Merge queue tick at admission was orchestration, not a software qualification gate. Original795browser15passes and796mounted evidence remain valid through measured unchanged runtime/package/UAT-module/helper identity, not a later UAT or whole old Test-map claim. The measured primitive-leaf correction, all original source/candidate-removal/inactive/rebackup/rejection assertions and physical-retirement controls remain. No buffer, deadline, skip, retry, validation, namespace or durability bypass was introduced; earlier failures and profiling limitations remain historical evidence.

AC1–4 and software DoD are satisfied by these actual source-bound results, existing ADR125/126 and the independent exact-source review. B/source-d6f4947-native-Windows-evidence and G/completion-metadata-20261108/source-admission-and-preservation.json retain receipts; G is the repair evidence directory identified by the current checkpoint. Final completion changes only the three actual task records and QA README. All product/Test/workflow/script/packaging trees must remain identical; the metadata successor requires its own exact review and OWN full CI before conditional auto-merge. Original objects, sources, captures, refs, histories and122 artifacts remain preserved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Retained Eval recovery is qualified on source d6f494767bd38fe7462a4200f93ec39658035d81 with the original45-second restore default. Its own Windows original rebackup, all26 policy cases and native-pool controls pass; one unchanged FIFO-primitive skip remains. Validation, durability and physical ownership are preserved. Exact source review and genuine source gates pass. Completion metadata still requires its own review and CI before auto-merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
