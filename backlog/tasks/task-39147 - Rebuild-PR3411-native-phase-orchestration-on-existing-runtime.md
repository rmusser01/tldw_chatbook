---
id: TASK-39147
title: Rebuild PR3411 native phase orchestration on existing runtime
status: Done
created_date: 2026-11-07 15:27
assignee:
- '@codex'
labels:
- console
- agents
- workflows
priority: high
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3411
- TASK-38943
- TASK-38944
documentation:
- backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
- Docs/superpowers/specs/2026-11-07-native-phase-orchestration-rebuild-design.md
- Docs/superpowers/plans/2026-11-07-native-phase-orchestration-rebuild.md
updated_date: 2026-11-07 23:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace PR3411's orchestration implementation with a thin phase coordinator using the existing native agent lifecycle. Preserve approved coding phases, shared-worktree edits, approvals, bounded execution and useful history. This supersedes continued patching of the rejected coordination design; it is not an application rewrite. Work only in the existing checkout, preserve unrelated changes, and stop shared-source edits while another writer owns them. No publication or merge is authorized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tracked ADR, design, implementation plan and salvage/removal manifest identify exact source provenance and preserve unrelated checkout/index content; no hidden source export or new worktree is used.
- [x] #2 One thin Console phase coordinator provides read-only planning, initial-plan and each-phase approval, ordered implementation, verification, independent review and final report; restart requires explicit freshly validated resume.
- [x] #3 Native AgentService/FleetCoordinator own typed child acceptance, cancellation, completion and physical resource drain. No orchestration authority depends on Python function/code/default/closure identity, stack frames, caller locals or self-attestation; ordinary chats have no such workflow gate.
- [x] #4 All agents use the same selected writable local Workspace binding with finite cumulative budgets, strict bounded plan data, dependency order, tool narrowing, actual permission/revision checks and final invocation checks after hook mutations.
- [x] #5 Shared-tree conflicts serialize correctly and scoped writes require observed-content or observed-absence preconditions; unrelated dirty files survive Stop, cancellation and restart without automatic Git mutation.
- [x] #6 Restart reconciliation distinguishes uncertain prior effects from live workers, can finish and release obsolete recovery protections, and preserves live-worker protection without blocking unrelated scratch work indefinitely.
- [x] #7 Bounded private results, history and charges survive restart. Supported result/change-window removal releases quota exactly while retaining history/accounting; existing storage upgrades remain transactional and compatible.
- [x] #8 Targeted functional, runtime failure/cancellation, permissions, concurrency, restart, storage, UI and ordinary-chat regression evidence plus lint/format and self-review are recorded. Full-suite execution, automatic commits, pushes, replacement PR publication and merge are excluded.
- [x] #9 Planning accounting failures after proven native return release real execution resources, retain known charges and unknown evidence, and remain explicitly reviewable; failed native scopes still retain protection.
- [x] #10 Upgraded closed history with saved filesystem ownership and no explicit recovery-review marker retains scoped protection until review.
- [x] #11 A renamed/replaced saved root, including a retargeted symlink, never grants the old history protection over its replacement.
- [x] #12 Unregistered Library construction acquires opaque protection before side effects; runtime-registered native read-only construction keeps bounded concurrency without implementation-identity checks.
- [x] #13 A recovery-journal write failure after proven native return releases real execution/configuration/intake resources, retains honest accounting/history protection and can be retried by explicit review after storage recovers.
- [x] #14 Shutdown closes proven nonlive coordinator entries without requiring historical review and without dropping live or unproven native resources.
- [x] #15 Historical folder fences require saved filesystem ownership and stop blocking replacement roots after projection; planning-only history without a saved root remains reviewable without claiming a new folder.
- [x] #16 Temporary metadata/read failures preserve established recovery fences; confirmed directory replacement or removal still retires only obsolete historical scopes.
- [x] #17 Saving equivalent or future system-prompt settings while planning runs or stops preserves the admitted input, completes native cleanup and permits shutdown without weakening real failed-scope protection.
- [x] #18 Re-delivered closing reviews are idempotent for explicitly reviewed terminal history with known or unknown accounting, preserve revision/liveness checks and never reopen removed payloads.
- [x] #19 The system-prompt editor keeps Apply/Clear/Cancel fully visible and usable at 80×24, including after resizing and Library feedback, with unchanged action results and targeted regression/UAT evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: replace cross-module orchestration and recovery contracts while preserving ADR285's approved product behavior. ADR356 supersedes implementation-attestation amendments only.
1. Verify PR head/base, checkout and concurrent writer status; preserve unrelated dirty/index content. Write replacement design and salvage/removal manifest in ordinary repository paths.
2. Restore the pure runtime boundary using repeatable LoopDeps.validate_tool_call injection alongside the once-per-batch effectful guard. Preserve final provider invocation checks after hook mutation and approval waits.
3. Establish typed native child acceptance/cancellation/completion/drain contracts in AgentService/FleetCoordinator/ExecutionOwner; remove implementation-introspection authority without dropping setup-failure/cancellation cleanup.
4. Replace orchestration with one app-owned phase coordinator, reuse bounded models/history/UI selectively, and remove ordinary-chat workflow attestation.
5. Reconcile uncertain prior effects separately from live workers with a supported protection-release endpoint; preserve charges and live-worker protection.
6. Verify storage/quota/window removal, explicit restart/resume, shared-tree conflicts and dirty-file preservation through targeted tests and a complete native Console journey. Record evidence, lint/format and self-review before Done.
Detailed plan/evidence: Docs/superpowers/plans/2026-11-07-native-phase-orchestration-rebuild.md. Salvage/removal manifest: Docs/superpowers/specs/2026-11-07-native-phase-orchestration-rebuild-design.md. No automatic commits/resets/stashes/pushes, hidden export/worktree, publication or merge; only exact task paths may be staged, preserving existing staged entries.

Execution and evidence: all six stages were implemented inline. The detailed plan
retains each RED, correction, qualified pass and known baseline limitation. The
salvage manifest records exact decisions for independent PR fixes and the current
checkout's supported archive lineage. Final acceptance, preservation and staging
are recorded below after verification.

Follow-up review fixes (authorized 2026-11-07):
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: direct corrections to the approved native-return, historical-review/root identity and runtime-owned effect registration contracts.
1. Turn the four reproduced review cases into targeted regressions; preserve the current index and unrelated file bytes before edits.
2. Generalize returned-planning accounting disposition using actual producer error, physical proof and native attempt rows. Keep known charges and real failed-scope protection.
3. Select unreviewed closed history using the fixed review marker and validate saved root identity before projecting its protection.
4. Replace partial-argument classification with a runtime-owned Library construction resource; keep unknown factories opaque and native read-only behavior bounded. Trace/update its callers and fixtures.
5. Run targeted positive/negative cleanup, recovery, Library and existing native journey checks; lint/format/self-review, update notes/status, and stage only exact task files.
Second follow-up (three re-review defects, authorized 2026-11-07):
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: correct the existing separation of native physical return, durable accounting review and identity-scoped historical denial; no new storage/schema or execution boundary is needed.
1. Save the full index and dirty/untracked bytes, then add failing tests for secondary journal refusal, shutdown without review, planning-only replacement roots and stale installed overlays.
2. Retire proven native resources into a retryable recovery disposition before journal recording. Preserve actual charges/native rows and scoped historical protection; explicit review retries recording after storage recovers.
3. Let shutdown discard only proven nonlive coordinator entries while retaining durable unreviewed history. Preserve actual failed-scope/live-worker protection.
4. Require saved root evidence for historical projection and revalidate installed recovery overlays at admission; never infer ownership of a replacement from its pathname.
5. Run targeted RED/GREEN and native/restart checks, lint/format/self-review, update the task/ADR/architecture/evidence, and stage exact task paths only.
Third follow-up (three review defects, authorized 2026-11-07):
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: correct the already approved ownership, frozen admitted input and idempotent recovery contracts without a new schema or authority boundary.
1. Preserve index/file bytes and add failing regressions for unreadable metadata, same/changed prompt saves during running/stopping planning, and lost closing-review delivery with known/unknown accounting.
2. Preserve uncertain folder ownership on metadata errors; prune only positively observed replacement/removal, retaining real native claims and other current roots.
3. Consume an immutable admitted planning prompt, independent of later session-settings replacement, while retaining actual source/resource/permission and required-payload checks.
4. Recognize already reviewed terminal history as an idempotent closing acknowledgement with fresh revision and live-resource checks; do not reopen removed results.
5. Verify targeted failures/controls and complete native cleanup/restart, update notes/ADR/architecture/testing evidence, and stage only exact task paths.

Prompt-editor UAT defect fix (authorized 2026-11-07):
ADR required: no new ADR
ADR path: N/A (existing ADR150/161 modal/component rules apply)
Reason: routine geometry correction using the canonical scrolling/docked-action pattern, without a new service or ownership contract.
1. Snapshot the full index/dirty files and reproduce clipped painted action bounds in a production-CSS regression at 80×24, with wide and resize/status controls.
2. Make the existing dialog scroll with native docked actions; preserve Library feedback visibility and all Apply/Clear/Cancel results. Rebuild generated CSS using build_css.py.
3. Run targeted prompt-modal tests and design-token/component/bundle checks, then rerun mounted prompt UAT at narrow and wide sizes and inspect screenshots.
4. Update the task and UAT evidence; stage exact task files only and audit unrelated index/file bytes and HEAD.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebuilt workflow coordination on actual native resource ownership. The Console
coordinator retains ordered planning, explicit approvals, task execution,
verification/review and final reporting, while AgentService/FleetCoordinator own
children and ExecutionOwner owns physical drain. Removed orchestration code,
closure, caller-frame and descriptor attestation, including ordinary-chat gates.
The pure loop receives repeatable task validation separately from its once-per-
batch effectful guard; final permission/scope/revision checks remain in place.

Returned canceled planners no longer occupy native capacity solely for unknown
usage or an accounting clock refusal. Explicit recovery review releases only
historical protection, retains unknown charges, and refuses real live/unproven
resources. Native prelaunch fleet fencing returns unused preparations through
cleanup. Post-return delivery errors preserve cancellation without resurrecting
resources. Confirmed history/result removal frees task/window/planning payload
quota. Additive AgentRuns v30 stores immutable review revision outside payload
quota; exact analyzed SQLite archives remain supported. Generic controls cannot
reopen terminal history. Existing UI patterns handle worker refusal without a
leaked coroutine or disabled retry.

Core changes: Agents runtime/service/scope/history; Console preparation/wave/
coordinator and ordinary source integration; AgentRuns migrations/recovery/store;
worktree recovery protection; existing workflow setup/status controls; targeted
fixtures/tests. No new dependency, stylesheet, alternate executor or automatic worktree Git mutation.
ADR required: yes — [ADR-356](../decisions/356-native-phase-orchestration-runtime-ownership.md).
[Architecture](../../Docs/Architecture/agent-runtime.md),
[exact salvage manifest](../../Docs/superpowers/specs/2026-11-07-native-phase-orchestration-rebuild-design.md)
and the incident-based testing lesson were updated.

Validation is targeted: the detailed plan records native end-to-end and restart/
backup evidence, 256 runtime/permission cases, 120 history/storage cases, final
83-case storage/recovery and 61-case bounds/freshness/window cohorts, mounted UI at
two sizes, ordinary-chat wrappers, repeated cancellation and actual failed-scope
controls. Cohorts overlap; counts are not summed as unique tests. Final commands
and qualifications are in the [plan](../../Docs/superpowers/plans/2026-11-07-native-phase-orchestration-rebuild.md).

Follow-up review corrected all four reproduced defects without restoring
implementation-identity checks. Returned planners now retire after actual physical
proof on clock, SQL and post-commit delivery errors; known token charges and
committed plans/completion remain intact, while incomplete settlements stay
review-required. The existing history action also permits review with known
accounting. Legacy closed history requires the immutable explicit-review marker,
and saved directory-chain identity prevents a replaced folder/symlink target from
inheriting a fence. Native Library construction is registered with its actual
runtime/app; arbitrary partials are opaque before effects, late constructor swaps
are refused, and transparent native wrappers/read-only tools stay bounded.

Follow-up targeted evidence: 24 recovery/accounting/Library cases passed; the final
17-case accounting/recovery rerun passed; 18 lifecycle/recovery controls passed,
including all three failed-scope retention cases. Three mounted history cases and
two selected controller/direct-RAG cases passed. Cohorts overlap. The isolated
complete native correction/Stop/backup/restore/rebackup journey passed in 156.23s.
ADR356 and the agent-runtime architecture document describe the corrected
contracts. Exactly 13 existing task-owned paths changed in this follow-up.

The wider UI batch is qualified, not called green: 76 passed, 24 failed, 1 xfailed.
Fourteen failures occur during old config reselection before Library construction;
selected factory/controller cases pass under the existing bootstrap profile. Four
representative field/custody/Chat-start failures reproduce against the unchanged
staged pre-fix code executed only in memory. Other UI parameterizations remain
uncertified. A first concurrent native journey timed out shortly before reaching
the expected state; its isolated rerun passed with unchanged deadlines. See the
plan for exact commands, logs and limitations.

Second follow-up fixes all three re-review defects. Native resource return is
recorded independently of accounting success: a failed recovery journal keeps a
retryable receipt and captured folder protection while the actual owner, source,
configuration and intake/primary slots retire. Review retries with the displayed
revision; an already-committed recording is idempotent. Shutdown drops only proven
nonlive coordinator entries, leaving durable native rows, charges and the review
marker untouched. Historical projection requires saved root ownership, and every
admission path revalidates installed scopes without dropping other current roots
or real live claims. Planning-only history remains reviewable without fencing a
folder by pathname. No implementation-identity checks were restored.

Second follow-up validation: 50 focused accounting/recovery/Library tests passed;
5 delivery/revision and failed-scope controls passed (overlapping cohort); the full
native correction/Stop/backup/restore/rebackup journey passed in 114.59s. REDs,
fixture qualifications and exact logs are in the plan. ADR356, architecture and the
existing incident-based testing lesson describe the corrected contract. There are
no new lint findings or changed-region formatting failures. Exactly 11 existing
task paths changed; original HEAD and all unrelated index/file bytes are preserved.

Third follow-up fixes all three remaining defects. Established folder fences
survive transient I/O, permission and incomplete metadata; positive root
replacement/removal still retires obsolete scopes without releasing live claims.
Planning freezes admitted prompt text rather than checking the identity of mutable
session settings, so equivalent/future prompt saves during running or stopping
preserve provider input and permit actual cleanup/shutdown. A fresh-revision retry
acknowledges already reviewed closed/removed history without mutating its immutable
review marker, charges or removed payloads; real liveness checks remain in place.
No implementation-identity checks were restored.

Third follow-up validation: all 14 new regression cases failed before correction;
final focused regressions passed 71 tests in 382.26s, including real failed-scope,
journal retry, shutdown, startup recovery, replacement-root and ordinary Library
controls. No new lint findings (324 inherited) or changed-region format failures.
ADR356, architecture, the detailed plan and the existing incident-based testing
lesson were updated. Exactly 10 existing task paths changed; unrelated index/file
bytes and HEAD are preserved by the final staging audit.

At third-fix close, complete native journeys were unqualified: two source runs timed
out at the unchanged five-second initial planning wait, while completing cleanup.
An unchanged staged pre-fix comparison loaded only in memory passed planning but
timed out at the 35-second correction wait, reaching the expected state later and
retiring normally. This is a different timing failure, not proof that the current
initial-planning timeout is pre-existing. The earlier 114.59s complete pass belongs
to the preceding source revision. Exact logs and qualifications are in the plan;
no deadlines were relaxed and no failed run is called passing.

Follow-up end-to-end/UAT verification changed no production source. Functional
native correction, fresh verification, independent review, Stop, backup, isolated
restore and rebackup passed in 217.49s with explicit 240s overall/60s status UAT
bounds. The original deadline failures remain qualified, not called passing. Five
mounted acceptance cases passed: four prompt/closing-retry cases in 163.59s and a
real two-phase approval/checkpoint/result journey in 56.37s. The original command/
setup UAT also passed; two original approval probes stopped before workflow actions.
UAT found an unchanged 80×24 system-prompt editor clipping defect, so this is not a
bug-free UI certification. Exact cases, original failures, corrected probe waits,
screenshots and limits are in
[the UAT report](../../Docs/superpowers/qa/2026-11-07-task-39147-end-to-end-uat.md).
Only five verification documents/evidence paths changed; production and unrelated
staged/file content remain preserved.

The prompt-editor follow-up fixes the 80×24 clipping found by UAT. The existing
frame now scrolls and its Clear/Cancel/Apply row docks at the bottom. Library
feedback scrolls after layout refresh so its newly wrapped lines stay above the
footer. All widget ids, action results, safety behavior and visual tokens remain
unchanged; no implementation-identity checks were restored. Existing ADR150/161
apply; this routine geometry correction requires no new ADR.

The new production-CSS regression covers painted action labels, physical clicks
at 80×24 and 140×42, resize round trips, long prompt retention, the entire painted
Library outcome and keyboard focus. Its original clipped-action RED and the
stricter wrapped-feedback RED are retained; all seven final cases pass in 204.41s.
All six existing persistence/Library controls pass against the final code in
213.53s. Mounted running/Stop UAT, actual resource cleanup and
fresh images are recorded with exact timings and qualifications in the UAT report
and plan. The first layout UAT passed all four running/Stop cases at 80×24/140×42
in 379.34s. Both final-code narrow cases passed again in 254.35s after the feedback
timing correction, with unchanged deadlines and proven native shutdown. New test lint/format, inherited-lint comparison and prompt-bundle
reproducibility pass. The broader CSS batch retains unrelated portrait-sheet
drift, unchanged numeric debt and a class-scan timeout as explicit failures.
Only the ten paths in `/tmp/pr39147-prompt-layout-paths.json` belong to this fix;
final index/file/HEAD preservation is audited in
`/tmp/pr39147-prompt-layout-final-preservation.json`.

Limits: no full suite, external provider, cross-platform hardware or unrelated PR
schema-lineage qualification. Four isolated-worktree baseline failures and an
order-sensitive killed-MCP ordinary case are documented, not called passing.
Current-checkout v24–29 history upgrades are supported; the pinned PR's differently
numbered v30 and unrelated migration families were not copied/certified. Existing
lint/format debt is preserved; task changes introduce no new lint findings.
CI handoff authorized by the user after task close: create a TASK-39147-only
branch and commit, push it, and run CI while retaining all original checkout
staging and file content. The branch uses the verified native-workflow baseline
`e08627ac38392fb890984169376b1c242c3a2f3e`; only the 72 task-owned paths are
included in the new commit. It excludes the unrelated staged portrait and Library
work. CI is to run against this published branch. Integration onto the newer,
diverged remote `dev` remains a separate qualification; no merge is authorized.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ADR-356 inline and completed all three review follow-ups without
restoring implementation-identity checks. The 71 targeted regressions cover honest
native return/journal retry, shutdown, established and replaced-root fences,
admitted prompt saves and idempotent closing review/removal.

Subsequent functional end-to-end verification passed in 217.49s, including Stop,
coherent backup, isolated restore and rebackup, using explicit 240s overall/60s
status UAT waits. Five mounted acceptance cases passed, including real two-phase
Approvals/Acceptances, prompt edits during running/Stop and known/unknown lost
closing-review delivery. Original timing and two pre-workflow UI probe failures
remain documented. The subsequent prompt-editor correction keeps all three
actions visible at 80×24 using native scrolling/docking, including after resize
and wrapped Library feedback. Targeted regressions, persistence controls, mounted
running/Stop UAT and painted before/after evidence are recorded in the UAT report.
No bug-free UI or latency claim is made.

Exactly ten task-owned paths cover this layout fix, regression tests, task notes
and evidence. Unrelated file/index content and HEAD remain preserved. No commit,
push, reset, stash, replacement PR or merge. No full suite, remote provider,
other-platform or other PR archive-lineage qualification; broader CSS drift/debt
and scan-timeout failures remain explicitly qualified.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
