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
updated_date: 2026-11-08 21:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace PR3411's orchestration implementation with a thin phase coordinator using the existing native agent lifecycle. Preserve approved coding phases, shared-worktree edits, approvals, bounded execution and useful history. This supersedes continued patching of the rejected coordination design; it is not an application rewrite. Work only in the existing checkout, preserve unrelated changes, and stop shared-source edits while another writer owns them. Task/QA closure and draft PR setup are explicitly authorized in the 2026-11-08 closeout below; merge remains excluded.
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
- [x] #8 Targeted functional, runtime failure/cancellation, permissions, concurrency, restart, storage, UI and ordinary-chat regression evidence plus lint/format and self-review are recorded. Full-suite execution and merge remain excluded; task-only commits, pushes and replacement PR setup are separately authorized in the closeout below.
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
- [x] #20 The explicitly approved minimal inherited CI repairs pass unchanged diagnostic, query-plan, worker and markup gates, with genuine plan evidence and targeted DOM-loss/literal-rendering regressions.
- [x] #21 The derived-artifact job selects a runner that provides its unchanged pinned Python build; the immutable artifact-version gate and all existing checks/budgets remain enforced.
- [x] #22 Passive trace/provider recovery projection reads the currently installed controller without initializing or refreshing configuration, remains responsive during a held configuration rebuild, and yields no recovery state before a controller exists; action-time checks remain unchanged.
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

CI repair follow-up (authorized monitor/fix/push/rerun until green):
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: restore existing native-workflow and ordinary-chat compatibility; no new storage schema or authority boundary.
1. Read all failing job logs and reproduce representative failures on the task tree and untouched native-workflow baseline. Record inherited failures separately from newly introduced regressions.
2. Correct old-version migration fixtures, native optional-owner setup and ordinary hook custody; remove dead implementation-attestation declarations and consolidate workflow Settings integration without raising size budgets.
3. Diagnose mounted capture refusals, retain real source/resource/permission/revision fences, and verify supported ordinary provider and UI paths without restoring implementation-identity checks.
4. Use targeted tests against in-memory Git sources, lint/format and exact branch diffs. Preserve the original checkout HEAD, real index and every staged/unstaged file.
5. Commit only task-related repairs to codex/task-39147-native-phase-rebuild, push normally and rerun Derived Artifacts on the current head until green. No unrelated fixes, newer-dev integration or merge.

CI repair follow-up 2 (2026-11-08), before implementation:
ADR required: no. ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md.
Reason: correct synthetic mounted test-profile ownership and native cold MCP publication acknowledgement while preserving existing source/recovery contracts.
1. Fetch new-head CI logs and reproduce each remaining mounted preparation refusal.
2. Use existing factory profile arguments to give affected native mounted fixtures the configured private profile; retain real DBs and all source checks. Ordinary gateway fixtures must use file-backed capture sources with agent runtime explicitly disabled and exact factory-owned database cleanup.
3. Transfer the native MCP publisher's exact permission candidate/path/binding under the existing ordinary claim even when agent runtime is disabled; consume that resource receipt once without function/code attestation. Run the actual send/recovery regressions and explicit cold/custom-source refusal controls. Fix only demonstrated task-related gaps; do not weaken stock capture or restore implementation checks.
4. Record exact results, preserve the original index/374 paths and the newly observed unrelated untracked `300`, then commit/push only task-related repairs and rerun after active CI ends.


CI repair follow-up 3 (2026-11-08), before implementation:
ADR required: no. ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md.
Reason: correct existing ordinary Stop and physical-read ownership behavior without adding a boundary or relaxing any fence.
1. Retain the actual Space-on-Stop CI failure and isolated local passes separately. Diagnose missing explicit Stop records without widening UI deadlines or claiming the intermittent failure is fixed.
2. Add a deterministic concurrent native-read/stream Stop regression, including app/session attribution, repeated Stop and physical-read retirement. Correct only reproduced control-flow defects; retain native settlement and live-resource protection.
3. Run targeted Stop/read/dispatch controls and mounted Space/Enter checks, update exact task/UAT evidence and preservation, then commit/push only task fixes and rerun after all active task CI jobs finish.


CI repair follow-up 4 (2026-11-08), before implementation:
ADR required: no. ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md.
Reason: make existing mounted fixtures use their actual runtime custody and strengthen poll-lifetime evidence without weakening resource or recovery fences.
1. Retain all three completed old-head failures (12 test jobs pass, 3 fail), including exact Space/Resend/skill-custody logs; cancel only that run's redundant queued aggregate after all tests finish.
2. Seed interrupt projection through existing native precapture/admission/wait APIs so cold skill preparation has its actual original runtime record. Preserve cold/custom-source custody refusals.
3. Preserve the Resend test's original assertion and 0.65-second observation; assert its timer survives the held admission read and include real poll/state diagnostics. Do not turn an isolated local pass into a claimed CI fix or increase waits.
4. Rerun targeted mounted/native-custody controls, retain failures/limits, update task/evidence and preservation, and publish only task changes before dispatching one exact-head workflow.


Fifth CI follow-up (aggregate gate, authorized task scope):
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: direct worker-refusal presentation and diagnostic-record corrections; inherited baseline repairs require separate scope approval.
1. Reconstruct task scope from d5d10f866b versus e08627a after temporary evidence loss; capture fresh full real-index/HEAD/status and 375 path-state preservation before edits.
2. Retain the actual aggregate log and attribute each failure against immutable baseline source. Do not call the workflow green because its 15 test jobs passed.
3. Cover worker refusal after notice removal, then make its presentation tolerate the absent widget. Review the added constant-only native-child diagnostic and update only its owned inventory row.
4. Run targeted regression and static attribution checks, preserve remaining inherited failures honestly, self-review and publish only authorized task changes. Baseline-only fixes remain pending explicit scope approval; no budgets, assertions, resource/revision/recovery gates or test selections may be relaxed.


Sixth CI follow-up (minimal inherited repairs, explicitly approved 2026-11-08):
ADR required: no new ADR
ADR path: N/A (existing ADR356 runtime ownership and ADR150/161 UI conventions remain applicable)
Reason: repair existing DOM-loss/literal-rendering defects and record reviewed diagnostics/real plans for unchanged indexes; no schema, runtime boundary or authority changes.
1. Audit the full real index/status/HEAD and all 375 preserved paths. Advance the existing private index from published 6aaaeb33ab without resetting it; retain prior evidence and the explicit scope answer.
2. Add narrow RED regressions for workflow view completion after partial DOM removal and Console Undo messages containing Rich markup. Protect view projection with NoMatches handling and render external Undo text literally; keep command/revision and cleanup behavior.
3. Capture actual production-query EXPLAIN QUERY PLAN results on the current AgentRuns schema with sqlite_stat1 absent. Pin all three existing workflow indexes with positive plans and dropped-index controls; add plan-pinned census rows only after those tests pass.
4. Review statement changes for all eight inherited stale diagnostic owners against their last pinned source before regenerating the inventory. Retain all privacy/sink checks and do not relax scanners, census counts, assertions or budgets.
5. Run the targeted regressions, all four unchanged failing checkers, affected existing controls and lint/format/diff checks. Record exact RED/GREEN evidence and limits, audit preservation, commit-tree/CAS and push only this task branch, then dispatch one exact-head Derived Artifacts run after checking no task run is active. Follow actual logs until overall current-head success; no merge, newer-dev integration or full local sweep.


Seventh CI follow-up (runner compatibility, before implementation):
ADR required: no new ADR
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
Reason: retain the existing immutable build/runtime contract by selecting a supported runner for its already pinned Python; no runtime, artifact, dependency or authority contract changes.
1. Retain actual aggregate log 128785469019: all 15 test jobs and the four repaired static gates pass, but setup-python cannot provide pinned 3.12.11 on Ubuntu26.04. Verify supported Linux images in the official actions/python-versions manifest.
2. Strengthen the existing Python-pin workflow regression with its supported runner, observe RED, then pin only the aggregate runner to Ubuntu24.04. Preserve Python3.12.11, asset inputs/output checks, all other jobs, timeout budgets and gates.
3. Run targeted workflow positive/negative gate controls, lint/diff and preservation checks. Record actual failures/limits, commit-tree/CAS and push only this task branch. Do not dispatch while the existing run's merge-queue tick is active, and never cancel it; dispatch once after it completes and follow exact-head CI.


Eighth CI follow-up (passive recovery config-lock stall, before implementation):
ADR required: no new ADR
ADR path: backlog/decisions/220-console-human-decision-coordination-ownership.md
Reason: use the existing injected live-controller read contract for passive display; action dispatch and native admission/resource authority stay in their current owners.
1. Retain actual UI10 log128794172577 on6350ea97c1. The captured watchdog stack shows transcript sync -> passive trace recovery -> controller initialization/core-state refresh -> blocking settings read under the held config-rebuild lock. The unchanged case passes in local isolation (20.01s), so do not call that a fix or widen its waits.
2. Force the entire existing recovery-region refresh inside the held-lock response-rule regression, preserving all existing deadlines/assertions and adding a no-controller inert-display control. Publish the actual controller slot in the existing callback fixture. Observe deterministic RED before production edits.
3. Replace passive trace recovery's initializing accessor with the already injected current-controller accessor and make the two provider-recovery display builders read the installed screen controller once; keep the actual current preparation/context-hold/message projection and leave recovery actions/permission/revision/native cleanup unchanged.
4. Verify held-lock maintenance/rebuild Stop cases, callback/action order and no-controller behavior plus targeted recovery controls; compare lint/format, diagnostics and diffs without raising ratchets. Record exact evidence, preserve all checkout/index states, commit/push and rerun only after the active run and its queue tick complete. Never cancel unrelated runs or the tick; no merge/full sweep/newer-dev integration.
Documentation and draft-PR closeout (explicitly authorized 2026-11-08):
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md (existing ADR220 and ADR124 also apply)
Reason: record already verified behavior/CI and task completion; this closeout changes no runtime or architectural contract.
1. Reverify run42509127239 on exact published2d4241b, all22 acceptance criteria and complete real checkout/index/375-path preservation.
2. Use the Backlog CLI on task metadata only to check AC8 and mark Done; record exact source-head CI, qualified local failures and historical evidence without relabeling earlier runs.
3. Verify that only this task and its QA Markdown change, all executable/build inputs equal the green source, task parsing/criteria and links pass, and preservation remains intact.
4. Commit through the existing private index and commit-tree/CAS, push only the task branch, and create a draft PR against dev. Record its inherited baseline/divergence and leave merge/newer-dev integration unqualified.
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


CI follow-up (2026-11-08): the initial task-only Derived Artifacts run
[42426760674](https://github.com/rmusser01/tldw_chatbook/actions/runs/42426760674)
finished its test jobs with failures; overall CI is not green. The task remains
In Progress and AC #8 stays open during the authorized repair loop.

The first repair batch initializes a standalone controller's runtime slot, binds
legacy hook custody to the published runtime, and includes already-installed V2
handlers in the ordinary turn's pre-admitted opaque footprint. The Stop regression
now covers agent-runtime disabled as well as enabled and still proves exclusion
of conflicting work until physical return. Native fixture handoffs use the real
runtime store/controller/bridge APIs; the lifetime stub supplies its actual store.
Synthetic pre-workflow migration fixtures remove later workflow schema before
claiming to be v15/v17. No production migration or source/permission/revision fence
is relaxed. Settings reads provider identity directly, shares existing Goal and
Workflow control glue, and fits the existing size budget; dead attestation
constants are removed and the app's ratchet budget is reduced to its smaller size.
No implementation-identity checks are restored.

The expanded Stop/native-follow-up regression batch passes all 5 cases (48.02s),
and the originally failing mounted Settings streaming-save case passes (62.54s).
Exact partial verification and remaining mounted-preparation failures are recorded
in the [UAT evidence](../../Docs/superpowers/qa/2026-11-07-task-39147-end-to-end-uat.md).
A baseline control also reproduces the mounted preparation refusal on `e08627a`;
that is evidence of an inherited blocker, not a passing compatibility result.
The original dev HEAD, all index entries and all 374 dirty/untracked path states
remain unchanged. Repair commits use only Git blobs and a temporary index.


Second CI repair batch (2026-11-08): the run on `ab472e8` completed all 15 test
jobs with 11 failures and 4 successes. Only its redundant aggregate remained
queued; that task run was cancelled after the tests finished. It is not green.

Native cold MCP permission publication now retains its actual candidate, store,
profile binding and path under the existing opaque ordinary claim when agent
runtime is disabled as well as enabled. The receipt is consumed once; actual
source/path/resource checks remain, and function/code attestation is removed from
that acknowledgement. Regression coverage includes both modes and explicit
replaced-source, path, callback, missing-context and kill-switch refusals.

Affected mounted fixtures now use their configured private profile and real file
storage. Gateway-only fixtures explicitly disable agent runtime; factory-owned
file databases are physically closed even if an app field is replaced. Native
fixture handoffs drain the prior scratch owner and use runtime binding/preparation
APIs. Send observers follow the shared visible request before native/legacy
dispatch; tests wait for actual custody retirement. Snapshot fixtures rebind
imported config-module aliases alongside existing callable aliases, preserving the
native config-installation guard. No production UI, timeout or budget is relaxed.

Targeted checks pass: 15 MCP/publication/refusal controls (39.01s), four mounted
Retry/native-start/Settings controls (79.74s), Send acknowledgement plus visible
attention (2 cases, 32.71s), idle composer Enter plus the complete pending-interrupt
projection journey (2 cases, 125.55s), and the fixture ownership controls (2 cases,
11.79s). Focused Dictate/Stop observer controls pass 2 cases in 58.31s. Separate
earlier representative ordinary Stop/Resend/attention/no-controller
controls passed 4 cases in 77.19s. Cohorts overlap. Exact trees, retained REDs and
limits are in the UAT report. Ruff comparison adds zero findings; diff checks pass.
AC #8 stays open until the actual pushed head passes CI.

The original dev HEAD, full real index and original 374 dirty/untracked states
remain unchanged. A separate writer added untracked `300` between heartbeats;
round-two auditing also preserves that path, for 375 checked states. No checkout
file was edited; repairs and task/evidence updates use Git blobs and the private
index. Existing ADR356 applies; this direct ownership correction needs no new ADR.


Third CI repair batch (2026-11-08): the run on `3a2f2a4` passes the admission
lane and multiple previously failing UI lanes, but its focused Space-on-Stop case
fails to observe the explicit user-stop transcript row. That exact mounted failure
passes in isolation locally; it is retained as intermittent evidence, not called
fixed from those local passes alone.

A deterministic native-read barrier exposes a related Stop defect: an app-scoped
or same-session preparation read caused Stop to return before finalizing an already
streaming response. Stop now cancels those original reads and continues through
existing native stream settlement. Unrelated-session reads remain running, repeated
Stop records one user marker, and finite read ownership remains live until physical
return. No resource, permission, revision, recovery or implementation-identity
check is added or removed by this control-flow correction.

The new regression has 2 failing cases and 1 passing unrelated-session control
before correction (22.63s). Final Stop, dispatch-CAS and repeated native-read
cancellation controls pass 11 cases in 65.84s; both focused Stop UI cases pass in
53.50s. Assertions, test/UI deadlines and budgets are unchanged. Exact packets,
source trees and the original CI qualification are in the UAT report. The task
remains In Progress with AC #8 open for exact-head CI. Existing ADR356 applies;
no new ADR is required. Dev HEAD, full real index and all 375 dirty/untracked
states remain preserved; changes use only the private Git index and objects.


Fourth CI follow-up (2026-11-08): all test jobs on `3a2f2a4` completed with
12 successes and 3 failures. The saved failures are focused Space Stop, Resend
incremental publication and the interrupt journey's cold skill-custody refusal.
Only the redundant queued aggregate was cancelled after every test job finished.
The separate Stop correction is published as `591fcd64b1`; it still requires CI.

The interrupt fixture now seeds through the existing native precapture, runtime
admission and wait APIs. Cold skill preparation keeps its actual original request
record instead of relying on a warm catalog during direct controller submission;
the source/custody refusal itself remains unchanged. The Resend test now requires
its exact poll timer to survive the held admission callback and reports real
state/poll diagnostics on failure. Its original content assertion and 0.65-second
observation are retained; no latency fix is claimed from isolated local passes.

Final mounted Resend and all six interrupt projection journeys pass 2 cases in
193.17s (individual calls 26.39s and 159.73s). Supported skill capture and existing
missing/custom-source refusal controls pass 3 cases in 5.03s. The original CI
failures and the separate local message-pump timeout remain qualified in the UAT
report. No production source changes in this fourth batch, no implementation
attestation restored, no weakened assertion/deadline/budget, and zero new Ruff
findings. Task status and AC #8 remain open for the next exact-head CI. The real
dev HEAD, complete index and all 375 preserved path states remain unchanged.


Fifth CI aggregate follow-up (2026-11-08): d5d10f866b passes all 15 test jobs,
but aggregate 128681770614 fails four static checks. Immutable e08627a controls
reproduce the inherited diagnostic, index-plan, worker-census and Console Undo
markup failures. The task adds one worker-refusal lookup and one constant-only
native-child warning. Added removed-notice regression: 1 failed/1 passed in
3.73s; corrected worker-refusal handling: 2 passed in 3.74s. The exact AgentService
diagnostic owner row is reviewed/refreshed; other stale baseline rows are retained.
Both changed Python files format cleanly, add zero Ruff findings, and diff checks
pass. Source/count attribution and exact packets are in the UAT report. No identity
attestation or relaxed check/budget was introduced; existing ADR356 applies.
Temporary state loss required reconstruction from d5d10f866b's 96-path baseline
diff and a fresh full-index/status/HEAD plus 375-state preservation capture before
edits. The fresh audit passes; lost historical snapshot bytes cannot be recreated.
Scope approval for minimal inherited baseline gate repairs is pending; no knowingly
redundant selected CI run is dispatched. AC #8 remains open and status In Progress.
Later user authorization covers task-only commits/pushes; merge and newer-dev
integration remain excluded. No full local suite or additional live UAT is claimed.


Sixth CI follow-up (explicit scope reply: “Include the minimal baseline CI repairs”):
workflow view handlers now tolerate controls removed during a finite read or before
a queued button event. Busy/read flags still reset, stale checkpoint review is
cleared, and native operations retain all existing revision/permission/cleanup
checks. Undo error/path text and its numeric success message render literally.
No worker/markup census allowance or scanner changed.

All three existing workflow indexes now have genuine no-sqlite_stat1 production
query-plan pins with populated native rows and dropped-index controls. No index,
schema or migration changed. Diagnostic statement review against verified pinned
source c5ceca1247 covers eight stale rows; four previously unlisted workflow owners
(11 calls) were also reviewed and recorded. Those four were present in the original
aggregate log but omitted from the fifth-round handoff summary. Sink topology and
privacy metadata remain unchanged; no diagnostic source was changed.

Isolated RED: 12 failures in 8.35s on faf92ff9c741f2b87c1d02ba1aaf07272609d6ec.
Final executable tree 7867d352de1d7937b0ad32892ed6a47232cc0925 passes 39 workflow/storage
cases (10.48s), 12 Undo unit cases (9.37s), all four unchanged static gates, zero-new-
Ruff comparison and formatting of all five format-clean modules plus the changed
Undo notification ranges. Existing whole-chat-screen formatting debt is retained.
A broader 54-case batch had 51 passes and three mounted Undo profile-setup refusals;
all three also fail unchanged on pre-repair 6aaaeb33ab before exercising Undo.
Separate candidate Undo-only execution reproduces those refusals. They are outside
the selected CI census and are retained as an unqualified local limit, not hidden
by the focused passing cohorts. No full local sweep or fresh live UAT is claimed.
Detailed packets and self-review are in the UAT report. The full real index/status,
dev HEAD and 375 preserved path states still audit unchanged. AC #8 stays open for
actual exact-head CI; no merge or newer-dev integration. Existing ADR356 applies;
these routine view/evidence repairs require no new ADR.


Seventh CI follow-up: run42493042973 on 1a921835b6 passes all 15 test jobs
and all four repaired static gates in Linux CI. Its aggregate128785469019 still
fails: ubuntu-latest now selected Ubuntu26.04, for which setup-python has no
3.12.11 x64 build. The later Mermaid version refusal is a consequence of that
failed setup; the immutable Python/input/output checks remain correct.

Only the aggregate runner is pinned to Ubuntu24.04, which the official
`actions/python-versions` manifest provides for the unchanged Python3.12.11.
Manifest blob25851576302b86c37ffa01ee654d024d6d06b6de records 22.04/24.04 Linux x64,
with no26.04 build. The existing Python-pin regression fails before correction
(1 failure in0.81s); targeted workflow/dispatch controls pass43 cases in5.17s
on executable tree9f6394c7085310d3acb49bc27011a9a9d5a71d2e. Parsed YAML confirms
every other job, step, check, Python pin and timeout budget is identical.
Zero new Ruff findings and diff checks pass; existing CI-test-module formatting
debt is retained. This is configuration/manifest evidence, not an actual Ubuntu
rerun. The old run's queued merge tick128791336060 is left alone and blocks a
new dispatch until it completes. AC8/status remain open for actual new-head CI.
No production/asset/checker changes, relaxed budgets, newer-dev integration or
merge. Full-index/status/HEAD and375-state preservation passes. ADR124's existing
immutable build contract applies; no new ADR is needed.



Eighth CI follow-up (2026-11-08): passive recovery no longer refreshes configuration.
Actual run42501214598 on6350ea97c1 has fourteen successful test jobs and one UI10
failure: held-rebuild Stop times out. Job128794172577's captured UI stack reaches
`load_settings` through transcript sync -> passive trace recovery -> initializing
controller/core-state refresh. That passive method is byte-identical on e08627a
and6350 (SHA256 d4530fe564cb3045646c69729e8081b69ce77d307ecd991e506a180b2ae72ec5),
so this is an approved minimal inherited repair. Local unchanged isolation passes
in20.01s; it does not establish a fix. The new regression forces the full mounted
recovery refresh under the existing held lock, retaining the original1s Stop wait,
2s watchdog and all actual cleanup/current-session/draft assertions. It also covers
live provider recovery reads and absent-controller display. Test-only tree
1b8e8cd4dd771ab98cdc97cf7ca3e3eddb239c29 gives **7 failed in24.62s**, all expected
blocking/initializer refusals (`/tmp/pr39147-ci-round8-region-red.txt`). Earlier
fixture-assignment error is excluded from clean RED evidence.

Passive trace projection now reads the existing injected current-controller
accessor; both provider display builders read the installed screen controller
once. An absent controller yields no recovery state. Actual recovery actions,
preparation/context-hold projection, resource/permission/revision fences and all
budgets remain intact. ADR220 applies; no new ADR. Executable tree
a53cc1078a23f12f043b42ca636550aca25e5fb9: **9 passed in27.60s** (all four held/no-
controller response-rule variants plus five live recovery binding cases), packet
`/tmp/pr39147-ci-round8-recovery-green.txt`. Separate trace/provider recovery
controls give **29 passed,1 failed in19.21s**; the mounted provider case refuses
`raw_source_selection_changed` during app construction, before recovery behavior.
Unchanged6350 reproduces it (**1 failed in9.00s**); packets
`/tmp/pr39147-ci-round8-recovery-controls.txt` and
`/tmp/pr39147-ci-round8-provider-baseline.txt`. No passing mounted-provider claim
or extra fixture repair is made. Prior mounted Undo limits remain recorded.

Formatting-only test lines produce051adf8e2b7e363a1521d979fd8b8aae5b580621 with
identical test AST; production sources stay byte-identical to the passing focused
tree. All four changed Python files have zero new Ruff findings and every changed
range formats cleanly; inherited whole-module formatting debt is retained.
`/tmp/pr39147-ci-round8-verification.txt` proves unchanged logger AST, initializer,
trace/provider action paths and the complete workflow, including runner/Python/
checks/budgets; diff check passes. Self-review traces all three passive callbacks
through the mounted region and confirms no skipped assertion, changed allowance,
new authority or removed fence. Full real index/status/HEAD and375 preserved states
audit unchanged. Later changes are evidence Markdown only. This partial local
overlay is not full-filesystem/isolated-Python or Linux qualification; no full suite,
external-provider/other-platform UAT, newer-dev integration, PR or merge. AC8 stays
open for actual overall success on the eventual current pushed head. Dispatch is
held until the existing run and its separate queue tick complete; neither is cancelled.
Green CI closeout — 2026-11-08 (user-authorized documentation and PR setup).
[Derived Artifacts run42509127239](https://github.com/rmusser01/tldw_chatbook/actions/runs/42509127239)
completed **SUCCESS** on exact [2d4241b78e51bb59f3808fa805bfca0f51ac4e2f](https://github.com/rmusser01/tldw_chatbook/commit/2d4241b78e51bb59f3808fa805bfca0f51ac4e2f),
tree78c789cf0d79d191f6f8e9508df79aaf61c9f462. All15 selected Linux test jobs,
aggregate128832706229 and queue tick128834131516 succeeded (17/17 jobs).
The aggregate covers reproducible styles/Mermaid, pinned Python3.12.11 on Ubuntu24.04,
profile census, diagnostic inventory, task IDs/readability, schema/index decisions,
worker contracts, UTC writers, markup and the intact UI census. The run was
dispatched once; old-head passes were not substituted for this exact-head result.

All22 acceptance criteria are now checked, including AC8's targeted tests,
recorded lint/changed-range formatting and self-review. The earlier native lifecycle,
permission/revision, shared-tree, restart/SQLite, ordinary-chat and mounted UAT
cohorts remain attached to their original versions and bounds. AC20/21/22 are also
supported by actual Linux CI. The task is Done for its stated, scoped acceptance;
this is not a claim that every repository test or inherited ratchet passes.

The user now explicitly authorizes the task/QA closure commit, normal task-branch
push and replacement PR setup. That updates the original publication exclusion;
merge and newer-dev integration remain outside this closeout. Only two Markdown
files change after the qualified source commit. Executable/build inputs remain
byte-identical to2d4241b; its green run does not become an exact-commit result for
the later documentation/PR head. PR-triggered checks qualify that new head.

Preservation is reverified against the fresh round5 capture: real deve08627a HEAD,
full real index/status and all375 path states (including untracked300) unchanged.
This proves continuity from that capture only; older missing temporary bytes cannot
be reconstructed. The completed CI heartbeat was deleted; it is not restarted.

Retained limits: no full local suite, new external-provider/other-platform UAT,
latency certification or unrelated archive lineage. The three mounted Undo and one
mounted provider app-construction refusals reproduce on their recorded unchanged
parents; selected CI and focused passes do not erase them. Inherited whole-module
lint/format/size debt remains. The branch comparison with dev includes94 baseline
commits before the9 TASK-39147 commits,1057 paths from its merge-base, and3584
base-side commits at inspection. Open a draft with that integration debt explicit;
no merge, rebase or auto-merge is certified by this closure.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-39147 meets its22 scoped acceptance criteria and is Done. The replacement
coordinator uses native lifecycle/resource ownership under ADR356, preserves real
permission/revision/recovery fences, and removes orchestration implementation-identity
authority. Follow-up native-return/recovery, ordinary-chat, mounted prompt-editor
and minimal inherited CI repairs retain their targeted RED/GREEN and UAT evidence.

[Derived Artifacts run42509127239](https://github.com/rmusser01/tldw_chatbook/actions/runs/42509127239)
is overall green on exact [2d4241b78e51bb59f3808fa805bfca0f51ac4e2f](https://github.com/rmusser01/tldw_chatbook/commit/2d4241b78e51bb59f3808fa805bfca0f51ac4e2f):
all15 selected test jobs, aggregate and queue tick succeeded. AC8 is checked.
The subsequent closure changes task/QA Markdown only and retains that exact-source
qualification; new PR-head checks remain separate.

Full-suite, new external-provider/other-platform UAT, latency and unrelated archive
lineages are not certified. Earlier failed timings, local baseline fixture refusals
and inherited lint/format/size debt remain documented in the QA report. Real checkout,
index and375 preserved path states remain unchanged. Draft PR setup is now explicitly
authorized; its heavily diverged dev comparison remains an integration/review limit.
No merge, auto-merge or newer-dev integration is part of this completion.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
