---
id: TASK-39147
title: Rebuild PR3411 native phase orchestration on existing runtime
status: In Progress
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
updated_date: 2026-11-08 21:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace PR3411's orchestration implementation with a thin phase coordinator using the existing native agent lifecycle. Preserve approved coding phases, shared-worktree edits, approvals, bounded execution and useful history. This supersedes continued patching of the rejected coordination design; it is not an application rewrite. Work only in the existing checkout, preserve unrelated changes, and stop shared-source edits while another writer owns them. Task/QA closure and draft PR setup are explicitly authorized in the 2026-11-08 closeout below; the subsequent user-authorized integration/merge follow-up below supersedes that earlier exclusion.
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
- [ ] #23 PR3480 integrates the native workflow with current dev, preserves shipped features and runtime fences, passes targeted integration and actual PR-head CI, and satisfies current-head review/thread requirements before authorized merge or queue auto-merge.
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
Current-dev rebase and merge follow-up (explicitly authorized 2026-11-08):
ADR required: existing ADR amendment
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md#current-dev-integration-allocation-pr3480-2026-11-08
Reason: current dev25 review-boundary data collides with the standalone workflow25 lineage; preserve shipped25 and append workflows after it under the existing ownership contract.
1. Pin actual PR head3184178 and remote dev36a21dc, retain full checkout/index/375-path preservation, inspect upstream-equivalent commits and all review/queue requirements. Rebase through immutable Git blobs/private indexes without a checkout, worktree or source export.
2. Replay the commit sequence on current dev, resolving real content/schema/API conflicts against both current callers and ADR356. Keep current-dev features, all resource/permission/revision/recovery fences and unchanged assertion/timeout budgets; never restore implementation-identity authority.
3. Record conflict decisions and add narrow RED regressions for demonstrated semantic gaps; run targeted tests and unchanged static/asset gates on the integrated candidate. Keep historical standalone CI/UAT evidence separate from new integration results.
4. Verify preservation and task scope, update task/QA/PR description, then push the rewritten task branch with an exact force-with-lease. Mark PR ready and use its actual pull_request CI; never dispatch a workflow to substitute for the PR-required rollup.
5. Fetch actual failure logs before minimal repairs and rerun until current-head CI is green. With MERGE_QUEUE=on, arm auto-merge only after Qodo reviewed the current head and every current-head thread is resolved; let the queue own subsequent rebase/merge. No unrelated run or automation cancellation.

Current-dev integration verification/repair sequence (2026-11-08):
1. Finish conflict replay without publishing intermediate trees; preserve current dev review, mix-in, archive, extracted-owner and CI contracts.
2. Remove unused historical attestation residue that references current moved policy names; confirm ordinary config import and CI module collection. Initial immutable replay1df731a fails conftest collection with NameError:get_canvas_execution_enabled; the sole residual record has no callers.
3. Preserve shipped AgentRuns25 review data, append workflow26..31 with exact migration/recovery alignment, and first demonstrate a populated dev25 upgrade regression.
4. Verify each recorded extracted-owner integration gap with targeted existing-module controls, then actual merged diagnostics/artifacts and unchanged checks.
5. Reverify preservation and remote lease, push only the task branch, qualify actual PR-head CI/current-head Qodo review and threads before queue auto-merge.
ADR required: existing ADR356 amendment; no new ADR. Import residue removal is a routine correction implementing its prohibition on implementation-identity authority. All real resource/permission/revision/recovery fences and budgets remain mandatory.

Legacy integration completion: add narrow real-SQLite RED controls for exact standalone workflow25–30 history, then use one additive installed bridge in constructor and archive recovery. Preserve closed/unreviewed history, all old values and audit rows; unknown catalog, failure and cancellation must refuse/roll back. Existing ADR356 amendment above applies.

Extracted child composition correction: the unchanged delayed-child empty-snapshot regression fails on integrated source6f18273 (80 preceding controls pass). Diagnostic-only tracing shows the child using a later parent slot because the shared current-review composition omitted startup_instruction_snapshot. Pass that already-captured argument in the native child kwargs, retaining explicit None and all review composition fields; rerun without diagnostics.

Recovery action fixture parity: the existing live binding fixture fails the same missing session_input_snapshot on unchanged dev36a21d (1 failed3.21s) before passive projection. Supply the current snapshot revision and original result handoff in this fixture; keep action-time production initialization, exact session/draft/order assertions and all recovery fences unchanged.

Native preparation API parity: a name audit finds7 undefined references for current-dev recovery_provider/code_review_provider/on_tool_surface_selected inside the extracted shared preparation. A clean forwarding regression gives3 intended failures3.13s (earlier unrelated bootstrap fixture errors excluded). Add the three existing public parameters to the shared context signature and forward them once; preserve their current construction/action and tool-surface guards.
Review worker start parity: the unchanged current-dev started-before-error control fails on the integrated candidate because the generic uncertainty branch cancels a review worker after Thread.start actually launched it. Preserve the workflow-specific unknown-start retention branch; for an already-started review child, retain its association and proceed through the existing native admission Future and physical worker settlement, matching the shipped review contract. Existing ADR356 applies; no budget or cleanup proof is relaxed.
Reference extracted-owner correction: the unchanged real-expansion control fails because the current plugin-context owner bypasses native custody. Route its first-pass expansion through the existing original reference preparation, retaining one operation through parsing, resolver/Git reads and host assignment. Preserve command literal arguments, rule selector exclusion, exact per-candidate offsets and resumed non-expansion. Keep the parser on the admitted worker; preflight uses only the stock syntax scanner. Existing ADR356 applies.
Review executor classification: unchanged current-dev bounded review execution passes2 controls on dev36a21d but fails before launch on the integrated candidate with native RawCLI executor changed. Select additional stock-executor custody only for the actual stock executor it owns. Custom bounded executors retain the existing ordinary tool operation and review boundary/admission/revision/permission/receipt path, without receiving a stock process-drain claim. Keep every stock executor resource and uncertainty fence and all required-output restrictions; verify unchanged review matrices and native RawCLI custody controls. Existing ADR356 resource ownership contract applies.
Current-dev size ratchets: integrated source exceeds21 unchanged ceilings, including AgentService by1620 lines. Move the existing native execution lifecycle into a same-package helper behind unchanged service entry points, preserving decorators, exact arguments, live owner-module globals and normalized function ASTs. Remove only demonstrated unused attestation data; preserve every actual resource check. Re-run targeted lifecycle tests and unchanged ratchets; never raise budgets. Existing ADR356; mechanical extraction introduces no new owner or contract.
Child scope fixture parity: the unchanged raise-path control reaches the current _persist call but its fixture rejects the new budget_tokens_known keyword before the intended child-only failure. Accept and forward this existing keyword for parent persistence, retaining the exact child exception, scope-order assertions and terminal cleanup checks. No production behavior change.
Generated worker size: the actual unchanged builder produces5911 lines versus the retained5859 ceiling. Preserve all added regular-file, descriptor, bounded-read, compare-and-swap and retained-earlier-effect fences. Remove its demonstrated unused planning-reader data and condense transferred API docstrings while preserving confinement, denylist, ancestor, same-read stamp and per-target CAS contracts. Verify executable AST parity apart from removed unused metadata/docstrings and regenerate solely with the unchanged builder; no artifact/checker/budget exception or new ADR.
Current plugin reference controls: seven unchanged cases fail after restoring native reference preparation because their zero-argument fake resolver has no admitted source or real file. Use an actual private named-Workspace binding and real evidence file; observe calls to the real authority-aware resolver without replacing its result. Preserve literal selector, injection, resume, exact read-count and requirement-ceiling assertions. Separate the macOS root_boot_identity_unavailable setup limit; do not alter that fence.
Current-dev attempt lifetime parity: real admitted file fixtures now pass initial plugin reference projection; four unchanged resume controls still observe reuse of the paused pending plugin run ID. Allocate a fresh pending ID on each resumed Send, preserving the exact frozen requirement ceiling and all other maximum fields. Warm Send read-count controls still show an extra pre-commit read. The attempted constructor keyword was rejected (TypeError; excluded) and reverted: the existing engine already derives its negative-only reader from the target owner. Trace actual extra reads before any further change; preserve fresh launch/admission and native Stop custody.
The retained-requirement fixture's bare object deep-copies to a different identity, so the original maximum fence correctly refuses before plugin verification. Use an unissued immutable stand-in whose copy behavior matches the real opaque ceiling; keep the intended permission refusal and read-count assertions.
Actual hook-read attribution: diagnostic-only thread profiling identifies the extra pre-commit read in console_ordinary_tree._capture_maximum -> v2_configuration, after the original Send read and before the fresh final admission. Share the existing owner/session/current-read projection with initial footprint classification; it can only choose a conservative worktree scope and never selects or grants a hook. Later footprint recapture stays fresh; native borrow still requires OPAQUE for hook effects and every actual launch/admission check remains fresh. Re-run unchanged read-count, foreign/stale, revocation and positive-target controls without diagnostics.
Pinned runner relocation: dev moved the Python3.12.11 setup and artifact checks from the aggregate to preflight. The replay kept Ubuntu24.04 on the now-verdict-only aggregate while preflight still used ubuntu-latest. Move the existing supported-runner pin to the job actually installing that exact Python, and restore dev’s aggregate runner. Prove the existing pin regression targets preflight; every other YAML job, step, dependency, check, version and timeout stays identical to dev. Existing ADR124 applies.
### PR #3480 follow-up for managed qualification, startup and Tool Recovery

Trace each failure to the existing installed SQLite, startup or owned Console boundary. Keep the workflow schema catalogs byte-identical, retain private detached snapshot checks and check actual mounted Stop behavior with its original limits. Verify the original performance tests and leave outstanding failures visible.

ADR required: no. ADR path: [125](../decisions/125-lock-safe-private-sqlite-validation.md), [256](../decisions/256-detached-portable-profile-import.md), [097](../decisions/097-boot-budget-ratchets.md) and [356](../decisions/356-native-phase-orchestration-runtime-ownership.md). These changes reuse the installed SQLite owner, existing startup budgets and the admitted native MCP capture boundary.

Reproduce the two macOS managed-hook failures with the real Darwin boot identity. Replace the provisional initialization test's pre-custody Stop double with a consented legacy Stop hook and observe its real dispatch. Trace the separate post-compaction refusal before changing the production path. Retain approval timing, denial, terminal, audit and memory assertions. Existing ADR356 applies. No new ADR is required for fixture and integration fixes.

Architecture review follow-up. Switch the two new private participant imports to the existing public name for the same operation, then verify the workflow launch and unused settlement paths. Classify the remaining Agents/Chat crossings and identity ratchet deltas against the exact dev base. Keep the existing guards, ownership and budgets. ADR356 applies. The alias migration requires no new ADR.

Cold mounted receipt follow-up. Keep the configuration object and its deferred preparation witness intact through registration. Carry the already issued archive origin separately on the request and add it only to the existing execution copy. Retain the same exact proof acceptance and release operations and verify the original cold receipt, archive origin controls and refusal cases. ADR356 applies. This repairs the existing custody contract and requires no new ADR.

Storage architecture follow-up. Scope legacy catalog assembly to the builder and compare every resulting catalog against the prior tree. Review the counted SQLite staging and bounded input reads against their actual installed boundaries. Keep the architecture checks and recovery behavior. Existing ADR126, ADR256 and ADR356 apply. No new ADR is required because catalog matching, ownership and staging contracts stay the same.

Screen projection follow-up. Move direct turn Undo All planning and per-root outcome assembly into the existing review selection module. Keep the screen event, confirmation, native preparation and apply lifetime, and card publication in place. Keep workflow controller creation and command dispatch in existing Console wiring. Delegate the existing Undo apply sequence with explicit screen callbacks. Verify both original projection ceilings, mounted workflow command routing and Undo All controls. ADR required: no. Existing ADR path: [089](../decisions/089-console-per-turn-change-review-ownership.md) and [210](../decisions/210-console-region-ownership.md). The extraction leaves the same review, consent and revert authorities in place.

Archive catalog follow-up. Project only the existing archive recovery names from the accepted user proof and original current received claim. Keep the existing admission and registry authentication, require the same issuer, turn and session when binding and dispatching, and verify the mounted result and restore cases with the original cleanup and refusal controls. ADR required: no. Existing ADR path: [166](../decisions/166-agent-assisted-archive-recovery.md) and [356](../decisions/356-native-phase-orchestration-runtime-ownership.md). This repairs the existing catalog projection and leaves recovery authorization and storage contracts in place.

<!-- SECTION:PLAN:END -->

### Remaining current-dev CI repair

ADR required: no new ADR.
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md.
Reason: repair the original native phase and metadata owners without adding
storage, tool permission, or execution authority.

1. Preserve the native accepted phase, SQL fences and original returned
   preparation at their owning runtime boundary. Keep Console coordination and
   exact native wave resource validation.
2. Qualify the original stock async MCP metadata read under its actual bounded
   source and policy ownership, retaining opaque admission for other sources.
3. Recheck the published overlap, native workflow, negative controls, original
   startup budgets and the unchanged architecture guard. Review the exact diff
   before publishing the rebased PR branch.

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
Current-dev integration for [PR #3480](https://github.com/rmusser01/tldw_chatbook/pull/3480): replayed onto `36a21dc4d270940e3576b49487a71b37c78896f8`, keeping current review, prompt, archive, tool, hook and extracted runtime owners. Shipped AgentRuns v25 review data keeps its allocation; workflow migrations occupy v26–31. Exact standalone v25–30 workflow catalogs upgrade additively in constructor and bounded recovery, with cancellation, rollback and unknown-catalog refusal. Existing ADR356 records the allocation and compatibility contract.

Preserved current plugin resume IDs, reference offsets and literal/rule behavior using real admitted files; restored native read custody and the same Send-owned hook receipt for initial footprint classification. Fresh final permission/launch checks and OPAQUE effect borrowing remain in place. Review children retain actual started-worker admission and return; workflow unknown starts remain protected. The shared native reply forwards current review/recovery and tool-surface inputs.

Moved existing implementation bodies into smaller helpers behind the original entry points, preserving live owner bindings and checked AST parity. Six measured module ceilings decreased, none increased. Actual source builders reproduce 15 tracked CSS outputs and the 5,853-line remote worker, under its 5,859 ceiling. Diagnostic review matched all 18 changed historical owner pins and reviewed eight new owner locations; only owner/digest rows changed, with identical privacy candidates, sinks and exclusions. The Python 3.12.11 runner pin now follows artifact checks to preflight on Ubuntu 24.04; parsed YAML otherwise equals current dev. Existing ADR124 applies. Three unmerged colliding tasks moved to TASK-39433, TASK-39434 and TASK-39435 with provenance and reviewed references; landed claimants are byte-identical.

Local targeted evidence and its limits are in the QA notes. The previous standalone green run is historical; the rebased PR's actual pull_request CI and current-head Qodo/thread requirements still need verification. AC23 remains open and the task remains In Progress.

### PR #3480 follow-up: reference refusal and lint ratchet

Restored the original visible refusal when reference custody changes; the original body remains discarded. The current dev Ruff ratchet applies to the incoming code: made the 18 new zip length contracts explicit and renamed the internal native Git timeout parameter at its two callers without changing any deadline. Partial submission and receipt-prefix loops still consume only the available prefix. All existing assertions, permissions, retirement checks, timeouts and lint baselines are unchanged. See ADR-355 and the QA receipt; AC23 remains open pending actual PR CI and review.

### PR #3480 Windows checkout follow-up

The Windows managed-plugin job stopped during checkout at a 262-character absolute QA receipt path. Both affected PR workflows now use the repository's existing Windows Git long-path setup before their initial checkout. The repair preserves the receipt and its content-addressed references. Windows qualification is pending on the repaired head.

### PR #3480 qualification and startup follow-up

On fd2872f9500b67d388566c77d20ff6dda4f790a0, [Eval recovery](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733203) completed successfully across all three runner platforms. [Managed plugin qualification](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733161), [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733128) and [Tool Recovery](https://github.com/rmusser01/tldw_chatbook/actions/runs/42539733126) exposed separate failures. The follow-up routes workflow catalog replays and detached normalization through existing private SQLite validation, defers startup imports until their operations need them, scopes workflow button styling by class and records the resolved MCP capture mode after native admission. AC23 remains open. See the [QA evidence](../../Docs/superpowers/qa/2026-11-07-task-39147-end-to-end-uat.md) for qualification limits.

The macOS provisional initialization fixture now observes one actual consented Stop after approved startup and zero Stops on the four refusal paths. Standalone post-compaction still reaches an unresolved MCP ceiling on the exact-head fixture. That failure and the warm UI-ready budget remain open.

The exact PR-head project-chat lane exposed the same MCP state mismatch as Tool Recovery. The staged state transition passes all four mounted project-chat variants and retains the missing-README rejection. The character provenance fixture now installs its intended store on the runtime before screen wiring. All thirteen character cases pass locally. The held-Enter and P0 failures still require verification.

The architecture lane reproduced six PR-specific failures. Workflow child launch and unused settlement now use the existing public name of the same counted storage operation. The participant guard and both targeted workflow paths pass locally. The Agents/Chat boundary and code-identity counts remain open.

The mounted handoff now retains the original captured configuration through registration and carries the issued archive proof separately. It feeds that same proof into the existing execution copy and cleanup. Cold mounted acceptance, queue-before-publication and changed-trust controls passed locally. The existing stock proof cases and the two targeted Console P0 cases also passed locally. See QA for the remaining storage-source and CI limits.

The storage architecture follow-up keeps catalog construction local to its builder and records the actual persistence census changes for detached SQLite staging and read-only workspace inputs. Both architecture modules and all 38 workflow migration cases passed locally, with identical current and legacy catalogs. ChatScreen projection limits and the other architecture-review checks remain open.

The Console projection follow-up moves Undo All planning and apply coordination to the existing review selection module and lazy workflow construction and dispatch to Console wiring. ChatScreen retains confirmation, native currentness and result publication. The original architecture modules passed with 24520 lines and 741 direct methods. The real mounted native card reverted its original file and retired its worker locally. See QA for local fixture and CI limits.

The mounted workflow fixture now selects its named workspace in the real registry before Console resume. Resume previously reconciled the fixture's active chat to Default, which correctly has no ready folder. The existing command, registry-refusal recovery, and Default passive-binding controls pass locally.

The two mounted draft tests now hold the actual native MCP permission read and call through to its original implementation. The other fixtures retain their existing routes. Both original draft assertions passed on af48390f5b6aa062f1bb5098b4277fdd99a61635 in 22.75 seconds, covering the sent notice after a non-append edit and its absence after a hidden draft was replaced. The cross-tab Send refusal still needs a hold while its named UI hook gate owns the attempt.

The cross-tab compatibility test holds the original launch callback while the UI hook gate still owns the spoken Send. The fixture calls through to real admission and keeps the original timeout, named refusal, single-send and preserved-draft assertions. The two Persona handoff cases use the existing in-memory recording store from screen construction onward. All three cases passed on 161fff615aa2d9ced165a10f5dd20b4fc1a613bf in 20.27 seconds. Both changed test modules passed pinned fatal Ruff checks. UI lane 19 also reported a held character render that never streamed and a mounted archive recovery case that refused an uninstalled MCP source. The latter still reproduces locally.

Both failing Environment fixtures now retain the original Change Review root initializer installed by app wiring. Replacing it after the native issuer captured its owner prevented the folder from becoming ready. The two original mounted tests passed on f33d2607e826ab1272e86b66a4f4ad06fa0db922 in 130.57 seconds, covering real readiness, revocation, worker delivery and retired registry leases. Pinned fatal Ruff passed for both fixture modules. The production initializer, fences, assertions and deadlines are unchanged.

Ordinary catalog capture now projects the two reserved archive recovery names only for the app's accepted user proof matching the current same-store received claim. Binding and dispatch require the original runtime issuer and a genuine provider for that same session and turn. The existing registry, resolved Library policy, confirmation and storage checks still run. The mounted archive fixture uses its isolated profile, original permission store and saved Library policy. Refresh and retired-database cleanup passed in 26.67 seconds and all four button and agent restore variants passed in 90.28 seconds on a043a3a3. The catalog binding and dispatch control, existing session close and foreign cursor controls passed in 12.76 seconds on e85bab25. Three separate native dispatch cases still refuse changed storage on the earlier archive scratch and its prior candidate. GitHub verification remains open.

The Response Rules and Inspector layout fixtures now build their apps inside the active isolated profile. The original native MCP source checks accept the fixture's real bound store. All four original Response Rules variants, both held transport cleanup controls and the original collapsed Inspector case passed locally. The private candidate still needs exact-head CI. See the QA report for the baselines and scope.

Native hook command and plugin preparation now share the original worker join in hook ownership. The join preserves actual worker retirement and cancellation propagation. Four native command custody cases passed and the broader native preparation module matched the unchanged baseline's eight failed node IDs. The two architecture guards report one fewer crossing each. Both still fail on other crossings. ADR230 and ADR356 apply to this mechanical ownership relocation.

Two forecast preflight fixtures now await the original skill trust owner and use its real published service before calling the synchronous context capture. All 20 forecast tests and the isolated bound-skill case passed locally. Standalone cold capture and PostCompact remain separate. No new ADR is needed for fixture initialization.

The Console workflow setup, status and history dialogs now have entries and rules in the existing modal wide tier. ADR150 and ADR161 apply. The base widths and live resize behavior were verified on the mounted dialogs. The original command cases, design and performance guards, generated CSS and exact PR selector preflight passed locally. The private candidate still needs exact-head CI.

The citation edit readiness fixture records the original deferred sync callable and checks that stale or cancelled edits schedule nothing. The full 12-case citation readiness module passed locally. Existing citation and edit authority checks still apply. No new ADR is needed for this test-only worker contract correction.

- Admission-sensitive lane 1 fixture checks now use the mounted destination defaults and await publication of the original skill trust before capture. All ten targeted controls passed locally on d1681aa7. Cold native-start custody and the required CI gate remain open.

Four Temporary-chat quit and close cases now use the original profile-bound MCP source. All nine quit and close tests passed locally. Two more draft replay tests use the existing native permission read hold and passed alongside both earlier draft controls. These fixture-only changes leave production custody in place. The published required aggregate failed and AC23 remains open.

The cold readiness fixture observes the real publication worker schedule while leaving its async callable in place. Both targeted readiness cases passed. The image-only Send fixture now observes the actual runtime custody task with the original launch callback installed. All six attachment readiness tests passed together.

Both first-send import guards now reach the provider locally and expose additional cold loads. The existing idle warm-up payload has one module of headroom. Both MCP run-start scheduling failures also reproduce against passing dev controls. These gates remain open.


The Tools chip now uses the actual run-start switch observation when it belongs to the app's live stock MCP source. Unobserved, failed and custom reads retain the original display path. ADR230 applies to the attempt-local observation. Both scheduling failures and the no-observation control passed locally. Direct composition now retains its own switch observation when publishing counts. The catalog fixture explicitly selects the full-tools profile it asserts against. All 14 catalog and scheduling cases passed together. First-send imports and the required CI gate remain open.

Standalone owned PostCompact now resolves the pending MCP definition maximum under the original native hook reservation. It checks the original capture before and after preparation, uses the existing guarded permission publisher for a cold service, and passes a narrowed copy to the existing hook composer. The provider still intersects that maximum with its reviewed component selection. Seven local macOS controls passed, including first publication and refusal of copied configuration and changed service. ADR230, ADR345 and ADR356 apply to the existing ownership and permission boundaries. The required PR check, warm UI budget and architecture-review failures remain open.

The mounted first-send path keeps custody execution, acceptance, persistence and hook initialization on its existing resident owners. Workflow and agent preparation use their original registrations. Reference readers and the writer validator load when those sources are present. Both original import guards passed locally, including Automatic's exact two Library cold loads. The boot and preimport budgets passed. The warm UI census and five architecture-review assertions still fail. ADR230, ADR345 and ADR356 cover the existing ownership boundaries. AC23 remains open.

The existing provider endpoint module now owns the same goal reference constructor, with its result type loaded on use. Goal and workflow authorization compare the same provider, endpoint and credential references. The warm census shed the goal coordinator and script runner and measured 1063/1038. Exact Send import guards, both preimport payloads and five scoped goal controls passed locally. The warm budget, architecture review and AC23 remain open. Existing ADR097 and ADR356 apply.

Console checks the same saved workflow display setting before loading the workflow line. The enabled command and navigation paths passed locally, and the default warm census shed the workflow presentation module at 1062/1038. Existing ADR097 and ADR356 apply. Runtime admission, recovery and the required PR gate remain separate from the view setting. AC23 remains open.

Native plugin hook preparation imports its shared worker join when capturing or joining actual work. The same original issuer and join implementation retain cancellation and resource settlement. Twelve qualified local macOS cases passed, and the warm census shed five hook package modules at 1057/1038. Existing ADR097 and ADR345 apply. AC23 remains open.

The shared provider contract now owns the unchanged endpoint comparison and redaction helpers. Native workflow child preparation imports the same comparison from that owner. Existing Console import paths still work. Scoped endpoint and native launch controls passed, and the upstream guard reports one fewer edge. Existing ADR343 applies. AC23 remains open.

The shared Change Review module now defines the same Undo refusal type. Its original import path remains available. Native ownership can load at startup without loading the revert engine. Eleven engine and four native confirmation controls passed. Warm startup measured 1056/1038. Existing ADR089 applies. AC23 remains open.

The workflow coordinator imports its existing wave module at the original execution and quiescence call sites. The checks and execution bodies are unchanged. Actual overlapping children and coordinator launch and settlement passed with Full agent tools enabled in isolated fixtures. Warm startup measured 1053/1038. The default-light workflow and required PR gate remain open. Existing ADR097 and ADR356 apply.

The original RawCLI issuer now lives with its resident runtime, preserving the initial binding and loading command custody on capture. Five native launch, terminal, refusal and disposal controls passed with exact spawned sources. Two first-send RawCLI persistence cases still fail on both trees. Warm startup measured 1052/1038. Existing ADR094 and ADR356 apply. AC23 remains open.

The native child imports its unchanged first-request plan from AgentService. Console re-exports the same class. Eight qualified native launch, authority replacement and refund cases passed, and the upstream guard removed only the original Console bridge edge. Five architecture assertions and AC23 remain open. Existing ADR356 applies.

Fleet attention loads the existing receipt service at settlement. The same direct and lazy receipt classes and connection checks run for the original callback. Seven focused native controls and both Send and payload guards passed. Warm startup measured 1051/1038. Three mounted cases still hit the same source-selection refusal on both trees. ADR097 and ADR356 apply. AC23 remains open.

The workflow coordinator loads the same automatic work context at its two existing uses. The original context, checks and accounting remain. Seventy-three targeted cases and both Send and payload guards passed. Warm startup measured 1050/1038. ADR097 and ADR356 apply. AC23 remains open.

The raw executor owns the shared invocation context and the original model binding. The same source, tool selection, approval, operation and process cleanup checks remain. Six native controls and both Send and payload guards passed. The upstream scan removed one provider-to-Console import. ADR094 and ADR356 apply. AC23 remains open.

Native commands and Console plugin preparation use the same submission context at its shared hook owner. The original scope, source checks and worker lifetimes remain. Twelve native tests and both Send and payload guards passed. Warm startup measured 1049/1038, and the upstream scan removed one prohibited import. Existing ADR097, ADR230, ADR345 and ADR356 apply. AC23 remains open.

The original plugin hook issuer lives with the preparation lifetime owner. Runtime attaches the same issuer at startup, and the existing worker request loads at capture. Twelve native tests and both Send and payload guards passed. Warm startup measured 1048/1038. Existing ADR097, ADR230, ADR345 and ADR356 apply. AC23 remains open.

Workflow preparation loads the original phase class at its existing exact-class check. The native Undo issuer lives with existing review ownership and attaches at the same Runtime site. Nineteen workflow and four Undo cases passed on the combined tree, including native worker return and actual mounted restore. Both Send and payload guards passed. Warm startup measured 1046/1038. Existing ADR084, ADR089, ADR097 and ADR356 apply. AC23 remains open.

Change Review attaches the same native issuers from the tracking module and loads the original worker sources at capture. Native initialization, snapshot and Undo controls passed locally. One additional failure in the mixed run reproduced on both trees in isolation. Boot and Send payload limits passed and warm startup measured 1045/1038. Existing ADR084, ADR089, ADR097 and ADR356 apply. AC23 remains open.

Workflow storage validation imports its original exact source class from the source owner at the existing check. All 19 native preparation cases and both Send and preimport payload guards passed. The paired warm trace shed console_agent_sources and measured 1044/1038. An untraced run also loaded the timing-sensitive character conversation search and measured 1045/1038. Existing ADR097 and ADR356 apply. AC23 remains open.

The config owner again defines the recorders for its original deferred Collections and first-Send producers. The capture sites and callable records are identical. All 17 scoped native Collections and persistence controls, both Send guards and both preimport payloads passed. Boot measured 685/688, and warm startup shed config_source_records to reach 1043/1038. Existing ADR097 and ADR356 apply. AC23 remains open.

Workflow task narrowing imports the same AutomaticWorkRefused class at its existing raise and catch sites. All 67 task-scope cases, both Send and preimport payload guards and the boot budget passed. Warm startup shed automatic_work_budget to reach 1042/1038. Existing ADR097 and ADR356 apply. AC23 remains open.

Native hooks share the same submission origin enum with Console through the existing agent models owner. The original Console export still resolves to that enum. All 69 installed cases passed on the independent change and 12 passed again after integration. The architecture scan removed one upstream import and still has five open assertions. Existing ADR345 and ADR356 apply.

Workflow and Goal share the original storage lifetime helper with its existing activation owner. Seven scoped native Goal and storage cases passed, including borrowed mounted dispatch, and all 19 native workflow preparation cases passed. The borrowed CLI case still times out in the same retained worker path on the changed and unchanged trees. Both Send and preimport payload guards passed. Boot measured 687/688 and warm startup measured 1041/1038. Existing ADR097, ADR134 and ADR356 apply. AC23 remains open.

Four draft race cases now hold the native permission read and call the original. The committed-spoken-Send case also holds the real Enter dispatch because a busy Console can skip acknowledgement paint. All six targeted cases passed with the original draft, provider and notice assertions. Existing ADR356 applies. AC23 remains open.

The held-Enter scenario waits for the original first-Send settings task to retire before issuing the spoken Send. The actual settings publication fence refused the earlier overlapping attempt. The targeted reruns verified the parked Enter and the original draft, provider and duplicate-send assertions. Existing ADR356 applies. AC23 remains open.

Exact local loaders qualified four detached portable-cache cases and 32 project-chat report and workflow gate checks on the current candidate. The tests use the real cache getter, archive producer and receipt checker. See the QA report for loader scope and startup follow-up. AC23 remains open.

The lane 8 round-trip fixtures hold the original native MCP permission read. The acknowledgement fixture observes actual received intake while leaving the original context builder installed. All seven targeted tests passed with the original paint, draft and provider assertions. Existing ADR356 applies. AC23 remains open.

The workspace executor owns the shared SSH tool-call attachment and binder. Console uses the same attachment for original capture and retirement under ADR181 and ADR356. The focused SSH and import evidence is in QA. The architecture review, warm UI limit and AC23 remain open.

The Console loads compaction hooks on actual compaction. Shared Library search state owns the original query validation, source normalization and result type, and the Console opens the original modal on demand. The existing warm UI census passes at 1038. Both first-Send checks and the payload and boot limits pass locally. Existing ADR097, ADR052 and ADR003 apply. Architecture review and AC23 remain open.

The execution owner now holds the original ordinary tree binding used by native model and tool workers. Console keeps the existing admission, publication and retirement points. Eight Agents-to-Console imports moved to the existing owner. The scoped native, capacity and import checks are recorded in QA. ADR134 and ADR356 apply. Architecture review and AC23 remain open.

Model Raw CLI capture now lives beside the original executor-owned invocation. It still qualifies the same Console runtime, provider and stock executor before taking custody. Native process, import closure and warm census evidence is in QA. ADR093 and ADR356 apply. AC23 remains open.

Native Goal tool dispatch now closes its borrowed Goal storage scope before the admitted ordinary source proves physical return. The existing authorization stays bound across the worker and storage checks. QA records the real borrowed CLI and mounted results. ADR134 and ADR356 apply. AC23 remains open.

The existing Agents run context owns the original workflow preparation binding and readers. Console keeps the exact native reply and registration checks, and its existing readers load the owner on use. QA records the passing native and import controls and the separate changed-receiver failures. ADR097 and ADR356 apply. AC23 remains open.

The first request preserves bounded file reads already selected by the same live native planning catalog. Ordinary light-tool filtering and all invocation checks still apply to their original scopes. QA records actual file reads, plan settlement and remaining refusal cases. ADR285 Task 8F and ADR356 apply. AC23 remains open.

The approved native workflow child keeps its existing fleet and original bounded local tools with the default light setting. The same live preparation and registry supply the tools, and the original task and permission checks govern dispatch. QA records two concurrent native edits under one Change Review window. ADR104, ADR285 Task 8D2 and ADR356 apply. AC23 remains open.

The native wave replacement fixture observes the existing receiving closure in its dispatch module. Both replacement cases now reach their original refusal assertions and leave the outside path untouched. ADR285 and ADR356 govern the unchanged production checks. AC23 remains open.

The shared run context owns the existing agent source issuance and reader used by both native hook families. Console still creates and checks the source at the original site. QA records the source lifetime controls and the remaining architecture failures. ADR356 applies. AC23 remains open.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

### Workflow root provenance

The workflow models own the existing persisted directory-chain fingerprint. The workflow service and returned-planning recovery use the same encoder. The service keeps its existing alias for test callers. The bytes, captured filesystem metadata, approval, worktree protection and recovery rules are unchanged. ADR356 governs the existing behavior and requires no amendment.

On 96628c11, 15 startup, historical protection and native planning cases and all six returned-planning recovery cases passed. The latter include journal failure, recording delivery, unreviewed shutdown and directory and symlink replacement. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scans remove one private crossing (44 to 43), with 14 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Shared Raw CLI runtime qualification

The existing Raw CLI executor checks the same exact runtime class for model shell capture and workflow verifiers. The verifier still requires its original catalog provider and app-installed runtime. Model capture still checks the platform, provider and executor. Existing approval, process and verifier evidence requirements remain in effect under ADR093 and ADR356.

On 61219ddf, all 15 scoped native Raw CLI and workflow verifier cases passed, including the original real spawned worker, model relay, replaced runtime refusal, bounded output and native checkpoints. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan drops only the verifier service's direct Console import (14 to 13), with 43 private crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Agent recovery source observation

Console obtains the existing agent recovery source observation through the activation owner. The owner uses the same guard and task identity, returning the existing accepted tuple or observing the original selectors afresh. The activation guard stays private to these callers and all execution and recovery admission behavior is unchanged. ADR285 and ADR356 govern the existing ownership.

On 0f81aced, 54 storage, reference and recovery activation cases, 15 native hook cases, and six native planning, phase-wave and settlement cases passed. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan drops six private imports (43 to 37), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Unused Textual callback record

The combined tree bb8ba736 removes the unused Textual callback record and its sole import. It changes no dispatch behavior. The real two-child workflow overlap and both received and voice reference controls passed. Both Send, payload and boot guards passed, and warm UI measured 1038. The unchanged scan now counts 67 code references against 56, with the other counts, boundaries and five failing assertions intact.

### Workflow runtime and hook preflight

Workflow callers use the named runtime and idle-hook checks from their existing task-scope owner. The same validation runs at the original call sites. ADR285 and ADR356 govern the existing ownership.

On 8839b56e, eight mounted controls passed, including the four hook custody cases, native two-task overlap, cancellation after baseline, duplicate approval and shutdown after initial commit. The original task scope, child launch and partial-attempt cohort had 99 passes and the same failed repair assertion on both the trial and unchanged ad32a04c. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes four private crossings (37 to 33), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Catalog ownership of built-in source comparisons

The catalog owner compares its original built-in provider type and retained pure planning inventory. Ordinary composition and workflow preparation use the same checks at their existing sites. ADR285 and ADR356 govern this boundary.

On 2c658512, all four native planning controls passed, including actual bounded reads, refusal of direct writes and suppression of optional tools. Both trees returned 12 passes and the same three ordinary catalog assertion failures. Those cases remain open. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes three private crossings (33 to 30), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Workflow operation and capacity checks

The execution capacity owner validates the original operation and capacity objects. The existing launch, wait and close callers validate at their original points. ADR285 and ADR356 govern this boundary.

On 8b284f2d, all 27 original child launch cases and six mounted overlap, refusal, cancellation, duplicate approval and shutdown cases passed. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes one private crossing (30 to 29), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Typed workflow launch results

The existing AgentService result types provide the wait and launch return consumed by Console. The previous private names remain aliases for original callers. The original admission and physical return checks still govern those results under ADR285 and ADR356.

On fdd98e42, five native mounted cases passed, including real overlap, cancellation, failed creation return, duplicate approval and shutdown. The manual-gap child fails at the same assertion on both the trial and unchanged d1cbf5c2. The broader launch comparison matched 30 passes, one exclusion assertion failure and ten freshness setup or teardown errors. Those failures remain open. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1038. The unchanged architecture scan removes three private crossings (29 to 26), with 13 upstream crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Catalog import of the native planning reader

Built-in construction calls the existing native planning reader directly from its run-context owner. The active binding, exact class and runtime registration checks are the same under ADR285 and ADR356.

On 65e9ba2e, four native planning cases and 71 task-scope and ordinary catalog cases passed. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1037. The unchanged architecture scan removes the catalog's upward Console import (13 to 12), with 26 private crossings and the same five failed assertions. The published required check still fails and AC23 remains open.

### Native wave resource checks

The wave owner exposes the original resource validation to the native task binder. All live wave validations and their ordering are retained. Two calls with no wave had no effect and were removed. The original quiescence checks still verify physical return under ADR285 and ADR356.

On d7dd4e7d, 67 task scope and six native wave and coordinator cases passed, including real two-child overlap, cancellation after baseline, shutdown, tool return, changed receiver refusal and coordinator retirement. Both Send and payload guards passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan reports one fewer private crossing, from 26 to 25. Twelve upstream crossings and the same five architecture failures remain. The published PR head still fails the required aggregate and AC23 remains open.

### Unused Library and provider gateway records

Removed two unused Conversation Library metadata records and two unused provider gateway planning records. The live class record, native Library construction registration, planning dispatch and provider loop cleanup remain in their existing owners under ADR356.

The Conversation Library comparison matched 91 passes and seven existing ordinary construction failures. The exact combined tree passed mounted grounded Send. The zero-match reload passed in isolation on both trees, and both mixed runs reproduced the same intermittent stale line after the persisted record was removed. The gateway comparison matched 58 passes and one existing OpenAI Stop transport assertion. All four native planning controls passed on the combined tree, including bounded tools, optional provider suppression and direct-write refusal. The original two Send, two payload and boot guards passed, with 687 boot and 1037 warm UI modules. The original architecture scan reports 64/56 code, 32/27 defaults and 49/43 keyword defaults, with 12 upstream and 25 private crossings. Five architecture assertions and published CI remain red. AC23 stays open.

### Unused Library policy coordinator record

Removed one unused capture tuple from the coordinator. Its actual native validation records, saved policy reads and fail-closed paths remain in place under ADR356.

The policy batch passed 22 cases on both trees. Its parent-only loader could not collect four private child nodes. With the exact child loader all four original project Library policy combinations passed on c896e5e4. Grounded Send and Send Once without Library also passed mounted. Both Send guards, both payload guards and boot passed, with 687 boot and 1037 warm UI modules. The original architecture scan reports 63/56 code, 32/27 defaults and 49/43 keyword defaults with 12 upstream and 25 private crossings. The same five architecture assertions and published CI remain red. AC23 stays open.

### Native agent storage ownership

The existing native agent source, writer validation and physical return checks now live together in the Agents storage owner. Console retains the original aliases. The original selected databases, same-tree check, exact handle return and uncertain cleanup custody remain under ADR285 and ADR356.

On a16c0925, 15 original Goal, physical unwind and native MCP worker cases and six mounted workflow cases passed. Both Send guards, both payload guards and boot passed, with 687 boot and 1037 warm UI modules. The same-loader comparison on the original storage trial matched 84 passes and two reference failures, three passes and 28 ordinary failures, and 38 passes and four scoped hook failures. The stalled negative-drain case was excluded from the scoped comparison and remains open. The architecture scan removes three upstream crossings, from 12 to 9. The same five assertions and the published required check still fail. AC23 stays open.

### Native legacy hook attachment

The existing legacy hook owner names the same source class used by the runtime and workflow preflight. Both test fixtures install their runtime on the fixture app before invoking the normal attachment path. The non-chat case still verifies empty metadata and leaves the chat store and agent bridge unset. Native hook permission, source, process and host return checks remain under ADR345.

On 8cbf1611, all 32 original legacy hook cases and four mounted workflow hook validations passed with real macOS qualification. Both original Send and payload guards and boot passed, with 687 boot and 1037 warm UI modules. The original architecture scan removes one private crossing, from 25 to 24, with nine upstream and the same five failed assertions. Published CI still fails and AC23 stays open.

### Direct Raw CLI storage admission

The existing activation owner exposes the fresh selected-path observation and the original retained Raw CLI storage scope. Console uses them at the existing admission and late receipt storage points. The same source tuple, task and thread admission, physical operation, SQLite handles and cleanup checks remain under ADR285 and ADR356.

On 323a1eb4, ten original native Raw CLI cases passed on macOS with the exact spawned sources. These include cold receipt storage, a real SQLite terminal row, canonical readback, source loss, held admission, host disposal, model relay and worker containment. Both Send and payload guards and boot passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan removes one private crossing, from 24 to 23. Nine upstream crossings and the same five failed assertions remain. Published CI still fails and AC23 stays open.

### First-Send configuration validation ownership

The installed config owner checks the records it issues for first-Send persistence. Console keeps the captured original checker and its existing current module, runtime, controller, store, quiescence and mutation CAS checks. ADR356 applies to this ownership correction.

On 2458dcdb, both trees completed the real post-Send configuration CAS on an isolated app profile. Replacing the captured checker kept the config bytes and identity intact without invoking the replacement. This directly qualifies publication. The separate survivor test still fails before reaching checker replacement, and the two mounted flag tests fail at the same configuration setup on both trees. All 41 Collections and atomic first-Send tests, both Send and payload guards, and boot passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan reports 62/56 code, 31/27 defaults and 48/43 keyword defaults, with nine upstream, 23 private crossings and five failed assertions. Published CI still fails and AC23 stays open.

### MCP catalog entry ownership

The MCP provider reads its original catalog table and entry at the existing ordinary bind. Console retains the collision view, definition ceiling and exact table and entry identity checks at dispatch. The unused Console storage alias is gone. The real storage return operation and its test-facing alias stay in place. ADR285 and ADR356 apply to this ownership correction.

On fa847dff and unchanged 512cc575, the same two provider and mounted composition probes passed. They observed the exact table and entry after stock composition, refused an absent entry and refused an entry from a composed subclass. The five original provider tests passed. Both native batches returned seven passes and the same two failures at the killed-MCP replacement assertion and the deferred-permission worker observation. Those cases remain unqualified. Both Send and payload guards and boot passed. Boot measured 687 modules and warm UI measured 1037. The original architecture scan still has five failures, nine upstream imports and 21 private crossings. Published CI still fails and AC23 stays open.

### Live dev reconciliation and local size checks

The private reconciliation incorporates dev b498bc3a. It retains dev's approval
observations, terminal status reporting, and workflow checks alongside native
retirement and custody. The workflow and remote artifact selection passed
287 cases and the approval selection passed 73 cases on the reconciled sources.

On executable tree 4d19509e, the installed config lifetime owner holds the
original recorders and first-Send validator. Config still publishes the
records from its own namespace. The Personas dictionary statistics import
still loads at use. The original config and Personas size limits pass at
8,956/8,983 and 16,828/16,829 lines. All 216 selected size and boot tests
passed, as did 41 Collections and atomic first-Send cases, both first-Send
import guards, both preimport payload guards, and warm UI at 1,037/1,038.
The exact diagnostic scanner reproduces the reconciled manifest byte for byte.

The exact trial and unchanged reconciliation both completed the direct
first-Send configuration CAS and refused a replaced checker before
invocation. Both dictionary suites returned 18 passes and the same 70
failures, including storage selection refusals in mounted fixtures. The
five architecture-review failures remain open. This is local evidence
against dev b498bc3a. AC23 awaits qualified PR-head CI and review.

On c493f95d, all four original mounted project-chat UAT variants passed
in 115.54 seconds with the real loopback provider and isolated profiles.
The original checker accepted all four complete receipts.


### Ordinary provider observations

The Local and Library providers own their original class comparisons.
Ordinary composition reads the original immutable Local effects and
uses the existing catalog reader for the ordered built-in gate table.

On 2c007c7f, all 201 size checks, 85 catalog cases, both mounted Library
cases, and the six original Send, payload, boot, and warm checks passed.
The original checker accepted complete receipts from all four mounted
project-chat variants. Direct controls matched the original objects and
refused subclasses. The diagnostic inventory reproduced byte for byte.
The original native SSH selection returned four passes and the same
three assertion failures on this tree and a0d1eb69. Four private boundary
crossings were removed. Five architecture assertions remain open.


### Native workflow preparation interface

Console uses the named native child preparation type and the existing
task preparation operation from their Agents owners. The storage check
still captures the same class object. Task preparation retains its
accepted phase, registry, scope, and lock checks.

On 2cae33c5, all 94 task scope and child launch cases, four mounted wave
cases, six finite native return cases, and six Send and startup checks
passed. The size pins and diagnostic inventory pass. Two private
crossings were removed. Five architecture assertions remain open.


### Virtual CLI source interface

The Virtual CLI provider owns the existing static callback observations.
Console reads the same provider and parser registry classes by their named
bindings. Native checks retain the original callback slots, catalog, parser,
executor, and resource return.

On 841d6455, all 127 provider, parser, and SSH composition cases and all
six Send and startup checks passed. The actual macOS native comparison
returned six passes on both trees, including physical SSH return and four
remote source replacement refusals. The same two local Virtual CLI cases
failed on both trees at the nonopaque maximum and callback invocation
assertions. The size pins and diagnostic inventory pass. Four private
crossings were removed. Five architecture assertions remain open.


### Remote dispatch adapter ownership

The remote executor owns the existing wire-to-text adapter and its original
execute and ping records. Console loads the same adapter when it builds the
remote executor and preserves the old import. The typed remote Git and URI
refusals and direct preparation checks remain at their existing boundaries.

The tightened tree b6e602c5 passed all 20 SSH composition cases and all 201
size checks. The controller pin now records the measured 28,728 lines.
Its identical production blobs passed all 12 native SSH cases and all six
Send and startup checks. The combined loopback and composition run matched
the unchanged tree at 59 passes and one read-count assertion failure.
The diagnostic inventory passes. Five architecture assertions remain open.


### Shared workspace SSH custody

The workspace executor owns model SSH capture and the existing request,
master, and session return checks beside their original context binding.
Console retains direct preparation issuance and its original instruction and
registry checks. Both paths share the same physical return operations.
The existing activation owner provides the shared callable source checks.

On 394f197a, all six Send and startup checks, 41 atomic first-Send and
Collections cases, and 17 exact remote bundle cases passed. All four
mounted project-chat cases passed and the receipt checker accepted them.
The native SSH comparison matched 13 passes and the two existing local
Virtual CLI failures. The three initial bare-child probe failures came
from the local test loader inserting the application into those children.
The exact isolated bundle run kept the original guard and used the original
bundle bytes. The size checks and diagnostic inventory pass. Five
architecture assertions remain open with eight upstream dependencies and
six private crossings.


### Workflow planning interfaces

Console uses the existing planning issuer and asks the Raw Shell provider
to compare its live schema with the original prospective schema. The
same registry, bound read scope, policy, and metadata eligibility checks
remain in their original order.

On af246da0, all 99 task scope and Raw Shell provider cases, six native
planning cases, and six Send and startup checks passed. The native
comparison also passed all six cases on the unchanged tree. The size
checks and diagnostic inventory pass. Three private crossings were
removed. Five architecture assertions remain open.


### Native return validation

The prepared reply and ordinary tree owners perform their existing type
and source checks before recovery recording and storage retention. The
phase owner exposes the original unused-acceptance validator. The original
read, error, owner, and physical return checks still run at the same
points before recording or releasing resources.

On fa258d00, six storage and settlement cases and ten planning recovery
cases passed. All ten recovery cases also passed on the unchanged tree.
The unchanged private import guard reports zero crossings, and size and
diagnostic inventory checks pass. Four architecture assertions remain
open. The first mixed import run observed a cold RAG profile read on the
UI loop during manual Send. Both isolated manual controls passed, and all
six original Send and startup checks passed on the repeated mixed run.


### Workspace consent readers

The Workspace consent owner checks the original registry methods at the
existing native hook preparation boundary. Console retains its original
controller, registry, project, and consent checks. The exact installed
reader tuple and registry namespace remain in the check.

On fbebacd9, all 69 installed hook cases, both mounted consent cases, and
all six Send and startup checks passed. The direct source controls refused
replaced methods, instance shadows, changed defaults, and a copied
namespace. All size checks and the diagnostic inventory pass. Four
architecture assertions remain open.


### Native skill publication and hook source records

The app skill owner records the candidate after the actual app slot
assignment using the original marker, caller, thread, task, and slot
checks. Console still checks the original winner and issues the receipt.
The controller hook reader uses the existing callable source record
while retaining its original class, descriptor, and namespace checks.

On 83b39ed9, four native receipt cases passed. The stock trust and
builder comparison matched eight passes and two failures on each tree.
The original mounted Enter and button tests both failed before the
configuration SQL observation and remain unqualified. On 9c9b5eb7,
all 12 installed hook controls, both mounted consent cases, and all six
Send and startup checks passed. All three identity caps, the private
import guard, size, and diagnostic inventory pass. Eight upstream
dependencies remain in the one failing architecture assertion.


### Ordinary execution tree ownership

Agents owns the original admitted execution tree, its exact source and
member checks, and physical return validation. Console retains ordinary
admission and publication. Workflow preparation reads the original local
provider and Raw CLI sources from their owners. The old Console test
import still resolves to the same receipt encoder.

On 84f2ab1e, native two-child overlap, bounded planning reads, all eight
Goal lifetime cases, and all six Send and startup checks passed. The
warm census reached 1038 of 1038, and boot used 687 modules against the
688 limit. The preceding caller update also passed 12 native receipt,
return, and Goal controls. All three identity caps, private import and
size checks passed, and the diagnostic inventory matched byte for byte.
Seven upstream dependencies remain in the one failing architecture
assertion. The earlier provider comparison had 133 passes and the same
two failures on both trees. The Library reservation count and reserved
skill dispatch cases remain unqualified.

On retained tree 5ecb8581, the four mounted project-chat UAT variants
passed, and the original checker accepted all four complete receipts.
The missing-README control failed after the real file read as expected,
and the negative receipt checker accepted the rejection. All seven
native planning and shell checks passed, including primary planning,
three prospective verifier cases and three original revocation cases.

The shared runtime producer module now owns the original finite
preparation read, worker binding, observers, and physical retirement.
Console keeps the original imports for callers and fixtures. Workflow
and the installed Collections initializer use the same read at its
defining module. The original runtime producer class and the moved
finite read body are unchanged.

On executable 10e04ea6, 32 finite read and producer cases, four native
workflow cases, both first-Send guards, both payload guards, and the
boot and warm census passed. Boot measured 686 against the strict 688
limit. Warm measured 1038 against 1038. All four mounted project-chat
cases passed and the original checker accepted every complete receipt.
The diagnostic inventory matched byte for byte. The architecture
batch passed 222 cases, with the same seven upstream dependencies
reported by its one failing assertion. The mixed Collections,
first-Send, and attachment batch matched the unchanged reconciled
tree with 66 passes and one image-only Send return assertion failure.
The image-only case remains open. AC23 remains open.

The image-only Send fixture awaits the original visible action from a
Textual worker. The worker receives the native custody outcome, which
the button pump hands off to the runtime. The original assertion,
staged bytes, provider payload, and turn custody checks are preserved.

On executable 5d32fb86 all 67 selected Collections, atomic first-Send,
and attachment readiness cases passed. Both spoken review outcomes
passed separately. Two observed isolated runs confirmed native
custody acceptance and a successful worker result. The reconciled
tree remains private with the upstream architecture assertion and
AC23 open.

On reconciled tree e1ea1597, all 99 selected architecture and SQLite
inventory controls passed, including the original ownership, projection,
participant naming, and raw connection census cases reported by CI.
All three original lane 6 failures passed locally. Nine admission
controls also passed, including manual Send racing a prepared native
start, all six target revision variants, and two mount cleanup cases.

The lane 15 cross-session Send still timed out at the second provider
gate while the same case passed on dev b498bc3a. The two neighboring
Send-start cases passed. Local observation saw two disjoint scratch
paths, each with an opaque footprint for pending MCP capture. The first
original lease yielded a claim while the second yielded none. The
resource and permission checks remain in place. The upstream dependency
assertion and AC23 remain open.

On the reconciled sources, all eight Response Rules and Inspector
layout cases and all four original Library outcome cases passed locally.
The mounted slash Stop and native archive result refresh passed. All
nine selected installed hook cases passed with the actual macOS boot
identity, including approved provisional initialization, warm and cold
PostCompact, and the changed-source and copied-capture refusals.
The exact published Linux and macOS job results remain separate.
AC23 remains open.


On reconciled tree 6dea1ee1, all four previously failing Perf Guard cases
passed locally, covering boot modules, warm UI modules, the CSS rule
count, and preimport LOC. All nine selected UI lane 8 and 22 cases also
passed, covering pending projection and failed-worker retirement, bound
workspace consent and revocation, Send acknowledgement, and both tab
round-trip draft cases. The published required check and the local
upstream dependency check still fail. AC23 remains open.

On reconciled tree 5cfc7f56, all 146 selected modal tier and citation
readiness cases passed locally. They included the original reverse tier
guard and the citation edit readiness failure reported by lanes 32 and
31. All nine selected Stop, failed-commit discard, and spoken Send
controls passed, including the original failures from lanes 7, 11,
17 and 20 and the adjacent Temporary setup Stop control. The published
required check still fails. Cross-session Send and the local upstream
dependency assertion remain open, along with AC23.

Ordinary Virtual CLI binding records the stock provider callback slots
using the existing descriptor-free observation. The receiving check
refuses a changed callback before review or invocation. The original
local replacement test failed on 52d6a6d6 by invoking the substitute
and passed on a8087cac. Eight native controls passed, including local
held-worker retirement and real SSH stderr return. The original stock
local nonopaque footprint case still failed on both trees. Both Send
checks, both payload limits, boot and warm UI limits passed on the
exact repeat. The first mixed run reproduced an intermittent RAG
import during manual Send. Both isolated manual controls passed.
The upstream dependency assertion, published required check and
AC23 remain open.

### Run-start projection replay fixture

The local run-start coalescing test encountered a full-sync replay already
queued by the original core-state path. Its test-driven tick deferred before
projection even though no sync was running. This occurred after provider
validation. The published lane 3 failure timed out at validation itself.

The fixture uses the adjacent readiness predicate to wait for the real replay
and pending display reads before each existing test-driven sync. It keeps
the original deadlines, held provider, clock, manual ticks and assertions.
The trace on c3ae8586 saw the original pending replay and verified both
manual ticks and the freshness assertions. On 7a4b10a9 all ten targeted lane
3, 4 and 5 tests passed in 95.95 seconds. No new ADR is required for this
test-only change. The published required check and AC23 remain open.


### Reconciled Stop, durable projection and Resend checks

On private tree `d3d630af`, all nine selected published lane 31 tests
passed in 115.60 seconds. They exercised four Stop geometries, three
durable-commit projection cases, vision-block focus and warm layout.
The original citation case was covered separately by the 146-case
modal tier and citation run.

All nine published lane 18 tests passed in 119.93 seconds with a scoped
exception observer. The mounted Resend click completed in place, and the
archive test exercised repeated restores and resource retirement. The
observer saw no mix-in snapshot exception. An earlier mixed run did
refuse the retry after the click, and that refusal remains unexplained.
The 212-case retry, SQLite and Resend comparison returned 209 passes
and the same three viewless attachment failures on the earlier private
tree and live dev. Those cases remain unqualified.

Twelve original Chat Start cases passed in 30.92 seconds. They cover
literal request receipts, manual withdrawal, Stop and mount cleanup.
Cold managed skill custody still needs verification. The published
required check failed and AC23 remains open.

Local evidence is in `/tmp/pr39147-live-dev-published-ui31-1506.txt`,
`/tmp/pr39147-live-dev-ui18-nine-mixin-observe-exact-1501.txt`,
`/tmp/pr39147-live-dev-mixin-retry-storage-controls-1449.txt`,
`/tmp/pr39147-live-dev-mixin-retry-storage-dev-1455.txt` and
`/tmp/pr39147-live-dev-chat-start-lifetime-controls-1506.txt`.

### Cold native Chat Start custody

A Chat Start with an original pending skill snapshot now uses the existing
received turn and custody task. The original coordinator authorization,
configuration and literal draft reach the same durable cutoff and receipts.
The managed task retires through the existing runtime without opening a
prompt chain. The direct warm path remains available under ADR219 and ADR285.

On local tree `5d69fc90`, the cold completion and source Stop regressions
and both warm mounted controls passed. The completion confirmed the
conversation and ledger receipts, literal request, provider output, one
generation charge and complete custody retirement. The source Stop control
withdrew while the actual preparation was held, preserving the original
draft with no durable turn or generation charge. On the same cold setup,
unchanged production refused before acceptance. The observed positive
confirmed the original conversation checkpoint and exact attempt before
terminal settlement retired the row.

The same production blobs passed 14 original Chat Start tests, six
Send/payload/startup guards and the exact diagnostic inventory. The
architecture check still reports the seven upstream dependencies. Published
CI and review remain open under AC23.

Local evidence is in
`/tmp/pr39147-live-dev-cold-start-stop-1527.txt`,
`/tmp/pr39147-live-dev-cold-start-stock-selector-baseline-1527.txt`,
`/tmp/pr39147-live-dev-cold-start-stock-selector-observe-1527.txt` and
`/tmp/pr39147-live-dev-cold-custody-budget-1516.txt`.

### Project-chat receipts after cold Chat Start repair

All four original mounted project-chat variants passed on private tree
`ff075b09` in 118.08 seconds. Each ran in its own profile and used the real
loopback provider. The original checker accepted all four complete receipts.
See `/tmp/pr39147-live-dev-cold-start-uat-1533.txt`, the matching JUnit file,
and `/tmp/pr39147-live-dev-cold-start-uat-check-1533.txt`.

Two of the three previously failing Send-start tests passed on the same tree.
They cover queued and received starts ending at their own provider request.
The cross-session case still failed waiting for the second provider gate,
consistent with the earlier opaque MCP footprint observation.
See `/tmp/pr39147-live-dev-lane15-retained-1533.txt`. The published check and
AC23 remain open.

### Current-dev rebase and migration review

Replayed the 106 published commits onto live dev `b498bc3a` and retained
the locally qualified integration tree. An independent review identified
two stale migration fixtures and a race between AgentRuns constructors.
The runtime now rechecks the legacy catalog and each workflow version under
the same SQLite writer lock as the original DDL. This implements ADR356's
existing additive migration with the same version allocation, SQL and
per-step durability.

The file-backed concurrent regressions reproduced the duplicate v26 audit
row and duplicate legacy review-boundary column before the repair. Both
passed after it, preserving the exact current catalog, original history
and foreign keys. All 75 targeted Goal, automatic work, workflow and
archive recovery migration tests passed. Independent re-review found no
critical or important issue in the migration follow-up.

The cross-session Send case still fails while waiting for the second
provider gate. The architecture review still reports seven upstream
workflow dependencies. AC23 and published CI and review remain open.

### Cross-session Send and native workflow dependencies

Owned Console preparation now captures stock MCP metadata and policy on the
original finite worker before ordinary admission. Disjoint Sends with complete
metadata can hold their own bounded scratch claims. Cold permission publication
respects an existing opaque owner. Custom, unresolved and changed definitions
still require the original admission, and the admitted path retains the downgrade
write and audit. The configuration witness allows independent log publication.

The original accepted phase, SQL fences and wave now live with native execution.
Console retains the actual preparation and coordination. Native planning
validates the phase's original producer at both existing return checks. Remaining
acceptance requires the original issued cell, active reader and physical owner.
All seven upstream edges are gone, and the original architecture guard,
private-import check, identity caps and size limits pass.

On source-identical executable `440e7362`, the three original Send-start cases,
237 architecture and workflow-policy checks, six startup/Send/payload guards,
and four mounted project-chat cases passed. The original checker accepted every
complete receipt. The test-only follow-up `a18eb53d` passed all 16 combined MCP
controls, including cold publication, live deny, source replacement, cancellation,
late log initialization, deferred downgrade and a held opaque owner. The native
workflow repair separately passed real overlap, cancellation, bounded reads,
returned planning, six settlement controls and the Git-backed correction journey.
Independent review confirmed the final fixes and the test-only delta. The larger
MCP comparison reproduced the same eight adjacent failures on the original
candidate, and those cases remain unqualified. PR-head CI remains open under AC23.

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Current-dev integration reopened by the user after standalone task acceptance.
PR3480 must be rebased, conflict-resolved and qualified on its actual PR head before
authorized merge/auto-merge. The earlier22 checked criteria and green source2d4241b
remain historical scoped evidence; new AC23 is open. The latest authorization
supersedes prior exclusions of newer-dev integration, a lease-protected rebase push
and merge. The real checkout/index and every unrelated path remain protected.

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

### Exact-head CI follow-up for PR #3480

Derived Artifacts run 42666948278 failed before the PR test lanes because
two backlog records claimed TASK-39147. The earlier native rebuild keeps
that ID. The later review-recovery task is TASK-40127 under the TASK-19601
owner rule. The original checker now passes both the filename and
frontmatter scan on the repair tree.

The full pinned Ruff scan found the inherited B905 and PLE1205 overages on
dev too. The repair retains used aliases, makes the existing zip behavior
explicit, and preserves both Console log messages and the diagnostic sink.
The exact scan passes the original F and ratchet limits. The tab-sync
fixture now represents the missing workflow row. The compact setup probe
checks the actual worker namespace while retaining the original alias,
issued request and physical return checks. Existing ADRs 355 and 356 apply.
Exact PR-head CI and AC23 remain open.

### November 11 rebase and CI repairs

Replayed the 112 published commits onto dev `ae9f9e8d21`, retaining its explicit
storage provider and shared switch-interval owner at the Console's moved
call sites. The default-config pin now includes the seven commented workflow
settings. Removing those comments reproduces the dev template byte for byte.

The ledger receives the existing fence and dispatch validators from the
original phase issuer before durable acceptance. The accounting driver
installs the original return and evidence validators before its first
quiescence cell. Registration is per ledger, rejects replacement, and uses a
separate lock. Validation and pre-commit checks retain their original
order. ADR-356 covers the native ownership boundary.

The app service readers, Workspace records and execution owner binding live
with existing shared owners. The mounted tests exercise actual profile-owned
workspace consent, streaming and provider selection. Spoken Send carries its
captured draft across command classification, checks whether another Send
already took it, and preserves escaped literal chat. The source checks,
budgets, and the project-chat receipt checker remain in force. ADR-230 covers
the Send receipt and ADR-355 covers the lint guards.

The original admission-sensitive tests, layer and lookup guards, and
startup limits were checked in targeted runs. GitHub required CI and the
current-head review remain part of AC23.

### CI follow-up on rebased PR #3480

[Derived Artifacts run 43085804799](https://github.com/rmusser01/tldw_chatbook/actions/runs/43085804799)
on `56f05ff8d3` exposed recovery-worker fixture drift, native Workspace
test setup on a runner with system Git LFS filters, and UI projection
races. All four project-chat receipts passed and the missing-README
control was correctly rejected after `fs_read`.

Recovery fixtures capture Textual's lazy worker callables and drive the
original replay. The config test uses the actual finite preparation
owner. The positive Workspace fixtures isolate host Git system config
at the existing test command seam. Their full-snapshot readers await
real skill trust publication. The production filter, consent, storage,
and cleanup checks still run.

Library outcome lines use only the current database, session and
conversation. Returning to an already shown path retires obsolete
publication. If a return requests the same key again, only that exact
read can publish. Pending and failed reads still retain valid lines
from the same session. ADR-079 covers the existing durable sidecar.
The Watchlists poll waits for the recreated widget. The Console resume
test observes the original stopped timer and exact resumed callback,
including when independent CORE work needs its own retry under ADR-121.

Targeted verification covered the affected tests, the original Git
refusals and startup limits, and both directions of the new Library
read-order regression. Exact PR-head required CI and Qodo review remain
part of AC23.
