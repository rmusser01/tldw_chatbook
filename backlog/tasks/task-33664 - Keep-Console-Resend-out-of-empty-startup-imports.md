---
id: TASK-33664
title: Keep Console Resend out of empty startup imports
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-03 03:47'
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
- [x] #3 App import, storage, CSS and source artifact guards remain unchanged and pass; focused independent review approves.
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

## Required known-work trace integration — October 2

ADR required: no new ADR
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md
Reason: compose landed TASK-33801 known-work idle-check avoidance with existing admitted maintenance and physical cleanup. No new storage/runtime/permission/performance contract.
1. Published a0708ef has all four required jobs PASS and exact-head Qodo clear, with four resolved threads/no page gaps. Live dev2612fc (PR2959 trace known-work and PR2925 audit documentation) is CONFLICTING/DIRTY; strict protection requires the preserving update. Reopen TASK33664 AC3 before implementation and capture exact prior121 and incoming13 hashes.
2. Rebase onto exact2612fc while preserving all prior work. Direct overlap is three files: console_runtime, console_trace_maintenance and legacy migration tests. Retain upstream expect_work initialization/generation signalling exactly; put its flag consumption and conditional idle check in the existing admitted _run_admitted_batch, keeping run_batch/core admission, write recheck, physical custody and SQL unchanged. Retain all upstream new tests and all owned tests. Other118 approved and10 incoming files stay byte-exact. Preserve QA report/data as shipped, without claiming this PR reruns its wider audit.
3. Prove exact hunk composition, all unchanged hashes and twelve-line AGENT_WAKE preflight refund. Run both bounded legacy migration/parking modules, affected runtime scheduling/GC/admission controls and exact wake/refund/nonreplay/close custody neighbors. Verify source-loaded cwd/PYTHONPATH; preserve non-green evidence and all earlier limitations. Incoming TASK33801 remains Done and TASK33802 typing helper-spawn flake remains To Do; no closure or deterministic cause claim for that separate flake.
4. Obtain focused independent immutable review of source, SQL/core/cleanup, signalling/work-flag semantics, wake/refund and artifact/test preservation. After root functional runs and reviewer finish, run the original five storage/import/UI-ready/boot-CSS cases with exact incoming sources and unchanged pins/ceilings/counts/warmup/timeouts/work. Check fatal/added-line/new-file static, CSS and relevant diagnostic/worker/task artifacts; no full suite/dependency installation or foreign cleanup.
5. Recheck AC3/close through CLI only after qualification; update existing plan/review/follow-up/task/PR body with exact evidence and inherited limits. Doc-only closure preserves reviewed hashes. Publish once EXACT observed a0708ef lease, then require fresh current-head Qodo/all four CI jobs/up-to-date protection before normal --match-head-commit merge, verify actual MERGED parents/tree, pause heartbeat.

## Cold worker qualification fixture repair — October 2

ADR required: no new ADR
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: test-only separation of admission-count behavior from the existing legal elapsed-time yield; no production or Performance guard change.
1. Retain initial root54 selection NON-GREEN53pass/1failure154.235s, unchanged isolated1pass5.280s and external real-clock boundary observations. The failure did not record its clock, so do not claim a deterministic load cause; the unchanged contract explicitly permits zero-row yield before first normalization.
2. The owned cold admission-count case keeps every original assertion and real SQLite/worker/admission/retirement behavior, using the existing injected clock seam to remove unrelated wall-clock variability. Production100ms bound and all original Performance sources remain exact.
3. Add a real private-profile fresh-worker controlled-clock zero-row yield and second admitted retry regression. Assert pending row/checkpoint preservation, normalization only on retry, at most two admissions per attempt, and zero registered worker handles after both attempts. This detects ignoring the elapsed guard, deleting/claiming an unprocessed row, dropping pending work, duplicate admission or lost physical cleanup.
4. Verify the controlled yield catches a temporary external ignore-time mutation; run the owned repair and bounded legacy/parking neighbors without changing production. Get fresh independent immutable review with original assertion AST and production/Performance hash preservation; retain all earlier non-green evidence before final original budgets and closure.

## Required Roleplay quit integration — October 2

> For agentic workers: execute inline in the authorized worktree; request an independent immutable review after composition. Preserve the completed source and all earlier qualification records.

**Goal:** Satisfy live strict up-to-date protection while retaining PR #2963 / TASK-33622.14 and every approved orchestration source.
**Architecture:** Preserve the incoming Roleplay draft guard and lazy delegation exactly. Only the generated diagnostic inventory overlaps this patch; compose its owner entries from unchanged production sources, without replacing either side's ownership evidence.
**Tech Stack:** Python, Textual, existing Backlog CLI, Git and existing source-reproduction guards.
**Spec:** Incoming TASK-33622.14 and the user's protected-merge authorization; this adds no product behavior beyond the landed implementation.

ADR required: no new ADR
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/097-boot-budget-ratchets.md; existing ADR199 unchanged.
Reason: mechanical preserving integration of the accepted global quit choke point and existing budget contracts; no new UI, owner, runtime, schema, authority, dependency, gate or ceiling.

Files: preserve eight non-overlap incoming files from dev2d34cbf80d1d7569abf0490e5c9821101d892661; compose Docs/security/production-diagnostic-inventory.json; update this plan, existing review/follow-up and TASK33664 qualification records only. Incoming module-size row tightens PersonasScreen to16397 and must stay exact; all Performance guards remain unchanged.

- [ ] Confirm all four published637ebd jobs PASS, exact-head Qodo clear/four threads resolved, live protection strict/enforced for admins and actual dev2d34cbf MERGEABLE/BEHIND. Reopen TASK33664 AC3 through CLI and preserve prior131 reviewed and incoming9 hashes before rebase.
- [ ] Rebase once onto exact2d34cbf; resolve only actual conflicts, preserving both diagnostic owner deltas. Prove all prior131 non-overlap hashes and eight incoming non-overlap hashes exact, plus every tracked package/Native/Packaging/Performance blob except the two exact incoming Roleplay sources. Leave incoming TASK33622.14 Done as shipped, other programs unchanged.
- [ ] Run Tests/UI/test_roleplay_quit_guard.py, Tests/Architecture/test_quit_flow_prompt_choke_point.py, the two PersonasScreen module-size nodes and affected aggregate navigation/quit neighbors only, with cwd/PYTHONPATH pinned to this checkout. Preserve all positive and non-green results; do not claim new physical Ctrl+Q/relaunch, reference-backend or wider suite qualification.
- [ ] Reproduce the diagnostic inventory and CSS bundles; check fatal/added-line/new-file static plus affected diagnostic/worker/task artifacts. Request independent immutable source/quit/delegation/inventory/guard preservation review; reviewers write reports only, no source/Git/task changes. No new source repair is planned.
- [ ] After functional/review/artifact activity settles, run the original five storage/import/UI-ready/boot-CSS cases unchanged. Record exact sources, limits/counts/warnings and inherited debt; recheck AC3 and Done through CLI only after qualification. Documentation-only closure must preserve the independent source manifest.
- [ ] Publish once with EXACT observed637ebd161272e0ab903bee5ac587205b52d8d5fb lease, update concise PR body, verify refs/body/current-head Qodo and require all four fresh jobs. Merge normally with --match-head-commit only after live strict protection is met; verify MERGED parents/tree/concurrent changes and pause heartbeat.

## Strict-base offline capture preservation — October 2

Goal: retain the next actual strict-protection base acc45cdc2e7ec90157e478a0e4b499084830edf9 (PR2927/TASK33640) before the one exact637 lease publication. This follow-up occurs after all FOUR637 CI jobs passed; no CI for a newly published candidate is running. Roleplay source daf2bd is independently approved; its original editor-setup and bare-profile failures reproduce on exact incoming2d and remain NON-GREEN limits, with unchanged architecture/global-quit/artifact/static positive evidence.

ADR required: no new ADR
ADR path: existing ADR031/097/199 unchanged; incoming TASK33640 owns its shipped capture tooling.
Reason: preserve incoming test/fixture/documentation files and deletion exactly, with no package/native/Performance change or new tool, runtime, provider, schema, permission or dependency decision.

1. Capture the incoming37 surviving file hashes and one retired fixture-tool deletion against exactacc45; verify no direct patch or prior140 manifest overlap. Preserve all140 approved source hashes. Leave incoming TASK33640 In Progress/unchecked AC5 exactly as shipped; do not perform, certify or broaden live captures, no-key network probes or provider allowance changes.
2. Add this prospective plan to existing TASK33664 via CLI before preserving rebase. Rebase onto exactacc45 only; preserve Roleplay/owned production, every Performance source, incoming tests/fixtures and original assertion/marker/config-admission behavior. No test readiness adapter or masking is authorized; automatic review rejected the proposed external adapter and it was not executed.
3. Run incoming capture-tool loopback tests and offline parser/replay/no-auth-fixture/preset tests only. Their existing skipped cases remain skipped and disclosed; do not run capture.py, touch user keys or call live providers. Reproduce affected task/artifact/static checks, authenticate the union manifest and complete independent immutable source/evidence review with inherited Roleplay limits retained.
4. After all root consumers/review/artifact activity settles, run the ORIGINAL five budget cases on the final candidate unchanged, with exact REPO_ROOT cwd/PYTHONPATH. Record count/ceiling warnings and all positive/NON-GREEN/skip limits. Close TASK33664 AC3 via CLI only after its original guards and independent review approve; documentation-only closure must preserve every final reviewed hash.
5. Publish once with exact observed637ebd161272e0ab903bee5ac587205b52d8d5fb lease; verify remote head/base/body/Qodo and four fresh jobs. Do not rebase a newly published head while its jobs run. Normal protected --match-head-commit merge requires fresh-head Qodo/no actionable threads, all four jobs and actual strict up-to-date state; verify MERGED parents/tree/concurrent changes then pause heartbeat.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred the three existing eager Resend imports to actual message resend, refused-echo dispatch and transcript action projection consumers; updated the existing legitimate test mock to the owning module. No new owner, dependency, UI behavior or budget. ADR097 directly governs the repair; no new ADR required.
Original UI-ready RED1034/1033 also reproduces on exact incoming Resend source. Final original five-case tested/untested storage/import/UI/CSS selection passes85.975s with681/686imports and1033/1033UI-ready, unchanged performance-source bytes and original drift warnings/no UI headroom. Real click/keyboard/duplicate worker, held-preflight publication and task33663 authority controls pass; final independent UI/runtime reviews approve2cfbb01c76. Static94patchPython/tennewRuff+format/whitespace and diagnostic/worker/index/UI/timestamp/CSS artifacts pass. Evidence and inherited limits are retained in the final review; no raw-suite/full-suite/live-provider/Windows result is claimed.

Preserved incoming PR2962 implementation of the same three deferred Resend imports, owning-module test mock and full boot-source lesson. Only the upstream module alias method/comments and local test alias differ from approved2cf;92other reviewed Python/CSS files remain exact. Both independent preservation reviews approve immutablea7d9dcbad73ac12ecbc7bd3f515e48a139e0eae9. Whole13-case Resend UI selection passes54.462s and final original five-case tested/untested storage/import/UI/CSS budgets pass75.893s, exit0. Imports remain681/686,UI-ready1033/1033 with unchanged sources, limits and drift warnings. Current subprocess guards explicitly pin PYTHONPATH to exactREPO_ROOT; no main-checkout measurement is claimed. Final93-file fatal/added-line,tennewRuff/format/whitespace and CSS reproduction pass. Exact source/AST and full lesson preservation proofs are retained; inherited upstream task33661 EOF whitespace is outside diff against actualef8. No new ADR; existing ADR097/199 apply. Detailed positive/non-green evidence and separate limits remain in the final review. Fresh published-head Qodo/four checks and protected merge remain.

Preserving PERF-07 integration is independently approved at immutable source 13e261a2582d661d8b41fa3111937659d589241a on dev ecc0a531c855bc9e80906bff90180fd2045f7159. The rebase is clean: all 96 previously approved Python/CSS files and all eight incoming files are exact, with no direct overlap. Incoming config memo remains behind admission and re-verifies path posture; sensitive contexts resolve their memoized raw inputs afresh. Existing ADR-126 D2 applies; no new ADR, source repair, owner, permission, schema, dependency or ceiling.
Independent config/recovery selection passes 11 cases (8.363s); independent replay passes 13 (51.881s), custody/refund/nonreplay/saved-close passes 12 (65.239s). Root incoming memo/bundle/sensitive-path/mounted consumers have 24 passes and one inherited config-retarget fixture failure (45.702s). The same raw_source_selection_changed failure reproduces on exact incoming ecc with cwd/PYTHONPATH pinned (1.511s); both non-green logs/XML are retained, with no gate/test masking. All twelve incoming memo tests pass. Final original five budgets pass 83.786s, exit 0, at 681/686 imports and 1033/1033 UI-ready with unchanged sources, workload and warnings.
Final 93-file fatal/added-line, ten new Ruff/format, whitespace, CSS, diagnostic, worker and 4789 task guards pass. Evidence is /private/tmp/pr2918-perf07-*. All schema76/both gates/frozen AgentRuns and runtime/replay/queue/hook/refund/physical custody bytes remain exact. Incoming TASK33266 retains shipped To Do status; separate open tasks and all earlier limits remain. Fresh published-head Qodo, four CI jobs and normal protected merge remain delivery gates.

Preserving release integration is independently approved at immutable source 353afb6ad48b7336945f4ca959c66c7192e25b7b on dev f3aeb32fb3d230c0774c7fc349729d9f75c96366 (PR #2961, retaining main PR #2950 evaluation parity). All four prior e1377 jobs and edited exact-head Qodo were green before the actual conflict was handled. The sole append-only testing-lesson conflict preserves the complete upstream file and complete 12,598-byte local tail. All 96 reviewed Python/CSS and eight prior PERF-07 files remain exact, as do 16 other incoming files. Across all 3,388 tracked package/native paths only upstream version 0.2.3 metadata and EvaluationSpec.case_sensitive differ. Runtime, schema 76/both gates/frozen AgentRuns, recovery, permissions, replay, hooks, refunds, physical custody and CSS remain approved. Existing ADR-032/097 boot/098 app-only/199 apply; no new ADR, owner, dependency, source repair, gate or ceiling.
Root 46 incoming evaluation schema/client, synchronized metadata, app-only native authority, source-digest and two real mounted initial-hook polling/readiness refusal/restored Resend checks pass in 35.230s, exit 0. Independent immutable review is Ready with no actionable findings: every original API assertion survives, explicit True/False and partial omission remain covered, and removing only the incoming three-line Resend absence entry reproduces the original census bytes. Final original five budgets: 5 passes in 75.284s, exit 0, zero failures/errors/skips; app imports 681/686 and UI-ready 1033/1033. Original drift and stale pytest cleanup warnings remain, with zero UI headroom. All four actual guard sources are exact incoming f3, including the added Resend absence assertion; no pins, ceilings, counts, warmup, timeout or measured-work changes.
Final 93 patch Python fatal/added-line, ten new full Ruff/format and whitespace checks pass; all CSS bundles reproduce. Diagnostics remain 638 owners/16 sinks (1429/56/7604 calls); workers remain 340 lookups/161 functions, 69 waits/27 roots, with no new sites. All 4,791 task IDs/paths are unique/readable. Evidence: /private/tmp/pr2918-release-*. Selections overlap and are not summed. Incoming TASK-33645 and TASK-33803 retain shipped In Progress statuses; no package/index/installed/native release is performed or certified. TASK-33266/33648/33662/33560 and all earlier positive and NON-GREEN limits remain unchanged.
The initial ENOSPC rebase attempt left a clean plan HEAD and no rebase/index-lock state. Only 176 ignored regenerable bytecode cache directories (87,919,735 bytes) in the authorized checkout were removed; all source/evidence and other work were preserved. The same rebase completed 30 commits when disk space became available. No aggregate-resource or performance-cause qualification follows. Fresh published-head Qodo/all four jobs and protected head-pinned merge remain delivery gates; no merge is claimed.

Preserving known-work trace integration qualifies immutable source e963c33d0476ba3ac8c868a8202232e44acfe1a4 on actual dev2612fc. After all four prior a0708ef jobs/Qodo passed, strict protection required the actual conflict update. Upstream initialization/generation/conditional probe compose within the existing admitted batch; all production SQL/100ms bound, runtime/schema/privacy/refund/nonreplay/physical custody and original Performance bytes stay exact. ADR097/199 apply; no new ADR/owner/gate/ceiling.
Initial root54 selection is NON-GREEN53pass/1cold first-row assertion failure154.235s, retained with unchanged isolated1pass5.280s. The admission-count fixture now uses the existing clock seam with all86 original assertions exact; a separate real cold-worker0-row/checkpoint/retry/≤2admissions/zerohandle control detects an external ignore-time mutation at the intended assertion. Both affected modules29PASS37.151s. The original failure clock was unrecorded; no deterministic host-load cause is claimed. Append-only incident lesson preserves the entire prior prefix.
Independent final review approves all131 hashes,129 unchanged entries and test/lesson qualification, no findings. Original5budgets PASS107.723s at681/686imports,1033/1033UI-ready with exact sources/limits/workload and original warnings. Original93/10 static plus owned fixture/whitespace, CSS12, diagnostics638/16, workers340/161/69/27 and4793 task guards pass; inherited whole-fileI001/format remains exact. Incoming TASK33801 Done/TASK33802 To Do/QA artifacts retain shipped status/content; no wider audit or separate program closure. All earlier positive/NON-GREEN limits remain. Documentation closure preserves approved hashes; exact observeda0708ef lease publication and fresh-head Qodo/four jobs/protected merge remain delivery gates.

Required Roleplay/capture preservation is independently approved at immutable79b2f251865ddc16aae052e723d5f6078ec0f5a1 on actual devacc45cdc2e7ec90157e478a0e4b499084830edf9. All177 prior/incoming hashes and the incoming deletion are authenticated; package/native/Performance bytes remain exact against approveddaf2. Existing ADR031/097/199 apply; no new ADR/source repair/owner/gate/ceiling. The original five guards PASS107.713s, exit0, zerofail/error/skip at681/686imports and1033/1033UI-ready, exact incoming guard sources/cwd/PYTHONPATH/limits/work and original warnings. Incoming capture/preset modules145PASS/51retainedSKIP5.432s, loopback/offline only. Static93/10/added-line/whitespace pass with1794 inherited whole-file findings;4794 tasks unique/readable; exact diagnostic/CSS/worker reproduction remains valid.
Roleplay original editor and bare-navigation setup failures reproduce on exact incoming dev; seven architecture/size and three existing global-quit cases pass, while new Roleplay behavior and remaining bare neighbors stay unqualified/unexecuted. The rejected readiness adapter was neither created nor executed. Canonical review retains raw interrupted/NON-GREEN/baseline evidence and no deterministic timing/load cause. Incoming TASK33622.14 Done and TASK33640 In Progress/uncheckedAC5 remain exact; no live-provider/physicalCtrlQ/full-suite/Windows/aggregate-resource/release/wider-audit certification. All earlier evidence and separate programs remain. RecheckAC3/Done follows guards and independent approval; documentation closure preserves177 hashes before exact637 lease publication and fresh-head Qodo/fourCI/strict protected merge.
<!-- SECTION:NOTES:END -->
