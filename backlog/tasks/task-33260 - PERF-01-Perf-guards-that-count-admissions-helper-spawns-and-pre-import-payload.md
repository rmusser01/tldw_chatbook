---
id: TASK-33260
title: >-
  PERF-01: Perf guards that count admissions, helper spawns and pre-import
  payload
status: Done
assignee:
  - '@codex'
created_date: '2026-09-28 18:01'
updated_date: '2026-10-03 00:03'
labels:
  - performance
  - ci
  - testing
  - perf-audit-2026-09
dependencies: []
references:
  - qa/perf-structural-audit-2026-09-27/report.md
  - qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The two biggest regressions since the 09-04 perf review shipped with every perf guard green. Per-call ADR-126 storage admission and a private-SQLite helper spawn per connection went unseen. The keystroke census reported 0 work per key while each key ran 27-69 guarded load_settings calls. The Console mount profiler is broken by route reuse. The screen pre-import payload ratchet is red (554/500 modules) but perf-guard.yml never runs it. About 14 startup guards are masked by RecoveryRequired raised at module-scope APP_CONFIG. Every later PERF task needs a guard that sees its unit. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-01; every issue with file:line is listed under PERF-01 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The keystroke census and a new idle/visit census count config admissions, storage admissions, private-SQLite helper spawns and open() calls; today's measured counts are pinned as ceilings so any increase fails, with target 0 per keystroke/idle tick documented for PERF-06 (TASK-33265) to tighten
- [x] #2 The screen pre-import payload guard runs in perf-guard.yml; it is either paid down or re-pinned at the measured value with a follow-up task named in the pin
- [x] #3 run_console_mount_profile.py produces a profile against the reusable Console route
- [x] #4 Startup/footer guards previously masked by RecoveryRequired('raw_source_selection_changed') run and report real results
- [x] #5 The stale CSS-source meta-test matches the current source count
- [x] #6 Resend execution stays off first paint and broken-row action projection; the unchanged UI-ready module ceiling holds, and existing Resend click/key, refusal and partial-output behavior passes.
- [x] #7 Mounted pending-projection and orphan-decision Close checks retain every scenario and assertion with independent app/controller/worker lifetimes while avoiding redundant private interpreter startups within unchanged per-child deadlines.
- [x] #8 The connection-evidence normalization comparison uses equivalent initialized readiness builds, retains a nonzero baseline and exact equality, and existing cached/injected/derived support-set contracts remain green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
AC adjustment (controller directive, 2026-09-28): a guard that fails on today's dev cannot merge as a required gate, so AC#1 is now a RATCHET -- measured counts pinned as ceilings (any increase fails), target 0 per keystroke / idle tick documented for PERF-06 (TASK-33265), PERF-08 (TASK-33267) and PERF-09 (TASK-33268) to lower.

1. AC#1: in Tests/Performance/test_console_keystroke_work_census.py, count inside the harness only (monkeypatch, no production change): outermost config admissions (Backup_Recovery.config_participants.operation), storage admissions (storage_admission._acquire_storage, the one seam every acquire_storage import binding routes through), private-SQLite helper spawns (DB.private_sqlite_process.HelperLease.start) and os.open. Add an idle-tick census (credential-poll tick, the 4/s idle unit) and a warm Console-visit census (Console -> Library -> Console on the reusable route). Measure, pin ceilings with comments naming TASK-33265/33267/33268.
2. AC#2: add test_screen_preimport_payload_budget.py to perf-guard.yml; re-pin its budgets at the measured value with TASK-33276 (PERF-17) named as paydown owner, add the ADR-097 exception-ledger row, refresh the snapshot via scripts/update_boot_budget_snapshots.py --only preimport.
3. AC#3: run_console_mount_profile.py -- hook the warm-visit path (ChatScreen.on_screen_resume / Library suspend) instead of on_mount/_on_unmount, which no longer fire per visit on reusable routes.
4. AC#4: make the ~14 startup/footer/TTS guards that hit RecoveryRequired('raw_source_selection_changed') under the per-test env redirect run for real (bootstrap_profile marker / private_profile_test, the conftest's established opt-ins). Stop and report if it needs a production change.
5. AC#5: fix test_snapshots_are_real_not_hollow (6 -> 5 CSS sources).
6. Verify: full Tests/Performance vs a pristine 9cd9aad65f worktree baseline; preflight.sh.

PR2953 CI follow-up (2026-10-02): ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md; existing ADR-125/126 real admission seams. Reason: test-only correction of the existing settled-idle census, with no runtime, authority or ceiling change. Reproduce the cold pending migration counting17 admissions over8 ticks; run and assert its real one-time completion outside the measured phase; retain8 warm ticks, every real counter and anti-vacuity canary; run the exact plain guard and relevant trace completion/parking checks, lint and artifact preflight, then require all final-head PR gates and resolved Qodo before normal merge.

PR2953 typing-fixture follow-up: ADR required: no; existing ADR-097/125/126 apply. Call-through diagnostics found the five-second media startup helper finishing within milliseconds of the first measured key; it is a plain asyncio task outside the Textual worker drain. Hold only this private fixture startup schedule and await the same real cleanup once before counting. Retain all measured operations, ticks, counters, canaries and ceilings; run the exact plain guard plus the existing census canaries and inspect actual cleanup evidence. The original aggregate typing-helper failure was not directly attributed; this removes the confirmed unowned setup window, without claiming production cost reduction.

PR2953 latest-dev UI-ready follow-up: ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md. Reason: mechanically place existing pure eligibility in the already-resident message-action owner and defer execution to its existing first-use paths; preserve public exports and behavior, with no new runtime boundary or budget exception. Upstream dev and this PR both fail at 1034 modules against unchanged 1033. Add a fresh-process guard proving broken-row action projection does not import execution, and pin execution absent in the real warm UI-ready census. Move the three existing pure helpers into console_message_actions, keep compatibility re-exports, and defer the two execution imports until explicit Resend. Adjust the defining-module monkeypatch. Verify real warm census, full affected Resend shape/action/click/key tests and boot guard neighbors; peer review, artifact/lint checks and current-source native qualification; require fresh final-head Qodo/all four CI gates before normal merge. No ceiling or snapshot refresh.

PR2953 UI timeout follow-up: ADR required: no. ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; existing ADR-126 private-profile test isolation. Reason: remove redundant private interpreter startup only in this workstream's mounted checks, preserving all assertions and fresh per-scenario app/controller/store/worker ownership. Exact head8e UIjob111015408914 reached74% without assertion failure, then hit unchanged20m limit; its four pending-projection children cost111s and Close file353s. Consolidate only four projection journeys into one bounded private child with distinct case dirs, and both no-owning-turn Close kinds into their existing local-loop pattern. Preserve every body, safe cleanup, independent apps and unchanged180s deadline. Measure the affected mounted groups, prove unchanged scenario bodies, review isolation, then rerun combined latest-dev guards/native/preflight and fresh final-head CI. Do not collapse the full Close file or change global CI/caps.

PR2953 latest-dev evidence-counter follow-up: ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md. Reason: test-only setup correction of the inherited TASK-33005.2 normalization comparison. Its isolated first baseline counts3669 versus9 after support-set initialization; source arithmetic identifies exactly3660 static catalog normalizations. Perform one real readiness build without evidence before profiling either branch, preserving the nonzero baseline and exact equality assertion and measuring the first shared-evidence lookup. Do not alter production, support-cache semantics or any census ceiling. Verify the exact comparison plus existing cache/keyed-evidence guards in a fresh private process; retain the red log and record the current-base validation.

PR2953 remaining UI timeout (2026-10-02, head45189cd3): ADR required: no. ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; existing ADR-126 private-profile isolation and ADR-097 ratchets apply. Reason: test-only grouping of this workstream's existing mounted Close scenarios; no production, authority, deadline, CI setting or budget change. UI Fast Lane reached97% with no assertion failure before its unchanged20m limit; the14 Close children took273.95s. Measure nine navigation/recovery nodes through their ordinary private wrappers, then place their unchanged complete scenario bodies into two bounded private children with fresh apps/controllers/stores/workers, distinct real-DB directories, per-scenario monkeypatch contexts and existing factory cleanup between scenarios. Keep all five heavier children separate and the180s child timeout unchanged. Compare actual timings and original-body ASTs, review isolation, run changed children, zero-new lint/range-format and artifact preflight. Retain exact unchanged production/native/performance qualification with explicit source-equivalence evidence. Require fresh final-head Qodo and all four GitHub gates before normal merge; local timing is not a CI completion guarantee.

Latest-dev PERF-07 integration: preserve the already-approved ADR-126 D2 memo implementation from dev ecc0a531c8 unchanged, without adding a boundary or exception. Rebase after the bounded local grouping measurement. Read TASK-33266/ADR-126 and the eight upstream changed files; run the full seven-child Close suite, projection and compact approval mounted groups, upstream data-dir memo contracts and exact three Perf Guard groups on the combined tree. Requalify existing native approval/Close journeys and actual loaded source origins serially after tests; retain old-base receipts and explicit new-base identities. Finish fresh lint/artifact checks, task/QA notes, safe push and final-head Qodo/all four CI gates.

Latest-dev trace-work integration (PR2959/TASK33801):
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: Qualify the inherited known-work idle-check optimization unchanged; no new production change, boundary or budget exception.
1. Read the upstream task/runtime/maintenance and real-write regressions; prove all27 PR patches unchanged and the two changed worker owners retain exactly the upstream method/class semantics.
2. Run affected full trace migration/parking, runtime shutdown and chat-create integration contracts, followed serially by all three exact Perf Guard groups in private profiles. Preserve timers, real seams, canaries, caps and all original receipts.
3. Record current source hashes and fresh lint/preflight. Existing native receipts retain original runtime bytes and are historical; no new native UI claim for this upstream worker optimization. Safe-push the qualified combined head and require fresh final-head Qodo and all four gates before normal merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Original guard implementation was test/CI-only. PR2953 follow-up also defers existing Resend execution imports as documented below.

AC#1 (ratchet, per the controller's AC adjustment): Tests/Performance/test_console_keystroke_work_census.py gains _count_storage_units (wraps config_participants.operation [outermost per thread], storage_admission._acquire_storage [the one global every acquire_storage binding reaches], HelperLease.start, and os.open via a sys.addaudithook 'open' event with mode=None -- NOT by replacing os.open: raw_participants requires os.open in os.supports_dir_fd, and a wrapper turns every config read into RecoveryRequired('raw_source_selection_changed')). New test test_console_storage_units_stay_within_their_ratchets censuses five phases: typing burst (24 keys), typing pause (the 0.2 s trailing draft-spend refresh, fired once), credential-poll tick and legacy trace-maintenance tick (each driven directly, per tick), and a warm Console visit (Library -> Console on the reusable route) plus a get_user_data_dir canary (anti-vacuity). Wall-clock loops are held still only in this mode (credential poll stopped, trace maintenance captured not scheduled, draft-spend refresh delayed, scheduler poll stretched to 1 h via the scratch config); the four existing census tests run exactly as before. Pinned ceilings (maxima over 19 runs on macOS): typing 27 config / 54 storage / 0 helpers / 19,224 opens; pause 22/53/3/18,359; poll per tick 1/2/0/790.6; trace per tick 0/2/1/715.4; visit 37/107/9/35,351. Config admissions reproduce exactly; storage/helpers jitter downward (executor-thread connection reuse), os_opens upward in ~37-open steps (source not isolated) so os_opens gets OS_OPENS_JITTER_SLACK=1.05. Paydown owners named in the pin: TASK-33265 (PERF-06), TASK-33267 (PERF-08), TASK-33268 (PERF-09), TASK-33269 (PERF-10).

AC#2: test_screen_preimport_payload_budget.py added to perf-guard.yml's boot-ratchet step. Re-pinned at 9cd9aad65f's measurement (modules 554 exactly; LOC 410,347 + 15,000 and library 125,111 + 10,000 ADR-097 standard slack, because LOC moves with every edited line), TASK-33276 (PERF-17) named as paydown owner; ADR-097 exception-ledger row added (owner sign-off to be recorded on the PR); snapshot refreshed via scripts/update_boot_budget_snapshots.py --only preimport.

AC#3: run_console_mount_profile.py hooks ChatScreen.on_screen_resume (warm visits never mount) and times the outgoing screen's suspend as well as unmount; new default --phase warm (single warm_resume variant); production/controls kept but documented as compose-time A/Bs that no longer differ on the reusable route. Smoke test test_profiler_measures_a_warm_visit_on_the_reusable_console_route runs one real iteration in a subprocess.

AC#4: the masking was build_test_app_config/load_settings under the per-test env redirect (config participant bound to the collection-time profile). @pytest.mark.bootstrap_profile (conftest's existing per-node opt-in) on the 3 startup guards and the footer guard; the footer guard's 3 s boot wait was too short once unmasked and is now a 60 s condition wait. The same mark unmasks 6 three-turn prepare_workspace_runtime guards (pass) and the scripted mounted sample (now fails for its real reason: ScriptedGateway lacks cached_context_window). The 4 RAG citation-benchmark guards still raise RecoveryRequired with the mark (they re-select config themselves) -- left for a follow-up.

AC#5: test_snapshots_are_real_not_hollow expects the 5 current boot CSS sources.

Also: lessons-testing-evidence.md entry (os.open wrapper trap; hold wall-clock loops before trusting a count).

Files: .github/workflows/perf-guard.yml, Tests/Performance/{test_console_keystroke_work_census,run_console_mount_profile,test_console_mount_profile,test_screen_preimport_payload_budget,test_app_startup_performance,test_footer_token_timer_retired,test_console_three_turn_profile,test_boot_budget_ratchet_messages}.py, Tests/Performance/boot_budget_snapshots/preimport_payload.json, backlog/decisions/097-boot-budget-ratchets.md, backlog/docs/lessons-testing-evidence.md.

PR2953 CI follow-up (2026-10-02): settled-idle setup now completes and asserts the real fresh trace migration before its eight measured batches. Diagnostic RED counted3+7*2 admissions (17/8); plain GREEN and four actual cold-completion/read-only/parking/wake contracts pass without errors/skips. Runtime, real seams, anti-vacuity canaries and ceilings unchanged. Ruff adds zero diagnostics against dev in22 modified Python files; census retains inherited import/format diagnostics. Read-only independent review clear; incident added to lessons-testing-evidence.md. Existing ADR-097/125/126 apply; no new ADR. QA: settled_idle_census_followup in Docs/superpowers/qa/2026-10-01-console-tool-ux-config-integration.json. PR2953 still requires fresh final-head CI/Qodo before normal merge.

The private settled-typing fixture holds its media startup timer and awaits the same real cleanup before measuring keys. Two frozen call-through diagnostics observed the media helper finishing 116ms/508ms before the first key; neither captured the original aggregate helper1 failure, so this is setup-race prevention, not exact failure attribution or production cost reduction. The exact ordinary guard now passes with all admission/helper/open seams, eight idle batches, canaries and ceilings unchanged. Independent source review clear; current Ruff delta and artifact preflight pass. Existing ADR-097/125/126 apply; final PR2953 CI/review/merge checkpoint remains.

Reopened for exact final-head UI latency failure: PR run37060344703 and pure latest-dev run37058139830 both measure1034>1033. Three eager Resend imports load execution from first-paint consumers. Existing budget and snapshot will remain unchanged.

Latest-dev Resend correction: the three pure eligibility helpers now live in the already-resident message-action owner, with compatibility exports preserved; execution imports only on explicit Resend. Both new startup regressions fail before and pass after; real warm census1033/1033, unchanged limit/snapshot. All64 Resend core cases,11 actual UI/click/key/timer cases and22 boot neighbors pass. Other action logic is AST-identical to dev. Broader action-file run has120 passes/10 unchanged Canvas or legacy-label failures, exactly reproduced with the dev action owner in an otherwise combined-tree private process (not a pure full-dev checkout); initial pre-bootstrap comparison error retained. This follow-up introduces production import deferral, unlike the original test-only guard work. Artifact preflight and zero-new Ruff33 pass. Native approval25 qualifies the preceding bb865 combined candidate; newly advanced dev185c845 requires rebase/current-source verification before Done or merge.

Latest readiness-base qualification (dev185c845, code anchore4577e27): all230 scoped cases pass with zero failures/errors/skips, including193 behavior/grouping/cache cases and37 exact three-group Perf Guard cases. Warm census1033/1033; both tested/untested ordinary storage variants retain real seams, canaries, ticks and ceilings. Four projection journeys and two orphan-decision kinds retain fresh apps/controllers/workers in two private children; full body AST preserved except two explicit loop-local lambda defaults. One paired timing91.55->64.84s, final bound Close rerun23.55s; unchanged180s/20m deadlines, no CI guarantee. Inherited isolated normalization counter red3669 vs9 was exactly3660 cold static support-set calls; one real no-evidence setup build preserves positive/exact comparison and measures first shared lookup; five existing related contracts pass. Native approval26(9 journeys/111pins) and Close28(6 real journeys+six-kind geometry/183pins), current loaded origins, exit0/cleanup/no-egress/real-profile invariants pass. Initial external Close path rejection retained. AX and wider native requests/races remain unqualified; earlier10 broad action failures retain explicit baseline limits. Current preflight(census124), zero-new Ruff34, edited-range format and independent review pass. Existing ADR-097/094/126 apply; no new ADR. QA readiness_base_final_followup in combined integration JSON. Final published-head CI/Qodo/normal merge remains PR2953.

Final rebase onto dev ef8fd5d38a512be299af17b1e0d5b367a352a5d6 incorporates overlapping PR2962. Resolved only equivalent Resend import/monkeypatch placement, retaining all225 exact qualified production/test/helper/artifact pins and stronger execution-free row projection. All upstream lesson/task metadata retained. The230-case/native/preflight receipts remain original; explicit source-equivalence proof and limits recorded in QA overlapping_resend_rebase. Fresh final-published-head CI/Qodo still required before normal merge.

Reopened after published45189cd3 UI Fast Lane reached97% then hit20m. No assertions failed; all Close cases had passed. Derived artifacts failed because the required UI dependency was cancelled. Continue bounded startup reduction only in selected Close fixtures; no global CI changes.

Remaining UI timeout grouping locally qualified: nine unchanged scenario bodies/63 assertions execute through two ordinary private-profile wrappers,28.42+38.44=66.86s versus95.55s baseline. All five heavy bodies unchanged; scoped patches exit before per-scenario existing factory drains/unfreeze/GC. Independent review clear, zero-new Ruff34 and edited-range format pass. No production or CI/deadline changes; local single-pair saving is not a CI guarantee. Keep In Progress for dev ecc0a531c8 combined-tree qualification and fresh final-head gates.

Final qualification on dev ecc0a531 at code anchor 17235d9d: all 25 patches rebase unchanged and all eight upstream files match dev. Fresh 65 scoped cases pass with zero failures/errors/skips, including all mounted Close/projection/compact-approval groups, upstream memo/readiness contracts and all three exact Perf Guard groups. Native approval27 (nine journeys) and Close29 (six real closes plus geometry) pass; independently verified source pins/origins, normal process/socket cleanup, no egress and real-profile invariance. A post-completion shell capacity error is preserved separately; space recovered without pruning or rerunning. Preflight, zero-new Ruff across 34 files, range format and independent review pass; ceilings, snapshots, counters, canaries, ticks and deadlines remain unchanged. Original receipts retain their identities and explicit limits. Existing ADR-094/097/126 apply; no new ADR. QA path_memo_base_final_followup records evidence. PR2953 still requires fresh final-head CI/Qodo and normal protected merge. No full suite.

Latest dev e6ab66b0 (PR2959/TASK33801 known-work idle-check optimization) integrates cleanly: all27 PR patches unchanged, changed worker/class AST exactly upstream and remaining modules/controller unchanged.147 fresh scoped cases pass with zero failures/errors/skips:110 real trace/write/lifetime/chat-create contracts plus all37 exact Perf Guard cases. Of230 prior pins,228 unchanged; two upstream trace-owner changes and two additional upstream regression-file pins identified (232 current). Zero-new Ruff34 and fresh preflight pass; original counters, ticks, canaries, caps, snapshots and CI settings remain. No new production patch or full suite. Prior native27/29 and46/65/230 receipts keep original identities; current worker owners have real SQLite and mounted performance evidence, with no new native replay claim. Existing ADR-097/126 apply. QA trace_work_base_final_followup records qualification/limits. Final published-head CI/Qodo/normal merge remains PR2953.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
