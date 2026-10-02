---
id: TASK-33260
title: >-
  PERF-01: Perf guards that count admissions, helper spawns and pre-import
  payload
status: Done
assignee:
  - '@codex'
created_date: '2026-09-28 18:01'
updated_date: '2026-10-02 18:57'
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test/CI-only change; no production code touched.

AC#1 (ratchet, per the controller's AC adjustment): Tests/Performance/test_console_keystroke_work_census.py gains _count_storage_units (wraps config_participants.operation [outermost per thread], storage_admission._acquire_storage [the one global every acquire_storage binding reaches], HelperLease.start, and os.open via a sys.addaudithook 'open' event with mode=None -- NOT by replacing os.open: raw_participants requires os.open in os.supports_dir_fd, and a wrapper turns every config read into RecoveryRequired('raw_source_selection_changed')). New test test_console_storage_units_stay_within_their_ratchets censuses five phases: typing burst (24 keys), typing pause (the 0.2 s trailing draft-spend refresh, fired once), credential-poll tick and legacy trace-maintenance tick (each driven directly, per tick), and a warm Console visit (Library -> Console on the reusable route) plus a get_user_data_dir canary (anti-vacuity). Wall-clock loops are held still only in this mode (credential poll stopped, trace maintenance captured not scheduled, draft-spend refresh delayed, scheduler poll stretched to 1 h via the scratch config); the four existing census tests run exactly as before. Pinned ceilings (maxima over 19 runs on macOS): typing 27 config / 54 storage / 0 helpers / 19,224 opens; pause 22/53/3/18,359; poll per tick 1/2/0/790.6; trace per tick 0/2/1/715.4; visit 37/107/9/35,351. Config admissions reproduce exactly; storage/helpers jitter downward (executor-thread connection reuse), os_opens upward in ~37-open steps (source not isolated) so os_opens gets OS_OPENS_JITTER_SLACK=1.05. Paydown owners named in the pin: TASK-33265 (PERF-06), TASK-33267 (PERF-08), TASK-33268 (PERF-09), TASK-33269 (PERF-10).

AC#2: test_screen_preimport_payload_budget.py added to perf-guard.yml's boot-ratchet step. Re-pinned at 9cd9aad65f's measurement (modules 554 exactly; LOC 410,347 + 15,000 and library 125,111 + 10,000 ADR-097 standard slack, because LOC moves with every edited line), TASK-33276 (PERF-17) named as paydown owner; ADR-097 exception-ledger row added (owner sign-off to be recorded on the PR); snapshot refreshed via scripts/update_boot_budget_snapshots.py --only preimport.

AC#3: run_console_mount_profile.py hooks ChatScreen.on_screen_resume (warm visits never mount) and times the outgoing screen's suspend as well as unmount; new default --phase warm (single warm_resume variant); production/controls kept but documented as compose-time A/Bs that no longer differ on the reusable route. Smoke test test_profiler_measures_a_warm_visit_on_the_reusable_console_route runs one real iteration in a subprocess.

AC#4: the masking was build_test_app_config/load_settings under the per-test env redirect (config participant bound to the collection-time profile). @pytest.mark.bootstrap_profile (conftest's existing per-node opt-in) on the 3 startup guards and the footer guard; the footer guard's 3 s boot wait was too short once unmasked and is now a 60 s condition wait. The same mark unmasks 6 three-turn prepare_workspace_runtime guards (pass) and the scripted mounted sample (now fails for its real reason: ScriptedGateway lacks cached_context_window). The 4 RAG citation-benchmark guards still raise RecoveryRequired with the mark (they re-select config themselves) -- left for a follow-up.

AC#5: test_snapshots_are_real_not_hollow expects the 5 current boot CSS sources.

Also: lessons-testing-evidence.md entry (os.open wrapper trap; hold wall-clock loops before trusting a count).

Files: .github/workflows/perf-guard.yml, Tests/Performance/{test_console_keystroke_work_census,run_console_mount_profile,test_console_mount_profile,test_screen_preimport_payload_budget,test_app_startup_performance,test_footer_token_timer_retired,test_console_three_turn_profile,test_boot_budget_ratchet_messages}.py, Tests/Performance/boot_budget_snapshots/preimport_payload.json, backlog/decisions/097-boot-budget-ratchets.md, backlog/docs/lessons-testing-evidence.md.

PR2953 CI follow-up (2026-10-02): settled-idle setup now completes and asserts the real fresh trace migration before its eight measured batches. Diagnostic RED counted3+7*2 admissions (17/8); plain GREEN and four actual cold-completion/read-only/parking/wake contracts pass without errors/skips. Runtime, real seams, anti-vacuity canaries and ceilings unchanged. Ruff adds zero diagnostics against dev in22 modified Python files; census retains inherited import/format diagnostics. Read-only independent review clear; incident added to lessons-testing-evidence.md. Existing ADR-097/125/126 apply; no new ADR. QA: settled_idle_census_followup in Docs/superpowers/qa/2026-10-01-console-tool-ux-config-integration.json. PR2953 still requires fresh final-head CI/Qodo before normal merge.

The private settled-typing fixture holds its media startup timer and awaits the same real cleanup before measuring keys. Two frozen call-through diagnostics observed the media helper finishing 116ms/508ms before the first key; neither captured the original aggregate helper1 failure, so this is setup-race prevention, not exact failure attribution or production cost reduction. The exact ordinary guard now passes with all admission/helper/open seams, eight idle batches, canaries and ceilings unchanged. Independent source review clear; current Ruff delta and artifact preflight pass. Existing ADR-097/125/126 apply; final PR2953 CI/review/merge checkpoint remains.
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
