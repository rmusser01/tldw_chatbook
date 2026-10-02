---
id: TASK-33370
title: >-
  135 DB and UI tests fail with RecoveryRequired('raw_source_selection_changed')
  under the per-test environment redirect
status: In Progress
assignee: []
created_date: '2026-09-28 20:12'
updated_date: '2026-10-02 02:40'
labels:
  - testing
  - backup-recovery
  - tech-debt
dependencies:
  - TASK-33260
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A local run of Tests/DB, Tests/ChaChaNotesDB and the files calling search_conversations_page produced 187 distinct failure tracebacks on dev. 135 of them raise tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired('raw_source_selection_changed').

The affected files:
- mostly the Subscriptions/Watchlists DB suites (test_subscriptions_db_watchlists*.py, test_subscriptions_db*.py)
- Tests/UI/test_personas_lore.py (46 setup errors)
- several ChaChaNotes migration suites

These tests are blind: they report the recovery bootstrap, not the behaviour they pin. PERF-01 (TASK-33260) hit the same class in the perf guards, which failed when build_test_app_config -> load_settings ran under the per-test env redirect. It fixed them with the conftest's existing @pytest.mark.bootstrap_profile. PR CI lanes do not run most of these files, so nothing red surfaced. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared cause is identified and fixed at the common seam (conftest or fixture), not by marking tests one by one, unless per-test marking is shown to be the correct contract
- [x] #2 The previously RecoveryRequired-blinded tests run their real assertions; any that then fail for another reason are fixed or listed with a named owner
- [x] #3 backlog/docs/lessons-testing-evidence.md records how to recognise this failure class
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the named DB, Personas and config fixture refusals on current dev in a private environment.
2. Trace collection-time source binding and per-test redirects; fix the shared test seam while preserving production admission and host isolation.
3. Run the affected original assertions and meaningful isolation regressions; document any distinct failures with named owners.
4. Run scoped lint/format/Bandit and independent review, record evidence and prepare reviewable changes.
ADR required: no. Test fixture correction preserves ADR-126; reference backlog/decisions/126-complete-local-backup-and-recovery.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scope addition (2026-09-28, found while verifying PERF-06): Tests/test_config_*.py fails 186 tests identically at base c174e30f6b and on the PERF-06 branch. The TASK-32804.1 warm-read tests skip with 'raw_source_selection_changed' for the same reason. Under @pytest.mark.bootstrap_profile the same config reads work, so the marker, or the shared conftest seam it uses, is the likely fix for this set too.
2026-10-01 scoped follow-up: shared conftest families retain the collection-time private profile; independent config tests select a real fresh module through the existing test helper and rebind only explicit consumers. No participant registry, production guard or deadline is reset. Fixed eager-import creation/absence setup, real default-profile selection, lock protocols, guarded-function observations, captured-source retarget assertions and the actual public warm-cache race hook. Legacy cache/publication assertions now match the unchanged recovery rollback/structured result contract; unsafe forced reads remain refused. Prior baseline 560 cases had 292 failures/15 errors/2 skips. Current finite selection /private/tmp/backup-followup-check-pxzt67k1 passed all565 with0 failures/errors/skips; stronger /private/tmp/backup-followup-check-wzpm3jic passed43 and /private/tmp/backup-followup-check-igf4nkk1 passed136. Final four-case /private/tmp/backup-followup-check-rlwf4wjg passed all4, including populated-cache decryption recovery, actual warm-path race and original benchmark CLI privacy/budget assertions. Independent final immutable review /private/tmp/backup-followup-final-independent-review-zxtrobnq/report.md PASS, no actionable P1/P2. Scoped static /private/tmp/backup-followup-final-static-oyhol6q7/corrected-bandit-comparison.json has0 new non-assert Bandit findings,27 identical existing non-assert findings;15 new B101 observations are pytest assertions. Ruff has0 new findings; compile/diff passed. Updated lessons-testing-evidence.md through official Backlog document tools. Evidence above precedes the pending rebase onto current dev922440b93e; assess upstream schema/Console changes and rerun the finite affected selection before publication. Status remains In Progress pending integration.
Post-rebase integration: normal commit 5ffd78a967125c024a2741829aa548b28342f5c4 rebased cleanly onto dev922440b93e83b4dd7086086de21b1276c477fc8e as70152bb3dc534cb278b8f84b755248ed9dfbbccc. All23 final reviewed files remained byte-identical (/private/tmp/backup-followup-rebase-2ddydqpv/summary.json). The finite current-dev phase /private/tmp/backup-followup-dev-integration-65ec4aj9/summary.json verified7611 tracked Python/SQL source bytes stable and707 actual cases PASS,0 failures/errors/skips, including the original config/DB/environment selection, complete benchmark module with original CLIprivacy/budget assertion,46Personas, admission runtime, new v74 migration and actual ChaChaNotes schema-policy check. Network guard and Null keyring were enabled before app imports. No full suite or repeated native matrix; original native backup evidence retains its original identities. Status remains In Progress pending PR integration.
Independent latest-dev rebase assessment /private/tmp/backup-followup-rebase-independent-review-er5zsq_e/report.md passed with no actionable findings. All 23 reviewed source pins match exact 70152 HEAD; the reviewer independently counted current JUnit at 707 PASS, including 72 benchmark, 46 Personas and 18 v74 cases. No application/native tests were run by that reviewer. Only Task tracking text changes after the qualified application/test bytes.
Draft PR #2955: https://github.com/rmusser01/tldw_chatbook/pull/2955, against dev 922440b93e83b4dd7086086de21b1276c477fc8e. Published evidence head 7fef82e2f0fbc165487673bbda19dfdfc7ef8927 changes only Task records after tested runtime 70152bb3dc534cb278b8f84b755248ed9dfbbccc; all 7611 tested Python/SQL bytes remain identical. Pending current-head CI and PR integration. The admission-performance proposal remains unapproved and has no implementation in this PR.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Config, DB and UI fixtures now select and retain real private source lifetimes. Production admission and deliberate-retarget refusals are unchanged. Independent code and rebase reviews passed. Latest-dev finite integration at 70152bb3dc534cb278b8f84b755248ed9dfbbccc passed all 707 cases, including original assertions and v74 migration/schema-policy checks. Static analysis has no new non-assert Bandit or Ruff findings. Pending PR integration; status remains In Progress.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
