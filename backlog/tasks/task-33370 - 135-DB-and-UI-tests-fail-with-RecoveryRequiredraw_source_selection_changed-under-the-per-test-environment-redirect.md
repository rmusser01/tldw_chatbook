---
id: TASK-33370
title: 135 DB and UI tests fail with RecoveryRequired('raw_source_selection_changed')
  under the per-test environment redirect
status: In Progress
assignee: []
created_date: 2026-09-28 20:12
updated_date: 2026-10-02 06:54
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Scope addition (2026-09-28, found while verifying PERF-06): Tests/test_config_*.py fails 186 tests identically at base c174e30f6b and on the PERF-06 branch. The TASK-32804.1 warm-read tests skip with 'raw_source_selection_changed' for the same reason. Under @pytest.mark.bootstrap_profile the same config reads work, so the marker, or the shared conftest seam it uses, is the likely fix for this set too.
2026-10-01 scoped follow-up: shared conftest families retain the collection-time private profile; independent config tests select a real fresh module through the existing test helper and rebind only explicit consumers. No participant registry, production guard or deadline is reset. Fixed eager-import creation/absence setup, real default-profile selection, lock protocols, guarded-function observations, captured-source retarget assertions and the actual public warm-cache race hook. Legacy cache/publication assertions now match the unchanged recovery rollback/structured result contract; unsafe forced reads remain refused. Prior baseline 560 cases had 292 failures/15 errors/2 skips. Current finite selection /private/tmp/backup-followup-check-pxzt67k1 passed all565 with0 failures/errors/skips; stronger /private/tmp/backup-followup-check-wzpm3jic passed43 and /private/tmp/backup-followup-check-igf4nkk1 passed136. Final four-case /private/tmp/backup-followup-check-rlwf4wjg passed all4, including populated-cache decryption recovery, actual warm-path race and original benchmark CLI privacy/budget assertions. Independent final immutable review /private/tmp/backup-followup-final-independent-review-zxtrobnq/report.md PASS, no actionable P1/P2. Scoped static /private/tmp/backup-followup-final-static-oyhol6q7/corrected-bandit-comparison.json has0 new non-assert Bandit findings,27 identical existing non-assert findings;15 new B101 observations are pytest assertions. Ruff has0 new findings; compile/diff passed. Updated lessons-testing-evidence.md through official Backlog document tools. Evidence above precedes the pending rebase onto current dev922440b93e; assess upstream schema/Console changes and rerun the finite affected selection before publication. Status remains In Progress pending integration.
Post-rebase integration: normal commit 5ffd78a967125c024a2741829aa548b28342f5c4 rebased cleanly onto dev922440b93e83b4dd7086086de21b1276c477fc8e as70152bb3dc534cb278b8f84b755248ed9dfbbccc. All23 final reviewed files remained byte-identical (/private/tmp/backup-followup-rebase-2ddydqpv/summary.json). The finite current-dev phase /private/tmp/backup-followup-dev-integration-65ec4aj9/summary.json verified7611 tracked Python/SQL source bytes stable and707 actual cases PASS,0 failures/errors/skips, including the original config/DB/environment selection, complete benchmark module with original CLIprivacy/budget assertion,46Personas, admission runtime, new v74 migration and actual ChaChaNotes schema-policy check. Network guard and Null keyring were enabled before app imports. No full suite or repeated native matrix; original native backup evidence retains its original identities. Status remains In Progress pending PR integration.
Independent latest-dev rebase assessment /private/tmp/backup-followup-rebase-independent-review-er5zsq_e/report.md passed with no actionable findings. All 23 reviewed source pins match exact 70152 HEAD; the reviewer independently counted current JUnit at 707 PASS, including 72 benchmark, 46 Personas and 18 v74 cases. No application/native tests were run by that reviewer. Only Task tracking text changes after the qualified application/test bytes.
Draft PR #2955: https://github.com/rmusser01/tldw_chatbook/pull/2955, against dev 922440b93e83b4dd7086086de21b1276c477fc8e. Published evidence head 7fef82e2f0fbc165487673bbda19dfdfc7ef8927 changes only Task records after tested runtime 70152bb3dc534cb278b8f84b755248ed9dfbbccc; all 7611 tested Python/SQL bytes remain identical. Pending current-head CI and PR integration. The admission-performance proposal remains unapproved and has no implementation in this PR.
2026-10-01 latest-dev rebase check at495e0abbb: 720 collected,717 passed,3 failures,0errors/skips; /private/tmp/backup-followup-check-lkk8d2dg/summary.json. Read-only diagnosis /private/tmp/backup2955-rebase-diagnosis-ywn7yo_f/report.md and bounded one-execution controls /private/tmp/backup2955-rebase-controls-5j6knddj/report.md show two writers preserve original conflict/lost-update assertions with private file output instead of undrained PIPE, retaining20s readiness/30s completion/0.75s write barrier. The control's omitted tomllib import was corrected by executing the retained assertion tail, not rerunning child mutations. Real cache/native tracing proves PERF06 warm rederivation retains literal0.17 from validated bootstrap cache while forced reads refuse the unsafe outside0.99 symlink twice and preserve outside bytes/mode. Adapt test output transport and warm value assertion only; retain rollback identity/source, refusal/security/serialization assertions. Add explicit Null keyring/network guard before app imports to these fresh writer children; do not infer product security regression or change production.
Rebased followup source495e0abb on latest devab4df999: the three originally failing cases now pass as the actual pytest cases,3PASS/0fail-error-skip and0undrained network attempts, receipt /private/tmp/backup-followup-check-i2484u9p/summary.json. Child output transport uses private files, with original20s readiness/30s completion/.75s coordination and exact conflict/lost-update assertions retained; each child installs Null keyring and the network guard before app imports and asserts0network. The warm cache check now asserts literal validated0.17 while forced refusal/cache/source identity and unsafe outside0644 mode checks remain. Scoped format/compile2/diff pass. Paired static /private/tmp/backup2955-rebase-static-5q8havbp/paired-findings.json has0newRuff and0newnonassertBandit by rule/file/severity/confidence; five existing Bandit findings and one existing Ruff alias finding remain. Initial static summary retained with over-count from changed context/line prefixes; corrected pairing does not label all findings clean. Independent review and normal followup-branch publication pending; no production change.
Independent latest-dev adaptation review /private/tmp/backup2955-rebase-review-ap8fqvd1/report.md passed both spec and code quality, no actionable findings. Reviewer verified immutable patch6813ef2b/source pins and parsed original720-case JUnit717PASS/3FAIL plus corrected actual3-case JUnit3PASS, with distinct identities and unchanged writer deadlines. New child network assertions reached;0newpaired static findings, existing debt retained. Normal followup-branch commit/publication and fresh required exact-head CI remain pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Config, DB and UI fixtures select and retain real private source lifetimes; production admission and deliberate-retarget refusals remain unchanged. Earlier dev922 qualification passed707 cases. Rebased onto devab4 (PERF06): the finite720-case phase passed717 and exposed three test-seam failures; those three actual cases now pass after private output transport and a literal validated warm-cache value assertion, with original deadlines and safety assertions retained. Independent scoped review passed both spec and quality; paired static checks have no new findings and retain existing debt. The two phases are not one fresh all720-pass run. Pending normal publication, current-head required CI and PR integration.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
