---
id: TASK-33373
title: >-
  RAG citation-provenance benchmark guards fail: 4 hit RecoveryRequired and the
  CLI writes admission gate files under HOME
status: In Progress
assignee: []
created_date: '2026-09-28 20:12'
updated_date: '2026-10-02 02:40'
labels:
  - testing
  - rag
  - backup-recovery
dependencies:
  - TASK-33370
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Performance/test_rag_citation_provenance_benchmark.py has 5 failing guards on dev. Four hit RecoveryRequired('raw_source_selection_changed'); they select their own config, so the bootstrap_profile marker PERF-01 used does not help. test_cli_never_reads_or_writes_host_config_data_or_secrets fails because the benchmark CLI writes recovery-bootstrap admission gate files under HOME. That may be a real host-isolation defect in the CLI, not just a test problem. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The HOME write is explained, and either the CLI stops writing under the host HOME or the guard's contract is corrected with a documented reason
- [x] #2 The 4 RecoveryRequired guards run their real assertions and pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce actual benchmark CLI host-HOME changes and the four configuration refusals with disposable sentinel profiles.
2. Trace every app import/config/control-root access before and after the existing benchmark isolation context.
3. Make the smallest isolation correction without weakening HOME/secret/network assertions; run all benchmark guard cases.
4. Run scoped lint/format/Bandit and independent review; record evidence.
ADR required: no. Correct existing benchmark isolation contract under ADR-126.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-10-01 scoped fix: migration ran after the inner benchmark isolation context had retired, so its real DB owner enrolled the caller HOME (metadata-only diagnostic observed112 host admissions). One private profile now owns validation imports and every measured group; HOME and USERPROFILE share it, and both restore afterward. Direct callers must supply an already selected private profile before work. No application guard, migration budget, credential/backend permission or timeout changed. Root conftest retention also allows the four real Console first-token guards to reach/pass their original assertions. Original host-state regression and new direct-runner refusal passed in final four-case /private/tmp/backup-followup-check-rlwf4wjg (4/4,0 fail/error/skip); the CLI test preserves complete host bytes, private output/redaction checks, original overall budget and exit-code checks. Earlier phase /private/tmp/backup-followup-check-pmf0m1gw retained a migration budget miss49.715 vs100; do not erase it or claim an established cause. Diagnostic after fix observed0 host admissions/controls/network. Independent immutable code review /private/tmp/backup-followup-final-independent-review-zxtrobnq/report.md passed; remaining integration verification will follow the dev rebase. TASK-33267 remains separate, awaiting ADR amendment approval.
Post-rebase integration: normal commit 5ffd78a967125c024a2741829aa548b28342f5c4 rebased cleanly onto dev922440b93e83b4dd7086086de21b1276c477fc8e as70152bb3dc534cb278b8f84b755248ed9dfbbccc. All23 final reviewed files remained byte-identical (/private/tmp/backup-followup-rebase-2ddydqpv/summary.json). The finite current-dev phase /private/tmp/backup-followup-dev-integration-65ec4aj9/summary.json verified7611 tracked Python/SQL source bytes stable and707 actual cases PASS,0 failures/errors/skips, including the original config/DB/environment selection, complete benchmark module with original CLIprivacy/budget assertion,46Personas, admission runtime, new v74 migration and actual ChaChaNotes schema-policy check. Network guard and Null keyring were enabled before app imports. No full suite or repeated native matrix; original native backup evidence retains its original identities. Status remains In Progress pending PR integration.
Independent latest-dev rebase assessment /private/tmp/backup-followup-rebase-independent-review-er5zsq_e/report.md passed with no actionable findings. All 23 reviewed source pins match exact 70152 HEAD; the reviewer independently counted current JUnit at 707 PASS, including 72 benchmark, 46 Personas and 18 v74 cases. No application/native tests were run by that reviewer. Only Task tracking text changes after the qualified application/test bytes.
Draft PR #2955: https://github.com/rmusser01/tldw_chatbook/pull/2955, against dev 922440b93e83b4dd7086086de21b1276c477fc8e. Published evidence head 7fef82e2f0fbc165487673bbda19dfdfc7ef8927 changes only Task records after tested runtime 70152bb3dc534cb278b8f84b755248ed9dfbbccc; all 7611 tested Python/SQL bytes remain identical. Pending current-head CI and PR integration. The admission-performance proposal remains unapproved and has no implementation in this PR.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validation imports and every measured benchmark group share one private HOME/USERPROFILE/config/data lifetime. Direct callers must select it before work. Latest-dev finite integration passed all 707 cases, including the original CLI host-snapshot, redaction and unchanged performance-budget assertions. Earlier budget misses remain documented without an invented timing cause. Independent code and rebase reviews passed. Pending PR integration; status remains In Progress.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
