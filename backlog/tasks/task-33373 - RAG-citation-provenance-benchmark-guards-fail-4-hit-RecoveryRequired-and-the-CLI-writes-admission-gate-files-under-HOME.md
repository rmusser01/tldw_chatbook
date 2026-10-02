---
id: TASK-33373
title: 'RAG citation-provenance benchmark guards fail: 4 hit RecoveryRequired and
  the CLI writes admission gate files under HOME'
status: In Progress
assignee: []
created_date: 2026-09-28 20:12
updated_date: 2026-10-02 18:43
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-01 scoped fix: migration ran after the inner benchmark isolation context had retired, so its real DB owner enrolled the caller HOME (metadata-only diagnostic observed112 host admissions). One private profile now owns validation imports and every measured group; HOME and USERPROFILE share it, and both restore afterward. Direct callers must supply an already selected private profile before work. No application guard, migration budget, credential/backend permission or timeout changed. Root conftest retention also allows the four real Console first-token guards to reach/pass their original assertions. Original host-state regression and new direct-runner refusal passed in final four-case /private/tmp/backup-followup-check-rlwf4wjg (4/4,0 fail/error/skip); the CLI test preserves complete host bytes, private output/redaction checks, original overall budget and exit-code checks. Earlier phase /private/tmp/backup-followup-check-pmf0m1gw retained a migration budget miss49.715 vs100; do not erase it or claim an established cause. Diagnostic after fix observed0 host admissions/controls/network. Independent immutable code review /private/tmp/backup-followup-final-independent-review-zxtrobnq/report.md passed; remaining integration verification will follow the dev rebase. TASK-33267 remains separate, awaiting ADR amendment approval.
Post-rebase integration: normal commit 5ffd78a967125c024a2741829aa548b28342f5c4 rebased cleanly onto dev922440b93e83b4dd7086086de21b1276c477fc8e as70152bb3dc534cb278b8f84b755248ed9dfbbccc. All23 final reviewed files remained byte-identical (/private/tmp/backup-followup-rebase-2ddydqpv/summary.json). The finite current-dev phase /private/tmp/backup-followup-dev-integration-65ec4aj9/summary.json verified7611 tracked Python/SQL source bytes stable and707 actual cases PASS,0 failures/errors/skips, including the original config/DB/environment selection, complete benchmark module with original CLIprivacy/budget assertion,46Personas, admission runtime, new v74 migration and actual ChaChaNotes schema-policy check. Network guard and Null keyring were enabled before app imports. No full suite or repeated native matrix; original native backup evidence retains its original identities. Status remains In Progress pending PR integration.
Independent latest-dev rebase assessment /private/tmp/backup-followup-rebase-independent-review-er5zsq_e/report.md passed with no actionable findings. All 23 reviewed source pins match exact 70152 HEAD; the reviewer independently counted current JUnit at 707 PASS, including 72 benchmark, 46 Personas and 18 v74 cases. No application/native tests were run by that reviewer. Only Task tracking text changes after the qualified application/test bytes.
Draft PR #2955: https://github.com/rmusser01/tldw_chatbook/pull/2955, against dev 922440b93e83b4dd7086086de21b1276c477fc8e. Published evidence head 7fef82e2f0fbc165487673bbda19dfdfc7ef8927 changes only Task records after tested runtime 70152bb3dc534cb278b8f84b755248ed9dfbbccc; all 7611 tested Python/SQL bytes remain identical. Pending current-head CI and PR integration. The admission-performance proposal remains unapproved and has no implementation in this PR.
Latest logging-dev rebase source d18e990586a8b193a3fbf12248bf1b266e6eca85 on dev92a95170: the actual finite phase /private/tmp/backup-followup-check-01o7b3nc/summary.json collected106, passed94, failed12,0errors/skips/undrainednetwork. All72 original benchmark cases PASS including original CLI host-state/redaction/budget assertions; new logging17, adapted config3 and installed ChaChaNotes schema1 PASS. All12 distinct failures are worker-event tests refused at real app construction before assertions with raw_source_selection_changed; TASK-33370 owns their scoped source-lifetime diagnosis/correction. Prior720-case717/3 and corrected3-case phases retain original identities, not one all-green phase. No production/admission/deadline changes or original native reruns. The admission performance design is now requester-approved and proceeds separately.
Current-dev rebase qualification at06165659/dev eba4305: original benchmark module72/72PASS within finite777 phase(776PASS0FAIL0ERROR1inapplicable upstream oracle SKIP), zero undrained parent network and7622 tracked Python/SQL pins unchanged. Receipts /private/tmp/backup2955-final-rebase-o7vk5h6b/summary.json and /private/tmp/backup-followup-check-d1aocb76; no numeric/input/corpus/privacy/selection-lifetime/deadline changes or full sweep. Exact upstream source applicability /private/tmp/backup2955-upstream-delete-review.md: ChaChaNotesv74 and exercised constructor/transaction/migration methods unchanged; real ChatPersistenceService constructor gains only thread-local release holder, new Delete methods are not called by benchmark or reconstructed probe. Actual added ready census/modal CSS source-wiring nodes pass. Earlier92-source72PASS and113-source72PASS are separate, retained; current phase supplies new source identity. Shared config-consumer cleanup independently reviewed underTASK33370. PR2955 publication/current requiredCI/integration pending; performance amendment remains separately trackedTASK33560.
2026-10-02 current-dev rebase: f33799c1417a463a024e30c181f5605f736df27c is based on dev6958e8dfa99a66680b1aec09fed65df0f1a91955. All11 prior patches are unchanged (=). Whole-branch independent review of d75/dev eba passed SPEC and QUALITY, /private/tmp/backup2955-final-independent-review.md; its exact777-case evidence remains tied to061/dev eba. Fresh affected rebase verification /private/tmp/backup2955-folder-rebase-hvffe1u2 passed94/94,0failures/errors/skips/undrained parent network; all7627 Python/SQL pins remained stable. Originalbenchmark72,workers13,cleanup3,ready/CSS2 and private-profile reporting4 pass. Parent zero is not arbitrary logging-child isolation proof; no new full suite/native/performance qualification. Tasks remain In Progress pending protected PR2955 integration. Private benchmark profile covers every import and measured group; original numeric/deadline/privacy assertions stay intact. New upstream Console/screen cleanup is being assessed for applicability; no speculative production fix.
2026-10-02 PR2955 scoped Qodo followup at base176126644f3cbf81ecdc4a2c069a40cf8d6cc502/dev6958: document run_benchmark Args/Returns/Raises while retaining its lifetime-bound private-profile prerequisite and every output/budget/deadline. Related test-source cleanup/isolation correction is tracked with TASK-33370; verify affected benchmark/worker routes under the existing fresh private Null/network guarded runner and preserve exact source receipts. No new paired probe or performance implementation is included.
PR2955 scoped Qodo followup at BASE176126644f3cbf81ecdc4a2c069a40cf8d6cc502: reproduced both fixture defects before the fix (4 RED cases: two actual post-teardown ProviderSettingsError escapes, and two canonical bootstrap-file byte leaks). Restore now maps only exact selected-config-defined classes as well as functions/modules in newly imported application namespaces; nested selections, imported-library classes, old consumers and real retired-source refusal remain guarded. The console save self-test and TASK21124 fast-path module select an owned independent actual source before cache warmup, patches or writes; canonical collection bytes are explicitly checked unchanged. Added concise annotations/docs for source selection and teardown, and Args/Returns/Raises for benchmark run_benchmark without executable benchmark changes.

Evidence retained: RED /private/tmp/backup2955-qodo-red-1hi88cxg/summary.json (runner /private/tmp/backup-followup-check-ar2o28ry); focused GREEN15 /private/tmp/backup2955-qodo-green-regressions-jdmdix6d/summary.json (runner /private/tmp/backup-followup-check-uryoocjh). Final affected GREEN432 /private/tmp/backup2955-qodo-green-affected-fi3c9tru/summary.json (runner /private/tmp/backup-followup-check-0iipx0vg): all top-level config modules, helper5, benchmark72 and worker13, zero failures/errors/skips/undrained parent network attempts. All 7,627 tracked Python/SQL hashes stable; final changed sources match tested pins and retained test-source copies. The focused15 retains its own earlier doc/format identity; final432 is the final source identity. Fresh private HOME/USERPROFILE/XDG/config, Null keyring and network guard preceded application imports; original test/CLI budgets preserved.

Final paired static /private/tmp/backup2955-qodo-final-static-3n3y_mz7/summary.json: six files compile, diff-check passes, Ruff25 existing findings unchanged, Bandit6 existing non-test-assert findings unchanged (only pytest B101 excluded; benchmark B101 retained), identical preexisting format debt, zero new findings. AST checks preserve executable benchmark code, teardown body, root isolation policy and every original assertion. This is bounded affected verification, not a full-suite/native/admission-probe rerun or all-child network claim. Production guards/resources, registries and all budgets/deadlines unchanged. Normal scoped commit and independent review follow; tasks remain In Progress.
2026-10-02 final followup qualification: shared helper restoration now includes exact helper-owned config exception classes, with owned mutating Console/fast-path profiles. The original four semantic regressions failed before correction; focused15 and affected432 passed at exact7d19f2af with 0fail/error/skip. Independent SPEC and QUALITY reviews passed all seven Qodo items; paired static checks add zero findings and retain documented prior debt. Original/ corrected reports and transcript-derived operation evidence retain their explicit limits.

Normal rebases preserved the six reviewed Python files byte/mode-identically: 7d19→c9ec20aa on dev30ca (30PASS), then c9→d61613af16d1aef9d50cfc5143df2c41614187a3 on deve92b01515f9547aab2cfb14cd4d93a7a07419775. Each13-patch range-diff has12equal patches and one document-context-only change; official document editor preserved both lessons sections and LF. Fresh42 actual cases passed at exactd616, 0fail/error/skip, with7,736 tracked Python/SQL byte/stat pins stable and zero undrained parent network attempts. This includes schema75 upgrade/rollback, backup validation, mounted Console continuations and benchmark/dependency controls. Independent /private/tmp/backup2955-expanded-final-independent-review.json (SHA8437e129dcee05d264a37203456799da32d5169afc308b4daa649aad4b92cf33) joined all7,736 pins to immutableGit and all20selectors to42JUnit cases; no actionable findings. Source-stat/network history remain receipt limits; no numeric, idle or native qualification is inferred.

Evidence: /private/tmp/backup2955-qodo-independent-spec-review.md; /private/tmp/backup2955-qodo-independent-quality-review.md; /private/tmp/backup2955-expanded-rebase-proof-ddziepmo/summary.json; /private/tmp/backup2955-expanded-rebase-check-h0terovr/summary.json. PR https://github.com/rmusser01/tldw_chatbook/pull/2955 remains pending publication of this reviewed source, exact-current-head required CI, review resolution and protected integration. Historical phases retain their own identities and unsuccessful results.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The citation benchmark uses one private HOME/USERPROFILE/config/data lifetime through validation and all original groups, including repository storage and migrations. Direct callers require a private profile; original budgets, inputs and host/privacy assertions remain intact.

The original benchmark module passed72/72 on current source06165659cdf72f947554cf23ec1cd44f9ba430e2/dev eba4305 inside the777-case finite phase(776 passed,0 failed/errors,1 inapplicable upstream oracle skip). All7622 Python/SQL source pins remained stable, with private environment/Null keyring/network guard established before app imports. Evidence /private/tmp/backup2955-final-rebase-o7vk5h6b and /private/tmp/backup-followup-check-d1aocb76. Static upstream assessment confirms ChaChaNotes remainsv74; exercised transaction/migration methods are unchanged, and only the actual readiness/modal CSS checks were added.

PR #2955 current-head required CI and protected integration remain pending. These are benchmark-followup checks, not new admission-performance qualification; Task33560 retains the approved performance work.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
