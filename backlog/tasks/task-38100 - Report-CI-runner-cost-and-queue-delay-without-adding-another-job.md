---
id: TASK-38100
title: Report CI runner cost and queue delay without adding another job
status: In Progress
created_date: 2026-10-30 06:15
assignee:
- '@codex'
labels:
- ci
- testing
priority: high
updated_date: 2026-11-01 12:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the cost of each required qualification visible so coverage growth and scheduling changes can be judged against total runner work and queue delay.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The required workflow reports completed test-job runner minutes and queue delay with explicit scope, attempts and missing-data handling.
- [x] #2 Cost reporting adds no runner job, cannot change qualification verdicts, and does not change test targets, permissions of application jobs or merge protection.
- [x] #3 A documented initial warning budget is derived from the measured audit baseline; exceeding it reports a warning without suppressing required tests.
- [x] #4 Focused behavioral tests exercise the reporting calculation and actual workflow integration, including API errors and incomplete timestamps.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md. Reason: observe CI cost with a narrow non-gating diagnostic exception. Follow Docs/superpowers/plans/2026-10-29-ci-cost-reporting.md: execute failing tests against the actual inline script, implement the one-step report and fail-closed contract exception, run focused checks and an offline audit sample, obtain independent review, and record local versus hosted evidence before delivery.
2026-11-01 backlog integration follows Docs/superpowers/plans/2026-11-01-pr3317-backlog-integration.md. Apply only the source report commit onto the current PR3316 integration, preserve all application jobs and verdicts, rebase after confirmed PR3316 merge, run focused local contracts, and obtain independent requesting-code-review before authorized merge. Live PR is non-draft; historical draft delivery wording is superseded. Hosted summary, actual API pagination and timing remain unqualified under the user waiver. ADR required: no additional ADR beyond the source ADR103 amendment.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added one bounded, checkout-free cost report after the required aggregate verdicts. Job-local actions-read access reads the current attempt and all recorded jobs, deduplicates IDs, includes failed/cancelled completed execution, excludes skipped/control jobs, and discloses invalid/missing timestamps. Reports observed execution minutes rather than billing; job wait includes dependency/concurrency waits. A 500-minute warning budget has about 26 percent headroom above the audit median of 397 minutes. Only this diagnostic step can tolerate failure; all qualification verdicts remain fail-closed.

Validation: 18 initial red tests, then 195 focused reporting/gate/queue tests passed with 28 unchanged exhaustive shard cases deselected. The 19 report tests passed again after adding an offline replay run-ID input. Saved run 40909862501 produced 401.3 minutes across 34 application jobs and correctly disclosed 33 inverted created/start timestamp rows. The one valid wait sample is not a workflow-wide latency measurement. Full Ruff check/format on the new test, scoped lint on the changed contract, and diff checks pass. Independent review found no actionable findings and exercised seven additional read-only probes.

Modified derived-artifacts.yml, the queue-pressure contract and new test_ci_cost_report.py; added the cost-reporting spec and plan and amended ADR-103. ADR required: yes; backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md records the narrow diagnostic exception. No application test selection, dependencies, timeouts or merge protection changed; no runner job added.

Hosted summary rendering, timestamp availability and timeout behavior still require qualification. Local tests verify calls to Octokit pagination, not real multipage delivery. No full suite or live performance improvement is claimed. Task remains In Progress pending hosted evidence and integration after PR3316.
2026-11-01 final integration onto confirmed PR3316 merge a68971781f3826a108d8116f06bb778aeffa3ae4. All three patches preserve source behavior. Parsed workflow comparison confirms only aggregate actions: read and the final diagnostic step differ; all prior jobs/verdicts remain. Five affected modules: 195 passed, 32 unchanged exhaustive shard-distribution cases deselected, stable source hashes. Receipt pr3317-final-b907bde1; details in Docs/superpowers/qa/2026-11-01-pr3317-backlog-integration.md. Scoped Ruff/AST/diff checks pass. No real pagination, hosted cancellation, throughput or billing qualification is claimed; hosted CI is user-waived. Independent final-head review and exact-lease merge follow. ADR-103 applies; no new ADR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
