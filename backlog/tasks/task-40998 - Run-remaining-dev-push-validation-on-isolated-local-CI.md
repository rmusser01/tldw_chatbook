---
id: TASK-40998
title: Run remaining dev-push validation on isolated local CI
status: In Progress
created_date: 2026-11-14 17:57
assignee:
- '@codex'
updated_date: 2026-11-15 23:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the owner-requested migration of development-branch CI execution to the existing local server, then establish a qualified baseline for the separate runtime-optimization comparison.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pushes to dev route all 14 Library shell shards, Perf Guard and the Backup_Recovery owner suites to the configured disposable test pool; clearing its variable preserves hosted rollback and other event routing remains unchanged.
- [x] #2 Existing test selections, isolation, latency budgets, per-test and job deadlines, Library parallelism, dependency ordering and required-status policy remain intact.
- [ ] #3 Focused routing and workflow contracts, independent review and actual local dev-push Library, Perf Guard and Backup_Recovery owner jobs pass before the migration is declared complete; deployed controller and stop deadlines accommodate the longest job without changing CPU, memory, isolation or workflow budgets.
- [ ] #4 After successful local dev qualification, a fresh full normal pull_request Derived Artifacts run on the final migration head is retained with per-attempt timing, all job records, runner/environment evidence and contention limits; tested/deployed contents match before the optimization comparison starts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing decision)
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Extend the existing accepted runner-placement policy from PR/control work to remaining dev-push Linux work, under the explicit owner instruction on2026-11-14. No new isolation or runtime boundary.
1. Inventory dev-push workflows and retain historical PR3529 evidence; confirm dependencies, limits and deployed pool compatibility.
2. Amend ADR359 for dev-push Library and Perf Guard placement. Add event/ref/variable routing cases, observe expected failures, then update only their runs-on expressions and relevant operational docs.
3. Run focused workflow/routing contracts and independent review without local full-suite sweeps. Preserve all runtime/test commands and limits.
4. Schedule around legitimate local work; create a migration PR with a normal full pull_request gate for integration. Keep required-review and queue policy; do not manually dispatch the required gate.
5. After normal merge, verify actual dev-push Library and Perf Guard execution on local guests, fix any failures without weakening budgets, and then rerun the final successful normal pull_request gate in a clear local workload window for a fresh baseline after deployment, verifying tested/deployed content and environment equivalence. Handoff to PR3532 owner; preserve failed attempts separately.
6. New dev coverage on 2026-11-15 adds Backup_Recovery owner suites. Include its dev-only routing and preserve four pytest workers, its 55-minute job/50-minute step/300-second per-test limits, existing deselections and non-required policy. Amend ADR-359 for a bounded 60-minute test-pool execution/drain limit and 65-minute service-stop grace; validate the deadline hierarchy and gracefully drain active guests before applying it. Qualify this actual dev job alongside Library and Perf Guard. Keep updates in this chat as the owner subsequently requested.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The remaining dev-push workflow inventory contains Library shell (14 shards) and Perf Guard (one job); preflight, lint, aggregate and merge control already use the local control pool. Updated only the two jobs' runs-on expressions to select CI_TEST_RUNNER for push to refs/heads/dev, with ubuntu-latest fallback. PR, main-push and manual behavior remains unchanged. The existing runner image, credentials, private-network policy, fleet resources and controller are untouched.

Expanded the existing routing evaluator across dev/main/feature refs and both label variables, including the Perf Guard workflow. Before the workflow edits, four dev/local-label cases failed as expected and558 passed. Afterward,783 focused routing/workflow/queue-pressure/shard-count contracts passed in31.21s; the one storage-lease harness contract passed separately with normal conftest loading in0.51s. Parsed old/new workflow objects are identical after restoring the two runs-on fields: all commands, tests, fixture flags, isolation, budgets,20-minute job caps, Library max-parallel7, dependencies and required-status semantics are unchanged. Ruff check/format, whitespace and backlog validators pass. Logs: /tmp/tldw-ci-dev-routing-{red,focused,harness}.log.

Amended ADR359 before implementation under the owner's2026-11-14 instruction to finish dev-job migration. Updated operational routing/rollback and event-specific qualification guidance. Existing local closeout files for PR3529 accompany this follow-up: its three finished repair tasks, integration status, and historical raw baseline artifacts. That successful88m09s gate is retained as historical evidence and will not substitute for the newly requested final baseline.

Live qualification and new baseline remain pending. After merge, actual Library and Perf Guard dev-push execution must pass on local runner identities. Then rerun the final migration head's normal pull_request gate in a clear window, verify tested/deployed source equivalence and retain per-attempt timing, including the rerun request timestamp rather than counting the original attempt's age as execution time. Perf Guard's real hardware latency thresholds are unchanged; source contracts do not establish hardware compatibility or timing. PR3457 currently occupies the pool; no unrelated work was cancelled. Independent review found no blockers: it independently inventoried every workflow and verified event/ref/rollback semantics, unchanged commands and deadlines, guest tooling, and hardware-sensitive Perf Guard limits. Its sequence concern is addressed by measuring a fresh normal PR rerun after actual dev qualification, rather than promoting a predeployment observation. Publication and live evidence remain pending.
2026-11-15 amendment: rebased onto cc6006fcf59328353004206bba8724c399216749, which adds the dev-only Backup_Recovery owner job. Its runs-on now uses the same dev/ref/variable selection and hosted rollback as Library and Perf Guard. Parsed old/new workflow objects are identical after restoring that single field: all 14 files, eight existing deselections, four workers, 300-second per-test, 50-minute step and 55-minute job budgets, and non-required status remain unchanged. ADR359 now accommodates the job with 60-minute test execution/drain and 65-minute service stop limits; control15m, pool sizing, resources, image and security boundaries remain unchanged. The deadline contract failed against the old45m controller; two dev-routing cases also failed as expected. Routing now passes591 cases. Broader targeted workflow/queue/shard contracts passed819 cases, with their harness-dependent storage-lease test passing separately under normal conftest loading (0.39s). Ruff passes; the changed fixture/routing files pass format. The large queue-pressure file already had unrelated formatting drift at HEAD; its sole assertion replacement adds none. Backlog file/id and ADR validators pass. Related deterministic fixture repair is tracked by TASK41176. Controller activation, independent review, actual local dev jobs and new clean-window baseline remain pending. All updates stay in this chat.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
