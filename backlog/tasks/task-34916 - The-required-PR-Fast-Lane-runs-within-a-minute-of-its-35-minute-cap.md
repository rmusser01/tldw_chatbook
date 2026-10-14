---
id: TASK-34916
title: The required PR Fast Lane runs within a minute of its 35-minute cap
status: In Progress
created_date: 2026-10-12 22:22
labels:
- ci
- tests
priority: high
assignee:
- '@claude'
updated_date: 2026-10-14 15:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR Fast Lane is one of the jobs behind the required check, and it has a 35-minute cap. Over 34 pull-request runs measured on 2026-10-12 its median was 31.6 minutes and its slowest 34.0; one run on PR #3091 took 34 minutes 10 seconds. When a job reaches its cap GitHub cancels it, and a cancelled job leaves no test report, so the next few slow tests added to the lane will turn into failures with no explanation, which the merge queue then counts against the PR. The owner's rule for CI time is to add a shard rather than raise the minutes. Measured during the review of PR #3091 (TASK-34789).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The PR Fast Lane's slowest run over at least 20 pull-request runs leaves at least 5 minutes under its cap
- [x] #2 No test is dropped from the required check to achieve it
- [x] #3 The contract test that pins the lanes and their caps reflects the new layout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure: per-job and per-step durations of the PR Fast Lane over the recent pull_request runs (gh api jobs).
2. The lane is two serial pytest steps of about 15 min each; move the admission-sensitive step into its own job (new runner, same install line, its own pytest process as before), so the PR Fast Lane keeps only the sandboxed step. Cap stays 35; the new job gets its own cap sized from the measured step.
3. Add the new job to the required aggregator's needs with its own verdict step.
4. Update the contract tests that pin the lanes (test_ci_queue_pressure_contract, test_derived_artifacts_workflow, test_console_p0_regression_gate, test_personal_context_regression_gate) and ADR-103.
5. Verify: Tests/CI, YAML parse, actionlint if present, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Split the PR Fast Lane's two serial pytest invocations into two parallel jobs instead of raising the cap.

Measured (gh api .../actions/runs/<id>/jobs, 90 pull_request runs of derived-artifacts.yml with both steps completed, 2026-10-12 19:39Z to 2026-10-14 14:26Z): the job took median 31.5 / p90 34.5 / max 39.5 min and 4 of the 90 were cancelled at the 35-min cap. The two steps are halves of it: 'Run fast PR contract' median 15.2 / max 19.9 min (Tests/UI/test_mcp_workbench.py alone is ~8.8 min of it), 'Run admission-sensitive suites' median 15.4 / max 18.7 (test_console_runtime_ownership 4.3, test_roleplay_header 3.3, test_roleplay_hostile_names 2.1, test_console_chat_create_integration 1.9 min); setup ~0.8 min.

Change: new job pr-fast-lane-admission ('PR Fast Lane (admission-sensitive)') holds the admission-sensitive step with its argv byte-for-byte unchanged (checked by parsing both workflow versions), the same install line and the same 35-min cap the owner set for these suites (TASK-33910.2). It was always its own pytest process, and each process makes its own mkdtemp bootstrap profile (Tests/conftest.py), so a fresh runner changes nothing it relied on. Dropped its 'if: always()' (it is the only test step in its job now). The required aggregate needs it and has a 'Require successful PR fast lane (admission-sensitive)' verdict in the same !cancelled() shape.

Expected durations (replay of the same 90 runs: job minus the moved step, and setup plus the moved step): PR Fast Lane median 16.1 / p90 17.9 / max 20.8 min (14.2 min under the cap); admission job median 16.3 / p90 17.5 / max 20.1 min (one hung test at --timeout=300, ~5.3 min with the private-profile wrapper's report, still ends ~10 min under). Cost: one more runner per PR (ADR-103 note).

AC#1 is left unchecked on purpose: it is about 20 measured runs of the new layout, which only the PR's and later PRs' CI can produce. Verify with: gh run list --workflow derived-artifacts.yml --event pull_request --limit 40 --json databaseId, then the max of 'PR Fast Lane' job durations from gh api repos/rmusser01/tldw_chatbook/actions/runs/<id>/jobs must be <= 30 min.

Tests: Tests/CI + Tests/Packaging/test_python_runtime_floor.py green (910 passed, 1 skipped); negative controls: dropping the new job from needs, deleting its verdict, or weakening its verdict to == 'failure' each fail _assert_required_aggregation.

Files: .github/workflows/derived-artifacts.yml, Tests/CI/test_ci_queue_pressure_contract.py, Tests/CI/test_derived_artifacts_workflow.py, Tests/CI/test_console_p0_regression_gate.py, Tests/CI/test_personal_context_regression_gate.py, backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
