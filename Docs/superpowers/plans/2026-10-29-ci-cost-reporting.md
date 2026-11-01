# CI cost reporting implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, then obtain one independent whole-change review.

**Goal:** Report qualification cost and waiting time without adding runner jobs or changing required verdicts.

**Architecture:** One bounded github-script step in the existing aggregate reads job metadata and writes its summary. Workflow contracts allow failure only for that diagnostic step.

**Tech Stack:** GitHub Actions, JavaScript, pytest/PyYAML and Node for behavioral tests.

**Spec:** `Docs/superpowers/specs/2026-10-29-ci-cost-reporting.md`

ADR required: yes (amend existing decision).
ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`.
Reason: CI cost reporting and a narrow exception to aggregate step failure policy.

## Constraints and review focus

- No additional jobs, checkout or dependencies; preserve all qualification steps and selectors.
- Only `Report qualification cost` can tolerate failure; mutations on verdicts still fail contracts.
- No billing claim, no invented zero for missing data, and no duplicate or cross-attempt counting.
- Current attempt and cumulative observations are distinct; aggregate/tick work is excluded explicitly.
- API and summary failures cannot green or redden the test verdict.

## Implementation

- [x] Add a real-inline-script harness and failing calculation/metadata/workflow tests in `Tests/CI/test_ci_cost_report.py`.
- [x] Add the reporting step and job-local actions-read permission in `.github/workflows/derived-artifacts.yml`; update the aggregate contract for the narrow diagnostic exception.
- [x] Run the focused reporting and workflow contracts; exercise a saved audit sample without GitHub writes.
- [x] Amend ADR-103, record exact evidence in TASK-38100, and obtain independent review with regression tests for any consequential finding.
- [ ] Publish as a follow-up to #3316 after reconciling the base; keep hosted evidence distinct from local checks.

## Execution ledger

- Created a separate managed worktree from #3316 head `383bf3da799b4276fb124e9225e494a468ae9a6d`; the first PR remains untouched while its Qodo-credit exception is pending.
- RED: 18 reporting tests failed because the workflow had no report step. GREEN: 195 focused tests passed (28 unchanged exhaustive shard-distribution cases deselected) using `python -m pytest --noconftest -o addopts='' -p no:cacheprovider Tests/CI/test_ci_cost_report.py Tests/CI/test_derived_artifacts_workflow.py Tests/CI/test_ci_queue_pressure_contract.py Tests/CI/test_cancel_superseded_ci.py Tests/CI/test_merge_queue_workflow.py -q -k 'not ui_gate_shards_cover' --tb=short`. After allowing the harness to use an observed run ID, all 19 report tests passed again. Two config warnings reflect the intentionally minimal pytest/PyYAML environment.
- Offline replay of saved run 40909862501: 34 measured application jobs, 401.3 runner-minutes. The API sample has created-at timestamps later than start for 33 jobs; the report disclosed 33 incomplete wait rows rather than inventing zero waits. Its 6.4-minute median/max wait covers only the one valid row and is not a workflow-wide latency result.
- Independent review found no actionable findings and separately exercised seven read-only JavaScript probes. It confirmed unchanged application jobs and correct attempt/all endpoint selection. Its limitations remain: fake API tests verify pagination calls, not actual multipage delivery; hosted timestamp availability, summary rendering and timeout behavior require hosted qualification.
- Full Ruff check and format check of the new test, scoped syntax lint on the changed contract, and `git diff --check` pass. No full suite or GitHub settings change. No additional change was needed in `test_derived_artifacts_workflow.py`; its existing graph/verdict checks already pass.
