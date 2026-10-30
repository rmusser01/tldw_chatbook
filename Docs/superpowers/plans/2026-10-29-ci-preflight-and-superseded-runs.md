# CI preflight and superseded runs implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement the approved scope inline, then request one independent whole-change review.

**Goal:** Free runners held by obsolete commits and avoid test fan-out after predictable gate failures.

**Architecture:** A checkout-free cleanup job lives in the existing queue workflow, outside the test workflow's concurrency group. Artifact checks and static selector validation move before the unchanged test lanes; the stable required check becomes a verdict-only aggregate.

**Tech Stack:** GitHub Actions, actions/github-script, Python 3.12, pytest, PyYAML, Node for the inline-script test harness.

**Spec:** `Docs/superpowers/specs/2026-10-29-ci-preflight-and-superseded-runs.md`

ADR required: yes (amend existing decisions).
ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md` and `backlog/decisions/218-in-repo-merge-queue.md`.
Reason: change CI admission ordering and add narrowly scoped Actions cancellation authority.

## Global constraints

- Same required context, all existing selectors, 28 UI shards, max-parallel 17, timeouts and application dependency lists.
- Preserve current-head evidence, fork/bot boundaries, failed-job retries, queue kicks and full manual dispatch semantics.
- No full test sweep, branch-protection change, queue admission redesign or live backlog cancellation from this session.

## Review focus

- Live PR changes head between discovery and cancellation: stop without cancelling new-head evidence.
- API results include unrelated branches, repositories, PR numbers or missing metadata: preserve them.
- Failed/skipped/cancelled preflight: no expensive lanes and no successful required aggregate.
- Push and queue-kick events: skipped application lanes must not hide failed preflight.
- A failed checker must not hide subsequent checker reports; no guard or test selector disappears.

## Task 1: Superseded-run cleanup

Files: `.github/workflows/merge-queue.yml`, `Tests/CI/test_cancel_superseded_ci.py`, `Tests/CI/test_merge_queue_workflow.py`.

- [x] Add behavioral tests executing the real inline script through a mocked Octokit API, plus trigger/permissions/isolation contracts. Observe expected failures before implementation.
- [x] Implement the synchronize-only job and identity/head checks from the spec. Keep the queue's original events and permissions separate.
- [x] Verify current-head retention, unrelated/missing metadata, pagination, head races, HTTP 409 and other API errors using the focused tests.

## Task 2: Preflight before test fan-out

Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_derived_artifacts_workflow.py`, `Tests/CI/test_ci_queue_pressure_contract.py`.

- [x] Add failing contracts for preflight dependencies, explicit aggregate rejection, minimal preflight installation and conserved guards/targets.
- [x] Move existing artifact steps to preflight, add the two existing static selector tests, and make every application lane depend on preflight.
- [x] Keep the required aggregate always running, verdict-only, and explicitly require successful preflight on every event.
- [x] Run targeted CI contracts with `--noconftest` in a clean pytest/PyYAML environment, then the actual preflight checks. Compare workflow selections against the base revision.

## Task 3: Review and delivery

- [x] Run focused queue rules/actions/workflow regressions, formatter/linter and `git diff --check`.
- [x] Obtain independent review; resolve consequential findings with reproducing tests.
- [x] Record local evidence and any hosted qualification still outstanding in TASK-37800.
- [ ] Publish the locally verified branch as a draft PR. Do not arm or merge it.

## Execution ledger

- Started from `origin/dev` at `ff815f8559744049be36295d9468811eb4e3c020` in a managed worktree. The approved first two audit changes are the implementation scope; wider admission control remains a follow-up.
- Initial behavioral and graph contracts failed before implementation. Independent review found that cancelling an old run's queue tick could interrupt the controller approving replacement CI. Added eight failing cases (six verdict/tick states, job-inventory failure and empty branch), then preserved every run whose required verdict or queue tick appears. All 46 cleanup tests pass after the fix; no other actionable review findings remain.
- Final targeted command: `python -m pytest --noconftest -o addopts='' -p no:cacheprovider --basetemp=.test-tmp/task37800-final-2 Tests/CI/test_cancel_superseded_ci.py Tests/CI/test_derived_artifacts_workflow.py Tests/CI/test_ci_queue_pressure_contract.py Tests/CI/test_merge_queue_workflow.py Tests/CI/test_merge_queue_rules.py Tests/CI/test_merge_queue_actions.py Tests/CI/test_ui_lane_shard_count_statements.py Tests/CI/test_pr_workflows_dispatch_safe.py Tests/CI/test_console_p0_regression_gate.py Tests/CI/test_response_rules_critical_gate.py -k 'not ui_gate_shards_cover' --tb=short -q`: **631 passed, 28 deselected**, 64.66 seconds. The 28 deselected cases exhaustively redistribute the unchanged UI targets across shard counts 1 through 28. Two config warnings reflect the deliberately minimal environment lacking asyncio/timeout plugins.
- All 13 real artifact guards and both selector tests passed locally. The unchanged Mermaid guard requires Python 3.12.11 and exact LF source bytes: a normal Windows checkout and an initial Git archive inherited `core.autocrlf=true` and failed reproduction. An isolated export with `core.autocrlf=false`, verified against Git blob bytes, passed under 3.12.11. No asset source or pin changed. The other guards ran under local Python 3.12.10; hosted preflight retains 3.12.11.
- Parsed workflow comparison against the base confirmed all seven application job definitions are identical except their new preflight dependency, and all 13 moved guards are identical. Scoped Ruff checks (`E9,F63,F7,F82`) on the five changed test files, formatting of the new test file, and `git diff --check` pass. Existing test-file formatting was preserved.
- No full suite, live cancellation, merge, auto-merge or protection change was performed. TASK-37800 stays In Progress pending hosted qualification. GitHub scheduling, cancellation effects and throughput improvement have not been measured for this branch.
