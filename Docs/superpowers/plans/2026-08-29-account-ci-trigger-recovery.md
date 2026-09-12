# Account CI Trigger Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `tldw_chatbook` from duplicating its heavy CI workload so the shared `rmusser01` Actions queue can drain and fresh pull-request checks can reach verdicts.

**Architecture:** Enforce the approved workload policy at GitHub Actions trigger boundaries, before runs enter the account queue. Keep the ordinary suite on PRs into `dev`, keep a release run on pushes to `main`, and gate ordinary jobs out of the scheduled event so only `nightly-deep` consumes scheduled runners.

**Tech Stack:** GitHub Actions YAML, Python 3.12, PyYAML, pytest, Ruff, GitHub CLI

---

### Task 1: Pin the new trigger and schedule contract

**Files:**
- Modify: `Tests/CI/test_ci_queue_pressure_contract.py`
- Modify: `Tests/CI/test_github_actions_test_workflow.py`
- Read: `.github/workflows/test.yml`
- Read: `.github/workflows/derived-artifacts.yml`
- Read: `.github/workflows/css-bundle-guard.yml`
- Read: `.github/workflows/perf-guard.yml`
- Read: `.github/workflows/backlog-guard.yml`

- [ ] **Step 1: Replace the obsolete PR #602 job-guard contract**

In `Tests/CI/test_ci_queue_pressure_contract.py`, replace the promotion-guard constants and test with trigger-level assertions:

```python
SCHEDULE_SKIP = "${{ github.event_name != 'schedule' }}"
ALWAYS_SCHEDULE_SKIP = "${{ always() && github.event_name != 'schedule' }}"
ORDINARY_TEST_JOB_CONDITIONS = {
    "core-tests": SCHEDULE_SKIP,
    "artifact-lease-spike": SCHEDULE_SKIP,
    "artifact-lease-shape": SCHEDULE_SKIP,
    "artifact-lease-gate": ALWAYS_SCHEDULE_SKIP,
    "ui-tests": SCHEDULE_SKIP,
    "textual-minimum": SCHEDULE_SKIP,
    "test-summary": ALWAYS_SCHEDULE_SKIP,
}


def test_dev_merge_creates_one_heavy_tests_run() -> None:
    triggers = _workflow("test.yml").get("on", _workflow("test.yml").get(True))
    assert triggers["pull_request"]["branches"] == ["dev"]
    assert triggers["pull_request"]["types"] == PULL_REQUEST_TYPES
    assert triggers["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in triggers


def test_schedule_skips_the_ordinary_test_suite() -> None:
    jobs = _workflow("test.yml")["jobs"]
    for job_name, expected in ORDINARY_TEST_JOB_CONDITIONS.items():
        assert jobs[job_name].get("if") == expected
    assert jobs["nightly-deep"]["if"] == (
        "github.event_name == 'schedule' || "
        "github.event_name == 'workflow_dispatch'"
    )
    assert jobs["all-tests"]["if"] == "github.event_name == 'workflow_dispatch'"


def test_focused_guards_ignore_the_permanent_main_promotion_pr() -> None:
    for workflow_name in STANDALONE_JOB_GUARDS:
        workflow = _workflow(workflow_name)
        triggers = workflow.get("on", workflow.get(True))
        assert triggers["pull_request"]["branches"] == ["dev"]
        assert triggers["push"]["branches"] == ["dev", "main"]
```

Also assert that the changed workflows no longer contain `pull_request.number` or a PR #602 job-level exception.

- [ ] **Step 2: Update existing shape assertions for schedule exclusions**

In `Tests/CI/test_github_actions_test_workflow.py`, replace `PROMOTION_GUARD` and `ALWAYS_PROMOTION_GUARD` with `SCHEDULE_SKIP` and `ALWAYS_SCHEDULE_SKIP`. Update the artifact-lease description to state that PRs use Ubuntu, `main` pushes/manual runs use the three-OS spike, and schedules use `nightly-deep` instead of the ordinary spike.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_github_actions_test_workflow.py \
  Tests/CI/test_derived_artifacts_workflow.py \
  -q --confcutdir=Tests/CI
```

Expected: failures show that `test.yml` still targets `dev` pushes and `main` PRs, ordinary jobs still carry PR #602 guards, schedules still admit ordinary jobs, and focused guards have no `pull_request.branches` filter.

### Task 2: Apply the minimum workflow trigger fix

**Files:**
- Modify: `.github/workflows/test.yml`
- Modify: `.github/workflows/derived-artifacts.yml`
- Modify: `.github/workflows/css-bundle-guard.yml`
- Modify: `.github/workflows/perf-guard.yml`
- Modify: `.github/workflows/backlog-guard.yml`

- [ ] **Step 1: Narrow the heavy workflow at the trigger boundary**

Change the start of `.github/workflows/test.yml` to:

```yaml
on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["dev"]
    types: [opened, synchronize, reopened, ready_for_review]
  workflow_dispatch:
  schedule:
    - cron: '30 8 * * *'
```

- [ ] **Step 2: Exclude ordinary jobs from the scheduled event**

Set these exact conditions:

```yaml
# core-tests, artifact-lease-spike, artifact-lease-shape, ui-tests,
# textual-minimum
if: ${{ github.event_name != 'schedule' }}

# artifact-lease-gate, test-summary
if: ${{ always() && github.event_name != 'schedule' }}
```

Remove every PR #602-specific job condition and update nearby comments. Keep `all-tests`, `nightly-deep`, and both `max-parallel: 3` settings unchanged.

- [ ] **Step 3: Narrow synchronization-capable focused guards**

Add `branches: [dev]` below `pull_request:` in the four focused guard workflows. Keep their existing push branches and paths. Remove the obsolete PR #602 job conditions and comments.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the test-first workflow change**

```bash
git add \
  .github/workflows/test.yml \
  .github/workflows/derived-artifacts.yml \
  .github/workflows/css-bundle-guard.yml \
  .github/workflows/perf-guard.yml \
  .github/workflows/backlog-guard.yml \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_github_actions_test_workflow.py
git commit -m "fix(ci): stop account-wide workflow duplication"
```

### Task 3: Verify the complete local CI contract

**Files:**
- Verify: `.github/workflows/*.yml`
- Verify: `Tests/CI/test_*.py`

- [ ] **Step 1: Run all affected CI contract tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_github_actions_test_workflow.py \
  Tests/CI/test_derived_artifacts_workflow.py \
  Tests/CI/test_task2062_1_gguf_import_evidence.py \
  Tests/CI/test_task2062_2_gguf_source_evidence.py \
  -q --confcutdir=Tests/CI
```

Expected: all selected tests pass. Do not run the repository-wide suite; AGENTS.md requires targeted verification unless the owner requests a full sweep.

- [ ] **Step 2: Run Ruff on changed Python contracts**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_github_actions_test_workflow.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_github_actions_test_workflow.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Parse all changed YAML and inspect the diff**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c \
  'from pathlib import Path; import yaml; [yaml.safe_load(p.read_text()) for p in map(Path, [".github/workflows/test.yml", ".github/workflows/derived-artifacts.yml", ".github/workflows/css-bundle-guard.yml", ".github/workflows/perf-guard.yml", ".github/workflows/backlog-guard.yml"])]'
git diff --check
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- .github/workflows Tests/CI
```

Expected: YAML parsing and diff checks exit 0; the diff contains only the approved trigger/condition contract and its tests.

### Task 4: Record implementation evidence without closing live criteria

**Files:**
- Modify: `backlog/tasks/task-22250 - CI runs are swept by simultaneous burst cancellations.md`

- [ ] **Step 1: Add an implementation update**

Record the authenticated account evidence, changed trigger contract, focused local verification, ADR decision, and the fact that the two live acceptance criteria remain open until GitHub produces a post-change PR verdict. Do not mark the task Done.

- [ ] **Step 2: Commit the task evidence**

```bash
git add 'backlog/tasks/task-22250 - CI runs are swept by simultaneous burst cancellations.md'
git diff --cached --check
git commit -m "docs: record account CI recovery evidence"
```

### Task 5: Open the recovery PR and drain only obsolete work

**Files:**
- Remote branch: `codex/task-22250-account-ci-recovery`
- GitHub pull request: base `dev`

- [ ] **Step 1: Rebase on the latest `origin/dev` and repeat Task 3 verification**

Confirm `git log --oneline origin/dev..HEAD` contains only this task's commits and `git merge-base --is-ancestor origin/dev HEAD` exits 0.

- [ ] **Step 2: Push and open the PR**

Push the branch and create a PR whose body includes the account usage evidence, trigger reduction, test evidence, ADR result, and explicit live-verdict requirement.

- [ ] **Step 3: Audit the queue using current authority**

Build the preservation set from current open-PR head SHAs plus current `dev`/`main` SHAs in both `tldw_chatbook` and `tldw_server`. Cancel only obsolete idle queued/pending runs. Preserve runs with executing jobs and record zero-job HTTP 409 ghost entries.

- [ ] **Step 4: Verify post-change runner admission**

Inspect the recovery PR by head SHA, not PR number alone. Success requires a newly created `Tests` run to start jobs and eventually produce a completed verdict. Queue-count reduction alone is not evidence.

### Task 6: Address review and merge only after live proof

**Files:**
- GitHub PR review threads and Qodo comments
- Potentially the same workflow/test/task files when a finding is valid

- [ ] **Step 1: Inspect every review comment**

Use `superpowers:receiving-code-review`. Verify each finding against GitHub event semantics and the repository contract; implement valid findings test-first and respond to or resolve every thread.

- [ ] **Step 2: Rebase on the latest `dev` and repeat targeted verification**

After any base movement, rerun the complete Task 3 commands and confirm the pushed head SHA is the one being observed.

- [ ] **Step 3: Complete TASK-22250 only with live evidence**

Check both remaining acceptance criteria only when the recovery PR's test workflows complete without a simultaneous sweep and `Tests` has a completed verdict. Add the final live run URLs and results to Implementation Notes, then set the task status to Done.

- [ ] **Step 4: Merge and verify the exact head**

Merge only after local verification, review resolution, and live criteria. Confirm the PR state is `MERGED`, `mergeCommit` is non-null, and `headRefOid` equals the verified SHA.
