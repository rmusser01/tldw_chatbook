# TASK-31550 UI CI Root Install Implementation Plan

> **For Codex:** Implement this plan test-first and stop if the focused regression does not demonstrate the diagnosed failure.

**Goal:** Make every comprehensive UI CI shard install the repository distribution so bundled packages such as `tldw_profile_core` are importable during test setup.

**Architecture:** Preserve the existing root packaging boundary and the twelve-way UI shard design. Add the missing editable install to the UI job and protect that job-specific precondition with a workflow-shape regression test.

**Tech Stack:** GitHub Actions YAML, pytest, setuptools editable installs.

**Backlog task:** `backlog/tasks/task-31550 - Install-the-root-project-in-comprehensive-UI-CI-shards.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a CI installation correction that preserves existing packaging and runtime boundaries.

## Task 1: Pin the missing UI-job installation contract

**Files:**

- Modify: `Tests/CI/test_github_actions_test_workflow.py`
- Test: `Tests/CI/test_github_actions_test_workflow.py`

1. Add a focused test that extracts the `ui-tests` job, locates its dependency-install and pytest steps, and requires `pip install -e .` to occur before pytest.
2. Run only that test with `--confcutdir=Tests/CI`.
3. Confirm it fails because the current UI job has no editable root install.

## Task 2: Restore the root package install

**Files:**

- Modify: `.github/workflows/test.yml`
- Test: `Tests/CI/test_github_actions_test_workflow.py`

1. Add `pip install -e .` to the UI job's existing dependency-install step.
2. Re-run the focused regression and confirm it passes.
3. Run the complete CI workflow contract file.

## Task 3: Verify the packaging boundary

**Files:**

- Verify: `pyproject.toml`
- Test: `Tests/Packaging/test_profile_core_packaging.py`

1. Run the profile-core packaging tests to prove the root distribution still includes the bundled package.
2. Build a temporary no-dependency editable environment and use `importlib.util.find_spec` to confirm `tldw_profile_core` is discoverable after the same install command used by CI.
3. Run whitespace/static checks for the touched files.

## Task 4: Record comprehensive evidence

**Files:**

- Modify: `backlog/tasks/task-31550 - Install-the-root-project-in-comprehensive-UI-CI-shards.md`

1. Re-run the comprehensive workflow on the corrected branch when it is publishable.
2. Verify that UI reports contain executed test outcomes rather than setup-wide `ModuleNotFoundError` results.
3. Check every acceptance criterion, add concise implementation notes, and mark the task Done only if all Definition of Done conditions are satisfied.
