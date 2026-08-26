# PR 1642 Qodo Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Qodo's valid public-docstring finding while preserving and
proving the approved retry-safe lease and dependency-free maintainer contracts.

**Architecture:** This is a documentation-only runtime change. Existing tests
remain the authority for lease retry and `python -S` execution; new focused tests
pin the docstring sections and explicit `--output` behavior. No lease, path, or
filesystem implementation changes are admitted.

**Tech Stack:** Python 3.11+, pytest, standard-library subprocess execution,
Ruff.

**ADR required:** no

**ADR path:**
`backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`

**Reason:** ADR-050 already governs exact lease ownership and retry. This plan
changes documentation and tests only; it does not alter an architecture
boundary.

---

### Task 1: Pin the review contracts

**Files:**
- Modify: `Tests/UI/test_model_curated_view.py`
- Modify: `Tests/TTS/test_audio_cpp_artifact_catalog.py`

- [ ] **Step 1: Write the failing Model Library docstring test**

Add a parametrized test over the five public helpers in
`model_curated_view.py`. Require `Args:` and `Returns:` in each docstring.

- [ ] **Step 2: Write the failing maintainer-script docstring test**

Add a parametrized test over `validate_commit`, `refresh_manifest_bytes`, and
`main`. Require `Args:` and `Returns:` in each docstring and `Raises:` for all
three functions, matching their exposed validation, parser, and I/O failures.

- [ ] **Step 3: Write the explicit output-path regression**

Extend the direct dependency-free command coverage with the complete invocation:

```bash
python -S scripts/refresh_audio_cpp_artifact_manifest.py \
  --commit <exact 40-hex commit> \
  --manifest <empty fixture manifest> \
  --output <tmp destination>
```

Assert success, empty stdout, and
`output_path.read_bytes() == expected_bytes`, including the trailing newline.

- [ ] **Step 4: Verify RED**

Run the three new tests. Expected: the two docstring tests fail on missing
sections; the existing output behavior test passes.

- [ ] **Step 5: Commit the tests**

Commit the RED tests separately so the behavioral requirement is reviewable.

### Task 2: Add the minimal documentation

**Files:**
- Modify: `tldw_chatbook/UI/Screens/model_curated_view.py`
- Modify: `scripts/refresh_audio_cpp_artifact_manifest.py`

- [ ] **Step 1: Expand the eight public docstrings**

Document each argument and return value. Add `Raises:` to `validate_commit`,
`refresh_manifest_bytes`, and `main`, matching their existing public failure
contracts. The `main` docstring must state that `--manifest` and `--output` are
explicit trusted-maintainer paths and that `--output` intentionally permits an
arbitrary destination. Do not change executable statements.

- [ ] **Step 2: Verify GREEN**

Run the three new tests. Expected: all pass.

- [ ] **Step 3: Re-run unchanged ownership boundaries**

Run:

```bash
pytest -q \
  Tests/Model_Artifacts/test_operation_leases.py::test_release_raises_stable_error_without_masking_unlock_failure \
  Tests/Model_Artifacts/test_operation_leases.py::test_unlock_failure_retains_real_shared_lock_until_release_retry \
  Tests/TTS/test_audio_cpp_artifact_catalog.py::test_refresh_command_runs_directly_without_network_for_empty_manifest
```

Expected: four parametrized cases pass. This proves the rejected lease/path
recommendations did not alter their approved contracts.

- [ ] **Step 4: Run focused suites and static checks**

Run the two changed test files, Ruff check/format check on changed Python files,
and `git diff --check`.

- [ ] **Step 5: Run the repository test gate and self-review**

Run `pytest -q`. If the environment produces existing failures, run the
identical command on unmodified `origin/dev`, compare exact failing node IDs,
and document only unchanged baseline outcomes. Review the complete branch diff
for executable changes, privacy, dependency, and task-scope drift.

- [ ] **Step 6: Commit the implementation**

Commit the docstring-only runtime change.

### Task 3: Close the task and publish the remediation

**Files:**
- Modify: `backlog/tasks/task-16301 - Address-PR-1642-Qodo-review-feedback.md`

- [ ] **Step 1: Record evidence and close TASK-16301**

Edit the five-digit task file directly: check all ACs, add concise
implementation notes, retain the ADR-050 link, and set status to Done. Do not
use `backlog task edit 16301`, which can create a ghost task file.

- [ ] **Step 2: Verify branch hygiene**

Run Backlog listing/parsing, `git status --short backlog/`, and confirm no
`backlog/tasks/task-task- - .md` ghost exists. Run `git diff --check`.

- [ ] **Step 3: Commit the task closeout**

Commit the direct task-file update, then confirm the worktree is clean.

- [ ] **Step 4: Push and open a separate PR against `dev`**

Summarize the accepted docstring fix and the tested technical dispositions for
the lease and explicit output-path recommendations.

- [ ] **Step 5: Reply to Qodo on PR 1642**

Post a concise top-level disposition linking the follow-up PR: accepted
docstrings fixed; lease retry retained because closing would destroy exact
authority; central path validator rejected because it breaks the approved
dependency-free arbitrary-destination maintainer contract.
