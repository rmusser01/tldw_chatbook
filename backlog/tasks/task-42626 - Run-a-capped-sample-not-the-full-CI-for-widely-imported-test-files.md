---
id: TASK-42626
title: Run a capped sample, not the full CI, for widely imported test files
status: Done
assignee:
  - '@claude'
created_date: '2026-11-26 15:30'
updated_date: '2026-11-26 16:10'
labels:
  - ci
dependencies:
  - TASK-42521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner wants dev PRs to merge quickly after sanity checks, with the nightly and release runs as the broad gates (ADR-366). The selection still sent a PR to the full run, about 63 jobs including 42 UI shards, whenever it touched a test module imported by more than 8 test files, a shared helper under `Tests/`, or more than 20 test files of its own.

PR #3769 is the case that cost the most. Its preflight said `Shared test module (9+ importers): Tests/Chat/test_console_dispatch_recovery.py`, and its full run took 3 h 26 min, because jobs waited up to about 2 h 50 min for runners on the account pool shared with tldw_server. The one-at-a-time merge queue held every armed PR behind it for 4.5 h. A light-tier PR (#3518) went from rebase to merge in 21 minutes.

These cases should run the light tier with a capped, deterministic sample of the affected test files. Inputs that can break CI itself, the schema, and the profile isolation harness keep the full run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A changed or deleted test module imported by more than 8 test files runs the light tier with at most 8 of its importers (direct before transitive, then by path) instead of the full run
- [x] #2 A changed or deleted non-test Python file under `Tests/` samples its importing test files the same way, except `Tests/private_profile.py` and `Tests/real_profile_guard.py`, which still take the full run
- [x] #3 More than 20 selected test files are cut to 20, keeping the PR's own changed test files first, instead of taking the full run
- [x] #4 The same diff and tree give the same targets whatever order the file system lists the tree in
- [x] #5 An importer or data reader the light job cannot run is left out of the sample; the PR's own changed test files of those kinds still take the full run
- [x] #6 The plan's reason says what it sampled
- [x] #7 ADR-366 records the amendment, with the cost accepted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In `scripts/ci/tier_policy.py`, sort the Tests corpus and make `_importers` breadth first with depths.
2. Replace the shared-infrastructure rule with an explicit profile-harness set.
3. Sample importers per changed module and cap the total, skipping unrunnable importers.
4. Replace the four tests that pinned the broad behaviour; check each new test fails on dev's policy.
5. Append the ADR-366 amendment and update `scripts/ci/README.md`.
6. Replay #3769 and recent dev merges offline.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed:**
- **`scripts/ci/tier_policy.py`:**
  - The Tests corpus is read in sorted order.
  - `_importers` searches breadth first and returns each file's depth, still stopping past `IMPORTER_CAP`.
  - A changed or deleted `.py` under `Tests/` samples its importers. The sample takes at most `IMPORTER_CAP` (8) runnable files, by (depth, path).
  - The total keeps the PR's own changed test files first, by path, then the samples in change order. It is cut at `LIGHT_FILE_CAP` (20).
  - Importers and data readers that `_full_reason` or the lane checks flag (helpers, HEAVY, deselected, split, push-only) are skipped. The PR's own changed files of those kinds still go broad.
  - The lane checks moved into the `lane_nodes`/`lane_reason` closures, so both paths use them.
  - The old "Shared test infrastructure" rule is now `PROFILE_HARNESS` (`Profile isolation harness: <path>`).
  - The reason lists each sample, for example `sampled 8 of 9+ importers of <path>`.
- **`Tests/CI/test_tiered_ci.py`:** four tests that pinned the broad behaviour are replaced, and `hooks_v2_owned_rig.py` leaves the broad list. New tests cover:
  - the cap with direct before transitive importers, and a deleted module;
  - determinism over six shuffled `rglob` orders;
  - own changed files first under the total cap;
  - unrunnable importers skipped (the same files changed by the PR still go broad);
  - a helper sampling its importers;
  - the two harness files and `conftest.py` still broad.
- **ADR-366:** new amendment section. **`scripts/ci/README.md`:** the harness and sampling sentences.

**Evidence (2026-11-26, branch `ci/42626-tier-sample` from dev 52351c4e0b):**
- All seven new test cases fail on dev's `tier_policy.py`. The determinism test also fails when only the corpus `sorted(` is removed.
- `Tests/CI/test_tiered_ci.py`: 96 passed. `Tests/CI -n 4`: 2013 passed, 2 skipped.
- `./scripts/preflight.sh` under a scratch HOME passes with `PYTHON=.venv/bin/python` (3.12.11). Left to pick its own interpreter, it chose python3.14. The Mermaid check then refused it with "use the pinned Python build version", because it pins 3.12.11.
- `ruff check tldw_chatbook --select F` (the CI gate) and `Tests/lint/ruff_gate.py` pass. Widened to `scripts Tests/CI`, ruff reports 4 findings, all in files dev already had in that state.
- PR #3769 (base 434eb71f95, head 7a453d4d36, root = a detached worktree of the head) stays **broad** under the new policy: `Schema code: tldw_chatbook/Chat/console_dispatch_repository.py`. It also changes `tldw_chatbook/DB/assistant_generation_state.py`, a schema input. `SCHEMA_TEXT` matches `console_dispatch_repository.py` only because a Python parameter is named `user_version`; the file has no DDL. With its four `tldw_chatbook/` files dropped, the rest runs light with 24 targets: `sampled 8 of 9+ importers` of each of its three changed test modules. Dev's policy sends that subset broad.
- Over the last 60 dev merges, replayed against one tree (the #3769 head, so approximate), 25 run light under dev's policy and 35 under this one. The remaining 25 broad runs: 21 CI, build or test configuration; 2 own HEAVY files; 1 own file split across lane steps; 1 own deselected file.

**Trade-offs:**
- The importer count in the reason is a lower bound, because `_importers` still stops past the cap.
- A sample drawn from those first 9 can be smaller than 8 when some are helpers.
- A helper a `conftest.py` imports (for example `Tests/network_guard.py`) samples like any other helper, so the suites under that conftest run in full only at nightly and release.
- The `user_version` false positive in `SCHEMA_TEXT` is left as it is, outside this task.
<!-- SECTION:NOTES:END -->
