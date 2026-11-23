---
id: TASK-42521
title: Run sanity checks, not the full CI, on dev PRs that leave CI inputs alone
status: Done
assignee:
  - '@claude'
created_date: '2026-11-23 21:00'
updated_date: '2026-11-23 21:00'
labels:
  - ci
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner intended the tiered CI (ADR-366) to let dev PRs merge quickly after sanity checks. The nightly full run on dev and the full release run into main were meant to be the broad gates. That never happened.

- **The switch was off.** `CI_TIERED_ENABLED` was never set, so every dev PR ran all 62 jobs. With runners starved, PRs such as #3683 waited about two days in the merge queue.
- **The selection rule would have blocked it anyway.** Any test file, doc, backlog doc or "sensitive" path name forced the full run. Replayed over the last 60 dev merges, only 5 would have gone light.

The owner set the variable on 2026-11-23 and chose to loosen the rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A dev PR runs the full CI only when it changes CI, build, test-harness or schema inputs, deletes an application file, or carries `ci:full`; every other dev PR runs the light tier
- [x] #2 The light tier runs preflight, `Tests/CI`, the smoke set, the PR's own changed test files and its subsystem's smoke nodes, each test file in its own pytest process
- [x] #3 Replayed over the last 60 dev merges, most go light
- [x] #4 ADR-366 records the activation and the amended selection rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed:**
- **`scripts/ci/tier_policy.py`:** the old allowlist with a broad fallback is replaced by a short list of broad triggers:
  - CI and build inputs: `.github/`, `scripts/ci/`, `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements*.txt`.
  - DB migrations.
  - Non-`test_*.py` Python files under `Tests/`.
  - Deleted application files.
  - Type changes and unmerged paths.
  - An empty change range.

  Changed `test_*.py` files outside `Tests/CI` become light targets; `Tests/CI` already runs whole. Docs, backlog and other sources add nothing beyond the smoke set and their subsystem's smoke nodes. `HIGH_RISK_NAMES` is gone.
- **`derived-artifacts.yml` `dev-light`:** each selected file now runs in its own pytest process, because bootstrap_profile and sandboxed suites can't share one (TASK-32873). Every failure is reported, and exit 5 (nothing collected) passes. Other changes: a 300 s per-test timeout, a 30-minute job cap (was 15), and `hypothesis` added to the install.
- **`Tests/CI/test_tiered_ci.py`:** the broad and light cases are re-pinned, and three tests are new or rewritten: one for changed test files, one for sensitive names, and one for deletions.
- **ADR-366:** marked Accepted, with an amendment section covering the activation and the new rule.

**Verification:**
- `Tests/CI/test_tiered_ci.py` 57 passed, and `Tests/CI` passes in full.
- Replay over the last 60 dev merges at b31e3e9241: the old rule sends 5 of 60 light, the new rule 37. Of the 23 still broad, 18 change a workflow (mostly lane additions) and 5 change shared test infrastructure. A light run selects up to 17 changed test files, median 6.
- `Tests/CI` 1830 passed, 2 skipped (`-n 4`).
- This PR itself runs broad, because it changes `.github/` and `scripts/ci/`.
<!-- SECTION:NOTES:END -->
