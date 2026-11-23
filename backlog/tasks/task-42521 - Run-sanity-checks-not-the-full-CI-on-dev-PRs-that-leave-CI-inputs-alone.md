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
- [x] #1 A dev PR runs the full CI only when it changes CI, build, test-configuration, shared-test or schema inputs, deletes an application file, selects more test files than the light job can run, or carries `ci:full`; every other dev PR runs the light tier
- [x] #2 The light tier runs preflight, `Tests/CI`, the smoke set, the PR's changed test files under `Tests/` and the test files importing them, and its subsystem's smoke nodes, each test file whole and in its own pytest process
- [x] #3 Replayed over the last 60 dev merges, the light tier covers far more than the 5 the old rule allowed
- [x] #4 ADR-366 records the activation and the amended selection rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed:**
- **`scripts/ci/tier_policy.py`:** the old allowlist with a broad fallback (and `HIGH_RISK_NAMES`) is replaced by broad triggers. The full list is in ADR-366's amendment:
  - CI, build and pytest-config inputs, and `packages/`.
  - Schema directories (`DB`, `db`, `migrations`).
  - Shared test infrastructure: non-test Python under `Tests/`, Architecture/Performance data, and test modules imported by more than 8 files or deleted while imported.
  - The two files no lane runs whole.
  - More than 20 selected files.
  - Deleted application files.
  - `ci:full`.

  Changed test files and their importers run light. A file selected whole drops its own smoke nodes.
- **`scripts/ci/run_light.py`** (new; `dev-light` calls it): runs one pytest process per file, using the lanes' plugin autoload and `--timeout=180`. It reports every red file and warns when a file collects nothing. The job cap is 30 minutes (was 15), and `hypothesis` is installed as in console-p0-gate.
- **`Tests/CI/test_tiered_ci.py`:** re-pins the broad and light cases and adds tests for:
  - overlap (a whole file vs its own nodes);
  - importers and the importer cap;
  - the file cap;
  - the runner's grouping, refusals and exit codes;
  - the light job's wiring.
- **ADR-366:** Accepted, with an amendment.

**Review:** an independent local review found 1 Critical, 4 Important and 5 Minor issues, all fixed in the second commit:
- **C1:** a changed file was narrowed to its own smoke nodes.
- **I1:** imported `test_*.py` modules were treated as ordinary test files.
- **I2:** schema changes outside `DB/migrations` went light.
- **I3:** huge or deselect-listed files were selected whole.
- **I4:** `packages/` was not covered.
- **M1:** a test file that collected nothing passed silently.
- **M2:** the light job's plugins and timeout differed from the lanes'.
- **M3:** pytest config and guard data at depth went light.
- **M4:** the runner had no tests.
- **M5:** the ADR and task wording overstated the code.

The label-trigger part of I3 is documented instead of adding `labeled` to the workflow, because any bot label would then start a full 62-job run.

**Verification:**
- `Tests/CI` 1851 passed, 2 skipped (`-n 4`); `test_tiered_ci.py` 79 passed.
- Replay over the last 60 dev merges at b31e3e9241: the old rule sends 5 light, the new rule 27.
  - Of the 33 still broad, 18 change a workflow, 7 change a widely imported test module, 5 touch the two whole-file-excluded UI files, 2 change `Tests/conftest.py` and 1 changes Architecture data.
  - A light run selects a median of 6 files, and the slowest selection took 26.5 s at load average about 40.
- This PR itself runs broad, because it changes `.github/` and `scripts/ci/`.
<!-- SECTION:NOTES:END -->
