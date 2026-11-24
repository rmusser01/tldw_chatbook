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

**Review:** an independent local review ran in three rounds.
- **Round 1:** 1 Critical, 4 Important and 5 Minor findings, fixed in the second commit:
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
- **Round 2:** no Critical or Important findings. Its Minors are fixed in the third commit:
  - schema code outside the DB directories (`CREATE TABLE`/`user_version`);
  - test data files now select the test files that read them;
  - parenthesized imports, and importers of importers;
  - the file cap now counts only the PR's own files;
  - files the lanes run only by node now run those nodes instead of the whole file;
  - tests for the importer-is-infrastructure branch;
  - the replay counts and the `ci:full` re-run note.
- **Round 3 (final pass):** no Critical findings. Fixed in the fourth commit:
  - **I1:** node ids containing commas were cut short. Census entries are now read as whole lines, and workflow steps as shell words.
  - **I2:** push and nightly jobs counted as lanes, and `--deselect` was ignored. Lanes are now the jobs the required check needs. Files only other jobs run, files a lane deselects nodes from, and files split across lane steps go broad.
  - **Minors:**
    - uppercase DDL is matched case-sensitively;
    - deleted data files select their readers;
    - the ADR now says light runs only lane-listed nodes.
- **Round 4 (confirmation):** I1 and I2 confirmed closed. One new Important finding: step names (`Run the gated Tests/UI slice`) made `Tests/UI` a "lane directory", so node-only UI files ran whole. Fixed in the fifth commit, which also fixes two latent Minors:
  - only `run:` scripts are read, so names and env no longer count;
  - shlex strips comments, so a quoted `#` is safe;
  - the plain-word fallback drops quotes.
  - A real-repo test pins `test_settings_advanced_disclosures.py` to exactly its census nodes.
- **The label trigger (part of I3):** documented instead of adding `labeled` to the workflow, since any bot label would then start a full 62-job run. The reviewer agreed.

**Verification:**
- `Tests/CI` 1865 passed, 2 skipped (`-n 4`); `test_tiered_ci.py` 92 passed.
- **Replay** over the last 60 dev merges at b31e3e9241:
  - The old rule sends 5 light; the new rule sends 23.
  - Of the 37 still broad: 20 change CI or test configuration (mostly lane additions to the workflow), 10 change a widely imported test module, 5 touch the two whole-file-excluded files, 1 has lane nodes split across steps and 1 changes Architecture data.
  - A light run selects a median of 5 files (max 13), and selection takes a median of 0.5 s (max 2.2 s).
- This PR itself runs broad, because it changes `.github/` and `scripts/ci/`.
<!-- SECTION:NOTES:END -->
