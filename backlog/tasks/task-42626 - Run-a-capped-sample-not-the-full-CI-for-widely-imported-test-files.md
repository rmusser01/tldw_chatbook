---
id: TASK-42626
title: Run a capped sample, not the full CI, for widely imported test files
status: Done
assignee:
  - '@claude'
created_date: '2026-11-26 15:30'
updated_date: '2026-11-26 21:30'
labels:
  - ci
dependencies:
  - TASK-42521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner wants dev PRs to merge quickly after sanity checks, with the nightly and release runs as the broad gates (ADR-366). The selection still sent a PR to the full run, about 63 jobs including 42 UI shards, whenever it touched a test module imported by more than 8 test files, a shared helper under `Tests/`, or more than 20 test files of its own.

PR #3769 is the case that cost the most. Its preflight said `Shared test module (9+ importers): Tests/Chat/test_console_dispatch_recovery.py`, and its full run took 3 h 26 min, because jobs waited up to about 2 h 50 min for runners on the account pool shared with tldw_server. The one-at-a-time merge queue held every armed PR behind it for 4.5 h. A light-tier PR (#3518) went from rebase to merge in 21 minutes.

These cases should run the light tier with a capped, deterministic sample of the affected test files. Inputs that can break CI itself, the schema, and anything a `conftest.py` loads keep the full run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A changed or deleted test module imported by more than 8 test files runs the light tier with at most 8 of its importers (direct before transitive, then by path) instead of the full run
- [x] #2 A changed or deleted non-test Python file under `Tests/` samples its importing test files the same way
- [x] #3 More than 20 selected test files are cut to 20, keeping the PR's own changed test files first, instead of taking the full run
- [x] #4 The same diff and tree give the same targets whatever order the file system lists the tree in
- [x] #5 An importer or data reader the light job cannot run is left out of the sample; the PR's own changed test files of those kinds still take the full run
- [x] #6 The plan's reason says what it sampled
- [x] #7 ADR-366 records the amendment, with the cost accepted
- [x] #8 A changed or deleted path that any `conftest.py` loads, directly or through other `Tests` modules and including imports inside functions, takes the full run; no hand-kept list decides it
- [x] #9 Application code versions its schema only through `PRAGMA user_version`; a Python name `user_version` no longer sends a module broad
- [x] #10 `scripts/ci/README.md` describes the policy as it now runs
- [x] #11 Importers are resolved by module path, so a same-named module in another package or a mention in prose is not an importer
- [x] #12 Only test files the light job can run count toward a sample, and a change whose dependents the light job cannot run at all takes the full run
- [x] #13 The 20 slots go to the PR's own changed test files first, then to each sample in turn, and the reason names what was sampled, skipped or cut, including the PR's own files it left out
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In `scripts/ci/tier_policy.py`, sort the Tests corpus and make `_importers` breadth first with depths.
2. Replace the shared-infrastructure rule with an explicit profile-harness set.
3. Sample importers per changed module and cap the total, skipping unrunnable importers.
4. Replace the four tests that pinned the broad behaviour; check each new test fails on dev's policy.
5. Append the ADR-366 amendment and update `scripts/ci/README.md`.
6. Replay #3769 and recent dev merges offline.
7. Fix round 1 (coordinator review): derive the conftest-loaded set instead of the profile-harness list; narrow `user_version` to the PRAGMA form; bring the README up to date; re-run the replays and verification.
8. Fix round 2 (independent review I1, I2, M1-M4, N1-N3): resolve imports with `ast` for both the importer search and the conftest set; count only runnable files and go broad when none can run; share the cap in turn and write the notes after the cut; kill the reviewer's mutations; measure the light runner.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed (`scripts/ci/tier_policy.py`):**
- **Sampling.**
  - A changed or deleted test module or helper under `Tests/` samples the runnable test files that import it. A data file under `Tests/` samples the runnable files that read it, or that import a helper that reads it.
  - Each sample holds at most `IMPORTER_CAP` (8). The selection holds at most `LIGHT_FILE_CAP` (20): the PR's own changed test files first, by path, then each sample in turn, one file per round.
  - Unrunnable dependents are skipped and noted: helpers, HEAVY, deselected, split and push-only files. `Tests/CI` runs whole, so it is not counted.
  - The PR's own changed files of those unrunnable kinds still go broad.
- **Import resolution (fix round 2, review I1).**
  - `_imports(name, text)` resolves a file's imports with `ast`, as pytest's prepend import mode loads them: `import Tests.a.mod`, `from Tests.a import mod`, `from Tests.a.mod import X`, relative imports, and bare imports of a module in the importer's own directory. Nested imports count. So do the imports of a child script a test embeds as a string: a template that does not parse whole has its `import`/`from` lines parsed one by one. Results are cached by (name, text).
  - Every package `__init__.py` on the way counts, including the importer's own packages. Paths no file has are kept, so a deletion counts.
  - A file that does not parse counts as an importer.
  - `_importers` uses `stem in text` only as a prefilter; for `__init__.py` the prefilter is the package name, or a file inside the package.
  - The old name-only `_import_pattern` is gone.
- **The search (review I2, M3).** `_importers` runs breadth first in sorted corpus order. It follows helpers and unrunnable files without counting them, and stops at the ninth runnable test file. The depth sort is gone, because BFS order already puts direct importers first.
- **Something must run (review I2).** A change with dependents but none the light job can run (and none in `Tests/CI`) goes broad: `No test file the light job can run depends on: <path>`.
- **The conftest-loaded set (fix rounds 1 and 2).** `_conftest_loaded` follows every `Tests/**/conftest.py`'s imports transitively through the same resolver. A changed or deleted member goes broad with `Loaded by conftest.py: <path>`, and so does a data file a member reads. `PROFILE_HARNESS` is gone.
- **Schema code (fix round 1).** `SCHEMA_TEXT` matches `PRAGMA user_version`, not any `user_version`. All 54 occurrences in `tldw_chatbook/` (reads, writes and docstrings) are written that way. On dev the narrowing drops exactly one file, `Chat/console_dispatch_repository.py`, which has no DDL.
- **The reason (review M1, M2, N3).** Notes are written after the 20-file cut. Each capped, cut or skipped module or data file gets `sampled <ran> of <found>[+] importers|readers of <path>[, skipping N the light job cannot run]`. A cut adds `cut to 20 test files[, not running the PR's own <files>]`.

**Other files:**
- **`Tests/CI/test_tiered_ci.py`:**
  - Earlier rounds: the cap and determinism, own files first, unrunnable importers skipped, a helper sampling its importers, the conftest-loaded set (synthetic and real), the PRAGMA narrowing, and `conftest.py` still broad.
  - Round 2: two same-named modules plus prose and every import form; child scripts embedded as strings (plain, a template, `;`-joined fragments); unrunnable and helper importers taking no slot; a `Tests/CI` importer not counted; breadth first; a data file read by a helper, a heavy file and a normal file, plus heavy-only and helper-only data going broad; the cap shared in turn with notes after the cut; readers capped per file.
- **ADR-366:**
  - The Status line says "amended by TASK-42521 and TASK-42626".
  - The amendment names the three 2026-11-23 bullets it replaces and says what is counted.
  - It covers the conftest-loaded rule, the PRAGMA narrowing, the something-must-run rule and the reason format.
- **`scripts/ci/README.md`:** active-policy header, event table, the selection section as the code runs it, and a note on the rollout section.

**Evidence (2026-11-26, branch `ci/42626-tier-sample` from dev 52351c4e0b):**
- **Fail on old:**
  - The 7 first-round test cases fail on dev's `tier_policy.py`.
  - The 9 fix-round-1 cases fail on f8b476384b.
  - The 7 new round-2 tests, and the 2 whose reason wording changed, fail on 5c83750feb.
  - The child-script test also fails on the round-2 resolver before string scanning.
  - The determinism test also fails when only the corpus `sorted(` is removed.
- **Mutations (all 9 killed on the final policy):** a (no runnable filter), b (cap before the runnable filter), b2 (count every file), c (no `Tests/CI` exclusion), d (DFS), f (name-only matching), g (light with nothing runnable), h (notes counting the sample rather than what ran), and i (no child-script imports). For e, the depth sort was deleted.
- **Precision and recall,** checked on the final resolver against the reviewer's ground-truth regex (extended with the `from Tests.pkg import mod` form), over all 6,135 Tests files:
  - No file fails to parse.
  - **Recall:** 0 import lines the regex confirms are missed. An earlier cut missed child scripts, for example the UI tests that import `Tests/windows_private_fixture_runner.py` inside a script string. That led to the string scanning.
  - **Precision:** 15 of the 7,863 resolver edges are not confirmed by the line-anchored regex. Spot checks show they are real child-script imports the regex cannot see (quoted `"from Tests.x import y\n"` fragments, `;`-joined statements).
- **The reviewer's probes P1-P11 now give these results:**
  - broad for a helper only a conftest imports, for a conftest-imported test module, and for heavy-only data;
  - 8 runnable importers when 9 helpers sort first;
  - 6, 6 and 5 for three shared modules, with notes after the cut;
  - readers capped at 8;
  - the dropped own files named.
- **Real-tree probes:**
  - `Tests/UI/test_console_dispatch_recovery.py` runs its 2 real UI importers.
  - `Tests/UI/response_rules_fixtures.py` runs its 3 real importers' lane nodes.
  - `thread_diagnostics.py` reports `sampled 8 of 9+ ..., skipping 1`.
  - `required_doubles.py` and `Tests/Agents/__init__.py` are conftest-loaded, so broad.
- **All 235 helpers:** 203 run light and 32 are broad, all as conftest-loaded. None goes broad for having no runnable dependents.
- **Verification:**
  - `Tests/CI/test_tiered_ci.py`: 111 passed.
  - `Tests/CI -n 4`: 2028 passed, 2 skipped.
  - `./scripts/preflight.sh` under a scratch HOME with `PYTHON=.venv/bin/python` (3.12.11): all derived-artifact checks passed.
  - These all pass: `ruff check tldw_chatbook --select F`; `ruff check scripts/ci Tests/CI/test_tiered_ci.py --select F`; `ruff format --check` on both files; `Tests/lint/ruff_gate.py`.
- **PR #3769** (base 434eb71f95, head 7a453d4d36, root = a detached worktree of the head) still runs **broad** under the final policy, for one reason only: `Schema input: tldw_chatbook/DB/assistant_generation_state.py`. With that `DB/` file dropped, it runs light: 26 targets, 24 files. The reason is `cut to 20 test files; sampled 8 of 9+ importers of Tests/Chat/test_console_dispatch_recovery.py; sampled 7 of 9+ importers of Tests/UI/test_console_prompt_queue.py`. The UI `test_console_dispatch_recovery.py` now runs its two real importers rather than eight same-named Chat ones. Dev's policy sends that subset broad.
- **Last 60 first-parent dev merges** (through 5dcba48401), replayed against one tree (the #3769 head, so approximate): 26 run light under dev's policy and 35 under the final one. The remaining 25 broad runs: 21 change CI, build or test configuration, 2 their own HEAVY files, 1 its own file split across lane steps, and 1 its own deselected file. None changed class between round 1 and round 2.
- **Light runner timing.** `scripts/ci/run_light.py` ran as the dev-light job runs it: `CI_TEST_TARGETS=<json> python scripts/ci/run_light.py`, Python 3.12.11, a scratch HOME/XDG/TMPDIR, on this 18-core Mac. The selection was merge 1b9e3f8a55's real capped plan: 63 targets, the 20-file cap plus 2 subsystem smoke files and the smoke set, 26 pytest processes with `Tests/CI`.
  - It took **37 min 19 s wall**, over the job's 30-minute cap.
  - `Tests/CI` alone took 10 min 21 s, run serially in one process. The 25 selection processes took 25 min 4 s.
  - My mutation, recall and probe runs overlapped part of it, so this is an upper bound, but the margin is not there.
  - Four files failed in that run: `test_console_rewind_restore.py` (41 `RecoveryRequired: raw_source_selection_changed`), `test_console_custom_capture_worker_feedback.py` (3 event-wait timeouts), and 2 each in `test_console_workflow_controls.py` and `test_console_workflow_navigation.py`. Each file fails again when run alone through the isolated pytest wrapper with a fresh profile (42, 3, 2 and 1 failures). So they fail on this macOS checkout of dev's code; this branch changes only the CI policy, its tests and docs.

**Trade-offs:**
- The `+` in `of 9+` means the search stopped at the cap; the true count is unknown.
- A file inside a package counts as loading its `__init__.py`, as pytest imports it, so an `__init__.py` change samples the package's own tests.
- Bare imports resolve only to a module beside the importer, as the review asked. A rootless directory that relies on another `sys.path` entry is not followed.
- A full 20-file selection measured 37 min locally against the job's 30-minute cap, and `Tests/CI` alone took 10 min serially. The cap (or how `Tests/CI` runs in the light job) needs revisiting. This task does not change it.
<!-- SECTION:NOTES:END -->
