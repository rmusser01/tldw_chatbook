---
id: TASK-42626
title: Run a capped sample, not the full CI, for widely imported test files
status: Done
assignee:
  - '@claude'
created_date: '2026-11-26 15:30'
updated_date: '2026-11-27 00:30'
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
- [x] #12 Over the cap, a sample holds only dependents a PR lane runs. The full run is kept only for a laned dependent the light job cannot run when nothing laned would run in its place. With no laned dependent, the plan stays light and says so.
- [x] #13 The PR's own changed test files and the under-the-cap dependents share the 20 slots, and sampled files take at most 8 in all, handed out in turn. The reason names what was sampled, skipped or cut, including the PR's own files it left out.
- [x] #14 The light runner's wall time on the worst realistic selection under the final caps is measured against the job's 30-minute cap
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
9. Fix round 3 (coordinator): sample only dependents a PR lane runs, keep the full run only where it reaches what light cannot, and cap sampled files at 8 per PR; re-measure the light runner on the worst selection.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed (`scripts/ci/tier_policy.py`):**
- **Dependents.** A changed or deleted test module or helper under `Tests/` depends on the test files that import it, directly or through other modules. A data file under `Tests/` depends on the files that read it, or that import a helper that reads it.
- **Two modes (round 3).**
  - **Under the cap:** with at most `IMPORTER_CAP` (8) runnable dependents, all of them run, laned or not, as dev's rule did.
  - **Over the cap:** the change is sampled. Only dependents a PR lane runs are eligible (whole, by node, through a lane directory, or in the UI census), up to 8, which is a subset of what the full run would run. Unlaned dependents are skipped and noted as `skipping N no PR lane runs`. The nightly runs the same lanes, so they run only at release.
  - A PR samples at most 8 files in all, handed out in turn. Its own changed test files, then the under-the-cap dependents, take the 20 slots (`LIGHT_FILE_CAP`) first. The total never exceeds 20.
- **The full run only where it reaches what light cannot (round 3).**
  - A change goes broad when a dependent that a PR lane (or a push or nightly job) runs is one the light job cannot run (heavy, deselected, split, push-only), and nothing laned would run in its place: `Only the full run reaches the laned dependents of <path>: ...`.
  - When no PR lane runs any dependent, the plan stays light and says so: `no PR lane runs a dependent of <path>; only the release run covers its N test files`.
- **Import resolution (round 2).**
  - `_imports(name, text)` resolves a file's imports with `ast`, as pytest's prepend import mode loads them: `Tests.*` absolute imports, `from pkg import mod`, relative imports, bare imports of a module beside the importer, and nested imports.
  - Package `__init__.py` files count: the importer's own and every one on the way.
  - It also reads child scripts a test embeds as a string; a template that does not parse whole has its `import`/`from` lines parsed one by one.
  - Paths no file has are kept, so a deletion counts. A file that does not parse counts as an importer.
  - `stem in text` is only a prefilter.
- **The search.** `_importers` runs breadth first in sorted corpus order and follows every dependent. It stops once it has IMPORTER_CAP + 1 files that count: runnable files a PR lane runs. The depth sort is gone, because BFS order already puts direct importers first.
- **The conftest-loaded set.** `_conftest_loaded` follows every `Tests/**/conftest.py`'s imports through the same resolver. A changed or deleted member goes broad (`Loaded by conftest.py: <path>`), and so does a data file a member reads. `PROFILE_HARNESS` is gone.
- **Schema code.** `SCHEMA_TEXT` matches `PRAGMA user_version`, not any `user_version`. All 54 occurrences in `tldw_chatbook/` (reads, writes and docstrings) are written that way. On dev the narrowing drops exactly one file, `Chat/console_dispatch_repository.py`, which has no DDL.
- **The reason.** Notes are written after the cut. Each sampled, cut or skipped module or data file gets `sampled <ran> of <found>[+] importers|readers of <path>[ a PR lane runs][, skipping N no PR lane runs][, skipping N the light job cannot run]`. When a cap applied, the reason also says `8 sampled files in all`, or `cut to 20 test files[, not running the PR's own <files>]`.

**Other files:**
- **`Tests/CI/test_tiered_ci.py`.**
  - Earlier rounds:
    - the cap and determinism, and own files first;
    - unrunnable importers skipped, and a helper sampling its importers;
    - the conftest-loaded set (synthetic and real), the PRAGMA narrowing, and `conftest.py` still broad;
    - same-named modules, prose and every import form, and child scripts;
    - unrunnable and helper importers taking no slot, a `Tests/CI` importer not counted, and breadth first;
    - data readers and reading helpers, and the cap shared in turn with notes after the cut.
  - Round 3 gives the sampling-mode trees a PR lane (`_laned`), and adds:
    - a sample holding only laned dependents, with the search going past 20 unlaned ones;
    - broad only when the full run reaches what light cannot, in both modes;
    - light with the "no PR lane" note when nothing laned depends;
    - the 8-file sample budget, with an under-the-cap module outside it;
    - a `Tests/CI` importer, with `Tests/CI` laned, not counted toward the nine.
- **ADR-366.**
  - The Status line says "amended by TASK-42521 and TASK-42626".
  - The amendment names the three 2026-11-23 bullets it replaces and says what is counted.
  - It covers the two modes and the budget (with the 1,131 / 4,707 lane counts and the 37 min measurement as the reason), when the full run is kept, the conftest rule, the PRAGMA narrowing and the reason format.
- **`scripts/ci/README.md`:** active-policy header, event table, the selection section as the code runs it, and a note on the rollout section.

**Evidence (2026-11-26, branch `ci/42626-tier-sample` from dev 52351c4e0b):**
- **Fail on old:**
  - The 7 first-round test cases fail on dev's `tier_policy.py`.
  - The 9 round-1 cases fail on f8b476384b.
  - 7 round-2 tests, and the 2 whose reason wording changed, fail on 5c83750feb.
  - The 3 new round-3 tests, and the 6 updated for lanes or reason wording, fail on 280dc7b24f.
  - The determinism test also fails when only the corpus `sorted(` is removed.
- **Mutations: all 13 killed on the final policy.**
  - a: no runnable filter.
  - b: cap before the runnable filter.
  - b2: the search counts every file.
  - c: no `Tests/CI` exclusion.
  - d: DFS.
  - f: name-only matching.
  - g: never broad for laned dependents the light job cannot run.
  - h: notes before the cut.
  - i: no child-script imports.
  - j: sample unlaned files too.
  - k: no 8-file sample budget.
  - l: the search stops on runnable rather than laned files.
  - m: laned-only under the cap too.
  - For e, the depth sort was deleted.
- **Precision and recall,** checked on the round-2 resolver against the reviewer's ground-truth regex (extended with the `from Tests.pkg import mod` form) over all 6,135 Tests files:
  - Every file parses.
  - **Recall:** 0 import lines the regex confirms are missed.
  - **Precision:** 15 of the 7,863 resolver edges are not confirmed by the line-anchored regex. Spot checks show they are real child-script imports: quoted `"from Tests.x import y\n"` fragments and `;`-joined statements.
- **All 235 helpers on the final policy:**
  - 123 run all their dependents;
  - 60 are sampled;
  - 16 stay light with the "no PR lane" note;
  - 4 go broad because only the full run reaches their laned dependents (`Tests/Backup_Recovery` diagnostics helpers);
  - 32 go broad as conftest-loaded.

  Deciding takes 0.34 s median and at most 10.7 s per helper (147 s for all 235), including searches that exhaust a closure.
- **Verification:**
  - `Tests/CI/test_tiered_ci.py`: 114 passed.
  - `Tests/CI -n 4`: 2031 passed, 2 skipped. The first run had one timing failure, in `Tests/CI/test_bootstrap_root_cleanup.py::test_owned_root_is_removed[fork-child-terminated]` (the child not yet reaped). That test passed 3 of 3 alone and in the re-run.
  - `./scripts/preflight.sh` under a scratch HOME with `PYTHON=.venv/bin/python` (3.12.11): all derived-artifact checks passed.
  - These all pass: `ruff check tldw_chatbook --select F`; `ruff check scripts/ci Tests/CI/test_tiered_ci.py --select F`; `ruff format --check`; `Tests/lint/ruff_gate.py`.
- **PR #3769** (base 434eb71f95, head 7a453d4d36, root = a detached worktree of the head) still runs **broad**, for one reason: `Schema input: tldw_chatbook/DB/assistant_generation_state.py`. With that file dropped, it runs light: 22 targets, 18 files. The reason is `8 sampled files in all; sampled 4 of 9+ importers of Tests/Chat/test_console_dispatch_recovery.py a PR lane runs, skipping 32+ no PR lane runs; sampled 4 of 9+ importers of Tests/UI/test_console_prompt_queue.py a PR lane runs, skipping 18+ no PR lane runs, skipping 1 the light job cannot run`. The UI `test_console_dispatch_recovery.py` runs both of its importers, under the cap.
- **Last 60 first-parent dev merges** (through 5dcba48401), replayed against one tree (the #3769 head, so approximate): 26 run light under dev's policy and 35 under the final one. The remaining 25 broad runs: 21 change CI, build or test configuration, 2 their own HEAVY files, 1 its own file split across lane steps, and 1 its own deselected file. None changed class between rounds 1, 2 and 3; only the samples and reasons changed.
- **Light runner, re-measured (round 3).** `scripts/ci/run_light.py` ran as the dev-light job runs it: `CI_TEST_TARGETS=<json> python scripts/ci/run_light.py`, Python 3.12.11, a scratch HOME/XDG/TMPDIR, on this 18-core Mac. The selection was the largest light plan of the 60 merges under the final caps: merge 1b9e3f8a55, 68 targets, 25 files, 26 pytest processes with `Tests/CI`.
  - None of my jobs ran alongside it. Another session's pytest runs were on the machine (load average 6.1 before, 11.5 after).
  - It took **20 min 14 s wall** (1,214 s), inside the job's 30-minute cap.
  - `Tests/CI` took 3 min 1 s, 15 % of the total. The 25 selection processes took 16 min 17 s, and process start-up the remaining ~56 s.
  - Round 2's 37 min 19 s for the same merge had my mutation, recall and probe jobs running alongside (`Tests/CI` alone then took 10 min 21 s), so that figure was inflated.

**Observation (no fix here).** These four unlaned UI files fail on macOS on dev's code (Python 3.12.11, isolated profile), each run alone:
- `Tests/UI/test_console_rewind_restore.py`: 42 failed, 41 of them `RecoveryRequired: raw_source_selection_changed`;
- `Tests/UI/test_console_custom_capture_worker_feedback.py`: 3 failed, event-wait timeouts;
- `Tests/UI/test_console_workflow_controls.py`: 2 failed;
- `Tests/UI/test_console_workflow_navigation.py`: 1-2 failed.

No PR lane runs them, which is the round-3 reason samples avoid unlaned files.

**Trade-offs:**
- The `+` in `of 9+` means the search stopped at the cap; the true count is unknown, and so is the matching unlaned count.
- A file inside a package counts as loading its `__init__.py`, as pytest imports it, so an `__init__.py` change has the package's own tests as dependents.
- Bare imports resolve only to a module beside the importer, as the review asked.
- An over-the-cap change whose laned dependents are all past the 8-file budget runs none of them (`sampled 0 of ...`). The nightly and release runs cover them.
<!-- SECTION:NOTES:END -->
