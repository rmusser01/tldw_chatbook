---
id: TASK-42145
title: "tldw_test_config_* bootstrap profile directories accumulate in TMPDIR"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 20:49
labels:
- follow-up
- tests
- disk
priority: low
references:
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the skillsrec stream (round 18), which counted about 430 of them. `Tests/conftest.py` creates the bootstrap profile root with `tempfile.mkdtemp(prefix="tldw_test_config_")` (line 56) when it inherits none, and removes it in `pytest_sessionfinish` (lines 1324-1325), only in the owning process and never in an xdist worker.

On 2026-11-20 at 19:58 UTC `$TMPDIR` held 437 of these directories, about 200 MB in all, the oldest from 2026-11-08. Measured at d85e90bc9e through the test wrapper: a passing serial run, a failing run, a `-n 2` run, `--collect-only` and a private-profile test each left none; `pytest --version` left one, because it loads `Tests/conftest.py` (`testpaths = ["Tests"]`) and never reaches `pytest_sessionfinish`. Runs killed or interrupted mid-session are the other likely source; that was not measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The sources of left-behind directories are named, including measurements for `pytest --version` or `--help` and for a run killed mid-session
- [x] #2 Each source that can be closed leaves no directory, shown by a before-and-after count
- [x] #3 Any source left open is named in the notes with the reason
<!-- AC:END -->

## Implementation Plan

1. Measure each run mode with its own `TMPDIR` (so other sessions' directories never count): `--version`, `--help`, pass, fail, `-n 2`, `--collect-only`, SIGINT, SIGTERM and SIGKILL mid-session, serial and xdist.
2. Close what an in-process hook can close: an `atexit` removal of the owned root, registered right after `mkdtemp` (so it runs after every later atexit writer), in `Tests/conftest.py` (and the same in `Tests/UI/conftest.py`, which has its own `mkdtemp`); a SIGTERM handler only if it is safe.
3. Re-measure the same modes; name the sources left open with the reason. Existing directories in `$TMPDIR` are not touched.

## Implementation Notes

- **How measured.** Each run mode ran through the test wrapper with its own fresh `TMPDIR` under the scratchpad (a probe script plus a temporary one-test file, both gone now), so other sessions' directories never counted. Then I counted the `tldw_test_config_*` directories left behind. For kills, the test wrote its pid (and its parent's, the xdist controller) and slept 90 s; the probe sent the signal 2 s later. A background job in a non-interactive shell starts with SIGINT ignored, so the launcher reset SIGINT to default, as a terminal's Ctrl-C would see it. At base (dc371f0842):

  | mode | left | after the fix |
  |---|---|---|
  | `pytest --version` | 1 (9 files: config.toml, the admission registry, lock) | 0 |
  | `pytest --help` | 1 | 0 |
  | `--confcutdir=Tests/UI <a UI file> --version` (Tests/UI/conftest.py's own `mkdtemp`) | 1 | 0 |
  | pass, fail, `-n 2`, `--collect-only`, a `--timeout=3` firing (signal method) | 0 | 0 |
  | SIGINT mid-test, serial and to the `-n 2` controller | 0 | 0 |
  | SIGTERM mid-test, serial and to the `-n 2` controller | 1 | 0 |
  | SIGHUP mid-test, serial | 1 | 0 |
  | SIGKILL mid-test, serial and to the `-n 2` controller | 1 | 1 |

- **Fix.** New `remove_sandbox_at_exit(root)` in `Tests/real_profile_guard.py`, which both conftests import first already. Each conftest calls it right after it `mkdtemp`s a root it owns: `Tests/conftest.py` with 3 lines inside the existing owner branch, never in an xdist worker; `Tests/UI/conftest.py` with 1 line.
  - It began as a new module, `Tests/bootstrap_root_cleanup.py`. Tests/CI then failed `test_perf_guard_runs_when_a_file_its_steps_load_changes`, because every file `Tests/conftest.py` loads must be in `perf-guard.yml`'s path triggers, and the brief allows no workflow edit. `real_profile_guard.py` is already listed.
  - It registers `atexit` removal. That is registered before the app imports, so it runs after their exit hooks (`storage_admission._shutdown` rewrites the admission registry).
  - SIGTERM and SIGHUP get a handler, installed only where the disposition is still `SIG_DFL`. It removes the root, restores `SIG_DFL` and re-sends the signal. A SIGTERM'd serial run still exits with rc 143 at once (measured).
  - `pytest_sessionfinish`'s removal is unchanged.
- **Test.** New `Tests/CI/test_bootstrap_root_cleanup.py` (fast lane via its `Tests/CI` target). A plain-Python child calls `remove_sandbox_at_exit` on a directory, then exits normally or SIGTERMs itself. The test requires rc 0 or -15 and the directory gone. Red with a temporary no-op body: both failed on `assert not root.exists()`, before and after the move. Green: 2 passed.
- **Sources left open (AC#3).**
  - SIGKILL, and `os._exit` (pytest-timeout's `thread` method, the default only on Windows), cannot be caught in-process.
  - A handler someone else installs first, or a daemon thread that writes after the exit hooks, can still leave or recreate a root.
  - I did not touch the 443 directories already in `$TMPDIR`, which other sessions share. Their shapes:
    - 204 bare `config data home` and 18 with files: serial-run roots;
    - 101 with `gw0..gw3` and 41 with other worker counts or files: xdist controllers that never reached their session end, which fits a killed `-n 4` run;
    - 76 with only `home` (3 more: `config home` twice, one empty). That is a root removed at session end and then recreated through `HOME`; 56 of them hold the 7 recovery-bootstrap admission files. No run I tried reproduced it (bootstrap UI and Skills tests serially and at `-n 2`, at base). An in-process writer at exit is now covered by the `atexit` removal; an orphaned child process that outlives pytest is not.
- **Covering runs.** `Tests/test_real_profile_guard.py`, `Tests/UI/test_console_fixture_ownership.py`, `Tests/App/test_app_shutdown.py`, `Tests/test_bootstrap_profile_config_reset.py`, `Tests/test_private_profile_coverage.py` and the two new CI files at -n 4: 133 passed, 2 skipped (load 37). The guard tests run `Tests/UI/conftest.py` through `runpy`. Since review rev1 they record the `remove_sandbox_at_exit` call instead of running it, so they install no exit removal or handler on the worker.
- **Review rev1 (fixes).**
  - *Important, fork safety.* A fork child inherited the `atexit` entry and the handlers, so terminating it removed the parent's live root. Chunking's `ebook_chapters` strategy does exactly that: a fork context, then `p.terminate()`, and fork is the Linux default. The helper now records `owner = os.getpid()` and removes the root only in that process, from both the `atexit` callable and the handler. Every process still re-raises the signal.
    - New `fork-child-terminated` case: the child terminates a fork child (exit -15) and the root must still exist, then must be gone once the child exits.
    - Red on the earlier code: `'child=-15 kept=False' == 'child=-15 kept=True'`. Green after the fix. The reviewer's `fork_probe.py` now reports the parent's root kept.
  - *Minor.* New `sigterm-ignored` case pins the branch that leaves an existing disposition alone. The child sets SIGTERM to `SIG_IGN`, then calls the helper and SIGTERMs itself; it must exit 0 with the root removed by `atexit`. Red with the `is SIG_DFL` check removed: `assert -15 == 0`.
  - *Minor.* `_run_ui_bootstrap` in `Tests/test_real_profile_guard.py` now replaces `guard.remove_sandbox_at_exit` with a recorder, so the guard tests no longer install handlers on xdist workers. The owning cases assert it got the fresh root; the inherited-root cases assert no call. Red with the UI conftest's call removed: `assert [] == [PosixPath('...')]` in both owning cases.
  - Runs. Changed files plus `Tests/App/test_app_shutdown.py`: serially, 100 passed, 2 skipped. At -n 4 with the earlier covering set: 132 passed, 2 skipped, 1 failed. The failure is a `test_console_fixture_ownership.py::test_fixture_retires_captured_databases_not_replaced_attributes` case (`instance_lock is not None`).
    - It fails at base dc371f0842 with the same files (2 of 3 runs), so it was luck that my earlier -n 4 runs passed.
    - Paired at -n 4, it fails only beside `Tests/App/test_app_shutdown.py` (2 of 2), never beside the other three files, and passes alone (27 passed, twice).
  - Tests/CI at -n 4 after these fixes: 1817 passed, 2 skipped (load 28). Preflight and the ruff gate pass.
- **Review rev2 (fixes).** The root conftest's call, the source of the leftover directories, is now pinned. `test_pytest_version_leaves_no_bootstrap_root` runs `python -m pytest --version -p no:cacheprovider` from the repo root, with `TMPDIR=tmp_path` and without `TLDW_TEST_CONFIG_ROOT`, `TLDW_TEST_CONFIG_ROOT_OWNER` or `PYTEST_XDIST_WORKER`. It requires rc 0 and no `tldw_test_config_*` in `tmp_path`, and takes 1.4 s. Red with the conftest call removed: `assert [PosixPath('.../tldw_test_config_4b2tr_gi')] == []`, one directory left. Green: the file has 5 passed. The two stale notes above (the `runpy` sentence, the Files list) are corrected.
- Tests/CI at -n 4 after the move: 1815 passed, 2 skipped (load 48). After the move the run modes measured the same as the table: 0 left for every mode except SIGKILL.
- `backlog/docs/lessons-testing-evidence.md`: the `pytest --version` lesson no longer says the root stays in `$TMPDIR`.
- Files: `Tests/real_profile_guard.py`, `Tests/conftest.py`, `Tests/UI/conftest.py`, `Tests/CI/test_bootstrap_root_cleanup.py`, `Tests/test_real_profile_guard.py`, `backlog/docs/lessons-testing-evidence.md`.
