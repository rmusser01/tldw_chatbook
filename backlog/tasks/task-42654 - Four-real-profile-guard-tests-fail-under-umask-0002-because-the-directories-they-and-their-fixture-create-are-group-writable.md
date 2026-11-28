---
id: TASK-42654
title: "Four real-profile guard tests fail under umask 0002 because the directories they and their fixture create are group-writable"
status: To Do
assignee: []
created_date: 2026-11-28 19:37
updated_date: 2026-11-28 19:53
labels:
- follow-up
- tests
priority: low
references:
- Tests/test_real_profile_guard.py
- tldw_chatbook/Utils/private_paths.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 29 (2026-11-28) while checking TASK-42651. Under umask `0002`, four tests in `Tests/test_real_profile_guard.py` fail. That is Ubuntu's default for regular users. The failures reproduce in an `ubuntu:24.04` container run as uid 1001, and on macOS with the umask set for the pytest process. Under umask `022` the file gives 65 passed and 2 skipped. In the container 61 of the other 63 pass and 2 are skipped, on dev 2df861ec7d and with TASK-42651's change alike. The app's refusals are correct; the tests depend on the developer's umask. There are two causes.

1. **The three `test_an_xdist_workers_session_end_refusal_fails_the_run` params** (`both-conftests`, `root-conftest-only`, `ui-rooted`).
   - The test creates `(tmp_path / "sandbox").mkdir()` and passes it to its child session as `TLDW_TEST_CONFIG_ROOT`. Under umask 0002 that directory is 0775.
   - The child refuses to start: "Chatbook cannot start: its private storage location is not secure on this machine". The cause is `shared_writable_parent`, from the ancestor walk in `Utils/private_paths.py`.
2. **`test_the_incidents_own_dir_fd_writer_cannot_replace_the_profile_file`.**
   - It runs in-process and has no sandbox. The shared `fake_profile` fixture creates `home/.config/tldw_cli/sub` with `mkdir(parents=True)`, so the whole chain is 0775.
   - `ChatScreen._save_sidebar_state` then fails with `PrivatePathError` before the guarded replace, the guard records nothing, and the test's `_take_about` assertion sees `[]`.

`Path.mkdir(parents=True, mode=0o700)` applies the mode to the leaf only; parents get the default permissions. So each ancestor has to be created private explicitly. This is the trap in the lesson "Private test parents must be private too" (TASK-32560/TASK-32561). A test-wide umask change would be riskier, because the umask is process-wide and these tests run late writer threads.

**CI status:**
- No PR lane runs the file. `test.yml`'s core shards (`pytest Tests --ignore=Tests/UI`, called by release qualification and publish) collect it.
- No CI log has reached these four nodes yet:
  - in release candidate qualification run 44955437105 their shards stopped at `--maxfail=1` first;
  - its other runs (the 2026-10-03 direct runs and the 2026-11-19 publish runs on `main`) tested commits without the file; it reached dev on 2026-10-03 (#2976; 4f6718c7ba).
- The hosted runners' umask is not verified.
- The three child sessions run `pytest -n 2`, so a lane for this file must install pytest-xdist. No PR job in `derived-artifacts.yml` installs it today, so AC #3 needs a PR job's install line changed (or a new job).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four tests pass under umask `0002` with every assertion kept, because every directory the tests and the `fake_profile` fixture create is created private (each ancestor explicitly), not by changing the process umask
- [ ] #2 A test builds those directories under a forced umask `0002` (in a child process, or with the umask set and restored around the setup only) and checks none is group- or world-writable; it fails on the unfixed tree
- [ ] #3 The file runs in a PR lane that installs pytest-xdist
<!-- AC:END -->
