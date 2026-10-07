---
id: TASK-33665
title: Stop test runs from writing the real profile
status: Done
assignee: []
created_date: '2026-10-02 19:30'
labels:
  - tests
  - isolation
  - safety
dependencies: []
references:
  - Tests/conftest.py
  - Tests/UI/conftest.py
  - Tests/UI/app_factory.py
  - Tests/app_module_patches.py
  - tldw_chatbook/config.py
  - tldw_chatbook/profile_paths.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: at 15:56–15:57 PDT on 2026-10-02, a pytest run wrote to the owner's REAL profile. The run came from an agent's scratch tree, which had been corrupted into a mix of two commits.

What changed:
- `~/.config/tldw_cli/ui_state.toml` was rewritten to the empty default sidebar state.
- `~/.local/share/tldw_cli/default_user/tldw_chatbook_library_collections.db` was opened and closed. A clean close removes its `-wal`/`-shm` files, and the link count of `default_user` fell from 72 to 68. On APFS the link count counts every entry, so 4 entries went: at least the 2 WAL files, plus 2 unidentified.

Unchanged: config.toml, every other database, and the instance lock.

The writer was traced, read-only, to an app built with `_build_test_app()` with the Console mounted, running while the real HOME was in effect and TLDW_CONFIG_PATH was unset. The Console's teardown flush wrote the sidebar snapshot, and app construction opened the one database the factory does not redirect.

The isolation is environment redirection alone, and it has these gaps:
- **No guard.** No production code refuses to write under the real HOME when running under pytest.
- **Library collections path.** `_build_test_app` does not redirect `get_library_collections_db_path`, which calls config's own unpatched `get_user_data_dir`.
- **Session end.** `pytest_sessionfinish` restores the real HOME and unsets TLDW_CONFIG_PATH while late threads, `to_thread` writes or an unmounting app may still run.
- **Frozen config path.** `DEFAULT_CONFIG_PATH` is frozen at the first import of config.py. A run that imports `tldw_chatbook` before the root conftest has set HOME (for example one rooted at Tests/UI, whose conftest sets only TLDW_CONFIG_PATH) freezes the real path.
- **XDG is ignored.** XDG_* variables have no effect on app paths, so the isolation can look stronger than it is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Under pytest, any attempt in a test process (pytest itself, xdist workers, private-profile children) to write a file under the real user's `~/.config/tldw_cli` or `~/.local/share/tldw_cli` fails loudly and names the path, the writer and its thread; a test proves the guard trips. Python subprocesses that are not pytest and C-level writers are documented gaps.
- [x] #2 An app built by `_build_test_app()` resolves the library-collections database, and every other per-profile path, inside the test sandbox; a test asserts it.
- [x] #3 Nothing the suite starts can write under the real HOME after the session ends (the environment is no longer restored to the real HOME at `pytest_sessionfinish`); the late-thread or unmount case is covered by a test.
- [x] #4 A run rooted at Tests/UI, or one that imports `tldw_chatbook` early, cannot freeze the real config path; a test demonstrates it.
- [x] #5 The lessons doc records the 2026-10-02 incident and states the guard as the protection, not environment variables alone.
<!-- AC:END -->

## Implementation Notes

**The guard.** `Tests/real_profile_guard.py` is a `sys.addaudithook` guard, installed first by both `Tests/conftest.py` and `Tests/UI/conftest.py`. It refuses the following whenever the path, its realpath or a case variant falls under the real user's `~/.config/tldw_cli` or `~/.local/share/tldw_cli`:
- write-mode `open`;
- `os.remove`, `rmdir`, `mkdir`, `rename`/`replace`, `truncate`, `link`, `symlink`, `chmod`, `utime`, `chown`, `chflags` and the xattr calls;
- `shutil.rmtree`, `move` and `copyfile`;
- `sqlite3.connect`, including `file:` URIs (parsed with urllib).

Further rules:
- Deleting or moving a directory that contains a protected root is refused too.
- Paths given relative to a directory file descriptor are resolved: `F_GETPATH` on macOS, `/proc/self/fd` on Linux.
- The real home comes from `pwd`, and is exported once as `TLDW_TEST_REAL_HOME` so xdist workers and private-profile children inherit it rather than their sandbox HOME.
- There is no off switch.

**How a refusal surfaces:**
- It raises `RealProfileWriteError` at the call site and is recorded with the stack and thread name.
- The autouse fixture `refuse_real_profile_writes` fails the test even when the writer swallowed the exception.
- `session_end_check` fails the run. An xdist worker sends its refusals through `config.workeroutput` to the controller (`pytest_testnodedown`).
- An atexit handler prints anything refused later.

**Session end.** `pytest_sessionfinish` (root and UI) no longer restores the real HOME, XDG_* or TLDW_CONFIG_PATH. Late threads and atexit writers now land in the dead sandbox. This is the real fix for dir-fd opens, which the `open` audit event cannot place: `raw_participants`' pinned IO and `Utils/private_paths.atomic_private_write_*` both write that way, and so did the incident's own `ui_state.toml` writer. Nothing depended on the restore.

**Other changes:**
- `Tests/UI/conftest.py` moves HOME before any tldw import, so `DEFAULT_CONFIG_PATH` cannot freeze to the real path.
- The test app factory now redirects `get_library_collections_db_path` and `get_tts_profiles_db_path`. Its test asserts that every DB path the app holds is under the factory's own data dir.

**Rulings:**
- No sitecustomize to guard non-pytest Python subprocesses. Many tests spawn Python, and a sitecustomize risks the Packaging import-closure tests. The gap is documented, along with C-level writers and SQLite `ATTACH`/`VACUUM INTO`.
- AC#1 and AC#3 were reworded to state exactly that.

**Evidence:**
- `Tests/test_real_profile_guard.py`: 44 tests plus 2 skipped (xattr is absent on macOS). They run against a temporary stand-in profile, so a broken guard writes into tmp_path, never the real home. They include the incident's own dir-fd writer and an xdist session-end refusal.
- Each fix was RED first.
- Negative controls fail the right tests: dir-fd join off (10 fail), workeroutput off, atexit off, hook off (10 of 15 at the first round), `_is_write_open` broken, and the factory patch removed.
- A review agent found the dir-fd, xdist, pwd-fallback and TTS-path gaps; all are fixed.
- Full suite (`Tests`, `-n 8`, `--junitxml`) on the committed guard: about 113k recorded cases, including the guard's own 44 passing, and **zero guard refusals**. So no existing test writes the real profile.
  - That run's final stretch was disturbed when the worktree was rebased under it, so it was interrupted at 99% and its failure count is not meaningful.
  - `Tests/Architecture/test_rail_bool_coercion_is_shared.py` was excluded because its `object()` parametrize ids break xdist collection; that bug predates this change.
- `./scripts/preflight.sh` passes, and ruff reports no new findings against dev.
