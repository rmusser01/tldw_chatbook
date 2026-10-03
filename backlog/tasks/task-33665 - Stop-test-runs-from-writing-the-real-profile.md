---
id: TASK-33665
title: Stop test runs from writing the real profile
status: To Do
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
- [ ] #1 Under pytest, any attempt to write a file under the real user's `~/.config/tldw_cli` or `~/.local/share/tldw_cli` fails loudly and names the path and the writer; a test proves the guard trips.
- [ ] #2 An app built by `_build_test_app()` resolves the library-collections database, and every other per-profile path, inside the test sandbox; a test asserts it.
- [ ] #3 Nothing the suite starts can write under the real HOME after `pytest_sessionfinish` restores the environment; the late-thread or unmount case is covered by a test.
- [ ] #4 A run rooted at Tests/UI, or one that imports `tldw_chatbook` early, cannot freeze the real config path; a test demonstrates it.
- [ ] #5 The lessons doc records the 2026-10-02 incident and states the guard as the protection, not environment variables alone.
<!-- AC:END -->
