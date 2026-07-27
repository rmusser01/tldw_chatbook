---
id: TASK-950
title: >-
  verify_trusted_directory rejects every macOS temp directory because /var is a
  symlink
status: To Do
assignee: []
created_date: '2026-07-27 17:30'
labels:
  - bug
  - macos
  - security
  - db
priority: critical
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`verify_trusted_directory` (`tldw_chatbook/Utils/private_paths.py:1107`) walks a path component by component with `_open_directory_component` and rejects anything that is not a real directory. On macOS `/var` is a **symlink** to `/private/var`, and the system temp directory lives under it — so the walk hits the symlink and raises `PrivatePathError: link_or_non_regular: NotADirectoryError`.

Every SQLite open under a macOS temp directory therefore fails, via `_connect_registered_sqlite` (`tldw_chatbook/DB/private_sqlite.py:907`).

**Large parts of the suite are red on `origin/dev` right now because of this.** Measured on a clean detached checkout of `2c33cb616` with no other changes:

| Suite | Result on clean dev |
|---|---|
| `Tests/UI/test_destination_visual_parity_correction.py` | **96 failed**, 4 passed |
| `Tests/UI/test_watchlists_destination_shell.py` | **47 failed**, 1 passed |
| `Tests/Watchlists` | **15 failed**, 130 passed |

This is not specific to Watchlists — it is every test that opens SQLite under a temp directory, which on macOS is effectively every test that builds an app.

In `Tests/Watchlists` it is partly **masked** by a second, unrelated breakage: `Tests/UI/test_screen_navigation.py` assigns `app.current_runtime_backend`, which became a read-only property, so those tests die at construction with `AttributeError` before reaching the path guard. Note that "fixing" that harness line in isolation makes things **worse**, not better: it drives previously-passing tests into this guard. Both need fixing together, this one first.

Do not fix this by loosening the guard's symlink rejection wholesale — rejecting symlinked components is the point of it. The likely correct treatment is to resolve the path before walking, or to treat the platform's own temp root as trusted, but that is a judgement call for whoever owns this subsystem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A SQLite database can be opened under the system temp directory on macOS
- [ ] #2 The guard still rejects a symlinked component that is not part of the platform's own temp root
- [ ] #3 `Tests/UI/test_destination_visual_parity_correction.py`, `Tests/UI/test_watchlists_destination_shell.py` and `Tests/Watchlists` all pass on macOS from a clean `dev` checkout
- [ ] #4 A test covers the macOS `/var` → `/private/var` case specifically, and fails against the current code
- [ ] #5 `Tests/UI/test_screen_navigation.py` no longer assigns the read-only `current_runtime_backend` property, and is fixed only alongside #1 so it does not unmask this
- [ ] #6 The security intent of the guard is stated in its docstring, so the next person does not weaken it to make a test pass
<!-- AC:END -->
