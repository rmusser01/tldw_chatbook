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
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`verify_trusted_directory` (`tldw_chatbook/Utils/private_paths.py:1107`) walks a path component by component with `_open_directory_component` and rejects anything that is not a real directory. On macOS `/var` is a **symlink** to `/private/var`, and the system temp directory lives under it — so the walk hits the symlink and raises `PrivatePathError: link_or_non_regular: NotADirectoryError`.

Every SQLite open under a macOS temp directory therefore fails, via `_connect_registered_sqlite` (`tldw_chatbook/DB/private_sqlite.py:907`).

**`Tests/Watchlists` is currently red on `origin/dev` because of this** — 15 failures at `2c33cb616`, reproduced on a clean detached checkout of dev with no other changes. The count is the number of tests that build a database under `tmp_path`, not a property of the Watchlists suite; anything else opening SQLite under a temp directory on macOS will fail the same way.

It is currently **masked** by a second, unrelated breakage: `Tests/UI/test_screen_navigation.py` assigns `app.current_runtime_backend`, which became a read-only property, so tests die at construction with `AttributeError` before reaching the path guard. Fixing that harness line (seed `_runtime_policy_projection_snapshot` instead) reveals the same 15 failures with this error instead.

Do not fix this by loosening the guard's symlink rejection wholesale — rejecting symlinked components is the point of it. The likely correct treatment is to resolve the path before walking, or to treat the platform's own temp root as trusted, but that is a judgement call for whoever owns this subsystem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A SQLite database can be opened under the system temp directory on macOS
- [ ] #2 The guard still rejects a symlinked component that is not part of the platform's own temp root
- [ ] #3 `Tests/Watchlists` passes on macOS from a clean `dev` checkout
- [ ] #4 A test covers the macOS `/var` → `/private/var` case specifically, and fails against the current code
- [ ] #5 `Tests/UI/test_screen_navigation.py` no longer assigns the read-only `current_runtime_backend` property
- [ ] #6 The security intent of the guard is stated in its docstring, so the next person does not weaken it to make a test pass
<!-- AC:END -->
