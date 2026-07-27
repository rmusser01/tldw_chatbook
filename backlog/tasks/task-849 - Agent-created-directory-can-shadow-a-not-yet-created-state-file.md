---
id: TASK-849
title: Agent-created directory can shadow a not-yet-created state file
status: Done
assignee: []
created_date: '2026-07-27 02:36'
updated_date: '2026-07-27 05:07'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The denylist's direct-child rule is gated on whether a path is an existing directory, so an agent can create a directory named after a state file the app has not created yet (verified: search_history.db/ is permitted). The app's later attempt to open that path as a database would fail. Denial of service only -- no disclosure and no gate bypass. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An agent cannot create a directory whose name collides with a known app state file,Existing container subdirectories under the user data dir stay reachable,A regression test covers the collision
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact defect: create get_user_data_dir()/search_history.db as a directory and confirm is_sensitive_path returns False; then confirm the full attack chain is reachable via WriteFileTool.execute(create_directories=True) under a widened sandbox root, and that the app's own sqlite3.connect on that path then fails (denial of service).
2. Add refuses_new_directory_chain(), which walks upward from a not-yet-created target directory, consulting is_sensitive_path on each not-yet-existing ancestor (reusing the existing gate, never loosening it) and stopping at the first EXISTING ancestor -- so a legitimate pre-existing container is never blocked.
3. Wire it into WriteFileTool.execute's create_directories=True path, before Path.mkdir(parents=True).
4. Add regression tests: the collision itself (unit-level walk + tool-level end to end through WriteFileTool), plus two "still works" tests (a brand-new subdirectory nested inside an existing container, and the real default sandbox configuration end to end).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced the full chain first: created get_user_data_dir()/search_history.db as a directory and confirmed is_sensitive_path() returned False; then, under a widened sandbox root (== get_user_data_dir()), called the real WriteFileTool.execute(file_path="search_history.db/note.txt", create_directories=True) and confirmed it silently created search_history.db/ as a directory (the tool never validated the intermediate directory level, only the final leaf file); then confirmed sqlite3.connect() on that path fails outright (denial of service).

Root cause: WriteFileTool's is_sensitive_path(path) check only ever validated the FINAL file being written, never the new directory levels Path.mkdir(parents=True) creates on the way there.

Fix: added refuses_new_directory_chain(target_dir, context=None) to Utils/sensitive_paths.py. It walks upward from target_dir while each level does not yet exist, calling the EXISTING is_sensitive_path on each such level (never a separate check, never a loosened gate) and stopping at the first already-existing ancestor -- mirroring exactly what Path.mkdir(parents=True) would actually create. Wired into WriteFileTool.execute immediately before its own mkdir call.

Deliberately did NOT change is_sensitive_path's own "is an existing directory" gate: an already-existing search_history.db/ directory (predating this fix, or created by any other means) still reads as an ordinary reachable container, since there is no name-independent way to distinguish a legitimate pre-existing container from an illegitimate one, and the whole point of that gate is to avoid a name enumeration. The fix closes the hole at CREATION time instead, which is also the only place the agent tools can actually cause the collision.

Verified both directions end to end through the real WriteFileTool: the collision (search_history.db/note.txt) is now refused before any directory is created; a brand-new subdirectory nested inside an EXISTING container (tool_sandbox/brand_new_subdir/note.txt) still succeeds, under both the default and a widened sandbox root; the real unmocked default sandbox configuration still works end to end (pre-existing test, unmodified, still passes).

Verified: `pytest Tests/Utils/ Tests/Tools/ Tests/Agents/ -q` -> 893 passed, 0 failed.

Files: tldw_chatbook/Utils/sensitive_paths.py, tldw_chatbook/Tools/file_operation_tools.py, Tests/Utils/test_sensitive_paths.py, Tests/Tools/test_file_tool_sandbox.py.
<!-- SECTION:NOTES:END -->
