---
id: TASK-854
title: Fix MCP server's media DB lookup to use the real config key/accessor
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:34'
updated_date: '2026-07-27 16:28'
labels:
  - security
  - mcp
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP/server.py:154-155 calls get_cli_setting("database", "media_db", "media_library.db"); the actual config key is media_db_path (declared config.py:2233, live in config at line 30, read via get_media_db_path() at config.py:4622). Because the key "media_db" doesn't exist, the lookup always falls through to the relative literal, so the MCP server creates/opens ./media_library.db relative to whatever directory it was launched from -- a live reproduction showed it landing at .../wt-path-hardening/media_library.db (a repo checkout) rather than the app's real .../default_user/tldw_chatbook_media_v2.db. The two lines immediately above this one correctly call get_chachanotes_db_path() for the conversations DB, so this is an isolated miss, not a pattern. Utils/sensitive_paths.py:98 denylists get_media_db_path()'s result -- the DB the MCP server does not actually use -- so everything the MCP media tools read or write through this path has no denylist coverage at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP/server.py resolves the media DB path via get_media_db_path() (or the equivalent media_db_path config key), matching what the rest of the app opens
- [x] #2 A grep-based check (or test) confirms no other get_cli_setting("database", ...) call site in the codebase uses a key that isn't one of the declared *_db_path names
- [x] #3 A test starts/constructs the MCP server's media DB handle and asserts its resolved path equals get_media_db_path()'s return value, rather than asserting a literal filename
- [x] #4 The resolved media DB path is covered by is_sensitive_path() once the fix lands (verified by test, not by inspection)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the wrong-db behaviour with a sandboxed HOME/no-TLDW_CONFIG_PATH probe.
2. Fix MCP/server.py's _init_databases() to call get_media_db_path() instead of the undeclared 'media_db' key.
3. Grep the codebase (AST-based, in a test) for any other get_cli_setting('database', ...) call using a key that is not a declared *_db_path name.
4. Add a regression test that drives the real _init_databases() and asserts the resolved media DB path.
5. Add a test asserting the resolved path is covered by is_sensitive_path().
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed MCP/server.py:154 to call config.get_media_db_path() instead of get_cli_setting("database", "media_db", "media_library.db") (key never existed; real key is media_db_path). Reproduced before/after with a sandboxed HOME probe: before, the buggy key resolved to the CWD-relative literal 'media_library.db' (would land at .../wt-path-accessors/media_library.db); after, it resolves to get_media_db_path()'s real, per-profile path (.../default_user/tldw_chatbook_media_v2.db in the sandbox). While fixing this, discovered _init_databases() also called MediaDatabase(media_db_path) with NO client_id (a required positional arg -- TypeError at construction) and constructed NotesInteropService/CharacterInteropService with signatures that don't match either class (CharacterInteropService does not exist anywhere in the codebase). The missing client_id was fixed (adjacent, required to make AC #3 constructible at all -- CLI_APP_CLIENT_ID, matching every other MediaDatabase call site). The NotesInteropService/CharacterInteropService construction bugs are a SEPARATE, pre-existing defect (the whole _init_databases() method has apparently never been exercised successfully) and were left alone as out of this task's scope -- flagged for a follow-up task rather than fixed here. New test Tests/MCP/test_server_media_db_path.py drives the real _init_databases() (stubbing only the two unrelated broken collaborators) and asserts media_db.db_path == get_media_db_path(), that it is_sensitive_path()-covered, and an AST-based repo-wide grep (also serves AC #2) confirming no other get_cli_setting('database', ...) call site uses an undeclared key -- zero additional offenders found. Files: tldw_chatbook/MCP/server.py, Tests/MCP/test_server_media_db_path.py (new).
<!-- SECTION:NOTES:END -->
