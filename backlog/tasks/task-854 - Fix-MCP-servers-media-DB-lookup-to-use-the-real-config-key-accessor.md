---
id: TASK-854
title: Fix MCP server's media DB lookup to use the real config key/accessor
status: To Do
assignee: []
created_date: '2026-07-27 04:34'
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
- [ ] #1 MCP/server.py resolves the media DB path via get_media_db_path() (or the equivalent media_db_path config key), matching what the rest of the app opens
- [ ] #2 A grep-based check (or test) confirms no other get_cli_setting("database", ...) call site in the codebase uses a key that isn't one of the declared *_db_path names
- [ ] #3 A test starts/constructs the MCP server's media DB handle and asserts its resolved path equals get_media_db_path()'s return value, rather than asserting a literal filename
- [ ] #4 The resolved media DB path is covered by is_sensitive_path() once the fix lands (verified by test, not by inspection)
<!-- AC:END -->
