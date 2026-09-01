---
id: TASK-26013
title: MCP server-spawn configuration guard
status: Done
assignee: []
created_date: '2026-08-31 15:44'
updated_date: '2026-09-01 23:19'
labels:
  - security
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Any command can be saved as an MCP server and spawned. Verified on origin/dev: MCP/local_store.py sanitizes environment values and rejects secret-shaped literals (:42-50), but a named grep for authorized_keys, IOC and suspicious across tldw_chatbook/MCP returns zero - nothing inspects the command itself. A pasted server config is a code-execution primitive, and imported configs (MCP/mcp_import.py:51 reads Claude Desktop JSON) are exactly the untrusted-input path. Hermes blocks shell-interpreter egress and persistence shapes plus known indicators at both save time and spawn time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A server configuration whose command matches a known dangerous shape is refused at save time with the reason stated
- [x] #2 The same check runs at spawn time, so a configuration edited on disk cannot bypass the save-time check
- [x] #3 Imported configurations pass through the identical check as hand-entered ones
- [x] #4 Shapes covered include at minimum: piping a remote fetch into an interpreter, writing to shell startup files or authorized_keys, and inline interpreter invocation of encoded payloads
- [x] #5 A refusal names the matched rule and does not silently drop the server from the list
- [x] #6 Ordinary server configurations (npx, uvx, python -m, a plain binary path) are unaffected, asserted by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure screen_spawn_command(command, args) in MCP/spawn_guard.py covering the AC#4 shapes\n2. TDD dangerous-shape + ordinary-config cases\n3. Wire at save (local_store.save_profile), spawn (client.connect_to_server), import (mcp_import.parse) - one function, three sites\n4. Integration tests per chokepoint
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One pure guard applied at all three chokepoints.

Approach:
- screen_spawn_command(command, args) in MCP/spawn_guard.py returns a SpawnGuardVerdict(rule, reason) for the first matching dangerous shape, else None (raise_on_match variant for callers preferring an exception). Rules (AC#4): remote-fetch-piped-to-interpreter (curl/wget/... | sh/bash/python), shell-startup-or-authorized_keys-write (>>/>/tee/dd of= targeting .bashrc/.zshrc/.profile/authorized_keys/.ssh/cron/LaunchAgents/...), inline-interpreter-encoded-payload (sh/python/node/perl/... -c/-e/-enc with base64/eval/exec/atob/Buffer.from, plus PowerShell -enc always). Scans the full command line since a shape can span command+args; MCP spawns are create_subprocess_exec (no shell) so the real vector is an interpreter -c payload, but scanning is defensive.
- Wired identically at: save (local_store.save_profile raises ValueError naming the rule, list untouched - AC#1/#5), spawn (client.connect_to_server refuses with a logged rule and returns False before any subprocess - AC#2), import (mcp_import.parse_mcp_servers_json raises ValueError - AC#3). On-disk edits cannot bypass because the spawn-time check is independent of save.
- AC#6: npx/uvx/python -m/plain binary/docker run/node dist/index.js all pass (python -m is not -c; no fetch|pipe, sensitive path, or encoded marker). Asserted by tests; the full MCP suite (464) stays green.

Tests: Tests/MCP/test_spawn_guard.py (22: dangerous shapes, ordinary configs, raise variant, and save/spawn/import integration).

Files: tldw_chatbook/MCP/spawn_guard.py (new), tldw_chatbook/MCP/local_store.py, tldw_chatbook/MCP/client.py, tldw_chatbook/MCP/mcp_import.py, Tests/MCP/test_spawn_guard.py.
<!-- SECTION:NOTES:END -->
