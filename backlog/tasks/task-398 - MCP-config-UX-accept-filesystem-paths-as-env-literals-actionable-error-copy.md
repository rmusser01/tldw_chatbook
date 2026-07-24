---
id: TASK-398
title: 'MCP config UX: accept filesystem paths as env literals + actionable error copy'
status: To Do
assignee: []
created_date: '2026-07-21 02:16'
labels:
  - mcp
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ATHF third-party UAT (2026-07-21) found real onboarding friction: a user configuring a local MCP server whose required env var is a filesystem path (ATHF needs ATHF_WORKSPACE=/path/to/hunts) cannot type that path into the Add-server form's env-literal field — the strict env_literals whitelist only accepts booleans/ints/decimals/URLs/log-levels, so a path is rejected with 'Literal env key X must use an explicit safe operational literal or an env placeholder'. Workaround (works today): env PLACEHOLDER $VAR + export the value before launching chatbook; also the LEGACY env path already accepts a path via _LEGACY_SAFE_PATH_PATTERN. Two problems: (1) the strict path rejects a legitimate non-secret filesystem path that the legacy path accepts — inconsistent; (2) the error copy doesn't tell the user HOW to fix it (use $NAME + export). Filesystem paths are extremely common in stdio MCP server configs (workspace roots, config files, db paths) — this will bite most third-party server onboarding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A non-secret filesystem path is either accepted as a safe env literal (matching the legacy path pattern) OR the rejection copy names the exact fix (use a $NAME placeholder and export it),The Add-server form surfaces the actionable guidance inline,Decision recorded on whether strict/legacy path acceptance should converge
<!-- AC:END -->
