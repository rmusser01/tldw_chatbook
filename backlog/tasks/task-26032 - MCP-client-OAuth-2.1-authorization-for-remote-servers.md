---
id: TASK-26032
title: 'MCP client: OAuth 2.1 authorization for remote servers'
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
labels:
  - mcp
  - interop
  - auth
dependencies:
  - TASK-25900
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Most hosted MCP servers require OAuth, which chatbook cannot perform. Verified on origin/dev: a named grep for oauth across tldw_chatbook/MCP returns zero, and MCP/local_store.py:38 stores environment placeholders only. Hermes carries a full PKCE, dynamic-client-registration and token-storage stack with interactive login flows. Depends on task-25900 (HTTP and SSE transports) - authorization is meaningless until there is a remote server to authorize against.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A remote MCP server requiring OAuth can be authorized interactively and subsequently connects without re-prompting
- [ ] #2 The flow uses PKCE and supports dynamic client registration where the server offers it
- [ ] #3 Tokens are stored using the same protection as other stored credentials, never in plaintext config, and never logged
- [ ] #4 Token refresh happens transparently; an unrecoverable refresh failure surfaces as an actionable readiness state rather than a generic connection error
- [ ] #5 The authorization endpoint is subject to the existing SSRF egress policy
- [ ] #6 A user can revoke a server's authorization and the stored tokens are actually removed
- [ ] #7 Servers not requiring authorization are unaffected
<!-- AC:END -->
