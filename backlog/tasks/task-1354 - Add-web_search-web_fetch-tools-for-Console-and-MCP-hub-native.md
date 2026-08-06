---
id: TASK-1354
title: Add web_search + web_fetch tools for Console and MCP (hub-native)
status: To Do
assignee: []
created_date: '2026-08-05 05:49'
labels:
  - web-tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Claude-Code-style WebSearch/WebFetch for the Console agent runtime and the FastMCP server: one implementation registered as builtin tools in the MCP hub, inheriting On/Off/Ask permissions (search=On, fetch=Ask with per-domain session approvals). web_fetch is lightweight-first (httpx + trafilatura) with one-time Playwright escalation, behind a new general egress guard (redirect re-validation, metadata endpoints always blocked, private/loopback policy block|ask|allow).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 web_search+web_fetch callable in Console with approvals,Both tools exposed via MCP server,Egress guard blocks SSRF incl. redirect hops,Config template gains [tools]+[webfetch],Tests green incl. hub/permission/egress units
<!-- AC:END -->
