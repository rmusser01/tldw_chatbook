---
id: TASK-33106
title: Refuse ADR-183 character writes from server-backed Console sessions
status: To Do
created_date: 2026-09-27 17:30
dependencies:
- TASK-32955
labels:
- mcp
- characters
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32955 (PR #2863) added a server-mode refusal to ADR-183's `create_character` and `update_character`, but only for the standalone MCP server. That server has no Console session, so it reads the default profile's runtime source. In-process calls, where a Console agent reaches ADR-183's tools through the MCP bridge, deliberately keep dev's behaviour. The default-profile check could have refused a local session or let a server-backed session write.

So a Console agent in a server-backed session can still write LOCAL character cards through the bridge, while the Console's own `character_save` refuses in that case. The in-process path should refuse using the calling session's own runtime source, not the default profile's.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Console agent in a server-backed session that calls `create_character` or `update_character` through the in-process MCP bridge is refused with the same message the Console character tools use.
- [ ] #2 A Console agent in a local session keeps working as today.
- [ ] #3 The standalone MCP server's behaviour (TASK-32955) is unchanged.
<!-- AC:END -->
