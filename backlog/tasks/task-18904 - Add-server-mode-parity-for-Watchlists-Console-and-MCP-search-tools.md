---
id: TASK-18904
title: Add server-mode parity for Watchlists Console and MCP search tools
status: To Do
assignee: []
created_date: '2026-08-14 23:16'
updated_date: '2026-08-18 00:00'
labels:
  - watchlists
  - agent-tools
  - mcp
dependencies: []
references:
  - rmusser01/tldw_server TASK-13022
  - TASK-16222
  - Docs/superpowers/specs/2026-08-14-watchlists-agent-search-tools-design.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the current server-mode unsupported outcome for watchlists_search_items and watchlists_get_item with exact user-visible parity backed by tldw_server TASK-13022, while retaining the existing local behavior, permissions, privacy shaping, and byte bounds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Server mode exposes the same two tool names, input schemas, status outcomes, evidence fields, ordering, and continuation semantics as local mode.
- [ ] #2 Chatbook consumes tldw_server TASK-13022 through a transport safe for the synchronous agent worker and never blocks Textual's event loop.
- [ ] #3 Server watchlists map to collection scope; human source and watchlist resolution retains exact-name precedence, bounded disambiguation, and explicit canonical IDs.
- [ ] #4 Server statuses and dates map explicitly to the local public vocabulary, with no invented timestamps or aggregate last-updated field.
- [ ] #5 Chatbook wraps server continuation with a backend and query fingerprint so local, server, and mismatched cursors fail closed.
- [ ] #6 All server evidence passes through the existing strict less-than-30-KiB JSON packer, field allowlist, URL sanitization, terminal-control stripping, and untrusted-evidence labeling.
- [ ] #7 The existing local:__local__ permission, kill-switch, approval, definition-hash, and external MCP exposure gates remain authoritative with no new principal or mutation capability.
- [ ] #8 Tests cover Console and external MCP invocation, local/server switching, auth and transport failures, malformed cursors, scope ambiguity, parity fixtures, no event-loop blocking, and unchanged local-mode results.
<!-- AC:END -->
