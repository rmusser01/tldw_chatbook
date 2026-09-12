---
id: TASK-31510
title: Defer legacy MCP server-target JSON sync off the pre-paint path
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - mcp
  - boot
dependencies: []
priority: low
---

## Description (the why)

`TldwCli.__init__` calls `_wire_server_context_provider` (`app.py:8442`),
which runs `upsert_legacy_config_target`
(`MCP/server_target_store.py:300-342`): a synchronous JSON read of
`mcp_server_targets.json` and, when the legacy config differs, a synchronous
write -- on the main thread, before first paint. Only users with a legacy
`tldw_api` base_url pay it, and the file is small, but it is blocking I/O on
the pre-paint critical path that first-use or post-paint wiring would serve
equally well. Evidence: `Docs/Design/2026-09-04-holistic-perf-review.md`
section 7.

## Acceptance Criteria (the what)

- [ ] The legacy-target reconciliation no longer performs file I/O before first paint (deferred wiring or first-use), with behavior for consumers unchanged
- [ ] Existing server-target store tests stay green
