---
id: TASK-22861
title: Expose bounded Watchlists receipts and briefing query tools
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:16'
labels:
  - watchlists
  - console
  - mcp
  - briefings
dependencies:
  - TASK-22859
  - TASK-22860
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console agents inspect sources, collections, durable operation receipts, and full briefing content with provenance while external MCP receives only explicitly approved metadata and receipts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared tools list bounded source, collection, briefing-receipt, and operational metadata using canonical IDs, stable order, filter-bound cursors, redacted URLs, and the established 30 KiB result ceiling.
- [ ] #2 `watchlists_get_operation_status` accepts only exact `local:watchlist_run:<id>` or `local:briefing:<id>` receipts and returns their owning entity, timestamps, normalized state, retry/cancel capability, and inspection destination.
- [ ] #3 `watchlists_get_briefing` is Console-only and returns bounded Markdown plus ordered selected/cited durable provenance, generated/untrusted labels, truncation metadata, and explicit legacy/missing-reference markers.
- [ ] #4 External MCP can list bounded metadata/receipts but cannot receive article snippets, article bodies, briefing Markdown, or selected/cited provenance arrays, regardless of stored permission state.
- [ ] #5 “Read latest briefing” deterministically resolves the newest completed receipt for one collection while reporting newer non-readable attempts as operational context.
- [ ] #6 Server Watchlists mode and missing/old read-only storage return structured unsupported/feature-unavailable outcomes without initializing or mutating the local database.
- [ ] #7 Service, provider, external-MCP, read-only-database, redaction, cursor, truncation, and documentation-contract tests cover every new query.
<!-- AC:END -->
