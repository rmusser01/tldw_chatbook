---
id: TASK-33274
title: 'PERF-15: Agent runtime - persistent worker, pooled client, fewer hydrations'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33268
labels:
- performance
- agents
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each Console agent send builds a new _ModelCallLifeline thread, a new event loop and a new httpx.AsyncClient (33.7 ms SSL plus a cold TLS handshake). Each tool call runs on a new bare thread and leaks the DB handles it opens. Trace writes open and close a fresh connection per operation. MCP tool calls run about 130 ms of admission, governance and audit I/O on the loop. Conversation open runs three full-hydration derivations of the same agent history. probe_initial_catalog is O(N^2). The StreamGate re-scans its buffer per chunk (843 ms per 100 KB reply). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-15; every issue with file:line is listed under PERF-15 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Agent model calls and tool calls run on a persistent worker with a reused event loop and HTTP client
- [ ] #2 Tool calls do not leak DB handles (leak test)
- [ ] #3 Agent trace writes reuse a connection
- [ ] #4 MCP tool calls do no admission/governance I/O on the event loop
- [ ] #5 Conversation open hydrates agent history once, and StreamGate work per chunk is O(chunk)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
