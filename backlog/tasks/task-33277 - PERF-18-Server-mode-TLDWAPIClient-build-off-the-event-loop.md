---
id: TASK-33277
title: 'PERF-18: Server-mode TLDWAPIClient build off the event loop'
status: To Do
created_date: 2026-09-28 18:03
labels:
- performance
- server-mode
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
For server-mode users, the first TLDWAPIClient construction imports a 1,257-model schema surface (575-854 ms) synchronously on the event loop, about 0.1 s after first paint. About 30 from_config services each own a private httpx pool and SSLContext. mcp_server_targets.json is re-read on every build_client(). Binary downloads are fully buffered. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-18; every issue with file:line is listed under PERF-18 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first client build and schema import happen off the event loop
- [ ] #2 Slice schema classes defer pydantic build until first use
- [ ] #3 Server-backed services share one pooled client
- [ ] #4 No loop stall above 50 ms is attributable to client construction on the server-mode boot probe
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
