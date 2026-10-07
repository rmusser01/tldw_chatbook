---
id: TASK-33279
title: 'PERF-20: HTTP client reuse - pooled sessions, cached SSLContext, nothing built
  on the loop'
status: To Do
created_date: 2026-09-28 18:03
labels:
- performance
- network
- llm
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every hosted LLM call builds and discards a requests.Session: 15.2 ms against 0.9 ms reused on loopback, plus a real TCP/TLS handshake per send and per agent step. Every httpx client rebuilds an SSLContext from certifi (12-24 ms), and about 15 sites do so on the event loop. That includes watchlist feed checks and Settings/Console/wizard probes. Research 'Ask Follow-up' runs a blocking chat_api_call on the loop. webbrowser.open runs on the loop (at least 73 ms on macOS). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-20; every issue with file:line is listed under PERF-20 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Hosted provider calls reuse a pooled session per provider/base URL
- [ ] #2 httpx clients share a cached SSLContext
- [ ] #3 No HTTP client is constructed on the event loop
- [ ] #4 Research Ask Follow-up and URL opening run off the event loop
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
