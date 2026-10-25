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
updated_date: 2026-10-25 23:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every hosted LLM call builds and discards a requests.Session: 15.2 ms against 0.9 ms reused on loopback, plus a real TCP/TLS handshake per send and per agent step. Every httpx client rebuilds an SSLContext from certifi (12-24 ms), and about 15 sites do so on the event loop. That includes watchlist feed checks and Settings/Console/wizard probes. Research 'Ask Follow-up' runs a blocking chat_api_call on the loop. webbrowser.open runs on the loop (at least 73 ms on macOS). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-20; every issue with file:line is listed under PERF-20 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).

### Post-PR #3248 book audit: follow-up request admission

Additional bounded evidence for the existing Research Ask Follow-up objective (AC #4), not a second owner: with the actual ResearchWindow, LocalResearchEngine, scope/local service and private SQLite, a held asynchronous answerer lets the independent App tick 17 times in about 0.35 seconds while the window's enabled Clear events action remains queued. Release publishes the actual answer and clears the events. The matched unheld control clears immediately. The hold replaces only the answerer boundary; it does not measure the default network call or natural provider latency.

When addressing the already recorded blocking default chat_api_call, also ensure the originating window handler returns promptly; awaiting a properly asynchronous provider still retains that window's FIFO message queue. Keep current-run result ownership and explicit insufficient-evidence behavior intact. Completed [TASK-16325](./task-16325%20-%20Persist-structured-claims-and-enable-follow-up-QA-over-run-evidence.md)/[TASK-16334](./task-16334%20-%20Expose-budgets-and-follow-up-QA-in-the-Research-window.md) own claims and Q&A delivery, not this pending-request responsiveness. Canonical [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md) and [ADR-068](../decisions/068-local-research-execution-engine.md) apply. Book: Chapter 7 PDF145-146/158 / printed136-137/149. Audited merged dev: `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05`. [Audit and controlled observations](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md).
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
