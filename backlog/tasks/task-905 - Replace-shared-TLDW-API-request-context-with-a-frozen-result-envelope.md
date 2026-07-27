---
id: TASK-905
title: Replace shared TLDW API request context with a frozen result envelope
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:16'
labels:
  - architecture
  - state
  - ingest
  - reliability
  - privacy
dependencies:
  - TASK-647
references:
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eliminate cross-request context contamination by returning each completed TLDW API ingestion with its own detached immutable context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The worker returns a frozen result envelope whose response and every detached context field are repr-hidden; context retains no mutable request model.
- [ ] #2 Success handling validates the envelope before UI lookup, valid ingestion continues after the originating screen is gone, and no shared _last_tldw_api_request_context attribute remains.
- [ ] #3 Interleaved completions preserve their exact request context while failure and public worker cancellation cannot reuse stale context.
- [ ] #4 Envelope representations and exercised log, notification, status, or exception payloads disclose none of the response, keyword, author, custom-prompt, or input-reference sentinels.
- [ ] #5 Focused direct-function, normal production TldwCli where needed, privacy, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
