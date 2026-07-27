---
id: TASK-905
title: Retire unreachable TLDW API worker context and handlers
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
Remove the orphaned pre-Library TLDW API worker pipeline and its shared request context now that the production ingest route is owned by Library and no worker producer remains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No `_last_tldw_api_request_context` application attribute, writer, reader, dynamic access, or compatibility property remains.
- [ ] #2 The unproducible `api_calls` worker group, its completion routing, payload-bearing media-ingest handlers, and their compatibility exports are removed rather than recreated behind a new envelope.
- [ ] #3 The normal production `TldwCli` still resolves the `ingest` alias to Library and exercises the live Library ingest owner without importing or querying the retired handler pipeline.
- [ ] #4 The live Library local/server ingest request mapping and public cancellation path remain covered; no retired selector, handler, or worker-group reference remains in production.
- [ ] #5 Focused Library, normal production `TldwCli`, privacy/static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
