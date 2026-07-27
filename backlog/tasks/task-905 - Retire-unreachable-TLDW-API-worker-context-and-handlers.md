---
id: TASK-905
title: Retire unreachable TLDW API worker context and handlers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 22:44'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/033-application-session-state-ownership.md
Reason: Existing ADRs require payload-free diagnostics and forbid application-root request ownership; latest dev has no producer for the retired pipeline.

1. Prove the api_calls producer and MediaIngestScreen are absent and freeze exact executable plus historical-reference sentinels.
2. Remove the orphan field, handlers, routing, exports, and stale production references.
3. Exercise the live Library ingest owner and public cancellation seam in the normal production TldwCli.
4. Verify direct Library request mapping, full production behavior, static absence, formatting, and compilation.
<!-- SECTION:PLAN:END -->
