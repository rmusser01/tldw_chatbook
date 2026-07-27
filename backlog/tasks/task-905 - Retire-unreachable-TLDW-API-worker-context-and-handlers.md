---
id: TASK-905
title: Retire unreachable TLDW API worker context and handlers
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 23:25'
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
- [x] #1 No `_last_tldw_api_request_context` application attribute, writer, reader, dynamic access, or compatibility property remains.
- [x] #2 The unproducible `api_calls` worker group, its completion routing, payload-bearing media-ingest handlers, and their compatibility exports are removed rather than recreated behind a new envelope.
- [x] #3 The normal production `TldwCli` still resolves the `ingest` alias to Library and exercises the live Library ingest owner without importing or querying the retired handler pipeline.
- [x] #4 The live Library local/server ingest request mapping and public cancellation path remain covered; no retired selector, handler, or worker-group reference remains in production.
- [x] #5 Focused Library, normal production `TldwCli`, privacy/static, formatting, compile, and authorized integration checks pass.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the orphaned pre-Library worker graph after reconciling the plan against current dev. The verified census found no `api_calls` producer, `tldw_api_events.py`, `MediaIngestScreen`, or mounted `#tldw-api-*` widget; Library remains the sole production ingest owner. Deleted the payload-bearing completion module, root request-context field, worker-registry branch/method/import, and compatibility exports while preserving Notes exports plus `ollama_api` and `model_download`. Corrected stale production comments without recreating an envelope or compatibility owner.

Added precise production AST/source guards for direct, descriptor, symbol, selector, module, quoted-group, `getattr`/`setattr`/`delattr`/`hasattr`, subscript, `vars(...).get`, and `__dict__.get` access. The normal production `TldwCli` test routes the legacy `ingest` alias to exact `LibraryScreen`, exercises real form state across fresh navigation, seeds a real server-origin app-registry job, clicks the rendered Cancel action, and verifies the public seam receives the exact batch id and the real registry reconciles to cancelled. No test/simplified App, unbound method, or fabricated worker event is used. Existing direct tests retain local-file and URL server-request mapping coverage.

TDD evidence: the first static RED failed on the root class field and dead handler dynamic `getattr` while the production Library behavior already passed; review-driven RED then proved `hasattr`, `vars(...).get`, and `__dict__.get` were not recognized before the guard repair.

Verification: 73 focused Library/production-app/ownership tests passed in 251.19s; the repaired guard subset passed 4 tests; compileall passed; Ruff lint passed; the exact eight-file Ruff format gate reports all files formatted; `git diff --check` passed; the retired production-reference sweep is empty. Independent specification re-review is compliant and independent code-quality review is clean, including three consecutive cancellation-flow runs.

ADR required: yes. Existing ADR-029 (payload-safe private-data boundary) and ADR-033 (narrow application-session ownership) govern this deletion; no new ADR was needed.

Plan deviations/review corrections: latest-dev audit added comment-only cleanup for two obsolete production references; spec review required two pre-existing Ruff line wraps and broader dynamic-access sentinels. The branch was repeatedly rebased as dev advanced; current upstream additions through `ca19c5142` are non-overlapping Watchlists QA documentation.
<!-- SECTION:NOTES:END -->
