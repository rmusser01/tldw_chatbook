---
id: TASK-652
title: Remove duplicate Media root state and stop mutation bubbling
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 19:51'
labels:
  - architecture
  - state
  - media
  - reliability
dependencies:
  - TASK-647
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make MediaWindow the sole Media view and selection owner and ensure one production Media event performs one scoped mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media destination root reactives, duplicate search generation/timers, and all of their writers, watchers, and legacy emitters are removed; legacy Chat sidebar pagination and selection fields remain assigned to TASK-650.
- [x] #2 The Media destination stops mutation events before awaiting work and the duplicate app-level mutation handler registration is removed.
- [x] #3 One real metadata event performs exactly one scoped mutation and refresh, while older item-detail and search completions cannot overwrite newer destination state.
- [x] #4 MediaScreen snapshots and restores only the actual MediaWindow owner.
- [x] #5 Normal production TldwCli Media checks plus focused ownership, static, formatting, compile, and authorized integration checks pass.
- [x] #6 Concurrent metadata edits are last-request-wins in durable storage, including when the initiating MediaWindow is replaced before the writes settle.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Existing ADRs make MediaWindow the production state/action owner.

1. Reproduce one metadata event reaching both the destination and app.
2. Stop handled messages before work and keep durable metadata mutation alive in an app-owned worker without root Media data.
3. Order same-record metadata persistence at the long-lived scope-service boundary and guard stale detail/search/metadata presentation with destination-local generations.
4. Remove duplicate reactive and ordinary app-root Media state plus legacy handlers and tests.
5. Verify the real production TldwCli, snapshots, privacy, ownership, static checks, and integration gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented destination-owned Media state and exact-once event handling.

- Moved MediaRuntimeState construction, browse/detail caches, selection, and search/detail/metadata generations into MediaWindow; removed the listed TldwCli and MediaScreen mirrors plus legacy app handlers and emitters.
- Stopped every handled Media message at the destination before mutation, await, or worker launch. Metadata persistence runs in an app-owned generic worker with an immutable detached request, while presentation remains guarded by the exact mounted owner, selection, browse identity, and local generation.
- Serialized same-record metadata writes in the long-lived media scope service with synchronous request reservation, so slower earlier edits cannot overwrite newer durable values within one MediaWindow or across destination replacement.
- Unified mutation-triggered refreshes with the stale-safe browse worker, retained last-valid-page correction, and allowed valid completions while a real production modal is above the still-mounted Media route.
- Reduced media_events.py to message contracts, removed obsolete root-only search tests, updated retained direct-function contracts, and replaced simplified mounted MediaWindow restore tests with full production TldwCli coverage.
- MediaScreen now saves and restores only live MediaWindow fields; live and missing restored selections are verified on fresh production destinations.

ADR required: yes
ADR paths: backlog/decisions/033-application-session-state-ownership.md and backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: these existing ADRs already assign destination state/actions and stale-safe async presentation; no new decision was introduced.

Verification:
- Production Media plus ownership sentinels: 55 passed, 1 dependency warning, 467.66s.
- Runtime-policy, Chat function, and MediaScreen integration selection: 28 passed, 1 dependency warning, 18.95s.
- Media scope-service functions: 73 passed, 2.55s; MediaRuntimeState functions: 3 passed, 0.13s.
- Ruff format check: 14 files already formatted; Ruff check passed; compileall passed; git diff --check passed.
- Independent review found a modal-owner race, retained-test contract drift, and durable metadata ordering race; all confirmed defects were fixed. A suggested keyword-refresh coupling was rejected after an import/mount census proved the legacy CollectionsTagWindow has no production mount and Media type transitions issue a fresh browse query.

Verified evidence:
- One real metadata message produced one mutation and one refresh and stopped at MediaWindow.
- Real message propagation ended at MediaWindow for navigation, selection, delete/undelete, read-later, highlight, analysis, and collapse messages.
- Older detail, ordinary search, and mutation-triggered refresh completions did not overwrite newer state; metadata survived Media teardown without stale presentation.
- Same-window and replacement-window reverse-order metadata writes left the newer edit in durable SQLite storage.
- Metadata failure diagnostics excluded sentinel private values.

Deviations and adjacent findings:
- Added production regressions for modal ownership, missing restored records, and guarded last-page correction after independent review exposed those cases.
- Two separate pre-existing issues were verified but intentionally left outside this task acceptance criteria: local FTS queries containing a raw hyphen can raise SQLite no-such-column errors, and Media analysis Save as Note references a nonexistent TldwCli.notes_db instead of the current Notes ownership seam.
<!-- SECTION:NOTES:END -->
