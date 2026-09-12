---
id: TASK-2510
title: Source type options offer values the local service rejects
status: Done
assignee:
  - '@codex'
created_date: '2026-08-05'
updated_date: '2026-08-25 15:05'
labels:
  - watchlists
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Watchlists New source form reflect the active backend's actual
creation contract so users cannot choose dead-end source types or lose an
in-progress draft when switching backends. The current shared option list
offers Local values that its service rejects and sends Local-only fields to
the Server create signature, reducing every failure to a generic toast.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local New source offers only RSS, Atom, and Web page; Server New source offers only RSS, Site, and Forum; the existing source-filter vocabulary remains unchanged.
- [x] #2 An open create form updates immediately when the backend changes and preserves name, URL, Active state, tags, destination, cadence, and noise drafts while normalizing only an incompatible type to RSS.
- [x] #3 Local submissions retain cadence and noise fields; Server submissions omit both and successfully match the real Server create signature.
- [x] #4 Unsupported form types are rejected before dispatch with exact, backend-specific, markup-safe recovery copy; unrelated failures remain generic.
- [x] #5 Creation routing, destination filing, and confirmation stay bound to the backend shown at submission even if the selector changes before the worker executes; post-completion refreshes target the visible backend.
- [x] #6 Focused Watchlists tests cover contracts, draft preservation, payload routing, validation, recovery copy, backend capture, focus order, and supported-width geometry.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed plan: `Docs/superpowers/plans/2026-08-24-watchlists-backend-source-types.md`

1. Publish Local and Server create-form source-type contracts and route the
   active tuple through the scope service and UI controller.
2. Split the Sources pane's filter and create vocabularies, preserve the full
   draft, render backend-specific fields, and reject unsupported form values
   before event dispatch.
3. Live-sync an open pane when the backend changes and carry the
   submission-time backend through creation, destination filing, and
   confirmation.
4. Extend full-shell focus/geometry coverage, run only focused Watchlists
   tests and scoped static checks, then record evidence and close the task.
5. Verify Qodo review findings against the mounted-form lifecycle, add a
   regression for immediate Server-to-Local submission, apply bounded
   maintenance fixes, document any rejected architectural suggestions, and
   rerun the focused merge gate.

ADR required: no
ADR path: N/A
Reason: This is a bounded correction inside the existing Watchlists
local/server routing boundary; it changes no schema, service ownership, API
contract, dependency, or long-lived application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Published ordered backend-owned create-form contracts: Local exposes RSS,
  Atom, and Web page while retaining its broader import/programmatic types;
  Server exposes RSS, Site, and Forum. The scope service and controller route
  those machine values without taking ownership of UI labels.
- Split the Sources pane's unchanged filter vocabulary from its backend-aware
  create vocabulary. The mounted pane live-syncs backend changes, keeps the
  form open, preserves the complete draft, hides Local-only controls on
  Server, and rejects stale unsupported values before dispatch with exact
  markup-disabled recovery copy.
- Bound creation, destination filing, and confirmation to the backend captured
  when Create was pressed. Source-list refresh still follows the backend that
  is visible when the request completes.
- Modified source files:
  `tldw_chatbook/Subscriptions/local_watchlists_service.py`,
  `tldw_chatbook/Subscriptions/server_watchlists_service.py`,
  `tldw_chatbook/Subscriptions/watchlist_scope_service.py`,
  `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py`,
  `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`, and
  `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`.
- Modified focused tests:
  `Tests/Subscriptions/test_local_watchlists_service.py`,
  `Tests/Subscriptions/test_server_watchlists_service.py`,
  `Tests/Subscriptions/test_watchlist_scope_service.py`,
  `Tests/Watchlists/test_watchlists_backend_controller.py`,
  `Tests/Watchlists/test_watchlists_sources_pane.py`,
  `Tests/Watchlists/test_watchlists_collections_screen.py`,
  `Tests/UI/full_app_destination_context.py`, and
  `Tests/UI/test_watchlists_source_create_form.py`.
- Geometry/tab-order verification for Local RSS, Local Web page, and Server RSS
  at `(160, 42)` and `(235, 52)`: **12 passed, 22 deselected**. No CSS change
  or bundle regeneration was necessary.
- Final review exposed an incomplete test double in the collections-screen
  integration coverage: blanket `AsyncMock` controllers turned the new
  synchronous `create_form_source_types` seam into a coroutine during backend
  watcher execution. The exact Reader recovery test failed red with
  `TypeError: 'coroutine' object is not iterable`; all four controller doubles
  in that module now share the real mixed sync/async API shape.
- Expanded user-authorized focused verification: the originally failing
  Reader recovery test is **1 passed, 2 warnings**; the complete related
  collections-screen module is **88 passed, 2 warnings**; and the prior
  six-module Task-2510 suite is **155 passed, 2 warnings**. The warnings were
  an environment dependency-version warning and Python's `audioop`
  deprecation; pytest also reported unrelated permission warnings while
  cleaning pre-existing temporary test garbage.
- Qodo follow-up identified a real Server-to-Local submit/recompose race. A
  regression first reproduced `NoMatches` when Create was pressed before the
  Local-only cadence control remounted; submission now falls back to the
  preserved draft cadence. The follow-up also centralizes the one-hour
  default and documents the Active-switch handler.
- Qodo's generic-validator and Pydantic-payload suggestions were deliberately
  not applied: the create type is a controlled Select backed by each service's
  published contract and is validated again by the selected service, while
  Watchlists messages/controllers consistently use dictionary payloads.
  Either suggestion would duplicate backend validation or introduce a new
  cross-module model boundary outside this bounded task and its approved
  design.
- Post-review focused merge gate: **244 passed, 2 warnings** across the three
  services, backend controller, Sources pane, full-shell create form, and
  collections screen. The new exact race regression is also independently
  **1 passed, 1 warning**; the full Sources pane module is **40 passed, 1
  warning**.
- Scoped Ruff over all 14 Task-2510 source/test files, including the corrected
  collections-screen integration tests: **passed**. `git diff --check`:
  **passed**.
- Self-review found every acceptance criterion satisfied: filter options are
  unchanged; Local's broader persistence types remain accepted; Server sends
  no cadence/noise keys; specialized copy is limited to pane pre-dispatch
  validation; the mounted pane survives backend changes; and submitted-backend
  routing, filing, confirmation, and visible-backend refresh are independently
  covered.
- Approved design specification:
  `Docs/superpowers/specs/2026-08-24-watchlists-backend-source-types-design.md`.
- ADR required: no. ADR path: N/A. Reason: bounded correction inside the
  existing Watchlists local/server routing boundary.
- Lessons assessment: the final-review regression demonstrated that blanket
  async controller doubles become incomplete when a controller gains a
  synchronous seam. The incident, evidence, and explicit mixed-interface-double
  rule are recorded in `backlog/docs/lessons-testing-evidence.md`.
- Verification was intentionally scoped to modified Watchlists functionality,
  per explicit user direction. No repository-wide suite, performance,
  licence, or unrelated static checks were run or claimed.
<!-- SECTION:NOTES:END -->
