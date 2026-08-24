---
id: TASK-2510
title: Source type options offer values the local service rejects
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-05'
updated_date: '2026-08-24 23:30'
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
- [ ] #1 Local New source offers only RSS, Atom, and Web page; Server New source offers only RSS, Site, and Forum; the existing source-filter vocabulary remains unchanged.
- [ ] #2 An open create form updates immediately when the backend changes and preserves name, URL, Active state, tags, destination, cadence, and noise drafts while normalizing only an incompatible type to RSS.
- [ ] #3 Local submissions retain cadence and noise fields; Server submissions omit both and successfully match the real Server create signature.
- [ ] #4 Unsupported form types are rejected before dispatch with exact, backend-specific, markup-safe recovery copy; unrelated failures remain generic.
- [ ] #5 Creation routing, destination filing, and confirmation stay bound to the backend shown at submission even if the selector changes before the worker executes; post-completion refreshes target the visible backend.
- [ ] #6 Focused Watchlists tests cover contracts, draft preservation, payload routing, validation, recovery copy, backend capture, focus order, and supported-width geometry.
<!-- AC:END -->

## Implementation Plan

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

ADR required: no  
ADR path: N/A  
Reason: This is a bounded correction inside the existing Watchlists
local/server routing boundary; it changes no schema, service ownership, API
contract, dependency, or long-lived application structure.
