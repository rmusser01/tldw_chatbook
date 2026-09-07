---
id: TASK-19055
title: Add opt-in app-wide floating Persona Buddy
status: Done
assignee: []
created_date: '2026-08-20 17:47'
updated_date: '2026-08-22 06:26'
labels: []
dependencies:
  - TASK-19053
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users an explicitly enabled, app-wide floating visual companion for one selected local Persona, driven only by trusted application lifecycle state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy is default-off and mounts only after the user explicitly selects an eligible local Persona, persisted profile-locally as `(source = local, local_persona_id)`; Workbench highlight, Console actor, and server-source changes never silently retarget it. If the selected Persona is disabled, soft-deleted, missing, or loses its local binding, the view hides with a stable path-free unavailable reason while the enabled preference remains; restoring or explicitly replacing the Persona re-resolves the view, and no selection mounts nothing.
- [x] #2 An app-owned controller survives screen navigation without retaining screen/widget references and resolves the pinned state priority, all nine built-ins, source-scoped leases, safe custom triggers, and exact Persona/binding/version identity.
- [x] #3 A native Textual 8 floating view is bottom-right by default, draggable, resizable, focusable, collapsible, closable, bounded to the viewport, and never steals focus on state changes; it provides keyboard move, resize, reset, collapse, and close actions without shadowing terminal-convention, reserved, or existing global bindings, and collapses to a labelled compact control when its minimum geometry cannot fit.
- [x] #4 Geometry/enabled/open/collapsed preferences persist profile-locally and are never exported; geometry re-clamps after every viewport change, and splash/auth/recovery/modal surfaces safely hide or cover the Buddy so it cannot intercept input behind them.
- [x] #5 `sprite_frames` animation pauses while hidden/collapsed, respects reduced motion, and falls back through state, idle, and portrait without blanking the UI; frame and availability failures report stable path-free categories.
- [x] #6 Same-owner Buddy work is serialized across replacement screens; DB, resolve, decode, and frame-preparation work runs off the event loop, uncancellable work is shielded and drained before releasing serialization, view targets are weak or identity-fenced, and authority is revalidated after every await. Stale work and replaced views cannot repaint or remove the current view.
- [x] #7 No third-party window dependency, taskbar, snapping desktop, maximize system, model-directed state, or default Persona is introduced.
- [x] #8 Production-shaped Pilot tests cover normal, wide, and 80x24 layouts, compositor output, and zero flow/`fr` budget; isolated real-terminal verification covers mouse drag/resize, keyboard controls, focus, modal hit testing, navigation, viewport resize, and geometry restore. Evidence includes born-RED→GREEN tests at the actual seams, assigned-worktree import provenance, isolated HOME/XDG/config/data roots set before app import, and mutation proof for authority, lease, cancellation, geometry, and modal-input guards; real SQLite repository coverage is required only if persistence storage changes. Impeccable review follows the final visible change; scoped Ruff, format, compile, diff, and static checks plus diagnostic, privacy, architecture, and governance gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`
Reason: ADR-074 already defines the app-owned Buddy controller, local-Persona authority, native Textual overlay, trusted-state, persistence, and async-lifetime boundaries; this task implements that accepted design without introducing another architectural choice.

1. Add born-RED pure contracts for strict profile-local Buddy preferences, explicit local-Persona selection, state priority, source-scoped leases, and authority generations; implement the smallest app-owned controller that satisfies them.
2. Add born-RED async controller tests for active Persona Visual resolution, fallback rendering, reduced motion, bounded frame preparation, serialization, shield-and-drain cancellation, stale-authority refusal, and lifecycle shutdown; integrate only the existing Persona Visual repository/runtime and image-rendering seams off the event loop.
3. Add born-RED Textual tests for the lightweight floating view, native compositor placement, bounded drag/resize, keyboard controls, compact collapse, focus stability, animation pause, modal coverage, navigation replacement, and viewport re-clamping; mount it through the shared app-screen chrome without consuming flow or fractional layout budget.
4. Add born-RED Personas Workbench tests and explicit actions to select, show, hide, and disable the Buddy; persist only local Persona identity and UI preferences, reject server-backed/inactive/deleted/missing Personas, and refresh through existing Persona and Persona Visual lifecycle seams without silent retargeting.
5. Add born-RED trusted-lifecycle adapter tests for Console run, approval, realtime voice, wake, tool, error, explicit, and custom source-scoped states; keep adapters path-free, model-independent, and free of screen/widget retention.
6. Run only touched-component verification in an isolated profile: focused unit/Pilot/architecture/privacy tests, mutation probes, a bounded real-terminal interaction harness, final Impeccable review/detector, scoped Ruff/format/compile/diff/CSS/governance checks, then document evidence and close TASK-19055.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an opt-in, app-owned Persona Buddy from preferences through rendering and trusted Console lifecycle integration. The work added strict profile-local selection and geometry persistence, source-scoped state leases, immutable Persona Visual resolution, bounded frame preparation, a native Textual floating view, Workbench actions, and content-free Console adapters while preserving the separation between operational Buddy states and Shared Visual Identity reactions.

Authority and lifetime handling use exact Persona/profile/binding/cache generations, screen/view identity fences, serialized app-owned preference writes, shield-and-drain cancellation, and terminal disposal. Unavailable Personas hide without losing the enabled selection; restore or matching visual publication re-resolves and remounts. No schema, server write path, model-directed state, default Persona, or third-party window dependency was introduced. ADR-074 remained the governing decision and no new ADR was required.

Verification was born-RED to GREEN with mutation proof across preference authority, lease ownership, async fences, rendering bounds, modal/input safety, navigation replacement, publication rebound, and Console disposal. The exact touched-component gate passed 578 tests; current-base follow-up gates passed 221, 24, and 125 tests; the exact performance guard passed 9 tests; CSS bundles reproduced; the diagnostic inventory verified 521 owners and 7 sink files; and the isolated POSIX terminal probe passed all 18 checks. Qodo's final review reported zero bugs and zero rule violations with all five threads resolved. Modified areas include `Persona_Buddy`, shared app/screen lifecycle wiring, Personas Workbench, Console producers, focused tests, the terminal probe, generated CSS, and the diagnostic inventory.
<!-- SECTION:NOTES:END -->
