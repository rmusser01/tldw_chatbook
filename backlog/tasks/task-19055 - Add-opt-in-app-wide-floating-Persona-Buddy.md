---
id: TASK-19055
title: Add opt-in app-wide floating Persona Buddy
status: To Do
assignee: []
created_date: '2026-08-20 17:47'
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
- [ ] #1 Buddy is default-off and mounts only after the user explicitly selects an eligible local Persona, persisted profile-locally as `(source = local, local_persona_id)`; Workbench highlight, Console actor, and server-source changes never silently retarget it. If the selected Persona is disabled, soft-deleted, missing, or loses its local binding, the view hides with a stable path-free unavailable reason while the enabled preference remains; restoring or explicitly replacing the Persona re-resolves the view, and no selection mounts nothing.
- [ ] #2 An app-owned controller survives screen navigation without retaining screen/widget references and resolves the pinned state priority, all nine built-ins, source-scoped leases, safe custom triggers, and exact Persona/binding/version identity.
- [ ] #3 A native Textual 8 floating view is bottom-right by default, draggable, resizable, focusable, collapsible, closable, bounded to the viewport, and never steals focus on state changes; it provides keyboard move, resize, reset, collapse, and close actions without shadowing terminal-convention, reserved, or existing global bindings, and collapses to a labelled compact control when its minimum geometry cannot fit.
- [ ] #4 Geometry/enabled/open/collapsed preferences persist profile-locally and are never exported; geometry re-clamps after every viewport change, and splash/auth/recovery/modal surfaces safely hide or cover the Buddy so it cannot intercept input behind them.
- [ ] #5 `sprite_frames` animation pauses while hidden/collapsed, respects reduced motion, and falls back through state, idle, and portrait without blanking the UI; frame and availability failures report stable path-free categories.
- [ ] #6 Same-owner Buddy work is serialized across replacement screens; DB, resolve, decode, and frame-preparation work runs off the event loop, uncancellable work is shielded and drained before releasing serialization, view targets are weak or identity-fenced, and authority is revalidated after every await. Stale work and replaced views cannot repaint or remove the current view.
- [ ] #7 No third-party window dependency, taskbar, snapping desktop, maximize system, model-directed state, or default Persona is introduced.
- [ ] #8 Production-shaped Pilot tests cover normal, wide, and 80x24 layouts, compositor output, and zero flow/`fr` budget; isolated real-terminal verification covers mouse drag/resize, keyboard controls, focus, modal hit testing, navigation, viewport resize, and geometry restore. Impeccable review follows the final visible change; scoped Ruff, format, compile, diff, and static checks pass with mutation evidence for authority, lease, and cancellation guards.
<!-- AC:END -->
