# PR 2427 coordinated lifecycle repair plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Preserve other workers' edits; verify each bounded change before integration.

**Goal:** Repair fresh Work replacement and exact Notes in-place follow-up ownership without regressions or stale content.

**Architecture:** The existing browse shell retires its own exact instances; it always mounts fresh Work. Notes reports an in-place bulk update to the existing canvas-sync coordinator, which attaches follow-ups to the canvas that actually recomposes. No reuse cache, registry, new scheduler or global cleanup.

**Tech Stack:** Python 3.12 test environment, Textual 8, pytest, existing SQLite lifecycle fixtures.

ADR required: yes
ADR path: backlog/decisions/141-library-work-retirement-and-in-place-sync-followups.md
Reason: explicit cross-module sync disposition and Work-retirement contracts; implements the user's approved coordinated repair.

## Task 1: Safe fresh Work replacement

Files: `tldw_chatbook/Widgets/Library/library_browse_reader_shell.py`,
`Tests/UI/test_library_media_reader_shell.py`.

- [x] Run the existing rapid replacement regression RED; retain original three Notes gutter regressions.
- [x] Strengthen rapid coverage with same-ID fresh-object/current-copy checks and actual TextArea render after both route directions.
- [x] Hide outgoing Work before awaits. Before mount, retire an exact hidden non-current same-ID child; defer ordinary retirement behind refresh with exact-parent/current-owner guards. Never discard the incoming fresh object.
- [x] If that retirement still races, establish the needed refresh-completion boundary in this owner using the actual Textual lifecycle, not a delay or registry.
- [x] Verify the controls, three original route/autosave tests, complete reader-shell and phase-C files; obtain independent review.

## Task 2: Coordinator-owned in-place follow-ups

Files: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`,
`tldw_chatbook/UI/Library_Modules/canvas_sync.py`,
`Tests/UI/test_library_notes_reader.py`, `Tests/UI/test_library_canvas_sync_defects.py`.

- [x] Reproduce old callback surviving an in-place bulk update followed by editor-owned sync; cover explicit callback exact-once behavior.
- [x] Have `sync_state` return a narrow in-place-bulk disposition; existing non-bulk paths retain behavior. Apply guarded presentation without replacing fields.
- [x] In the coordinator, use the disposition to clear superseded Work follow-ups and attach the exact current follow-up to the list canvas that actually recomposes; retain existing automatic focus veto and explicit `then` semantics.
- [x] Remove `_run_in_place_follow_up`; do not add another callback polling owner.
- [x] Verify original bulk preview, field text/selection/undo, newer focus, mode transition and controlled stale-callback cases; obtain independent review.

## Task 3: Integration, evidence and publication

- [x] Run focused former failures plus complete architecture ratchets; preserve or lower all existing caps and review diagnostics before regeneration.
- [x] Save verified repair checkpoint; rebase onto fetched latest dev. Union fresh-adoption success, resident-sync failure propagation and route validation with captured-focus handoff.
- [x] Freeze sources for complete affected-owner/native-resource qualification using the existing observation-only runner; no whole-repository sweep or forced cleanup.
- [x] Run derived preflight and scoped lint, compare inherited lint baseline, review all changes independently.
- [ ] Publish with the exact remote lease, address final-head Qodo/CI findings, and merge normally only when all local and published-head gates clear.

Test invocation: `.venv/bin/python -I -m pytest <named affected files/nodes> -q --tb=short --show-capture=no --basetemp=<fresh per-user TMPDIR directory>/pytest`.
Expected RED is a contract assertion or captured lifecycle error; expected GREEN is exit zero with all named cases passing. Record full counts and native retained-resource evidence in `backlog/docs/pr-2427-rebase-reconciliation.md`.

## Post-qualification dev churn

Frozen evidence on runtime d034c2d9c9: 1581 passed, one existing skip across the
16 affected native-observed files; zero final SQLite/instance-lock retention.
The shell's exact-owner retirement needed no additional registry or scheduler.

- [x] Save evidence and rebase onto dev e574c81d22 or newer after inspecting churn.
- [x] Preserve upstream Library column defaults, boot sentinel contract and Watchlists width behavior while retaining the reviewed PR lifecycle/resource repairs.
- [x] Verify changed width/boot/Watchlists owners and affected Library geometry/focus integration, preflight and unchanged caps; independently review conflict resolution.
- [ ] Publish, address final-head reviews/checks and merge normally.

ADR required: no new ADR
ADR path: existing backlog/decisions/086-library-adaptive-reader-shell.md and upstream Watchlists width design
Reason: reconcile accepted upstream contracts without introducing a new boundary or layout policy.
