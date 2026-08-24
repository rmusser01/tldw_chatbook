---
id: TASK-21134
title: >-
  Perf small-residue batch - setup-modal snow idle burn, casefold UDF, MCP log parses, executemany, connection closes, misc
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - cleanup
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21134). Each item
verified on the pin; none warrants a task alone, together they are a real tax.

1. Setup-modal snow animation ticks a full-screen refresh at 5 Hz - measured 9.9% of a core
   idle (vs 1.8% configured) for every not-yet-configured user: honor reduced-motion by default
   on low-core machines or lower the tick rate (console_setup_modal.py:88+).
2. `unicode_casefold` Python-UDF in WHERE + ORDER BY on watchlists agent-tool queries
   (Subscriptions_DB.py:2908-2985).
3. MCP execution log: two full-file JSON parses + fsync-on-close per tool invocation
   (MCP/execution_log.py:156).
4. Re-chunk per-chunk INSERT loop -> executemany (library_rechunk_service.py:265-271).
5. GC-leaked `with conn:` (no close) in sync_state / event_state / writing / research /
   tamagotchi stores.
6. `EnhancedStatusWidget` recompose-per-status-message during ingest (status_widget.py:82-140).
7. Media-viewer match-nav restyles the whole document per click (library_media_content.py:16-53
   - cache the match list, restyle two lines).
8. Trajectory brush-drag rebuilds the ledger DataTable per mouse-move
   (trajectory_screen.py:861-867 - throttle to once per frame or use the existing worker path).
9. `CAPABILITY_REGISTRY` builds 1,323 frozen dataclasses (62% server-only) and runs
   `validate_registry_completeness()` in production at every import (registry.py:1358,1414 -
   move validation to tests; lazy-build the server partition).
10. Dormant sqlite owners to verify-then-retire: Sync_Interop/notes_mirror.py (no prod caller),
    Widgets/Tamagotchi/tamagotchi_storage.py (never imported), Kanban boot connect with no UI,
    Third_Party/aider/repomap.py diskcache (no prod caller).

## Acceptance Criteria

- [ ] Each numbered item is either fixed as described or explicitly declined with a reason in the task notes
- [ ] No behavior change beyond the stated performance mechanics; touched areas keep their existing tests green
