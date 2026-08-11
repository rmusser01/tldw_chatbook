---
id: TASK-15454
title: Console rail search: move DB work inside its debounce and re-guard the workspace tray
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified first-hand: the `#console-workspace-conversation-search` handler (`chat_screen.py:1843-1907`) debounces only the FTS worker; synchronously per keystroke, on the event loop, it invalidates the persisted-rows TTL cache, reads workspace labels twice (including a possible write transaction via `ensure_default_workspace`), runs one `list_workspace_conversations` SELECT per workspace, reads starred ids at least twice, and then calls `_sync_console_workspace_context()` — which recomposes the workspace context tray unconditionally across up to 3 tray instances (~180-450 widgets), because the tray's equality guard was deliberately reverted (`Widgets/Console/console_workspace_context.py:546-560`, pinned by a test) after a full-equality guard caused a click-targeting regression.

Fix direction: move everything between `:1858-1893` inside the debounced timer callback; then design a NARROWER structural-key guard for the tray that avoids the historical regression (the reverted guard failed on full equality — a structural key over row identity/order can skip no-op recomposes without recreating the click-targeting bug). Stability constraints: keep the existing pinning test, and add a regression test for the click-targeting case that forced the revert before re-introducing any guard. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Zero SQLite queries and zero tray recompose on the keystroke path before the debounce fires (evidence)
- [ ] #2 Tray recomposes only on structural change, with the historical click-targeting regression covered by a test
- [ ] #3 Search results and rail behavior unchanged (existing surface green)
<!-- AC:END -->
