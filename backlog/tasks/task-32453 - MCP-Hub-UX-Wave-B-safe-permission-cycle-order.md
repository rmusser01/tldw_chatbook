---
id: TASK-32453
title: 'MCP Hub UX Wave B: safe permission cycle order'
status: Done
assignee: []
created_date: '2026-09-11 19:25'
updated_date: '2026-09-11 19:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reorder the Permissions Space-cycle so the first press from Inherit lands on Ask, not Allow (Inherit → Ask → Allow → Off). Allow becomes a deliberate second press. Store helper, legend copy, and every pinned test move together in one change (UX program 2026-09-11, safety assumption adopted at review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 cycle_ui_state(None) returns ask (not allow),Legend reads Space cycles Inherit → Ask → Allow → Off,All pinned tests updated to the new order in the same change,Permission regression suites green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: tests-first (updated pins watched RED against the old ring), then the one-map store change. Commit 728d3ea on `feat/mcp-hub-ux-wave-a`.
- Change surface: `permission_store._CYCLE_UI_STATES` (None→ask→allow→deny→None), `cycle_ui_state` docstring, Permissions legend constant. `cycle_global` deliberately unchanged — from the global default (ask) one press already goes to deny, so Allow is never the first stop there either.
- Pinned tests updated in the same commit: `test_permission_resolution.py` full-loop; `test_mcp_permissions_mode.py` first-press event + two verbatim legend pins; `test_mcp_workbench.py` — 12 flow tests adjusted intent-preserving (one-press flows now assert the ask/override marker; flows that need `allow` press twice). TASK-627 built-in tests post explicit `new_state` values and needed only comment updates.
- Trap caught by TDD: a blanket string replacement initially rewrote a structurally identical assertion in the re-allow test (which legitimately expects `allow` after Re-allow); the immediate test run caught it and it was reverted.
- ADR check: none required (interaction order within existing states; no storage/schema/interface change — rationale recorded in the store comment and program review).
- Historical phase docs/specs (2026-07) that describe the old ring order were left as dated records; Docs/User_Guide/mcp.md does not pin the order (doc-contract verified).
- Verification: Tests/MCP/test_permission_resolution.py + test_permission_store.py 119 passed; combined A+B run 515 passed (workbench/permissions-mode/resolution/store).
<!-- SECTION:NOTES:END -->
