---
id: TASK-32502
title: 'MCP Hub: rail keyboard navigation (F2)'
status: Done
assignee: []
created_date: '2026-09-11 23:10'
updated_date: '2026-09-11 23:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the F2 gap from the 2026-09-11 UX review: the rail is Tab-only today. Add up/down (and j/k alias) bindings on MCPRail that move the selection through its rows via the existing ServerSelected path, mirroring table-cursor semantics, with focus kept on the rail so consecutive presses work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 up/down and j/k on the rail move selection through All-servers, server rows and the Agent tools row,Selection never wraps; clamps at the ends,Consecutive presses work without re-focusing (focus follows the new row),Existing click/selection behavior unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- F2 from the 2026-09-11 UX review (the one finding dropped during wave decomposition; surfaced in the remaining-work audit). Branch `feat/mcp-hub-ux-wave-d`.
- `up`/`down` + `j`/`k` bindings on `MCPRail` post the same `ServerSelected` a click posts, clamped at both ends (no wrap); Selects keep their own arrows while focused (widget focus wins).
- The load-bearing fix the RED tests forced: every selection triggers the host's resync → rail recompose, which DESTROYS the focused row — so focus restoration is scheduled at COMPOSE time (sync_state notes the rail owned focus; compose consumes the flag once the new rows exist and refocuses the selected row). A sync_state-time `call_after_refresh` races the recompose and restores against children that are about to be destroyed.
- Tests exercise the full loop with a harness that mirrors the workbench contract (host re-syncs the rail after every ServerSelected); paced presses match the real typing cadence across recomposes. One transient failure in the 367-test sweep did not reproduce (known destination-shells flake class).
- ADR check: none required (keybinding follows ADR-031 htop-style conventions; no structural change).
<!-- SECTION:NOTES:END -->
