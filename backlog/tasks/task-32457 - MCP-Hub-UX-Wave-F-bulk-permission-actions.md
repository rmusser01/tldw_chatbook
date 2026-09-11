---
id: TASK-32457
title: 'MCP Hub UX Wave F: bulk permission actions'
status: Done
assignee: []
created_date: '2026-09-11 21:12'
updated_date: '2026-09-11 21:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-149 Wave F: shift+space applies the cursor row's next cycled state to a server's VISIBLE tool rows; C clears its visible overrides. Filter is the scope; every write stays an ordinary profile-scoped set_tool_state call; raw-shell rows skipped and named in the echo. Plan: Docs/superpowers/plans/2026-09-11-mcp-hub-bulk-permission-actions.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 shift+space bulk-applies the row's next state to the server's visible tool rows with one echo,C clears only visible overrides and teaches the full-clear recipe,Global row and zero-visible-rows are hint no-ops not toasts,Raw-shell rows are skipped and named in the echo,First write failure stops the batch with a reason toast,Footer and legend copy updated with their pins
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Executed inline per Docs/superpowers/plans/2026-09-11-mcp-hub-bulk-permission-actions.md (3 tasks, TDD). Branch `feat/mcp-hub-ux-wave-d` (program waves D/E/F share the stacked branch).
- Canvas (single writer preserved): `shift+space`/`C` bindings post `BulkStateRequested(server_key, tool_names, new_state, profile_context)` / `BulkClearRequested(...)` — the canvas alone knows visibility, so it computes the scope (C sends only overridden rows); no-op cases (global row, zero matching visible rows) flash a transient hint line under the legend via `flash_hint()` (the spec's "existing hint Static"), never a toast. Footer + legend copy landed with their pins (2 verbatim legend tests, destination footer test).
- Workbench: `_apply_bulk_tool_states` executes the batch as N ordinary `_call_profile_scoped(service.set_tool_state, ...)` calls — no batch API, per-row audit logging unchanged. Raw-shell and vanished-tool rows are skipped and named in the echo; first service failure stops the batch with the Wave-A reason toast; one `_sync_permissions_mode(echo=...)` at the end.
- **Design catch from the RED tests:** one captured PermissionProfileContext cannot span the batch — each write changes the profile digest, so write #2 fails `stale_profile` (the single-press path never hits this; it writes once). Fixed with a per-write context re-capture from `_tool_policy_inventory()` plus a selection-identity guard (profile id + selector generation unchanged, else the stale toast and stop; partial writes stand, each idempotent).
- Wave-B interplay verified: first `shift+space` from Inherit applies Ask to the visible set.
- Spec open questions resolved to defaults (in-plan): per-row execution-log entries unchanged; shift+space on a server-default row applies that row's own next state.
- ADR: ADR-149 (linked); no additional ADR.
- Verification: test_mcp_workbench 348 passed (full); permissions/servers/rail/tools 185 passed; permission resolution/store 119 passed; doc-contract unchanged at the pre-existing 39 failures; ruff before=after on touched files.
<!-- SECTION:NOTES:END -->
