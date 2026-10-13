---
id: TASK-34927
title: Add-a-staleness-gate-to-the-MCP-workbench-per-visit-catalog-reload
status: Done
assignee: []
created_date: 2026-10-13 14:51
labels:
- performance
- ui-perf
- wave-3-screen-switch
dependencies: []
priority: medium
updated_date: 2026-10-13 18:03
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP screen is non-reusable, so every visit re-mounts and re-runs the full catalog reload: service.load_context() + _collect_snapshots() + _sync_children() over every configured server (mcp_workbench.py:963-991 -> :1127-1162), with no staleness check against the previous visit's data. Switching to MCP repeatedly pays the full reload each time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Re-entering MCP with unchanged config and services skips the full catalog reload (staleness check)
- [x] Explicit refresh path still forces a full reload
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _start_initial_load/reload/_collect_snapshots and what identifies loaded data
2. Skip the full reload when config/service state is unchanged since the last visit
3. Keep explicit refresh forcing a reload
4. Tests + targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Found the right seam already maintained by the service: `run_catalog_fingerprint()` (TASK-33620.15.1) — the governance-grade identity of everything a catalog composition reads (permission store + local service admission-checked state), already used to validate cached run compositions, so it must already be correct or those would break.
- `reload()` gained `force: bool = False` and routes snapshot collection through `_collect_snapshots_stale_gated`: local-source reloads compare `(catalog fingerprint, current_config_identity())` against a cache on the app instance (outliving the per-visit widget); a match reuses the cached snapshots — `_sync_children()` still runs, so the fresh DOM is fully populated. Any catalog/permission/lifecycle/config change alters the fingerprint and forces the full walk. Server-source mode, admission refusals (fingerprint raises), missing fingerprint support, and `force=True` all bypass the gate.
- The two screen-side paths that must never be gated pass `force=True`: the `r` manual refresh and the runtime-backend-change refresh (mcp_screen.py).
- Context load (`load_context`) stays uncached: it is a small persisted-state read and carries user selection state.
- Tests: new `Tests/UI/test_mcp_snapshot_stale_gate.py` (4 tests: cache-hit reuse, fingerprint-change full walk, force + server-source bypass, admission-refusal fallback with no cache write) — 4/4 pass. Regression: `Tests/UI/test_mcp_workbench.py` + `test_mcp_workbench_lifetime.py` — 361/361 passed.
- ADR required: no — revisit cache over an existing service-maintained fingerprint.
