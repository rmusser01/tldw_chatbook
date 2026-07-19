---
id: TASK-236
title: Batch the residual N+1 store loads in get_external_servers
status: Done
assignee: ['@claude']
created_date: '2026-07-16 15:19'
labels:
  - mcp
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3's get_catalog_bundle() cut local_external_catalog from 2N+1 to N+2 store loads, but LocalMCPControlService.get_external_servers() still does list_profiles + one get_discovery_snapshot per profile. Extend the bundle usage inside the local service so a full catalog read is one load (PR #639 Task-2 review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full external-server catalog read performs a single store load,Governance gating unchanged,Existing control-plane tests stay green unmodified
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Add a store-level joined reader (one load) with the exact key normalization get_discovery_snapshot uses.
2. Rewire get_external_servers to it, keeping governance gating and the returned shape unchanged; duck-typed fallback for legacy store doubles.
3. Unmocked load-counting + shape-equivalence tests on the real store.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Note: the description's get_catalog_bundle()/local_external_catalog names no longer exist on dev (post-#639 refactors); the residual N+1 it describes was live in LocalMCPControlService.get_external_servers — list_profiles (1 full state load) + get_discovery_snapshot per profile (N loads; every LocalMCPStore accessor re-reads and re-parses the store file).

Fix: LocalMCPStore.get_external_catalog() joins profiles with snapshots from ONE load(), using the same _text(profile_id) key normalization as get_discovery_snapshot. get_external_servers consumes the pairs; governance gate (mcp.external_profiles.list.local) and the returned dict shape are unchanged. A getattr fallback keeps duck-typed store doubles (e.g. test_local_control_service's FakeLocalStore) on the legacy per-item path — required to keep existing control-plane tests green UNMODIFIED per the AC.

Tests: Tests/MCP/test_external_catalog_single_load.py — 3, real LocalMCPStore on tmp_path (load-count == 1 for the whole catalog, per-profile shape equivalence vs the individual accessors incl. None-snapshot profiles, store-level join equivalence). Full Tests/MCP/: 314 passed, zero modified.

Files: MCP/local_store.py, MCP/local_control_service.py, Tests/MCP/test_external_catalog_single_load.py.
<!-- SECTION:NOTES:END -->
