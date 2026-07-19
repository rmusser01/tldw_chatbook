---
id: TASK-236
title: Batch the residual N+1 store loads in get_external_servers
status: To Do
assignee: []
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
- [ ] #1 Full external-server catalog read performs a single store load,Governance gating unchanged,Existing control-plane tests stay green unmodified
<!-- AC:END -->
