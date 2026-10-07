---
id: TASK-32684
title: Expose owned plugin MCP tools with scoped connection leases
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:28'
updated_date: '2026-10-01 05:42'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32683
  - TASK-32672
  - TASK-32673
  - TASK-32675
  - TASK-32678
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let plugin skills invoke reviewed MCP tools through normal authority while shared connections retain per-workspace ownership.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Owned profiles and tools register through existing MCP/catalog services with exact namespace, definition digest, mappings and normal permission enforcement.
- [x] #2 Portable variable expansion applies only to approved fields with host variables last; configuration save does not launch a server and explicit testing uses normal authority.
- [x] #3 Connection reuse requires equivalent reviewed execution, configuration, credential binding and qualified session isolation; otherwise each scope receives a separate connection.
- [x] #4 Cancelling A detaches/cancels A requests without killing authorized B transport; idle processes retain data ownership, uncertain outcomes remain counted, and standalone mutation services reject package-owned edits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements accepted reviewed MCP mapping publication, component ceilings and scoped connection/root custody through existing owners.
1. Read the actual MCP/catalog, plugin admission/coordinator/revocation/root flow and reviewed M4 increment, including R65-R69 constraints and qualification fixtures.
2. Establish owned registration/connection RED after current M3 positive controls.
3. Integrate exact owned profile/tool mappings, portable expansion, immutable ceilings and scope leases while preserving current protected storage, producer guards, native cleanup and plugin process qualification.
4. Run targeted owned multiplexed stdio/HTTP lifecycle/root and credential/catalog neighbors plus static/resource checks; document limits and commit only M4-owned files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented owned schema-4 data-only profiles, exact authenticated connection/tool configuration reviews, namespace-safe normal MCP/catalog registration, component ceilings, scoped sharing/cancellation and idle/uncertain root custody. Preserved current native guards by reacquiring retained-task admission; retained actual shared stdio requests after timeout without killing B; preserved H3 approval metadata and exact portable literals/data bindings. ADR-162/163 apply. Complete 287-node task covering set verified through 286 passes plus corrected original-deletion/final native controls; 144 stdio regressions, 21 final settlement checks, 8 actual native retirement checks and 14 final auth checks pass. Credential cancellation harness race reproduced on frozen M3 and corrected without production timeout changes. Evidence and limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. Native managed application/graph composition belongs to I1.
<!-- SECTION:NOTES:END -->
