---
id: TASK-32684
title: Expose owned plugin MCP tools with scoped connection leases
status: To Do
assignee: []
created_date: '2026-09-16 04:28'
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
- [ ] #1 Owned profiles and tools register through existing MCP/catalog services with exact namespace, definition digest, mappings and normal permission enforcement.
- [ ] #2 Portable variable expansion applies only to approved fields with host variables last; configuration save does not launch a server and explicit testing uses normal authority.
- [ ] #3 Connection reuse requires equivalent reviewed execution, configuration, credential binding and qualified session isolation; otherwise each scope receives a separate connection.
- [ ] #4 Cancelling A detaches/cancels A requests without killing authorized B transport; idle processes retain data ownership, uncertain outcomes remain counted, and standalone mutation services reject package-owned edits.
<!-- AC:END -->
