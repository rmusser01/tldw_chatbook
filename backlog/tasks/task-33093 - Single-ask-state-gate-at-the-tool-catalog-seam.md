---
id: TASK-33093
title: Single ask-state gate at the tool catalog seam
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, tools, security]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The pending-gate, stamp, and approval engine is implemented five times — local, virtual-CLI, raw-shell, and MCP tool providers plus the builtin gate — all resolving against the same MCP permission store, with near-verbatim admitted-root checks copied between providers. One gate component at the existing ToolCatalogRegistry.invoke_by_name dispatch choke point (keyed by server and tool, owning stamps, session grants, arg-rules, persist, and audit) removes the copies and makes future remote-tool permission parity hold by construction instead of by a sixth copy. Security-sensitive: the local provider's gate_error versus deny taxonomy and its documented failure-fix history encode real incidents; the unified gate must adopt the strictest semantics or it regresses those fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One gate component keyed by server key and tool name owns stamps, session grants, arg-rules, persist, and audit.
- [ ] #2 All providers route through the unified gate at the invoke_by_name seam.
- [ ] #3 The local failure taxonomy (gate_error versus deny) is preserved with regression tests referencing the original incident fixes.
- [ ] #4 Batch decisions and revoke flows are behavior-identical across all providers.
- [ ] #5 Dedicated test coverage exists for the unified gate itself.
<!-- AC:END -->
