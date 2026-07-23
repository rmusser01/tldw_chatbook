---
id: TASK-400
title: Fix MCP navigation crash by requiring Textual 8
status: To Do
assignee: []
created_date: '2026-07-23 22:51'
updated_date: '2026-07-23 22:52'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-23-mcp-textual-runtime-floor-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the application from crashing when users navigate to the MCP destination after installing a Textual version that predates the Select.NULL API used by the MCP UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Package metadata requires Textual 8.0.0 or newer
- [ ] #2 The MCP destination mounts without a Select.NULL AttributeError
- [ ] #3 A regression test prevents lowering the Textual runtime floor below 8.0.0
- [ ] #4 Focused MCP tests pass on the supported runtime
<!-- AC:END -->
