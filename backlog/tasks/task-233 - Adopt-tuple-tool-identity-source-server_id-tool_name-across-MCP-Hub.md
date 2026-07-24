---
id: TASK-233
title: 'Adopt tuple tool identity (source, server_id, tool_name) across MCP Hub'
status: To Do
assignee: []
created_date: '2026-07-16 15:18'
labels:
  - mcp
  - phase4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3 packs tool identity into a '::'-joined string (tool_id = server_key::name) parsed with partition('::'). Profile-id charset validation (rejecting ':' and whitespace) shipped as the minimum guard, but spec §8's identity model is a tuple, and the Phase 4 permission store will key on tool identity — the packed string must be replaced before that. PHASE 4 PREREQUISITE (from PR #639 final review I3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tool identity flows as (source, server_id, tool_name) through tools-mode row keys, inspector messages, and test_hub_tool routing,No '::' packing/parsing remains in the execute path,Legacy colon-bearing profile ids on disk are handled (load-time migration or rejection with clear copy),Execution-log records carry the tuple fields
<!-- AC:END -->
