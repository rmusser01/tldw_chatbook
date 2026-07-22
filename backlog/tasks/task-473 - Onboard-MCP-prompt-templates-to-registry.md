---
id: TASK-473
title: 'Onboard MCP-exposed prompt templates to the Internal Prompts registry'
status: To Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - enhancement
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the program. MCP/prompts.py exposes conversation-analysis, document-generation, media-analysis, search-synthesis and character-interaction prompt templates over MCP as hardcoded f-strings. Onboard them to the registry so they are user-editable and consistent with the app's own prompts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MCP prompt templates are registered with byte-identical defaults (parity tests)
- [ ] #2 The MCP server resolves them via the registry
- [ ] #3 An override reaches the MCP-exposed template output (integration test)
<!-- AC:END -->
