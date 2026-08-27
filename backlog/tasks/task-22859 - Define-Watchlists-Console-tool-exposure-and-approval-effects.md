---
id: TASK-22859
title: Define Watchlists Console tool exposure and approval effects
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:16'
labels:
  - watchlists
  - console
  - mcp
  - security
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the fail-closed catalog contract that distinguishes Console-only Watchlists commands and private item/briefing reads from metadata reads eligible for external MCP exposure, with approval effects derived from code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every local tool descriptor declares an explicit Console/external-MCP exposure value; missing or invalid exposure prevents provider construction.
- [ ] #2 Approval-effect metadata is code-owned and distinguishes private read, local mutation, network access, and possible LLM spend without relying on model-supplied arguments or unenforced risk tags.
- [ ] #3 Console composition can register every approved Watchlists read/command descriptor, while external MCP publishes only bounded source, collection, operation, and briefing-receipt metadata descriptors even when a Console-only article or briefing tool has a persisted Allow.
- [ ] #4 Read-only project bindings and `allow_write=False` omit all mutating Watchlists commands as well as filesystem writes.
- [ ] #5 Catalog, permission, definition-hash, kill-switch, and external-publication tests pin the fail-closed contract and retain ADR-032 refusal behavior.
- [ ] #6 ADR-032 and local-tool documentation describe the exposure/effect boundary without implying that descriptor exposure grants authorization.
<!-- AC:END -->
