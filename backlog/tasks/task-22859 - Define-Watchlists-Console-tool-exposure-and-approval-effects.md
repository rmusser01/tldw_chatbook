---
id: TASK-22859
title: Define Watchlists Console tool exposure and approval effects
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 06:17'
labels:
  - watchlists
  - console
  - mcp
  - security
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the fail-closed catalog contract that distinguishes Console-only Watchlists commands and private item/briefing reads from metadata reads eligible for external MCP exposure, with approval effects derived from code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every local tool descriptor declares an explicit Console/external-MCP exposure value; missing or invalid exposure prevents provider construction.
- [x] #2 Approval-effect metadata is code-owned and distinguishes private read, local mutation, network access, and possible LLM spend without relying on model-supplied arguments or unenforced risk tags.
- [x] #3 Console composition can register every approved Watchlists read/command descriptor, while external MCP publishes only bounded source, collection, operation, and briefing-receipt metadata descriptors even when a Console-only article or briefing tool has a persisted Allow.
- [x] #4 Read-only project bindings and `allow_write=False` omit all mutating Watchlists commands as well as filesystem writes.
- [x] #5 Catalog, permission, definition-hash, kill-switch, and external-publication tests pin the fail-closed contract and retain ADR-032 refusal behavior.
- [x] #6 ADR-032 and local-tool documentation describe the exposure/effect boundary without implying that descriptor exposure grants authorization.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing descriptor, external-publication, read-only filtering, and approval-card behavior tests.
2. Implement required LocalToolExposure and LocalApprovalEffect metadata and derive external publication from descriptors.
3. Carry code-owned effects into pending calls and render plain-language approval effects without changing authorization semantics.
4. Verify ADR-032's already-present approved addendum and update Console tool documentation.
5. Run the task-targeted pytest, Ruff, documentation contract, and diff checks; self-review and independently review the task.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032 owns the synthetic local principal, approval semantics, and external MCP publication boundary; this task implements its approved addendum.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a fail-closed LocalToolSpec exposure/effect contract and explicit inventory. External MCP publication now derives solely from descriptor exposure; Console-only Watchlists item/detail/briefing content cannot be published by persisted Allow. Code-owned effects flow through pending calls and the Console controller into production approval cards without inspecting raw arguments; allow_write=False filters all mutates_local descriptors. Updated the Console guide and verified ADR-032's existing addendum. Added descriptor, publication, permission, controller, formatter, and production-mounted Textual regressions.

Verification: fresh targeted suite 389 passed, 5 skipped because optional mcp_unified is not installed; Ruff and git diff --check passed. Independent review approved after the mounted production-CSS regression was added. ADR required: yes; existing backlog/decisions/032-local-agent-tool-permission-boundary.md applies. No new ADR or dependency.
<!-- SECTION:NOTES:END -->
