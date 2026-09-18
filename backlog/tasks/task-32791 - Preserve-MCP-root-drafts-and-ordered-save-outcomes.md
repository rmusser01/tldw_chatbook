---
id: TASK-32791
title: Preserve MCP root drafts and ordered save outcomes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 13:53'
updated_date: '2026-09-18 14:27'
labels:
  - mcp
  - ui
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep local MCP root edits and save receipts consistent with actual configuration and runtime authority through refresh, overlapping saves and navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root guidance accurately names local MCP serving and Hub tests, blank current-directory fallback, and separate Console file authority.
- [x] #2 Read-only refresh and source/mode navigation preserve unsaved root edits and local error or success feedback; validation and failed saves retain retryable drafts.
- [x] #3 Explicit saves retain submission order and exact draft revisions; older outcomes never overwrite newer drafts, including A-to-B-to-A edits.
- [x] #4 Admitted saves and their bounded latest receipt survive departing or recreating MCP; app shutdown closes admission and drains admitted writes.
- [x] #5 Partial cache-refresh success remains truthful across repeat saves, queued writes refuse changed configuration identity, and presentation-refresh failure cannot misreport persistence.
- [x] #6 Targeted UI, ownership, configuration, governance and native private theme/size journeys qualify the change; ledgers and ADR are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-18-mcp-root-settings.md: reproduce draft and save races; add narrow app-owned ordered writes; preserve exact draft revisions and partial outcomes; correct scope copy; verify targeted and native paths; update ledgers and draft PR2707. ADR required: yes. ADR path: backlog/decisions/168-mcp-root-save-lifetime.md. Reason: admitted root writes and receipts must outlive disposable MCP screens.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ordered app-owned MCP root saves with shielded observers, shutdown admission/drain and bounded receipts. Mounted drafts retain explicit identity across refresh, ABA edits and overlapping saves; recreated views use accurate receipt copy. Config path, publication generation and file revision prevent stale outcomes from replacing later settings, including external TOML reloads. Root copy now describes local MCP serving/Hub tests and separate Console authority. Validation/save receipts are visible beneath Save at compact size. ADR-168 and QA README record 26 passing root checks, adjacent targeted checks, independent review and ten inspected final native captures with clean private shutdown. Related Permissions resize clipping reproduces on unchanged baseline6c0e317ab7 and remains separately open; prior-head Windows GGUF SelectOverlay CI failure is also not closed. No full suite or provider/tool execution. Updated root owner, workbench/canvas, app shutdown, config guidance, tests, diagnostic inventory and review ledgers.
<!-- SECTION:NOTES:END -->
