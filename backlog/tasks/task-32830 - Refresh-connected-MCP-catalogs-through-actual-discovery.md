---
id: TASK-32830
title: Refresh connected MCP catalogs through actual discovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 02:02'
updated_date: '2026-09-19 02:20'
labels:
  - mcp
  - ui
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh tools must discover the current server catalog instead of reporting success with cached tools. Connection state and existing permissions must remain truthful before and after refresh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refreshing a connected local profile replaces cached discovery with the current tools, resources and prompts while keeping it connected.
- [x] #2 Refreshing an initially disconnected profile discovers fresh capabilities and restores its disconnected state.
- [x] #3 Observe and launch denials leave an existing session and its persisted catalog intact; failed refresh remains recoverable through the UI.
- [x] #4 Targeted regressions and a real local stdio journey verify catalog changes, failure recovery and process cleanup with documented visual bounds.
- [x] #5 A refresh rejected by launch permission or an existing connection attempt leaves that other pending connection running.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add isolated regressions for changed discovery, original connection state, permission denials, and failed refresh recovery.
2. Reuse the existing reconnect/discover operation while preserving the original connected state and both observe and launch permission gates.
3. Run targeted service regressions and a real local stdio connection/refresh/failure/recovery journey; inspect native captures and verify process cleanup.
4. Complete independent review, QA evidence and task notes; save a bounded draft PR against dev.

ADR required: no
ADR path: N/A (existing ADR-161 and ADR-111 apply)
Reason: Restores the advertised reconnect-and-refresh behavior through existing service and transport interfaces; no new storage, permission, transport or application boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refresh now uses the existing connect/discover path, preserving the original connected state and observe/launch gates. Post-connect failures clean up only the established session identity; rejected refresh leaves another pending connection untouched.

Validation: 15 distinct targeted cases pass, including nine isolated real-stdio regressions. Four native dark/light 80x24/170x48 journeys and sixteen rendered/inspected captures verify catalog changes, failure, retry and final disconnect. All twenty fixture processes and the app exited; defaults unchanged, lock released and ten private databases healthy. All seven preflight guards pass; no introduced Ruff diagnostics, new files and changed ranges formatted. Independent review findings on pending connection ownership and atomic fixture release are resolved.

Evidence: Docs/superpowers/qa/2026-09-18-mcp-connection-refresh/README.md. Scope is local stdio catalog lifecycle; compact toolbar clipping belongs to PR2712, connected execution and further screen reviews remain open. No full test sweep.

ADR required: no. Existing ADR-161 and ADR-111 apply; service/transport/policy/storage boundaries unchanged. Files: local_control_service.py, real stdio fixture/tests, QA evidence and review ledgers. Owner visual approval and current-head remote checks remain merge gates.
<!-- SECTION:NOTES:END -->
