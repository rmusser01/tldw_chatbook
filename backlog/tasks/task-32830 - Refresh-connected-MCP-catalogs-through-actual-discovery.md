---
id: TASK-32830
title: Refresh connected MCP catalogs through actual discovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 02:02'
updated_date: '2026-09-20 22:00'
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
ADR required: no. ADR path: N/A; existing ADR-111 and ADR-161 apply. Reason: restore existing catalog-refresh semantics within current service, transport and permission boundaries. Resume existing PR2714 from merged PR2713 dev 5e0f9f82c3. 1. Re-run saved real-stdio regressions before integrating the service patch. 2. Integrate the minimal service repair while preserving current boundaries. 3. Modernize and validate the native evidence runner; run focused neighbors and fresh four-cell native qualification. 4. Complete independent review and update the existing PR with a current visual gallery. Owner visual approval and current-head CI/review remain merge gates.

Qodo follow-up: verify the reported cleanup interleaving against real client scheduling, pin any ownership issue with a focused regression, validate fixture state/trace paths beneath an explicit trusted root with malicious-path coverage, document both public methods, and add isolated service branch tests alongside existing wire tests. Re-run affected checks and native evidence for changes, preserve the approved UI, then answer all review threads and complete current-head CI/latest-dev merge gates. Existing ADR-111/161 apply; no new architecture or ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refresh now uses the existing connect/discover path, preserving the original connected state and observe/launch gates. Post-connect failures clean up only the established session identity; rejected refresh leaves another pending connection untouched.

Validation: 15 distinct targeted cases pass, including nine isolated real-stdio regressions. Four native dark/light 80x24/170x48 journeys and sixteen rendered/inspected captures verify catalog changes, failure, retry and final disconnect. All twenty fixture processes and the app exited; defaults unchanged, lock released and ten private databases healthy. All seven preflight guards pass; no introduced Ruff diagnostics, new files and changed ranges formatted. Independent review findings on pending connection ownership and atomic fixture release are resolved.

Evidence: Docs/superpowers/qa/2026-09-18-mcp-connection-refresh/README.md. Scope is local stdio catalog lifecycle; compact toolbar clipping belongs to PR2712, connected execution and further screen reviews remain open. No full test sweep.

ADR required: no. Existing ADR-161 and ADR-111 apply; service/transport/policy/storage boundaries unchanged. Files: local_control_service.py, real stdio fixture/tests, QA evidence and review ledgers. Owner visual approval and current-head remote checks remain merge gates.

Integrated PR2714 on merged PR2713 dev 5e0f9f82c3 without conflicts. Four real-stdio regressions failed on unchanged dev; all 15 service cases pass after repair. Current qualification totals 49 distinct focused cases, seven artifact guards, no introduced lint diagnostics and independent review clear. Fixed reviewer finding in the native harness by moving fixture files into its exclusive evidence directory; live sentinel symlinks remain unchanged. Four real native cells and 32 inspected SVGs verify fresh catalog, failure/retry, original connection state and final disconnect. All twenty fixture processes and app exited, private databases healthy, lock released, defaults unchanged, zero network attempts. Immediate compact toasts overlap inspector actions; recorded with feedback captures as remaining UI follow-up, alongside separate PR2712 toolbar clipping and compact catalog scrolling. Current QA: Docs/superpowers/qa/2026-09-18-mcp-connection-refresh/current-dev/README.md. Existing ADR-111 and ADR-161, no new ADR. Keep In Progress pending current-head CI/review and fresh owner visual approval.
<!-- SECTION:NOTES:END -->
