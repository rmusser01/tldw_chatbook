---
id: TASK-89.4
title: Clarify Library Import Export as destination-owned source workflow
status: Done
dependencies:
- TASK-89.1
labels:
- library
- import-export
- ux
priority: medium
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Library Import/Export feel like a destination-owned source workflow rather than a confusing jump to generic ingest tooling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library clearly explains which import/export workflows are owned by Library and which deeper operations route to Ingest or Media.
- [x] #2 The Import/Export mode shows available source import/export actions, prerequisites, blocked states, and recovery copy.
- [x] #3 Route handoffs preserve context and make return-to-Library behavior obvious where supported.
- [x] #4 The UI distinguishes source-level Library Import/Export from full Media ingestion, artifact export, and generic file management.
- [x] #5 Focused regressions cover visible actions, blocked states, and route handoff copy.
- [x] #6 Actual CDP/Textual-web screenshot QA is approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a Library UI/UX routing and copy clarification only; it does not change storage, schema, sync policy, runtime boundaries, security, or service contracts.

1. Add failing mounted regressions for Import/Export mode visibility, explicit owner/handoff copy, blocked export state, and route handoff buttons.
2. Update destination-shell compatibility coverage so the left Library Import/Export action stays native while a dedicated import action emits the ingest route.
3. Implement the smallest LibraryScreen changes: native import-export mode rows, inspector rows, action widgets, status row copy, and explicit route handlers.
4. Run focused pytest red/green verification plus broader Library shell/parity checks.
5. Capture an actual Textual-web/CDP screenshot for approval before marking the task Done.
<!-- SECTION:PLAN:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not duplicate full Ingest or Media workflows inside Library.
- Do not move artifact import/export ownership into Library.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the direct left-rail `Import/Export Sources` route jump with a Library-native Import/Export mode.
- Added explicit center-pane copy for Library-owned source acquisition framing, Ingest handoff, Media review ownership, artifact/export boundary, generic file-management boundary, and return-to-Library recovery.
- Added inspector/action rows with dedicated `Open Ingest` and `Open Media` handoff buttons plus disabled `Export Library sources` recovery copy.
- Added focused mounted regressions for native mode activation, visible boundaries, disabled export state, route handoff buttons, and updated destination-shell compatibility behavior.
- ADR required: no. ADR path: N/A. Reason: UI routing/copy clarification only, no storage/schema/sync/runtime/security/service boundary change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Import/Export now behaves like a destination-owned source acquisition workflow. The left Library action opens a native Import/Export mode, while explicit inner actions hand off to Ingest or Media with visible return-path and ownership copy. Source export remains honestly blocked with recovery guidance.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria completed.
- [x] Implementation plan followed.
- [x] Focused automated regressions added and run.
- [x] Broader Library shell/parity verification run.
- [x] Actual CDP/Textual-web screenshot captured and approved.
- [x] QA notes updated.
- [x] ADR check completed.
<!-- DOD:END -->
