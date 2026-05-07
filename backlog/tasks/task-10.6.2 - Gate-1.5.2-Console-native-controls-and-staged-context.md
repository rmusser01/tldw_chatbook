---
id: TASK-10.6.2
title: 'Gate 1.5.2: Console native controls and staged context'
status: Done
assignee: []
created_date: '2026-05-07 03:37'
updated_date: '2026-05-07 04:58'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies:
  - TASK-10.6.1
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move provider/model readiness and staged context into Console-owned widgets while preserving legacy chat behavior behind compatibility seams.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console owns visible provider model persona RAG/source readiness controls outside the transcript region.
- [x] #2 Staged Library Search/RAG and live-work context appears in a native staged-context tray with provenance and recovery copy.
- [x] #3 Provider/model selector changes still synchronize with existing chat send behavior and compact controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted regression proving a pending Console handoff renders provider/model/persona/RAG/source labels plus staged context provenance in native Console regions.
2. Introduce Console-owned control bar and staged-context tray widgets backed by the pure display-state contracts from TASK-10.6.1.
3. Wire the widgets into `ChatScreen` without changing route IDs or removing the legacy transcript/composer compatibility surface in this slice.
4. Add focused TCSS source rules for the new Console regions and regenerate the bundled stylesheet.
5. Run focused Console, chat shell, and handoff regressions plus diff hygiene before marking acceptance criteria complete.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Console-native control and staged-context widgets backed by the pure display-state contracts from TASK-10.6.1. `ChatScreen` now mounts the control bar above the transcript, routes staged live-work payloads through a native tray with provenance/recovery rows, and keeps the embedded legacy chat window available for transcript/composer compatibility until later Gate 1.5 slices. The Console compact model bar uses the existing `CompactModelBar` sync path and routes its sidebar toggle to the embedded chat settings sidebar. TCSS source rules were added and the modular stylesheet was regenerated; the CSS build exited 0 with the pre-existing missing `features/_evaluation_v2.tcss` warning.
<!-- SECTION:NOTES:END -->
