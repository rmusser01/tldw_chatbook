---
id: TASK-10.6.1
title: 'Gate 1.5.1: Console native display-state contracts'
status: Done
assignee: []
created_date: '2026-05-07 03:37'
updated_date: '2026-05-07 04:25'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create pure Console display-state seams and mounted red regressions that define the native Console chrome before replacing legacy ChatWindowEnhanced internals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console display-state helpers expose provider model persona source RAG tool approval artifact and recovery labels without reading widget internals.
- [x] #2 Mounted regressions prove the current Console still exposes legacy embedded chrome that Gate 1.5 must remove or isolate.
- [x] #3 Existing Gate 1 Console shell tests remain runnable as compatibility guardrails.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the existing Console Gate 1 shell handoff and shell-bar baseline.
2. Add failing pure Console display-state tests for provider/model context staged provenance and inspector recovery labels.
3. Add strict xfailed mounted guardrails documenting the current embedded legacy Console chrome that later Gate 1.5 tasks must remove.
4. Implement minimal pure dataclasses in tldw_chatbook/Chat/console_display_state.py with no Textual imports.
5. Run focused display-state and Console shell verification plus diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added tldw_chatbook/Chat/console_display_state.py with pure ConsoleControlState ConsoleStagedContextState ConsoleInspectorState and ConsoleDisplayRow contracts. Added unit coverage for provider/model/persona/RAG/source/tool/approval/artifact/recovery labels, including falsy label preservation and explicit Chatbook-save capability state. Added strict xfailed mounted Console internals guardrails that record the current legacy ChatWindowEnhanced chrome and missing native composer as intentional Gate 1.5 follow-up red states. Existing Gate 1 Console shell handoff and shell-bar baseline remains green.
<!-- SECTION:NOTES:END -->
