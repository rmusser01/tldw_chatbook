---
id: TASK-6.4
title: Phase 5.4 Apply recovery taxonomy to optional dependency blockers
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 04:03'
updated_date: '2026-05-05 04:04'
labels:
  - unified-shell
  - phase-5
  - recovery
  - optional-dependencies
milestone: Unified Shell UX
dependencies: []
documentation:
  - >-
    Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
parent_task_id: TASK-6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply the shared Phase 5 recovery taxonomy to visible optional-dependency blocked states so users can understand missing local extras such as embeddings/RAG, transcription, speech, or local model dependencies and recover through setup or install guidance instead of generic failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Visible optional-dependency blocker states expose what is unavailable, why, next action, recovery target, authority owner, stable selector, and disabled tooltip copy where a disabled control is present.
- [x] #2 At least two high-impact optional-dependency surfaces are covered with taxonomy-aligned recovery copy, prioritizing RAG/embeddings and media transcription or speech/local model setup if present in the shell UI.
- [x] #3 Automated regressions verify representative optional-dependency recovery copy and stable selectors/tooltips.
- [x] #4 Durable Phase 5 QA evidence records verification, residual risks, and whether Phase 5 is ready for final closeout QA.
- [x] #5 Parent Phase 5 tracking is updated without marking the phase verified unless final running-app closeout QA is also completed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Identify visible optional-dependency blocker states in shell-adjacent UI, prioritizing RAG/embeddings and media transcription or speech/local model setup.
2. Add failing UI or unit regressions proving those states lack Phase 5 recovery taxonomy fields, stable selectors, or disabled tooltips.
3. Implement the smallest shared recovery helper or screen-local wiring needed to expose taxonomy-aligned optional-dependency recovery copy.
4. Update Phase 5 QA evidence, roadmap, and Backlog tracking for TASK-6.4 without prematurely closing Phase 5.
5. Run focused tests, py_compile on touched files, and diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a shared optional-dependency recovery helper and applied it to two visible blocker surfaces: Search/RAG missing embeddings extras and STTS local speech missing TTS/STT extras. Search/RAG now shows persistent taxonomy recovery copy and disabled tooltips for the search input/button, while STTS local speech status uses the same recovery fields and install guidance. Added focused regressions plus Phase 5 evidence and tracking updates without closing the parent Phase 5 task.
<!-- SECTION:NOTES:END -->
