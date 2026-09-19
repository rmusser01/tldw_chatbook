---
id: TASK-32713
title: Keep RAG answer failure recovery visible beside Run
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 05:49'
updated_date: '2026-09-17 06:00'
labels:
  - library
  - search-rag
  - recovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The answer-generation review reproduced compact layouts where the query retains focus and Run re-enables after a provider failure, but the failure and retry guidance are below the viewport. Users need immediate, accurate feedback beside the query and a fresh result when they retry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Answer-generation failures are visible beside Run at wide and compact sizes in both themes, while provider disclosure and detailed recovery remain available.
- [x] #2 Keyboard retry preserves the query and source choices, clears stale failure feedback while busy, and displays a fresh cited answer after success.
- [x] #3 Mounted and native controlled-provider checks verify exception and empty-answer recovery without changing retrieval or provider contracts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/003-settings-library-rag-defaults.md apply.
Reason: restore visible recovery feedback for the existing answer phase without new storage, provider boundaries or interaction structure.
1. Retain the mounted red reproduction (both compact themes fail; wide layouts pass), then extend it through exception/empty-answer and successful retry, with disclosure and busy-state assertions.
2. Extend the existing persistent notice with a brief answer-failure message. Reuse the existing gate synchronization and token-backed styling; retain detailed answer errors below.
3. Run targeted answer/retrieval tests and static checks, obtain independent review, and verify native dark/light wide/compact journeys with real local retrieval and a controlled provider seam. Inspect captures and shutdown/storage; update evidence, audit and task, then commit locally. No full suite, push or merge.
Task allocation: 267 refs and 27 worktrees swept, maximum 32712; CLI offered 32713, assigned 32713 before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Answer-generation failure now appears in the existing persistent notice beside Run, while provider disclosure and the detailed Answer error/retry hint remain available. The existing busy gate clears the notice during retry and a successful cited reply removes it. Only the panel notice helper changes; no new state, style or provider boundary.

Eight mounted exception/empty-answer-to-success journeys verify both themes and sizes, notice/disclosure/Run paint, query/scope/focus retention, a held retry and citation validation. The pre-fix reproduction failed in both compact themes (2 failed / 2 wide passes). Final targeted checks: 326 passed without exclusions. New files pass Ruff lint/format, the changed production range is formatted, and baseline lint remains 5/5 with no additions. Independent review found no actionable issue.

Eight native journeys passed at 170x48 and 80x24 in both themes, with 16 real keyword searches adapted to RAG mode and 16 controlled provider calls. All 12 captures were inspected; sources/default files stayed unchanged, ten private databases passed quick_check, and normal exit/PID absence were verified. Two initial runner assumptions were corrected: keyword Media evidence carries source labels, and an inspection-only scroll must be restored before the next theme. Both failed attempts exited normally and were closed. These are verification corrections, not production changes or qualification of real provider behavior/semantic retrieval/factual grounding. Answer inspection scrolling is programmatic.

Evidence: Docs/superpowers/qa/2026-09-17-rag-answer-recovery/README.md. Updated the Library workflow audit. ADR required: no; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/003-settings-library-rag-defaults.md apply. No new generalizable lesson beyond existing evidence/focus guidance. Next review: keyboard access to generated answers and citation warnings. No full suite, push or merge.
<!-- SECTION:NOTES:END -->
