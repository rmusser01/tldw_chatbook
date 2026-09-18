---
id: TASK-32714
title: Verify keyboard reading of RAG answers and citation warnings
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 06:07'
updated_date: '2026-09-17 06:26'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The component review needs evidence that keyboard users can read generated answers and citation warnings beyond the compact viewport. The existing scroll route should be verified and documented without adding redundant controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard scrolling exposes all lines of short and long generated answers, their uncited or invalid-marker warnings, and neutral validated-citation notes in both themes and wide/compact layouts.
- [x] #2 Keyboard reading preserves the query, source choices and answer, leaves evidence actions reachable, and causes no additional retrieval or generation.
- [x] #3 The user guide documents the verified Tab and page-scrolling route; mounted and native evidence clearly separates controlled provider responses from real-provider qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md govern the unchanged UI.
Reason: verification and guidance for existing keyboard behavior; no new bindings, widgets, styles or runtime boundaries.
1. Verify the existing Tab-to-evidence and Page Up/Page Down reading routes with production CSS, short/long replies, all three citation states, both sizes/themes, unchanged input/state and no repeat calls. Treat probe assertion mistakes separately from product findings.
2. Document the verified keyboard reading route and source-dependent Tab sequence in Docs/User_Guide/library/search-and-rag.md.
3. Run targeted keyboard/citation checks and static checks, review independently, then exercise native keyboard-only reading with real local retrieval and controlled replies. Inspect captures and private persistence/shutdown; close the task/audit and commit locally. If a real defect appears, amend scope before repair. No full suite, push or merge.
Task allocation: maximum 32713 across 267 refs and 27 worktrees; CLI offered 32714, assigned 32714.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the existing evidence-card Page Up/Page Down route without production UI changes. Added 24 production-CSS cases covering short/long replies, both themes/sizes, and uncited/invalid/validated citation feedback; full compositor rows are reconstructed in each direction so normal wrapped page boundaries remain valid. Query/source choices/answer and call counts stay unchanged, and an evidence action remains reachable. Updated the Search/RAG guide to explain this route and replace its fixed-five-Tab claim.
Validation: 24 navigation tests plus 65 answer-service tests pass; Ruff lint/format and whitespace checks pass. Twelve native long-answer journeys pass with real local keyword retrieval adapted to RAG and controlled provider replies; all 16 captures inspected, 10 private databases healthy, source/default files unchanged, no messages created, normal shutdown/exact PID absence verified. Independent review's stale pending-status sentence was corrected and confirmed resolved.
Evidence: Docs/superpowers/qa/2026-09-17-rag-answer-navigation/README.md; audit: Docs/superpowers/reports/2026-09-14-library-workflow-audit.md. Initial probes corrected a whole-sentence-per-frame assertion; two native attempts failed between-cell return-to-query setup, so final cells explicitly reset the query viewport before measured keyboard reading. These are documented limitations, not product fixes. Real provider behavior, semantic retrieval, factual grounding and keyboard return across resize remain unqualified.
ADR required: no; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md govern unchanged UI. No new runtime/storage/dependency/security/license boundary. No full suite, push or merge. Task-ID check: 310 refs/27 worktrees, no collision. Next review: query return and resize transitions.
<!-- SECTION:NOTES:END -->
