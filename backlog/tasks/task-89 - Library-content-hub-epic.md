---
id: TASK-89
title: Library content hub epic
status: To Do
labels:
- library
- ux
- content-hub
- epic
priority: high
documentation:
- Docs/superpowers/qa/product-maturity/screen-qa/library/notes.md
- Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-20-library.md
- Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-23-workspaces-library-depth.md
- Docs/superpowers/specs/2026-05-23-citation-snippet-carry-through-epic-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Library a first-class landing page and center hub for ingested content across Notes, Media, Conversations, Collections, Workspaces, Import/Export, Search/RAG, Study, Flashcards, and Quizzes. Users should understand what content exists, which module owns deeper work, and which handoffs are available without falling into thin placeholder screens or unclear ownership boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library exposes a coherent content-hub model for Notes, Media, Conversations, Collections, Workspaces, Search/RAG, Import/Export, Study, Flashcards, and Quizzes.
- [ ] #2 Each Library sub-screen or mode has an explicit owner, source-of-truth, available actions, blocked-state recovery, and Console/RAG/study handoff behavior.
- [ ] #3 Actual CDP/Textual-web QA verifies the Library workbench and every changed sub-screen with screenshots approved before UI PRs are created.
- [ ] #4 No source visibility is hidden by workspace switching; workspace only controls staging/manipulation eligibility.
- [ ] #5 Route handoffs that remain outside Library are labeled as intentional destination ownership, not accidental exits.
- [ ] #6 Focused regressions cover hub status, module ownership, action availability, handoff payloads, keyboard access, and empty/error/recovery states.
- [ ] #7 Child implementation slices follow the content-hub contract produced by TASK-89.1 rather than redefining source authority, workspace eligibility, or handoff payloads independently.
- [ ] #8 Library visual acceptance is verified in the actual rendered app: it fills the Textual/Web canvas, preserves top destination tabs and status bar, uses clear columns, keeps focus visible, and remains usable at supported terminal widths.
<!-- AC:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not turn Library into a second live agent Console.
- Do not rebuild Ingest, Media, Study, Flashcards, Quizzes, or full citation persistence/export inside this epic.
- Do not hide global Library content when the active workspace changes; workspace context gates staging and manipulation only.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
