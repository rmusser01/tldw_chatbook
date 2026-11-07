---
id: TASK-34908
title: Retire the Window-in-Screen double generation
status: To Do
created_date: 2026-10-12 17:54
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
12 legacy Window classes (13.2k LOC) are hosted by thin screen wrappers that translate lifecycle across UI generations; research_screen.py and writing_screen.py are line-for-line identical boilerplate. A generic WindowHostScreen (or folding each Window body into its screen compose_content) retires a whole layer of indirection.

Evidence (verified on dev df4985e): Hosting wrappers re-implement _pending_restore_state deferral and save/restore dance (compare research_screen.py:40-72 with writing_screen.py:48-68). Chatbooks nests further full Window widgets (Chatbooks_Window_Improved.py:858+884). Blockers: STTS_Events and LLM_Management_Events target the Window classes; 80+ test files across the 12 windows; study stts llm screens carry real logic beyond hosting so their folds are per-file effortful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Generic host lands for the simple boilerplate hosts (research writing) first,Remaining hosted destinations migrate incrementally (chatbooks logs study stts llm),UI/ChatbookCreationWindow.py deleted (never constructed),app.py and app_speech.py symbol references migrated,Each migration lands with its per-window test suite updated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
