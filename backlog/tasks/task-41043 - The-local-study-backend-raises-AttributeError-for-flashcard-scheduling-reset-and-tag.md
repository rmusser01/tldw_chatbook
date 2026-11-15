---
id: TASK-41043
title: "The local study backend raises AttributeError for flashcard scheduling reset and tag suggestions"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- bug
- study
priority: low
references:
- tldw_chatbook/Study_Interop/local_study_service.py
- tldw_chatbook/Study_Interop/study_scope_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In `Study_Interop/local_study_service.py`, `reset_flashcard_scheduling` and `list_flashcard_tag_suggestions` call methods of the same names on `LearningRepository`. That class has neither method, and neither did `CharactersRAGDB` before TASK-36949.2 moved the learning data out, so both calls raise AttributeError.

`StudyScopeService` forwards both calls to whichever backend is active, and the server backend implements them, so a local profile fails where a server profile works. Nothing in the app calls them yet.

Found by TASK-36949.2 (2026-11-06); re-read on dev ecad404bd4 (2026-11-15).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both methods work against the local learning repository, or refuse with an explicit unsupported-locally error that the scope service passes on
- [ ] #2 A test covers each of the two local methods
<!-- AC:END -->
