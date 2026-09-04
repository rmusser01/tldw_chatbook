---
id: TASK-31262
title: Sweep earlier-wave stale present-tense shim-footer comments
status: To Do
assignee: []
created_date: '2026-09-04 05:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This wave's final review found a shim-footer comment in library_rag_search_controller.py still claiming, in present tense, that LibraryScreen 'carries' a generated state shim block that an earlier cleanup task had already deleted. library_collections_controller.py and library_export_controller.py already carry the corrected past-tense phrasing and serve as the template. The same stale present-tense claim (a shim block 'the shim block LibraryScreen carries') survives, uncorrected, in two earlier-wave controllers whose own screen-side shim blocks were similarly deleted at their cleanup PRs: library_conversation_reader_controller.py (~line 913) and library_conversations_controller.py (~line 1712). Sweep these stragglers to match the corrected template.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 library_conversation_reader_controller.py's shim-footer comment is reworded to past tense, matching library_collections_controller.py's corrected template phrasing
- [ ] #2 library_conversations_controller.py's shim-footer comment is reworded the same way, and the controller-ratchet file's pinned line count for either file is re-measured and re-pinned in the same commit if the rewording changed it
<!-- AC:END -->
