---
id: TASK-40327
title: "Opening Find in the Library Media viewer re-parses the whole document"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- media
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/Widgets/Library/library_media_viewer.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (lib stream; TASK-39152 recorded the extra parse as debt and pinned it). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Opening Find on a large document pays a full Markdown re-parse (the test's 49 KB case) for a toggle that only adds a bar, and it recomposes every viewer child, so focus has to be rescued afterwards. Introduced by 7b04023420.

**Path.** `_toggle_library_media_find` (`UI/Screens/library_screen.py:35041-35068`) sets `find_open=True` and syncs the viewer. In the media controller's sync, `viewer.find_open != state` fails the unchanged check (`library_media_controller.py:4463`), so it assigns `viewer.find_open` (`:4510`) and calls `viewer.refresh(recompose=True)` (`:4561`). `LibraryMediaViewer.compose` rebuilds `LibraryMediaContentBody(content=...)` (`Widgets/Library/library_media_viewer.py:755-767`), whose `Markdown.update` re-parses the whole document. `Tests/UI/test_library_shell.py:11886-11889` pins exactly 2 parses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening and closing Find causes no parse of the document after the first; the latency proxy test is tightened from 2 parses to 1
- [ ] #2 The content body widget keeps its identity across a Find toggle, and focus stays where it was
<!-- AC:END -->
