---
id: TASK-32101
title: >-
  Library Conversations: `#library-conversation-use-source` has no compose site;
  its two handlers are dead; hand-off copy and annotation leftovers
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - library
  - conversations
  - cleanup
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32056 review (PR #2523): nothing composes `#library-conversation-use-source`, yet `library_conversations_controller.py:1638` and `library_screen.py:35604` handle its press; `_open_console_tooltip()` still says 'Press Link to workspace' in the generic blocked state where the Link button is hidden; `library_conversation_reader_controller.py:150` annotates the injected callable as `Callable[[], str]` (it returns `tuple[str, bool]`); `media-and-conversations.md` describes a 'no active workspace' message-only case the code cannot produce; the blocked state paints 'Open in Console' twice (button label + Static line). Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dead handlers are removed or the control is composed
- [ ] #2 The tooltip never names an action that is not on screen
- [ ] #3 The annotation, docstring and guide sentence match the code
- [ ] #4 The blocked state paints the action name once
<!-- AC:END -->

## Critique #9 evidence (2026-09-10)

Assessor A read the blocked state as TWO 'Open in Console' controls (a plain button and the `○ Open in Console · not in this workspace` line beneath it) and found the `c` accelerator silent when blocked (captures 54/55). Fold into this task: render one control in the blocked state, and make `c` say the same sentence the control says (a toast) instead of nothing.
